"""User-facing CLI for verified GGUF meshes."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import math
import os
import re
import signal
import socket
import ssl
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from verallm.mesh.llama_cpp import (
    DEFAULT_LLAMA_RPC_PORT,
    DEFAULT_LLAMA_SERVER_PORT,
    build_llama_server_command,
    build_rpc_worker_command,
    command_preview,
    normalize_rpc_endpoint,
    resolve_binary,
    rpc_plan_from_mesh,
    run_command,
)
from verallm.mesh.http_auth import (
    DEFAULT_VALIDATOR_ALLOWLIST_MAX_AGE_SECONDS,
)
from verallm.mesh.proof import (
    PROOF_SAMPLE_BPS_DENOMINATOR,
    is_proof_capable_rpc_worker_binary,
    normalize_proof_sample_bps,
)
# ggml_proof / gguf_manifest import torch + numpy at module load. Every
# name from them is imported inside its use site instead: interactive
# commands (chat, manage, fleet) must start in milliseconds, and on a box
# whose disk a 200 GB model load is saturating, a torch import at CLI
# start was observed to look like a hard hang.
from verallm.mesh.types import (
    SUPPORTED_TRACE_MANIFEST_FORMATS,
    CapabilityAd,
    MeshMember,
    MeshSpec,
    StageRange,
    canonical_json_bytes,
    load_mesh_spec,
    save_json,
)
from verallm.mesh.state import (
    MESH_SPEC_FILE,
    MeshJoinToken,
    admit_mesh_worker,
    create_mesh_state,
    join_mesh,
    load_mesh_state,
    mesh_state_path,
    refresh_worker_mesh_state,
    save_mesh_state,
    state_admitted_compute_stage_count,
    state_capabilities,
    state_internal_auth_secret,
    state_mesh_spec,
    update_worker_mesh_state,
)
from verallm.mesh.worker import (
    DEFAULT_PROOF_ARTIFACT_TIMEOUT,
    DEFAULT_WORKER_PORT,
    deferred_audit_decision,
    post_json,
    probe_worker,
    serve_worker,
    verify_deferred_mesh_audit_bundle,
    verify_mesh_inference_artifact,
)
from verallm.mesh.verification_snapshot import (
    MIN_SECURE_TRACE_CANDIDATES_PER_REQUEST,
    MeshCoordinatorIdentity,
    MeshVerificationPolicy,
    MeshVerificationSnapshot,
    build_mesh_verification_snapshot,
    derive_mesh_verification_stage_bindings,
    sign_mesh_verification_snapshot,
)
# smoke pulls gguf_manifest (torch) transitively; imported in cmd_smoke.

GGML_RPC_TRACE_MAX_ELEMS_FALLBACK = 4_194_304
VERIFICATION_SNAPSHOT_FILE = "verification_snapshot.json"


def _default_mesh_dir() -> Path:
    return Path.cwd() / ".verathos" / "meshes"


def _positive_seconds(value: str) -> float:
    seconds = float(value)
    if not math.isfinite(seconds) or seconds <= 0:
        raise argparse.ArgumentTypeError("must be a positive number of seconds")
    return seconds


def _subtensor_network_arg(value: str) -> str:
    """A network name OR an explicit chain endpoint URL.

    Long-running chain readers (epoch follower, allowlist refresher) must
    be able to target an operator-run node: the public entrypoints
    rate-limit busy boxes (observed HTTP 429 stalling snapshot
    rotation).
    """

    network = str(value or "").strip()
    if network in ("", "test", "finney") or network.startswith(
        ("ws://", "wss://", "http://", "https://")
    ):
        return network
    raise argparse.ArgumentTypeError(
        "must be '', 'test', 'finney', or an explicit ws(s)://:http(s):// "
        "chain endpoint URL"
    )


def _default_trace_dir(mesh: str | Path = "") -> Path:
    if mesh:
        path = Path(mesh)
        if path.is_dir():
            return path / "ggml-traces"
    return Path.cwd() / ".verathos" / "ggml-traces"


def _normalize_hf_selector(value: str) -> str:
    return "".join(ch.lower() for ch in value if ch.isalnum())


def _resolve_cached_llama_hf_gguf(llama_hf: str) -> Path | None:
    """Resolve a locally cached GGUF file for llama.cpp's -hf syntax."""

    raw = str(llama_hf or "").strip()
    if not raw:
        return None
    repo_id, _, selector = raw.partition(":")
    repo_id = repo_id.strip()
    selector = selector.strip()
    if not repo_id:
        return None

    def cached_snapshot_from_env() -> Path | None:
        cache_root = (
            Path(os.environ["HF_HUB_CACHE"])
            if os.environ.get("HF_HUB_CACHE")
            else Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
            / "hub"
        )
        repo_cache = cache_root / ("models--" + repo_id.replace("/", "--"))
        snapshots = repo_cache / "snapshots"
        if not snapshots.exists():
            return None
        candidates = [path for path in snapshots.iterdir() if path.is_dir()]
        return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None

    snapshot_dir: Path | None = None
    if os.environ.get("HF_HUB_CACHE") or os.environ.get("HF_HOME"):
        snapshot_dir = cached_snapshot_from_env()
    try:
        if snapshot_dir is None:
            from huggingface_hub import snapshot_download

            snapshot_dir = Path(
                snapshot_download(
                    repo_id,
                    allow_patterns=["*.gguf"],
                    local_files_only=True,
                )
            )
    except Exception:
        snapshot_dir = cached_snapshot_from_env()
    if snapshot_dir is None or not snapshot_dir.exists():
        return None

    # rglob: repos that publish several quants keep each in its own
    # subdirectory (UD-IQ2_M/, UD-Q3_K_XL/), so a flat glob finds nothing.
    ggufs = sorted(snapshot_dir.rglob("*.gguf"))
    if not ggufs:
        return None
    if selector:
        normalized_selector = _normalize_hf_selector(selector)
        matched = [
            path
            for path in ggufs
            if normalized_selector in _normalize_hf_selector(path.name)
        ]
        if matched:
            ggufs = matched

    first_shards = [path for path in ggufs if "-00001-of-" in path.name.lower()]
    if first_shards:
        return first_shards[0]
    return max(ggufs, key=lambda path: path.stat().st_size)


def _suggest_ggml_trace_max_elems_from_model_ref(model_ref: str) -> int | None:
    from verallm.mesh.gguf_manifest import (
        suggest_ggml_trace_max_elems_from_gguf_model,
    )

    raw = str(model_ref or "").strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    if path.exists() and path.is_file():
        return suggest_ggml_trace_max_elems_from_gguf_model(path)
    cached_gguf = _resolve_cached_llama_hf_gguf(raw)
    if cached_gguf is not None:
        return suggest_ggml_trace_max_elems_from_gguf_model(cached_gguf)
    return None


def _print_json(value: dict) -> None:
    print(json.dumps(value, sort_keys=True, indent=2))


def _color_enabled() -> bool:
    """ANSI color only on a real terminal and never against NO_COLOR."""
    return sys.stdout.isatty() and not os.environ.get("NO_COLOR")


def _c(text: str, code: str) -> str:
    """Wrap text in an ANSI style when color is enabled.

    Codes: 1=bold 2=dim 32=green 31=red 33=yellow 36=cyan 35=magenta.
    """
    if not _color_enabled():
        return text
    return f"\033[{code}m{text}\033[0m"


def _interactive_terminal() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def _write_or_print_json(value: dict, path: str = "") -> None:
    if path:
        Path(path).write_text(
            json.dumps(value, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        return
    _print_json(value)


def _load_json_file(path: str | Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"{path} must contain a JSON object")
    return data


def _without_sample_lists(value):
    if isinstance(value, dict):
        return {
            key: _without_sample_lists(item)
            for key, item in value.items()
            if not (key == "samples" and isinstance(item, list))
        }
    if isinstance(value, list):
        return [_without_sample_lists(item) for item in value]
    return value


def _stream_chat_completions(
    endpoint: str,
    payload: dict,
    *,
    content_only: bool = False,
    timeout: float = 120.0,
) -> None:
    req = Request(
        endpoint.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload, sort_keys=True).encode("utf-8"),
        headers={"Accept": "text/event-stream", "Content-Type": "application/json"},
        method="POST",
    )
    event = ""
    data_lines: list[str] = []
    try:
        with urlopen(req, timeout=timeout) as resp:
            while True:
                raw = resp.readline()
                if not raw:
                    break
                line = raw.decode("utf-8").rstrip("\r\n")
                if line.startswith("event:"):
                    event = line[len("event:"):].strip()
                elif line.startswith("data:"):
                    data_lines.append(line[len("data:"):].strip())
                elif line == "":
                    if not data_lines:
                        event = ""
                        continue
                    data = "\n".join(data_lines)
                    if data == "[DONE]":
                        if content_only:
                            print()
                        return
                    parsed = json.loads(data)
                    evt = event or parsed.get("event") or "message"
                    if content_only and evt == "message":
                        for choice in parsed.get("choices", []):
                            delta = choice.get("delta", {}) if isinstance(choice, dict) else {}
                            text = delta.get("content", "") if isinstance(delta, dict) else ""
                            if text:
                                print(text, end="", flush=True)
                    elif not content_only:
                        print(json.dumps({"event": evt, "data": parsed}, sort_keys=True))
                    event = ""
                    data_lines = []
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from stream: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"failed to connect to stream: {exc.reason}") from exc


def _mesh_path(mesh: str | Path) -> Path:
    path = Path(mesh)
    if path.is_dir():
        return path / "mesh.json"
    return path


def _state_exists(mesh: str | Path) -> bool:
    return mesh_state_path(mesh).exists()


def _save_mesh_spec(path: str | Path, spec: MeshSpec) -> Path:
    return save_json(_mesh_path(path), spec.to_dict())


def _load_mesh_spec(path: str | Path) -> MeshSpec:
    if _state_exists(path):
        return state_mesh_spec(load_mesh_state(path))
    return load_mesh_spec(_mesh_path(path))


def _mesh_bound_ctx_size(
    spec: MeshSpec,
    requested: int | None,
    *,
    flag_name: str,
) -> int | None:
    """Clamp the llama context to what the committed mesh spec allows.

    ``spec.max_context_len`` is the PER-REQUEST contract, and the serve
    budget must hold at least that much - but it may hold MORE: the KV
    auto-fit sizes the unified pool to the machine's measured fit so real
    headroom serves concurrency (overlapping validator canaries,
    private-API traffic). The old exact-equality rule predates the
    measured-fit budget and crash-looped every serve launched above the
    contract. Per-request enforcement lives in the
    admission ledger, not here.
    """

    if spec.max_context_len:
        if requested is not None and requested < spec.max_context_len:
            raise SystemExit(
                f"{flag_name} must be at least the mesh max_context_len "
                f"({spec.max_context_len}); a smaller unified KV cannot "
                "serve the advertised per-request contract"
            )
        return requested or spec.max_context_len
    return requested


def _mesh_summary(path: str | Path, spec: MeshSpec) -> dict:
    return {
        "path": str(_mesh_path(path)),
        "mesh_id": spec.mesh_id,
        "mode": spec.mode,
        "model_id": spec.model_id,
        "members": len(spec.members),
        "mesh_spec_hash": spec.spec_hash_hex(),
        "stage_assignment_hash": spec.stage_assignment_hash_hex(),
        "rpc_plan": rpc_plan_from_mesh(spec).to_dict(),
        "spec": spec.to_dict(),
    }


def _mesh_internal_auth_secret(path: str | Path) -> str:
    """Load a mesh control secret without ever including it in output."""

    return state_internal_auth_secret(load_mesh_state(mesh_state_path(path)))


def _atomic_write_canonical_json(
    path: str | Path,
    value: Mapping[str, Any],
) -> Path:
    """Atomically replace ``path`` with deterministic compact JSON."""

    output = Path(path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_json_bytes(value) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return output


def _load_verification_snapshot(path: str | Path) -> MeshVerificationSnapshot:
    data = _load_json_file(path)
    snapshot = MeshVerificationSnapshot.from_dict(data)
    snapshot.validate(require_signature=True)
    return snapshot


def _snapshot_loader_for_serve(
    *,
    explicit_path: str,
    state_path: Path | None,
    required: bool = False,
) -> Callable[[], MeshVerificationSnapshot] | None:
    """Resolve and validate a snapshot while retaining request-time reloads."""

    path: Path | None = None
    if explicit_path:
        path = Path(explicit_path).expanduser()
        if not path.is_file():
            raise SystemExit(f"verification snapshot does not exist: {path}")
    elif state_path is not None:
        candidate = state_path.parent / VERIFICATION_SNAPSHOT_FILE
        if candidate.is_file() or required:
            path = candidate
    if path is None:
        if required:
            raise SystemExit(
                "--require-verification-snapshot requires persisted coordinator mesh state"
            )
        return None

    # A pool coordinator starts before its remote stages join. In required
    # mode the loader is installed immediately, while inference fails closed
    # until the final assignment has been signed and atomically published.
    if path.is_file():
        try:
            _load_verification_snapshot(path)
        except Exception as exc:
            raise SystemExit(f"invalid verification snapshot {path}: {exc}") from exc
    elif explicit_path:
        raise SystemExit(f"verification snapshot does not exist: {path}")

    return lambda: _load_verification_snapshot(path)


def _mesh_has_proof_endpoint(spec: MeshSpec | None) -> bool:
    return bool(spec and any(member.proof_endpoint for member in spec.members))


def _proof_collection_label(args: argparse.Namespace, spec: MeshSpec | None) -> str:
    if args.proof_url:
        return "url"
    if args.proof_trace_dir:
        return "embedded"
    if _mesh_has_proof_endpoint(spec):
        return "member_endpoints"
    return "none"


def _broadcast_mesh_update(
    *,
    spec: MeshSpec,
    join_secret: str,
    skip_endpoints: set[str],
    timeout: float,
) -> list[dict[str, str]]:
    errors: list[dict[str, str]] = []
    normalized_skip = {endpoint.rstrip("/") for endpoint in skip_endpoints}
    payload = {
        "join_secret": join_secret,
        "mesh_id": spec.mesh_id,
        "mesh_spec_hash": spec.spec_hash_hex(),
        "stage_assignment_hash": spec.stage_assignment_hash_hex(),
        "mesh": spec.to_dict(),
    }
    for member in sorted(spec.members, key=lambda item: item.stage_index):
        endpoint = member.endpoint.rstrip("/")
        if endpoint in normalized_skip:
            continue
        try:
            post_json(
                endpoint + "/v1/mesh/update",
                payload,
                timeout=timeout,
                internal_auth_secret=join_secret,
            )
        except Exception as exc:
            errors.append({"endpoint": endpoint, "error": str(exc)})
    return errors


def cmd_create(args: argparse.Namespace) -> None:
    spec = MeshSpec.new_private_mesh(
        coordinator_uid=args.uid,
        coordinator_hotkey=args.hotkey,
        endpoint=args.endpoint,
        model_id=args.model_id,
        model_package_ref=args.package_ref,
        model_package_hash=args.package_hash,
        model_tensor_manifest_root=args.model_tensor_manifest_root,
        tokenizer_hash=args.tokenizer_hash,
        quantization_scheme=args.quantization_scheme,
        activation_dtype=args.activation_dtype,
        max_context_len=args.max_context_len,
        total_layers=args.layers,
    )
    out_dir, _, token = create_mesh_state(
        spec=spec,
        root=Path(args.output_dir) if args.output_dir else _default_mesh_dir(),
    )
    print(f"Wrote mesh: {out_dir}")
    print(f"mesh_id: {spec.mesh_id}")
    print(f"mesh_spec_hash: {spec.spec_hash_hex()}")
    print(f"stage_assignment_hash: {spec.stage_assignment_hash_hex()}")
    print(f"join_token: {token.encode()}")


def cmd_snapshot_create(args: argparse.Namespace) -> None:
    """Build and sign the validator-facing snapshot for one coordinator."""

    state_path = mesh_state_path(args.mesh)
    if not state_path.is_file():
        raise SystemExit(
            "snapshot-create requires a persisted coordinator mesh state"
        )
    state = load_mesh_state(state_path)
    if str(state.get("role", "")) != "coordinator":
        raise SystemExit("snapshot-create requires coordinator mesh state")
    spec = state_mesh_spec(state)

    from verallm.chain.wallet import derive_evm_address
    from verallm.mesh.receipt_signing import load_hotkey_keypair, load_hotkey_seed

    keypair = load_hotkey_keypair(args.wallet_name, args.wallet_hotkey)
    if keypair.ss58_address != spec.coordinator_hotkey:
        raise SystemExit("wallet hotkey does not match mesh coordinator_hotkey")
    hotkey_seed = load_hotkey_seed(
        args.wallet_name,
        args.wallet_hotkey,
        keypair=keypair,
    )
    derived_evm_address = derive_evm_address(hotkey_seed).lower()
    declared_evm_address = str(args.coordinator_evm_address).strip().lower()
    if declared_evm_address != derived_evm_address:
        raise SystemExit(
            "--coordinator-evm-address does not match the coordinator hotkey"
        )

    issued_at_unix = (
        int(args.issued_at_unix)
        if args.issued_at_unix is not None
        else int(time.time())
    )
    expires_at_unix = (
        int(args.expires_at_unix)
        if args.expires_at_unix is not None
        else issued_at_unix + int(args.ttl_seconds)
    )
    coordinator = MeshCoordinatorIdentity(
        chain_id=args.chain_id,
        netuid=args.netuid,
        coordinator_uid=spec.coordinator_uid,
        coordinator_hotkey=spec.coordinator_hotkey,
        coordinator_evm_address=declared_evm_address,
        model_index=args.model_index,
    )
    policy = MeshVerificationPolicy(
        profile=args.policy_profile,
        trace_manifest_format=args.trace_manifest_format,
        base_proof_sample_bps=args.base_proof_sample_bps,
        organic_decode_sample_bps=args.organic_decode_sample_bps,
        canary_decode_sample_bps=args.canary_decode_sample_bps,
        proof_ops_per_request=args.proof_ops_per_request,
        proof_trace_candidates_per_request=(
            args.proof_trace_candidates_per_request
        ),
        deferred_proof_enabled=bool(args.deferred_proof_enabled),
    )
    snapshot = build_mesh_verification_snapshot(
        spec,
        coordinator=coordinator,
        policy=policy,
        generation=args.generation,
        epoch=args.epoch,
        issued_at_unix=issued_at_unix,
        expires_at_unix=expires_at_unix,
        stage_bindings=derive_mesh_verification_stage_bindings(
            spec,
            internal_auth_secret=state_internal_auth_secret(state),
        ),
    )
    signed = sign_mesh_verification_snapshot(snapshot, keypair)
    output = (
        Path(args.output).expanduser()
        if args.output
        else state_path.parent / VERIFICATION_SNAPSHOT_FILE
    )
    _atomic_write_canonical_json(output, signed.to_dict())
    print(f"Wrote verification snapshot: {output}")
    print(f"snapshot_hash: {signed.snapshot_hash_hex()}")
    print(f"stage_coverage_hash: {signed.stage_coverage_hash_hex()}")


def _pm2_jlist() -> list[dict]:
    """Running PM2 processes; empty when PM2 is absent or errors.

    Local implementation: verallm must not import neurons (the dependency
    points the other way), so neurons.wizard's helper is not reachable here.
    """
    import shutil

    pm2 = shutil.which("pm2")
    if not pm2:
        return []
    try:
        result = subprocess.run(
            [pm2, "jlist"], capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0 and result.stdout.strip():
            return json.loads(result.stdout)
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    return []


def _worker_status_payload(registry: Mapping[str, Any]) -> dict[str, Any]:
    """This machine's units joined with PM2 state and the pool's public view."""
    from verallm.mesh import units as units_module
    from verallm.mesh.worker import post_json

    units = units_module.units_from_registry(registry)
    pm2_by_name = {
        str(process.get("name", "")): process for process in _pm2_jlist()
    }
    manager_endpoint = str(registry.get("manager_endpoint", "")).rstrip("/")
    overview: dict[str, Any] = {}
    overview_error = ""
    if manager_endpoint:
        from verallm.mesh.onboarding import loopback_equivalent

        # Loopback twin second: a coordinator box often cannot hairpin to
        # its own public IP while the manager answers fine locally.
        for endpoint in dict.fromkeys(
            (manager_endpoint, loopback_equivalent(manager_endpoint))
        ):
            try:
                overview = post_json(
                    endpoint + "/v1/operator/overview", {}, timeout=5.0
                )
                overview_error = ""
                break
            except (RuntimeError, OSError) as exc:
                overview_error = str(exc)
    pool_workers = dict(overview.get("workers", {}))
    meshes = dict(overview.get("meshes", {}))
    rows = []
    for unit in units:
        process = pm2_by_name.get(unit.pm2_name, {})
        pm2_status = str(
            (process.get("pm2_env") or {}).get("status", "not registered")
        )
        pool_view = pool_workers.get(unit.worker_id, {})
        mesh_key = str(pool_view.get("mesh", "") or "")
        role = ""
        if mesh_key and meshes.get(mesh_key, {}).get("driver") == unit.worker_id:
            role = "driver"
        elif mesh_key:
            role = "member"
        rows.append(
            {
                "worker_id": unit.worker_id,
                "pm2_name": unit.pm2_name,
                "gpu_index": unit.gpu_index,
                "gpu_label": (
                    units_module.gpu_group_label(unit.gpu_indices)
                    if unit.gpu_indices
                    else str(unit.gpu_index)
                ),
                "gpu_name": unit.gpu_name,
                "pm2_status": pm2_status,
                "pool_status": str(pool_view.get("status", "") or ""),
                "stale": bool(pool_view.get("stale", False)),
                "mesh": mesh_key,
                "role": role,
                "model_id": str(
                    meshes.get(mesh_key, {}).get("model_id", "") or ""
                ),
            }
        )
    return {
        "pool_id": registry.get("pool_id", ""),
        "manager_endpoint": manager_endpoint,
        "manager_error": overview_error,
        "units": rows,
    }


def cmd_mesh_add_model(args: argparse.Namespace) -> None:
    """Add a GGUF that is already on this machine to the pool.

    The pool only knows what workers advertise from their catalog files;
    this writes the catalog entry (building the tensor manifest the drive
    path requires) and restarts idle PM2 worker units so they rejoin and
    advertise it. Purely local: nothing is registered on any chain — the
    on-chain ModelSpec is the subnet owner's, and `mesh deploy` is what
    reads it.
    """

    from verallm.mesh import render
    from verallm.mesh import units as units_module
    from verallm.mesh.gguf_manifest import (
        build_gguf_tensor_manifest,
        load_gguf_tensor_manifest,
        save_gguf_tensor_manifest,
    )

    path = Path(args.gguf).expanduser().resolve()
    if not path.is_file() or path.suffix != ".gguf":
        raise SystemExit(f"{path} is not a .gguf file")
    shards = units_module.gguf_shard_set(path)
    model_id = args.model_id or units_module.derive_gguf_model_id(
        shards[0].name
    )
    model_bytes = sum(shard.stat().st_size for shard in shards)

    manifest_path = shards[0].parent / f"{model_id}.tensor-manifest.json"
    manifest = None

    def _matches(candidate: dict[str, Any]) -> bool:
        recorded = sorted(
            (Path(str(f.get("path", ""))).name, int(f.get("n_bytes", 0) or 0))
            for f in candidate.get("model_files", [])
        )
        return recorded == sorted(
            (s.name, s.stat().st_size) for s in shards
        )

    supplied = str(getattr(args, "manifest", "") or "")
    if supplied:
        # A store-fetched manifest is already verified against the
        # subnet's registered root; reusing it skips the local rebuild.
        candidate = load_gguf_tensor_manifest(supplied)
        if not _matches(candidate):
            raise SystemExit(
                f"the supplied manifest {supplied} does not describe "
                f"these files"
            )
        manifest = candidate
        save_gguf_tensor_manifest(manifest, manifest_path)
        print(f"  using published tensor manifest {supplied}")
    if manifest is None and manifest_path.is_file():
        try:
            candidate = load_gguf_tensor_manifest(manifest_path)
            if _matches(candidate):
                manifest = candidate
                print(f"  reusing tensor manifest {manifest_path}")
        except (OSError, ValueError, KeyError):
            manifest = None
    if manifest is None:
        with render.spinner(
            f"building the tensor manifest for {model_id} "
            "(hashes every tensor once)..."
        ):
            manifest = build_gguf_tensor_manifest([str(s) for s in shards])
            save_gguf_tensor_manifest(manifest, manifest_path)
        print(f"  wrote tensor manifest {manifest_path}")

    blocks = {
        int(m.group(1))
        for record in manifest.get("tensors", [])
        if (m := re.match(r"blk\.(\d+)\.", str(record.get("name", ""))))
    }
    layers = (max(blocks) + 1) if blocks else 0

    entry: dict[str, Any] = {
        "model_id": model_id,
        "llama_model": str(shards[0]),
        "manifest": str(manifest_path),
        "layers": layers,
        "model_bytes": model_bytes,
    }
    hf_files = [
        part.strip()
        for part in str(getattr(args, "hf_files", "") or "").split(",")
        if part.strip()
    ]
    if getattr(args, "hf_repo", "") and hf_files:
        entry["hf_repo"] = args.hf_repo
        entry["hf_files"] = hf_files

    registry = None
    try:
        registry = units_module.load_unit_registry()
    except Exception:
        registry = None
    catalog_paths: list[Path] = []
    if getattr(args, "catalog", ""):
        catalog_paths = [Path(args.catalog).expanduser()]
    elif registry:
        catalog_paths = [
            Path(unit.catalog)
            for unit in units_module.units_from_registry(registry)
        ]
    if not catalog_paths:
        catalog_paths = [Path.home() / ".verathos" / "pool-catalog.json"]

    for catalog_path in catalog_paths:
        try:
            existing = json.loads(catalog_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            existing = []
        if not isinstance(existing, list):
            existing = []
        existing = [
            item
            for item in existing
            if str((item or {}).get("model_id", "")) != model_id
        ]
        existing.append(entry)
        catalog_path.parent.mkdir(parents=True, exist_ok=True)
        catalog_path.write_text(json.dumps(existing, indent=1))
        print(f"  catalog updated: {catalog_path}")

    print(
        f"added {model_id}: {len(shards)} file(s), "
        f"{model_bytes / 1e9:.1f} GB, {layers} layers"
    )

    if getattr(args, "no_restart", False) or not registry:
        print(
            "restart the worker units so they advertise it: "
            "verathos mesh start --all (or pm2 restart <unit>)"
        )
        return
    # Workers advertise catalogs at join time only; restart the IDLE units
    # so the pool learns the model now. FAIL CLOSED: only a worker whose
    # live status is exactly "idle" restarts. An unknown or empty status
    # (manager unreachable, worker mid-assignment) once counted as idle
    # and a restart killed two members mid-model-load, aborting their
    # mesh; a skipped restart only delays catalog advertisement.
    statuses: dict[str, str] = {}
    for pool in _local_pool_views():
        for worker_id, worker in pool["workers"].items():
            statuses[worker_id] = str(worker.get("status", "") or "")
    for unit in units_module.units_from_registry(registry):
        status = statuses.get(unit.worker_id, "")
        if status != "idle":
            print(
                f"  {unit.pm2_name}: NOT restarted "
                f"({status or 'state unknown'}); restart it once it is "
                "idle (verathos mesh start --all)"
            )
            continue
        subprocess.run(["pm2", "restart", unit.pm2_name], capture_output=True)
        print(f"  {unit.pm2_name}: restarted to advertise the model")


def _local_mesh_processes() -> list[dict[str, Any]]:
    """Pool manager/worker processes running on this machine right now.

    The unit registry only knows PM2-managed units; a worker launched by
    hand (`verathos mesh pool worker ...`) is just as real and must show
    up in status instead of the machine claiming it never joined a pool.
    """

    try:
        listing = subprocess.run(
            ["ps", "-eo", "pid=,args="],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    pm2_pids = {
        int((process.get("pid") or 0))
        for process in _pm2_jlist()
        if process.get("pid")
    }
    rows: list[dict[str, Any]] = []
    for line in (listing.stdout or "").splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) != 2 or not parts[0].isdigit():
            continue
        pid, argv = int(parts[0]), parts[1]
        if "<defunct>" in argv:
            continue
        if " mesh pool serve" in argv:
            kind = "manager"
        elif " mesh pool worker" in argv:
            kind = "worker"
        else:
            continue
        rows.append(
            {
                "pid": pid,
                "kind": kind,
                "pm2": pid in pm2_pids,
                "args": argv[:160],
            }
        )
    return rows


def _local_pool_views() -> list[dict[str, Any]]:
    """Every pool whose state lives on this machine, with its LIVE view."""

    from verallm.mesh.onboarding import pool_management_status
    from verallm.mesh.pool import (
        POOL_ADMIN_TOKEN_FILE,
        known_pool_dirs,
        load_pool_token_file,
    )

    views: list[dict[str, Any]] = []
    seen: set[str] = set()
    for pool_dir in known_pool_dirs():
        resolved = str(Path(pool_dir).resolve())
        if resolved in seen or not (Path(pool_dir) / "pool-state.json").exists():
            continue
        seen.add(resolved)
        # An unreadable admin token and a dead manager are different
        # problems; reporting both as "manager DOWN" sends the operator
        # to the wrong fix.
        token_error = ""
        try:
            load_pool_token_file(Path(pool_dir) / POOL_ADMIN_TOKEN_FILE)
        except (OSError, ValueError) as exc:
            token_error = str(exc)
        status = (
            pool_management_status(Path(pool_dir), timeout=2.0)
            if not token_error
            else None
        )
        views.append(
            {
                "pool_dir": str(pool_dir),
                "pool_id": Path(pool_dir).name,
                "manager_reachable": status is not None,
                "token_error": token_error,
                "workers": dict((status or {}).get("workers", {})),
                "meshes": dict((status or {}).get("meshes", {})),
            }
        )
    return views


def cmd_worker_status(args: argparse.Namespace) -> None:
    from verallm.mesh import units as units_module

    registry_error = ""
    try:
        registry = units_module.load_unit_registry()
    except Exception as exc:
        registry = None
        registry_error = str(exc)
    pools = _local_pool_views()
    processes = _local_mesh_processes()
    payload = (
        _worker_status_payload(registry)
        if registry and registry.get("units")
        else None
    )

    if getattr(args, "json", False):
        _print_json(
            {
                "units": payload,
                "pools": pools,
                "processes": processes,
                **(
                    {"registry_error": registry_error}
                    if registry_error
                    else {}
                ),
            }
        )
        return

    if registry_error:
        print(f"warning: unit registry is unreadable: {registry_error}")
    if payload is None and not pools and not processes:
        print("this machine has not joined a mesh pool.")
        print("start one, or join an existing pool with its token:")
        print("  verathos mesh setup")
        return

    if payload is not None:
        print(
            f"pool {payload['pool_id']}  manager={payload['manager_endpoint']}"
        )
        if payload["manager_error"]:
            print(f"  manager unreachable: {payload['manager_error']}")
        for row in payload["units"]:
            line = (
                f"  {row['worker_id']:24s} gpu{row.get('gpu_label', row['gpu_index'])} {row['gpu_name']}\n"
                f"    pm2={row['pm2_status']}"
            )
            if row["pool_status"]:
                line += f"  pool={row['pool_status']}"
                if row["stale"]:
                    line += " (stale)"
            if row["mesh"]:
                line += f"  mesh={row['mesh']} ({row['role']}, {row['model_id']})"
            print(line)

    from verallm.mesh import render as _render

    styled = _render.use_style()
    for pool in pools:
        if payload is not None or pool is not pools[0]:
            print()
        if pool.get("token_error"):
            manager_state = _render.red(
                f"admin token unreadable: {pool['token_error']}",
                styled=styled,
            )
        elif pool["manager_reachable"]:
            manager_state = _render.green("manager healthy", styled=styled)
        else:
            manager_state = _render.red("manager DOWN", styled=styled)
        print(
            _render.bold(
                f"pool {pool['pool_id']} on this machine", styled=styled
            )
            + f" ({manager_state})"
            + _render.dim(f"  state={pool['pool_dir']}", styled=styled)
        )
        for worker_id in sorted(pool["workers"]):
            worker = pool["workers"][worker_id]
            capability = worker.get("capability") or {}
            status = str(worker.get("status", "?") or "?")
            if worker.get("stale"):
                status += " (stale)"
            mesh = str(worker.get("mesh", "") or "") or "-"
            print(
                f"  worker {worker_id:24s} "
                + _render.badge(f"{status:16s}", styled=styled)
                + f" gpu={capability.get('gpu_name', '?')}  mesh={mesh}"
            )
        for mesh_key in sorted(pool["meshes"]):
            mesh = pool["meshes"][mesh_key]
            status = str(mesh.get("status", "?") or "?")
            if status == "serving" and not mesh.get("routing_ready", True):
                status += " (NOT routable: stale member)"
            print(
                f"  mesh   {mesh_key:24s} "
                + _render.badge(f"{status:16s}", styled=styled)
                + f" model={mesh.get('model_id', '?')}"
            )

    if processes:
        print()
        print("running mesh processes on this machine:")
        for row in processes:
            managed = "pm2" if row["pm2"] else "manual"
            print(
                f"  pid {row['pid']:>7d}  {row['kind']:8s} [{managed}] "
                f"{row['args']}"
            )
    else:
        print()
        print("no mesh manager/worker processes are running on this machine.")


def _resolve_pm2_targets(
    args: argparse.Namespace, *, allow_all: bool = False
) -> list[str]:
    """Map a target argument to PM2 unit names via the unit registry."""
    from verallm.mesh import units as units_module

    if getattr(args, "manager", False):
        return ["verathos-pool-manager"]
    registry = units_module.load_unit_registry()
    units = units_module.units_from_registry(registry) if registry else []
    target = str(getattr(args, "target", "") or "").strip()
    if target:
        for unit in units:
            if target in (unit.worker_id, unit.pm2_name):
                return [unit.pm2_name]
        # A raw PM2 name still works even without a registry entry.
        return [target]
    if allow_all and getattr(args, "all", False):
        return [unit.pm2_name for unit in units]
    if len(units) == 1:
        return [units[0].pm2_name]
    if not units:
        raise SystemExit(
            "no PM2-managed mesh worker units on this machine; pass a PM2 "
            "unit name, or --manager for the pool manager. Manually "
            "launched workers are not PM2-managed — `verathos mesh status` "
            "lists their pids."
        )
    listing = "\n  ".join(unit.pm2_name for unit in units)
    raise SystemExit(
        "several mesh units on this machine; name one:\n  " + listing
    )


def _require_pm2() -> None:
    import shutil

    if shutil.which("pm2") is None:
        raise SystemExit(
            "PM2 is not installed on this machine. Manually launched "
            "workers are not PM2-managed; `verathos mesh status` lists "
            "their pids."
        )


def cmd_mesh_logs(args: argparse.Namespace) -> None:
    _require_pm2()
    names = _resolve_pm2_targets(args, allow_all=True)
    if len(names) == 1:
        subprocess.run(["pm2", "logs", names[0], "--lines", str(args.lines)])
        return
    # `pm2 logs <name>` streams forever, so a sequential loop would never
    # reach the second unit; multi-target logs are snapshots instead.
    for name in names:
        subprocess.run(
            ["pm2", "logs", name, "--lines", str(args.lines), "--nostream"]
        )


def cmd_mesh_stop(args: argparse.Namespace) -> None:
    from verallm.mesh import units as units_module

    _require_pm2()
    names = _resolve_pm2_targets(args)
    registry = units_module.load_unit_registry()
    if registry is not None and not getattr(args, "manager", False):
        payload = _worker_status_payload(registry)
        serving = [
            row
            for row in payload["units"]
            if row["pm2_name"] in names and row["mesh"]
        ]
        for row in serving:
            print(
                f"note: {row['worker_id']} is serving mesh {row['mesh']}. "
                "To tear down the mesh itself, run:\n"
                f"  verathos mesh pool stop --mesh-key {row['mesh']} "
                "--pool-token-file <admin-token>"
            )
    for name in names:
        subprocess.run(["pm2", "stop", name])


def cmd_mesh_start(args: argparse.Namespace) -> None:
    _require_pm2()
    names = _resolve_pm2_targets(args, allow_all=True)
    for name in names:
        subprocess.run(["pm2", "start", name])


def cmd_status(args: argparse.Namespace) -> None:
    if not args.mesh:
        if args.probe:
            raise SystemExit("--probe requires a mesh spec path")
        cmd_worker_status(args)
        return
    spec = _load_mesh_spec(args.mesh)
    payload = _mesh_summary(args.mesh, spec)
    internal_auth_secret = (
        _mesh_internal_auth_secret(args.mesh) if _state_exists(args.mesh) else ""
    )
    if args.probe:
        probes = []
        for member in sorted(spec.members, key=lambda item: item.stage_index):
            try:
                probes.append(
                    probe_worker(
                        member.endpoint,
                        timeout=args.timeout,
                        internal_auth_secret=internal_auth_secret,
                    ).to_dict()
                )
            except Exception as exc:
                probes.append({"endpoint": member.endpoint, "status": "error", "error": str(exc)})
        payload["probes"] = probes
    _print_json(payload)


def _parse_layer_range(raw: str) -> StageRange:
    try:
        start_raw, end_raw = raw.split(":", 1)
        return StageRange(int(start_raw), int(end_raw))
    except Exception as exc:
        raise argparse.ArgumentTypeError("layer range must be START:END") from exc


def _parse_member(raw: str, *, stage_index: int) -> MeshMember:
    """Parse UID,HOTKEY,ENDPOINT,START:END[,BACKEND[,PAYOUT_BPS[,ROLE[,PROOF_ENDPOINT]]]]."""

    parts = [part.strip() for part in raw.split(",")]
    if len(parts) < 4:
        raise argparse.ArgumentTypeError(
            "member must be UID,HOTKEY,ENDPOINT,START:END[,BACKEND[,PAYOUT_BPS[,ROLE[,PROOF_ENDPOINT]]]]"
        )
    uid_raw, hotkey, endpoint, layers_raw = parts[:4]
    backend = parts[4] if len(parts) >= 5 and parts[4] else (
        "gguf_stage" if stage_index == 0 else "gguf_stage_worker"
    )
    payout_bps = int(parts[5]) if len(parts) >= 6 and parts[5] else (
        10000 if stage_index == 0 else 0
    )
    role = parts[6] if len(parts) >= 7 and parts[6] else (
        "coordinator" if stage_index == 0 else "worker"
    )
    proof_endpoint = parts[7] if len(parts) >= 8 else ""
    if len(parts) > 8:
        raise argparse.ArgumentTypeError("member has too many comma-separated fields")

    return MeshMember(
        uid=int(uid_raw),
        hotkey=hotkey,
        endpoint=endpoint,
        stage_index=stage_index,
        layers=_parse_layer_range(layers_raw),
        role=role,
        backend=backend,
        proof_endpoint=proof_endpoint,
        payout_bps=payout_bps,
    )


def cmd_assign(args: argparse.Namespace) -> None:
    base = _load_mesh_spec(args.mesh)
    members = [_parse_member(raw, stage_index=i) for i, raw in enumerate(args.member)]
    spec = MeshSpec(
        mesh_id=base.mesh_id,
        mode=args.mode or base.mode,
        coordinator_uid=base.coordinator_uid,
        coordinator_hotkey=base.coordinator_hotkey,
        model_id=base.model_id,
        model_package_ref=base.model_package_ref,
        model_package_hash=base.model_package_hash,
        model_tensor_manifest_root=base.model_tensor_manifest_root,
        tokenizer_hash=base.tokenizer_hash,
        quantization_scheme=base.quantization_scheme,
        activation_dtype=args.activation_dtype or base.activation_dtype,
        max_context_len=base.max_context_len,
        proof_trace_manifest_format=base.proof_trace_manifest_format,
        total_layers=base.total_layers,
        members=members,
        epoch=base.epoch,
        expires_at_unix=base.expires_at_unix,
        created_at_unix=base.created_at_unix,
        signatures={},
    )
    spec.validate()
    out = _save_mesh_spec(args.output or args.mesh, spec)
    print(f"Wrote mesh spec: {out}")
    print(f"mesh_id: {spec.mesh_id}")
    print(f"mesh_spec_hash: {spec.spec_hash_hex()}")
    print(f"stage_assignment_hash: {spec.stage_assignment_hash_hex()}")


def _capability_from_args(args: argparse.Namespace) -> CapabilityAd:
    return CapabilityAd(
        uid=args.uid,
        hotkey=args.hotkey,
        endpoint=args.endpoint,
        supported_backends=_supported_backends_from_args(args),
        proof_modes=args.proof_mode or ["verathos-gemv1"],
        cached_model_package_hashes=args.package_hash,
        gpu_name=args.gpu_name,
        vram_gb=args.vram_gb,
        rpc_endpoint=normalize_rpc_endpoint(args.rpc_endpoint) if args.rpc_endpoint else "",
        proof_endpoint=args.proof_endpoint,
        network_reachability=args.network_reachability,
        relay_hint=args.relay_hint,
    )


def _supported_backends_from_args(args: argparse.Namespace) -> list[str]:
    backends = list(args.backend or ["gguf_stage_worker"])
    if getattr(args, "rpc_endpoint", "") and "llama_cpp_rpc" not in backends:
        backends.append("llama_cpp_rpc")
    return backends


# Resolved at import, NOT inside the preexec closure: preexec_fn runs in
# the post-fork child of a heavily threaded serve, where an import or
# dlopen can deadlock on a lock some other thread held at fork. The
# closure below must make a single raw syscall and nothing else.
try:
    import ctypes as _ctypes

    _LIBC_PRCTL = _ctypes.CDLL("libc.so.6", use_errno=True).prctl
except Exception:
    _LIBC_PRCTL = None
_PR_SET_PDEATHSIG = 1


def _die_with_parent_preexec():
    """preexec_fn: backend children terminate when their serve dies.

    llama-server and the rpc worker are spawned by the SERVE process; when
    the serve crashes or is killed, they survived as orphans holding the
    mesh ports and VRAM, wedging every later stop/relaunch on the
    fail-closed unowned-listener guard.
    PR_SET_PDEATHSIG delivers SIGTERM the moment the parent exits, so the
    backend can never outlive its serve. Linux-only; other platforms
    degrade to the previous behaviour.

    CAVEAT (kernel semantics): the death signal fires when the spawning
    THREAD exits, not the process - so the supervisor thread that spawns
    a backend must stay alive for the serve's whole lifetime (see the
    supervisor loops: they wait on their stop events instead of
    returning early).
    """

    if _LIBC_PRCTL is None:
        return
    try:
        _LIBC_PRCTL(_PR_SET_PDEATHSIG, int(signal.SIGTERM), 0, 0, 0)
    except Exception:
        pass


def _terminate_process(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def _wait_port_free(
    host: str,
    port: int,
    *,
    timeout_s: float = 10.0,
    poll_s: float = 0.25,
    clock: Callable[[], float] = time.monotonic,
    sleeper: Callable[[float], None] = time.sleep,
) -> bool:
    """Block until (host, port) accepts a fresh bind, or time out.

    Killing a backend releases its PROCESS before the kernel releases its
    LISTENER (lingering sockets, PDEATHSIG-async children), so a respawn
    that races the teardown binds against the dying instance, exits
    immediately, and used to feed the crash-loop breaker with failures
    that said nothing about the backend. A bind probe is the only honest
    "free" signal: connect probes cannot tell a dying listener from a
    healthy one.
    """

    deadline = clock() + max(0.0, float(timeout_s))
    while True:
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            probe.bind((host, int(port)))
            return True
        except OSError:
            if clock() >= deadline:
                return False
        finally:
            probe.close()
        sleeper(max(0.05, float(poll_s)))


def _rpc_endpoint_is_ready(endpoint: str, *, timeout: float = 0.5) -> bool:
    if os.environ.get("VERATHOS_LLAMA_RPC_TCP_PROBE", "").strip() != "1":
        return True
    host, port_raw = normalize_rpc_endpoint(endpoint).rsplit(":", 1)
    try:
        with socket.create_connection((host, int(port_raw)), timeout=timeout):
            return True
    except OSError:
        return False


def _llama_extra_arg_value_from(
    extra_args: list[str] | None, *names: str
) -> str | None:
    values = [str(item) for item in (extra_args or [])]
    for idx, item in enumerate(values):
        if item in names and idx + 1 < len(values):
            return values[idx + 1]
        for name in names:
            if item.startswith(name + "="):
                return item.split("=", 1)[1]
    return None


def llama_n_parallel_from_args(args: argparse.Namespace) -> int:
    raw = _llama_extra_arg_value_from(
        getattr(args, "llama_extra_arg", None), "--parallel", "-np"
    )
    try:
        return max(1, int(raw)) if raw is not None else 1
    except ValueError:
        return 1


def tensor_split_for_rpc_plan(plan: Any, requested: str = "") -> str:
    """Use the committed ordered split or reject a conflicting override."""

    normalized_requested = ",".join(
        item.strip()
        for item in str(requested or "").split(",")
        if item.strip()
    )
    if not plan.rpc_endpoints:
        return normalized_requested
    committed = str(plan.tensor_split_arg)
    if normalized_requested and normalized_requested != committed:
        raise ValueError(
            "configured llama tensor split does not match the committed RPC plan"
        )
    return committed


def committed_rpc_stage_count(spec: MeshSpec) -> int:
    """Count members that have committed an RPC endpoint.

    The serve supervisor decides the join wait state from this BEFORE the
    strict all-RPC binding validation runs. While a multi-box mesh is still
    admitting members, the joined subset forms a valid smaller all-RPC plan
    (one member = the whole model), so validating the launch config against
    it raises "llama device order does not match the committed all-RPC plan"
    even though nothing is wrong. That state must read as waiting, never as
    a plan error.
    """

    return sum(1 for member in spec.members if member.rpc_endpoint)


def validate_all_rpc_runtime_binding(
    spec: MeshSpec,
    plan: Any,
    *,
    device: str,
    n_gpu_layers: Any,
) -> None:
    """Require runtime flags that exactly realize an all-RPC assignment."""

    compute_members = [
        member for member in spec.members if member.layers.end > member.layers.start
    ]
    coordinator_members = [
        member for member in spec.members if member.role == "coordinator"
    ]
    all_rpc = bool(compute_members) and (
        len(coordinator_members) == 1
        and coordinator_members[0].layers.end
        == coordinator_members[0].layers.start
        and len(plan.rpc_endpoints) == len(compute_members)
    )
    if not all_rpc:
        return
    requested_devices = ",".join(
        item.strip() for item in str(device or "").split(",") if item.strip()
    )
    committed_devices = ",".join(
        f"RPC{index}" for index in range(plan.total_devices)
    )
    if requested_devices != committed_devices:
        raise ValueError(
            "llama device order does not match the committed all-RPC plan: "
            f"expected {committed_devices}"
        )
    if str(n_gpu_layers or "").strip().lower() != "all":
        raise ValueError(
            "committed all-RPC meshes require --llama-n-gpu-layers all"
        )


def decide_trace_manifest_format(args: argparse.Namespace, *, n_parallel: int = 1) -> str:
    """Single source of truth for the proof-capture manifest format.

    The coordinator pins the result into the mesh spec
    (``proof_trace_manifest_format``) so every member of a multi-node mesh
    runs the same profile: a mixed mesh (one member writing op-manifest rows,
    another not) fails aggregate verification with "proof payload is missing
    op manifest membership".
    """

    # Slot-view attribution (--parallel > 1) reconstructs each request's
    # batch slices from the op manifest, so it needs v3 rows + graph markers
    # whatever the audit profile. This must be decided before the
    # candidates-only "none" profile below, which predates slot view: a
    # parallel serve pinned to "none" captures no manifest rows at all and
    # every stage commitment fails with "no v3 manifest rows for the slot
    # view op template".
    if n_parallel > 1:
        return "compact-raw-v3"
    # A decode audit is additive to the base GGML proof.  When both gates are
    # configured the native trace must retain the full v3 op manifest used by
    # the base challenge as well as the final-logit rows used by a sampled
    # decode audit.  The legacy ``decode`` profile is decode-only and would
    # otherwise discard the base-proof capture on every organic request.
    if bool(args.require_proof and args.proof_sample_bps > 0):
        # Base light/hard proofs alone need the compact manifest: slot-view
        # leaves verify against the op manifest even when the decode-audit
        # rate is zero .
        return "compact-raw-v3"
    if bool(
        args.require_proof
        and args.decode_audit_bps == 0
        and int(args.proof_trace_candidates_per_request) > 0
    ):
        return "none"
    if args.decode_audit_bps == 0 and (args.rpc_worker or args.proof_gguf_manifest):
        return "compact-raw-v3"
    if args.decode_audit_bps > 0 and args.proof_sample_bps == 0:
        return "decode"
    if (
        args.proof_sample_bps >= PROOF_SAMPLE_BPS_DENOMINATOR
        and args.decode_audit_bps == 0
    ):
        return "none"
    return "compact"


def coordinator_computes_from_args(args: argparse.Namespace) -> bool:
    """True when the coordinator's llama-server computes layers locally.

    A device list that is entirely RPC backends (e.g. ``RPC0,RPC1``) offloads
    every layer to workers, making the coordinator orchestration-only.
    """

    if str(getattr(args, "llama_capture_trace_dir", "") or ""):
        # Local-stage serving: the device list is local CUDA, but those
        # layers belong to the FIRST MEMBER's committed stage (llama-server
        # is that member's compute process and captures into its trace dir).
        # The coordinator stays orchestration-only, exactly as in an all-RPC
        # mesh, so stage proofs keep their per-worker identity.
        return False
    devices = [
        item.strip()
        for item in str(getattr(args, "llama_device", "") or "").split(",")
        if item.strip()
    ]
    return not devices or any(not d.upper().startswith("RPC") for d in devices)


def _serving_identity_from_args(
    args: argparse.Namespace,
) -> tuple[Callable[[str], str] | None, str, str, Callable[[bytes], str] | None]:
    """Resolve receipt and EVM identities without loading a coldkey.

    Wallet-backed serving always derives both identities from the serving
    hotkey.  Explicit EVM flags are accepted as a consistency assertion in
    that mode, never as an override of the hotkey-derived identity.  A
    ``--coordinator-sign-file`` (written by the pool worker for a driver
    that holds no wallet) resolves both identities to the POOL MANAGER's
    coordinator wallet, signing over the authenticated worker channel.
    """

    wallet_name = str(getattr(args, "wallet_name", "") or "")
    wallet_hotkey = str(getattr(args, "wallet_hotkey", "") or "")
    evm_address = str(getattr(args, "evm_address", "") or "")
    evm_private_key = str(getattr(args, "evm_private_key", "") or "")
    sign_file = str(getattr(args, "coordinator_sign_file", "") or "").strip()
    if sign_file and wallet_name:
        raise SystemExit(
            "--coordinator-sign-file conflicts with --wallet-name: a machine "
            "holding the coordinator wallet signs locally"
        )

    if bool(wallet_name) != bool(wallet_hotkey):
        raise SystemExit("--wallet-name and --wallet-hotkey must be provided together")

    receipt_signer = None
    if wallet_name:
        from verallm.chain.wallet import derive_evm_address, derive_evm_private_key
        from verallm.mesh.receipt_signing import (
            load_hotkey_keypair,
            load_hotkey_seed,
            sign_receipt_hash,
        )

        keypair = load_hotkey_keypair(wallet_name, wallet_hotkey)
        hotkey_seed = load_hotkey_seed(
            wallet_name,
            wallet_hotkey,
            keypair=keypair,
        )
        derived_address = derive_evm_address(hotkey_seed)
        derived_private_key = derive_evm_private_key(hotkey_seed)
        if evm_address and evm_address.lower() != derived_address.lower():
            raise SystemExit("--evm-address does not match the serving hotkey")
        if (
            evm_private_key
            and evm_private_key.lower().removeprefix("0x")
            != derived_private_key.lower().removeprefix("0x")
        ):
            raise SystemExit("--evm-private-key does not match the serving hotkey")

        args.hotkey = keypair.ss58_address
        receipt_signer = lambda receipt_hash: sign_receipt_hash(receipt_hash, keypair)
        evm_address = derived_address
        evm_private_key = derived_private_key
        print(f"signing receipts with hotkey {args.hotkey}", flush=True)
    elif sign_file:
        if evm_private_key:
            raise SystemExit(
                "--evm-private-key conflicts with --coordinator-sign-file"
            )
        from verallm.mesh.delegated_signing import (
            coordinator_delegation_from_file,
        )
        from verallm.mesh.receipt_signing import sign_receipt_hash

        # The daemon installed its pinned HTTPS opener at join; THIS
        # process starts bare, and an https manager with a self-signed
        # cert fails every delegated signing call without the pin.
        import json as json_module

        try:
            delegation_data = json_module.loads(
                Path(sign_file).read_text(encoding="utf-8")
            )
        except (OSError, ValueError) as exc:
            raise SystemExit(
                f"cannot read --coordinator-sign-file: {exc}"
            ) from exc
        manager_endpoint = str(
            delegation_data.get("manager_endpoint", "") or ""
        )
        ca_sha256 = str(delegation_data.get("manager_ca_sha256", "") or "")
        if manager_endpoint.startswith("https://") and ca_sha256:
            from verallm.mesh.pool import MeshPoolToken

            _install_manager_tls_pin(
                MeshPoolToken(
                    pool_id=str(delegation_data.get("pool_id", "") or "x"),
                    manager_endpoint=manager_endpoint,
                    pool_secret=str(
                        delegation_data.get("pool_secret", "") or "x"
                    ),
                    manager_ca_sha256=ca_sha256,
                )
            )
        elif manager_endpoint.startswith("https://"):
            manager_ca_file = str(delegation_data.get("manager_ca_file", "") or "")
            if manager_ca_file:
                context = _manager_ca_file_context(manager_ca_file)
                _install_per_host_manager_tls_context(manager_endpoint, context)
        delegation = coordinator_delegation_from_file(sign_file)
        if (
            evm_address
            and evm_address.lower() != delegation.evm_address.lower()
        ):
            raise SystemExit(
                "--evm-address does not match the pool coordinator identity"
            )
        args.hotkey = delegation.coordinator_hotkey
        receipt_signer = lambda receipt_hash: sign_receipt_hash(
            receipt_hash, delegation.keypair
        )
        print(
            "signing receipts via the pool manager as "
            f"{delegation.coordinator_hotkey}",
            flush=True,
        )
        return (
            receipt_signer,
            delegation.evm_address,
            "",
            delegation.challenge_signer,
        )
    elif bool(evm_address) != bool(evm_private_key):
        raise SystemExit(
            "--evm-address and --evm-private-key must be provided together"
        )

    return receipt_signer, evm_address, evm_private_key, None


def _stage_proof_identity_from_args(
    args: argparse.Namespace,
) -> tuple[Callable[[str], str] | None, str]:
    """Load the dedicated proof-worker signer, never the coordinator hotkey."""

    key_file = str(getattr(args, "stage_proof_key_file", "") or "").strip()
    if not key_file:
        return None, ""
    from verallm.mesh.receipt_signing import (
        STAGE_PROOF_KEY_SCHEME,
        load_stage_proof_keypair_file,
        sign_stage_proof_receipt_body_hash,
    )

    try:
        keypair = load_stage_proof_keypair_file(key_file)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"cannot load --stage-proof-key-file: {exc}") from exc
    proof_key = str(keypair.ss58_address)
    if (
        str(getattr(args, "wallet_name", "") or "").strip()
        and proof_key == str(getattr(args, "hotkey", "") or "").strip()
    ):
        raise SystemExit(
            "--stage-proof-key-file must not reuse the coordinator/miner hotkey"
        )
    signer = lambda body_hash: sign_stage_proof_receipt_body_hash(
        body_hash,
        keypair,
        expected_proof_key=proof_key,
        proof_key_scheme=STAGE_PROOF_KEY_SCHEME,
    )
    return signer, proof_key


class BackendCrashBreaker:
    """Decides when a respawning backend has proven it will not come up.

    Counts CONSECUTIVE failed starts, with no time bound. The previous rule
    (N crashes inside a 120s window) could not see a slow loop at all: a
    238GB model takes ~2 min to load before it OOMs, so every crash landed
    outside the window, the window emptied between crashes, and the breaker
    never tripped. That left a mesh retrying a hopeless load for half an
    hour while the pool reported "driving", and it starved the KV auto-fit
    upstream, which only descends once a backend failure is reported.

    A consecutive count subsumes the rate rule: a fast loop reaches the
    limit immediately, and a slow one still reaches it. What matters is
    whether the backend ever came up, not how quickly it failed.

    A component that ran for HEALTHY_UPTIME_S before dying was serving, not
    failing to start, so its crash begins a new streak rather than counting
    toward one.
    """

    WINDOW_S = 120.0  # reported with the failure, no longer a trip condition
    CONSECUTIVE_LIMIT = 3
    HEALTHY_UPTIME_S = 600.0

    def __init__(self, now: Callable[[], float] = time.monotonic) -> None:
        self._now = now
        self._times: dict[str, list[float]] = {}
        self._streak: dict[str, int] = {}

    def note(
        self, component: str, *, uptime_s: float = 0.0
    ) -> tuple[bool, int, str]:
        """Record one crash; return (tripped, crash_count, reason)."""

        now = self._now()
        times = self._times.setdefault(component, [])
        times.append(now)
        while times and now - times[0] > self.WINDOW_S:
            times.pop(0)
        if uptime_s >= self.HEALTHY_UPTIME_S:
            self._streak[component] = 1
        else:
            self._streak[component] = self._streak.get(component, 0) + 1
        streak = self._streak[component]
        if streak >= self.CONSECUTIVE_LIMIT:
            recent = len(times)
            detail = f" ({recent}x within {self.WINDOW_S:.0f}s)" if recent > 1 else ""
            return True, streak, f"{streak}x in a row{detail}"
        return False, streak, ""


def cmd_serve(args: argparse.Namespace) -> None:
    # kill -USR1 <pid> dumps every thread's stack to stderr: the one
    # diagnostic that localizes a silent hang (a request thread blocked on
    # a dead dial or a lock) on machines where ptrace is unavailable.
    import faulthandler
    import signal as _signal

    try:
        faulthandler.register(_signal.SIGUSR1)
    except (ValueError, OSError):
        pass
    (
        receipt_signer,
        evm_address,
        evm_private_key,
        evm_challenge_signer,
    ) = _serving_identity_from_args(args)
    stage_receipt_signer, stage_proof_key = _stage_proof_identity_from_args(args)

    join_handler = None
    mesh_spec_handler = None
    mesh_update_handler = None
    mesh_spec_loader = None
    mesh_state_role = ""
    internal_auth_secret = ""
    persisted_state_path: Path | None = None
    if args.mesh and _state_exists(args.mesh):
        state_path = mesh_state_path(args.mesh)
        persisted_state_path = state_path
        state = load_mesh_state(state_path)
        mesh_state_role = str(state.get("role", ""))
        internal_auth_secret = state_internal_auth_secret(state)
        state_dir = state_path.parent
        if state.get("role") == "worker":
            try:
                refresh_worker_mesh_state(state_dir, timeout=args.mesh_sync_timeout)
                state = load_mesh_state(state_path)
            except Exception as exc:
                print(f"mesh spec refresh skipped: {exc}", file=sys.stderr, flush=True)
        mesh_spec = state_mesh_spec(state)
        capability = state_capabilities(state)[0]
        mesh_spec_loader = lambda: state_mesh_spec(load_mesh_state(state_path))
        if state.get("role") == "coordinator":
            # Pin the mesh-wide proof profile and record whether this
            # coordinator computes layers locally, so worker admissions
            # (assign_mesh_members) and every member's proof runtime agree.
            computes = coordinator_computes_from_args(args)
            pinned = (
                decide_trace_manifest_format(
                    args, n_parallel=llama_n_parallel_from_args(args)
                )
                if args.require_proof
                else ""
            )
            current_pin = str(state["mesh"].get("proof_trace_manifest_format", ""))
            if (
                bool(state.get("coordinator_computes", True)) != computes
                or (pinned and current_pin != pinned)
            ):
                state["coordinator_computes"] = computes
                if pinned:
                    state["mesh"]["proof_trace_manifest_format"] = pinned
                save_mesh_state(state_dir, state)
                save_json(
                    mesh_state_path(state_dir).with_name(MESH_SPEC_FILE),
                    state["mesh"],
                )
                mesh_spec = state_mesh_spec(state)
        if state.get("role") == "coordinator" and state.get("join_secret"):
            join_secret = str(state["join_secret"])
            coordinator_endpoint = capability.endpoint.rstrip("/")
            mesh_admission_lock = threading.Lock()

            def mesh_spec_handler(body: dict) -> dict:
                if str(body.get("join_secret", "")) != join_secret:
                    raise PermissionError("invalid join token")
                current = state_mesh_spec(load_mesh_state(state_path))
                requested_mesh_id = str(body.get("mesh_id", ""))
                if requested_mesh_id and requested_mesh_id != current.mesh_id:
                    raise ValueError("mesh_id mismatch")
                return {
                    "status": "ok",
                    "mesh": current.to_dict(),
                    "mesh_spec_hash": current.spec_hash_hex(),
                    "stage_assignment_hash": current.stage_assignment_hash_hex(),
                }

            def _join_handler_locked(body: dict) -> dict:
                if str(body.get("join_secret", "")) != join_secret:
                    raise PermissionError("invalid join token")
                current_state = load_mesh_state(state_path)
                expected_compute_stages = int(
                    current_state.get("expected_compute_stage_count", 0) or 0
                )
                current_compute_stages = state_admitted_compute_stage_count(
                    current_state
                )
                joined_capability = CapabilityAd.from_dict(body["capability"])
                current_capabilities = state_capabilities(current_state)
                admitted_at_endpoint = [
                    capability
                    for capability in current_capabilities
                    if capability.endpoint == joined_capability.endpoint
                ]
                already_admitted = bool(admitted_at_endpoint)
                exact_replay = any(
                    capability.ad_hash_hex() == joined_capability.ad_hash_hex()
                    for capability in admitted_at_endpoint
                )
                if current_state.get("mesh_finalized"):
                    if not exact_replay:
                        raise PermissionError("mesh member assignment is finalized")
                    # The first response may be lost after admission and final
                    # snapshot publication race one another.  Return the
                    # immutable result only for the byte-exact capability that
                    # was already admitted; never reopen assignment or broadcast
                    # a mutable update after finalization.
                    current = state_mesh_spec(current_state)
                    return {
                        "status": "joined",
                        "mesh": current.to_dict(),
                        "mesh_spec_hash": current.spec_hash_hex(),
                        "stage_assignment_hash": current.stage_assignment_hash_hex(),
                        "mesh_update_errors": [],
                    }
                if (
                    expected_compute_stages > 0
                    and current_compute_stages >= expected_compute_stages
                    and not already_admitted
                ):
                    raise PermissionError("mesh already has its assigned compute stages")
                updated = admit_mesh_worker(state_dir, joined_capability)
                updated_compute_stages = sum(
                    member.layers.end > member.layers.start
                    for member in updated.members
                )
                if (
                    expected_compute_stages > 0
                    and updated_compute_stages > expected_compute_stages
                ):
                    raise RuntimeError("mesh admission exceeded assigned compute stages")
                update_errors = _broadcast_mesh_update(
                    spec=updated,
                    join_secret=join_secret,
                    skip_endpoints={
                        coordinator_endpoint,
                        joined_capability.endpoint.rstrip("/"),
                    },
                    timeout=args.mesh_sync_timeout,
                )
                return {
                    "status": "joined",
                    "mesh": updated.to_dict(),
                    "mesh_spec_hash": updated.spec_hash_hex(),
                    "stage_assignment_hash": updated.stage_assignment_hash_hex(),
                    "mesh_update_errors": update_errors,
                }

            def join_handler(body: dict) -> dict:
                # ThreadingHTTPServer may receive multiple member joins at
                # once. Serialize the read/assign/write/broadcast transaction
                # so capabilities cannot be lost or reordered by races.
                with mesh_admission_lock:
                    return _join_handler_locked(body)
        elif state.get("role") == "worker":
            capability = state_capabilities(state)[0]
            join_token = MeshJoinToken.decode(str(state.get("join_token", "")))

            def mesh_update_handler(body: dict) -> dict:
                if str(body.get("join_secret", "")) != join_token.join_secret:
                    raise PermissionError("invalid join token")
                spec = MeshSpec.from_dict(body["mesh"])
                requested_mesh_id = str(body.get("mesh_id", ""))
                if requested_mesh_id and requested_mesh_id != spec.mesh_id:
                    raise ValueError("mesh_id mismatch")
                requested_spec_hash = str(body.get("mesh_spec_hash", ""))
                if requested_spec_hash and requested_spec_hash != spec.spec_hash_hex():
                    raise ValueError("mesh_spec_hash mismatch")
                requested_assignment_hash = str(body.get("stage_assignment_hash", ""))
                if (
                    requested_assignment_hash
                    and requested_assignment_hash != spec.stage_assignment_hash_hex()
                ):
                    raise ValueError("stage_assignment_hash mismatch")
                updated = update_worker_mesh_state(state_dir, spec)
                return {
                    "status": "updated",
                    "mesh_id": updated.mesh_id,
                    "mesh_spec_hash": updated.spec_hash_hex(),
                    "stage_assignment_hash": updated.stage_assignment_hash_hex(),
                }
    else:
        mesh_spec = _load_mesh_spec(args.mesh) if args.mesh else None
        missing = [
            name
            for name in ("uid", "hotkey", "endpoint")
            if getattr(args, name, None) in (None, "")
        ]
        if missing:
            raise SystemExit("--mesh or all of --uid, --hotkey, and --endpoint are required")
        capability = _capability_from_args(args)
    from verallm.mesh import local_dial

    # Only the connection address changes; capabilities and plans stay public.
    local_dial.configure(capability.endpoint)
    local_advertised_host = urlparse(capability.endpoint).hostname or ""
    verification_snapshot_loader = _snapshot_loader_for_serve(
        explicit_path=str(args.verification_snapshot or ""),
        state_path=persisted_state_path,
        required=bool(args.require_verification_snapshot),
    )
    backend_url = args.backend_url
    args.proof_sample_bps = normalize_proof_sample_bps(args.proof_sample_bps)
    args.decode_audit_bps = normalize_proof_sample_bps(args.decode_audit_bps)
    args.decode_audit_top_k = max(1, int(args.decode_audit_top_k))
    args.proof_tolerance_abs = float(args.proof_tolerance_abs)
    args.proof_tolerance_rel = float(args.proof_tolerance_rel)
    args.proof_artifact_timeout = float(args.proof_artifact_timeout)
    args.proof_ops_per_request = int(args.proof_ops_per_request)
    args.proof_trace_candidates_per_request = int(
        args.proof_trace_candidates_per_request
    )
    if args.proof_ops_per_request < 0:
        raise SystemExit("--proof-ops-per-request must be >= 0")
    if args.proof_trace_candidates_per_request < 0:
        raise SystemExit("--proof-trace-candidates-per-request must be >= 0")
    if args.proof_trace_max_elems < 0:
        raise SystemExit("--proof-trace-max-elems must be >= 0")
    if args.proof_tolerance_abs <= 0:
        raise SystemExit("--proof-tolerance-abs must be > 0")
    if args.proof_tolerance_rel <= 0:
        raise SystemExit("--proof-tolerance-rel must be > 0")
    if args.proof_artifact_timeout <= 0:
        raise SystemExit("--proof-artifact-timeout must be > 0")
    if args.require_proof and args.proof_sample_bps > 0 and args.proof_ops_per_request < 1:
        raise SystemExit("--proof-ops-per-request must be >= 1 when proof sampling is enabled")
    if args.decode_audit_bps > 0 and not args.require_proof:
        raise SystemExit("--decode-audit-bps requires --require-proof")
    args.proof_trace_candidates_per_request = max(
        args.proof_trace_candidates_per_request,
        args.proof_ops_per_request,
    )
    if args.require_proof and not args.proof_url:
        # Proofs are generated in THIS process (not delegated to a remote
        # --proof-url prover), so the native sumcheck backend is mandatory.
        # Fail closed rather than silently serve on the slow NumPy fallback: a
        # miner that fell back would look healthy while running the unaudited,
        # orders-of-magnitude-slower path.
        from zkllm.crypto.sumcheck_fast import assert_native_backend

        try:
            assert_native_backend("verified mesh serving")
        except RuntimeError as exc:
            raise SystemExit(str(exc))
    if args.require_proof:
        if args.rpc_worker and not is_proof_capable_rpc_worker_binary(args.rpc_worker_binary):
            raise SystemExit(
                "--require-proof requires a proof-capable RPC worker binary such as "
                "verathos-rpc-server; raw llama.cpp rpc-server is receipt-only"
            )
        if args.rpc_worker and not args.proof_trace_dir:
            args.proof_trace_dir = str(_default_trace_dir(args.mesh))
        proof_source_configured = (
            args.proof_url or args.proof_trace_dir or _mesh_has_proof_endpoint(mesh_spec)
        )
        proof_source_expected_after_join = (
            mesh_state_role == "coordinator" and args.llama_min_rpc_workers > 0
        )
        if (args.backend_url or args.llama_model or args.llama_hf) and not proof_source_configured:
            if not proof_source_expected_after_join:
                raise SystemExit(
                    "--require-proof with --backend-url, --llama-model, or --llama-hf requires "
                    "--proof-url, --proof-trace-dir, or member proof_endpoint entries in the mesh"
                )
    llama_proc = None
    rpc_proc = None
    llama_plan_hash = ""
    llama_stop = threading.Event()
    llama_lock = threading.Lock()
    rpc_cmd = None
    rpc_payload = None
    proof_trace_enable_file = ""
    proof_runtime_env = None
    proof_trace_path = (
        Path(args.proof_trace_dir) if args.require_proof and args.proof_trace_dir else None
    )
    if proof_trace_path is not None:
        proof_trace_enable_file = str(proof_trace_path / ".capture-enabled")

        # Trace janitor: capture dumps are consumed by the inline proof within
        # seconds of being written, but nothing deleted them afterwards — trace
        # dirs grew without bound (hundreds of MB/day per member; fatal on a
        # disk-tight box). TTL is generous vs the seconds-scale consumption so
        # a slow or deferred audit can never lose its witness.
        def _trace_janitor(root: Path = proof_trace_path) -> None:
            ttl_s = float(os.environ.get("VERATHOS_TRACE_TTL_S", "3600"))
            while True:
                try:
                    cutoff = time.time() - ttl_s
                    # os.scandir, not iterdir + is_file + stat: pathlib costs
                    # TWO stat syscalls per file plus a Path object, and this
                    # sweeps a directory holding thousands of dumps while the
                    # proof path wants the GIL. scandir carries the dirent's
                    # type and caches the stat, so one pass is one syscall per
                    # file. Profiled: the janitor was the single largest
                    # consumer of Python CPU in a serving member.
                    with os.scandir(root) as entries:
                        for entry in entries:
                            if entry.name.startswith("."):
                                continue  # never touch .capture-enabled
                            try:
                                if (
                                    entry.is_file(follow_symlinks=False)
                                    and entry.stat().st_mtime < cutoff
                                ):
                                    os.unlink(entry.path)
                            except OSError:
                                pass
                except Exception:
                    pass  # janitor must never take the server down
                time.sleep(300.0)

        threading.Thread(target=_trace_janitor, daemon=True).start()

    def _llama_extra_arg_value(*names: str) -> str | None:
        return _llama_extra_arg_value_from(args.llama_extra_arg, *names)

    def llama_n_parallel_setting() -> int:
        return llama_n_parallel_from_args(args)

    def llama_n_ubatch_setting() -> int:
        raw = _llama_extra_arg_value("--ubatch-size", "-ub")
        try:
            # llama-server default ubatch size is 512.
            return max(1, int(raw)) if raw is not None else 512
        except ValueError:
            return 512

    def suggested_trace_max_elems() -> int:
        from verallm.mesh.gguf_manifest import (
            GGML_TRACE_MAX_ELEMS_FLOOR,
            load_gguf_tensor_manifest,
            suggest_ggml_trace_max_elems_from_gguf_model,
            suggest_ggml_trace_max_elems_from_manifest,
        )

        if args.proof_trace_max_elems > 0:
            return int(args.proof_trace_max_elems)
        # Slot-view meshes (--parallel > 1) capture no dumps at serve time,
        # so the quartile cap only shrinks the challenge universe below the
        # anti-shrink floor and starves the decode-audit replay; cover every
        # provable tensor there instead. Single-slot audited serves do NOT
        # need full range: v3 manifests write a row for every dim-valid op
        # regardless of this cap (the cap bounds only the serve-time
        # candidate dumps). Raising it here instead was live-measured to
        # quadruple full-context prefill TTFT (15.5s to 69.6s at 26k
        # tokens) from multi-MB activation dumps of ffn-sized ops.
        full_range = llama_n_parallel_setting() > 1
        if args.proof_gguf_manifest:
            try:
                manifest = load_gguf_tensor_manifest(args.proof_gguf_manifest)
                return suggest_ggml_trace_max_elems_from_manifest(
                    manifest, full_range=full_range
                )
            except Exception:
                pass
        if args.llama_model:
            return suggest_ggml_trace_max_elems_from_gguf_model(
                args.llama_model, full_range=full_range
            )
        if args.llama_hf:
            suggested = _suggest_ggml_trace_max_elems_from_model_ref(args.llama_hf)
            if suggested is not None:
                return suggested
        if mesh_spec is not None:
            suggested = _suggest_ggml_trace_max_elems_from_model_ref(
                mesh_spec.model_package_ref
            )
            if suggested is not None:
                return suggested
        if args.rpc_worker:
            return GGML_RPC_TRACE_MAX_ELEMS_FALLBACK
        return GGML_TRACE_MAX_ELEMS_FLOOR

    def get_proof_runtime_env() -> dict[str, str] | None:
        nonlocal proof_runtime_env
        if proof_trace_path is None:
            return None
        if proof_runtime_env is not None:
            return proof_runtime_env
        proof_trace_path.mkdir(parents=True, exist_ok=True)
        proof_trace_enable_path = proof_trace_path / ".capture-enabled"
        proof_trace_enable_path.unlink(missing_ok=True)
        proof_runtime_env = dict(os.environ)
        proof_runtime_env["VERATHOS_GGML_TRACE_DIR"] = str(proof_trace_path)
        proof_runtime_env["VERATHOS_GGML_TRACE_ENABLE_FILE"] = str(proof_trace_enable_path)
        # Pin the backend's capture budget to exactly the candidate count the
        # proof commits (the verifier samples 1 of these). Without this each
        # backend uses its compiled default — Metal's is 64 vs 8 elsewhere —
        # so a Metal member dumped 8x more witnesses per request than the
        # protocol can ever use: pure GPU-sync + disk cost on the hot path.
        proof_runtime_env.setdefault(
            "VERATHOS_GGML_TRACE_MAX_OPS_PER_CAPTURE",
            str(max(1, int(args.proof_trace_candidates_per_request))),
        )
        # Stage-boundary activation capture must be armed wherever proof
        # capture is configured: the validator's boundary-chain requirement
        # (VERATHOS_MESH_REQUIRE_BOUNDARY_CHAIN, default ON) needs every
        # middle stage to report roots, and the patched runtime only writes
        # boundary.jsonl when this is set. Until now only the e2e harness
        # exported it, so a production mesh would have shipped receipts with
        # empty roots and failed the chain check at the first validator.
        proof_runtime_env.setdefault("VERATHOS_GGML_BOUNDARY_CAPTURE", "1")
        # Proof capture requires a DETERMINISTIC op stream. Upstream's CUDA
        # decode-GEMV fusions (ffn gate+up+glu, projection+add) splice ops
        # out of the graph depending on runtime conditions, so the same
        # request can capture an op in one window and lose it in the next;
        # a slot-view leaf drawn on a fused-away op then fails an HONEST
        # node ("no single-row decode instance captured", observed on
        # glm-dsa shared-expert and attention projections). Upstream ships
        # the kill-switch for exactly this; the cost is a few percent of
        # decode throughput on GEMV-bound models, only on proof-armed
        # serves. Operators can override with GGML_CUDA_DISABLE_FUSION=0
        # at their own risk.
        proof_runtime_env.setdefault("GGML_CUDA_DISABLE_FUSION", "1")
        # Deliberately NOT arming VERATHOS_GGML_RPC_GRAPH_TRACE: the rpc hook
        # dumps its op after the WHOLE graph ran, but ggml reuses intermediate
        # buffers within a graph, so mid-graph mirror witnesses carry stale
        # activations and fail float recomputation (~1.0 rel error) whenever
        # the beacon draws one — a guaranteed chat 500 under a wide candidate
        # window. Each stage's own at-execution hook (cpu/cuda/metal) already
        # dumps every op consistently; the mirror added no coverage.
        _mesh_has_rpc_members = bool(
            mesh_spec is not None
            and any(member.rpc_endpoint for member in mesh_spec.members)
        )
        # Every base-proof compute stage captures its own candidate witnesses
        # during serve and proves a stored candidate at audit
        # (trace_candidate_set_v1), without replay. This covers a single-node
        # coordinator, each rpc-worker, and a coordinator computing some layers
        # alongside remote workers. Base-only capture can remain manifest-free;
        # an additive decode audit keeps the candidates but uses compact-raw-v3
        # for the extra final-logit rows.
        _candidate_stage = bool(
            args.require_proof
            and args.proof_sample_bps > 0
            and int(args.proof_trace_candidates_per_request) > 0
        )
        pinned_format = (
            str(getattr(mesh_spec, "proof_trace_manifest_format", "") or "")
            if mesh_spec is not None
            else ""
        )
        if pinned_format:
            # The coordinator pinned the mesh-wide profile in the spec: it
            # overrides both the local decision and any inherited env so every
            # member captures identically.
            manifest_format = pinned_format
            proof_runtime_env["VERATHOS_GGML_TRACE_MANIFEST_FORMAT"] = manifest_format
        else:
            manifest_format = decide_trace_manifest_format(
                args, n_parallel=llama_n_parallel_setting()
            )
            proof_runtime_env.setdefault(
                "VERATHOS_GGML_TRACE_MANIFEST_FORMAT", manifest_format
            )
        manifest_only_sampled = (
            args.proof_sample_bps < PROOF_SAMPLE_BPS_DENOMINATOR
            and args.decode_audit_bps < PROOF_SAMPLE_BPS_DENOMINATOR
        )
        decode_full_audit_only = (
            manifest_format == "decode"
            and args.decode_audit_bps >= PROOF_SAMPLE_BPS_DENOMINATOR
        )
        # Capture candidate witnesses DURING serve so the audit proves from
        # stored candidates (trace_candidate_set_v1) instead of regenerating
        # the completion. Replay is O(generation-length) and diverges on MoE
        # (non-deterministic expert routing), so it must not be the proof
        # path. A leaf candidate stage (single-node coordinator or rpc-worker)
        # captures its 8 candidates; the legacy compact formats keep the old
        # behavior.
        # Slot-view serves take NO serve-time candidate dumps: witnesses
        # come from slot-view leaves, the tail ring, and the exclusive
        # probe window (whose selected-op dumps bypass this budget), and
        # the wide candidate window exists only as the beacon-selectable
        # ROW space. Leaving the dump budget coupled to that window was
        # live-measured writing 29 GB of activation dumps during ONE
        # 31k-token prefill and tripling its wall time; zeroing the
        # budget restored full engine speed (407 -> 1216 tok/s) with the
        # whole proof stack unchanged.
        _slot_view_serve = bool(
            args.require_proof
            and manifest_format == "compact-raw-v3"
            and (
                args.decode_audit_bps > 0
                or llama_n_parallel_setting() > 1
            )
        )
        trace_candidates = (
            0
            if _slot_view_serve
            else args.proof_trace_candidates_per_request
            if _candidate_stage
            else 0
            if args.rpc_worker
            or manifest_format == "compact"
            or manifest_format == "compact-raw"
            or manifest_format == "compact-raw-v2"
            or manifest_only_sampled
            or decode_full_audit_only
            else args.proof_trace_candidates_per_request
        )
        proof_runtime_env["VERATHOS_GGML_TRACE_MAX_OPS_PER_CAPTURE"] = str(trace_candidates)
        proof_runtime_env["VERATHOS_GGML_TRACE_MAX_OPS_PER_GRAPH"] = str(trace_candidates)
        proof_runtime_env.setdefault(
            "VERATHOS_GGML_TRACE_MAX_ELEMS",
            str(suggested_trace_max_elems()),
        )
        if args.proof_gguf_manifest:
            proof_runtime_env.setdefault("VERATHOS_GGML_TRACE_SKIP_SRC0_DUMP", "1")
        if os.environ.get(
            "VERATHOS_MESH_LIGHT_TAIL_CAPTURE", "1"
        ).strip().lower() in ("0", "false", "no"):
            # One operator switch reverts the probe-free light tier end to
            # end: the worker stops preferring tail material AND the runtime
            # stops ring-buffering final-logit rows.
            proof_runtime_env["VERATHOS_GGML_TAIL_RING"] = "0"
        elif llama_n_parallel_setting() > 1:
            # The tail ring binds positions by instance order, which only
            # holds with ONE backend slot; concurrent slots interleave
            # generations through the one process-wide ring. Ring off:
            # audit-drawn lights take the certified probe path and base
            # light leaves stay synthesized (capture-free at any slot
            # count), so concurrency never rides on ring ordering.
            proof_runtime_env["VERATHOS_GGML_TAIL_RING"] = "0"
        else:
            # The runtime's 60ms idle flush predates the reply-end signal
            # file and splits one generation's ring across drains whenever
            # the pipeline stalls longer than a token gap (cold caches, the
            # prefill-to-decode boundary). Only a reply-final group maps
            # tail_seq to positions, so a split ring cannot cover an audit
            # draw and every audit-drawn light falls back to probes. With
            # the signal handling reply end, idle is a pure safety net.
            proof_runtime_env.setdefault(
                "VERATHOS_GGML_TAIL_RING_FLUSH_MS", "750"
            )
        return proof_runtime_env

    def get_llama_capture_env() -> dict[str, str] | None:
        """Env for the llama-server child; local-stage mode redirects capture.

        With --llama-capture-trace-dir, llama-server IS the first member's
        compute process: its capture must ride the MEMBER's trace dir and
        arming file (owned and armed by the member serve process over the
        normal trace-capture channel), while the coordinator's own
        --proof-trace-dir stays separate for window bookkeeping. The enable
        file is deliberately never unlinked here - the member owns it.
        """
        base = get_proof_runtime_env()
        capture_dir = str(getattr(args, "llama_capture_trace_dir", "") or "")
        if not capture_dir:
            return base
        env = dict(base) if base is not None else dict(os.environ)
        capture_path = Path(capture_dir)
        capture_path.mkdir(parents=True, exist_ok=True)
        env["VERATHOS_GGML_TRACE_DIR"] = str(capture_path)
        env["VERATHOS_GGML_TRACE_ENABLE_FILE"] = str(
            capture_path / ".capture-enabled"
        )
        return env

    if args.rpc_worker:
        rpc_cmd = build_rpc_worker_command(
            binary=args.rpc_worker_binary,
            host=args.rpc_host,
            port=args.rpc_port,
            device=args.rpc_device,
            cache=args.rpc_cache,
            extra_args=args.rpc_extra_arg,
        )
        rpc_payload = {
            "rpc_worker_command": rpc_cmd,
            "rpc_worker_command_text": command_preview(rpc_cmd),
            "proof_required": bool(args.require_proof),
            "proof_sample_bps": int(args.proof_sample_bps),
            "proof_ops_per_request": int(args.proof_ops_per_request),
            "proof_trace_candidates_per_request": int(
                args.proof_trace_candidates_per_request
            ),
            "decode_audit_bps": int(args.decode_audit_bps),
            "decode_audit_top_k": int(args.decode_audit_top_k),
            "proof_capable_rpc_worker": is_proof_capable_rpc_worker_binary(
                args.rpc_worker_binary
            ),
            "proof_trace_dir": args.proof_trace_dir,
            "proof_collection": _proof_collection_label(args, mesh_spec),
        }
        if args.rpc_dry_run and not (args.llama_model or args.llama_hf):
            runtime_env = get_proof_runtime_env()
            if runtime_env is not None:
                rpc_payload["proof_runtime_env"] = {
                    key: runtime_env[key]
                    for key in (
                        "VERATHOS_GGML_TRACE_DIR",
                        "VERATHOS_GGML_TRACE_ENABLE_FILE",
                        "VERATHOS_GGML_TRACE_MAX_OPS_PER_CAPTURE",
                        "VERATHOS_GGML_TRACE_MAX_OPS_PER_GRAPH",
                        "VERATHOS_GGML_TRACE_MANIFEST_FORMAT",
                        "VERATHOS_GGML_TRACE_MAX_ELEMS",
                        "VERATHOS_GGML_TRACE_SKIP_SRC0_DUMP",
                        "VERATHOS_GGML_RPC_GRAPH_TRACE",
                        "VERATHOS_GGML_TAIL_RING",
                        "VERATHOS_GGML_TAIL_RING_FLUSH_MS",
                        "GGML_CUDA_DISABLE_FUSION",
                    )
                    if key in runtime_env
                }
            _print_json(rpc_payload)
            return
    if args.llama_model or args.llama_hf:
        if args.backend_url:
            raise SystemExit("--backend-url and llama.cpp launch options are mutually exclusive")
        if args.llama_model and args.llama_hf:
            raise SystemExit("--llama-model and --llama-hf are mutually exclusive")
        if mesh_spec is None:
            raise SystemExit("--mesh is required when using --llama-model or --llama-hf")
        if getattr(args, "slot_state_dir", ""):
            # Slot-state save/restore for bounded hard-audit probes. llama
            # refuses to boot on a missing --slot-save-path directory, so
            # create it first. The /slots actions this activates live on the
            # llama aux port, which binds loopback (--llama-host default);
            # nothing remote dials it - members use rpc/proof/mesh ports.
            # NOT named state_dir: cmd_serve's join/update handlers close
            # over `state_dir` (the MESH state dir, bound far above), and
            # rebinding it here pointed every member join at
            # <mesh>/slot-states/mesh-state.json.
            slot_save_root = Path(args.slot_state_dir).expanduser()
            slot_save_root.mkdir(parents=True, exist_ok=True)
            args.llama_extra_arg = list(args.llama_extra_arg or []) + [
                "--slot-save-path",
                str(slot_save_root),
                "--slots",
            ]

        def llama_payload_for(spec: MeshSpec) -> dict:
            plan = rpc_plan_from_mesh(spec)
            validate_all_rpc_runtime_binding(
                spec,
                plan,
                device=args.llama_device,
                n_gpu_layers=args.llama_n_gpu_layers,
            )
            tensor_split = tensor_split_for_rpc_plan(
                plan,
                args.llama_tensor_split,
            )
            ctx_size = _mesh_bound_ctx_size(
                spec,
                args.llama_ctx_size,
                flag_name="--llama-ctx-size",
            )
            cmd = build_llama_server_command(
                binary=args.llama_server_binary,
                model=args.llama_model,
                hf_model=args.llama_hf,
                host=args.llama_host,
                port=args.llama_port,
                rpc_endpoints=plan.rpc_endpoints,
                device=args.llama_device,
                n_gpu_layers=args.llama_n_gpu_layers,
                ctx_size=ctx_size,
                tensor_split=tensor_split,
                alias=args.llama_alias or spec.model_id,
                extra_args=args.llama_extra_arg,
            )
            return {
                "mesh_id": spec.mesh_id,
                "rpc_plan": plan.to_dict(),
                "llama_server_command": cmd,
                "llama_server_command_text": command_preview(cmd),
                "backend_url": f"http://{args.llama_host}:{args.llama_port}",
                "proof_required": bool(args.require_proof),
                "proof_sample_bps": int(args.proof_sample_bps),
                "proof_ops_per_request": int(args.proof_ops_per_request),
                "proof_trace_candidates_per_request": int(
                    args.proof_trace_candidates_per_request
                ),
                "decode_audit_bps": int(args.decode_audit_bps),
                "decode_audit_top_k": int(args.decode_audit_top_k),
                "proof_artifact_timeout": float(args.proof_artifact_timeout),
                "proof_url": args.proof_url,
                "proof_trace_dir": args.proof_trace_dir,
                "proof_collection": _proof_collection_label(args, spec),
            }

        payload = llama_payload_for(mesh_spec)
        if args.llama_dry_run:
            runtime_env = get_proof_runtime_env()
            if runtime_env is not None:
                payload["proof_runtime_env"] = {
                    key: runtime_env[key]
                    for key in (
                        "VERATHOS_GGML_TRACE_DIR",
                        "VERATHOS_GGML_TRACE_ENABLE_FILE",
                        "VERATHOS_GGML_TRACE_MAX_OPS_PER_CAPTURE",
                        "VERATHOS_GGML_TRACE_MAX_OPS_PER_GRAPH",
                        "VERATHOS_GGML_TRACE_MANIFEST_FORMAT",
                        "VERATHOS_GGML_TRACE_MAX_ELEMS",
                        "VERATHOS_GGML_TRACE_SKIP_SRC0_DUMP",
                        "VERATHOS_GGML_RPC_GRAPH_TRACE",
                        "VERATHOS_GGML_TAIL_RING",
                        "VERATHOS_GGML_TAIL_RING_FLUSH_MS",
                        "GGML_CUDA_DISABLE_FUSION",
                    )
                    if key in runtime_env
                }
            if rpc_payload is not None:
                payload["rpc_worker_command"] = rpc_payload["rpc_worker_command"]
                payload["rpc_worker_command_text"] = rpc_payload["rpc_worker_command_text"]
            _print_json(payload)
            return
    # Crash-loop breaker: a backend that dies N times inside WINDOW seconds is
    # not going to heal by being respawned a 363rd time (the 4090 arch-mismatch
    # incident retried silently 362x while the dashboard said "driving"). Stop
    # retrying, write backend-failure.json into the mesh dir where the pool
    # driver's readiness wait picks it up, and let the mesh error with the
    # real reason instead of a ready-timeout hours later.
    _CRASH_WINDOW_S = BackendCrashBreaker.WINDOW_S
    _breaker = BackendCrashBreaker()
    _proc_started: dict[str, float] = {"rpc": 0.0, "llama": 0.0}

    def _write_backend_failure(
        component: str,
        *,
        exit_code: int | None = None,
        crashes: int = 0,
        error: str = "",
    ) -> None:
        failure = {
            "component": component,
            "exit_code": exit_code,
            "crashes": int(crashes),
            "window_s": _CRASH_WINDOW_S,
            "unix": int(time.time()),
            **({"error": str(error)[:500]} if error else {}),
        }
        try:
            failure_root = (
                persisted_state_path.parent
                if persisted_state_path is not None
                else Path(args.mesh)
            )
            if args.mesh:
                (failure_root / "backend-failure.json").write_text(
                    json.dumps(failure)
                )
        except Exception:
            pass

    def _expected_assignment_complete() -> bool:
        if persisted_state_path is None:
            return False
        try:
            current_state = load_mesh_state(persisted_state_path)
            expected = int(
                current_state.get("expected_compute_stage_count", 0) or 0
            )
            return (
                expected > 0
                and state_admitted_compute_stage_count(current_state) >= expected
            )
        except Exception:
            return False

    def _note_backend_crash(component: str, exit_code: int | None) -> bool:
        started = _proc_started.get(component, 0.0)
        uptime = (time.monotonic() - started) if started else 0.0
        tripped, crashes, reason = _breaker.note(component, uptime_s=uptime)
        if not tripped:
            return False
        _write_backend_failure(
            component,
            exit_code=exit_code,
            crashes=crashes,
        )
        print(
            f"BACKEND CRASH LOOP: {component} exited {reason} "
            f"(last code {exit_code}); giving up, "
            "check the serve log for the first GGML_ASSERT/CUDA error line",
            flush=True,
        )
        return True

    if rpc_cmd is not None and rpc_payload is not None:
        rpc_cmd[0] = resolve_binary(rpc_cmd[0])
        rpc_env = get_proof_runtime_env()
        rpc_restart_delay = max(1.0, min(5.0, float(args.llama_reload_interval or 1.0)))

        def launch_rpc_worker() -> None:
            nonlocal rpc_proc
            while not llama_stop.is_set():
                if rpc_proc is None or rpc_proc.poll() is not None:
                    if rpc_proc is not None:
                        print(
                            f"llama.cpp RPC worker exited with code {rpc_proc.returncode}; restarting",
                            flush=True,
                        )
                        if _note_backend_crash("rpc", rpc_proc.returncode):
                            return
                    print(rpc_payload["rpc_worker_command_text"], flush=True)
                    rpc_proc = subprocess.Popen(
                        rpc_cmd,
                        env=rpc_env,
                        preexec_fn=_die_with_parent_preexec,
                    )
                    _proc_started["rpc"] = time.monotonic()
                llama_stop.wait(rpc_restart_delay)

        rpc_thread = threading.Thread(target=launch_rpc_worker, daemon=True)
        rpc_thread.start()
    if args.llama_model or args.llama_hf:
        backend_url = payload["backend_url"]
        from urllib.parse import urlparse as _urlparse

        _llama_parsed = _urlparse(backend_url)
        _llama_bind_host = _llama_parsed.hostname or "127.0.0.1"
        _llama_bind_port = int(_llama_parsed.port or 8080)

        def launch_llama(spec: MeshSpec) -> None:
            nonlocal llama_proc, llama_plan_hash
            # The min-worker count check must run BEFORE the strict all-RPC
            # binding validation inside llama_payload_for: while a multi-box
            # mesh is still admitting members, the committed plan legitimately
            # has fewer RPC stages than the launch config expects. That is a
            # wait state, not a plan error. Validating first raised "llama
            # device order does not match the committed all-RPC plan" on every
            # supervisor tick during the join, and became a TERMINAL drive
            # failure when the last member committed between the raise and the
            # monitor's assignment-completeness re-read.
            rpc_count = committed_rpc_stage_count(spec)
            if rpc_count < args.llama_min_rpc_workers:
                with llama_lock:
                    _terminate_process(llama_proc)
                    llama_proc = None
                if llama_plan_hash != f"waiting:{rpc_count}":
                    print(
                        f"waiting for {args.llama_min_rpc_workers} llama.cpp RPC workers "
                        f"(currently {rpc_count})",
                        flush=True,
                    )
                    llama_plan_hash = f"waiting:{rpc_count}"
                return
            payload = llama_payload_for(spec)
            plan = payload["rpc_plan"]
            plan_hash = str(plan["rpc_plan_hash"])
            unready = [
                endpoint
                for endpoint in plan["rpc_endpoints"]
                if not _rpc_endpoint_is_ready(
                    local_dial.endpoint_for_host(endpoint, local_advertised_host),
                    timeout=args.llama_rpc_ready_timeout,
                )
            ]
            if unready:
                with llama_lock:
                    if (
                        llama_proc is not None
                        and llama_proc.poll() is None
                        and llama_plan_hash == plan_hash
                    ):
                        return
                    _terminate_process(llama_proc)
                    llama_proc = None
                wait_key = "waiting-rpc:" + ",".join(unready)
                if llama_plan_hash != wait_key:
                    print(
                        "waiting for llama.cpp RPC endpoints: " + ",".join(unready),
                        flush=True,
                    )
                    llama_plan_hash = wait_key
                return
            with llama_lock:
                if llama_proc is not None and llama_proc.poll() is None and llama_plan_hash == plan_hash:
                    return
                if (
                    llama_proc is not None
                    and llama_proc.poll() is not None
                    and llama_plan_hash == plan_hash
                ):
                    # Same plan, dead process = a crash (plan changes replace
                    # the process legitimately and don't count). Exception: a
                    # near-instant death while the bind port is still held is
                    # the relaunch bind race, not a backend fault - the port
                    # wait below prevents the next one, and counting these
                    # tripped the breaker on healthy relaunches.
                    started = _proc_started.get("llama", 0.0)
                    uptime = (time.monotonic() - started) if started else 0.0
                    bind_race = (
                        uptime < 2.0
                        and not _wait_port_free(
                            _llama_bind_host, _llama_bind_port, timeout_s=0.0
                        )
                    )
                    if bind_race:
                        print(
                            "llama.cpp exited instantly with its port still "
                            "held (relaunch bind race); not counting toward "
                            "the crash breaker",
                            flush=True,
                        )
                    elif _note_backend_crash("llama", llama_proc.returncode):
                        llama_stop.set()
                        return
                _terminate_process(llama_proc)
                if not _wait_port_free(
                    _llama_bind_host, _llama_bind_port, timeout_s=10.0
                ):
                    print(
                        f"llama.cpp port {_llama_bind_port} still held after "
                        "terminating the previous instance; deferring the "
                        "respawn to the next supervisor tick",
                        flush=True,
                    )
                    return
                cmd = local_dial.rpc_command_for_host(
                    payload["llama_server_command"], local_advertised_host
                )
                cmd[0] = resolve_binary(cmd[0])
                print(payload["llama_server_command_text"], flush=True)
                llama_env = (
                    get_llama_capture_env() if args.require_proof and not args.rpc_worker else None
                )
                llama_proc = subprocess.Popen(
                    cmd,
                    env=llama_env,
                    preexec_fn=_die_with_parent_preexec,
                )
                _proc_started["llama"] = time.monotonic()
                llama_plan_hash = plan_hash

        def _park_spawner_thread() -> None:
            # PDEATHSIG is delivered when the spawning THREAD exits, not
            # the process: a supervisor that returns while its llama is
            # alive SIGTERMs the backend it just launched (one-shot mode
            # killed every serve at startup). Park until process death -
            # daemon threads die with the process, which is exactly when
            # the backend SHOULD receive its death signal.
            threading.Event().wait()

        def monitor_llama() -> None:
            while not llama_stop.is_set():
                try:
                    current = mesh_spec_loader() if mesh_spec_loader is not None else mesh_spec
                    if current is not None:
                        launch_llama(current)
                except Exception as exc:
                    print(f"llama-server supervisor error: {exc}", flush=True)
                    if isinstance(exc, (ValueError, FileNotFoundError)) and (
                        _expected_assignment_complete()
                    ):
                        _write_backend_failure(
                            "llama-plan",
                            error=str(exc),
                        )
                        print(
                            "TERMINAL LLAMA PLAN ERROR after final member "
                            "assignment; giving up",
                            flush=True,
                        )
                        llama_stop.set()
                        _park_spawner_thread()
                if args.llama_reload_interval <= 0:
                    _park_spawner_thread()
                llama_stop.wait(args.llama_reload_interval)
            if llama_proc is not None and llama_proc.poll() is None:
                # Stop requested with a live backend (e.g. rpc-side crash
                # breaker): parking keeps this thread's death signal out
                # of the picture so the shutdown path controls teardown.
                _park_spawner_thread()

        llama_thread = threading.Thread(target=monitor_llama, daemon=True)
        llama_thread.start()
    requested_server_role = str(getattr(args, "server_role", "auto") or "auto")
    if (
        requested_server_role != "auto"
        and mesh_state_role
        and requested_server_role != mesh_state_role
    ):
        raise SystemExit(
            f"--server-role {requested_server_role} conflicts with {mesh_state_role} mesh state"
        )
    server_role = (
        mesh_state_role if requested_server_role == "auto" and mesh_state_role
        else requested_server_role
    )
    if args.validator_auth and server_role == "worker":
        raise SystemExit("--validator-auth is only valid for a mesh coordinator")
    if args.require_verification_snapshot and server_role == "worker":
        raise SystemExit(
            "--require-verification-snapshot is only valid for a mesh coordinator"
        )
    if verification_snapshot_loader is not None and server_role != "worker":
        if not args.validator_auth or not args.require_validator_nonce:
            raise SystemExit(
                "snapshot-bound coordinators require --validator-auth and "
                "--require-validator-nonce"
            )

    try:
        serve_worker(
            capability=capability,
            host=args.host,
            port=args.port,
            mesh_spec=mesh_spec,
            mesh_spec_loader=mesh_spec_loader,
            join_handler=join_handler,
            mesh_spec_handler=mesh_spec_handler,
            mesh_update_handler=mesh_update_handler,
            backend_url=backend_url,
            proof_url=args.proof_url,
            require_proof=args.require_proof,
            proof_trace_enable_file=proof_trace_enable_file,
            proof_trace_dir=args.proof_trace_dir if not args.proof_url else "",
            proof_gguf_manifest_path=args.proof_gguf_manifest,
            proof_tolerance_abs=args.proof_tolerance_abs,
            proof_tolerance_rel=args.proof_tolerance_rel,
            proof_block_size=args.proof_block_size,
            proof_spot_checks=args.proof_spot_checks,
            proof_warmup=bool(
                args.require_proof
                and args.proof_trace_dir
                and not args.no_proof_warmup
            ),
            proof_decode_projection_warmup=bool(args.proof_decode_projection_warmup),
            proof_sample_bps=args.proof_sample_bps,
            defer_proof=args.defer_proof,
            proof_ops_per_request=args.proof_ops_per_request,
            proof_trace_candidates_per_request=args.proof_trace_candidates_per_request,
            decode_audit_bps=args.decode_audit_bps,
            decode_audit_top_k=args.decode_audit_top_k,
            proof_artifact_timeout=args.proof_artifact_timeout,
            llama_n_parallel=llama_n_parallel_setting(),
            llama_n_ubatch=llama_n_ubatch_setting(),
            proof_trace_manifest_format=decide_trace_manifest_format(
                args, n_parallel=llama_n_parallel_setting()
            ),
            slot_view_template_warmup=bool(
                args.require_proof
                and backend_url
                and args.proof_trace_dir
                and (
                    args.decode_audit_bps == 0
                    or llama_n_parallel_setting() > 1
                )
            ),
            local_stage_capture=bool(
                getattr(args, "llama_capture_trace_dir", "")
            ),
            receipt_signer=receipt_signer,
            stage_receipt_signer=stage_receipt_signer,
            stage_proof_key=stage_proof_key,
            server_role=server_role,
            validator_auth_enabled=bool(args.validator_auth),
            validator_allowlist_path=args.validator_allowlist_path,
            validator_allowlist_max_age_seconds=(
                args.validator_allowlist_max_age_seconds
            ),
            require_validator_nonce=bool(args.require_validator_nonce),
            internal_auth_secret=internal_auth_secret,
            evm_address=evm_address,
            evm_private_key=evm_private_key,
            evm_challenge_signer=evm_challenge_signer,
            # The exact fitted budget this serve was launched with: the
            # ladder respawns the serve on descent, so this value IS the
            # backend's real unified-KV size for the admission ledger.
            llama_ctx_budget=int(getattr(args, "llama_ctx_size", 0) or 0),
            slot_state_dir=str(getattr(args, "slot_state_dir", "") or ""),
            capacity_drain_file=str(
                getattr(args, "capacity_drain_file", "") or ""
            ),
            capacity_roster_file=str(
                getattr(args, "capacity_roster_file", "") or ""
            ),
            verification_snapshot_loader=verification_snapshot_loader,
            allow_loopback_dev_validator_routes=bool(
                args.allow_loopback_dev_validator_routes
            ),
        )
    finally:
        llama_stop.set()
        if args.llama_model or args.llama_hf:
            llama_thread.join(timeout=2)
        _terminate_process(llama_proc)
        _terminate_process(rpc_proc)


def cmd_probe(args: argparse.Namespace) -> None:
    _print_json(probe_worker(args.endpoint, timeout=args.timeout).to_dict())


def cmd_plan_units(args: argparse.Namespace) -> None:
    """Derive pool worker units (one per GPU group) for this host."""
    from verallm.mesh import units as units_module

    if args.backend == "cuda":
        gpus = units_module.detect_gpus()
    else:
        gpus = [units_module.GpuInfo(index=0, name=args.backend, vram_gb=0)]
    try:
        groups = units_module.parse_gpu_groups(
            args.gpus, [gpu.index for gpu in gpus]
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    planned = units_module.plan_worker_units(
        gpus,
        worker_id_base=args.worker_id_base,
        home=Path.home(),
        rpc_port_base=args.rpc_port,
        proof_port_base=args.proof_port,
        mesh_port_base=args.mesh_port,
        backend=args.backend,
        groups=groups,
    )
    _print_json({"units": [dataclasses.asdict(unit) for unit in planned]})


def cmd_register_units(args: argparse.Namespace) -> None:
    """Record this host's planned units so status/logs/stop can find them."""
    from verallm.mesh import units as units_module
    from verallm.mesh.pool import load_pool_token_file

    token = load_pool_token_file(args.token_file)
    registry = json.loads(Path(args.units_file).read_text())
    planned = [units_module.WorkerUnit(**unit) for unit in registry["units"]]
    target = units_module.save_unit_registry(
        planned,
        manager_endpoint=token.manager_endpoint,
        pool_id=token.pool_id,
        token_file=str(Path(args.token_file).resolve()),
    )
    _print_json({"status": "ok", "registry": str(target), "units": len(planned)})


# -- worker pool (docs/architecture/mesh_orchestration_ux.md) ---------------


# (host, port) -> SSLContext trusted only for that private pool manager.
# The registry is shared by token-pin, explicit CA-file, and coordinator-local
# trust paths. The installed urllib handler consults it per request, so public
# artifact/configuration hosts always retain the platform PKI trust store.
_POOL_API_TLS_CONTEXTS: dict[tuple[str, int], ssl.SSLContext] = {}


def _install_per_host_manager_tls_context(
    endpoint: str,
    context: ssl.SSLContext,
) -> None:
    """Install private TLS trust for exactly one manager host and port."""

    from urllib.parse import urlparse

    parsed = urlparse(str(endpoint or ""))
    if parsed.scheme != "https" or not parsed.hostname:
        return
    key = (parsed.hostname, parsed.port or 443)
    _POOL_API_TLS_CONTEXTS[key] = context

    import http.client as _http_client
    import urllib.request as _urllib_request

    default_context = ssl.create_default_context()

    class _ManagerPinnedHTTPSHandler(_urllib_request.HTTPSHandler):
        def https_open(self, req):
            from urllib.parse import urlparse as _urlparse

            target = _urlparse(req.full_url)
            pinned = _POOL_API_TLS_CONTEXTS.get(
                (target.hostname, target.port or 443)
            )
            return self.do_open(
                _http_client.HTTPSConnection,
                req,
                context=pinned if pinned is not None else default_context,
            )

    _urllib_request.install_opener(
        _urllib_request.build_opener(_ManagerPinnedHTTPSHandler())
    )
    from verallm.mesh.worker import register_pinned_tls_context

    register_pinned_tls_context(key[0], key[1], context)


def _install_manager_tls_pin(token) -> None:
    """Trust the manager's TLS certificate via the token's pin.

    Remote join tokens carry the SHA256 of the manager's DER certificate
    (the token travels over a secure channel anyway — it holds the pool
    secret — so it doubles as the TLS trust root; no CA files to hand
    out). The pinned certificate is fetched once, verified against the
    pin, and installed via a PER-HOST opener: only connections to the
    manager's host:port use the pinned context (hostname checking off -
    the pin is strictly stronger than a hostname match against a rented
    box's self-signed certificate); every other host keeps the default
    PKI trust. A process-wide anchor here once broke every verathos.ai
    store fetch in the same process.
    """

    pin = str(getattr(token, "manager_ca_sha256", "") or "").strip().lower()
    if not pin:
        return
    from urllib.parse import urlparse

    parsed = urlparse(str(token.manager_endpoint))
    if parsed.scheme != "https" or not parsed.hostname:
        return
    port = parsed.port or 443
    try:
        pem = ssl.get_server_certificate((parsed.hostname, port))
    except OSError as exc:
        raise SystemExit(
            f"could not reach the pool manager at {token.manager_endpoint} "
            f"to verify its pinned TLS certificate: {exc}"
        )
    der = ssl.PEM_cert_to_DER_cert(pem)
    observed = hashlib.sha256(der).hexdigest()
    if observed != pin:
        raise SystemExit(
            "the pool manager's TLS certificate does not match the join "
            "token's pin (possible interception or a re-minted manager "
            f"certificate). observed sha256 {observed[:16]}..., pinned "
            f"{pin[:16]}... — mint a fresh join token on the coordinator."
        )
    pin_dir = Path.home() / ".verathos" / "manager-pins"
    pin_dir.mkdir(parents=True, exist_ok=True)
    pin_file = pin_dir / f"{token.pool_id}.pem"
    pin_file.write_text(pem, encoding="utf-8")
    os.chmod(pin_file, 0o600)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    context.check_hostname = False
    context.verify_mode = ssl.CERT_REQUIRED
    context.load_verify_locations(cafile=str(pin_file))
    _install_per_host_manager_tls_context(token.manager_endpoint, context)


def _manager_ca_file_context(path: str) -> ssl.SSLContext:
    """Load explicit private CA trust; callers scope it to one manager."""
    ca_path = Path(path).expanduser()
    if not ca_path.is_file():
        raise SystemExit(f"--manager-ca-file is not a regular file: {ca_path}")
    try:
        context = ssl.create_default_context(cafile=str(ca_path))
    except (OSError, ssl.SSLError) as exc:
        raise SystemExit(f"--manager-ca-file is not a valid CA bundle: {ca_path}") from exc
    context.check_hostname = False
    return context


def _pool_token_from_args(
    args: argparse.Namespace,
    *,
    required_scope: str = "",
):
    """Resolve the preferred owner-only token file or the legacy raw flag."""

    manager_ca_file = str(getattr(args, "manager_ca_file", "") or "").strip()
    manager_ca_context = None
    if manager_ca_file:
        manager_ca_context = _manager_ca_file_context(manager_ca_file)

    from verallm.mesh.pool import MeshPoolToken, load_pool_token_file

    token_file = str(getattr(args, "pool_token_file", "") or "").strip()
    inline_token = str(getattr(args, "pool_token", "") or "").strip()
    if token_file and inline_token:
        raise SystemExit("use only one of --pool-token-file or --pool-token")
    try:
        if token_file:
            token = load_pool_token_file(token_file)
            if required_scope and token.scope != required_scope:
                raise ValueError(
                    f"pool token scope is {token.scope!r}; expected {required_scope!r}"
                )
            if str(getattr(token, "manager_ca_sha256", "") or "").strip():
                _install_manager_tls_pin(token)
            elif manager_ca_context is not None:
                _install_per_host_manager_tls_context(
                    token.manager_endpoint, manager_ca_context
                )
            return token
        if inline_token:
            token = MeshPoolToken.decode(inline_token)
            if required_scope and token.scope != required_scope:
                raise ValueError(
                    f"pool token scope is {token.scope!r}; expected {required_scope!r}"
                )
            if str(getattr(token, "manager_ca_sha256", "") or "").strip():
                _install_manager_tls_pin(token)
            elif manager_ca_context is not None:
                _install_per_host_manager_tls_context(
                    token.manager_endpoint, manager_ca_context
                )
            return token
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    raise SystemExit("--pool-token-file is required (or legacy --pool-token)")


def _add_pool_token_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--pool-token-file",
        help="Owner-only (0600 or 0400) file containing the pool token",
    )
    group.add_argument(
        "--pool-token",
        help=(
            "Legacy inline token; visible in shell history/process listings. "
            "Prefer --pool-token-file"
        ),
    )
    parser.add_argument(
        "--manager-ca-file",
        default="",
        help=(
            "CA bundle trusted only for this private/self-signed HTTPS pool "
            "manager"
        ),
    )


_DIALABLE_ENDPOINT_CACHE: dict[tuple[str, str], str] = {}


def _pool_dir_for_token(token) -> Path | None:
    """The local pool dir for a token's pool_id, when this box has it."""

    pool_id = str(getattr(token, "pool_id", "") or "")
    if not pool_id:
        return None
    canonical = Path.home() / ".verathos" / "pools" / pool_id
    if (canonical / "api-tls-cert.pem").is_file():
        return canonical
    from verallm.mesh.pool import known_pool_dirs

    for candidate in known_pool_dirs():
        base = Path(candidate)
        if base.name == pool_id and (base / "api-tls-cert.pem").is_file():
            return base
    return None


def _dialable_manager_endpoint(token) -> str:
    """The manager endpoint that actually answers for this pool.

    A coordinator box often cannot hairpin to its own public IP; every
    management command (launch, stop, probe, chat) then fails with a
    connection error while the manager listens fine on loopback. Probe
    the advertised endpoint first, fall back to its loopback twin only
    when THAT answers with the right pool id. Cached per process so
    per-call clients do not re-probe.
    """

    from verallm.mesh.onboarding import (
        loopback_equivalent,
        probe_manager,
    )

    advertised = token.manager_endpoint.rstrip("/")
    # EVERY management path resolves its endpoint here, so this is the one
    # chokepoint to trust the pool's own api-tls cert. Fixing only
    # _pool_client left probe/chat/apikey dying on the self-signed cert the
    # moment a launch succeeded.
    _install_pool_api_tls_trust(advertised, _pool_dir_for_token(token))
    _install_pool_api_tls_trust(
        loopback_equivalent(advertised), _pool_dir_for_token(token)
    )
    pool_id = str(getattr(token, "pool_id", "") or "")
    if not pool_id:
        return advertised
    cache_key = (advertised, pool_id)
    cached = _DIALABLE_ENDPOINT_CACHE.get(cache_key)
    if cached is not None:
        return cached
    result = advertised
    probe = probe_manager(advertised, timeout=2.0)
    if not (probe.reachable and probe.pool_id == pool_id):
        loopback = loopback_equivalent(advertised)
        if loopback != advertised:
            fallback = probe_manager(loopback, timeout=2.0)
            if fallback.reachable and fallback.pool_id == pool_id:
                result = loopback
    _DIALABLE_ENDPOINT_CACHE[cache_key] = result
    return result


def _resolve_pool_dir(args: argparse.Namespace) -> Path | None:
    """Best-effort pool directory, for local files like the API TLS cert.

    Mirrors ``_resolve_pool_context``'s dir search (explicit ``--pool``, an
    explicit token file's parent, ``$VERATHOS_POOL_TOKEN_FILE``'s parent, then
    a single discovered pool) but returns the directory instead of a token, so
    management calls can trust the pool's own certificate.
    """

    from verallm.mesh.pool import (
        POOL_ADMIN_TOKEN_FILE,
        POOL_TOKEN_FILE,
        known_pool_dirs,
    )

    pool_dir = str(getattr(args, "pool", "") or "").strip()
    if pool_dir:
        return Path(pool_dir).expanduser()
    token_file = str(getattr(args, "pool_token_file", "") or "").strip()
    if token_file:
        return Path(token_file).expanduser().parent
    env_file = os.environ.get("VERATHOS_POOL_TOKEN_FILE", "").strip()
    if env_file:
        return Path(env_file).expanduser().parent

    verathos_home = Path.home() / ".verathos"
    candidates: list[Path] = []
    seen: set[str] = set()
    for candidate in (
        list(known_pool_dirs())
        + sorted(verathos_home.glob("pools/pool-*"))
        + sorted(verathos_home.glob("*/pool-*"))
    ):
        resolved = str(Path(candidate).resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        if (Path(candidate) / POOL_ADMIN_TOKEN_FILE).is_file() or (
            Path(candidate) / POOL_TOKEN_FILE
        ).is_file():
            candidates.append(Path(candidate))
    return candidates[0] if len(candidates) == 1 else None


def _install_pool_api_tls_trust(endpoint: str, pool_dir: Path | None) -> None:
    """Trust the pool's own API-TLS cert for management calls to the manager.

    The management CLI runs ON the coordinator box, where the pool's
    self-signed ``api-tls-cert.pem`` sits in the pool dir. Management tokens
    carry no CA pin (unlike worker join tokens), so without this every
    management call to the ``--api-tls-port`` listener failed
    ``CERTIFICATE_VERIFY_FAILED: self-signed certificate`` even though the
    trust root was a file on the same box. Pin PER-HOST (hostname check off:
    the cert is a pool-local trust root, not a hostname assertion) so every
    OTHER https fetch in the process keeps real PKI trust.
    """

    from urllib.parse import urlparse

    if pool_dir is None:
        return
    parsed = urlparse(endpoint)
    if parsed.scheme != "https" or not parsed.hostname:
        return
    cert = Path(pool_dir) / "api-tls-cert.pem"
    if not cert.is_file():
        return
    key = (parsed.hostname, parsed.port or 443)
    if key in _POOL_API_TLS_CONTEXTS:
        return
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    context.check_hostname = False
    context.verify_mode = ssl.CERT_REQUIRED
    context.load_verify_locations(cafile=str(cert))
    _install_per_host_manager_tls_context(endpoint, context)


def _pool_client(args: argparse.Namespace):
    # Zero-flag resolution (explicit flags, --pool dir, env, single-pool
    # discovery); every management route rejects worker-scoped tokens with
    # guidance instead of an opaque 403 from the manager.
    token = _management_pool_token(args)
    from verallm.mesh.worker import post_json

    endpoint = _dialable_manager_endpoint(token)
    # Management tokens carry no CA pin; trust the pool's local api-tls cert so
    # https calls to the manager's api-tls port verify instead of failing
    # 'self-signed certificate'. Per-host, so other https keeps real PKI.
    _install_pool_api_tls_trust(endpoint, _resolve_pool_dir(args))

    def call(route: str, body: dict) -> dict:
        return post_json(
            endpoint + route,
            {"management_secret": token.pool_secret, **body},
            timeout=float(getattr(args, "timeout", 10.0)),
        )

    # Callers printing URLs for the operator (apikey create) need the real
    # dialable endpoint, not a hardcoded loopback guess.
    call.manager_endpoint = endpoint  # type: ignore[attr-defined]
    return call


def _add_optional_pool_token_arguments(parser: argparse.ArgumentParser) -> None:
    """Token flags that fall back to local discovery instead of requiring one."""
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--pool-token-file",
        help="Owner-only (0600 or 0400) file containing a pool token",
    )
    group.add_argument(
        "--pool-token",
        help="Legacy inline token; prefer --pool-token-file",
    )
    parser.add_argument(
        "--pool",
        default="",
        help="Pool state dir; reads its admin token (coordinator box)",
    )
    parser.add_argument(
        "--manager-ca-file",
        default="",
        help="CA bundle for a private/self-signed HTTPS pool manager",
    )


def _resolve_pool_context(args: argparse.Namespace):
    """Find a pool token: explicit flags, --pool dir, env, then discovery.

    Returns the token; management scope when available, worker scope as the
    read-only fallback on worker boxes. Exits with guidance when nothing
    resolves.
    """
    from verallm.mesh.pool import (
        POOL_ADMIN_TOKEN_FILE,
        POOL_TOKEN_FILE,
        load_pool_token_file,
    )

    if getattr(args, "pool_token_file", "") or getattr(args, "pool_token", ""):
        return _pool_token_from_args(args)

    token_errors: dict[str, str] = {}

    def _try(path: Path):
        try:
            return load_pool_token_file(path)
        except (OSError, ValueError) as exc:
            if path.is_file():
                # An existing-but-unloadable token (permissions, corruption)
                # must surface as ITS error, not as "no pool found".
                token_errors[str(path)] = str(exc)
            return None

    pool_dir = str(getattr(args, "pool", "") or "").strip()
    if pool_dir:
        base = Path(pool_dir).expanduser()
        for name in (POOL_ADMIN_TOKEN_FILE, POOL_TOKEN_FILE):
            token = _try(base / name)
            if token is not None:
                return token
        raise SystemExit(f"no readable pool token under {base}")

    env_file = os.environ.get("VERATHOS_POOL_TOKEN_FILE", "").strip()
    if env_file:
        token = _try(Path(env_file).expanduser())
        if token is not None:
            return token
        raise SystemExit(f"$VERATHOS_POOL_TOKEN_FILE is not readable: {env_file}")

    from verallm.mesh.pool import known_pool_dirs

    verathos_home = Path.home() / ".verathos"
    candidate_dirs: list[Path] = []
    seen: set[str] = set()
    for candidate in (
        list(known_pool_dirs())
        + sorted(verathos_home.glob("pools/pool-*"))
        + sorted(verathos_home.glob("*/pool-*"))
    ):
        resolved = str(Path(candidate).resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        if (Path(candidate) / POOL_ADMIN_TOKEN_FILE).is_file() or (
            Path(candidate) / POOL_TOKEN_FILE
        ).is_file():
            candidate_dirs.append(Path(candidate))

    def _dir_token(base: Path):
        for name in (POOL_ADMIN_TOKEN_FILE, POOL_TOKEN_FILE):
            token = _try(base / name)
            if token is not None:
                return token
        return None

    if len(candidate_dirs) == 1:
        token = _dir_token(candidate_dirs[0])
        if token is not None:
            # stderr: stdout may be a --json contract or a pipe.
            print(_c(f"pool: {candidate_dirs[0]}", "2"), file=sys.stderr)
            return token
    elif len(candidate_dirs) > 1:
        if _interactive_terminal():
            print(_c("pools on this machine:", "1"))
            for index, base in enumerate(candidate_dirs, start=1):
                token = _dir_token(base)
                endpoint = token.manager_endpoint if token else "?"
                print(
                    _c(f"  {index}) ", "2")
                    + _c(base.name, "1")
                    + _c(f"  {endpoint}  ({base})", "2")
                )
            while True:
                raw = input(
                    _c(f"select [1-{len(candidate_dirs)}, Enter=1]: ", "1")
                ).strip()
                if not raw:
                    raw = "1"
                if raw.isdigit() and 1 <= int(raw) <= len(candidate_dirs):
                    picked = candidate_dirs[int(raw) - 1]
                    token = _dir_token(picked)
                    if token is not None:
                        return token
                    # A valid pick with an unloadable token would loop
                    # forever on "pick a number from the list".
                    raise SystemExit(
                        f"the token under {picked} cannot be loaded:\n  "
                        + "\n  ".join(sorted(token_errors.values()))
                    )
                print(_c("  pick a number from the list", "33"))
        listing = "\n  ".join(str(path) for path in candidate_dirs)
        raise SystemExit(
            "several pools on this machine; pick one with --pool DIR:\n  "
            + listing
        )

    # Worker box: the unit registry records the join token file.
    from verallm.mesh import units as units_module

    registry = units_module.load_unit_registry()
    if registry and registry.get("token_file"):
        token = _try(Path(registry["token_file"]))
        if token is not None:
            return token

    if token_errors:
        listing = "\n  ".join(
            f"{path}: {error}" for path, error in sorted(token_errors.items())
        )
        raise SystemExit(
            "a pool exists on this machine but its token cannot be "
            "loaded:\n  " + listing
        )
    raise SystemExit(
        "no pool found on this machine.\n"
        "  new here?            verathos mesh setup\n"
        "  pool lives elsewhere? verathos mesh chat --pool <state-dir>\n"
        "  have a token file?    --pool-token-file <file> or "
        "$VERATHOS_POOL_TOKEN_FILE"
    )


def render_fleet(
    overview: Mapping[str, Any],
    *,
    suggestions: Sequence[Mapping[str, Any]] = (),
    reasons: Mapping[str, str] | None = None,
    model_id: str = "",
    managed: bool = True,
    styled: bool | None = None,
) -> str:
    """Human rendering of the operator overview plus placement advice.

    Reason and warn strings from the pool are printed verbatim: they are
    already written for humans and paraphrasing them here would create a
    second source of truth.

    ``styled`` gates color only (default: stdout is a TTY): stripped
    styled output is byte-identical to the plain form, so piped output
    stays parseable and ``--json`` remains the stable agent contract.
    """
    from verallm.mesh import panels
    from verallm.mesh import render as _render

    if styled is None:
        styled = _render.use_style()

    lines: list[str] = []
    header = _render.bold(
        f"pool {overview.get('pool_id', '?')}", styled=styled
    ) + f"  mode={overview.get('serving_mode', '?')}"
    if overview.get("owner_account"):
        header += f"  owner={overview.get('owner_account')}"
    lines.append(header)
    if not managed:
        lines.append(
            "(public view: worker detail is sanitized and mesh errors are "
            "redacted; pass an admin token for the full picture)"
        )

    view = {
        "live": True,
        "workers": dict(overview.get("workers", {})),
        "meshes": dict(overview.get("meshes", {})),
    }
    lines.append("")
    lines.extend(panels.pool_panel(view, styled=styled))
    lines.append("")
    lines.extend(panels.mesh_board(view, styled=styled))

    models = dict(overview.get("models", {}))
    if models:
        lines.append("")
        lines.append(
            _render.bold(f"  models ({len(models)})", styled=styled)
        )
        rows = []
        for name in sorted(models):
            model = models[name]
            ready = ""
            if "launch_ready" in model:
                ready = (
                    _render.green("launch-ready", styled=styled)
                    if model["launch_ready"]
                    else _render.red("NOT launch-ready", styled=styled)
                )
            rows.append(
                [
                    "    " + _render.bold(name, styled=styled),
                    (
                        f"{float(model.get('model_bytes', 0) or 0) / 1e9:.1f} GB"
                        if model.get("model_bytes")
                        else ""
                    ),
                    (
                        f"{model['layers']} layers"
                        if model.get("layers")
                        else ""
                    ),
                    ready,
                ]
            )
        lines.extend(_render.table(rows, indent="", styled=styled))

    if model_id:
        lines.append("")
        lines.append(
            _render.bold(f"  placement for {model_id}:", styled=styled)
        )
        placement_lines, _ = panels.placement_table(
            model_id,
            {"suggestions": list(suggestions), "reasons": dict(reasons or {})},
            view["workers"],
            styled=styled,
        )
        lines.extend(placement_lines)
        if not suggestions:
            lines.append("  no viable placement")
    return "\n".join(lines)


def cmd_fleet(args: argparse.Namespace) -> None:
    from verallm.mesh.pool import POOL_TOKEN_SCOPE_MANAGEMENT
    from verallm.mesh.worker import post_json

    token = _resolve_pool_context(args)
    managed = token.scope == POOL_TOKEN_SCOPE_MANAGEMENT
    endpoint = _dialable_manager_endpoint(token)
    body: dict[str, Any] = {}
    if managed:
        body["management_secret"] = token.pool_secret
    overview = post_json(
        endpoint + "/v1/operator/overview", body, timeout=float(args.timeout)
    )
    suggestions: list[dict[str, Any]] = []
    reasons: dict[str, str] = {}
    if args.model_id:
        advice = post_json(
            endpoint + "/v1/pool/recommend",
            {"model_id": args.model_id},
            timeout=float(args.timeout),
        )
        suggestions = list(advice.get("suggestions", []))
        reasons = dict(advice.get("reasons", {}))
    if args.json:
        _print_json(
            {
                "overview": overview,
                "suggestions": suggestions,
                "reasons": reasons,
            }
        )
        return
    print(
        render_fleet(
            overview,
            suggestions=suggestions,
            reasons=reasons,
            model_id=args.model_id,
            managed=managed,
        )
    )


def _default_signer_from_pool(args: argparse.Namespace, call) -> None:
    """Fill --wallet/--hotkey from the pool's stored miner identity.

    Chain-writing commands (deploy, renew, retire) sign as the identity the
    pool manager already runs with; only pools from before the identity
    persist still need the explicit flag.
    """

    if getattr(args, "wallet", None) or getattr(args, "private_key_file", None):
        return
    try:
        status = call("/v1/pool/status", {})
    except Exception:
        return
    wallet = str(status.get("wallet_name", "") or "")
    if not wallet:
        return
    args.wallet = wallet
    args.hotkey = str(status.get("wallet_hotkey", "") or "") or "default"
    print(
        f"signing as {wallet}/{args.hotkey} (the pool's miner identity; "
        "override with --wallet/--private-key-file)",
        flush=True,
    )


def _stored_registration(call, model_id: str = "") -> dict | None:
    """One stored registration from the manager, multi-model aware.

    Without a model id: unambiguous only when exactly one registration is
    stored; several stored registrations demand the model id instead of
    silently picking one.
    """

    response = call("/v1/pool/registration-state", {})
    registrations = {
        str(key): dict(value)
        for key, value in (response.get("registrations") or {}).items()
        if isinstance(value, dict)
    }
    single = response.get("registration")
    if isinstance(single, dict) and single.get("model_id"):
        registrations.setdefault(str(single["model_id"]), dict(single))
    if model_id:
        found = registrations.get(model_id)
        return dict(found) if found else None
    if len(registrations) == 1:
        return dict(next(iter(registrations.values())))
    if registrations:
        raise SystemExit(
            "several models are registered ("
            + ", ".join(sorted(registrations))
            + "); name the model"
        )
    return None


def _deploy_credentials(
    args: argparse.Namespace,
) -> tuple[str, bytes | None, str]:
    """Resolve the EVM signing key and, in wallet mode, the hotkey seed and
    the hotkey SS58 (needed to verify the hotkey holds a UID on the subnet).
    """
    from verallm.chain.cli_credentials import resolve_cli_evm_private_key

    private_key = resolve_cli_evm_private_key(
        wallet_name=args.wallet or None,
        hotkey_name=args.hotkey,
        private_key_file=args.private_key_file or None,
        required=True,
    )
    hotkey_seed = None
    hotkey_ss58 = ""
    if args.wallet:
        from verallm.mesh.receipt_signing import (
            load_hotkey_keypair,
            load_hotkey_seed,
        )

        keypair = load_hotkey_keypair(args.wallet, args.hotkey)
        hotkey_seed = load_hotkey_seed(args.wallet, args.hotkey, keypair=keypair)
        hotkey_ss58 = keypair.ss58_address
    return private_key, hotkey_seed, hotkey_ss58


def cmd_deploy(args: argparse.Namespace) -> None:
    from verallm.chain.config import ChainConfig
    from verallm.mesh.deploy import DeployConfig, run_deploy
    from verallm.mesh.probe import ProbeGateConfig

    subtensor_network = str(getattr(args, "subtensor_network", "") or "")
    config_path = ChainConfig.resolve_config_path(
        args.chain_config or None, subtensor_network or None
    )
    if not config_path:
        raise SystemExit(
            "pass --subtensor-network test|finney (uses the shipped chain "
            "config for that network) or an explicit --chain-config"
        )
    chain_config = ChainConfig.from_json(config_path)
    chain_endpoint = str(
        getattr(args, "subtensor_chain_endpoint", "") or ""
    ).strip()
    if chain_endpoint:
        # One node for BOTH sides of the deploy: EVM reads/writes (rpc_url,
        # ws:// converted to http://) and substrate UID resolution. Without
        # this every chain step crawled through the congested public RPC
        # even when the pool ran its own subtensor.
        from dataclasses import replace as _dc_replace

        chain_config = _dc_replace(
            chain_config,
            rpc_url=ChainConfig.resolve_rpc_url(
                chain_endpoint, subtensor_network or None
            ),
        )
    # Endpoint scheme posture: refuse before the pipeline spends anything.
    from verallm.chain.config import validate_registration_endpoint_scheme

    validate_registration_endpoint_scheme(
        args.endpoint, getattr(chain_config, "chain_id", None)
    )
    call = _pool_client(args)
    _default_signer_from_pool(args, call)
    private_key, hotkey_seed, hotkey_ss58 = _deploy_credentials(args)
    previous_registration = _stored_registration(call, str(args.model_id))
    config = DeployConfig(
        model_id=args.model_id,
        endpoint=args.endpoint,
        chain_config=chain_config,
        private_key=private_key,
        hotkey_seed=hotkey_seed,
        coordinator_hotkey_ss58=hotkey_ss58,
        subtensor_network=chain_endpoint or subtensor_network,
        uid=args.uid,
        netuid=getattr(chain_config, "netuid", None),
        # None = derive the registered context from the launch: min of the
        # MEASURED KV auto-fit and a time cap from a timing probe (the
        # honest default); an explicit value skips the derivation and is
        # validated against the measurement.
        max_context_len=args.max_context_len,
        validator_budget_s=args.validator_budget_s,
        workers=tuple(part for part in args.workers.split(",") if part.strip()),
        driver=args.driver,
        hf_repo=args.hf_repo,
        hf_files=tuple(part for part in args.hf_files.split(",") if part.strip()),
        model_bytes=args.model_bytes,
        probe=ProbeGateConfig(
            samples=args.probe_samples,
            hard_samples=args.hard_samples,
            min_tok_s=args.min_tok_s,
            full_context=not args.no_full_context_probe,
            full_context_budget_s=args.full_context_budget_s,
        ),
        assume_yes=args.yes,
        force=args.force,
        dry_run=args.dry_run,
        previous_registration=previous_registration,
    )

    def confirm(message: str) -> bool:
        try:
            return input(f"{message} [y/N]: ").strip().lower() in ("y", "yes")
        except (EOFError, KeyboardInterrupt):
            return False

    from verallm.mesh import render

    def _stage_line(line: str) -> None:
        # flush per stage line: deploys run for many minutes and are
        # routinely nohup'd into a file, where block-buffered stdout shows
        # NOTHING until exit — a live, healthy deploy is indistinguishable
        # from a hung one. Styling matches the rest of the CLI (TTY-gated
        # via render, so agents and logs still get plain text).
        text = line
        if line.startswith("[PASS]"):
            text = render.green("[PASS]") + line[len("[PASS]"):]
        elif "FAILED" in line:
            text = render.fail(line)
        elif line.startswith("["):
            end = line.find("]")
            if end > 0:
                text = render.cyan(line[: end + 1]) + render.dim(
                    line[end + 1:]
                )
        print(text, flush=True)

    report = run_deploy(
        config,
        call=call,
        out=_stage_line,
        confirm=confirm,
    )
    if args.json:
        _print_json(report.to_dict())
    # bittensor's substrate client leaves non-daemon websocket threads
    # behind, so a plain SystemExit hangs the interpreter at shutdown
    # (observed: a failed deploy sat silent for 4 hours after its
    # FAILED line). The report is printed and nothing needs cleanup past
    # this point: exit hard.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(1 if report.failed else 0)


def cmd_renew(args: argparse.Namespace) -> None:
    """Renew the on-chain lease once, from the manager's stored state.

    The escape hatch for operators who will not give the pool manager a
    signing key: run this from cron every few hours instead.
    """
    from verallm.chain.config import ChainConfig
    from verallm.mesh.registration import (
        registration_target_from_state,
        renew_once,
    )

    renew_config_path = ChainConfig.resolve_config_path(
        args.chain_config or None,
        str(getattr(args, "subtensor_network", "") or "") or None,
    )
    if not renew_config_path:
        raise SystemExit(
            "pass --subtensor-network test|finney or an explicit "
            "--chain-config"
        )
    chain_config = ChainConfig.from_json(renew_config_path)
    call = _pool_client(args)
    _default_signer_from_pool(args, call)
    private_key, _seed, _ss58 = _deploy_credentials(args)
    response = call("/v1/pool/registration-state", {})
    registrations = {
        str(key): dict(value)
        for key, value in (response.get("registrations") or {}).items()
        if isinstance(value, dict)
    }
    single = response.get("registration")
    if isinstance(single, dict) and single.get("model_id"):
        registrations.setdefault(str(single["model_id"]), dict(single))
    if not registrations:
        raise SystemExit(
            "no registration stored on the pool manager; run "
            "`verathos mesh deploy` first"
        )
    renewed = []
    failures = []
    for stored in registrations.values():
        # Per-model isolation: one expired lease or transient RPC error
        # must not stop the OTHER models' renewals - on a cron-driven
        # renew an early abort silently let every later lease lapse.
        try:
            target = registration_target_from_state(stored)
            outcome = renew_once(
                chain_config,
                target,
                int(stored["index"]),
                private_key=private_key,
            )
            call(
                "/v1/pool/registration-state",
                {"registration": {**stored, "expires_at": outcome.expires_at}},
            )
            renewed.append(
                {
                    "model_id": stored.get("model_id"),
                    "index": outcome.index,
                    "tx_hash": outcome.tx_hash,
                    "expires_at": outcome.expires_at,
                }
            )
        except Exception as exc:
            failures.append(
                {
                    "model_id": stored.get("model_id"),
                    "index": stored.get("index"),
                    "error": str(exc)[:300],
                }
            )
    _print_json(
        {
            "status": "ok" if not failures else "partial",
            "renewed": renewed,
            **({"failed": failures} if failures else {}),
        }
    )
    if failures:
        raise SystemExit(1)


def cmd_registration_status(args: argparse.Namespace) -> None:
    call = _pool_client(args)
    response = call("/v1/pool/registration-state", {})
    registrations = dict(response.get("registrations") or {})
    stored = response.get("registration")
    if not stored and len(registrations) == 1:
        stored = next(iter(registrations.values()))
    if not stored and not registrations:
        _print_json({"registration": None, "registrations": {}})
        return
    payload: dict[str, Any] = {
        "registration": stored,
        "registrations": registrations,
    }
    # Chain read-back needs ONE registration to compare; with several
    # stored, the JSON above already carries them all.
    if args.chain_config and stored:
        from eth_account import Account

        from verallm.chain.config import ChainConfig
        from verallm.chain.miner_lifecycle import resolve_registered_index
        from verallm.chain.miner_registry import MinerRegistryClient
        from verallm.mesh.registration import registration_target_from_state

        chain_config = ChainConfig.from_json(args.chain_config)
        private_key, _seed, _ss58 = _deploy_credentials(args)
        signer = Account.from_key(private_key).address
        entries = MinerRegistryClient(chain_config).get_miner_models(signer)
        target = registration_target_from_state(stored)
        try:
            index = resolve_registered_index(entries, target)
            entry = entries[index]
            payload["chain"] = {
                "index": index,
                "active": bool(entry.active),
                "expires_at": int(entry.expires_at),
                "expires_in_s": int(entry.expires_at) - int(time.time()),
            }
        except ValueError as exc:
            payload["chain"] = {"error": str(exc)}
    _print_json(payload)


def cmd_apikey(args: argparse.Namespace) -> None:
    """Mint/list/revoke keys for the pool's private OpenAI API.

    The API serves this pool's meshes to the operator's own tools
    (OpenAI-compatible, proofs always on), independent of subnet
    registration. Keys are stored hashed in the owner-only pool state;
    the cleartext key is printed exactly once here.
    """

    from verallm.mesh import render

    if args.action == "expose":
        from verallm.mesh import onboarding

        port = int(getattr(args, "port", 0) or 0)
        if port:
            print(
                f"reconfiguring the manager with an https API listener on "
                f"port {port} (workers keep their plain-http port)..."
            )
        else:
            print("disabling the https API listener...")
        manager_port = onboarding.set_manager_api_tls_port(port)
        deadline = time.monotonic() + 30
        healthy = False
        while time.monotonic() < deadline:
            try:
                import urllib.request as _request

                with _request.urlopen(
                    f"http://127.0.0.1:{manager_port}/healthz", timeout=3
                ) as response:
                    healthy = response.status == 200
                    break
            except Exception:
                time.sleep(1)
        if not healthy:
            raise SystemExit(
                "the manager did not come back healthy after the restart; "
                "check pm2 logs verathos-pool-manager"
            )
        if not port:
            print(render.ok("https API listener disabled; manager healthy"))
            return
        host = onboarding.detect_public_ip() or "<public-ip>"
        ok, detail = onboarding.test_api_tls_endpoint(host, port)
        url = f"https://{host}:{port}"
        if ok:
            print(render.ok(f"private API live on the internet: {url}"))
        else:
            print(
                render.warn(
                    f"listener is up locally but the self-test against "
                    f"{url} failed: {detail}"
                )
            )
        pool_dir = ""
        try:
            token = _management_pool_token(args)
            from verallm.mesh.pool import known_pool_dirs

            for candidate in known_pool_dirs():
                if candidate.name == token.pool_id:
                    pool_dir = str(candidate)
                    break
        except Exception:
            pass
        pin = ""
        if pool_dir:
            pin = onboarding.api_tls_pubkey_pin(Path(pool_dir))
            if pin:
                print(f"  certificate public-key pin: sha256//{pin}")
        print("  test it with your key:")
        if pin:
            print(
                f"    curl --insecure --pinnedpubkey 'sha256//{pin}' "
                f"-H 'Authorization: Bearer <key>' {url}/v1/models"
            )
        else:
            print(f"    curl -H 'Authorization: Bearer <key>' {url}/v1/models")
        return

    call = _pool_client(args)
    if args.action == "create":
        payload = call(
            "/v1/pool/api-keys", {"action": "create", "name": args.name}
        )
        endpoint = str(
            getattr(call, "manager_endpoint", "") or "http://127.0.0.1:9500"
        ).rstrip("/")
        print(render.ok("API key minted (shown ONCE, store it now):"))
        print(f"  {payload['api_key']}")
        print()
        print("use it with any OpenAI client against the pool manager:")
        print(
            f"  curl -H 'Authorization: Bearer <key>' {endpoint}/v1/models"
        )
        if any(
            marker in endpoint
            for marker in ("127.0.0.1", "localhost", "[::1]")
        ):
            print(
                render.dim(
                    "  (loopback endpoint: reachable on this machine only. "
                    "For internet access put TLS in front of the manager:\n"
                    "   sudo bash scripts/setup_https.sh --port 9543 "
                    "--backend-port 9500 --append\n"
                    "   then use https://<public-ip-or-domain>:9543/v1/...)"
                )
            )
        return
    if args.action == "revoke":
        if not args.key_id:
            raise SystemExit("revoke needs --key-id (from `apikey list`)")
        payload = call(
            "/v1/pool/api-keys", {"action": "revoke", "key_id": args.key_id}
        )
        _print_json(payload)
        return
    _print_json(call("/v1/pool/api-keys", {"action": "list"}))


def cmd_retire(args: argparse.Namespace) -> None:
    """Take the registered model OFF the subnet deliberately.

    `mesh pool stop` only tears down the local mesh and leaves the chain
    entry to lapse passively over 24h, which keeps the endpoint claimed
    (reusing the URL from another key then fails with an endpoint claim
    collision). Retiring deactivates the entry on chain, releasing the
    endpoint, and clears the stored registration so the lease renewer
    stops resurrecting it. --keep-serving keeps the mesh answering local
    (and private API) traffic; only the chain entry goes.
    """

    from eth_account import Account

    from verallm.chain.config import ChainConfig
    from verallm.chain.miner_registry import MinerRegistryClient
    from verallm.mesh import render

    subtensor_network = str(
        getattr(args, "subtensor_network", "") or ""
    )

    config_path = ChainConfig.resolve_config_path(
        args.chain_config or None,
        str(getattr(args, "subtensor_network", "") or "") or None,
    )
    if not config_path:
        raise SystemExit(
            "pass --subtensor-network test|finney or an explicit "
            "--chain-config"
        )
    chain_config = ChainConfig.from_json(config_path)
    chain_endpoint = str(
        getattr(args, "subtensor_chain_endpoint", "") or ""
    ).strip()
    if chain_endpoint:
        # One node for BOTH sides of the deploy: EVM reads/writes (rpc_url,
        # ws:// converted to http://) and substrate UID resolution. Without
        # this every chain step crawled through the congested public RPC
        # even when the pool ran its own subtensor.
        from dataclasses import replace as _dc_replace

        chain_config = _dc_replace(
            chain_config,
            rpc_url=ChainConfig.resolve_rpc_url(
                chain_endpoint, subtensor_network or None
            ),
        )
    call = _pool_client(args)
    # Pool identity BEFORE credential resolution, like deploy/renew: a
    # flagless retire otherwise dies inside _deploy_credentials before
    # the pool's stored signer can fill in.
    _default_signer_from_pool(args, call)
    private_key, _seed, _ss58 = _deploy_credentials(args)
    stored = _stored_registration(call, str(args.model_id or ""))
    if not stored:
        raise SystemExit(
            "no matching registration stored on the pool manager; nothing "
            "to retire"
        )
    model_id = str(stored.get("model_id", "") or "")
    index = int(stored["index"])
    mesh_key = str(stored.get("mesh_key", "") or "")
    if not args.yes:
        print(
            f"retire {model_id} @ index {index} "
            f"(endpoint {stored.get('endpoint', '?')}) from the subnet?"
        )
        if input("type 'retire' to confirm: ").strip() != "retire":
            raise SystemExit("aborted")

    if not args.keep_serving and mesh_key:
        print(f"stopping mesh {mesh_key}...")
        try:
            call("/v1/pool/stop", {"mesh_key": mesh_key})
        except Exception as exc:
            print(render.warn(f"mesh stop failed ({exc}); continuing"))
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            status = call("/v1/pool/status", {})
            if mesh_key not in (status.get("meshes") or {}):
                break
            time.sleep(3)

    registry = MinerRegistryClient(chain_config)
    signer = Account.from_key(private_key).address
    print(f"deactivating index {index} on chain (releases the endpoint)...")
    tx_hash = registry.deactivate_model(index, private_key=private_key)
    # Fresh client: the read-back must not hit this instance's 60s TTL cache.
    entries = MinerRegistryClient(chain_config).get_miner_models(signer)
    if index < len(entries) and entries[index].active:
        raise SystemExit(
            f"deactivateModel sent (tx {tx_hash}) but index {index} still "
            "reads active; NOT clearing the stored registration. Check the "
            "tx and re-run."
        )
    # model_id-scoped: a pool holds one registration PER model now, and a
    # bare clear on a multi-model pool is refused by the manager.
    call("/v1/pool/registration-state", {"clear": True, "model_id": model_id})
    _print_json(
        {
            "status": "ok",
            "retired": {"model_id": model_id, "index": index},
            "tx_hash": tx_hash,
            "kept_serving": bool(args.keep_serving),
        }
    )


def cmd_pool_create(args: argparse.Namespace) -> None:
    from verallm.mesh.pool import (
        POOL_ADMIN_TOKEN_FILE,
        POOL_TOKEN_FILE,
        create_pool_state,
    )

    if getattr(args, "network", ""):
        from verallm.mesh.onboarding import resolve_network

        binding = resolve_network(args.network)
        if binding is not None:
            args.chain_id = args.chain_id or binding["chain_id"]
            args.netuid = args.netuid or binding["netuid"]

    out, _token = create_pool_state(
        args.root,
        manager_endpoint=args.manager_endpoint,
        serving_mode=args.serving_mode,
        owner_account=args.owner_account,
        coordinator_address=args.coordinator_address,
        validator_shared_state_path=args.validator_shared_state,
        chain_id=args.chain_id,
        netuid=args.netuid,
        coordinator_uid=args.coordinator_uid,
        epoch=args.epoch,
        snapshot_ttl_seconds=args.snapshot_ttl_seconds,
    )
    print(f"Pool state: {out}")
    print(f"Worker token file: {out / POOL_TOKEN_FILE}")
    print(f"Pool admin token file: {out / POOL_ADMIN_TOKEN_FILE}")


def cmd_pool_upgrade(args: argparse.Namespace) -> None:
    """Flip a dev pool to subnet mode in place (state + tokens only).

    The interactive lane (`verathos mesh manage`, "go live on the subnet")
    additionally restarts the manager with the signing wallet and rejoins
    the local workers; this flag-driven command performs the state
    transition and prints those next steps for scripted setups.
    """

    from verallm.mesh import render
    from verallm.mesh.pool import upgrade_pool_state_to_subnet

    if getattr(args, "network", ""):
        from verallm.mesh.onboarding import resolve_network

        binding = resolve_network(args.network)
        if binding is not None:
            args.chain_id = args.chain_id or binding["chain_id"]
            args.netuid = args.netuid or binding["netuid"]

    try:
        pool_dir, token, endpoint_changed = upgrade_pool_state_to_subnet(
            args.pool,
            owner_account=args.owner_account,
            coordinator_address=args.coordinator_address,
            validator_shared_state_path=args.validator_shared_state,
            chain_id=args.chain_id,
            netuid=args.netuid,
            coordinator_uid=args.coordinator_uid,
            epoch=args.epoch,
            snapshot_ttl_seconds=args.snapshot_ttl_seconds,
            manager_endpoint=args.manager_endpoint or None,
        )
    except ValueError as exc:
        raise SystemExit(f"upgrade refused: {exc}")
    print(render.ok(f"pool {pool_dir.name} is now a subnet pool (in place)"))
    if endpoint_changed:
        print(
            render.warn(
                "the manager endpoint changed: worker tokens were re-minted "
                "(same secret, new endpoint). Every remote worker needs the "
                "new token: verathos mesh pool join-token --pool "
                f"{pool_dir}"
            )
        )
    print("next steps:")
    print(
        "  1. restart the manager WITH the signing wallet so snapshots and "
        "the lease renewer work:\n"
        "     pm2 delete verathos-pool-manager; then verathos mesh pool "
        f"serve --pool {pool_dir} ... --wallet-name <wallet> "
        "--wallet-hotkey <hotkey> --chain-config <chain_config.json> "
        "--subtensor-network <test|finney>\n"
        "     (the manage board's 'go live' action does this for you)"
    )
    print(
        "  2. restart every worker unit (pm2 restart <unit>); rejoining "
        "binds each worker's stage identity. Subnet drivers additionally "
        "need --wallet-name/--wallet-hotkey and a fresh validator "
        "allowlist (join_pool.sh flags)."
    )
    print("  3. register a model: verathos mesh deploy <model_id> ...")


def cmd_pool_join_token(args: argparse.Namespace) -> None:
    """Re-print the worker join one-liner.

    Before this command the token was effectively write-once: printed at
    pool creation and never again, leaving operators to cat token files by
    hand when adding a machine later.
    """

    from verallm.mesh import render
    from verallm.mesh.onboarding import (
        endpoint_is_loopback,
        join_command_for_token,
    )
    from verallm.mesh.pool import (
        POOL_TOKEN_FILE,
        known_pool_dirs,
        load_pool_token_file,
    )

    pool_dir = Path(args.pool).expanduser() if args.pool else None
    if pool_dir is None:
        candidates = [
            candidate
            for candidate in known_pool_dirs()
            if (candidate / POOL_TOKEN_FILE).exists()
        ]
        if not candidates:
            raise SystemExit(
                "no pool found on this machine; run this on the coordinator "
                "box or pass --pool DIR"
            )
        if len(candidates) > 1:
            candidates.sort(
                key=lambda c: (c / POOL_TOKEN_FILE).stat().st_mtime,
                reverse=True,
            )
            if _interactive_terminal():
                print("several pools on this machine:")
                for index, candidate in enumerate(candidates, start=1):
                    print(f"  {index}. {candidate}")
                while True:
                    raw = input(
                        f"select [1-{len(candidates)}, Enter=1 (newest)]: "
                    ).strip() or "1"
                    if raw.isdigit() and 1 <= int(raw) <= len(candidates):
                        candidates = [candidates[int(raw) - 1]]
                        break
                    print("  pick a number from the list")
            else:
                listing = "\n".join(f"  {candidate}" for candidate in candidates)
                raise SystemExit(
                    f"several pools on this machine:\n{listing}\n"
                    "pick one with --pool DIR"
                )
        pool_dir = candidates[0]
    token = load_pool_token_file(pool_dir / POOL_TOKEN_FILE)
    remote_endpoint = str(getattr(args, "endpoint", "") or "").strip().rstrip("/")
    if not remote_endpoint:
        # The pool knows its TLS listener and the box can detect its public
        # IP, so the REMOTE one-liner is mintable without any flag. Printing
        # only the loopback token here left operators guessing how to add a
        # second machine (live user feedback: "it has to be a selection").
        try:
            _state = json.loads(
                (pool_dir / "pool-state.json").read_text(encoding="utf-8")
            )
        except Exception:
            _state = {}
        _api_tls_port = int(_state.get("api_tls_port", 0) or 0)
        # The pool's STORED endpoint is authoritative when it is already a
        # remote https URL (the wizard resolves and persists it at setup):
        # re-deriving it from a public-IP probe silently skipped the CA-pin
        # block on containers where detection returns nothing, minting
        # tokens whose joins fail certificate verification. Probe only as
        # the fallback.
        _stored = str(_state.get("manager_endpoint", "") or "").strip().rstrip("/")
        if (
            _stored.startswith("https://")
            and not endpoint_is_loopback(_stored)
        ):
            remote_endpoint = _stored
        elif _api_tls_port:
            from verallm.mesh.onboarding import detect_public_ip

            _public_ip = detect_public_ip()
            if _public_ip:
                remote_endpoint = f"https://{_public_ip}:{_api_tls_port}"
                print(
                    render.dim(
                        "remote manager endpoint auto-detected: "
                        f"{remote_endpoint} (override with --endpoint)"
                    )
                )
    pubkey_pin = ""
    if remote_endpoint:
        from urllib.parse import urlparse

        parsed = urlparse(remote_endpoint)
        if parsed.scheme != "https" and not endpoint_is_loopback(remote_endpoint):
            raise SystemExit(
                "remote join tokens require an https:// manager endpoint "
                "(the manager's --api-tls-port listener serves the full API)"
            )
        from dataclasses import replace as dc_replace

        cert_path = pool_dir / "api-tls-cert.pem"
        if not cert_path.is_file():
            raise SystemExit(
                "the pool has no API TLS certificate yet; start the manager "
                "with --api-tls-port so it mints one, then re-run"
            )
        pem = cert_path.read_text(encoding="utf-8")
        der = ssl.PEM_cert_to_DER_cert(pem)
        cert_sha256 = hashlib.sha256(der).hexdigest()
        token = dc_replace(
            token,
            manager_endpoint=remote_endpoint,
            manager_ca_sha256=cert_sha256,
        )
        # curl-compatible public-key pin so even the bootstrap fetch is
        # authenticated against this pool's certificate.
        from verallm.mesh.onboarding import api_tls_pubkey_pin

        pubkey_pin = api_tls_pubkey_pin(pool_dir)
    print(
        f"{render.bold('pool')}    {token.pool_id}\n"
        f"{render.bold('manager')} {token.manager_endpoint}"
    )
    if endpoint_is_loopback(token.manager_endpoint):
        print(
            render.warn(
                "the manager endpoint is loopback: this token only works on "
                "THIS machine. Mint a remote token with --endpoint "
                "https://<public-ip>:<api-tls-port> to add other machines."
            )
        )
    print()
    if remote_endpoint and pubkey_pin:
        # The token is the WHOLE setup: chain coordinates (subtensor
        # network, netuid) come from the pool in the join response, so the
        # printed command carries no chain flags at all. The command must
        # run VERBATIM: no placeholders, nothing to edit. The join script
        # itself asks how the machine is reachable (IPs auto-detected)
        # when a terminal is present.
        print(render.section("Add a machine"))
        print(
            "  Copy-paste on any GPU machine (Linux or macOS). "
            "Nothing to edit; needs curl, python3, and tar on the box"
        )
        print(
            "  (bare containers: "
            + render.dim("apt-get update && apt-get install -y curl")
            + " first):"
        )
        print()
        print(
            render.cyan(
                f"  curl -fsSk --pinnedpubkey 'sha256//{pubkey_pin}' \\\n"
                f"    {token.manager_endpoint}/install.sh | bash -s -- \\\n"
                f"    --token {token.encode()}"
            )
        )
        print()
        print(f"  {render.bold('What happens next')}")
        for line in (
            "installs its own checkout and builds llama.cpp for that "
            "machine's GPU",
            "joins this pool and asks how the pool can reach the machine "
            "(addresses are auto-detected, never typed)",
            "no wallet needed: this manager keeps the coordinator wallet "
            "and signs for whichever worker it appoints as driver",
            "chain settings (network, netuid) come from the pool at join",
        ):
            print(f"    {render.dim('-')} {line}")
        print()
        print(f"  {render.bold('Optional flags')} (append only if needed)")
        for flag, meaning in (
            ("--member-only", "serve pipeline stages only, never drive"),
            (
                "--gpus 0,1",
                "which GPUs to enroll (default: all detected)",
            ),
            (
                "--advertise-host IP",
                "force the address other machines dial (default: detected)",
            ),
            (
                "--mesh-port N --proof-port N+2 --rpc-port N+3",
                "the box's PUBLISHED ports (port-mapped containers); "
                "mesh serving also binds mesh_port+1, so leave that slot "
                "free",
            ),
            (
                "--subtensor-network X --netuid N",
                "override the chain the pool advertises at join",
            ),
            (
                "--wallet-name W --wallet-hotkey H",
                "sign locally instead of through the manager",
            ),
        ):
            # Pad BEFORE colorizing: ANSI escapes would break the column.
            print(f"    {render.green(f'{flag:<36}')} {render.dim(meaning)}")
        print()
        print(f"  {render.bold('Already has a verathos checkout?')}")
        print(
            "    " + render.cyan("bash scripts/join_pool.sh --token <same token>")
        )
        print()
        print(f"  {render.bold('Security')}")
        for line in (
            "--pinnedpubkey authenticates the installer download against "
            "this pool's TLS key; the token pins every later connection",
            "the token only lets a machine join and serve; the admin "
            "token never leaves this box",
        ):
            print(f"    {render.dim('-')} {render.dim(line)}")
    else:
        print(render.section("Add this machine"))
        print("  From a verathos checkout on the GPU machine:")
        print()
        print(render.cyan(f"  bash scripts/join_pool.sh --token {token.encode()}"))
        print()
        print(
            render.dim(
                "after the next public release the one-liner works too:\n  "
                + join_command_for_token(token.encode())
            )
        )
        print()
        print(
            render.dim(
                "the token only lets a machine join and serve; the admin "
                "token never leaves this box"
            )
        )


def _pool_manifest_base_urls(args: argparse.Namespace) -> tuple[str, ...]:
    """Manifest store this pool's drivers may fetch from, from chain config.

    Without this the store is reachable only through an environment
    variable, so a fresh driver rebuilt the tensor manifest from scratch
    (hashing every tensor of, for a 238GB GGUF, a quarter of a terabyte)
    even though the pool's own chain config named a store serving it.
    """

    from verallm.mesh import render

    try:
        from verallm.chain.config import ChainConfig

        # Same resolution as every chain-touching command: an explicit
        # --chain-config wins, else --subtensor-network selects the
        # shipped config. A manager started with only the network flag
        # (a supported setup) otherwise silently lost the store and its
        # drivers fell back to the never-acceptable local manifest
        # rebuild.
        path = ChainConfig.resolve_config_path(
            str(getattr(args, "chain_config", "") or "").strip() or None,
            str(getattr(args, "subtensor_network", "") or "").strip() or None,
        )
        if not path:
            return ()
        return tuple(ChainConfig.from_json(path).mesh_manifest_base_urls or ())
    except Exception as exc:  # config is optional; never block serving
        print(
            render.dim(f"manifest store not configured ({exc})"),
            flush=True,
        )
        return ()


def cmd_pool_serve(args: argparse.Namespace) -> None:
    from verallm.mesh.pool import serve_pool_manager

    server = serve_pool_manager(
        args.pool,
        manifest_base_urls=_pool_manifest_base_urls(args),
        host=args.host,
        port=args.port,
        coordinator_address=args.coordinator_address or None,
        validator_shared_state_path=args.validator_shared_state or None,
        tls_certfile=args.tls_cert or None,
        tls_keyfile=args.tls_key or None,
        api_tls_port=int(getattr(args, "api_tls_port", 0) or 0),
        api_tls_certfile=getattr(args, "api_tls_cert", "") or None,
        api_tls_keyfile=getattr(args, "api_tls_key", "") or None,
        # Coordinator announce (proxy address-book registration) rides the
        # same chain config the lease renewer uses; without one the pool
        # is chain-blind and has nothing to announce to.
        announce_chain_config=getattr(args, "chain_config", "") or None,
    )
    api_tls_server = getattr(server, "api_tls_server", None)
    if api_tls_server is not None:
        threading.Thread(
            target=api_tls_server.serve_forever,
            daemon=True,
            name="mesh-api-tls",
        ).start()
        print(
            "private API https listener on "
            f"https://{args.host}:{args.api_tls_port} "
            "(self-signed unless --api-tls-cert/--api-tls-key given)",
            flush=True,
        )
    # Persist the chain identity the manager actually runs with, so status
    # surfaces (board, dashboard, operator score) can name the bound
    # wallet/hotkey/network instead of coming up empty. Best-effort: a
    # missing hotkey file must not stop the control plane.
    coordinator_hotkey_ss58 = ""
    if args.wallet_name:
        try:
            from verallm.mesh.receipt_signing import load_hotkey_keypair

            coordinator_hotkey_ss58 = str(
                load_hotkey_keypair(
                    args.wallet_name, args.wallet_hotkey
                ).ss58_address
            )
        except Exception as exc:
            print(
                f"note: could not resolve the hotkey SS58 ({exc}); "
                "status shows wallet names only",
                flush=True,
            )
    server.pool_manager.update_chain_identity(
        subtensor_network=getattr(args, "subtensor_network", ""),
        wallet_name=args.wallet_name,
        wallet_hotkey=args.wallet_hotkey if args.wallet_name else "",
        coordinator_hotkey_ss58=coordinator_hotkey_ss58,
    )
    if args.wallet_name and args.chain_config:
        # The manager is the coordinator box's only always-on process, so it
        # owns the 24h MinerRegistry lease renewal once a deploy has stored
        # the registration (POST /v1/pool/registration-state). Operators who
        # will not give the manager a signing key run `verathos mesh renew`
        # from cron instead.
        from verallm.chain.cli_credentials import resolve_cli_evm_private_key
        from verallm.chain.config import ChainConfig
        from verallm.mesh.registration import run_lease_renewer

        private_key = resolve_cli_evm_private_key(
            wallet_name=args.wallet_name,
            hotkey_name=args.wallet_hotkey,
            required=True,
        )
        chain_config = ChainConfig.from_json(args.chain_config)
        threading.Thread(
            target=run_lease_renewer,
            kwargs={
                "manager": server.pool_manager,
                "chain_config": chain_config,
                "private_key": private_key,
            },
            daemon=True,
            name="mesh-lease-renewer",
        ).start()
        print("lease renewer active (12h window, 15min ticks)", flush=True)
    scheme = "https" if args.tls_cert else "http"
    print(
        f"mesh pool manager listening on {scheme}://{args.host}:{args.port}",
        flush=True,
    )
    server.serve_forever()


def _require_native_prover_on_gpu_boxes() -> None:
    """Refuse to start a GPU proof worker on the NumPy sumcheck fallback.

    The fallback is 3-6x slower and silently degrades every hard proof
    this worker produces; a proof-capable GPU worker starting without
    the native zkllm extension is a broken install, not a preference.
    CPU-only machines (Mac members, dev laptops) keep the fallback with
    the existing warning, and VERATHOS_ALLOW_SLOW_SUMCHECK=1 remains the
    explicit dev escape hatch.
    """

    if os.environ.get("VERATHOS_ALLOW_SLOW_SUMCHECK", "") == "1":
        return
    name, _vram, _names, _per_gpu = _detect_gpu_capability()
    if not name:
        return
    try:
        from zkllm.crypto import sumcheck_fast

        if getattr(sumcheck_fast, "_HAS_CUDA", False) or getattr(
            sumcheck_fast, "_HAS_NATIVE", False
        ):
            return
    except Exception:
        pass
    raise SystemExit(
        "this GPU machine has no zkllm native extension, so every hard "
        "proof would run on the slow NumPy fallback.\n"
        "  fix:      re-run bash scripts/setup_mesh.sh "
        "(it builds the extension)\n"
        "  manually: cd zkllm/cuda && python build.py\n"
        "  dev only: VERATHOS_ALLOW_SLOW_SUMCHECK=1 to start anyway"
    )


def _detect_gpu_capability() -> tuple[str, int, list[str], list[int]]:
    """Best-effort (display_name, total_vram_gb, names, per_gpu_vram_gb).

    nvidia-smi does NOT respect CUDA_VISIBLE_DEVICES (it is a driver tool,
    not a CUDA app), so the mask is applied here: a worker pinned to a GPU
    group advertises exactly those GPUs, in mask order, and vram_gb is the
    SUM across them (a multi-GPU worker's placement capacity). Advertising
    unmasked GPUs is not cosmetic: the per-GPU list sizes the mesh device
    plan, and a 2-GPU worker advertising 4 makes the driver bind RPC
    devices its rpc-server never exposes. Returns ("", 0, [], []) on any
    failure; callers only use this when the operator did not pass values
    explicitly, so a detection failure degrades rather than lying.
    """

    import subprocess

    try:
        raw = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if raw.returncode != 0:
            return "", 0, [], []
        by_index: dict[int, tuple[str, int]] = {}
        for line in raw.stdout.strip().splitlines():
            head, _, mem = line.rpartition(",")
            index_text, _, name = head.partition(",")
            if not name:
                continue
            try:
                index = int(index_text.strip())
                vram = max(0, int(float(mem.strip()) / 1024))
            except ValueError:
                continue
            by_index[index] = (name.strip(), vram)
        mask = os.environ.get("CUDA_VISIBLE_DEVICES")
        if mask is not None:
            wanted: list[int] = []
            for part in mask.split(","):
                part = part.strip()
                if not part:
                    continue
                try:
                    wanted.append(int(part))
                except ValueError:
                    # UUID-style masks: no index mapping; keep all GPUs
                    # rather than misreporting an empty machine.
                    wanted = sorted(by_index)
                    break
            selected = [by_index[i] for i in wanted if i in by_index]
        else:
            selected = [by_index[i] for i in sorted(by_index)]
        names = [name for name, _ in selected]
        per_gpu = [vram for _, vram in selected]
        if not names:
            return "", 0, [], []
        display = names[0]
        if len(names) > 1:
            display = (
                f"{names[0]} x{len(names)}"
                if len(set(names)) == 1
                else ", ".join(names)
            )
        return display, sum(per_gpu), names, per_gpu
    except Exception:
        return "", 0, [], []


def cmd_pool_worker(args: argparse.Namespace) -> None:
    from verallm.mesh.pool import (
        POOL_TOKEN_SCOPE_WORKER,
        PoolWorkerConfig,
        pool_worker_loop,
    )

    _require_native_prover_on_gpu_boxes()
    catalog = json.loads(Path(args.catalog).read_text()) if args.catalog else []
    gpu_name = str(args.gpu_name or "")
    vram_gb = int(args.vram_gb or 0)
    detected_name, detected_vram, gpu_names, per_gpu_vram = _detect_gpu_capability()
    if not gpu_name or vram_gb <= 0:
        # Auto-detect so recommend() sees real capacity; explicit flags win
        # for the scalar fields, but the per-GPU breakdown (which drives the
        # multi-device tensor split) is always what nvidia-smi sees inside
        # this worker's CUDA_VISIBLE_DEVICES mask.
        gpu_name = gpu_name or detected_name
        vram_gb = vram_gb if vram_gb > 0 else detected_vram
    config = PoolWorkerConfig(
        token=_pool_token_from_args(args, required_scope=POOL_TOKEN_SCOPE_WORKER),
        manager_ca_file=(
            str(Path(args.manager_ca_file).expanduser().resolve())
            if getattr(args, "manager_ca_file", "") else ""
        ),
        repo_root=Path(args.repo_root or Path(__file__).resolve().parents[2]),
        workdir=Path(args.workdir),
        advertise_host=args.advertise_host,
        rpc_port=args.rpc_port,
        proof_port=args.proof_port,
        mesh_port=args.mesh_port,
        llama_server_binary=args.llama_server_binary,
        rpc_worker_binary=args.rpc_worker_binary,
        rpc_device=args.rpc_device,
        catalog=catalog,
        catalog_path=str(args.catalog or ""),
        wallet_name=str(getattr(args, "wallet_name", "") or ""),
        wallet_hotkey=str(getattr(args, "wallet_hotkey", "") or ""),
        validator_allowlist_path=str(
            getattr(args, "validator_allowlist_path", "") or ""
        ),
        validator_allowlist_max_age_seconds=float(
            getattr(
                args,
                "validator_allowlist_max_age_seconds",
                DEFAULT_VALIDATOR_ALLOWLIST_MAX_AGE_SECONDS,
            )
        ),
        stage_proof_key_file=(
            Path(args.stage_proof_key_file).expanduser()
            if str(getattr(args, "stage_proof_key_file", "") or "").strip()
            else None
        ),
        worker_id=args.worker_id,
        gpu_name=gpu_name,
        vram_gb=vram_gb,
        gpu_names=gpu_names,
        per_gpu_vram_gb=per_gpu_vram,
        heartbeat_s=args.heartbeat,
        member_only=bool(args.member_only),
    )
    allowlist_network = str(getattr(args, "subtensor_network", "") or "")
    allowlist_netuid = getattr(args, "netuid", None)
    refresher_started = {"done": False}

    def start_allowlist_refresher(network: str, netuid: int) -> None:
        # The coordinator refuses an allowlist whose metagraph refresh is
        # stale, so a subnet worker that can drive must keep its own copy
        # fresh (a standalone mesh miner has no vLLM miner process to do
        # it). Daemon thread: a chain hiccup logs and retries, never kills
        # the worker.
        if refresher_started["done"]:
            return
        refresher_started["done"] = True
        if not config.validator_allowlist_path:
            # Token-only workers set nothing up by hand; give the allowlist
            # its standard home so driver readiness needs zero flags.
            config.validator_allowlist_path = str(
                Path.home() / ".verathos" / "validator-allowlist.json"
            )
        import threading

        from verallm.mesh.allowlist import (
            preimport_chain_modules,
            run_allowlist_refresher,
        )

        # Import the chain stack HERE, on the main thread: doing it inside
        # the refresher races the worker's own startup imports and CPython
        # kills the first refresh with a module-lock deadlock.
        preimport_chain_modules()

        threading.Thread(
            target=run_allowlist_refresher,
            kwargs=dict(
                subtensor_network=network,
                netuid=int(netuid),
                out_path=config.validator_allowlist_path,
                chain_config_path=str(
                    getattr(args, "chain_config", "") or ""
                ),
                allow_extra=tuple(
                    part.strip()
                    for part in str(
                        getattr(args, "allow_validators", "") or ""
                    ).split(",")
                    if part.strip()
                ),
            ),
            daemon=True,
            name="allowlist-refresher",
        ).start()
        print(
            f"validator allowlist refresher: netuid {netuid} on "
            f"{network} -> {config.validator_allowlist_path}",
            flush=True,
        )

    if allowlist_network and allowlist_netuid is not None:
        start_allowlist_refresher(allowlist_network, int(allowlist_netuid))

    def on_join_info(joined: dict) -> None:
        # The stock installer waits for this exact stdout verdict. Do not rely
        # on library INFO visibility: imported logging handlers can suppress
        # that line even though the manager accepted the worker.
        _print_pool_join_accepted(joined)
        # A worker started with ONLY the token learns the pool's chain
        # coordinates from the join response; explicit flags always win.
        network = str(joined.get("subtensor_network", "") or "")
        netuid = joined.get("netuid")
        if network and netuid is not None:
            start_allowlist_refresher(network, int(netuid))

    print(f"pool worker joining {config.token.manager_endpoint}", flush=True)
    pool_worker_loop(config, on_join_info=on_join_info)


def _print_pool_join_accepted(joined: dict) -> None:
    """Emit the durable stock-installer join-success marker."""

    print(
        f"pool join accepted: worker '{str(joined.get('worker_id', '') or '')}'",
        flush=True,
    )


def cmd_pool_status(args: argparse.Namespace) -> None:
    _print_json(_pool_client(args)("/v1/pool/status", {}))


def cmd_pool_recommend(args: argparse.Namespace) -> None:
    _print_json(_pool_client(args)("/v1/pool/recommend", {"model_id": args.model_id}))


def _management_pool_token(args: argparse.Namespace):
    """Zero-flag token resolution, rejecting worker-scoped tokens."""
    from verallm.mesh.pool import POOL_TOKEN_SCOPE_MANAGEMENT

    token = _resolve_pool_context(args)
    if token.scope != POOL_TOKEN_SCOPE_MANAGEMENT:
        raise SystemExit(
            "this command needs a management token; this box only has a "
            "worker token. Run it on the coordinator box or pass "
            "--pool-token-file with the pool admin token."
        )
    return token


def _select_serving_mesh(
    meshes: Mapping[str, Any],
    requested: str,
    *,
    interactive: bool,
    workers: Mapping[str, Any] | None = None,
) -> tuple[str, Mapping[str, Any]]:
    """Resolve which serving mesh to talk to, asking when ambiguous."""
    if requested:
        mesh = meshes.get(requested)
        if mesh is None:
            raise SystemExit(f"unknown mesh: {requested}")
        return requested, mesh
    serving = sorted(
        (key, mesh)
        for key, mesh in meshes.items()
        if str(mesh.get("status", "")) == "serving"
    )
    if not serving:
        raise SystemExit(
            "no mesh is serving. Launch one first:\n"
            "  verathos mesh pool launch"
        )
    if len(serving) == 1:
        return serving[0]
    if not interactive:
        raise SystemExit(
            "pass --mesh-key; serving meshes: "
            + ", ".join(key for key, _ in serving)
        )
    from verallm.mesh import panels

    hardware_view = {"workers": dict(workers or {})}
    print(_c("serving meshes:", "1"))
    for index, (key, mesh) in enumerate(serving, start=1):
        where = panels.mesh_hardware_label(hardware_view, mesh)
        if not where:
            where = "workers=" + ",".join(mesh.get("members") or [])
        print(
            f"  {_c(str(index), '1;36')}. {key}"
            f"  {mesh.get('model_id', '?')}"
            f"  · {where}"
        )
    while True:
        raw = (
            input(f"which mesh? [1-{len(serving)}, Enter=1, b=back] ")
            .strip()
            .lower()
        )
        if raw in ("b", "back", "q"):
            raise SystemExit(0)
        if not raw:
            return serving[0]
        if raw.isdigit() and 1 <= int(raw) <= len(serving):
            return serving[int(raw) - 1]
        print("enter a number from the list")


_PROOF_TIER_LABELS = {
    "light": "light  what real user traffic rides",
    "hard": "hard   what a hard canary audit runs",
}


def _select_proof_tier() -> str:
    """Ask which canary shape this test chat should reproduce.

    Chat is an INSTRUMENT for checking the mesh works, not a product
    surface, so it offers exactly the two shapes a validator produces
    and nothing else. There is deliberately no third "auto" option: a
    tier that resolved differently here than it does for a canary would
    make the instrument lie about the thing it exists to test.
    """

    print()
    print(_c("which canary shape should this test chat reproduce?", "1"))
    print(
        _c("  1) light  ", "2")
        + "what real user traffic rides (milliseconds)"
    )
    print(
        _c("  2) hard   ", "2")
        + "what a hard canary audit runs: the nonce arrives only "
        "after your reply is committed"
    )
    try:
        choice = input(_c("select [1-2, Enter=1]: ", "1")).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return "light"
    return "hard" if choice == "2" else "light"


def _resolved_tier_label(final: Mapping[str, Any], requested: str) -> str:
    """The tier the worker actually produced, from its proof mode."""

    from verallm.mesh.ggml_proof import (
        VERATHOS_GGML_GEMM_PROOF_MODE,
        VERATHOS_GGML_LIGHT_PROOF_MODE,
    )

    mode = str(final.get("proof_mode", "") or "")
    if mode == VERATHOS_GGML_LIGHT_PROOF_MODE:
        return "light"
    if mode == VERATHOS_GGML_GEMM_PROOF_MODE:
        return "hard"
    return requested or "light"


def _proof_seconds(final: Mapping[str, Any]) -> float | None:
    """Seconds this reply spent proving, from driver-observed timings.

    Prefers the relay's MEASURED proving tail (generation end to the
    proof-bearing final). The subtraction estimate below mislabels
    per-token relay overhead as proving on slow-relay paths (a Mac reply
    showed 0.9s "proof" while its receipts took 25ms), so it is only the
    fallback for pre-measurement drivers. Sub-50ms is noise.

    total_s is the driver's whole-request wall time and engine_tps is
    llama.cpp's own decode rate, so total minus (ttft + decode) isolates
    the proving tail. Returns None when either input is missing rather
    than inventing a number.
    """

    measured = final.get("proof_wall_s")
    if isinstance(measured, (int, float)):
        return float(measured) if float(measured) >= 0.05 else None
    total = final.get("total_s")
    tps = final.get("engine_tps")
    usage = final.get("usage") or {}
    tokens = usage.get("completion_tokens") if isinstance(usage, Mapping) else None
    if not isinstance(total, (int, float)) or total <= 0:
        return None
    if not isinstance(tps, (int, float)) or tps <= 0:
        return None
    if not isinstance(tokens, (int, float)) or tokens <= 0:
        return None
    ttft = final.get("ttft_s")
    ttft_s = float(ttft) if isinstance(ttft, (int, float)) and ttft > 0 else 0.0
    remainder = float(total) - ttft_s - (float(tokens) / float(tps))
    if remainder < 0.05:
        return None
    return remainder


def cmd_mesh_setup(args: argparse.Namespace) -> None:
    """Guided human-first setup over the pool primitives."""

    from verallm.mesh.setup_wizard import run_mesh_setup

    raise SystemExit(run_mesh_setup(args))


def cmd_mesh_manage(args: argparse.Namespace) -> None:
    """The day-2 operator board over the same pool primitives."""

    from verallm.mesh.manage_board import run_mesh_manage

    raise SystemExit(run_mesh_manage(args))


def cmd_mesh_chat(args: argparse.Namespace) -> None:
    """Interactive verified chat with a serving mesh (streaming)."""

    token = _management_pool_token(args)
    endpoint = _dialable_manager_endpoint(token)

    def call(route: str, body: dict) -> dict:
        from verallm.mesh.worker import post_json

        return post_json(
            endpoint + route,
            {"management_secret": token.pool_secret, **body},
            timeout=15.0,
        )

    status = call("/v1/pool/status", {})
    mesh_key, mesh = _select_serving_mesh(
        status.get("meshes") or {},
        str(getattr(args, "mesh_key", "") or ""),
        interactive=_interactive_terminal(),
        workers=status.get("workers") or {},
    )

    proof_tier = str(getattr(args, "proof_tier", "") or "")
    if not proof_tier and _interactive_terminal():
        proof_tier = _select_proof_tier()
    # Piped/scripted sessions keep the historical no-override behavior.
    proof_tier = proof_tier or "auto"

    print()
    print(
        _c("verathos mesh chat", "1")
        + _c(
            f"  mesh={mesh_key}  model={mesh.get('model_id', '?')}"
            f"  workers={','.join(mesh.get('members') or [])}",
            "2",
        )
    )
    print(
        _c(
            "sleipnir mesh inference · every reply is gleipnir "
            "proof-verified end to end",
            "2",
        )
    )
    print(
        _c("proof tier: ", "2")
        + _c(_PROOF_TIER_LABELS.get(proof_tier, proof_tier), "1")
    )
    print(
        _c(
            "type your message; /tier switches the proof tier; "
            "/new starts over; /quit or Ctrl-D exits",
            "2",
        )
    )
    print()

    history: list[dict[str, str]] = []
    while True:
        try:
            prompt = input(_c("you › ", "1;36"))
        except (EOFError, KeyboardInterrupt):
            print()
            return
        prompt = prompt.strip()
        if not prompt:
            continue
        if prompt in ("/quit", "/exit", "quit", "exit"):
            return
        if prompt == "/new":
            history.clear()
            print(_c("(new conversation)", "2"))
            continue
        if prompt == "/tier":
            proof_tier = _select_proof_tier()
            print(
                _c("proof tier: ", "2")
                + _c(_PROOF_TIER_LABELS.get(proof_tier, proof_tier), "1")
            )
            continue
        history.append({"role": "user", "content": prompt})

        body = {
            "management_secret": token.pool_secret,
            "mesh_key": mesh_key,
            "messages": history,
            "max_tokens": int(args.max_tokens),
            "thinking": bool(args.thinking),
            "timeout": float(args.timeout),
            "proof_tier": proof_tier,
        }
        request = Request(
            endpoint + "/v1/pool/chat-stream",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        deltas: list[str] = []
        label_printed = False
        phase_line = ""
        final: dict[str, Any] | None = None
        try:
            with urlopen(request, timeout=float(args.timeout) + 30.0) as resp:
                for raw_line in resp:
                    line = raw_line.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    try:
                        event = json.loads(line[5:].strip())
                    except ValueError:
                        continue
                    kind = str(event.get("type", ""))
                    if kind == "phase":
                        phase_line = str(event.get("phase", ""))
                        if not label_printed and phase_line:
                            sys.stdout.write(
                                "\r\033[K" + _c(f"[{phase_line}]", "2")
                                if _color_enabled()
                                else f"[{phase_line}]\n"
                            )
                            sys.stdout.flush()
                        continue
                    if kind == "thinking":
                        continue
                    if kind == "delta":
                        if not label_printed:
                            if phase_line and _color_enabled():
                                sys.stdout.write("\r\033[K")
                            sys.stdout.write(_c("mesh › ", "1;35"))
                            label_printed = True
                        delta = str(event.get("delta", ""))
                        deltas.append(delta)
                        sys.stdout.write(delta)
                        sys.stdout.flush()
                        continue
                    if kind in ("done", "error"):
                        final = event
                        break
        except KeyboardInterrupt:
            print()
            print(_c("(interrupted; message dropped)", "2"))
            history.pop()
            continue
        except (HTTPError, URLError, OSError) as exc:
            print()
            detail = ""
            if isinstance(exc, HTTPError):
                # The response body carries the actual failure ("worker
                # offline", "proof endpoint failed: ..."); a bare status
                # line hides every root cause.
                try:
                    detail = exc.read(2048).decode("utf-8", "replace").strip()
                except Exception:
                    detail = ""
            print(
                _c(
                    f"✗ transport error: {exc}"
                    + (f" · {detail}" if detail else ""),
                    "31",
                )
            )
            history.pop()
            continue

        if label_printed:
            print()
        content = "".join(deltas)
        if final is None:
            print(_c("✗ stream ended without a result", "31"))
            history.pop()
            continue
        if not content and final.get("content"):
            content = str(final["content"])
            print(_c("mesh › ", "1;35") + content)
        error = str(final.get("error", "") or "")
        verified = bool(final.get("verified"))
        if final.get("type") == "error" or error or not verified:
            detail = error or str(final.get("error", "")) or "not verified"
            print(_c(f"✗ unverified: {detail[:200]}", "31"))
            history.pop()
            continue
        stages = final.get("proof_stages")
        declared = final.get("expected_stage_count") or stages
        receipts = final.get("receipts")
        ttft = final.get("ttft_s")
        tps = final.get("engine_tps")
        footer = (
            f"✓ verified ({stages}/{declared} stages, {receipts} receipts)"
        )
        # proof_mode is what the worker ACTUALLY produced, so an "auto"
        # session reports the tier it resolved to rather than the word auto.
        resolved_tier = _resolved_tier_label(final, proof_tier)
        footer += f" · {resolved_tier} tier"
        proof_s = _proof_seconds(final)
        if proof_s is not None:
            footer += f" · proof {proof_s:.1f}s"
        if isinstance(ttft, (int, float)) and ttft >= 0:
            footer += f" · ttft {ttft:.2f}s"
        if isinstance(tps, (int, float)):
            footer += f" · {tps:.0f} tok/s"
        print(_c(footer, "32") if _color_enabled() else footer)
        print()
        history.append({"role": "assistant", "content": content})


def cmd_pool_probe(args: argparse.Namespace) -> None:
    """Verified end-to-end self-test of a serving mesh through the pool.

    Sends the same probe-shaped chat the deploy gate uses (proof required,
    stream, temperature 0) and prints verified / receipts / TTFT / tok/s.
    With --gate it runs the full deploy probe gate (small samples plus the
    canary-shaped full-context probe) standalone, so an operator can test
    their own mesh at any time, not only during deploy. Exits nonzero when
    the probe does not verify or the gate fails.
    """

    from verallm.mesh.worker import post_json

    token = _management_pool_token(args)

    def call(route: str, body: dict) -> dict:
        # The HTTP timeout must outlive the in-band deadline the manager
        # enforces, or the client gives up right as the driver finishes.
        deadline = float(body.get("timeout", args.timeout) or args.timeout)
        return post_json(
            _dialable_manager_endpoint(token) + route,
            {"management_secret": token.pool_secret, **body},
            timeout=max(deadline, float(args.timeout)) + 15.0,
        )

    status = call("/v1/pool/status", {"timeout": 10.0})
    mesh_key, mesh = _select_serving_mesh(
        status.get("meshes") or {},
        str(getattr(args, "mesh_key", "") or ""),
        interactive=_interactive_terminal(),
    )

    if args.gate:
        from statistics import median

        from verallm.mesh.probe import ProbeGateConfig, run_probe_gate

        model_id = str(mesh.get("model_id", "") or "")
        model_entry = dict((status.get("models") or {}).get(model_id) or {})
        measured_ctx = int(model_entry.get("measured_ctx_budget", 0) or 0)
        registered_ctx = int(
            mesh.get("max_context_len")
            or model_entry.get("max_context_len")
            or measured_ctx
            or 32_768
        )
        config_kwargs: dict = {
            "samples": int(args.samples),
            "hard_samples": int(args.hard_samples),
            "full_context": not args.no_full_context,
            "max_context_len": registered_ctx,
            "measured_ctx_budget": measured_ctx,
        }
        if args.min_tok_s is not None:
            config_kwargs["min_tok_s"] = float(args.min_tok_s)
        gate = run_probe_gate(
            call=call,
            mesh_key=mesh_key,
            expected_snapshot_hash=str(
                mesh.get("verification_snapshot_hash", "") or ""
            ),
            expected_stage_count=len(mesh.get("members") or []),
            max_rtt_ms=0.0,
            config=ProbeGateConfig(**config_kwargs),
            log=lambda line: print(f"[gate] {line}", flush=True),
        )
        print(gate.render())
        ok_samples = [s for s in gate.samples if s.ok]
        tok_rates = [s.wall_tok_s for s in ok_samples if s.wall_tok_s > 0]
        ttfts = [s.ttft_s for s in ok_samples if s.ttft_s >= 0]
        summary = (
            f"light {len(ok_samples)}/{len(gate.samples)} verified, "
            f"hard {sum(1 for s in gate.hard_samples if s.ok and s.verified)}"
            f"/{len(gate.hard_samples)} verified, "
            f"median {median(tok_rates):.1f} tok/s, "
            f"ttft {median(ttfts):.2f}s, "
            if tok_rates and ttfts
            else ""
        )
        context_note = (
            f"measured context {measured_ctx}"
            if measured_ctx
            else "no measured context (launch on a subnet pool to measure)"
        )
        if gate.passed:
            print(
                _c("READY TO REGISTER", "1;32")
                + f": {summary}{context_note}. Next: `verathos mesh deploy "
                f"{model_id or '<model-id>'} ...` registers it on-chain."
            )
        else:
            failed = [
                check.name
                for check in gate.all_checks
                if check.kind == "hard" and not check.passed
            ]
            print(
                _c("NOT READY", "1;31")
                + f": failed {', '.join(failed)}. {summary}{context_note}."
            )
        raise SystemExit(0 if gate.passed else 1)

    from verallm.mesh.probe import probe_sample_from_chat_result

    result = call(
        "/v1/pool/chat",
        {
            "mesh_key": mesh_key,
            "probe": True,
            "stream": True,
            "thinking": False,
            "messages": [{"role": "user", "content": args.prompt}],
            "max_tokens": int(args.max_tokens),
            "timeout": float(args.timeout),
        },
    )
    sample = probe_sample_from_chat_result(result)
    _print_json(
        {
            "mesh_key": mesh_key,
            "verified": sample.verified,
            "receipts": sample.receipts,
            "proof_stages": sample.proof_stages,
            "ttft_s": sample.ttft_s,
            "total_s": sample.total_s,
            "engine_tps": sample.engine_tps,
            "wall_tok_s": round(sample.wall_tok_s, 2),
            "content": str(result.get("content", "") or "")[:400],
            "error": sample.error,
        }
    )
    if not sample.verified:
        raise SystemExit(1)


def cmd_pool_launch(args: argparse.Namespace) -> None:
    """Launch a mesh. With no flags on a terminal this is a guided flow:
    pick a model, review the pool's recommended placement (with its
    reasons), confirm, then watch the launch progress to serving."""

    call = _pool_client(args)
    interactive = _interactive_terminal()
    model_id = str(getattr(args, "model_id", "") or "")
    workers = [
        w.strip() for w in str(args.workers or "").split(",") if w.strip()
    ]
    driver = str(args.driver or "")

    if not model_id:
        if not interactive:
            raise SystemExit(
                "--model-id is required when not running on a terminal"
            )
        status = call("/v1/pool/status", {})
        models: dict[str, dict] = {}
        for worker in (status.get("workers") or {}).values():
            for entry in worker.get("catalog") or []:
                entry_id = str(entry.get("model_id", ""))
                if entry_id:
                    models.setdefault(entry_id, dict(entry))
        if not models:
            raise SystemExit(
                "no models registered in this pool. Register one first:\n"
                "  verathos mesh pool register-model --model-id ... "
                "--hf-repo ... --hf-files ..."
            )
        ordered = sorted(models)
        print(_c("registered models:", "1"))
        for index, entry_id in enumerate(ordered, start=1):
            entry = models[entry_id]
            size_gb = float(entry.get("model_bytes", 0) or 0) / 1e9
            print(
                f"  {_c(str(index), '1;36')}. {entry_id}"
                f"  layers={entry.get('layers', '?')}  size={size_gb:.1f}GB"
            )
        while True:
            raw = input(
                f"which model? [1-{len(ordered)}, default 1] "
            ).strip()
            if not raw:
                model_id = ordered[0]
                break
            if raw.isdigit() and 1 <= int(raw) <= len(ordered):
                model_id = ordered[int(raw) - 1]
                break
            print("enter a number from the list")

    if interactive and not workers:
        advice = call("/v1/pool/recommend", {"model_id": model_id})
        suggestions = list(advice.get("suggestions") or [])
        reasons = dict(advice.get("reasons") or {})
        if not suggestions:
            print(_c("no viable placement for this model:", "31"))
            for worker_id in sorted(reasons):
                print(f"  {worker_id}: {reasons[worker_id]}")
            raise SystemExit(1)
        print(_c(f"recommended placements for {model_id}:", "1"))
        for index, suggestion in enumerate(suggestions, start=1):
            row = (
                f"  {_c(str(index), '1;36')}."
                f" workers={','.join(suggestion.get('workers', []))}"
                f"  driver={suggestion.get('driver', '?')}"
                f"  link={suggestion.get('link_class', '?')}"
                f"  rtt={float(suggestion.get('max_rtt_ms', 0.0)):.0f}ms"
            )
            if suggestion.get("fetch"):
                row += (
                    f"  fetch~{float(suggestion.get('download_gb', 0.0)):.0f}GB"
                )
            print(row)
            if suggestion.get("warn"):
                print(_c(f"     warn: {suggestion['warn']}", "33"))
        for worker_id in sorted(reasons):
            print(_c(f"  {worker_id}: {reasons[worker_id]}", "2"))
        while True:
            raw = input(
                f"which placement? [1-{len(suggestions)}, default 1, "
                "or worker ids comma-separated] "
            ).strip()
            if not raw:
                chosen = suggestions[0]
                break
            if raw.isdigit():
                # A bare number is always a menu pick; out of range must
                # re-prompt, never fall through and become a "worker id".
                if 1 <= int(raw) <= len(suggestions):
                    chosen = suggestions[int(raw) - 1]
                    break
                print(_c(f"  pick a number from 1-{len(suggestions)}", "33"))
                continue
            custom = [w.strip() for w in raw.split(",") if w.strip()]
            if custom:
                chosen = {"workers": custom, "driver": custom[0]}
                break
        workers = list(chosen.get("workers", []))
        driver = str(chosen.get("driver", "") or (workers[0] if workers else ""))
        summary = (
            f"launch {model_id} on {','.join(workers)} (driver {driver})?"
        )
        confirmed = input(f"{_c(summary, '1')} [Y/n] ").strip().lower()
        if confirmed not in ("", "y", "yes"):
            raise SystemExit("cancelled")

    body: dict = {"model_id": model_id}
    if workers:
        body["workers"] = workers
    if driver:
        body["driver"] = driver
    result = call("/v1/pool/launch", body)
    if not interactive:
        mesh_key = str(result.get("mesh_key", ""))
        if not mesh_key or getattr(args, "no_wait", False):
            _print_json(result)
            return
        # Scripts deserve the same contract as the guided flow: when this
        # returns, the mesh routes. Returning at spawn time made every
        # scripted launch-then-chat race the serving transition ("mesh is
        # not serving yet" 400s until the members report in).
        # NOT args.timeout: that is the per-HTTP-call timeout (10s) shared
        # by every pool subcommand, and reusing it here capped the whole
        # launch wait at 30s - every launch that had to DOWNLOAD its model
        # then reported "timed out waiting for routing-ready" as a failure
        # while the launch was in fact proceeding.
        wait_timeout = max(30.0, float(getattr(args, "wait_timeout", 0) or 900))
        deadline = time.monotonic() + wait_timeout
        while True:
            time.sleep(5.0)
            try:
                status = call("/v1/pool/status", {})
            except (RuntimeError, OSError):
                status = {}
            mesh = (status.get("meshes") or {}).get(mesh_key) or {}
            state = str(mesh.get("status", "?"))
            if state == "fetching":
                # A download is bounded by the manager's disk-fit precheck
                # and reports progress via worker status; it must not burn
                # the routing deadline (17GB on a home line is minutes).
                deadline = max(deadline, time.monotonic() + wait_timeout)
            if state == "serving" and bool(mesh.get("routing_ready")):
                result["status"] = "serving"
                result["routing_ready"] = True
                break
            if state == "error":
                result["status"] = "error"
                result["error"] = str(mesh.get("error", "unknown"))
                break
            if time.monotonic() >= deadline:
                result["status"] = state
                result["routing_ready"] = bool(mesh.get("routing_ready"))
                result["error"] = "timed out waiting for routing-ready"
                break
        _print_json(result)
        if result.get("status") != "serving":
            raise SystemExit(1)
        return

    mesh_key = str(result.get("mesh_key", ""))
    if not mesh_key:
        _print_json(result)
        raise SystemExit(1)
    print(f"mesh {_c(mesh_key, '1')} launching...")
    spinner = "|/-\\"
    started = time.monotonic()
    tick = 0
    while True:
        time.sleep(5.0)
        tick += 1
        try:
            status = call("/v1/pool/status", {})
        except (RuntimeError, OSError):
            continue
        mesh = (status.get("meshes") or {}).get(mesh_key) or {}
        state = str(mesh.get("status", "?"))
        ready = bool(mesh.get("routing_ready"))
        elapsed = int(time.monotonic() - started)
        line = (
            f"{spinner[tick % 4]} {state}"
            + ("  routing-ready" if ready else "")
            + f"  ({elapsed}s)"
        )
        if _color_enabled():
            sys.stdout.write("\r\033[K" + line)
            sys.stdout.flush()
        else:
            print(line)
        if state == "serving" and ready:
            if _color_enabled():
                print()
            print(_c(f"✓ mesh {mesh_key} is serving", "32"))
            print("try it:  " + _c("verathos mesh chat", "1"))
            return
        if state == "error":
            if _color_enabled():
                print()
            print(_c(f"✗ launch failed: {mesh.get('error', 'unknown')}", "31"))
            print(
                "inspect with:  verathos mesh fleet  and the driver log "
                "under ~/.verathos/poolwork-*/"
            )
            raise SystemExit(1)


def cmd_pool_stop(args: argparse.Namespace) -> None:
    _print_json(_pool_client(args)("/v1/pool/stop", {"mesh_key": args.mesh_key}))


def cmd_pool_remove_worker(args: argparse.Namespace) -> None:
    _print_json(_pool_client(args)("/v1/pool/remove-worker", {"worker_id": args.worker_id}))


def cmd_pool_set_epoch(args: argparse.Namespace) -> None:
    _print_json(_pool_client(args)("/v1/pool/set-epoch", {"epoch": args.epoch}))


def cmd_pool_register_model(args: argparse.Namespace) -> None:
    body: dict[str, object] = {
        "model_id": args.model_id,
        "hf_repo": args.hf_repo,
        "hf_files": [f.strip() for f in args.hf_files.split(",") if f.strip()],
        "layers": args.layers,
        "model_bytes": args.model_bytes,
    }
    if args.proof_tolerance_abs is not None:
        body["proof_tolerance_abs"] = args.proof_tolerance_abs
    if args.proof_tolerance_rel is not None:
        body["proof_tolerance_rel"] = args.proof_tolerance_rel
    if getattr(args, "llama_ubatch", None) is not None:
        body["llama_ubatch"] = args.llama_ubatch
    if getattr(args, "llama_batch", None) is not None:
        body["llama_batch"] = args.llama_batch
    if args.model_index is not None:
        body["model_index"] = args.model_index
    if args.max_context_len is not None:
        body["max_context_len"] = args.max_context_len
    for name in (
        "model_package_hash",
        "model_tensor_manifest_root",
        "tokenizer_hash",
        "quantization_scheme",
    ):
        value = getattr(args, name, "")
        if value:
            body[name] = value
    manifest_urls = [
        part.strip()
        for part in str(getattr(args, "manifest_urls", "") or "").split(",")
        if part.strip()
    ]
    if manifest_urls:
        body["manifest_urls"] = manifest_urls
    _print_json(_pool_client(args)("/v1/pool/register-model", body))


def cmd_handshake(args: argparse.Namespace) -> None:
    spec = _load_mesh_spec(args.mesh)
    internal_auth_secret = (
        _mesh_internal_auth_secret(args.mesh) if _state_exists(args.mesh) else ""
    )
    response = post_json(
        args.endpoint.rstrip("/") + "/v1/stage/handshake",
        {
            "mesh_id": spec.mesh_id,
            "mesh_spec_hash": spec.spec_hash_hex(),
            "stage_assignment_hash": spec.stage_assignment_hash_hex(),
        },
        timeout=args.timeout,
        internal_auth_secret=internal_auth_secret,
    )
    _print_json(response)


def cmd_infer(args: argparse.Namespace) -> None:
    messages = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    messages.append({"role": "user", "content": args.prompt})
    payload = {
        "model": args.model,
        "messages": messages,
        "stream": bool(args.stream),
    }
    if int(getattr(args, "max_tokens", 0) or 0) > 0:
        payload["max_tokens"] = int(args.max_tokens)
    verathos: dict[str, object] = {}
    proof_tier = str(getattr(args, "proof_tier", "") or "")
    if proof_tier in ("light", "hard"):
        # Same wire field the manager chat lane uses: upgrade-only at the
        # coordinator ("hard" forces the hard relation on this reply;
        # "light" never downgrades a lane that must assert hard).
        verathos["proof_tier"] = proof_tier
    if args.validator_nonce:
        nonce = os.urandom(32).hex() if args.validator_nonce == "auto" else args.validator_nonce
        verathos["validator_nonce"] = nonce
    if args.proof_sample_bps is not None:
        verathos["proof_sample_bps"] = normalize_proof_sample_bps(args.proof_sample_bps)
    if args.decode_audit_bps is not None:
        verathos["decode_audit_bps"] = normalize_proof_sample_bps(args.decode_audit_bps)
    if args.decode_audit_top_k is not None:
        verathos["decode_audit_top_k"] = max(1, int(args.decode_audit_top_k))
    if verathos:
        payload["verathos"] = verathos
    if args.stream:
        _stream_chat_completions(
            args.endpoint,
            payload,
            content_only=args.content,
            timeout=args.timeout,
        )
        return
    response = post_json(
        args.endpoint.rstrip("/") + "/v1/chat/completions",
        payload,
        timeout=args.timeout,
    )
    if args.content:
        choices = response.get("choices", [])
        if choices and isinstance(choices[0], dict):
            message = choices[0].get("message", {})
            if isinstance(message, dict):
                print(message.get("content", ""))
                return
        print("")
        return
    _print_json(response)


def _mesh_artifact_for_verification(
    artifact: dict,
    *,
    response: dict | None = None,
) -> tuple[dict, dict | None]:
    if "receipt" in artifact and "response" in artifact:
        return artifact, response
    if isinstance(artifact.get("verathos_mesh"), dict):
        mesh_artifact = artifact["verathos_mesh"]
        if not isinstance(mesh_artifact, dict):
            raise SystemExit("verathos_mesh must be a JSON object")
        if isinstance(artifact.get("response"), dict):
            return mesh_artifact, artifact["response"]
        openai_response = dict(artifact)
        openai_response.pop("verathos_mesh", None)
        return mesh_artifact, response or openai_response
    if "receipt" in artifact:
        if response is None:
            raise SystemExit(
                "artifact metadata without a response requires --response"
            )
        return artifact, response
    raise SystemExit(
        "artifact must be a /v1/mesh/inference payload, a chat completion with "
        "verathos_mesh, or a verathos_mesh metadata object plus --response"
    )


def cmd_verify_artifact(args: argparse.Namespace) -> None:
    request = _load_json_file(args.request)
    artifact_data = _load_json_file(args.artifact)
    response = _load_json_file(args.response) if args.response else None
    artifact, openai_response = _mesh_artifact_for_verification(
        artifact_data,
        response=response,
    )
    spec = _load_mesh_spec(args.mesh) if args.mesh else None
    required_stage_indexes = (
        [int(item) for item in args.proof_stage_index]
        if args.proof_stage_index is not None
        else None
    )
    verify_mesh_inference_artifact(
        artifact,
        request,
        openai_response=openai_response,
        spec=spec,
        member_index=args.member_index,
        require_configured_proof=bool(args.require_configured_proof),
        require_cryptographic_proof=bool(args.require_cryptographic_proof),
        required_proof_stage_indexes=required_stage_indexes,
        require_spec_proof_stage_coverage=not bool(args.no_spec_proof_stage_coverage),
        deferred_randomness=args.deferred_randomness,
        require_deferred_proof_if_sampled=bool(
            args.require_deferred_proof_if_sampled
        ),
    )
    deferred_bundle_verified = False
    if args.deferred_audit_bundle:
        bundle = _load_json_file(args.deferred_audit_bundle)
        verify_deferred_mesh_audit_bundle(
            bundle,
            artifact,
            request,
            openai_response=openai_response,
            spec=spec,
            member_index=args.member_index,
            required_proof_stage_indexes=required_stage_indexes,
            require_spec_proof_stage_coverage=not bool(
                args.no_spec_proof_stage_coverage
            ),
        )
        deferred_bundle_verified = True
    receipt = artifact["receipt"]
    deferred_decision = deferred_audit_decision(
        receipt,
        randomness=args.deferred_randomness,
    )
    _print_json(
        {
            "ok": True,
            "mesh_id": receipt.get("mesh_id", ""),
            "mesh_spec_hash": receipt.get("mesh_spec_hash", ""),
            "request_id": receipt.get("request_id", ""),
            "receipt_hash": receipt.get("receipt_hash", ""),
            "proof_required": bool(receipt.get("proof_required", False)),
            "proof_sampled": bool(receipt.get("proof_sampled", False)),
            "proof_receipt_count": int(receipt.get("proof_receipt_count", 0)),
            "proof_mode": receipt.get("proof_mode", ""),
            "proof_deferred": bool(receipt.get("proof_deferred", False)),
            "proof_deferred_obligation": bool(
                receipt.get("proof_deferred_obligation", False)
            ),
            "proof_deferred_required": bool(
                receipt.get("proof_deferred_required", False)
            ),
            "proof_deferred_sampled": bool(
                deferred_decision.get("sampled", False)
            ),
            "proof_deferred_sample_value": int(
                deferred_decision.get("sample_value", -1)
            ),
            "deferred_audit_bundle_verified": deferred_bundle_verified,
            "decode_audit_required": bool(receipt.get("decode_audit_required", False)),
            "decode_audit_verified": bool(receipt.get("decode_audit_verified", False)),
        }
    )


def cmd_resolve_deferred_audit(args: argparse.Namespace) -> None:
    request = _load_json_file(args.request)
    artifact_data = _load_json_file(args.artifact)
    response = _load_json_file(args.response) if args.response else None
    artifact, openai_response = _mesh_artifact_for_verification(
        artifact_data,
        response=response,
    )
    payload = {
        "artifact": artifact,
        "openai_request": request,
        "deferred_randomness": args.deferred_randomness,
    }
    if openai_response is not None:
        payload["openai_response"] = openai_response
    bundle = post_json(
        args.endpoint.rstrip("/") + "/v1/mesh/proof/deferred-audit",
        payload,
        timeout=args.timeout,
    )
    spec = _load_mesh_spec(args.mesh) if args.mesh else None
    if spec is not None:
        verify_deferred_mesh_audit_bundle(
            bundle,
            artifact,
            request,
            openai_response=openai_response,
            spec=spec,
            member_index=args.member_index,
        )
    _write_or_print_json(bundle, args.output)


def cmd_rpc_plan(args: argparse.Namespace) -> None:
    spec = _load_mesh_spec(args.mesh)
    plan = rpc_plan_from_mesh(spec)
    payload = plan.to_dict()
    if args.model or args.hf:
        if args.model and args.hf:
            raise SystemExit("--model and --hf are mutually exclusive")
        validate_all_rpc_runtime_binding(
            spec,
            plan,
            device=args.device,
            n_gpu_layers=args.n_gpu_layers,
        )
        tensor_split = tensor_split_for_rpc_plan(plan, args.tensor_split)
        cmd = build_llama_server_command(
            binary=args.llama_server_binary,
            model=args.model,
            hf_model=args.hf,
            host=args.host,
            port=args.port,
            rpc_endpoints=plan.rpc_endpoints,
            device=args.device,
            n_gpu_layers=args.n_gpu_layers,
            ctx_size=_mesh_bound_ctx_size(
                spec,
                args.ctx_size,
                flag_name="--ctx-size",
            ),
            tensor_split=tensor_split,
            alias=args.alias or spec.model_id,
            extra_args=args.extra_arg,
        )
        payload["llama_server_command"] = cmd
        payload["llama_server_command_text"] = command_preview(cmd)
    _print_json(payload)


def cmd_rpc_worker(args: argparse.Namespace) -> None:
    cmd = build_rpc_worker_command(
        binary=args.binary,
        host=args.host,
        port=args.port,
        device=args.device,
        cache=args.cache,
        extra_args=args.extra_arg,
    )
    endpoint = normalize_rpc_endpoint(f"{args.advertise_host or args.host}:{args.port}")
    payload = {
        "rpc_endpoint": endpoint,
        "rpc_worker_command": cmd,
        "rpc_worker_command_text": command_preview(cmd),
    }
    if args.dry_run:
        _print_json(payload)
        return
    cmd[0] = resolve_binary(cmd[0])
    print(payload["rpc_worker_command_text"], flush=True)
    raise SystemExit(run_command(cmd))


def cmd_llama_server(args: argparse.Namespace) -> None:
    if bool(args.model) == bool(args.hf):
        raise SystemExit("exactly one of --model or --hf is required")
    spec = _load_mesh_spec(args.mesh)
    plan = rpc_plan_from_mesh(spec)
    validate_all_rpc_runtime_binding(
        spec,
        plan,
        device=args.device,
        n_gpu_layers=args.n_gpu_layers,
    )
    tensor_split = tensor_split_for_rpc_plan(plan, args.tensor_split)
    cmd = build_llama_server_command(
        binary=args.binary,
        model=args.model,
        hf_model=args.hf,
        host=args.host,
        port=args.port,
        rpc_endpoints=plan.rpc_endpoints,
        device=args.device,
        n_gpu_layers=args.n_gpu_layers,
        ctx_size=_mesh_bound_ctx_size(
            spec,
            args.ctx_size,
            flag_name="--ctx-size",
        ),
        tensor_split=tensor_split,
        alias=args.alias or spec.model_id,
        extra_args=args.extra_arg,
    )
    payload = {
        "mesh_id": spec.mesh_id,
        "rpc_plan": plan.to_dict(),
        "backend_url": f"http://{args.host}:{args.port}",
        "llama_server_command": cmd,
        "llama_server_command_text": command_preview(cmd),
    }
    if args.dry_run:
        _print_json(payload)
        return
    cmd[0] = resolve_binary(cmd[0])
    print(payload["llama_server_command_text"], flush=True)
    raise SystemExit(run_command(cmd))


def cmd_proof_adapter(args: argparse.Namespace) -> None:
    from verallm.mesh.ggml_proof import serve_ggml_proof_adapter

    serve_ggml_proof_adapter(
        trace_dir=args.trace_dir,
        host=args.host,
        port=args.port,
        tolerance_abs=args.tolerance_abs,
        tolerance_rel=args.tolerance_rel,
        proof_block_size=args.proof_block_size,
        spot_checks=args.spot_checks,
        gguf_manifest_path=args.gguf_manifest,
        warmup=not args.no_warmup,
    )


def cmd_build_runtime(args: argparse.Namespace) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "runtime" / "llama_cpp" / "build_verathos_rpc_server.sh"
    if not script.exists():
        raise SystemExit(f"runtime build script not found: {script}")

    cmd = [str(script), args.llama_dir]
    if args.build_dir:
        cmd.append(args.build_dir)

    env = os.environ.copy()
    build_flags = {
        "VERATHOS_BUILD_CUDA": args.cuda,
        "VERATHOS_BUILD_METAL": args.metal,
        "VERATHOS_BUILD_VULKAN": args.vulkan,
        "VERATHOS_BUILD_LLAMA_SERVER": args.llama_server,
    }
    for name, enabled in build_flags.items():
        if enabled:
            env[name] = "1"
    if args.jobs:
        env["VERATHOS_BUILD_JOBS"] = str(args.jobs)
    if args.cuda_architectures:
        env["VERATHOS_CUDA_ARCHITECTURES"] = str(args.cuda_architectures)

    payload = {
        "build_script": str(script),
        "command": cmd,
        "command_text": command_preview(cmd),
        "enabled_backends": [
            name
            for name, enabled in (
                ("cuda", args.cuda),
                ("metal", args.metal),
                ("vulkan", args.vulkan),
            )
            if enabled
        ],
        "build_llama_server": bool(args.llama_server),
        "env": {
            name: env[name]
            for name in (
                "VERATHOS_BUILD_CUDA",
                "VERATHOS_BUILD_METAL",
                "VERATHOS_BUILD_VULKAN",
                "VERATHOS_BUILD_LLAMA_SERVER",
                "VERATHOS_BUILD_JOBS",
                "VERATHOS_CUDA_ARCHITECTURES",
            )
            if name in env
        },
    }
    if args.dry_run:
        _print_json(payload)
        return

    print(payload["command_text"], flush=True)
    raise SystemExit(subprocess.run(cmd, env=env, check=False).returncode)


def cmd_build_proof_cache(args: argparse.Namespace) -> None:
    from verallm.mesh.gguf_manifest import (
        build_proof_weight_cache,
        load_gguf_tensor_manifest,
        proof_weight_cache_dir,
    )

    if args.cache_dir:
        os.environ["VERALLM_PROOF_WEIGHT_CACHE_DIR"] = args.cache_dir
    manifest = load_gguf_tensor_manifest(args.manifest)
    root = proof_weight_cache_dir()
    if root is None:
        raise SystemExit("proof-weight cache is disabled or the cache dir is not writable")
    started = time.time()
    stats = build_proof_weight_cache(
        manifest,
        progress=lambda msg: print(msg, flush=True),
    )
    _print_json(
        {
            "cache_dir": str(root),
            "elapsed_s": round(time.time() - started, 1),
            **stats,
        }
    )


def cmd_gguf_manifest(args: argparse.Namespace) -> None:
    from verallm.mesh.gguf_manifest import (
        build_gguf_tensor_manifest,
        save_gguf_tensor_manifest,
    )

    manifest = build_gguf_tensor_manifest(args.model)
    payload = {
        "model": args.model,
        "model_file_sha256": manifest["model_file_sha256"],
        "model_file_count": len(manifest.get("model_files", []) or []),
        "tensor_count": manifest["tensor_count"],
        "tensor_manifest_root": manifest["tensor_manifest_root"],
    }
    if args.output:
        path = save_gguf_tensor_manifest(manifest, args.output)
        payload["output"] = str(path)
    elif args.full:
        payload["manifest"] = manifest
    _print_json(payload)


def cmd_smoke(args: argparse.Namespace) -> None:
    from verallm.mesh.gguf_manifest import load_gguf_tensor_manifest
    from verallm.mesh.smoke import (
        SmokeConfig,
        default_package_hash,
        default_smoke_root,
        run_local_smoke,
    )

    if bool(args.llama_model) == bool(args.llama_hf):
        raise SystemExit("exactly one of --llama-model or --llama-hf is required")
    args.proof_sample_bps = normalize_proof_sample_bps(args.proof_sample_bps)
    args.proof_ops_per_request = int(args.proof_ops_per_request)
    args.proof_trace_candidates_per_request = int(
        args.proof_trace_candidates_per_request
    )
    args.proof_tolerance_abs = float(args.proof_tolerance_abs)
    args.proof_tolerance_rel = float(args.proof_tolerance_rel)
    args.decode_audit_bps = normalize_proof_sample_bps(args.decode_audit_bps)
    args.decode_audit_top_k = max(1, int(args.decode_audit_top_k))
    if args.proof_ops_per_request < 1:
        raise SystemExit("--proof-ops-per-request must be >= 1")
    if args.proof_trace_candidates_per_request < 1:
        raise SystemExit("--proof-trace-candidates-per-request must be >= 1")
    if args.proof_tolerance_abs <= 0:
        raise SystemExit("--proof-tolerance-abs must be > 0")
    if args.proof_tolerance_rel <= 0:
        raise SystemExit("--proof-tolerance-rel must be > 0")
    args.proof_trace_candidates_per_request = max(
        args.proof_trace_candidates_per_request,
        args.proof_ops_per_request,
    )
    package_ref = args.llama_hf or args.llama_model
    model_tensor_manifest_root = ""
    if args.proof_gguf_manifest:
        manifest = load_gguf_tensor_manifest(args.proof_gguf_manifest)
        model_tensor_manifest_root = str(manifest.get("tensor_manifest_root", ""))
    root = Path(args.output_dir) if args.output_dir else default_smoke_root()
    result = run_local_smoke(
        SmokeConfig(
            repo_root=Path(__file__).resolve().parents[2],
            output_root=root,
            model_id=args.model_id,
            package_hash=args.package_hash or default_package_hash(package_ref),
            model_tensor_manifest_root=model_tensor_manifest_root,
            layers=args.layers,
            llama_server_binary=args.llama_server_binary,
            rpc_worker_binary=args.rpc_worker_binary,
            llama_model=args.llama_model,
            llama_hf=args.llama_hf,
            llama_device=args.llama_device,
            rpc_device=args.rpc_device,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            samples=args.samples,
            direct_baseline=not args.no_baseline,
            require_proof=not args.no_require_proof,
            proof_sample_bps=args.proof_sample_bps,
            proof_ops_per_request=args.proof_ops_per_request,
            proof_trace_candidates_per_request=args.proof_trace_candidates_per_request,
            decode_audit_bps=args.decode_audit_bps,
            decode_audit_top_k=args.decode_audit_top_k,
            proof_gguf_manifest_path=args.proof_gguf_manifest,
            hf_home=args.hf_home,
            timeout=args.timeout,
        )
    )
    if args.result_json:
        Path(args.result_json).write_text(
            json.dumps(result, sort_keys=True, indent=2),
            encoding="utf-8",
        )
    _print_json(_without_sample_lists(result) if args.summary_only else result)


def cmd_join(args: argparse.Namespace) -> None:
    out_dir, spec = join_mesh(
        token=args.token,
        endpoint=args.endpoint,
        root=Path(args.output_dir) if args.output_dir else _default_mesh_dir(),
        uid=args.uid,
        hotkey=args.hotkey or None,
        backend=args.backend,
        package_hash=args.package_hash,
        gpu_name=args.gpu_name,
        vram_gb=args.vram_gb,
        rpc_endpoint=args.rpc_endpoint,
        proof_endpoint=args.proof_endpoint,
        timeout=args.timeout,
    )
    print(f"Joined mesh: {out_dir}")
    print(f"mesh_id: {spec.mesh_id}")
    print(f"mesh_spec_hash: {spec.spec_hash_hex()}")
    print(f"stage_assignment_hash: {spec.stage_assignment_hash_hex()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="verathos mesh",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "verathos mesh - verified GGUF inference on your own GPUs\n"
            "\n"
            "everyday commands:\n"
            "  setup     first-time guided setup: pool, workers, first mesh\n"
            "  manage    the operator board: launch/stop meshes, placement,"
            " machines\n"
            "  fleet     workers, meshes, and placement advice\n"
            "  status    this machine: units, pools, processes\n"
            "  chat      verified chat with a serving mesh\n"
            "  deploy    go live on the subnet (probe gate + on-chain"
            " registration)\n"
            "\n"
            "building blocks (scripts/agents):\n"
            "  pool ...  create/serve/worker/launch/stop/probe/join-token/"
            "recommend/...\n"
            "  add-model, logs, start, stop, renew, registration-status\n"
            "\n"
            "internals (rarely typed by hand):\n"
            "  create, snapshot-create, serve, join, assign, probe,\n"
            "  plan-units, register-units, handshake, infer, verify-artifact,\n"
            "  resolve-deferred-audit, rpc-plan, rpc-worker, llama-server,\n"
            "  proof-adapter, build-runtime, gguf-manifest, build-proof-cache,"
            " smoke\n"
            "\n"
            "`verathos mesh <command> -h` shows each command's flags."
        ),
    )
    sub = parser.add_subparsers(
        dest="command", required=True, metavar="<command>"
    )

    p_create = sub.add_parser("create", help="Create a private mesh spec for one UID")
    p_create.add_argument("--uid", type=int, required=True, help="Coordinator UID")
    p_create.add_argument("--hotkey", required=True, help="Coordinator hotkey SS58")
    p_create.add_argument("--endpoint", required=True, help="Coordinator mesh endpoint")
    p_create.add_argument("--model-id", required=True, help="Verathos model id")
    p_create.add_argument("--package-ref", default="", help="GGUF package ref or local path")
    p_create.add_argument("--package-hash", required=True, help="SHA-256 of package manifest")
    p_create.add_argument(
        "--model-tensor-manifest-root",
        default="",
        help="Merkle root from `verathos mesh gguf-manifest` for tensor-byte binding",
    )
    p_create.add_argument("--tokenizer-hash", default="", help="Tokenizer hash, if known")
    p_create.add_argument("--quantization-scheme", default="unknown")
    p_create.add_argument("--activation-dtype", default="f16", choices=["f16", "bf16", "f32", "q8"])
    p_create.add_argument(
        "--max-context-len",
        type=int,
        default=0,
        help="Optional runtime context limit; validator snapshots require a positive uint32",
    )
    p_create.add_argument("--layers", type=int, required=True, help="Transformer layer count")
    p_create.add_argument("--output-dir", default="", help="Root directory for mesh state")
    p_create.set_defaults(func=cmd_create)

    p_snapshot = sub.add_parser(
        "snapshot-create",
        help="Create a signed endpoint-free validator verification snapshot",
    )
    p_snapshot.add_argument(
        "mesh",
        help="Persisted coordinator mesh state directory or mesh-state.json",
    )
    p_snapshot.add_argument("--wallet-name", required=True)
    p_snapshot.add_argument("--wallet-hotkey", required=True)
    p_snapshot.add_argument("--chain-id", type=int, required=True)
    p_snapshot.add_argument("--netuid", type=int, required=True)
    p_snapshot.add_argument(
        "--coordinator-evm-address",
        required=True,
        help="EVM identity expected from the coordinator hotkey",
    )
    p_snapshot.add_argument("--model-index", type=int, required=True)
    p_snapshot.add_argument("--generation", type=int, required=True)
    p_snapshot.add_argument("--epoch", type=int, required=True)
    p_snapshot.add_argument(
        "--issued-at-unix",
        type=int,
        default=None,
        help="Snapshot issue time; defaults to the current Unix time",
    )
    expiry = p_snapshot.add_mutually_exclusive_group(required=True)
    expiry.add_argument("--expires-at-unix", type=int, default=None)
    expiry.add_argument("--ttl-seconds", type=int, default=None)
    p_snapshot.add_argument("--policy-profile", required=True)
    p_snapshot.add_argument(
        "--trace-manifest-format",
        required=True,
        choices=sorted(SUPPORTED_TRACE_MANIFEST_FORMATS),
    )
    p_snapshot.add_argument("--base-proof-sample-bps", type=int, required=True)
    p_snapshot.add_argument("--organic-decode-sample-bps", type=int, required=True)
    p_snapshot.add_argument("--canary-decode-sample-bps", type=int, required=True)
    p_snapshot.add_argument("--proof-ops-per-request", type=int, required=True)
    p_snapshot.add_argument(
        "--proof-trace-candidates-per-request",
        type=int,
        default=MIN_SECURE_TRACE_CANDIDATES_PER_REQUEST,
    )
    p_snapshot.add_argument("--deferred-proof-enabled", action="store_true")
    p_snapshot.add_argument(
        "--output",
        default="",
        help=(
            "Output JSON; defaults to verification_snapshot.json in the "
            "coordinator state directory"
        ),
    )
    p_snapshot.set_defaults(func=cmd_snapshot_create)

    p_status = sub.add_parser(
        "status", help="Inspect a mesh spec, or this machine's worker units"
    )
    p_status.add_argument(
        "mesh",
        nargs="?",
        default="",
        help="mesh.json or mesh directory; omit for this machine's worker status",
    )
    p_status.add_argument("--probe", action="store_true", help="Probe member endpoints")
    p_status.add_argument("--timeout", type=float, default=3.0)
    p_status.add_argument("--json", action="store_true", help="Worker view as JSON")
    p_status.set_defaults(func=cmd_status)

    p_join = sub.add_parser("join", help="Join a private mesh using a join token")
    p_join.add_argument("token")
    p_join.add_argument("--endpoint", required=True, help="This worker's reachable mesh endpoint")
    p_join.add_argument("--uid", type=int, default=None, help="Worker UID; defaults to coordinator UID")
    p_join.add_argument("--hotkey", default="", help="Worker hotkey; defaults to coordinator hotkey")
    p_join.add_argument(
        "--backend",
        default="gguf_stage_worker",
        choices=["gguf", "gguf_stage_worker", "llama_cpp_rpc"],
    )
    p_join.add_argument("--package-hash", default="", help="Cached model package hash, if present")
    p_join.add_argument("--gpu-name", default="")
    p_join.add_argument("--vram-gb", type=int, default=0)
    p_join.add_argument(
        "--rpc-endpoint",
        default="",
        help="llama.cpp RPC endpoint advertised by this worker, in host:port form",
    )
    p_join.add_argument(
        "--proof-endpoint",
        default="",
        help="Verathos mesh proof endpoint; defaults to --endpoint when --rpc-endpoint is set",
    )
    p_join.add_argument("--output-dir", default="", help="Root directory for worker mesh state")
    p_join.add_argument("--timeout", type=float, default=5.0)
    p_join.set_defaults(func=cmd_join)

    p_assign = sub.add_parser("assign", help="Set ordered stage members for a mesh")
    p_assign.add_argument("mesh", help="mesh.json or mesh directory")
    p_assign.add_argument(
        "--member",
        action="append",
        required=True,
        help="UID,HOTKEY,ENDPOINT,START:END[,BACKEND[,PAYOUT_BPS[,ROLE[,PROOF_ENDPOINT]]]]",
    )
    p_assign.add_argument("--mode", default="", choices=["", "private", "declared", "open"])
    p_assign.add_argument("--activation-dtype", default="", choices=["", "f16", "bf16", "f32", "q8"])
    p_assign.add_argument("--output", default="", help="Output mesh.json or directory")
    p_assign.set_defaults(func=cmd_assign)

    p_serve = sub.add_parser("serve", help="Run the lightweight mesh control server")
    p_serve.add_argument("--host", default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=DEFAULT_WORKER_PORT)
    p_serve.add_argument("--mesh", default="", help="Optional mesh.json or mesh directory")
    p_serve.add_argument(
        "--verification-snapshot",
        default="",
        help=(
            "Signed verification snapshot JSON; defaults to an existing "
            "verification_snapshot.json beside persisted mesh state"
        ),
    )
    p_serve.add_argument(
        "--require-verification-snapshot",
        action="store_true",
        help=(
            "Install the persisted snapshot loader before the final snapshot "
            "exists; validator inference remains fail-closed until publication"
        ),
    )
    p_serve.add_argument("--uid", type=int, default=None)
    p_serve.add_argument("--hotkey", default="")
    p_serve.add_argument(
        "--wallet-name",
        default="",
        help="Bittensor wallet name; with --wallet-hotkey, loads the hotkey to "
        "SIGN receipts (miner mode). --hotkey is then set from its SS58.",
    )
    p_serve.add_argument("--wallet-hotkey", default="", help="Hotkey name within --wallet-name")
    p_serve.add_argument(
        "--stage-proof-key-file",
        default="",
        help=(
            "Owner-only file containing the dedicated worker Sr25519 seed; "
            "never reuse the coordinator/miner hotkey"
        ),
    )
    p_serve.add_argument(
        "--evm-address",
        default="",
        help="Serving EVM address; wallet-backed serving derives and verifies this automatically",
    )
    p_serve.add_argument(
        "--evm-private-key",
        default="",
        help=argparse.SUPPRESS,
    )
    p_serve.add_argument(
        "--coordinator-sign-file",
        default="",
        help=(
            "Owner-only delegation file written by the pool worker: a "
            "driver without the coordinator wallet signs receipts and "
            "identity challenges through its pool manager"
        ),
    )
    p_serve.add_argument(
        "--capacity-drain-file",
        default="",
        help=(
            "Capacity-audit drain state file written by the pool worker "
            "daemon; while it marks an active audit window, admission "
            "returns the ordinary busy 503"
        ),
    )
    p_serve.add_argument(
        "--capacity-roster-file",
        default="",
        help=(
            "Signed capacity roster file written by the pool worker "
            "daemon; served on GET /capacity/roster for validators"
        ),
    )
    p_serve.add_argument(
        "--server-role",
        choices=["auto", "coordinator", "worker"],
        default="auto",
        help="HTTP security role; auto follows the persisted mesh state",
    )
    p_serve.add_argument(
        "--validator-auth",
        action="store_true",
        help="Require an allowlisted validator signature on coordinator inference routes",
    )
    p_serve.add_argument(
        "--validator-allowlist-path",
        default=os.environ.get(
            "VERATHOS_VALIDATORS_PATH",
            str(Path.home() / ".verathos" / "validators.json"),
        ),
        help="Validator allowlist JSON written by the miner metagraph refresh",
    )
    p_serve.add_argument(
        "--validator-allowlist-max-age-seconds",
        type=_positive_seconds,
        default=DEFAULT_VALIDATOR_ALLOWLIST_MAX_AGE_SECONDS,
        help=(
            "Reject validator requests when the allowlist refresh is older "
            "than this many seconds (default: 900)"
        ),
    )
    p_serve.add_argument(
        "--require-validator-nonce",
        action="store_true",
        help="Reject validator inference requests without a fresh proof nonce",
    )
    p_serve.add_argument(
        "--allow-loopback-dev-validator-routes",
        action="store_true",
        help=(
            "Explicit development mode: allow unsigned coordinator inference "
            "only from the local loopback interface"
        ),
    )
    p_serve.add_argument("--endpoint", default="")
    p_serve.add_argument(
        "--backend",
        action="append",
        default=[],
        choices=["vllm", "gguf", "gguf_stage", "gguf_stage_worker", "llama_cpp_rpc"],
        help="Supported backend; may be repeated",
    )
    p_serve.add_argument("--proof-mode", action="append", default=[])
    p_serve.add_argument("--package-hash", action="append", default=[])
    p_serve.add_argument("--gpu-name", default="")
    p_serve.add_argument("--vram-gb", type=int, default=0)
    p_serve.add_argument(
        "--rpc-endpoint",
        default="",
        help="llama.cpp RPC endpoint advertised by this node, in host:port form",
    )
    p_serve.add_argument("--proof-endpoint", default="")
    p_serve.add_argument("--network-reachability", default="unknown")
    p_serve.add_argument("--relay-hint", default="")
    p_serve.add_argument(
        "--mesh-sync-timeout",
        type=float,
        default=3.0,
        help="Seconds to wait for coordinator/worker mesh spec sync calls",
    )
    p_serve.add_argument(
        "--backend-url",
        default="",
        help="Local OpenAI-compatible backend URL, e.g. llama-server http://127.0.0.1:8080",
    )
    p_serve.add_argument(
        "--proof-url",
        default="",
        help="Verathos GGML proof endpoint URL; can be a worker or proof adapter",
    )
    p_serve.add_argument(
        "--proof-trace-dir",
        default="",
        help="Directory where a proof-capable RPC worker writes GGML witnesses",
    )
    p_serve.add_argument(
        "--proof-gguf-manifest",
        default="",
        help="GGUF tensor manifest JSON used to attach selected weight-byte openings",
    )
    p_serve.add_argument("--proof-tolerance-abs", type=float, default=8e-2)
    p_serve.add_argument("--proof-tolerance-rel", type=float, default=4e-2)
    p_serve.add_argument("--proof-block-size", type=int, default=64)
    p_serve.add_argument("--proof-spot-checks", type=int, default=8)
    p_serve.add_argument(
        "--proof-artifact-timeout",
        type=float,
        default=DEFAULT_PROOF_ARTIFACT_TIMEOUT,
        help="Seconds to wait for remote worker proof commitment/selection artifacts",
    )
    p_serve.add_argument(
        "--proof-sample-bps",
        type=int,
        default=PROOF_SAMPLE_BPS_DENOMINATOR,
        help="Inline Fiat-Shamir GGML proof sampling rate in basis points",
    )
    p_serve.add_argument(
        "--defer-proof",
        action="store_true",
        help=(
            "Commit proof metadata on the response path and require future-randomness "
            "proof bundles instead of synchronous selected replay"
        ),
    )
    p_serve.add_argument(
        "--proof-ops-per-request",
        type=int,
        default=1,
        help="GGML MUL_MAT witnesses to capture/prove when proof mode is active",
    )
    p_serve.add_argument(
        "--proof-trace-candidates-per-request",
        type=int,
        default=8,
        help="GGML trace candidates to commit before beacon-selecting proof ops",
    )
    p_serve.add_argument(
        "--proof-trace-max-elems",
        type=int,
        default=0,
        help=(
            "Maximum total tensor elements for a captured GGML proof op; "
            "0 auto-scales from GGUF metadata when available"
        ),
    )
    p_serve.add_argument(
        "--decode-audit-bps",
        type=int,
        default=0,
        help="Sampled GGUF decode/logit audit rate in basis points; requires --require-proof",
    )
    p_serve.add_argument(
        "--decode-audit-top-k",
        type=int,
        default=8,
        help="Top-K logits metadata to include in sampled decode audit openings",
    )
    p_serve.add_argument("--no-proof-warmup", action="store_true")
    p_serve.add_argument(
        "--proof-decode-projection-warmup",
        action="store_true",
        help=(
            "Preload final-projection GGUF proof weights for decode canary audits "
            "during worker startup instead of the first sampled canary"
        ),
    )
    p_serve.add_argument(
        "--require-proof",
        action="store_true",
        help="Reject receipt-only mesh compute and require bound GGML proof transcripts",
    )
    p_serve.add_argument("--llama-model", default="", help="Start llama-server with this GGUF model")
    p_serve.add_argument(
        "--llama-hf",
        default="",
        help="Start llama-server with llama.cpp --hf REPO[:QUANT], e.g. Qwen/Qwen2.5-7B-Instruct-GGUF:Q4_K_M",
    )
    p_serve.add_argument("--llama-server-binary", default="llama-server")
    p_serve.add_argument("--llama-host", default="127.0.0.1")
    p_serve.add_argument("--llama-port", type=int, default=DEFAULT_LLAMA_SERVER_PORT)
    p_serve.add_argument("--llama-device", default="")
    p_serve.add_argument(
        "--llama-capture-trace-dir",
        default="",
        help=(
            "local-stage mode: llama-server computes the first member's "
            "stage on local devices and captures into this (member-owned) "
            "trace dir; the coordinator stays orchestration-only"
        ),
    )
    p_serve.add_argument("--llama-n-gpu-layers", default=None)
    p_serve.add_argument("--llama-ctx-size", type=int, default=None)
    p_serve.add_argument("--llama-tensor-split", default="")
    p_serve.add_argument("--llama-alias", default="")
    p_serve.add_argument("--llama-extra-arg", action="append", default=[])
    p_serve.add_argument(
        "--slot-state-dir",
        default="",
        help=(
            "Directory for llama slot-state save/restore (bounded hard-audit "
            "probes). Enables llama-server --slot-save-path + --slots on the "
            "loopback aux port; empty disables"
        ),
    )
    p_serve.add_argument(
        "--llama-min-rpc-workers",
        type=int,
        default=0,
        help="Wait for at least this many joined RPC workers before starting llama-server",
    )
    p_serve.add_argument(
        "--llama-reload-interval",
        type=float,
        default=2.0,
        help="Seconds between mesh RPC plan checks; <=0 disables reload after first check",
    )
    p_serve.add_argument(
        "--llama-rpc-ready-timeout",
        type=float,
        default=0.5,
        help="TCP connect timeout for each joined llama.cpp RPC endpoint before launching llama-server",
    )
    p_serve.add_argument("--llama-dry-run", action="store_true")
    p_serve.add_argument("--rpc-worker", action="store_true", help="Start llama.cpp rpc-server too")
    p_serve.add_argument("--rpc-worker-binary", default="rpc-server")
    p_serve.add_argument("--rpc-host", default="0.0.0.0")
    p_serve.add_argument("--rpc-port", type=int, default=DEFAULT_LLAMA_RPC_PORT)
    p_serve.add_argument("--rpc-device", default="")
    p_serve.add_argument("--rpc-cache", action="store_true")
    p_serve.add_argument("--rpc-extra-arg", action="append", default=[])
    p_serve.add_argument("--rpc-dry-run", action="store_true")
    p_serve.set_defaults(func=cmd_serve)

    p_probe = sub.add_parser("probe", help="Probe a mesh node /health and /capability")
    p_probe.add_argument("endpoint")
    p_probe.add_argument("--timeout", type=float, default=3.0)
    p_probe.set_defaults(func=cmd_probe)

    p_plan_units = sub.add_parser(
        "plan-units", help="Derive one pool worker unit per GPU on this host"
    )
    p_plan_units.add_argument(
        "--worker-id-base",
        default=os.environ.get("HOSTNAME", "worker"),
        help="Base name; each unit becomes <base>-gpu<group>",
    )
    p_plan_units.add_argument(
        "--gpus",
        default="",
        help=(
            "GPU grouping: comma separates worker units, + groups GPUs into "
            "one unit (0,1,2+3 = three units). Default: ALL GPUs as one unit"
        ),
    )
    p_plan_units.add_argument(
        "--backend", default="cuda", choices=["cuda", "metal", "cpu"]
    )
    p_plan_units.add_argument("--rpc-port", type=int, default=50052)
    p_plan_units.add_argument("--proof-port", type=int, default=9402)
    p_plan_units.add_argument("--mesh-port", type=int, default=9443)
    p_plan_units.set_defaults(func=cmd_plan_units)

    p_register_units = sub.add_parser(
        "register-units",
        help="Record planned worker units in ~/.verathos/mesh-units.json",
    )
    p_register_units.add_argument(
        "--units-file", required=True, help="JSON from plan-units"
    )
    p_register_units.add_argument(
        "--token-file", required=True, help="Pool token file (owner-only)"
    )
    p_register_units.set_defaults(func=cmd_register_units)

    p_add_model = sub.add_parser(
        "add-model",
        help=(
            "Add a GGUF already on this machine to the pool (builds the "
            "tensor manifest, updates worker catalogs)"
        ),
    )
    p_add_model.add_argument("gguf", help="Path to a .gguf file (any shard)")
    p_add_model.add_argument(
        "--model-id", default="", help="Override the filename-derived id"
    )
    p_add_model.add_argument(
        "--hf-repo",
        default="",
        help="Optional HF download source so OTHER machines can fetch it",
    )
    p_add_model.add_argument(
        "--hf-files", default="", help="Comma-separated GGUF filenames"
    )
    p_add_model.add_argument(
        "--catalog",
        default="",
        help="Catalog file override (default: this machine's unit catalogs)",
    )
    p_add_model.add_argument(
        "--manifest",
        default="",
        help=(
            "Reuse a published tensor manifest (e.g. store-fetched) "
            "instead of building one"
        ),
    )
    p_add_model.add_argument(
        "--no-restart",
        action="store_true",
        help="Do not restart idle worker units to advertise the model",
    )
    p_add_model.set_defaults(func=cmd_mesh_add_model)

    p_fleet = sub.add_parser(
        "fleet",
        help="Workers, GPUs, meshes, and placement advice for a pool",
    )
    _add_optional_pool_token_arguments(p_fleet)
    p_fleet.add_argument(
        "--model-id", default="", help="Also show placement advice for a model"
    )
    p_fleet.add_argument("--json", action="store_true")
    p_fleet.add_argument("--timeout", type=float, default=10.0)
    p_fleet.set_defaults(func=cmd_fleet)

    p_setup = sub.add_parser(
        "setup",
        help=(
            "Guided setup: coordinator, GPU worker, or single-box pool "
            "(the human path over the pool commands)"
        ),
    )
    p_setup.set_defaults(func=cmd_mesh_setup)

    p_manage = sub.add_parser(
        "manage",
        help=(
            "The operator board: launch/stop/switch meshes, placement, "
            "machines, probes (day-2 human path)"
        ),
    )
    p_manage.add_argument(
        "--pool",
        default="",
        help="Pool state dir (default: the single local pool)",
    )
    p_manage.set_defaults(func=cmd_mesh_manage)

    p_chat = sub.add_parser(
        "chat",
        help="Interactive verified chat with a serving mesh (streaming)",
    )
    _add_optional_pool_token_arguments(p_chat)
    p_chat.add_argument(
        "--mesh-key",
        default="",
        help="mesh to chat with; picker shown when several are serving",
    )
    p_chat.add_argument("--max-tokens", type=int, default=4096)
    p_chat.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="per-reply deadline seconds",
    )
    p_chat.add_argument(
        "--thinking",
        action="store_true",
        help="let the model think before answering (slower)",
    )
    p_chat.add_argument(
        "--proof-tier",
        choices=("auto", "light", "hard"),
        # Empty default: an interactive session gets the startup tier
        # picker (falling back to "auto" when piped/scripted). An explicit
        # --proof-tier always skips the picker.
        default="",
        help=(
            "tier for YOUR chat replies only; the pool always runs both "
            "tiers on real traffic. auto = no override (dev chat "
            "resolves light like organic traffic, production chat is "
            "pinned hard as a self-test); light = openings-only "
            "milliseconds; hard = full GEMM relation inline per reply, "
            "seconds"
        ),
    )
    p_chat.set_defaults(func=cmd_mesh_chat)

    p_mesh_logs = sub.add_parser(
        "logs", help="Tail a mesh unit's PM2 logs (worker units or --manager)"
    )
    p_mesh_logs.add_argument("target", nargs="?", default="")
    p_mesh_logs.add_argument("--lines", type=int, default=50)
    p_mesh_logs.add_argument("--manager", action="store_true")
    p_mesh_logs.add_argument("--all", action="store_true")
    p_mesh_logs.set_defaults(func=cmd_mesh_logs)

    p_mesh_stop = sub.add_parser(
        "stop", help="Stop a mesh unit's PM2 process (never the whole fleet)"
    )
    p_mesh_stop.add_argument("target", nargs="?", default="")
    p_mesh_stop.add_argument("--manager", action="store_true")
    p_mesh_stop.set_defaults(func=cmd_mesh_stop)

    p_mesh_start = sub.add_parser(
        "start", help="Start a stopped mesh unit's PM2 process"
    )
    p_mesh_start.add_argument("target", nargs="?", default="")
    p_mesh_start.add_argument("--manager", action="store_true")
    p_mesh_start.add_argument("--all", action="store_true")
    p_mesh_start.set_defaults(func=cmd_mesh_start)

    def _add_signer_arguments(parser: argparse.ArgumentParser) -> None:
        # Not required: when neither is given, the command reads the pool's
        # stored miner identity from the manager (it runs with that wallet
        # anyway) instead of making the operator re-type what the pool
        # already knows.
        signer = parser.add_mutually_exclusive_group()
        signer.add_argument(
            "--wallet", help="Bittensor wallet (EVM key derived from hotkey "
            "seed); default: the pool's stored miner identity"
        )
        signer.add_argument(
            "--private-key-file",
            help="Owner-only file holding the EVM key (no hotkey seed: "
            "EVM binding then needs to exist already)",
        )
        parser.add_argument("--hotkey", default="default")

    p_deploy = sub.add_parser(
        "deploy",
        help="Launch a mesh, gate it on probes, then register it on-chain",
    )
    p_deploy.add_argument("model_id", help="On-chain mesh model id")
    _add_pool_token_arguments(p_deploy)
    p_deploy.add_argument(
        "--chain-config",
        default="",
        help="Explicit chain config JSON; defaults from --subtensor-network",
    )
    p_deploy.add_argument(
        "--subtensor-network",
        default="",
        choices=["", "test", "finney"],
        help="Target network; selects the shipped chain config and enables "
        "the hotkey-registered-on-netuid check (testnet and mainnet behave "
        "identically)",
    )
    p_deploy.add_argument(
        "--subtensor-chain-endpoint",
        default="",
        help="Your own subtensor node (ws:// or http://) for ALL chain "
        "reads/writes in this deploy; without it the shipped network "
        "default (public, often congested) is used. The pool's own flow "
        "passes the pool's node automatically.",
    )
    p_deploy.add_argument(
        "--endpoint",
        required=True,
        help="Public coordinator URL to register on MinerRegistry",
    )
    _add_signer_arguments(p_deploy)
    p_deploy.add_argument("--uid", type=int, default=None)
    p_deploy.add_argument(
        "--max-context-len",
        type=int,
        default=None,
        help="Override the registered context; default derives it as "
        "min(MEASURED KV auto-fit, time cap from a timing probe against "
        "the validator canary budget), and overrides are refused beyond "
        "10%% of the measurement",
    )
    p_deploy.add_argument(
        "--validator-budget-s",
        type=float,
        default=None,
        help="Canary budget the derived context cap is computed against; "
        "mirrors the validator's canary_full_context_inference_timeout "
        "(~900). Default: the gate's --full-context-budget-s",
    )
    p_deploy.add_argument(
        "--hard-samples",
        type=int,
        default=1,
        help="Explicit hard-tier proof probes in the gate (validator audit "
        "draws assert the hard relation)",
    )
    p_deploy.add_argument(
        "--workers", default="", help="Comma-separated placement override"
    )
    p_deploy.add_argument("--driver", default="")
    p_deploy.add_argument("--hf-repo", default="")
    p_deploy.add_argument("--hf-files", default="", help="Comma-separated")
    p_deploy.add_argument("--model-bytes", type=int, default=0)
    p_deploy.add_argument("--probe-samples", type=int, default=3)
    p_deploy.add_argument(
        "--min-tok-s",
        type=float,
        default=3.0,
        help="Hard throughput floor; ~1.0 is where canaries start timing out",
    )
    p_deploy.add_argument("--no-full-context-probe", action="store_true")
    p_deploy.add_argument(
        "--full-context-budget-s",
        type=float,
        default=540.0,
        help="Gate budget for the canary-shaped full-context probe "
        "(validator ceiling is 900)",
    )
    p_deploy.add_argument("--yes", action="store_true")
    p_deploy.add_argument(
        "--force",
        action="store_true",
        help="Register despite a failed gate (never overrides chain "
        "preconditions or reachability auth posture)",
    )
    p_deploy.add_argument("--dry-run", action="store_true")
    p_deploy.add_argument("--json", action="store_true")
    p_deploy.add_argument("--timeout", type=float, default=990.0)
    p_deploy.set_defaults(func=cmd_deploy)

    p_renew = sub.add_parser(
        "renew", help="Renew the on-chain lease once (cron escape hatch)"
    )
    _add_pool_token_arguments(p_renew)
    p_renew.add_argument(
        "--chain-config",
        default="",
        help="Explicit chain config JSON; defaults from --subtensor-network",
    )
    p_renew.add_argument(
        "--subtensor-network",
        default="",
        choices=["", "test", "finney"],
    )
    _add_signer_arguments(p_renew)
    p_renew.add_argument("--timeout", type=float, default=30.0)
    p_renew.set_defaults(func=cmd_renew)

    p_apikey = sub.add_parser(
        "apikey",
        help=(
            "Manage keys for the pool's private OpenAI API "
            "(/v1/chat/completions on the pool manager)"
        ),
    )
    p_apikey.add_argument(
        "action", choices=["create", "list", "revoke", "expose"]
    )
    p_apikey.add_argument(
        "--name", default="", help="label shown in `apikey list`"
    )
    p_apikey.add_argument(
        "--key-id", default="", help="id prefix from `apikey list` (revoke)"
    )
    p_apikey.add_argument(
        "--port",
        type=int,
        default=9543,
        help=(
            "expose: public https port for the API listener (0 disables); "
            "on rented boxes pick a provider-mapped port"
        ),
    )
    # Optional token args: on the coordinator box every apikey action works
    # with zero flags (single-pool discovery), which is the common case.
    _add_optional_pool_token_arguments(p_apikey)
    p_apikey.add_argument("--timeout", type=float, default=15.0)
    p_apikey.set_defaults(func=cmd_apikey)

    p_retire = sub.add_parser(
        "retire",
        help=(
            "Deactivate the registered model on chain (releases the "
            "endpoint claim) and clear the stored registration"
        ),
    )
    p_retire.add_argument(
        "model_id",
        nargs="?",
        default="",
        help="guard: must match the stored registration when given",
    )
    _add_pool_token_arguments(p_retire)
    p_retire.add_argument(
        "--chain-config",
        default="",
        help="Explicit chain config JSON; defaults from --subtensor-network",
    )
    p_retire.add_argument(
        "--subtensor-network",
        default="",
        choices=["", "test", "finney"],
    )
    _add_signer_arguments(p_retire)
    p_retire.add_argument(
        "--keep-serving",
        action="store_true",
        help=(
            "keep the mesh serving locally (chat / private API); only the "
            "on-chain entry is deactivated"
        ),
    )
    p_retire.add_argument("--yes", action="store_true")
    p_retire.add_argument("--timeout", type=float, default=60.0)
    p_retire.set_defaults(func=cmd_retire)

    p_reg_status = sub.add_parser(
        "registration-status",
        help="Show the stored registration and its on-chain lease",
    )
    _add_pool_token_arguments(p_reg_status)
    p_reg_status.add_argument("--chain-config", default="")
    p_reg_status.add_argument("--wallet", default="")
    p_reg_status.add_argument("--private-key-file", default="")
    p_reg_status.add_argument("--hotkey", default="default")
    p_reg_status.add_argument("--timeout", type=float, default=30.0)
    p_reg_status.set_defaults(func=cmd_registration_status)

    p_pool = sub.add_parser("pool", help="Worker pool + driver placement (orchestration v1)")
    pool_sub = p_pool.add_subparsers(dest="pool_command", required=True)

    pp = pool_sub.add_parser("create", help="Mint a new pool (state dir + pool token)")
    pp.add_argument("--root", default=str(Path.home() / ".verathos" / "pools"))
    pp.add_argument("--manager-endpoint", required=True, help="Public URL of `mesh pool serve`")
    pp.add_argument(
        "--serving-mode",
        required=True,
        # A pool is always miner-side. "subnet" = registered on the subnet as
        # a miner; the old spelling "validator" described the audience, not
        # the operator, and read as if a miner ran a validator. Still
        # accepted so existing scripts and pool state keep working.
        choices=["dev", "subnet", "validator"],
        help=(
            "dev = local only, no chain binding, nothing scoreable; "
            "subnet = this miner's pool on the subnet, signing the snapshots "
            "validators verify ('validator' is a deprecated alias for subnet)"
        ),
    )
    pp.add_argument(
        "--network",
        default="",
        choices=["", "testnet", "mainnet"],
        help=(
            "fill --chain-id/--netuid from the shipped chain config "
            "(testnet: netuid 405, mainnet: netuid 96); explicit flags "
            "still override"
        ),
    )
    pp.add_argument(
        "--owner-account",
        default="",
        help=(
            "Bittensor SS58 account that owns the operator dashboard; "
            "required in validator mode"
        ),
    )
    pp.add_argument(
        "--coordinator-address",
        default="",
        help="Coordinator miner EVM address used for exact validator-score mapping",
    )
    pp.add_argument(
        "--validator-shared-state",
        default="",
        help="Read-only path to the isolated validator shared_state.json",
    )
    pp.add_argument("--chain-id", type=int, default=None)
    pp.add_argument("--netuid", type=int, default=None)
    pp.add_argument("--coordinator-uid", type=int, default=None)
    pp.add_argument("--epoch", type=int, default=None)
    pp.add_argument(
        "--snapshot-ttl-seconds",
        type=int,
        default=86_400,
        help="Signed snapshot lifetime (validator mode only)",
    )
    pp.set_defaults(func=cmd_pool_create)

    pp = pool_sub.add_parser(
        "upgrade",
        help=(
            "Flip a dev pool to subnet mode IN PLACE (same pool id, same "
            "tokens; workers rejoin on restart)"
        ),
    )
    pp.add_argument("--pool", required=True, help="pool state directory")
    pp.add_argument(
        "--network",
        default="",
        choices=["", "testnet", "mainnet"],
        help="fill --chain-id/--netuid from the shipped chain config",
    )
    pp.add_argument(
        "--owner-account",
        default="",
        help="Bittensor SS58 account that owns the operator dashboard",
    )
    pp.add_argument(
        "--coordinator-address",
        default="",
        help="Coordinator miner EVM address used for exact validator-score mapping",
    )
    pp.add_argument(
        "--validator-shared-state",
        default="",
        help="Read-only path to the isolated validator shared_state.json",
    )
    pp.add_argument("--chain-id", type=int, default=None)
    pp.add_argument("--netuid", type=int, default=None)
    pp.add_argument("--coordinator-uid", type=int, default=None)
    pp.add_argument("--epoch", type=int, default=None)
    pp.add_argument(
        "--snapshot-ttl-seconds",
        type=int,
        default=86_400,
        help="Signed snapshot lifetime",
    )
    pp.add_argument(
        "--manager-endpoint",
        default="",
        help=(
            "new manager URL (e.g. https:// in front of a remote manager); "
            "omit to keep the current one. Changing it re-mints the token "
            "files with the new endpoint (secrets stay), so workers need "
            "the re-printed token"
        ),
    )
    pp.set_defaults(func=cmd_pool_upgrade)

    pp = pool_sub.add_parser("serve", help="Run the pool manager (control plane)")
    pp.add_argument("--pool", required=True, help="pool state directory")
    pp.add_argument("--host", default="0.0.0.0")
    pp.add_argument("--port", type=int, default=9500)
    pp.add_argument(
        "--coordinator-address",
        default="",
        help="Override the coordinator miner EVM address for validator-score mapping",
    )
    pp.add_argument(
        "--validator-shared-state",
        default="",
        help="Override the read-only isolated validator shared_state.json path",
    )
    pp.add_argument(
        "--tls-cert",
        default="",
        help="PEM certificate chain for built-in HTTPS; requires --tls-key",
    )
    pp.add_argument(
        "--tls-key",
        default="",
        help="PEM private key for built-in HTTPS; requires --tls-cert",
    )
    pp.add_argument(
        "--wallet-name",
        default="",
        help=(
            "Bittensor wallet whose hotkey-derived EVM key renews the "
            "on-chain lease; enables the in-manager lease renewer"
        ),
    )
    pp.add_argument("--wallet-hotkey", default="default")
    pp.add_argument(
        "--chain-config",
        default="",
        help="Chain config JSON; required for the lease renewer",
    )
    pp.add_argument(
        "--subtensor-network",
        default="",
        type=_subtensor_network_arg,
        help=(
            "bittensor network for metagraph-backed status (UID, incentive); "
            "derived from the pool's chain id when omitted"
        ),
    )
    pp.add_argument(
        "--api-tls-port",
        type=int,
        default=0,
        help=(
            "open a SECOND https listener for the private OpenAI API and "
            "dashboard (workers keep dialing the plain-http --port); 0 = off. "
            "Certs default to the pool's auto-minted self-signed pair"
        ),
    )
    pp.add_argument("--api-tls-cert", default="")
    pp.add_argument("--api-tls-key", default="")
    pp.set_defaults(func=cmd_pool_serve)

    pp = pool_sub.add_parser("worker", help="Join the pool and execute mesh assignments")
    _add_pool_token_arguments(pp)
    pp.add_argument("--workdir", required=True)
    pp.add_argument("--advertise-host", required=True, help="address the driver dials for this worker")
    pp.add_argument("--rpc-port", type=int, default=50052)
    pp.add_argument("--proof-port", type=int, default=9402)
    # NOT 9500: that is the pool MANAGER's default port, and a worker kills
    # whatever listens on its own ports before driving (_free_own_ports) — a
    # shared default would shoot the co-located control plane on a single box.
    pp.add_argument("--mesh-port", type=int, default=9443, help="coordinator port when driving")
    # Empty = AUTO: the worker detects its GPU arch and reuses or builds the
    # matching patched llama.cpp (plug-and-play; no hand-picked binaries).
    pp.add_argument("--llama-server-binary", default="")
    pp.add_argument("--rpc-worker-binary", default="")
    pp.add_argument("--rpc-device", default="CUDA0")
    pp.add_argument("--catalog", default="", help="json list: model_id, llama_model, manifest, layers, model_bytes")
    pp.add_argument("--wallet-name", default="", help="Bittensor wallet to sign receipts with (miner mode)")
    pp.add_argument("--wallet-hotkey", default="", help="Hotkey name within --wallet-name")
    pp.add_argument(
        "--validator-allowlist-path",
        default="",
        help="Local validator allowlist JSON (required when this worker drives a subnet mesh)",
    )
    pp.add_argument(
        "--validator-allowlist-max-age-seconds",
        type=_positive_seconds,
        default=DEFAULT_VALIDATOR_ALLOWLIST_MAX_AGE_SECONDS,
        help=(
            "Maximum accepted age of the local validator allowlist "
            "(default: 900)"
        ),
    )
    pp.add_argument(
        "--subtensor-network",
        default="",
        type=_subtensor_network_arg,
        help=(
            "With --netuid: refresh the validator allowlist from this "
            "network's metagraph every few minutes (a standalone mesh "
            "worker has no vLLM miner process to keep it fresh)"
        ),
    )
    pp.add_argument(
        "--netuid",
        type=int,
        default=None,
        help="Subnet netuid for the allowlist refresher",
    )
    pp.add_argument(
        "--chain-config",
        default="",
        help=(
            "Chain config JSON for the allowlist refresher's "
            "minValidatorStake filter (optional)"
        ),
    )
    pp.add_argument(
        "--allow-validators",
        default="",
        help="Comma-separated extra validator hotkeys to always allow",
    )
    pp.add_argument(
        "--stage-proof-key-file",
        default="",
        help=(
            "Persistent owner-only worker proof seed file; defaults to one "
            "auto-generated inside --workdir"
        ),
    )
    pp.add_argument("--repo-root", default="")
    pp.add_argument("--worker-id", default="")
    pp.add_argument("--gpu-name", default="")
    pp.add_argument("--vram-gb", type=int, default=0)
    # Match pool.DEFAULT_HEARTBEAT_S: command pickup and operator chats ride
    # the beat, so a slow CLI default re-adds the multi-second TTFT the
    # constant was lowered to remove.
    pp.add_argument("--heartbeat", type=float, default=0.5)
    pp.add_argument(
        "--member-only",
        action="store_true",
        help="Serve pipeline stages only; never drive (host coordinator + llama-server)",
    )
    pp.set_defaults(func=cmd_pool_worker)

    for name, fn in (
        ("status", cmd_pool_status),
        ("recommend", cmd_pool_recommend),
        ("launch", cmd_pool_launch),
        ("stop", cmd_pool_stop),
        ("register-model", cmd_pool_register_model),
        # Operator-facing name for the same source-teaching primitive:
        # "register" belongs to the on-chain ModelSpec (subnet owner /
        # deploy), never to an operator action.
        ("add-model-source", cmd_pool_register_model),
        ("remove-worker", cmd_pool_remove_worker),
    ):
        loop_help = {
            "status": "Full management view of workers and meshes",
            "recommend": "Ranked placement suggestions for one model",
            "launch": "Start a mesh (guided picker on a TTY)",
            "stop": "Tear one mesh down (frees its workers)",
            "register-model": "Low-level source/anchor write (deploy's tool)",
            "add-model-source": "Teach the pool a model's download source",
            "remove-worker": "Remove an OFFLINE worker's record",
        }
        pp = pool_sub.add_parser(name, help=loop_help.get(name, ""))
        _add_optional_pool_token_arguments(pp)
        pp.add_argument("--timeout", type=float, default=10.0)
        if name in ("recommend", "register-model", "add-model-source"):
            pp.add_argument("--model-id", required=True)
        if name == "launch":
            # Optional on purpose: with no flags on a terminal, launch is a
            # guided flow (model picker + recommended placement + confirm).
            pp.add_argument("--model-id", default="")
            pp.add_argument("--workers", default="", help="comma-separated worker ids (empty = recommend)")
            pp.add_argument("--driver", default="")
            pp.add_argument(
                "--no-wait",
                action="store_true",
                help="return at spawn instead of waiting for routing-ready",
            )
            pp.add_argument(
                "--wait-timeout",
                type=float,
                default=900.0,
                help=(
                    "seconds to wait for routing-ready; the window slides "
                    "while the driver is still downloading the model"
                ),
            )
        if name == "stop":
            pp.add_argument("--mesh-key", required=True)
        if name == "remove-worker":
            pp.add_argument("--worker-id", required=True)
        if name in ("register-model", "add-model-source"):
            pp.add_argument("--hf-repo", required=True)
            pp.add_argument("--hf-files", required=True, help="comma-separated GGUF filenames in the repo")
            pp.add_argument("--layers", type=int, default=0)
            pp.add_argument("--model-bytes", type=int, default=0)
            pp.add_argument("--proof-tolerance-abs", type=float, default=None)
            pp.add_argument("--proof-tolerance-rel", type=float, default=None)
            pp.add_argument(
                "--llama-ubatch",
                type=int,
                default=None,
                help=(
                    "Per-model llama micro-batch (mixed-batch TTFT floor); "
                    "MoE models with slow prefill want 512-1024"
                ),
            )
            pp.add_argument("--llama-batch", type=int, default=None)
            pp.add_argument(
                "--model-index",
                type=int,
                default=None,
                help="Pin this model to the coordinator's MinerRegistry model index",
            )
            pp.add_argument(
                "--manifest-urls",
                default="",
                help=(
                    "Comma-separated manifest store base URLs; drivers pull "
                    "the published tensor manifest instead of rebuilding it"
                ),
            )
            pp.add_argument(
                "--max-context-len",
                type=int,
                default=None,
                help=(
                    "Pin the model to the coordinator MinerRegistry context "
                    "limit; required by validator pools"
                ),
            )
            pp.add_argument("--model-package-hash", default="")
            pp.add_argument("--model-tensor-manifest-root", default="")
            pp.add_argument("--tokenizer-hash", default="")
            pp.add_argument("--quantization-scheme", default="")
        pp.set_defaults(func=fn)

    pp = pool_sub.add_parser(
        "probe",
        help=(
            "Verified end-to-end self-test of a serving mesh (proof-required "
            "chat through the pool; --gate runs the full deploy probe gate)"
        ),
    )
    _add_optional_pool_token_arguments(pp)
    pp.add_argument(
        "--mesh-key",
        default="",
        help="mesh to probe; auto-selected when exactly one mesh is serving",
    )
    pp.add_argument(
        "--prompt",
        default="Briefly describe what a Merkle tree is used for.",
    )
    pp.add_argument("--max-tokens", type=int, default=120)
    pp.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="in-band probe deadline seconds (HTTP timeout adds headroom)",
    )
    pp.add_argument(
        "--gate",
        action="store_true",
        help=(
            "run the full deploy probe gate (small samples + canary-shaped "
            "full-context) and exit nonzero on fail"
        ),
    )
    pp.add_argument("--samples", type=int, default=3)
    pp.add_argument(
        "--hard-samples",
        type=int,
        default=1,
        help="explicit hard-tier proof probes in the gate",
    )
    pp.add_argument("--min-tok-s", type=float, default=None)
    pp.add_argument("--no-full-context", action="store_true")
    pp.set_defaults(func=cmd_pool_probe)

    pp = pool_sub.add_parser(
        "set-epoch",
        help="Stage a later validator epoch for the next mesh launch",
    )
    _add_optional_pool_token_arguments(pp)
    pp.add_argument("--timeout", type=float, default=10.0)
    pp.add_argument("--epoch", type=int, required=True)
    pp.set_defaults(func=cmd_pool_set_epoch)

    pp = pool_sub.add_parser(
        "join-token",
        help="Print the worker join one-liner for a pool again (coordinator box)",
    )
    pp.add_argument(
        "--pool",
        default="",
        help="Pool state dir; discovered from this machine's pools when omitted",
    )
    pp.add_argument(
        "--endpoint",
        default="",
        help=(
            "Mint a REMOTE join token against this reachable manager URL "
            "(e.g. https://<public-ip>:<api-tls-port>). HTTPS is required "
            "for non-loopback endpoints; the token then pins the manager's "
            "TLS certificate so remote workers need no CA files."
        ),
    )
    pp.set_defaults(func=cmd_pool_join_token)

    p_handshake = sub.add_parser("handshake", help="Check that a node accepts a mesh spec")
    p_handshake.add_argument("endpoint")
    p_handshake.add_argument("--mesh", required=True, help="mesh.json or mesh directory")
    p_handshake.add_argument("--timeout", type=float, default=3.0)
    p_handshake.set_defaults(func=cmd_handshake)

    p_infer = sub.add_parser("infer", help="Send a non-streaming test request through a mesh")
    p_infer.add_argument("endpoint", help="Coordinator mesh endpoint")
    p_infer.add_argument("--model", required=True)
    p_infer.add_argument("--prompt", required=True)
    p_infer.add_argument("--system", default="")
    p_infer.add_argument(
        "--max-tokens",
        type=int,
        default=0,
        help=(
            "Completion token cap for this request. 0 (default) sends no "
            "cap and the coordinator applies its serve-side default "
            "(VERATHOS_MESH_DEFAULT_MAX_TOKENS, 4096)."
        ),
    )
    p_infer.add_argument("--stream", action="store_true")
    p_infer.add_argument("--content", action="store_true", help="Print only the assistant text")
    p_infer.add_argument(
        "--proof-tier",
        choices=("auto", "light", "hard"),
        default="auto",
        help=(
            "Proof tier for this request. There are exactly two tiers: light "
            "(openings-only, what all organic traffic rides) and hard (the "
            "full GEMM relation, what validator canaries run). auto = the "
            "lane default (light on a coordinator's organic lane). 'hard' is "
            "an upgrade-only request; 'light' cannot downgrade a lane that "
            "must assert the hard relation."
        ),
    )
    p_infer.add_argument(
        "--validator-nonce",
        default="",
        help="32-byte hex nonce for sampled mesh proof; use 'auto' for local testing",
    )
    p_infer.add_argument(
        "--proof-sample-bps",
        type=int,
        default=None,
        help=(
            "SAMPLING RATE override (bps of requests that carry a proof), "
            "NOT a proof tier: the tier is chosen by --proof-tier / the "
            "post-receipt nonce draw. Raising this makes THIS request carry "
            "its lane's proof instead of riding the sampled fraction."
        ),
    )
    p_infer.add_argument(
        "--decode-audit-bps",
        type=int,
        default=None,
        help=(
            "SAMPLING RATE override for the decode/logit audit (bps of "
            "requests audited), NOT a proof tier. The audit machinery is "
            "armed at serve time; this only raises how often it is drawn."
        ),
    )
    p_infer.add_argument(
        "--decode-audit-top-k",
        type=int,
        default=None,
        help="Top-K logits metadata for request decode/logit canaries",
    )
    p_infer.add_argument("--timeout", type=float, default=120.0)
    p_infer.set_defaults(func=cmd_infer)

    p_verify_artifact = sub.add_parser(
        "verify-artifact",
        help="Verify a saved mesh inference artifact without trusting the coordinator",
    )
    p_verify_artifact.add_argument("artifact", help="Artifact JSON to verify")
    p_verify_artifact.add_argument(
        "--request",
        required=True,
        help="Original OpenAI-compatible request JSON",
    )
    p_verify_artifact.add_argument(
        "--response",
        default="",
        help="OpenAI response JSON when the artifact is only verathos_mesh metadata",
    )
    p_verify_artifact.add_argument(
        "--mesh",
        default="",
        help="mesh.json or mesh directory for spec-bound verification",
    )
    p_verify_artifact.add_argument("--member-index", type=int, default=None)
    p_verify_artifact.add_argument("--require-configured-proof", action="store_true")
    p_verify_artifact.add_argument("--require-cryptographic-proof", action="store_true")
    p_verify_artifact.add_argument(
        "--deferred-randomness",
        default="",
        help="32-byte future randomness hex used to evaluate deferred audit sampling",
    )
    p_verify_artifact.add_argument(
        "--require-deferred-proof-if-sampled",
        action="store_true",
        help="Fail if supplied future randomness samples the artifact but proof receipts are absent",
    )
    p_verify_artifact.add_argument(
        "--deferred-audit-bundle",
        default="",
        help="Post-hoc deferred audit bundle JSON to verify against this artifact",
    )
    p_verify_artifact.add_argument(
        "--proof-stage-index",
        type=int,
        action="append",
        default=None,
        help="Require a proof receipt for this mesh stage index; repeatable",
    )
    p_verify_artifact.add_argument(
        "--no-spec-proof-stage-coverage",
        action="store_true",
        help="Do not infer required proof stages from mesh members with proof_endpoint",
    )
    p_verify_artifact.set_defaults(func=cmd_verify_artifact)

    p_resolve_deferred = sub.add_parser(
        "resolve-deferred-audit",
        help="Fetch a future-randomness audit proof bundle for a saved mesh artifact",
    )
    p_resolve_deferred.add_argument("artifact", help="Artifact JSON to audit")
    p_resolve_deferred.add_argument(
        "--request",
        required=True,
        help="Original OpenAI-compatible request JSON",
    )
    p_resolve_deferred.add_argument(
        "--response",
        default="",
        help="OpenAI response JSON when the artifact is only verathos_mesh metadata",
    )
    p_resolve_deferred.add_argument(
        "--endpoint",
        required=True,
        help="Coordinator or worker endpoint exposing /v1/mesh/proof/deferred-audit",
    )
    p_resolve_deferred.add_argument(
        "--deferred-randomness",
        required=True,
        help="32-byte future randomness hex that sampled the artifact",
    )
    p_resolve_deferred.add_argument(
        "--mesh",
        default="",
        help="mesh.json or mesh directory for local bundle verification",
    )
    p_resolve_deferred.add_argument("--member-index", type=int, default=None)
    p_resolve_deferred.add_argument("--timeout", type=float, default=120.0)
    p_resolve_deferred.add_argument("--output", default="", help="Write bundle JSON to this path")
    p_resolve_deferred.set_defaults(func=cmd_resolve_deferred_audit)

    p_rpc_plan = sub.add_parser("rpc-plan", help="Show llama.cpp RPC endpoints for a mesh")
    p_rpc_plan.add_argument("mesh", help="mesh.json or mesh directory")
    p_rpc_plan.add_argument("--model", default="", help="Optional GGUF model for command preview")
    p_rpc_plan.add_argument(
        "--hf",
        default="",
        help="Optional llama.cpp --hf REPO[:QUANT] for command preview",
    )
    p_rpc_plan.add_argument("--llama-server-binary", default="llama-server")
    p_rpc_plan.add_argument("--host", default="127.0.0.1")
    p_rpc_plan.add_argument("--port", type=int, default=DEFAULT_LLAMA_SERVER_PORT)
    p_rpc_plan.add_argument("--device", default="")
    p_rpc_plan.add_argument("--n-gpu-layers", default=None)
    p_rpc_plan.add_argument("--ctx-size", type=int, default=None)
    p_rpc_plan.add_argument("--tensor-split", default="")
    p_rpc_plan.add_argument("--alias", default="")
    p_rpc_plan.add_argument("--extra-arg", action="append", default=[])
    p_rpc_plan.set_defaults(func=cmd_rpc_plan)

    p_rpc_worker = sub.add_parser("rpc-worker", help="Run a llama.cpp rpc-server worker")
    p_rpc_worker.add_argument("--binary", default="rpc-server")
    p_rpc_worker.add_argument("--host", default="0.0.0.0")
    p_rpc_worker.add_argument("--advertise-host", default="")
    p_rpc_worker.add_argument("--port", type=int, default=DEFAULT_LLAMA_RPC_PORT)
    p_rpc_worker.add_argument("--device", default="")
    p_rpc_worker.add_argument("--cache", action="store_true")
    p_rpc_worker.add_argument("--extra-arg", action="append", default=[])
    p_rpc_worker.add_argument("--dry-run", action="store_true")
    p_rpc_worker.set_defaults(func=cmd_rpc_worker)

    p_llama_server = sub.add_parser(
        "llama-server",
        help="Run llama-server using RPC endpoints advertised by a mesh",
    )
    p_llama_server.add_argument("--mesh", required=True, help="mesh.json or mesh directory")
    p_llama_server.add_argument("--model", default="", help="GGUF model path")
    p_llama_server.add_argument(
        "--hf",
        default="",
        help="llama.cpp --hf REPO[:QUANT], e.g. Qwen/Qwen2.5-7B-Instruct-GGUF:Q4_K_M",
    )
    p_llama_server.add_argument("--binary", default="llama-server")
    p_llama_server.add_argument("--host", default="127.0.0.1")
    p_llama_server.add_argument("--port", type=int, default=DEFAULT_LLAMA_SERVER_PORT)
    p_llama_server.add_argument("--device", default="")
    p_llama_server.add_argument("--n-gpu-layers", default=None)
    p_llama_server.add_argument("--ctx-size", type=int, default=None)
    p_llama_server.add_argument("--tensor-split", default="")
    p_llama_server.add_argument("--alias", default="")
    p_llama_server.add_argument("--extra-arg", action="append", default=[])
    p_llama_server.add_argument("--dry-run", action="store_true")
    p_llama_server.set_defaults(func=cmd_llama_server)

    p_proof_adapter = sub.add_parser(
        "proof-adapter",
        help="Run the local GGML GEMM proof adapter for proof-capable RPC traces",
    )
    p_proof_adapter.add_argument("--trace-dir", required=True)
    p_proof_adapter.add_argument("--host", default="127.0.0.1")
    p_proof_adapter.add_argument("--port", type=int, default=9349)
    p_proof_adapter.add_argument("--tolerance-abs", type=float, default=8e-2)
    p_proof_adapter.add_argument("--tolerance-rel", type=float, default=4e-2)
    p_proof_adapter.add_argument("--proof-block-size", type=int, default=64)
    p_proof_adapter.add_argument("--spot-checks", type=int, default=8)
    p_proof_adapter.add_argument(
        "--gguf-manifest",
        default="",
        help="GGUF tensor manifest JSON used to attach selected weight-byte openings",
    )
    p_proof_adapter.add_argument("--no-warmup", action="store_true")
    p_proof_adapter.set_defaults(func=cmd_proof_adapter)

    p_build_runtime = sub.add_parser(
        "build-runtime",
        help="Patch and build proof-capable llama.cpp rpc-server binaries",
    )
    p_build_runtime.add_argument("llama_dir", help="Path to a llama.cpp checkout")
    p_build_runtime.add_argument("--build-dir", default="", help="Optional CMake build dir")
    p_build_runtime.add_argument("--cuda", action="store_true", help="Enable CUDA GGML proof hook")
    p_build_runtime.add_argument(
        "--cuda-architectures",
        default="",
        help="Optional CMAKE_CUDA_ARCHITECTURES override, e.g. 89-real for RTX 4090",
    )
    p_build_runtime.add_argument("--metal", action="store_true", help="Enable Metal GGML proof hook")
    p_build_runtime.add_argument("--vulkan", action="store_true", help="Enable Vulkan GGML proof hook")
    p_build_runtime.add_argument(
        "--llama-server",
        action="store_true",
        help="Also build llama-server next to the proof-capable rpc-server",
    )
    p_build_runtime.add_argument("--jobs", type=int, default=0)
    p_build_runtime.add_argument("--dry-run", action="store_true")
    p_build_runtime.set_defaults(func=cmd_build_runtime)

    p_gguf_manifest = sub.add_parser(
        "gguf-manifest",
        help="Build a GGUF tensor-byte manifest for model proof binding",
    )
    p_gguf_manifest.add_argument(
        "model",
        nargs="+",
        help="GGUF model path; split GGUF first shards auto-expand when siblings exist",
    )
    p_gguf_manifest.add_argument("--output", default="", help="Write full manifest JSON")
    p_gguf_manifest.add_argument("--full", action="store_true", help="Print full manifest")
    p_gguf_manifest.set_defaults(func=cmd_gguf_manifest)

    p_proof_cache = sub.add_parser(
        "build-proof-cache",
        help="Pre-build the content-addressed proof-weight cache for a manifest "
        "(one-time per machine; afterwards proving never reads the GGUF)",
    )
    p_proof_cache.add_argument("manifest", help="tensor-manifest.json path")
    p_proof_cache.add_argument(
        "--cache-dir", default="", help="Override VERALLM_PROOF_WEIGHT_CACHE_DIR"
    )
    p_proof_cache.set_defaults(func=cmd_build_proof_cache)

    p_smoke = sub.add_parser(
        "smoke",
        help="Run a local coordinator + proof-capable RPC worker mesh smoke test",
    )
    p_smoke.add_argument("--model-id", required=True)
    p_smoke.add_argument("--llama-model", default="", help="GGUF model path")
    p_smoke.add_argument(
        "--llama-hf",
        default="",
        help="llama.cpp -hf REPO[:QUANT], e.g. Qwen/Qwen2.5-7B-Instruct-GGUF:Q4_K_M",
    )
    p_smoke.add_argument("--package-hash", default="", help="Model package hash; defaults to SHA256(package ref)")
    p_smoke.add_argument("--layers", type=int, required=True)
    p_smoke.add_argument("--llama-server-binary", default="llama-server")
    p_smoke.add_argument("--rpc-worker-binary", default="verathos-rpc-server")
    p_smoke.add_argument("--llama-device", default="RPC0")
    p_smoke.add_argument("--rpc-device", default="CUDA0")
    p_smoke.add_argument("--prompt", default="Explain verified mesh inference in one sentence.")
    p_smoke.add_argument("--max-tokens", type=int, default=64)
    p_smoke.add_argument("--samples", type=int, default=1)
    p_smoke.add_argument(
        "--proof-sample-bps",
        type=int,
        default=PROOF_SAMPLE_BPS_DENOMINATOR,
        help="Inline Fiat-Shamir proof sampling rate in basis points",
    )
    p_smoke.add_argument(
        "--proof-ops-per-request",
        type=int,
        default=1,
        help="GGML MUL_MAT witnesses to capture/prove when proof mode is active",
    )
    p_smoke.add_argument(
        "--proof-trace-candidates-per-request",
        type=int,
        default=8,
        help="GGML trace candidates to commit before beacon-selecting proof ops",
    )
    p_smoke.add_argument("--proof-tolerance-abs", type=float, default=8e-2)
    p_smoke.add_argument("--proof-tolerance-rel", type=float, default=4e-2)
    p_smoke.add_argument(
        "--decode-audit-bps",
        type=int,
        default=0,
        help="Sampled GGUF decode/logit audit rate in basis points",
    )
    p_smoke.add_argument(
        "--decode-audit-top-k",
        type=int,
        default=8,
        help="Top-K logits metadata to include in sampled decode audit openings",
    )
    p_smoke.add_argument(
        "--proof-gguf-manifest",
        default="",
        help="GGUF tensor manifest JSON used to attach selected weight-byte openings",
    )
    p_smoke.add_argument("--no-require-proof", action="store_true", help="Run the mesh leg without proof capture")
    p_smoke.add_argument("--no-baseline", action="store_true", help="Skip direct llama.cpp RPC baseline")
    p_smoke.add_argument("--hf-home", default="", help="Optional Hugging Face cache directory")
    p_smoke.add_argument("--output-dir", default="", help="Smoke state/output root; defaults to a temp dir")
    p_smoke.add_argument("--result-json", default="", help="Write the full smoke result JSON to this path")
    p_smoke.add_argument("--summary-only", action="store_true", help="Print aggregate smoke results without per-sample lists")
    p_smoke.add_argument("--timeout", type=float, default=300.0)
    p_smoke.set_defaults(func=cmd_smoke)

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    # Library modules log (never print); the entry point decides visibility.
    # INFO keeps the operational telemetry (POOLCHAT-TIMING etc.) in the
    # worker logs that diagnostics grep for.
    # force=True: anything imported earlier (bittensor!) may have already
    # installed handlers, which would silently swallow INFO telemetry like
    # POOLCHAT-TIMING.
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    # `kill -USR1 <pid>` dumps every thread's stack to stderr (the PM2
    # error log) for ANY mesh process. The glm stage-proof server (mesh
    # serve on the proof port) has wedged with one thread grinding a
    # GIL-holding native computation for 20+ min while 250+ threads
    # futex-waited (accept queue full, every chat/probe timing out), and
    # rented containers ship without SYS_PTRACE so py-spy cannot attach:
    # this hook is the only way to see the grinder's Python frame.
    try:
        import faulthandler
        import signal as _signal

        faulthandler.register(_signal.SIGUSR1, all_threads=True)
    except (ImportError, AttributeError, ValueError):
        pass
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except KeyboardInterrupt:
        # The normal way out of a log tail or the chat REPL; never a bug.
        print()
        raise SystemExit(130)
    except EOFError:
        print()
        raise SystemExit(1)
    except (RuntimeError, ValueError, OSError) as exc:
        # Operator-facing failures (manager down, wrong token, missing
        # file, HTTP 4xx/5xx) end as one readable line, not a stack trace.
        raise SystemExit(f"error: {exc}")


if __name__ == "__main__":
    main()
