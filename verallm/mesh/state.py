"""Persistent state and join-token helpers for private Verathos meshes."""

from __future__ import annotations

import base64
import json
import secrets
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

from verallm.mesh.llama_cpp import (
    llama_tensor_split_layer_ranges,
    normalize_rpc_endpoint,
)
from verallm.mesh.private_files import write_owner_only_json, write_owner_only_text
from verallm.mesh.types import CapabilityAd, MeshMember, MeshSpec, StageRange, canonical_json_bytes, save_json
from verallm.mesh.worker import normalize_endpoint, post_json


JOIN_TOKEN_PREFIX = "vtmesh_"
MESH_SPEC_FILE = "mesh.json"
MESH_STATE_FILE = "mesh-state.json"
JOIN_TOKEN_FILE = "join-token.txt"


@dataclass(frozen=True)
class MeshJoinToken:
    """Private token used by workers to join one coordinator mesh."""

    mesh_id: str
    coordinator_endpoint: str
    join_secret: str
    coordinator_uid: int
    coordinator_hotkey: str
    model_id: str
    version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": int(self.version),
            "mesh_id": self.mesh_id,
            "coordinator_endpoint": self.coordinator_endpoint,
            "join_secret": self.join_secret,
            "coordinator_uid": int(self.coordinator_uid),
            "coordinator_hotkey": self.coordinator_hotkey,
            "model_id": self.model_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MeshJoinToken":
        token = cls(
            version=int(data.get("version", 1)),
            mesh_id=str(data["mesh_id"]),
            coordinator_endpoint=normalize_endpoint(str(data["coordinator_endpoint"])).rstrip("/"),
            join_secret=str(data["join_secret"]),
            coordinator_uid=int(data["coordinator_uid"]),
            coordinator_hotkey=str(data["coordinator_hotkey"]),
            model_id=str(data["model_id"]),
        )
        if token.version != 1:
            raise ValueError("unsupported mesh join token version")
        if not token.mesh_id:
            raise ValueError("mesh_id is required")
        if not token.join_secret:
            raise ValueError("join_secret is required")
        return token

    def encode(self) -> str:
        body = base64.urlsafe_b64encode(canonical_json_bytes(self.to_dict()))
        return JOIN_TOKEN_PREFIX + body.decode("ascii").rstrip("=")

    @classmethod
    def decode(cls, raw: str) -> "MeshJoinToken":
        if not raw.startswith(JOIN_TOKEN_PREFIX):
            raise ValueError(f"join token must start with {JOIN_TOKEN_PREFIX}")
        body = raw[len(JOIN_TOKEN_PREFIX):]
        body += "=" * (-len(body) % 4)
        data = json.loads(base64.urlsafe_b64decode(body.encode("ascii")).decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError("join token payload must be a JSON object")
        return cls.from_dict(data)


def default_mesh_root() -> Path:
    return Path.cwd() / ".verathos" / "meshes"


def mesh_dir(root: str | Path | None, mesh_id: str) -> Path:
    return (Path(root) if root else default_mesh_root()) / mesh_id


def mesh_spec_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_dir() or p.suffix == "":
        return p / MESH_SPEC_FILE
    return p


def mesh_state_path(path: str | Path) -> Path:
    p = Path(path)
    if p.name == MESH_STATE_FILE:
        return p
    if p.name == MESH_SPEC_FILE:
        return p.with_name(MESH_STATE_FILE)
    return p / MESH_STATE_FILE


def load_mesh_state(path: str | Path) -> dict[str, Any]:
    p = mesh_state_path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("mesh state must be a JSON object")
    if int(data.get("version", 0)) != 1:
        raise ValueError("unsupported mesh state version")
    return data


def save_mesh_state(path: str | Path, state: dict[str, Any]) -> Path:
    return write_owner_only_json(mesh_state_path(path), state)


def state_mesh_spec(state: dict[str, Any]) -> MeshSpec:
    return MeshSpec.from_dict(state["mesh"])


def state_capabilities(state: dict[str, Any]) -> list[CapabilityAd]:
    return [CapabilityAd.from_dict(item) for item in state.get("capabilities", [])]


def state_admitted_compute_stage_count(state: dict[str, Any]) -> int:
    """Count compute capabilities admitted beyond the placeholder spec.

    A new all-RPC coordinator state initially contains one coordinator
    capability and a temporary non-empty coordinator range so the MeshSpec is
    valid before any worker joins.  That placeholder is not an admitted RPC
    compute stage and must not consume ``expected_compute_stage_count``.
    """

    capabilities = state_capabilities(state)
    if bool(state.get("coordinator_computes", True)):
        return len(capabilities)
    return max(0, len(capabilities) - 1)


def state_internal_auth_secret(state: dict[str, Any]) -> str:
    """Return the private HMAC secret for coordinator/worker control calls.

    Coordinator state owns the join secret directly.  Worker state keeps the
    same secret only inside its private join token, so there is no second copy
    to drift or accidentally serialize into a mesh spec.
    """

    role = str(state.get("role", ""))
    if role == "coordinator":
        secret = str(state.get("join_secret", ""))
    elif role == "worker":
        raw_token = str(state.get("join_token", ""))
        secret = MeshJoinToken.decode(raw_token).join_secret if raw_token else ""
    else:
        raise ValueError("mesh state role must be coordinator or worker")
    if not secret:
        raise ValueError("mesh state does not contain an internal auth secret")
    return secret


def _coordinator_capability(spec: MeshSpec) -> CapabilityAd:
    coordinator = sorted(spec.members, key=lambda item: item.stage_index)[0]
    return CapabilityAd(
        uid=spec.coordinator_uid,
        hotkey=spec.coordinator_hotkey,
        endpoint=coordinator.endpoint,
        supported_backends=["gguf_stage"],
        proof_modes=["verathos-gemv1"],
        cached_model_package_hashes=[spec.model_package_hash],
    )


def _upsert_capability(
    capabilities: list[CapabilityAd],
    capability: CapabilityAd,
) -> list[CapabilityAd]:
    result: list[CapabilityAd] = []
    replaced = False
    for existing in capabilities:
        if existing.endpoint == capability.endpoint:
            result.append(capability)
            replaced = True
        else:
            result.append(existing)
    if not replaced:
        result.append(capability)
    return result


def assign_mesh_members(
    spec: MeshSpec,
    capabilities: list[CapabilityAd],
    *,
    coordinator_computes: bool = True,
    require_positive_vram: bool = False,
) -> MeshSpec:
    """Assign contiguous layer ranges in capability order.

    With ``coordinator_computes=False`` (a coordinator that offloads all
    compute to workers, e.g. ``--llama-device RPC0,RPC1``) the coordinator
    becomes an orchestration-only stage with an empty layer range and the
    workers tile every layer between them.  Ranges use llama.cpp's ordered
    tensor-split placement. Missing/zero VRAM becomes an explicit equal split
    in development mode; validator mode rejects it.
    """

    if not capabilities:
        raise ValueError("mesh requires at least one node")
    stageless_coordinator = not coordinator_computes and len(capabilities) > 1
    compute_count = len(capabilities) - 1 if stageless_coordinator else len(capabilities)

    compute_capabilities = (
        capabilities[1:] if stageless_coordinator else capabilities
    )
    # Per-DEVICE weights: a multi-GPU worker's rpc-server exposes one llama
    # device per GPU, so the tensor split runs over devices, not members. A
    # member's committed range is the union of its devices' contiguous
    # ranges. Legacy single-GPU members are one device worth vram_gb.
    advertised_device_weights = [
        [int(v) for v in capability.per_gpu_vram_gb]
        if capability.per_gpu_vram_gb
        else [int(capability.vram_gb)]
        for capability in compute_capabilities
    ]
    device_count = sum(len(weights) for weights in advertised_device_weights)
    if device_count > spec.total_layers:
        raise ValueError("more mesh devices than model layers; cannot assign non-empty ranges")
    all_positive = bool(advertised_device_weights) and all(
        weight > 0 for weights in advertised_device_weights for weight in weights
    )
    if require_positive_vram and not all_positive:
        raise ValueError(
            "validator mesh compute stages require positive vram_gb"
        )
    member_device_weights = (
        advertised_device_weights
        if all_positive
        else [[1] * len(weights) for weights in advertised_device_weights]
    )
    device_ranges = llama_tensor_split_layer_ranges(
        spec.total_layers,
        [weight for weights in member_device_weights for weight in weights],
    )
    compute_ranges: list[StageRange] = []
    device_pos = 0
    for weights in member_device_weights:
        first = device_ranges[device_pos]
        last = device_ranges[device_pos + len(weights) - 1]
        compute_ranges.append(StageRange(first.start, last.end))
        device_pos += len(weights)

    members: list[MeshMember] = []
    compute_pos = 0
    for idx, capability in enumerate(capabilities):
        is_coordinator = idx == 0
        device_weights: list[int] = []
        if stageless_coordinator and is_coordinator:
            layers = StageRange(0, 0)
            rpc_split_weight = 0
        else:
            layers = compute_ranges[compute_pos]
            if is_coordinator or not capability.rpc_endpoint:
                rpc_split_weight = 0
            else:
                weights = member_device_weights[compute_pos]
                rpc_split_weight = sum(weights)
                device_weights = weights if len(weights) > 1 else []
            compute_pos += 1
        backend = "gguf_stage" if is_coordinator else "gguf_stage_worker"
        if backend not in capability.supported_backends:
            backend = capability.supported_backends[0]
        members.append(
            MeshMember(
                uid=capability.uid,
                hotkey=capability.hotkey,
                endpoint=capability.endpoint,
                stage_index=idx,
                layers=layers,
                role="coordinator" if is_coordinator else "worker",
                backend=backend,
                # Private pool workers advertise their dedicated Sr25519
                # stage identity as the capability hotkey. Keep the explicit
                # proof-key field pinned to that same public identity so a
                # verification snapshot never has to infer a signer from a
                # coordinator/miner key.
                proof_key=capability.hotkey if not is_coordinator else "",
                rpc_endpoint=capability.rpc_endpoint,
                proof_endpoint=capability.proof_endpoint,
                rpc_split_weight=rpc_split_weight,
                rpc_device_weights=device_weights,
                payout_bps=10000 if is_coordinator else 0,
                capability_hash=capability.ad_hash_hex(),
            )
        )

    updated = MeshSpec(
        mesh_id=spec.mesh_id,
        mode=spec.mode,
        coordinator_uid=spec.coordinator_uid,
        coordinator_hotkey=spec.coordinator_hotkey,
        model_id=spec.model_id,
        model_package_ref=spec.model_package_ref,
        model_package_hash=spec.model_package_hash,
        model_tensor_manifest_root=spec.model_tensor_manifest_root,
        tokenizer_hash=spec.tokenizer_hash,
        quantization_scheme=spec.quantization_scheme,
        activation_dtype=spec.activation_dtype,
        max_context_len=spec.max_context_len,
        proof_trace_manifest_format=spec.proof_trace_manifest_format,
        total_layers=spec.total_layers,
        members=members,
        epoch=spec.epoch,
        expires_at_unix=spec.expires_at_unix,
        created_at_unix=spec.created_at_unix,
        signatures={},
    )
    updated.validate()
    return updated


def create_mesh_state(
    *,
    spec: MeshSpec,
    root: str | Path | None = None,
) -> tuple[Path, dict[str, Any], MeshJoinToken]:
    """Persist a coordinator mesh and return its join token."""

    capability = _coordinator_capability(spec)
    token = MeshJoinToken(
        mesh_id=spec.mesh_id,
        coordinator_endpoint=capability.endpoint,
        join_secret=secrets.token_urlsafe(32),
        coordinator_uid=spec.coordinator_uid,
        coordinator_hotkey=spec.coordinator_hotkey,
        model_id=spec.model_id,
    )
    state = {
        "version": 1,
        "role": "coordinator",
        "created_at_unix": int(time.time()),
        "join_secret": token.join_secret,
        "join_token": token.encode(),
        "mesh": spec.to_dict(),
        "capabilities": [capability.to_dict()],
    }
    out_dir = mesh_dir(root, spec.mesh_id)
    save_json(out_dir / MESH_SPEC_FILE, spec.to_dict())
    save_mesh_state(out_dir, state)
    write_owner_only_text(out_dir / JOIN_TOKEN_FILE, token.encode() + "\n")
    return out_dir, state, token


def admit_mesh_worker(state_dir: str | Path, capability: CapabilityAd) -> MeshSpec:
    """Add or update a worker capability in coordinator state."""

    state = load_mesh_state(state_dir)
    if state.get("role") != "coordinator":
        raise ValueError("only coordinator state can admit mesh workers")
    capabilities = _upsert_capability(state_capabilities(state), capability)
    spec = assign_mesh_members(
        state_mesh_spec(state),
        capabilities,
        coordinator_computes=bool(state.get("coordinator_computes", True)),
        # Subnet-mode meshes must never admit a zero-VRAM stage. The pool
        # manager writes the normalized "subnet" since the mode rename;
        # "validator" covers state dirs minted before it (pool.py owns the
        # canonical predicate but importing it here would be circular).
        require_positive_vram=(
            str(state.get("serving_mode", "")) in ("subnet", "validator")
        ),
    )
    state["mesh"] = spec.to_dict()
    state["capabilities"] = [item.to_dict() for item in capabilities]
    state["updated_at_unix"] = int(time.time())
    save_mesh_state(state_dir, state)
    save_json(mesh_state_path(state_dir).with_name(MESH_SPEC_FILE), spec.to_dict())
    return spec


def update_worker_mesh_state(state_dir: str | Path, spec: MeshSpec) -> MeshSpec:
    """Persist a coordinator-pushed mesh spec update on a worker."""

    state = load_mesh_state(state_dir)
    if state.get("role") != "worker":
        raise ValueError("only worker state can accept mesh updates")
    current = state_mesh_spec(state)
    if spec.mesh_id != current.mesh_id:
        raise ValueError("mesh update mesh_id mismatch")
    if spec.coordinator_uid != current.coordinator_uid:
        raise ValueError("mesh update coordinator_uid mismatch")
    if spec.coordinator_hotkey != current.coordinator_hotkey:
        raise ValueError("mesh update coordinator_hotkey mismatch")
    state["mesh"] = spec.to_dict()
    state["updated_at_unix"] = int(time.time())
    save_mesh_state(state_dir, state)
    save_json(mesh_state_path(state_dir).with_name(MESH_SPEC_FILE), spec.to_dict())
    return spec


def fetch_coordinator_mesh_spec(token: str, *, timeout: float = 5.0) -> MeshSpec:
    """Fetch the current mesh spec from a coordinator using a private join token."""

    join_token = MeshJoinToken.decode(token)
    response = post_json(
        urljoin(normalize_endpoint(join_token.coordinator_endpoint), "v1/mesh/spec"),
        {
            "join_secret": join_token.join_secret,
            "mesh_id": join_token.mesh_id,
        },
        timeout=timeout,
        internal_auth_secret=join_token.join_secret,
    )
    spec = MeshSpec.from_dict(response["mesh"])
    if spec.mesh_id != join_token.mesh_id:
        raise ValueError("coordinator returned a different mesh_id")
    return spec


def refresh_worker_mesh_state(state_dir: str | Path, *, timeout: float = 5.0) -> MeshSpec:
    """Pull and persist the latest mesh spec for a worker state directory."""

    state = load_mesh_state(state_dir)
    if state.get("role") != "worker":
        raise ValueError("only worker state can refresh from a coordinator")
    token = str(state.get("join_token", ""))
    if not token:
        raise ValueError("worker state does not contain a join token")
    return update_worker_mesh_state(
        state_dir,
        fetch_coordinator_mesh_spec(token, timeout=timeout),
    )


def join_mesh(
    *,
    token: str,
    endpoint: str,
    root: str | Path | None = None,
    uid: int | None = None,
    hotkey: str | None = None,
    backend: str = "gguf_stage_worker",
    package_hash: str = "",
    gpu_name: str = "",
    vram_gb: int = 0,
    per_gpu_vram_gb: list[int] | None = None,
    rpc_endpoint: str = "",
    proof_endpoint: str = "",
    timeout: float = 5.0,
    self_advertise_host: str = "",
) -> tuple[Path, MeshSpec]:
    """Join a coordinator and persist local worker state.

    ``self_advertise_host``: this worker's own public advertise host. When
    the coordinator's advertised endpoint lives on the SAME host, the join
    dials the loopback twin instead: many provider NATs cannot hairpin a
    box to its own public IP, and the driver-side member join then times
    out against an endpoint every other machine can reach.
    Only the dial URL changes; the spec and capability keep the
    advertised address.
    """

    join_token = MeshJoinToken.decode(token)
    rpc_endpoint = normalize_rpc_endpoint(rpc_endpoint) if rpc_endpoint else ""
    proof_endpoint = normalize_endpoint(proof_endpoint or endpoint).rstrip("/") if rpc_endpoint else (
        normalize_endpoint(proof_endpoint).rstrip("/") if proof_endpoint else ""
    )
    supported_backends = [backend]
    if rpc_endpoint and "llama_cpp_rpc" not in supported_backends:
        supported_backends.append("llama_cpp_rpc")
    capability = CapabilityAd(
        uid=join_token.coordinator_uid if uid is None else uid,
        hotkey=join_token.coordinator_hotkey if hotkey is None else hotkey,
        endpoint=normalize_endpoint(endpoint).rstrip("/"),
        supported_backends=supported_backends,
        proof_modes=["verathos-gemv1"],
        cached_model_package_hashes=[package_hash] if package_hash else [],
        gpu_name=gpu_name,
        vram_gb=vram_gb,
        per_gpu_vram_gb=(
            [int(v) for v in per_gpu_vram_gb]
            if per_gpu_vram_gb and len(per_gpu_vram_gb) > 1
            else []
        ),
        rpc_endpoint=rpc_endpoint,
        proof_endpoint=proof_endpoint,
    )
    coordinator_dial = normalize_endpoint(join_token.coordinator_endpoint)
    if self_advertise_host:
        parsed = urlparse(coordinator_dial.rstrip("/"))
        if parsed.hostname and parsed.hostname == str(self_advertise_host):
            coordinator_dial = (
                f"{parsed.scheme or 'http'}://127.0.0.1:{parsed.port or 80}/"
            )
    response: dict[str, Any] = {}
    update_errors: list[Any] = []
    for attempt in range(3):
        response = post_json(
            urljoin(
                coordinator_dial,
                "v1/mesh/join",
            ),
            {
                "join_secret": join_token.join_secret,
                "capability": capability.to_dict(),
            },
            timeout=timeout,
            internal_auth_secret=join_token.join_secret,
        )
        update_errors = list(response.get("mesh_update_errors") or [])
        if not update_errors:
            break
        if attempt < 2:
            time.sleep(0.25)
    if update_errors:
        raise RuntimeError(
            "coordinator could not update "
            f"{len(update_errors)} existing mesh member(s)"
        )
    spec = MeshSpec.from_dict(response["mesh"])
    out_dir = mesh_dir(root, spec.mesh_id)
    state = {
        "version": 1,
        "role": "worker",
        "joined_at_unix": int(time.time()),
        "join_token": token,
        "coordinator_endpoint": join_token.coordinator_endpoint,
        "mesh": spec.to_dict(),
        "capabilities": [capability.to_dict()],
    }
    save_json(out_dir / MESH_SPEC_FILE, spec.to_dict())
    save_mesh_state(out_dir, state)
    return out_dir, spec
