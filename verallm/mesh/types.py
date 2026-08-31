"""Canonical manifests for verified GGUF meshes.

These objects intentionally contain no runtime networking logic. They are the
stable, hashable protocol surface used by discovery, commitments, and future
mesh signing.
"""

from __future__ import annotations

import hashlib
import json
import platform
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping


SUPPORTED_BACKENDS = {"vllm", "gguf", "gguf_stage", "gguf_stage_worker", "llama_cpp_rpc"}
SUPPORTED_MESH_MODES = {"private", "declared", "open"}
SUPPORTED_ACTIVATION_DTYPES = {"f16", "bf16", "f32", "q8"}
SUPPORTED_TRACE_MANIFEST_FORMATS = {
    "none",
    "compact",
    "compact-raw",
    "compact-raw-v2",
    "compact-raw-v3",
    "decode",
}


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    """Serialize JSON deterministically for hashing and signing."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def _sha256_tagged(tag: bytes, *parts: bytes) -> bytes:
    h = hashlib.sha256(tag)
    for part in parts:
        h.update(part)
    return h.digest()


def _now_unix() -> int:
    return int(time.time())


def _is_hex_digest(value: str, *, bytes_len: int = 32) -> bool:
    if len(value) != bytes_len * 2:
        return False
    try:
        bytes.fromhex(value)
    except ValueError:
        return False
    return True


def _require_hex_digest(name: str, value: str, *, allow_empty: bool = False) -> None:
    if allow_empty and not value:
        return
    if not _is_hex_digest(value):
        raise ValueError(f"{name} must be a {32 * 2}-character hex SHA-256 digest")


@dataclass(frozen=True)
class StageRange:
    """Half-open transformer layer range owned by one stage."""

    start: int
    end: int

    def validate(
        self, total_layers: int | None = None, *, allow_empty: bool = False
    ) -> None:
        if self.start < 0:
            raise ValueError("stage range start must be >= 0")
        if self.end == self.start and allow_empty:
            # An orchestration-only stage (coordinator that offloads all
            # compute to workers) owns no layers.
            return
        if self.end <= self.start:
            raise ValueError("stage range end must be greater than start")
        if total_layers is not None and self.end > total_layers:
            raise ValueError("stage range exceeds total layer count")

    def to_dict(self) -> dict[str, int]:
        return {"start": int(self.start), "end": int(self.end)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "StageRange":
        return cls(start=int(data["start"]), end=int(data["end"]))


@dataclass
class MeshMember:
    """One UID or private worker participating in a mesh."""

    uid: int
    hotkey: str
    endpoint: str
    stage_index: int
    layers: StageRange
    role: str = "worker"
    backend: str = "gguf_stage_worker"
    proof_key: str = ""
    proof_endpoint: str = ""
    rpc_endpoint: str = ""
    rpc_split_weight: int = 0
    # Per-GPU split weights for a member whose rpc-server exposes several
    # devices. Empty = single device weighing rpc_split_weight (legacy).
    # When set, rpc_split_weight must equal the sum, and the member's layer
    # range is the union of its devices' contiguous tensor-split ranges.
    rpc_device_weights: list[int] = field(default_factory=list)
    payout_bps: int = 0
    capability_hash: str = ""

    def validate(self, total_layers: int) -> None:
        if self.uid < 0:
            raise ValueError("member uid must be >= 0")
        if not self.hotkey:
            raise ValueError("member hotkey is required")
        if not self.endpoint:
            raise ValueError("member endpoint is required")
        if self.stage_index < 0:
            raise ValueError("member stage_index must be >= 0")
        if self.backend not in SUPPORTED_BACKENDS:
            raise ValueError(f"unsupported member backend: {self.backend}")
        if isinstance(self.rpc_split_weight, bool) or not isinstance(
            self.rpc_split_weight, int
        ):
            raise ValueError("member rpc_split_weight must be an integer")
        if self.rpc_split_weight < 0:
            raise ValueError("member rpc_split_weight must be >= 0")
        if self.role == "coordinator" and self.rpc_split_weight != 0:
            raise ValueError("coordinator rpc_split_weight must be zero")
        if not self.rpc_endpoint and self.rpc_split_weight != 0:
            raise ValueError(
                "member rpc_split_weight must be zero without an rpc_endpoint"
            )
        if self.rpc_device_weights:
            if any(
                isinstance(weight, bool)
                or not isinstance(weight, int)
                or weight <= 0
                for weight in self.rpc_device_weights
            ):
                raise ValueError(
                    "member rpc_device_weights must be positive integers"
                )
            if sum(self.rpc_device_weights) != self.rpc_split_weight:
                raise ValueError(
                    "member rpc_device_weights must sum to rpc_split_weight"
                )
        if not 0 <= self.payout_bps <= 10000:
            raise ValueError("member payout_bps must be in [0, 10000]")
        _require_hex_digest("member capability_hash", self.capability_hash, allow_empty=True)
        # Only a coordinator may be stage-less: it orchestrates (and may
        # offload all compute to workers), so its layer range may be empty.
        self.layers.validate(total_layers, allow_empty=self.role == "coordinator")

    def to_dict(self) -> dict[str, Any]:
        return {
            "uid": int(self.uid),
            "hotkey": self.hotkey,
            "endpoint": self.endpoint,
            "stage_index": int(self.stage_index),
            "layers": self.layers.to_dict(),
            "role": self.role,
            "backend": self.backend,
            "proof_key": self.proof_key,
            "proof_endpoint": self.proof_endpoint,
            "rpc_endpoint": self.rpc_endpoint,
            "rpc_split_weight": int(self.rpc_split_weight),
            # Serialized only when set so single-device members keep their
            # historical dict shape (and hashes) byte-for-byte.
            **(
                {"rpc_device_weights": [int(w) for w in self.rpc_device_weights]}
                if self.rpc_device_weights
                else {}
            ),
            "payout_bps": int(self.payout_bps),
            "capability_hash": self.capability_hash,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MeshMember":
        return cls(
            uid=int(data["uid"]),
            hotkey=str(data["hotkey"]),
            endpoint=str(data["endpoint"]),
            stage_index=int(data["stage_index"]),
            layers=StageRange.from_dict(data["layers"]),
            role=str(data.get("role", "worker")),
            backend=str(data.get("backend", "gguf_stage_worker")),
            proof_key=str(data.get("proof_key", "")),
            proof_endpoint=str(data.get("proof_endpoint", "")),
            rpc_endpoint=str(data.get("rpc_endpoint", "")),
            rpc_split_weight=int(data.get("rpc_split_weight", 0)),
            rpc_device_weights=[
                int(w) for w in data.get("rpc_device_weights", [])
            ],
            payout_bps=int(data.get("payout_bps", 0)),
            capability_hash=str(data.get("capability_hash", "")),
        )


@dataclass
class MeshSpec:
    """Canonical mesh manifest for staged GGUF inference."""

    mesh_id: str
    mode: str
    coordinator_uid: int
    coordinator_hotkey: str
    model_id: str
    model_package_hash: str
    total_layers: int
    members: list[MeshMember]
    model_package_ref: str = ""
    model_tensor_manifest_root: str = ""
    tokenizer_hash: str = ""
    quantization_scheme: str = "unknown"
    activation_dtype: str = "f16"
    # Per-coordinator MinerRegistry limit. Zero is retained only for legacy
    # and dev manifests; validator snapshots require a positive uint32 value.
    max_context_len: int = 0
    # Proof-capture manifest format pinned by the coordinator so every member
    # runs the same profile. A mixed mesh (one member writing op-manifest
    # rows, another not) fails aggregate verification. Empty = unpinned
    # (each member derives it locally; pre-pinning behavior).
    proof_trace_manifest_format: str = ""
    version: int = 1
    epoch: int = 0
    expires_at_unix: int = 0
    created_at_unix: int = field(default_factory=_now_unix)
    signatures: dict[str, str] = field(default_factory=dict)

    @classmethod
    def new_private_mesh(
        cls,
        *,
        coordinator_uid: int,
        coordinator_hotkey: str,
        endpoint: str,
        model_id: str,
        model_package_hash: str,
        total_layers: int,
        model_package_ref: str = "",
        model_tensor_manifest_root: str = "",
        tokenizer_hash: str = "",
        quantization_scheme: str = "unknown",
        activation_dtype: str = "f16",
        max_context_len: int = 0,
        epoch: int = 0,
        expires_at_unix: int = 0,
        mesh_id: str = "",
    ) -> "MeshSpec":
        # Mesh identity is logical when the caller says so: a relaunch of
        # the same registration passes the previous mesh_id, so everything
        # keyed by it (mesh state dir, verification snapshot chain, opaque
        # stage ids) resumes instead of presenting the identical model as
        # new verification terms. A fresh mesh keeps the nonce-derived id.
        if mesh_id:
            if not re.fullmatch(r"mesh-[0-9a-f]{16}", mesh_id):
                raise ValueError("resumed mesh_id must be mesh-<16 hex>")
        else:
            seed = canonical_json_bytes(
                {
                    "uid": coordinator_uid,
                    "hotkey": coordinator_hotkey,
                    "endpoint": endpoint,
                    "model_id": model_id,
                    "model_package_hash": model_package_hash,
                    "nonce": uuid.uuid4().hex,
                }
            )
            mesh_id = "mesh-" + hashlib.sha256(seed).hexdigest()[:16]
        member = MeshMember(
            uid=coordinator_uid,
            hotkey=coordinator_hotkey,
            endpoint=endpoint,
            stage_index=0,
            layers=StageRange(0, total_layers),
            role="coordinator",
            backend="gguf_stage",
            payout_bps=10000,
        )
        spec = cls(
            mesh_id=mesh_id,
            mode="private",
            coordinator_uid=coordinator_uid,
            coordinator_hotkey=coordinator_hotkey,
            model_id=model_id,
            model_package_ref=model_package_ref,
            model_package_hash=model_package_hash,
            model_tensor_manifest_root=model_tensor_manifest_root,
            tokenizer_hash=tokenizer_hash,
            quantization_scheme=quantization_scheme,
            activation_dtype=activation_dtype,
            max_context_len=max_context_len,
            total_layers=total_layers,
            members=[member],
            epoch=epoch,
            expires_at_unix=expires_at_unix,
        )
        spec.validate()
        return spec

    def validate(self) -> None:
        if self.version != 1:
            raise ValueError("unsupported MeshSpec version")
        if not self.mesh_id:
            raise ValueError("mesh_id is required")
        if self.mode not in SUPPORTED_MESH_MODES:
            raise ValueError(f"unsupported mesh mode: {self.mode}")
        if self.coordinator_uid < 0:
            raise ValueError("coordinator_uid must be >= 0")
        if not self.coordinator_hotkey:
            raise ValueError("coordinator_hotkey is required")
        if not self.model_id:
            raise ValueError("model_id is required")
        if self.total_layers <= 0:
            raise ValueError("total_layers must be > 0")
        if (
            type(self.max_context_len) is not int
            or not 0 <= self.max_context_len < 2**32
        ):
            raise ValueError("max_context_len must fit uint32")
        if self.activation_dtype not in SUPPORTED_ACTIVATION_DTYPES:
            raise ValueError(f"unsupported activation dtype: {self.activation_dtype}")
        if self.proof_trace_manifest_format and (
            self.proof_trace_manifest_format not in SUPPORTED_TRACE_MANIFEST_FORMATS
        ):
            raise ValueError(
                "unsupported proof_trace_manifest_format: "
                f"{self.proof_trace_manifest_format}"
            )
        _require_hex_digest("model_package_hash", self.model_package_hash)
        _require_hex_digest(
            "model_tensor_manifest_root",
            self.model_tensor_manifest_root,
            allow_empty=True,
        )
        _require_hex_digest("tokenizer_hash", self.tokenizer_hash, allow_empty=True)
        if not self.members:
            raise ValueError("mesh must contain at least one member")

        payout_sum = 0
        coordinator_seen = False
        stage_indexes = set()
        for member in self.members:
            member.validate(self.total_layers)
            if member.stage_index == 0 and member.rpc_split_weight != 0:
                raise ValueError("coordinator rpc_split_weight must be zero")
            if member.stage_index in stage_indexes:
                raise ValueError("member stage_index values must be unique")
            stage_indexes.add(member.stage_index)
            payout_sum += member.payout_bps
            if (
                member.uid == self.coordinator_uid
                and member.hotkey == self.coordinator_hotkey
            ):
                coordinator_seen = True

        if not coordinator_seen:
            raise ValueError("coordinator must appear in members")
        if payout_sum not in (0, 10000):
            raise ValueError("member payout_bps must sum to 10000 or all be zero")
        if stage_indexes != set(range(len(self.members))):
            raise ValueError("member stage_index values must be contiguous from zero")

        # Compute stages (non-empty ranges) must tile [0, total_layers); an
        # orchestration-only coordinator (empty range) is excluded.
        ranges = sorted(
            (m.layers.start, m.layers.end)
            for m in self.members
            if m.layers.end > m.layers.start
        )
        if not ranges:
            raise ValueError("mesh must contain at least one compute stage")
        expected_start = 0
        for start, end in ranges:
            if start != expected_start:
                raise ValueError("member layer ranges must cover [0, total_layers) without gaps")
            expected_start = end
        if expected_start != self.total_layers:
            raise ValueError("member layer ranges must cover every layer")

    def to_dict(self, *, include_signatures: bool = True) -> dict[str, Any]:
        data = {
            "version": int(self.version),
            "mesh_id": self.mesh_id,
            "mode": self.mode,
            "coordinator_uid": int(self.coordinator_uid),
            "coordinator_hotkey": self.coordinator_hotkey,
            "model_id": self.model_id,
            "model_package_ref": self.model_package_ref,
            "model_package_hash": self.model_package_hash,
            "model_tensor_manifest_root": self.model_tensor_manifest_root,
            "tokenizer_hash": self.tokenizer_hash,
            "quantization_scheme": self.quantization_scheme,
            "activation_dtype": self.activation_dtype,
            "total_layers": int(self.total_layers),
            "members": [member.to_dict() for member in self.members],
            "epoch": int(self.epoch),
            "expires_at_unix": int(self.expires_at_unix),
            "created_at_unix": int(self.created_at_unix),
        }
        # Only serialized when pinned so pre-pinning spec hashes stay stable.
        if self.proof_trace_manifest_format:
            data["proof_trace_manifest_format"] = self.proof_trace_manifest_format
        if self.max_context_len:
            data["max_context_len"] = int(self.max_context_len)
        if include_signatures:
            data["signatures"] = dict(sorted(self.signatures.items()))
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MeshSpec":
        raw_max_context_len = data.get("max_context_len", 0)
        if type(raw_max_context_len) is not int:
            raise ValueError("max_context_len must be an integer")
        spec = cls(
            version=int(data.get("version", 1)),
            mesh_id=str(data["mesh_id"]),
            mode=str(data["mode"]),
            coordinator_uid=int(data["coordinator_uid"]),
            coordinator_hotkey=str(data["coordinator_hotkey"]),
            model_id=str(data["model_id"]),
            model_package_ref=str(data.get("model_package_ref", "")),
            model_package_hash=str(data["model_package_hash"]),
            model_tensor_manifest_root=str(data.get("model_tensor_manifest_root", "")),
            tokenizer_hash=str(data.get("tokenizer_hash", "")),
            quantization_scheme=str(data.get("quantization_scheme", "unknown")),
            activation_dtype=str(data.get("activation_dtype", "f16")),
            max_context_len=raw_max_context_len,
            proof_trace_manifest_format=str(
                data.get("proof_trace_manifest_format", "")
            ),
            total_layers=int(data["total_layers"]),
            members=[MeshMember.from_dict(item) for item in data["members"]],
            epoch=int(data.get("epoch", 0)),
            expires_at_unix=int(data.get("expires_at_unix", 0)),
            created_at_unix=int(data.get("created_at_unix", 0) or _now_unix()),
            signatures=dict(data.get("signatures", {})),
        )
        spec.validate()
        return spec

    def body_bytes(self) -> bytes:
        return b"VERATHOS_MESH_SPEC_BODY_V1" + canonical_json_bytes(
            self.to_dict(include_signatures=False)
        )

    def body_hash(self) -> bytes:
        return hashlib.sha256(self.body_bytes()).digest()

    def body_hash_hex(self) -> str:
        return self.body_hash().hex()

    def spec_hash(self) -> bytes:
        return _sha256_tagged(
            b"VERATHOS_MESH_SPEC_V1",
            canonical_json_bytes(self.to_dict(include_signatures=True)),
        )

    def spec_hash_hex(self) -> str:
        return self.spec_hash().hex()

    def stage_assignment_hash(self) -> bytes:
        assignments = [
            {
                "uid": member.uid,
                "hotkey": member.hotkey,
                "endpoint": member.endpoint,
                "stage_index": member.stage_index,
                "layers": member.layers.to_dict(),
                "backend": member.backend,
                "rpc_endpoint": member.rpc_endpoint,
                "rpc_split_weight": member.rpc_split_weight,
                **(
                    {
                        "rpc_device_weights": [
                            int(w) for w in member.rpc_device_weights
                        ]
                    }
                    if member.rpc_device_weights
                    else {}
                ),
                "proof_endpoint": member.proof_endpoint,
            }
            for member in sorted(self.members, key=lambda item: item.stage_index)
        ]
        return _sha256_tagged(
            b"VERATHOS_STAGE_ASSIGNMENT_V1",
            canonical_json_bytes(
                {
                    "mesh_id": self.mesh_id,
                    "model_package_hash": self.model_package_hash,
                    "total_layers": self.total_layers,
                    "assignments": assignments,
                    **(
                        {"max_context_len": int(self.max_context_len)}
                        if self.max_context_len
                        else {}
                    ),
                }
            ),
        )

    def stage_assignment_hash_hex(self) -> str:
        return self.stage_assignment_hash().hex()


@dataclass
class CapabilityAd:
    """Signed capability advertisement for mesh discovery."""

    uid: int
    hotkey: str
    endpoint: str
    supported_backends: list[str] = field(default_factory=lambda: ["gguf_stage_worker"])
    proof_modes: list[str] = field(default_factory=lambda: ["verathos-gemv1"])
    cached_model_package_hashes: list[str] = field(default_factory=list)
    platform_os: str = field(default_factory=lambda: platform.system().lower())
    platform_arch: str = field(default_factory=lambda: platform.machine().lower())
    gpu_name: str = ""
    vram_gb: int = 0
    # Per-GPU VRAM for a worker whose rpc-server exposes several devices.
    # Empty = one device worth vram_gb (legacy single-GPU workers).
    per_gpu_vram_gb: list[int] = field(default_factory=list)
    rpc_endpoint: str = ""
    proof_endpoint: str = ""
    network_reachability: str = "unknown"
    relay_hint: str = ""
    version: int = 1
    created_at_unix: int = field(default_factory=_now_unix)
    expires_at_unix: int = 0
    signatures: dict[str, str] = field(default_factory=dict)

    def validate(self) -> None:
        if self.version != 1:
            raise ValueError("unsupported CapabilityAd version")
        if self.uid < 0:
            raise ValueError("uid must be >= 0")
        if not self.hotkey:
            raise ValueError("hotkey is required")
        if not self.endpoint:
            raise ValueError("endpoint is required")
        if not self.supported_backends:
            raise ValueError("at least one backend is required")
        for backend in self.supported_backends:
            if backend not in SUPPORTED_BACKENDS:
                raise ValueError(f"unsupported backend: {backend}")
        for digest in self.cached_model_package_hashes:
            _require_hex_digest("cached_model_package_hashes entry", digest)
        if self.vram_gb < 0:
            raise ValueError("vram_gb must be >= 0")
        if self.per_gpu_vram_gb and any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in self.per_gpu_vram_gb
        ):
            raise ValueError("per_gpu_vram_gb entries must be >= 0 integers")

    def to_dict(self, *, include_signatures: bool = True) -> dict[str, Any]:
        data = {
            "version": int(self.version),
            "uid": int(self.uid),
            "hotkey": self.hotkey,
            "endpoint": self.endpoint,
            "supported_backends": list(self.supported_backends),
            "proof_modes": list(self.proof_modes),
            "cached_model_package_hashes": list(self.cached_model_package_hashes),
            "platform_os": self.platform_os,
            "platform_arch": self.platform_arch,
            "gpu_name": self.gpu_name,
            "vram_gb": int(self.vram_gb),
            # Only serialized when set: single-GPU ads keep their historical
            # dict shape (and ad hashes) byte-for-byte.
            **(
                {"per_gpu_vram_gb": [int(v) for v in self.per_gpu_vram_gb]}
                if self.per_gpu_vram_gb
                else {}
            ),
            "rpc_endpoint": self.rpc_endpoint,
            "proof_endpoint": self.proof_endpoint,
            "network_reachability": self.network_reachability,
            "relay_hint": self.relay_hint,
            "created_at_unix": int(self.created_at_unix),
            "expires_at_unix": int(self.expires_at_unix),
        }
        if include_signatures:
            data["signatures"] = dict(sorted(self.signatures.items()))
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CapabilityAd":
        ad = cls(
            version=int(data.get("version", 1)),
            uid=int(data["uid"]),
            hotkey=str(data["hotkey"]),
            endpoint=str(data["endpoint"]),
            supported_backends=[
                str(item) for item in data.get("supported_backends", ["gguf_stage_worker"])
            ],
            proof_modes=[str(item) for item in data.get("proof_modes", ["verathos-gemv1"])],
            cached_model_package_hashes=[
                str(item) for item in data.get("cached_model_package_hashes", [])
            ],
            platform_os=str(data.get("platform_os", platform.system().lower())),
            platform_arch=str(data.get("platform_arch", platform.machine().lower())),
            gpu_name=str(data.get("gpu_name", "")),
            vram_gb=int(data.get("vram_gb", 0)),
            per_gpu_vram_gb=[int(v) for v in data.get("per_gpu_vram_gb", [])],
            rpc_endpoint=str(data.get("rpc_endpoint", "")),
            proof_endpoint=str(data.get("proof_endpoint", "")),
            network_reachability=str(data.get("network_reachability", "unknown")),
            relay_hint=str(data.get("relay_hint", "")),
            created_at_unix=int(data.get("created_at_unix", 0) or _now_unix()),
            expires_at_unix=int(data.get("expires_at_unix", 0)),
            signatures=dict(data.get("signatures", {})),
        )
        ad.validate()
        return ad

    def body_hash(self) -> bytes:
        return _sha256_tagged(
            b"VERATHOS_CAPABILITY_AD_BODY_V1",
            canonical_json_bytes(self.to_dict(include_signatures=False)),
        )

    def body_hash_hex(self) -> str:
        return self.body_hash().hex()

    def ad_hash(self) -> bytes:
        return _sha256_tagged(
            b"VERATHOS_CAPABILITY_AD_V1",
            canonical_json_bytes(self.to_dict(include_signatures=True)),
        )

    def ad_hash_hex(self) -> str:
        return self.ad_hash().hex()


@dataclass
class StageReceipt:
    """Per-request responsibility receipt signed by a stage owner."""

    request_id: str
    commitment_hash: str
    uid: int
    stage_index: int
    layer_start: int
    layer_end: int
    input_activation_root: str
    output_activation_root: str
    proof_commitment_hash: str
    inference_ms: float
    bytes_in: int
    bytes_out: int
    error: str = ""
    signature: str = ""
    version: int = 1

    def validate(self) -> None:
        if self.version != 1:
            raise ValueError("unsupported StageReceipt version")
        if not self.request_id:
            raise ValueError("request_id is required")
        if self.uid < 0:
            raise ValueError("uid must be >= 0")
        if self.stage_index < 0:
            raise ValueError("stage_index must be >= 0")
        StageRange(self.layer_start, self.layer_end).validate()
        _require_hex_digest("commitment_hash", self.commitment_hash)
        _require_hex_digest("input_activation_root", self.input_activation_root)
        _require_hex_digest("output_activation_root", self.output_activation_root)
        _require_hex_digest("proof_commitment_hash", self.proof_commitment_hash)
        if self.inference_ms < 0:
            raise ValueError("inference_ms must be >= 0")
        if self.bytes_in < 0 or self.bytes_out < 0:
            raise ValueError("bytes_in and bytes_out must be >= 0")

    def to_dict(self, *, include_signature: bool = True) -> dict[str, Any]:
        data = {
            "version": int(self.version),
            "request_id": self.request_id,
            "commitment_hash": self.commitment_hash,
            "uid": int(self.uid),
            "stage_index": int(self.stage_index),
            "layer_start": int(self.layer_start),
            "layer_end": int(self.layer_end),
            "input_activation_root": self.input_activation_root,
            "output_activation_root": self.output_activation_root,
            "proof_commitment_hash": self.proof_commitment_hash,
            "inference_ms": float(self.inference_ms),
            "bytes_in": int(self.bytes_in),
            "bytes_out": int(self.bytes_out),
            "error": self.error,
        }
        if include_signature:
            data["signature"] = self.signature
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "StageReceipt":
        receipt = cls(
            version=int(data.get("version", 1)),
            request_id=str(data["request_id"]),
            commitment_hash=str(data["commitment_hash"]),
            uid=int(data["uid"]),
            stage_index=int(data["stage_index"]),
            layer_start=int(data["layer_start"]),
            layer_end=int(data["layer_end"]),
            input_activation_root=str(data["input_activation_root"]),
            output_activation_root=str(data["output_activation_root"]),
            proof_commitment_hash=str(data["proof_commitment_hash"]),
            inference_ms=float(data["inference_ms"]),
            bytes_in=int(data["bytes_in"]),
            bytes_out=int(data["bytes_out"]),
            error=str(data.get("error", "")),
            signature=str(data.get("signature", "")),
        )
        receipt.validate()
        return receipt

    def body_hash(self) -> bytes:
        return _sha256_tagged(
            b"VERATHOS_STAGE_RECEIPT_BODY_V1",
            canonical_json_bytes(self.to_dict(include_signature=False)),
        )

    def receipt_hash(self) -> bytes:
        return _sha256_tagged(
            b"VERATHOS_STAGE_RECEIPT_V1",
            canonical_json_bytes(self.to_dict(include_signature=True)),
        )

    def receipt_hash_hex(self) -> str:
        return self.receipt_hash().hex()


def stage_receipt_root(receipts: Iterable[StageReceipt]) -> bytes:
    """Compute a deterministic root over signed stage receipts."""

    ordered = sorted(
        receipts,
        key=lambda item: (item.request_id, item.stage_index, item.uid, item.layer_start),
    )
    if not ordered:
        return b""
    h = hashlib.sha256(b"VERATHOS_STAGE_RECEIPT_ROOT_V1")
    h.update(len(ordered).to_bytes(4, "little"))
    for receipt in ordered:
        receipt.validate()
        h.update(receipt.receipt_hash())
    return h.digest()


def save_json(path: str | Path, value: Mapping[str, Any]) -> Path:
    """Write a canonical JSON manifest."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return out


def load_mesh_spec(path: str | Path) -> MeshSpec:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return MeshSpec.from_dict(data)


def load_capability_ad(path: str | Path) -> CapabilityAd:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return CapabilityAd.from_dict(data)
