"""llama.cpp RPC launch helpers for Verathos meshes."""

from __future__ import annotations

import os
import hashlib
import shlex
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from verallm.mesh.types import MeshSpec, StageRange, canonical_json_bytes


DEFAULT_LLAMA_RPC_PORT = 50052
DEFAULT_LLAMA_SERVER_PORT = 8080


@dataclass(frozen=True)
class MeshRpcPlan:
    """Resolved llama.cpp RPC placement for a mesh."""

    mesh_id: str
    rpc_endpoints: list[str]
    rpc_split_weights: list[int]
    rpc_layer_ranges: list[dict[str, int]]
    rpc_arg: str
    # Per-endpoint device weight lists. A multi-GPU worker's rpc-server
    # exposes one llama device per GPU, and llama names them RPC0..RPCn-1
    # globally in endpoint order. Empty = one device per endpoint (legacy).
    rpc_device_weights: list[list[int]] = field(default_factory=list)

    def __post_init__(self) -> None:
        count = len(self.rpc_endpoints)
        if len(self.rpc_split_weights) != count or len(self.rpc_layer_ranges) != count:
            raise ValueError(
                "RPC plan endpoints, split weights, and layer ranges must align"
            )
        if not self.rpc_device_weights:
            # frozen dataclass: default-fill via object.__setattr__
            object.__setattr__(
                self,
                "rpc_device_weights",
                [[int(weight)] for weight in self.rpc_split_weights],
            )
        if len(self.rpc_device_weights) != count:
            raise ValueError(
                "RPC plan device weights must align with endpoints"
            )
        for weights, total in zip(self.rpc_device_weights, self.rpc_split_weights):
            if not weights or any(
                isinstance(w, bool) or not isinstance(w, int) or w <= 0
                for w in weights
            ):
                raise ValueError(
                    "RPC plan device weights must be positive integers"
                )
            if sum(weights) != total:
                raise ValueError(
                    "RPC plan device weights must sum to the member split weight"
                )
        normalized = [
            normalize_rpc_endpoint(endpoint) for endpoint in self.rpc_endpoints
        ]
        if normalized != self.rpc_endpoints:
            raise ValueError("RPC plan endpoints must be normalized")
        if len(set(normalized)) != count:
            raise ValueError("RPC plan endpoints must be unique")
        if any(
            isinstance(weight, bool) or not isinstance(weight, int) or weight <= 0
            for weight in self.rpc_split_weights
        ):
            raise ValueError("RPC plan split weights must be positive integers")
        for layer_range in self.rpc_layer_ranges:
            if set(layer_range) != {"start", "end"}:
                raise ValueError("RPC plan layer ranges must contain start and end")
            start = layer_range["start"]
            end = layer_range["end"]
            if (
                isinstance(start, bool)
                or not isinstance(start, int)
                or isinstance(end, bool)
                or not isinstance(end, int)
                or start < 0
                or end <= start
            ):
                raise ValueError(
                    "RPC plan layer ranges must be non-empty integer ranges"
                )
        if self.rpc_arg != ",".join(normalized):
            raise ValueError("RPC plan argument does not match its ordered endpoints")

    def ordered_members(self) -> list[dict]:
        return [
            {
                "rpc_endpoint": endpoint,
                "rpc_split_weight": weight,
                "layers": dict(layer_range),
                # Only included for multi-device members so single-device
                # plans keep their historical hash byte-for-byte.
                **(
                    {"rpc_device_weights": list(device_weights)}
                    if len(device_weights) > 1
                    else {}
                ),
            }
            for endpoint, weight, layer_range, device_weights in zip(
                self.rpc_endpoints,
                self.rpc_split_weights,
                self.rpc_layer_ranges,
                self.rpc_device_weights,
            )
        ]

    @property
    def total_devices(self) -> int:
        return sum(len(weights) for weights in self.rpc_device_weights)

    @property
    def tensor_split_arg(self) -> str:
        # One weight per llama DEVICE, flattened in endpoint order.
        return ",".join(
            str(weight)
            for weights in self.rpc_device_weights
            for weight in weights
        )

    def plan_hash(self) -> bytes:
        h = hashlib.sha256(b"VERATHOS_LLAMA_CPP_RPC_PLAN_V2")
        h.update(
            canonical_json_bytes(
                {
                    "mesh_id": self.mesh_id,
                    "rpc_members": self.ordered_members(),
                }
            )
        )
        return h.digest()

    def plan_hash_hex(self) -> str:
        return self.plan_hash().hex()

    def to_dict(self) -> dict:
        return {
            "mesh_id": self.mesh_id,
            "rpc_endpoints": list(self.rpc_endpoints),
            "rpc_split_weights": list(self.rpc_split_weights),
            "rpc_layer_ranges": [
                dict(layer_range) for layer_range in self.rpc_layer_ranges
            ],
            "rpc_members": self.ordered_members(),
            "rpc_arg": self.rpc_arg,
            "tensor_split_arg": self.tensor_split_arg,
            "rpc_plan_hash": self.plan_hash_hex(),
        }


def normalize_rpc_endpoint(endpoint: str) -> str:
    """Normalize a llama.cpp RPC endpoint in host:port form."""

    value = endpoint.strip()
    if value.startswith("tcp://"):
        value = value[len("tcp://"):]
    if value.startswith("http://") or value.startswith("https://"):
        raise ValueError("llama.cpp RPC endpoint must be host:port, not an HTTP URL")
    if not value or ":" not in value:
        raise ValueError("llama.cpp RPC endpoint must be host:port")
    host, port_raw = value.rsplit(":", 1)
    if not host:
        raise ValueError("llama.cpp RPC endpoint host is required")
    try:
        port = int(port_raw)
    except ValueError as exc:
        raise ValueError("llama.cpp RPC endpoint port must be an integer") from exc
    if not 1 <= port <= 65535:
        raise ValueError("llama.cpp RPC endpoint port must be in [1, 65535]")
    return f"{host}:{port}"


def llama_tensor_split_layer_ranges(
    total_layers: int,
    weights: list[int],
) -> list[StageRange]:
    """Reproduce llama.cpp repeating-layer placement for ordered weights."""

    total_layers = int(total_layers)
    normalized = [int(weight) for weight in weights]
    if total_layers <= 0:
        raise ValueError("total_layers must be positive")
    if not normalized or any(weight <= 0 for weight in normalized):
        raise ValueError("tensor split weights must all be positive")
    if len(normalized) > total_layers:
        raise ValueError(
            "more tensor split devices than model layers; "
            "cannot assign non-empty ranges"
        )

    total_weight = sum(normalized)
    placement_layers = total_layers + 1
    cumulative_weight = 0
    boundaries = [0]
    for weight in normalized[:-1]:
        cumulative_weight += weight
        boundaries.append(
            min(
                total_layers,
                (
                    placement_layers * cumulative_weight
                    + total_weight
                    - 1
                )
                // total_weight,
            )
        )
    boundaries.append(total_layers)
    ranges = [
        StageRange(boundaries[index], boundaries[index + 1])
        for index in range(len(normalized))
    ]
    if any(item.end <= item.start for item in ranges):
        raise ValueError(
            "tensor split gives a compute stage zero transformer layers"
        )
    return ranges


def rpc_plan_from_mesh(spec: MeshSpec) -> MeshRpcPlan:
    """Build the llama.cpp --rpc argument from mesh member metadata."""

    endpoints: list[str] = []
    split_weights: list[int] = []
    device_weight_lists: list[list[int]] = []
    layer_ranges: list[dict[str, int]] = []
    seen_endpoints: set[str] = set()
    ordered_members = sorted(spec.members, key=lambda item: item.stage_index)
    rpc_members = [member for member in ordered_members if member.rpc_endpoint]
    for member in rpc_members:
        if member.rpc_endpoint:
            endpoint = normalize_rpc_endpoint(member.rpc_endpoint)
            if endpoint in seen_endpoints:
                raise ValueError(f"duplicate RPC endpoint in mesh: {endpoint}")
            seen_endpoints.add(endpoint)
            if member.rpc_split_weight <= 0:
                raise ValueError(
                    "RPC mesh member split weights must be positive: "
                    f"stage {member.stage_index} ({endpoint})"
                )
            endpoints.append(endpoint)
            split_weights.append(member.rpc_split_weight)
            device_weight_lists.append(
                [int(w) for w in member.rpc_device_weights]
                or [member.rpc_split_weight]
            )
            layer_ranges.append(member.layers.to_dict())
    compute_members = [
        member
        for member in ordered_members
        if member.layers.end > member.layers.start
    ]
    if rpc_members and len(rpc_members) == len(compute_members):
        # The tensor split runs over DEVICES (a multi-GPU member's rpc-server
        # exposes one device per GPU); a member's committed range must equal
        # the union of its devices' contiguous ranges.
        device_ranges = llama_tensor_split_layer_ranges(
            spec.total_layers,
            [weight for weights in device_weight_lists for weight in weights],
        )
        expected_ranges: list[StageRange] = []
        device_pos = 0
        for weights in device_weight_lists:
            first = device_ranges[device_pos]
            last = device_ranges[device_pos + len(weights) - 1]
            expected_ranges.append(StageRange(first.start, last.end))
            device_pos += len(weights)
        actual_ranges = [member.layers for member in rpc_members]
        if actual_ranges != expected_ranges:
            raise ValueError(
                "RPC member layer ranges do not match committed tensor split"
            )
    return MeshRpcPlan(
        mesh_id=spec.mesh_id,
        rpc_endpoints=endpoints,
        rpc_split_weights=split_weights,
        rpc_layer_ranges=layer_ranges,
        rpc_arg=",".join(endpoints),
        rpc_device_weights=device_weight_lists,
    )


def resolve_binary(binary: str) -> str:
    """Return an executable path or raise a useful error."""

    if "/" in binary or "\\" in binary:
        path = Path(binary)
        if not path.exists():
            raise FileNotFoundError(f"{binary} does not exist")
        return str(path)
    resolved = shutil.which(binary)
    if not resolved:
        raise FileNotFoundError(f"{binary} was not found on PATH")
    return resolved


def normalize_llama_device(device: str) -> str:
    """Normalize user-facing llama.cpp device aliases."""

    value = device.strip()
    lower = value.lower()
    if lower.startswith("metal") and lower[len("metal"):].isdigit():
        return f"MTL{value[len('metal'):]}"
    return value


def build_rpc_worker_command(
    *,
    binary: str = "rpc-server",
    host: str = "0.0.0.0",
    port: int = DEFAULT_LLAMA_RPC_PORT,
    device: str = "",
    cache: bool = False,
    extra_args: list[str] | None = None,
) -> list[str]:
    """Build a llama.cpp rpc-server command."""

    cmd = [binary, "-H", host, "-p", str(int(port))]
    if device:
        cmd += ["--device", normalize_llama_device(device)]
    if cache:
        cmd.append("-c")
    cmd += list(extra_args or [])
    return cmd


def build_llama_server_command(
    *,
    binary: str = "llama-server",
    model: str = "",
    hf_model: str = "",
    host: str = "127.0.0.1",
    port: int = DEFAULT_LLAMA_SERVER_PORT,
    rpc_endpoints: list[str] | None = None,
    device: str = "",
    n_gpu_layers: int | str | None = None,
    ctx_size: int | None = None,
    tensor_split: str = "",
    alias: str = "",
    extra_args: list[str] | None = None,
) -> list[str]:
    """Build a llama.cpp OpenAI-compatible server command."""

    if bool(model) == bool(hf_model):
        raise ValueError("exactly one of model or hf_model is required")
    normalized_rpc = [normalize_rpc_endpoint(endpoint) for endpoint in (rpc_endpoints or [])]
    normalized_extra_args = list(extra_args or [])
    cmd = [binary]
    if hf_model:
        cmd += ["-hf", hf_model]
    else:
        cmd += ["--model", model]
    cmd += ["--host", host, "--port", str(int(port))]
    # Reasoning models: emit the thinking segment as
    # message.reasoning_content (and delta.reasoning_content while
    # streaming) instead of leaving it template-dependent. The mesh
    # response plumbing and the validator's output-sanity guard both
    # count reasoning as real generated output; without the explicit
    # format a thinking-heavy canary can burn its whole token budget
    # on reasoning and read as "empty_visible_output". Operator
    # extra_args may still override (later flags win in llama-server).
    if not any(
        arg == "--reasoning-format" or arg.startswith("--reasoning-format=")
        for arg in normalized_extra_args
    ):
        cmd += ["--reasoning-format", "deepseek"]
    if normalized_rpc:
        cmd += ["--rpc", ",".join(normalized_rpc)]
    if device:
        cmd += ["--device", normalize_llama_device(device)]
    if n_gpu_layers is not None:
        cmd += ["--n-gpu-layers", str(n_gpu_layers)]
    if ctx_size is not None:
        # 0 is llama-server's "use the model's trained maximum": the
        # unified-KV serve passes it when no registry limit pins the
        # budget, so the full advertised context is actually servable.
        if type(ctx_size) is not int or not 0 <= ctx_size < 2**32:
            raise ValueError("ctx_size must be a non-negative uint32 integer")
        if any(
            arg in {"--ctx-size", "-c"}
            or arg.startswith("--ctx-size=")
            or arg.startswith("-c=")
            or (arg.startswith("-c") and arg[2:].isdigit())
            for arg in normalized_extra_args
        ):
            raise ValueError(
                "extra_args must not override the committed ctx_size"
            )
        cmd += ["--ctx-size", str(ctx_size)]
    if tensor_split:
        cmd += ["--tensor-split", tensor_split]
    if alias:
        cmd += ["--alias", alias]
    cmd += normalized_extra_args
    return cmd


def command_preview(cmd: list[str]) -> str:
    """Shell-safe command preview."""

    return shlex.join(cmd)


def run_command(cmd: list[str], *, env: dict[str, str] | None = None) -> int:
    """Run a foreground llama.cpp process until it exits."""

    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    proc = subprocess.Popen(cmd, env=merged_env)
    try:
        return int(proc.wait())
    except KeyboardInterrupt:
        proc.terminate()
        try:
            return int(proc.wait(timeout=5))
        except subprocess.TimeoutExpired:
            proc.kill()
            return int(proc.wait())
