"""Pool worker unit planning for one host.

A worker unit owns a GROUP of GPUs on this machine. The default is one unit
owning every GPU (fastest: the model tensor-splits across local devices
inside one process, and RPC is only used between physical machines). The
operator can instead split the host into groups - "0,1,2+3" plans three
units - to run several independent models on one box. Each unit gets its
own worker id, PM2 unit name, ports, workdir, and catalog file, and is
pinned to its GPUs with CUDA_VISIBLE_DEVICES so the llama binaries AND the
Python proof stack (torch, zkllm) all land on the intended devices. The
planning lives here, in one testable module, and the shell installer only
renders what this module derives.

Port layout: unit slot i uses rpc_base + i, proof_base + i, and
mesh_base + 2*i. The mesh port strides by two because the driver's
llama-server binds mesh_port + 1 (see pool.py drive()). Slot 0 therefore
reproduces the original single-GPU defaults exactly.
"""
from __future__ import annotations

import csv
import io
import json
import re
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from verallm.mesh.private_files import read_owner_only_text, write_owner_only_json

DEFAULT_RPC_PORT_BASE = 50052
DEFAULT_PROOF_PORT_BASE = 9402
DEFAULT_MESH_PORT_BASE = 9443
MESH_PORT_STRIDE = 2
# mesh_base + 2*i must stay below the pool manager's default port 9500.
MAX_UNITS_PER_HOST = 28

UNIT_REGISTRY_VERSION = 1
UNIT_REGISTRY_PATH = Path.home() / ".verathos" / "mesh-units.json"

# Must match the pool control plane's worker id validator (pool.py).
_WORKER_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}")
# Default suffix budget: "-gpu" plus a two-digit index. Group suffixes can
# be longer; plan_worker_units passes the real budget in explicitly.
_MAX_SUFFIX_LEN = len("-gpu27")


@dataclass(frozen=True)
class GpuInfo:
    index: int
    name: str
    vram_gb: int


@dataclass(frozen=True)
class WorkerUnit:
    worker_id: str
    pm2_name: str
    gpu_index: int
    gpu_name: str
    vram_gb: int
    cuda_visible_devices: str
    rpc_device: str
    rpc_port: int
    proof_port: int
    mesh_port: int
    workdir: str
    catalog: str
    # Multi-GPU group fields. Defaults keep registries written before the
    # feature loading; gpu_index/gpu_name/vram_gb above stay the legacy
    # single-value views (first index, display name, SUMMED VRAM).
    gpu_indices: list[int] = field(default_factory=list)
    gpu_names: list[str] = field(default_factory=list)
    per_gpu_vram_gb: list[int] = field(default_factory=list)


def gpu_group_label(indices: Sequence[int]) -> str:
    """Human/id label for a GPU group: 2 -> "2", (2,3) -> "2-3", (0,2) -> "0.2".

    A contiguous ascending run renders as "first-last"; anything else joins
    with "." (both stay inside the pool worker-id charset).
    """
    ordered = list(indices)
    if len(ordered) == 1:
        return str(ordered[0])
    contiguous = all(b == a + 1 for a, b in zip(ordered, ordered[1:]))
    if contiguous:
        return f"{ordered[0]}-{ordered[-1]}"
    return ".".join(str(i) for i in ordered)


def parse_gpu_groups(
    spec: str, available: Sequence[int]
) -> list[tuple[int, ...]]:
    """Parse a GPU grouping spec against the GPUs present on this host.

    Grammar: comma separates worker units, "+" groups GPUs into one unit.
    "0,1,2+3" plans three units (gpu0, gpu1, and one unit owning 2 and 3).
    An empty spec or "all" means ONE unit owning every available GPU - the
    default topology, because splitting a single-box model over RPC between
    processes is strictly slower than a local tensor split.
    """
    text = str(spec or "").strip().lower()
    present = list(available)
    if not text or text == "all":
        if not present:
            raise ValueError("no GPUs available to plan")
        return [tuple(present)]
    groups: list[tuple[int, ...]] = []
    seen: set[int] = set()
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            indices = tuple(int(part) for part in token.split("+") if part.strip())
        except ValueError as exc:
            raise ValueError(f"bad GPU group {token!r}: {exc}") from None
        if not indices:
            continue
        if len(set(indices)) != len(indices):
            raise ValueError(f"GPU group {token!r} repeats an index")
        overlap = seen.intersection(indices)
        if overlap:
            raise ValueError(
                f"GPU index {sorted(overlap)} appears in more than one group"
            )
        missing = [i for i in indices if i not in present]
        if missing:
            raise ValueError(
                f"--gpus names indexes not present on this host: {missing}"
            )
        seen.update(indices)
        groups.append(tuple(sorted(indices)))
    if not groups:
        raise ValueError("no GPU groups in spec")
    return groups


def sanitize_worker_id_base(raw: str, *, reserve: int = _MAX_SUFFIX_LEN) -> str:
    """Fold an arbitrary hostname into a valid worker id base.

    The result plus any "-gpu<group>" suffix must satisfy the pool's worker
    id pattern, so map every disallowed character to "-", collapse runs,
    strip edge separators, and truncate to leave room for the suffix.
    """
    lowered = str(raw or "").strip().lower()
    folded = re.sub(r"[^a-z0-9_.-]+", "-", lowered)
    folded = re.sub(r"-{2,}", "-", folded).strip("-.")
    folded = folded[: 96 - reserve]
    folded = folded.rstrip("-.")
    if not folded or not folded[0].isalnum():
        folded = "worker"
    return folded


def detect_gpus() -> list[GpuInfo]:
    """Enumerate CUDA GPUs via nvidia-smi; empty when none are visible.

    Parsed with the csv module because GPU names may contain commas.
    """
    try:
        output = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    gpus: list[GpuInfo] = []
    for row in csv.reader(io.StringIO(output)):
        if len(row) < 3:
            continue
        try:
            index = int(row[0].strip())
            vram_gb = int(float(row[-1].strip()) / 1024)
            name = ",".join(cell for cell in row[1:-1]).strip()
        except ValueError:
            continue
        gpus.append(GpuInfo(index=index, name=name, vram_gb=vram_gb))
    gpus.sort(key=lambda gpu: gpu.index)
    return gpus


def plan_worker_units(
    gpus: Sequence[GpuInfo],
    *,
    worker_id_base: str,
    home: Path,
    rpc_port_base: int = DEFAULT_RPC_PORT_BASE,
    proof_port_base: int = DEFAULT_PROOF_PORT_BASE,
    mesh_port_base: int = DEFAULT_MESH_PORT_BASE,
    backend: str = "cuda",
    groups: Sequence[Sequence[int]] | None = None,
) -> list[WorkerUnit]:
    """Derive worker units (one per GPU group) with collision-free ports.

    ``groups`` is a list of GPU-index groups, each becoming one unit that
    owns those GPUs. None means the default topology: ONE unit owning every
    given GPU. Unit ordering follows the group sequence; port slots are
    sequential (slot i, not GPU index i) so a subset like --gpus 0,3 packs
    into the first two slots.
    """
    if not gpus:
        raise ValueError("no GPUs to plan worker units for")
    by_index = {gpu.index: gpu for gpu in gpus}
    if groups is None:
        groups = [tuple(gpu.index for gpu in gpus)]
    if len(groups) > MAX_UNITS_PER_HOST:
        raise ValueError(
            f"{len(groups)} units exceed the {MAX_UNITS_PER_HOST}-unit port budget"
        )
    labels = []
    for group in groups:
        missing = [i for i in group if i not in by_index]
        if missing:
            raise ValueError(f"GPU group names unknown indexes: {missing}")
        labels.append(gpu_group_label(group))
    reserve = max(len(f"-gpu{label}") for label in labels)
    base = sanitize_worker_id_base(worker_id_base, reserve=max(reserve, _MAX_SUFFIX_LEN))
    units: list[WorkerUnit] = []
    used_ports: set[int] = set()
    for slot, (group, label) in enumerate(zip(groups, labels)):
        members = [by_index[i] for i in group]
        worker_id = f"{base}-gpu{label}"
        if not _WORKER_ID_RE.fullmatch(worker_id):
            raise ValueError(f"derived worker id {worker_id!r} is not valid")
        rpc_port = rpc_port_base + slot
        proof_port = proof_port_base + slot
        mesh_port = mesh_port_base + MESH_PORT_STRIDE * slot
        # The driver's llama-server binds mesh_port + 1 as well.
        unit_ports = (rpc_port, proof_port, mesh_port, mesh_port + 1)
        if used_ports.intersection(unit_ports) or len(set(unit_ports)) != 4:
            raise ValueError(
                f"port collision planning unit {worker_id!r}: "
                f"rpc={rpc_port} proof={proof_port} mesh={mesh_port} "
                f"llama-aux={mesh_port + 1} (mesh serving always binds "
                "mesh_port+1 too, so proof/rpc must not sit directly above "
                "the mesh port — with a consecutive published range use "
                "e.g. mesh=N, proof=N+2, rpc=N+3)"
            )
        used_ports.update(unit_ports)
        if backend == "cuda":
            # The mask makes the unit's GPUs the only visible devices, so
            # in-mask indexes are always 0..n-1 for llama and torch alike.
            device = ",".join(f"CUDA{i}" for i in range(len(members)))
            visible = ",".join(str(gpu.index) for gpu in members)
        else:
            device = "MTL0"
            visible = ""
        names = [gpu.name for gpu in members]
        display_name = names[0] if len(set(names)) == 1 else ", ".join(names)
        if len(members) > 1 and len(set(names)) == 1:
            display_name = f"{names[0]} x{len(members)}"
        units.append(
            WorkerUnit(
                worker_id=worker_id,
                pm2_name=f"verathos-mesh-{worker_id}",
                gpu_index=members[0].index,
                gpu_name=display_name,
                vram_gb=sum(gpu.vram_gb for gpu in members),
                cuda_visible_devices=visible,
                rpc_device=device,
                rpc_port=rpc_port,
                proof_port=proof_port,
                mesh_port=mesh_port,
                workdir=str(home / ".verathos" / f"poolwork-gpu{label}"),
                catalog=str(home / ".verathos" / f"pool-catalog-gpu{label}.json"),
                gpu_indices=[gpu.index for gpu in members],
                gpu_names=names,
                per_gpu_vram_gb=[gpu.vram_gb for gpu in members],
            )
        )
    return units


def save_unit_registry(
    units: Sequence[WorkerUnit],
    *,
    manager_endpoint: str,
    pool_id: str,
    token_file: str,
    path: Path | None = None,
) -> Path:
    """Persist this host's planned units for status/logs/stop commands."""
    registry = {
        "version": UNIT_REGISTRY_VERSION,
        "manager_endpoint": manager_endpoint,
        "pool_id": pool_id,
        "token_file": token_file,
        "units": [asdict(unit) for unit in units],
    }
    target = path if path is not None else UNIT_REGISTRY_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    return write_owner_only_json(target, registry)


def load_unit_registry(path: Path | None = None) -> dict[str, Any] | None:
    """Load this host's unit registry; None when the host never joined."""
    target = path if path is not None else UNIT_REGISTRY_PATH
    if not target.exists():
        return None
    raw = read_owner_only_text(target, label="mesh unit registry")
    registry = json.loads(raw)
    if not isinstance(registry, Mapping):
        raise ValueError("mesh unit registry must be a JSON object")
    return dict(registry)


def units_from_registry(registry: Mapping[str, Any]) -> list[WorkerUnit]:
    return [WorkerUnit(**unit) for unit in registry.get("units", [])]


# -- local GGUF discovery ----------------------------------------------------

_GGUF_SHARD_RE = re.compile(r"^(?P<prefix>.+)-(?P<index>\d{5})-of-(?P<total>\d{5})\.gguf$")


def derive_gguf_model_id(filename: str) -> str:
    """A model id from a GGUF file name (shard suffixes stripped)."""

    match = _GGUF_SHARD_RE.match(filename)
    if match:
        return match.group("prefix")
    return filename[: -len(".gguf")] if filename.endswith(".gguf") else filename


def gguf_shard_set(path: Path) -> list[Path]:
    """Every shard belonging to the same model as ``path``, sorted.

    Refuses an incomplete shard set: launching from a partial download
    fails much later inside llama.cpp with an unhelpful error.
    """

    match = _GGUF_SHARD_RE.match(path.name)
    if not match:
        return [path]
    total = int(match.group("total"))
    shards = sorted(
        candidate
        for candidate in path.parent.glob(
            f"{match.group('prefix')}-*-of-{match.group('total')}.gguf"
        )
        if _GGUF_SHARD_RE.match(candidate.name)
    )
    if len(shards) != total:
        raise ValueError(
            f"{path.name} is one of {total} shards, but only "
            f"{len(shards)} are present in {path.parent}"
        )
    return shards


def discover_local_gguf_models(
    roots: Sequence[Path] | None = None,
) -> list[dict[str, Any]]:
    """GGUF models already on this machine, grouped by shard set.

    The pool only knows what workers advertise; nothing scans disk for
    them. This helper is that scan: the setup wizard uses it so a machine
    that already holds a model is never asked to download it again.
    """

    if roots is None:
        roots = [
            Path.home() / "models",
            Path.home() / ".verathos" / "mesh-models",
        ]
    # Group per directory so the same model in two roots yields two
    # candidate sets instead of one merged, seemingly-broken one.
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for root in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.gguf")):
            if not path.is_file():
                continue
            model_id = derive_gguf_model_id(path.name)
            entry = grouped.setdefault(
                (str(path.parent), model_id),
                {"model_id": model_id, "files": [], "bytes": 0},
            )
            entry["files"].append(path)
            entry["bytes"] += path.stat().st_size
    complete: dict[str, dict[str, Any]] = {}
    for entry in grouped.values():
        if entry["model_id"] in complete:
            continue  # first complete copy wins
        try:
            shards = gguf_shard_set(entry["files"][0])
        except ValueError:
            continue  # incomplete shard set: not servable, not offered
        if sorted(entry["files"]) != shards:
            continue
        entry["files"] = shards
        complete[entry["model_id"]] = entry
    return sorted(complete.values(), key=lambda entry: entry["model_id"])
