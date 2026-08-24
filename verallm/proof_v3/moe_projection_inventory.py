"""Bounded authenticated projection inventory for sparse-MoE profiles."""

from __future__ import annotations

import hashlib
import math
import re
import struct
from collections.abc import Callable
from dataclasses import dataclass

from verallm.proof_v3.errors import ProofV3Error, ProofV3VerificationError
from verallm.proof_v3.moe_runtime_semantics import MoeRuntimeSemanticsV3
from verallm.proof_v3.projection_manifest import (
    ProjectionManifestEntryV3,
    ProjectionManifestV3,
)


MOE_EXPERT_GATE_UP_OPERATION_V3 = "moe.expert.gate_up"
MOE_EXPERT_DOWN_OPERATION_V3 = "moe.expert.down"
MOE_ROUTER_OPERATION_V3 = "moe.router"
MOE_SHARED_GATE_OPERATION_V3 = "moe.shared_gate"
MOE_SHARED_GATE_UP_OPERATION_V3 = "moe.shared_gate_up"
MOE_SHARED_DOWN_OPERATION_V3 = "moe.shared_down"
MOE_CAPACITY_SELECTION_ABI_V3 = "moe.capacity_expert.postcommit.v1"

_DOMAIN = b"VERATHOS/PROOF_V3/MOE_PROJECTION_INVENTORY/V1"
_SELECTION_DOMAIN = b"VERATHOS/PROOF_V3/MOE_CAPACITY_SELECTION/V1"
_EXPERT_ENTRY = re.compile(r"^l([0-9]+)\.moe\.expert\.([0-9]+)\.(gate_up|down)$")
_LAYER_ENTRY = re.compile(
    r"^l([0-9]+)\.moe\.(router|shared_gate|shared_gate_up|shared_down)$"
)

__all__ = [
    "MOE_CAPACITY_SELECTION_ABI_V3",
    "MOE_EXPERT_DOWN_OPERATION_V3",
    "MOE_EXPERT_GATE_UP_OPERATION_V3",
    "MOE_ROUTER_OPERATION_V3",
    "MOE_SHARED_DOWN_OPERATION_V3",
    "MOE_SHARED_GATE_OPERATION_V3",
    "MOE_SHARED_GATE_UP_OPERATION_V3",
    "MoeProjectionIdentityV3",
    "build_moe_projection_entries_v3",
    "moe_expert_inventory_digest_v3",
    "parse_moe_projection_name_v3",
    "moe_projection_name_v3",
    "select_capacity_experts_v3",
    "select_capacity_rows_v3",
    "validate_moe_projection_inventory_v3",
]


@dataclass(frozen=True, slots=True, order=True)
class MoeProjectionIdentityV3:
    layer_index: int
    operation_id: str
    expert_id: int | None = None


def build_moe_projection_entries_v3(
    *,
    reader,
    layer_indices,
    num_experts: int,
    hidden_size: int,
    expert_intermediate_size: int,
    shared_expert_intermediate_size: int,
    chunk_size: int,
    progress: Callable[[int, int, str], None] | None = None,
) -> tuple[ProjectionManifestEntryV3, ...]:
    """Root the complete logical sparse-MoE inventory one matrix at a time.

    The checkpoint reader is authoritative for numerical values, while the
    qualified configuration supplies the exact logical geometry.  No live
    fused-kernel layout or backend-local expert mapping participates in this
    owner-side build.
    """

    from verallm.miner.proof_v3_projection_audit import (
        projection_weight_root_only_v3,
    )
    from verallm.proof_v3.economic_wire import scale_to_bits_v3

    numeric = (
        ("expert count", num_experts, 2, 1 << 16),
        ("hidden size", hidden_size, 1, 1 << 24),
        ("expert intermediate size", expert_intermediate_size, 1, 1 << 24),
        (
            "shared expert intermediate size",
            shared_expert_intermediate_size,
            1,
            1 << 24,
        ),
        ("chunk size", chunk_size, 1, 1 << 24),
    )
    for name, value, minimum, maximum in numeric:
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not minimum <= value < maximum
        ):
            raise ProofV3Error(f"MoE projection {name} is out of range")
    layers = tuple(layer_indices)
    if (
        not layers
        or any(isinstance(layer, bool) or not isinstance(layer, int) for layer in layers)
        or layers != tuple(sorted(set(layers)))
        or layers[0] < 0
        or layers[-1] >= 1 << 32
    ):
        raise ProofV3Error("MoE projection layer inventory is malformed")
    has_projection = getattr(reader, "has_projection", None)
    load_projection = getattr(reader, "get_projection_canonical", None)
    if not callable(has_projection) or not callable(load_projection):
        raise ProofV3Error(
            "MoE projection reader has no canonical checkpoint interface"
        )
    if progress is not None and not callable(progress):
        raise ProofV3Error("MoE projection progress callback is malformed")

    per_layer = 4 + 2 * num_experts
    total = len(layers) * per_layer
    entries: list[ProjectionManifestEntryV3] = []
    definitions = (
        (
            MOE_ROUTER_OPERATION_V3,
            "router",
            None,
            hidden_size,
            num_experts,
        ),
        (
            MOE_SHARED_GATE_OPERATION_V3,
            "shared_gate",
            None,
            hidden_size,
            1,
        ),
        (
            MOE_SHARED_GATE_UP_OPERATION_V3,
            "shared_gate_up",
            None,
            hidden_size,
            2 * shared_expert_intermediate_size,
        ),
        (
            MOE_SHARED_DOWN_OPERATION_V3,
            "shared_down",
            None,
            shared_expert_intermediate_size,
            hidden_size,
        ),
    )

    def add(
        *,
        layer_index: int,
        operation_id: str,
        projection: str,
        expert_id: int | None,
        expected_in: int,
        expected_out: int,
    ) -> None:
        if not has_projection(layer_index, projection, expert_id):
            raise ProofV3Error(
                "MoE checkpoint projection inventory is incomplete at "
                f"layer {layer_index}, projection {projection!r}, "
                f"expert {expert_id!r}"
            )
        weight = load_projection(layer_index, projection, expert_id)
        try:
            root, in_dim, out_dim, absmax = projection_weight_root_only_v3(
                weight,
                chunk_size,
            )
        finally:
            del weight
        if (in_dim, out_dim) != (expected_in, expected_out):
            raise ProofV3Error(
                "MoE checkpoint projection geometry disagrees with the "
                f"qualified configuration at layer {layer_index}, "
                f"projection {projection!r}, expert {expert_id!r}"
            )
        name = moe_projection_name_v3(
            layer_index=layer_index,
            operation_id=operation_id,
            expert_id=expert_id,
        )
        entries.append(
            ProjectionManifestEntryV3(
                name=name,
                root=root,
                orientation="out_in",
                in_dim=in_dim,
                out_dim=out_dim,
                scale_bits=scale_to_bits_v3(max(absmax, 1e-8) / 127.0),
            )
        )
        if progress is not None:
            progress(len(entries), total, name)

    for layer_index in layers:
        for operation, projection, expert, in_dim, out_dim in definitions:
            add(
                layer_index=layer_index,
                operation_id=operation,
                projection=projection,
                expert_id=expert,
                expected_in=in_dim,
                expected_out=out_dim,
            )
        for expert_id in range(num_experts):
            add(
                layer_index=layer_index,
                operation_id=MOE_EXPERT_GATE_UP_OPERATION_V3,
                projection="expert_gate_up",
                expert_id=expert_id,
                expected_in=hidden_size,
                expected_out=2 * expert_intermediate_size,
            )
            add(
                layer_index=layer_index,
                operation_id=MOE_EXPERT_DOWN_OPERATION_V3,
                projection="expert_down",
                expert_id=expert_id,
                expected_in=expert_intermediate_size,
                expected_out=hidden_size,
            )
    if len(entries) != total or len({entry.name for entry in entries}) != total:
        raise ProofV3Error("MoE projection inventory build is not exact")
    return tuple(sorted(entries, key=lambda entry: entry.name))


def moe_projection_name_v3(
    *,
    layer_index: int,
    operation_id: str,
    expert_id: int | None = None,
) -> str:
    if (
        isinstance(layer_index, bool)
        or not isinstance(layer_index, int)
        or not 0 <= layer_index < 1 << 32
    ):
        raise ProofV3Error("MoE projection layer index is malformed")
    if operation_id in {
        MOE_EXPERT_GATE_UP_OPERATION_V3,
        MOE_EXPERT_DOWN_OPERATION_V3,
    }:
        if (
            isinstance(expert_id, bool)
            or not isinstance(expert_id, int)
            or not 0 <= expert_id < 1 << 32
        ):
            raise ProofV3Error("MoE expert projection needs an expert id")
        suffix = operation_id.rsplit(".", 1)[-1]
        return f"l{layer_index}.moe.expert.{expert_id}.{suffix}"
    suffix_by_operation = {
        MOE_ROUTER_OPERATION_V3: "router",
        MOE_SHARED_GATE_OPERATION_V3: "shared_gate",
        MOE_SHARED_GATE_UP_OPERATION_V3: "shared_gate_up",
        MOE_SHARED_DOWN_OPERATION_V3: "shared_down",
    }
    try:
        suffix = suffix_by_operation[operation_id]
    except KeyError as exc:
        raise ProofV3Error("MoE projection operation is unsupported") from exc
    if expert_id is not None:
        raise ProofV3Error("MoE non-expert projection must not have an expert id")
    return f"l{layer_index}.moe.{suffix}"


def _parse_name(name: str) -> MoeProjectionIdentityV3 | None:
    match = _EXPERT_ENTRY.fullmatch(name)
    if match is not None:
        layer_text, expert_text, suffix = match.groups()
        layer = int(layer_text)
        expert = int(expert_text)
        if layer_text != str(layer) or expert_text != str(expert):
            raise ProofV3Error("MoE projection name is not canonical")
        operation = (
            MOE_EXPERT_GATE_UP_OPERATION_V3
            if suffix == "gate_up"
            else MOE_EXPERT_DOWN_OPERATION_V3
        )
        return MoeProjectionIdentityV3(layer, operation, expert)
    match = _LAYER_ENTRY.fullmatch(name)
    if match is None:
        return None
    layer_text, suffix = match.groups()
    layer = int(layer_text)
    if layer_text != str(layer):
        raise ProofV3Error("MoE projection name is not canonical")
    operation_by_suffix = {
        "router": MOE_ROUTER_OPERATION_V3,
        "shared_gate": MOE_SHARED_GATE_OPERATION_V3,
        "shared_gate_up": MOE_SHARED_GATE_UP_OPERATION_V3,
        "shared_down": MOE_SHARED_DOWN_OPERATION_V3,
    }
    return MoeProjectionIdentityV3(layer, operation_by_suffix[suffix])


def parse_moe_projection_name_v3(
    name: object,
) -> MoeProjectionIdentityV3 | None:
    """Parse one canonical MoE manifest name, or return ``None`` otherwise."""

    if not isinstance(name, str):
        raise ProofV3Error("MoE projection name must be a string")
    try:
        name.encode("ascii", "strict")
    except UnicodeEncodeError as exc:
        raise ProofV3Error("MoE projection name must be ASCII") from exc
    return _parse_name(name)


def _entry_bytes(entry: ProjectionManifestEntryV3) -> bytes:
    name = entry.name.encode("ascii", "strict")
    orientation = entry.orientation.encode("ascii", "strict")
    if len(name) >= 1 << 16 or len(orientation) >= 1 << 16:
        raise ProofV3Error("MoE projection entry identity is too long")
    if (
        not isinstance(entry.root, bytes)
        or len(entry.root) != 32
        or entry.root == bytes(32)
        or isinstance(entry.in_dim, bool)
        or not isinstance(entry.in_dim, int)
        or not 0 < entry.in_dim < 1 << 32
        or isinstance(entry.out_dim, bool)
        or not isinstance(entry.out_dim, int)
        or not 0 < entry.out_dim < 1 << 32
        or isinstance(entry.scale_bits, bool)
        or not isinstance(entry.scale_bits, int)
        or not 0 < entry.scale_bits < 1 << 64
    ):
        raise ProofV3Error("MoE projection entry is malformed")
    scale = struct.unpack("<d", struct.pack("<Q", entry.scale_bits))[0]
    if not math.isfinite(scale) or scale <= 0:
        raise ProofV3Error("MoE projection scale is malformed")
    return b"".join(
        (
            struct.pack("<H", len(name)),
            name,
            entry.root,
            struct.pack("<H", len(orientation)),
            orientation,
            struct.pack("<IIQ", entry.in_dim, entry.out_dim, entry.scale_bits),
        )
    )


def _moe_entries(
    manifest: ProjectionManifestV3,
) -> tuple[tuple[MoeProjectionIdentityV3, ProjectionManifestEntryV3], ...]:
    if not isinstance(manifest, ProjectionManifestV3):
        raise ProofV3Error("MoE projection manifest has an unexpected type")
    result = []
    for entry in manifest.entries:
        if not isinstance(entry, ProjectionManifestEntryV3):
            raise ProofV3Error("MoE projection entry has an unexpected type")
        identity = _parse_name(entry.name)
        if identity is not None:
            result.append((identity, entry))
    identities = tuple(item[0] for item in result)
    if len(identities) != len(set(identities)):
        raise ProofV3Error("MoE projection inventory contains duplicate identities")
    return tuple(sorted(result, key=lambda item: item[0]))


def moe_expert_inventory_digest_v3(manifest: ProjectionManifestV3) -> bytes:
    entries = _moe_entries(manifest)
    if not entries:
        raise ProofV3Error("MoE projection inventory is empty")
    hasher = hashlib.sha256()
    hasher.update(_DOMAIN)
    hasher.update(struct.pack("<I", len(entries)))
    for _identity, entry in entries:
        encoded = _entry_bytes(entry)
        hasher.update(struct.pack("<I", len(encoded)))
        hasher.update(encoded)
    return hasher.digest()


def validate_moe_projection_inventory_v3(
    *,
    manifest: ProjectionManifestV3,
    semantics: MoeRuntimeSemanticsV3,
) -> dict[MoeProjectionIdentityV3, ProjectionManifestEntryV3]:
    if not isinstance(semantics, MoeRuntimeSemanticsV3):
        raise ProofV3Error("MoE runtime semantics have an unexpected type")
    entries = dict(_moe_entries(manifest))
    expected: dict[MoeProjectionIdentityV3, tuple[int, int]] = {}
    for layer in semantics.layers:
        layer_index = layer.layer_index
        expected.update(
            {
                MoeProjectionIdentityV3(layer_index, MOE_ROUTER_OPERATION_V3): (
                    semantics.hidden_size,
                    semantics.num_experts,
                ),
                MoeProjectionIdentityV3(layer_index, MOE_SHARED_GATE_OPERATION_V3): (
                    semantics.hidden_size,
                    1,
                ),
                MoeProjectionIdentityV3(layer_index, MOE_SHARED_GATE_UP_OPERATION_V3): (
                    semantics.hidden_size,
                    2 * semantics.shared_expert_intermediate_size,
                ),
                MoeProjectionIdentityV3(layer_index, MOE_SHARED_DOWN_OPERATION_V3): (
                    semantics.shared_expert_intermediate_size,
                    semantics.hidden_size,
                ),
            }
        )
        for expert_id in range(semantics.num_experts):
            expected[
                MoeProjectionIdentityV3(
                    layer_index,
                    MOE_EXPERT_GATE_UP_OPERATION_V3,
                    expert_id,
                )
            ] = (
                semantics.hidden_size,
                2 * semantics.expert_intermediate_size,
            )
            expected[
                MoeProjectionIdentityV3(
                    layer_index,
                    MOE_EXPERT_DOWN_OPERATION_V3,
                    expert_id,
                )
            ] = (semantics.expert_intermediate_size, semantics.hidden_size)
    if set(entries) != set(expected):
        missing = sorted(set(expected) - set(entries))
        extra = sorted(set(entries) - set(expected))
        raise ProofV3Error(
            "MoE projection inventory is not exact "
            f"(missing={missing[:2]!r}, extra={extra[:2]!r})"
        )
    for identity, entry in entries.items():
        if (
            entry.orientation != "out_in"
            or (entry.in_dim, entry.out_dim) != expected[identity]
            or entry.row_sq
        ):
            raise ProofV3Error(
                f"MoE projection {entry.name!r} has unsupported geometry or storage"
            )
        _entry_bytes(entry)
    if moe_expert_inventory_digest_v3(manifest) != semantics.expert_inventory_digest:
        raise ProofV3Error(
            "MoE projection inventory does not match authenticated semantics"
        )
    return entries


def select_capacity_experts_v3(
    *,
    nonce: bytes,
    semantics: MoeRuntimeSemanticsV3,
    layer_index: int,
) -> tuple[int, ...]:
    """Select distinct hot-capacity experts independently of routed tokens."""

    if not isinstance(nonce, bytes) or len(nonce) != 32 or nonce == bytes(32):
        raise ProofV3VerificationError("MoE capacity nonce is malformed")
    semantics.layer_for(layer_index)
    count = semantics.capacity_experts_per_audited_layer
    expert_count = semantics.num_experts
    rejection_limit = (1 << 32) - ((1 << 32) % expert_count)
    selected: list[int] = []
    counter = 0
    while len(selected) < count:
        digest = hashlib.sha256(
            b"".join(
                (
                    _SELECTION_DOMAIN,
                    nonce,
                    semantics.digest(),
                    struct.pack("<II", layer_index, counter),
                )
            )
        ).digest()
        counter += 1
        for offset in range(0, len(digest), 4):
            candidate = struct.unpack_from("<I", digest, offset)[0]
            if candidate >= rejection_limit:
                continue
            expert_id = candidate % expert_count
            if expert_id not in selected:
                selected.append(expert_id)
                if len(selected) == count:
                    break
        if counter >= 1 << 20:
            raise ProofV3VerificationError("MoE capacity selection did not converge")
    return tuple(selected)


def select_capacity_rows_v3(
    *,
    nonce: bytes,
    semantics: MoeRuntimeSemanticsV3,
    layer_index: int,
    expert_id: int,
) -> tuple[int, int]:
    """Select one gate/up row and one down row independently of routing."""

    if not isinstance(nonce, bytes) or len(nonce) != 32 or nonce == bytes(32):
        raise ProofV3VerificationError("MoE capacity nonce is malformed")
    semantics.layer_for(layer_index)
    if (
        isinstance(expert_id, bool)
        or not isinstance(expert_id, int)
        or not 0 <= expert_id < semantics.num_experts
    ):
        raise ProofV3VerificationError("MoE capacity expert id is malformed")

    def draw(bound: int, label: int) -> int:
        limit = (1 << 256) - ((1 << 256) % bound)
        for counter in range(1 << 20):
            candidate = int.from_bytes(
                hashlib.sha256(
                    b"".join(
                        (
                            _SELECTION_DOMAIN,
                            b"/rows/v1",
                            nonce,
                            semantics.digest(),
                            struct.pack(
                                "<IIBI",
                                layer_index,
                                expert_id,
                                label,
                                counter,
                            ),
                        )
                    )
                ).digest(),
                "big",
            )
            if candidate < limit:
                return candidate % bound
        raise ProofV3VerificationError("MoE capacity row selection did not converge")

    return (
        draw(2 * semantics.expert_intermediate_size, 1),
        draw(semantics.hidden_size, 2),
    )
