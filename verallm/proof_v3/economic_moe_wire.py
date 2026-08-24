"""Canonical bounded nested wire for compact sparse-MoE recomputation."""

from __future__ import annotations

from dataclasses import dataclass

from verallm.proof_v3.economic_wire import (
    MAX_WEIGHT_RANGE_SIBLINGS,
    EconomicWeightRowRevealV3,
    _Reader,
    _Writer,
    _fixed32,
    _u_range,
)
from verallm.proof_v3.errors import ProofV3Error


ECONOMIC_MOE_WIRE_VERSION_V3 = 1
MAX_ECONOMIC_MOE_WIRE_BYTES_V3 = 112 << 20
MAX_MOE_AUDITED_LAYERS_V3 = 64
MAX_MOE_SELECTED_EXPERTS_V3 = 64
MAX_MOE_CAPACITY_EXPERTS_V3 = 64
MAX_MOE_HIDDEN_INDICES_V3 = 64
MAX_MOE_WEIGHT_RANGE_BYTES_V3 = 16 << 20
_MAGIC = b"V3MW"

__all__ = [
    "ECONOMIC_MOE_WIRE_VERSION_V3",
    "MAX_ECONOMIC_MOE_WIRE_BYTES_V3",
    "EconomicMoeCapacityRevealV3",
    "EconomicMoeExpertRevealV3",
    "EconomicMoeLayerRevealV3",
    "EconomicMoeWeightRangeRevealV3",
    "EconomicMoeWireV3",
    "decode_economic_moe_wire_v3",
    "encode_economic_moe_wire_v3",
]


def _sorted_indices(values, name: str, *, maximum: int) -> tuple[int, ...]:
    values = tuple(values)
    if not values or len(values) > maximum:
        raise ProofV3Error(f"{name} count is out of range")
    previous = -1
    for value in values:
        _u_range(value, f"{name} entry", bits=32)
        if value <= previous:
            raise ProofV3Error(f"{name} must be strictly increasing")
        previous = value
    return values


def _ordered_distinct_experts(values, name: str) -> None:
    seen = set()
    for value in values:
        _u_range(value, f"{name} expert id", bits=32)
        if value in seen:
            raise ProofV3Error(f"{name} contains a duplicate expert id")
        seen.add(value)


def _weight_rows(
    values,
    name: str,
    *,
    maximum: int = MAX_MOE_HIDDEN_INDICES_V3,
) -> tuple[EconomicWeightRowRevealV3, ...]:
    values = tuple(values)
    if not values or len(values) > maximum:
        raise ProofV3Error(f"{name} count is out of range")
    previous = -1
    for value in values:
        if not isinstance(value, EconomicWeightRowRevealV3):
            raise ProofV3Error(f"{name} has an unexpected type")
        if value.row_index <= previous:
            raise ProofV3Error(f"{name} must be strictly ordered")
        previous = value.row_index
    return values


@dataclass(frozen=True, slots=True)
class EconomicMoeWeightRangeRevealV3:
    """One contiguous set of complete int8 rows with one range multiproof."""

    first_row: int
    row_count: int
    chunk_blob: bytes
    range_siblings: tuple[bytes, ...]

    def __post_init__(self) -> None:
        _u_range(self.first_row, "MoE weight-range first row", bits=32)
        _u_range(
            self.row_count,
            "MoE weight-range row count",
            bits=32,
            positive=True,
        )
        if (
            not isinstance(self.chunk_blob, bytes)
            or not self.chunk_blob
            or len(self.chunk_blob) > MAX_MOE_WEIGHT_RANGE_BYTES_V3
        ):
            raise ProofV3Error("MoE weight-range blob length is out of range")
        siblings = tuple(self.range_siblings)
        if len(siblings) > MAX_WEIGHT_RANGE_SIBLINGS:
            raise ProofV3Error("MoE weight-range proof exceeds the wire bound")
        for digest in siblings:
            _fixed32(digest, "MoE weight-range sibling")
        object.__setattr__(self, "range_siblings", siblings)

    def encode(self, writer: _Writer) -> None:
        writer.pack("<II", self.first_row, self.row_count)
        writer.vbytes(
            self.chunk_blob,
            "MoE weight-range blob",
            MAX_MOE_WEIGHT_RANGE_BYTES_V3,
        )
        writer.pack("<H", len(self.range_siblings))
        for digest in self.range_siblings:
            writer.raw(digest)

    @classmethod
    def decode(cls, reader: _Reader) -> "EconomicMoeWeightRangeRevealV3":
        first_row, row_count = reader.unpack("<II")
        chunk_blob = reader.vbytes(
            "MoE weight-range blob",
            MAX_MOE_WEIGHT_RANGE_BYTES_V3,
        )
        sibling_count = reader.unpack("<H")[0]
        if sibling_count > MAX_WEIGHT_RANGE_SIBLINGS:
            raise ProofV3Error("MoE weight-range proof exceeds the wire bound")
        return cls(
            first_row=first_row,
            row_count=row_count,
            chunk_blob=chunk_blob,
            range_siblings=tuple(reader.read(32) for _ in range(sibling_count)),
        )


@dataclass(frozen=True, slots=True)
class EconomicMoeExpertRevealV3:
    """One selected routed expert in canonical router-rank order."""

    expert_id: int
    gate_up_rows: EconomicMoeWeightRangeRevealV3
    down_rows: tuple[EconomicWeightRowRevealV3, ...]

    def __post_init__(self) -> None:
        _u_range(self.expert_id, "MoE selected expert id", bits=32)
        if (
            not isinstance(self.gate_up_rows, EconomicMoeWeightRangeRevealV3)
            or self.gate_up_rows.first_row != 0
        ):
            raise ProofV3Error("MoE selected gate/up range is malformed")
        object.__setattr__(
            self,
            "down_rows",
            _weight_rows(self.down_rows, "MoE selected down rows"),
        )

    def encode(self, writer: _Writer) -> None:
        writer.pack("<I", self.expert_id)
        self.gate_up_rows.encode(writer)
        writer.pack("<I", len(self.down_rows))
        for reveal in self.down_rows:
            reveal.encode(writer)

    @classmethod
    def decode(cls, reader: _Reader) -> "EconomicMoeExpertRevealV3":
        expert_id = reader.unpack("<I")[0]
        gate_up_rows = EconomicMoeWeightRangeRevealV3.decode(reader)
        count = reader.count(
            "MoE selected down rows",
            MAX_MOE_HIDDEN_INDICES_V3,
        )
        return cls(
            expert_id=expert_id,
            gate_up_rows=gate_up_rows,
            down_rows=tuple(
                EconomicWeightRowRevealV3.decode(reader) for _ in range(count)
            ),
        )


@dataclass(frozen=True, slots=True)
class EconomicMoeCapacityRevealV3:
    """Route-independent post-commit opening for one logical expert."""

    expert_id: int
    gate_up_row: EconomicWeightRowRevealV3
    down_row: EconomicWeightRowRevealV3

    def __post_init__(self) -> None:
        _u_range(self.expert_id, "MoE capacity expert id", bits=32)
        if not isinstance(
            self.gate_up_row,
            EconomicWeightRowRevealV3,
        ) or not isinstance(self.down_row, EconomicWeightRowRevealV3):
            raise ProofV3Error("MoE capacity weight row has an unexpected type")

    def encode(self, writer: _Writer) -> None:
        writer.pack("<I", self.expert_id)
        self.gate_up_row.encode(writer)
        self.down_row.encode(writer)

    @classmethod
    def decode(cls, reader: _Reader) -> "EconomicMoeCapacityRevealV3":
        return cls(
            expert_id=reader.unpack("<I")[0],
            gate_up_row=EconomicWeightRowRevealV3.decode(reader),
            down_row=EconomicWeightRowRevealV3.decode(reader),
        )


@dataclass(frozen=True, slots=True)
class EconomicMoeLayerRevealV3:
    """One nonce-selected token's complete sparse-MoE recompute material."""

    layer_index: int
    token_position: int
    padding_multiple: int
    hidden_indices: tuple[int, ...]
    router_rows: EconomicMoeWeightRangeRevealV3
    selected_experts: tuple[EconomicMoeExpertRevealV3, ...]
    shared_gate_rows: EconomicMoeWeightRangeRevealV3
    shared_gate_up_rows: EconomicMoeWeightRangeRevealV3
    shared_down_rows: tuple[EconomicWeightRowRevealV3, ...]
    capacity_experts: tuple[EconomicMoeCapacityRevealV3, ...]

    def __post_init__(self) -> None:
        _u_range(self.layer_index, "MoE layer index", bits=32)
        _u_range(self.token_position, "MoE token position", bits=64)
        _u_range(
            self.padding_multiple,
            "MoE padding multiple",
            bits=16,
            positive=True,
        )
        hidden = _sorted_indices(
            self.hidden_indices,
            "MoE hidden indices",
            maximum=MAX_MOE_HIDDEN_INDICES_V3,
        )
        if (
            not isinstance(self.router_rows, EconomicMoeWeightRangeRevealV3)
            or self.router_rows.first_row != 0
        ):
            raise ProofV3Error("MoE router range is malformed")
        selected = tuple(self.selected_experts)
        if not selected or len(selected) > MAX_MOE_SELECTED_EXPERTS_V3:
            raise ProofV3Error("MoE selected expert count is out of range")
        if any(not isinstance(item, EconomicMoeExpertRevealV3) for item in selected):
            raise ProofV3Error("MoE selected expert has an unexpected type")
        _ordered_distinct_experts(
            tuple(item.expert_id for item in selected),
            "MoE selected experts",
        )
        expected_rows = hidden
        if any(
            tuple(row.row_index for row in item.down_rows) != expected_rows
            for item in selected
        ):
            raise ProofV3Error("MoE selected down rows do not match hidden indices")
        if (
            not isinstance(self.shared_gate_rows, EconomicMoeWeightRangeRevealV3)
            or self.shared_gate_rows.first_row != 0
            or self.shared_gate_rows.row_count != 1
            or not isinstance(
                self.shared_gate_up_rows,
                EconomicMoeWeightRangeRevealV3,
            )
            or self.shared_gate_up_rows.first_row != 0
        ):
            raise ProofV3Error("MoE shared full-row ranges are malformed")
        shared_down = _weight_rows(
            self.shared_down_rows,
            "MoE shared down rows",
        )
        if tuple(row.row_index for row in shared_down) != expected_rows:
            raise ProofV3Error("MoE shared down rows do not match hidden indices")
        capacity = tuple(self.capacity_experts)
        if not capacity or len(capacity) > MAX_MOE_CAPACITY_EXPERTS_V3:
            raise ProofV3Error("MoE capacity expert count is out of range")
        previous = -1
        for item in capacity:
            if not isinstance(item, EconomicMoeCapacityRevealV3):
                raise ProofV3Error("MoE capacity expert has an unexpected type")
            if item.expert_id <= previous:
                raise ProofV3Error(
                    "MoE capacity experts must be strictly ordered"
                )
            previous = item.expert_id
        object.__setattr__(self, "hidden_indices", hidden)
        object.__setattr__(self, "selected_experts", selected)
        object.__setattr__(self, "shared_down_rows", shared_down)
        object.__setattr__(self, "capacity_experts", capacity)

    def encode(self, writer: _Writer) -> None:
        writer.pack(
            "<IQH",
            self.layer_index,
            self.token_position,
            self.padding_multiple,
        )
        writer.pack("<I", len(self.hidden_indices))
        for index in self.hidden_indices:
            writer.pack("<I", index)
        self.router_rows.encode(writer)
        writer.pack("<I", len(self.selected_experts))
        for reveal in self.selected_experts:
            reveal.encode(writer)
        self.shared_gate_rows.encode(writer)
        self.shared_gate_up_rows.encode(writer)
        writer.pack("<I", len(self.shared_down_rows))
        for reveal in self.shared_down_rows:
            reveal.encode(writer)
        writer.pack("<I", len(self.capacity_experts))
        for reveal in self.capacity_experts:
            reveal.encode(writer)

    @classmethod
    def decode(cls, reader: _Reader) -> "EconomicMoeLayerRevealV3":
        layer_index, token_position, padding_multiple = reader.unpack("<IQH")
        hidden_count = reader.count(
            "MoE hidden indices",
            MAX_MOE_HIDDEN_INDICES_V3,
        )
        hidden_indices = tuple(
            reader.unpack("<I")[0] for _ in range(hidden_count)
        )
        router_rows = EconomicMoeWeightRangeRevealV3.decode(reader)
        selected_count = reader.count(
            "MoE selected experts",
            MAX_MOE_SELECTED_EXPERTS_V3,
        )
        selected = tuple(
            EconomicMoeExpertRevealV3.decode(reader)
            for _ in range(selected_count)
        )
        shared_gate_rows = EconomicMoeWeightRangeRevealV3.decode(reader)
        shared_gate_up_rows = EconomicMoeWeightRangeRevealV3.decode(reader)
        shared_down_count = reader.count(
            "MoE shared down rows",
            MAX_MOE_HIDDEN_INDICES_V3,
        )
        shared_down = tuple(
            EconomicWeightRowRevealV3.decode(reader)
            for _ in range(shared_down_count)
        )
        capacity_count = reader.count(
            "MoE capacity experts",
            MAX_MOE_CAPACITY_EXPERTS_V3,
        )
        capacity = tuple(
            EconomicMoeCapacityRevealV3.decode(reader)
            for _ in range(capacity_count)
        )
        return cls(
            layer_index=layer_index,
            token_position=token_position,
            padding_multiple=padding_multiple,
            hidden_indices=hidden_indices,
            router_rows=router_rows,
            selected_experts=selected,
            shared_gate_rows=shared_gate_rows,
            shared_gate_up_rows=shared_gate_up_rows,
            shared_down_rows=shared_down,
            capacity_experts=capacity,
        )


@dataclass(frozen=True, slots=True)
class EconomicMoeWireV3:
    semantics_digest: bytes
    layers: tuple[EconomicMoeLayerRevealV3, ...]

    def __post_init__(self) -> None:
        _fixed32(self.semantics_digest, "MoE semantics digest", nonzero=True)
        layers = tuple(self.layers)
        if not layers or len(layers) > MAX_MOE_AUDITED_LAYERS_V3:
            raise ProofV3Error("MoE audited layer count is out of range")
        previous = -1
        for layer in layers:
            if not isinstance(layer, EconomicMoeLayerRevealV3):
                raise ProofV3Error("MoE layer reveal has an unexpected type")
            if layer.layer_index <= previous:
                raise ProofV3Error("MoE layer reveals must be strictly ordered")
            previous = layer.layer_index
        object.__setattr__(self, "layers", layers)


def encode_economic_moe_wire_v3(value: EconomicMoeWireV3) -> bytes:
    if not isinstance(value, EconomicMoeWireV3):
        raise ProofV3Error("MoE wire value has an unexpected type")
    writer = _Writer()
    writer.pack("<4sH", _MAGIC, ECONOMIC_MOE_WIRE_VERSION_V3)
    writer.raw(value.semantics_digest)
    writer.pack("<I", len(value.layers))
    for layer in value.layers:
        layer.encode(writer)
    encoded = writer.finish()
    if len(encoded) > MAX_ECONOMIC_MOE_WIRE_BYTES_V3:
        raise ProofV3Error("MoE wire exceeds the byte limit")
    return encoded


def decode_economic_moe_wire_v3(encoded: bytes) -> EconomicMoeWireV3:
    if not isinstance(encoded, bytes) or len(encoded) > MAX_ECONOMIC_MOE_WIRE_BYTES_V3:
        raise ProofV3Error("MoE wire bytes are malformed")
    reader = _Reader(encoded, "economic MoE wire")
    magic, version = reader.unpack("<4sH")
    if magic != _MAGIC or version != ECONOMIC_MOE_WIRE_VERSION_V3:
        raise ProofV3Error("MoE wire header is unsupported")
    semantics_digest = reader.read(32)
    layer_count = reader.count(
        "MoE audited layers",
        MAX_MOE_AUDITED_LAYERS_V3,
    )
    result = EconomicMoeWireV3(
        semantics_digest=semantics_digest,
        layers=tuple(
            EconomicMoeLayerRevealV3.decode(reader)
            for _ in range(layer_count)
        ),
    )
    reader.finish()
    if encode_economic_moe_wire_v3(result) != encoded:
        raise ProofV3Error("MoE wire is not canonical")
    return result
