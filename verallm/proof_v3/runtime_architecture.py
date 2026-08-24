"""Fail-closed qualification of the runtime layer architecture.

The signed proof profile uses the protocol names ``full_attention`` and
``gdn``.  Hybrid Qwen model configs call the latter ``linear_attention``.
This module is the single conversion point and cross-checks the declaration
against the graph-integrated projection modules before a profile is built.
"""

from __future__ import annotations

from collections.abc import Sequence

from verallm.proof_v2.layout import (
    FULL_OUTPUT_OPERATION_ID,
    FULL_QKV_OPERATION_ID,
    GDN_BA_OPERATION_ID,
    GDN_OUTPUT_OPERATION_ID,
    GDN_QKVZ_OPERATION_ID,
    MLP_DOWN_OPERATION_ID,
    MLP_GATE_UP_OPERATION_ID,
)
from verallm.proof_v3.errors import ProofV3Error
from verallm.moe.detection import is_moe_layer
from verallm.vllm_plugin.capture_linear import (
    dense_execution_projection_specs,
)

__all__ = [
    "manifest_projection_suffix_v3",
    "manifest_projection_suffixes_v3",
    "qualify_runtime_layer_kinds_v3",
    "runtime_full_attention_layers_v3",
]

_DECLARED_TO_PROTOCOL = {
    "full_attention": "full_attention",
    "linear_attention": "gdn",
}
_COMMON_OPERATIONS = frozenset(
    (MLP_GATE_UP_OPERATION_ID, MLP_DOWN_OPERATION_ID)
)
_ARCHITECTURE_OPERATIONS = {
    "full_attention": frozenset(
        (FULL_QKV_OPERATION_ID, FULL_OUTPUT_OPERATION_ID)
    ),
    "gdn": frozenset(
        (GDN_QKVZ_OPERATION_ID, GDN_BA_OPERATION_ID, GDN_OUTPUT_OPERATION_ID)
    ),
}
_MANIFEST_SUFFIX_BY_OPERATION = {
    FULL_QKV_OPERATION_ID: "qkv",
    FULL_OUTPUT_OPERATION_ID: "o",
    GDN_QKVZ_OPERATION_ID: "gdn_qkvz",
    GDN_BA_OPERATION_ID: "gdn_ba",
    GDN_OUTPUT_OPERATION_ID: "gdn_o",
    MLP_GATE_UP_OPERATION_ID: "gate_up",
    MLP_DOWN_OPERATION_ID: "down",
}


def _text_config(config):
    if isinstance(config, dict):
        value = config.get("text_config")
        return value if isinstance(value, dict) else config
    value = getattr(config, "text_config", None)
    return value if value is not None else config


def _field(config, name: str):
    if isinstance(config, dict):
        return config.get(name)
    return getattr(config, name, None)


def _declared_layer_types(config, layer_count: int) -> tuple[str, ...]:
    text_config = _text_config(config)
    declared = _field(text_config, "layer_types")
    declared_count = _field(text_config, "num_hidden_layers")
    if declared_count is not None and (
        isinstance(declared_count, bool)
        or not isinstance(declared_count, int)
        or declared_count != layer_count
    ):
        raise ProofV3Error(
            "runtime layer count disagrees with the model configuration"
        )
    if declared is None:
        # Conventional transformer configs predate ``layer_types``.  They may
        # qualify only when the actual module inventory is uniformly full
        # attention; a hybrid model can never fall back through this branch.
        return ("full_attention",) * layer_count
    if (
        isinstance(declared, (str, bytes))
        or not isinstance(declared, Sequence)
        or len(declared) != layer_count
        or any(item not in _DECLARED_TO_PROTOCOL for item in declared)
    ):
        raise ProofV3Error(
            "model layer_types are missing, malformed, or unsupported"
        )
    return tuple(str(item) for item in declared)


def qualify_runtime_layer_kinds_v3(
    *,
    config,
    layers,
) -> tuple[str, ...]:
    """Return the exact protocol layer kinds after structural qualification."""

    layers = tuple(layers)
    if not layers:
        raise ProofV3Error("runtime model has no layers")
    declared = _declared_layer_types(config, len(layers))
    result = []
    all_architecture_operations = frozenset().union(
        *_ARCHITECTURE_OPERATIONS.values()
    )
    for layer_index, (layer, declared_kind) in enumerate(
        zip(layers, declared, strict=True)
    ):
        protocol_kind = _DECLARED_TO_PROTOCOL[declared_kind]
        specs = dense_execution_projection_specs(layer)
        operation_ids = tuple(item.operation_id for item in specs)
        if len(operation_ids) != len(set(operation_ids)):
            raise ProofV3Error(
                f"runtime layer {layer_index} has duplicate projection "
                "operations"
            )
        actual = frozenset(operation_ids)
        sparse_moe = is_moe_layer(layer)
        common = frozenset() if sparse_moe else _COMMON_OPERATIONS
        required = common | _ARCHITECTURE_OPERATIONS[protocol_kind]
        forbidden = (
            all_architecture_operations
            - _ARCHITECTURE_OPERATIONS[protocol_kind]
        )
        missing = required - actual
        extra_architecture = actual & forbidden
        mixed_dense_sparse = sparse_moe and bool(actual & _COMMON_OPERATIONS)
        if missing or extra_architecture or mixed_dense_sparse:
            raise ProofV3Error(
                f"runtime layer {layer_index} does not match declared "
                f"{declared_kind} projection inventory"
            )
        result.append(protocol_kind)
    return tuple(result)


def manifest_projection_suffix_v3(operation_id: str) -> str:
    """Map one qualified capture operation to its signed manifest suffix."""

    try:
        return _MANIFEST_SUFFIX_BY_OPERATION[str(operation_id)]
    except KeyError as exc:
        raise ProofV3Error(
            "runtime projection operation is not registered for proof v3"
        ) from exc


def manifest_projection_suffixes_v3() -> frozenset[str]:
    """Return the complete registered dense projection suffix inventory."""

    return frozenset(_MANIFEST_SUFFIX_BY_OPERATION.values())


def runtime_full_attention_layers_v3(
    layer_kinds,
) -> tuple[int, ...]:
    kinds = tuple(str(kind) for kind in layer_kinds)
    if not kinds or any(
        kind not in _ARCHITECTURE_OPERATIONS for kind in kinds
    ):
        raise ProofV3Error("runtime layer kinds are malformed")
    return tuple(
        index for index, kind in enumerate(kinds)
        if kind == "full_attention"
    )
