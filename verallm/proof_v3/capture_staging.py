"""Model-derived sizing for graph-integrated proof capture staging."""

from __future__ import annotations

import re

from verallm.proof_v3.errors import ProofV3Error
from verallm.proof_v3.economic_challenge import (
    PROMPT_CANDIDATE_ROWS_V3,
)
from verallm.proof_v3.projection_manifest import ProjectionManifestV3
from verallm.proof_v3.moe_runtime_semantics import (
    MoeRuntimeSemanticsV3,
    moe_runtime_encoding_bytes_v3,
)
from verallm.proof_v3.runtime_architecture import (
    manifest_projection_suffixes_v3,
)

CAPTURE_STAGING_BUDGET_BYTES_V3 = 5 << 29  # 2.5 GiB
MAX_CAPTURE_STAGING_ROWS_V3 = 2048

_LAYER_ENTRY = re.compile(r"^l([0-9]+)\.([a-z0-9_]+)$")

__all__ = [
    "CAPTURE_STAGING_BUDGET_BYTES_V3",
    "MAX_CAPTURE_STAGING_ROWS_V3",
    "capture_bytes_per_row_v3",
    "dense_capture_bytes_per_row_v3",
    "recommended_capture_staging_rows_v3",
    "recommended_dense_capture_staging_rows_v3",
    "response_stamp_capture_rows_v3",
    "sparse_moe_capture_bytes_per_row_v3",
]


def dense_capture_bytes_per_row_v3(
    manifest: ProjectionManifestV3,
) -> int:
    """Return exact fp16/bf16 buffer bytes allocated per scheduler row.

    Each registered dense projection owns one input and one output capture
    buffer. Each decoder layer additionally owns residual input/output
    buffers. ``gate_up`` is mandatory in the qualified dense shell and
    provides the residual hidden width without model-name assumptions.
    """

    if not isinstance(manifest, ProjectionManifestV3):
        raise ProofV3Error("capture staging manifest is malformed")
    suffixes = manifest_projection_suffixes_v3()
    projections: dict[int, dict[str, object]] = {}
    for entry in manifest.entries:
        match = _LAYER_ENTRY.fullmatch(entry.name)
        if match is None or match.group(2) not in suffixes:
            continue
        layer = int(match.group(1))
        suffix = match.group(2)
        layer_entries = projections.setdefault(layer, {})
        if suffix in layer_entries:
            raise ProofV3Error(
                "capture staging manifest has duplicate layer projections"
            )
        layer_entries[suffix] = entry
    layers = tuple(sorted(projections))
    if not layers or layers != tuple(range(len(layers))):
        raise ProofV3Error(
            "capture staging manifest layer inventory is not contiguous"
        )

    elements_per_row = 0
    for layer in layers:
        layer_entries = projections[layer]
        gate_up = layer_entries.get("gate_up")
        if gate_up is None:
            raise ProofV3Error(
                f"capture staging manifest lacks l{layer}.gate_up"
            )
        hidden = int(gate_up.in_dim)
        elements_per_row += 2 * hidden
        elements_per_row += sum(
            int(entry.in_dim) + int(entry.out_dim)
            for entry in layer_entries.values()
        )
    # Qualified runtime activation encodings are fp16.v1 or bf16.v1.
    return elements_per_row * 2


def recommended_dense_capture_staging_rows_v3(
    manifest: ProjectionManifestV3,
    *,
    budget_bytes: int = CAPTURE_STAGING_BUDGET_BYTES_V3,
    maximum_rows: int = MAX_CAPTURE_STAGING_ROWS_V3,
) -> int:
    """Choose a deterministic power-of-two scheduler chunk within budget."""

    if (
        isinstance(budget_bytes, bool)
        or not isinstance(budget_bytes, int)
        or budget_bytes <= 0
        or isinstance(maximum_rows, bool)
        or not isinstance(maximum_rows, int)
        or maximum_rows <= 0
    ):
        raise ProofV3Error("capture staging budget is malformed")
    bytes_per_row = dense_capture_bytes_per_row_v3(manifest)
    affordable = min(maximum_rows, budget_bytes // bytes_per_row)
    if affordable < 1:
        raise ProofV3Error(
            "capture staging budget cannot hold one scheduler row"
        )
    return 1 << (affordable.bit_length() - 1)


def sparse_moe_capture_bytes_per_row_v3(
    manifest: ProjectionManifestV3,
    *,
    semantics: MoeRuntimeSemanticsV3,
) -> int:
    """Return a platform-independent sparse capture allocation upper bound.

    The dense attention/GDN shell owns one input and output buffer per
    projection. Sparse blocks additionally own their input, router-logit and
    aggregate-output buffers, while decoder layers own all three residual
    boundaries. Expert weight matrices are authenticated static data and do
    not allocate one activation arena per expert.

    Qualified opaque-root adapters allocate a strict subset of these buffers,
    so using the complete generic sparse layout keeps the recommendation safe
    across Ampere buffer capture and split-mode Hopper/Blackwell capture.
    """

    if not isinstance(manifest, ProjectionManifestV3):
        raise ProofV3Error("capture staging manifest is malformed")
    if not isinstance(semantics, MoeRuntimeSemanticsV3):
        raise ProofV3Error("capture staging MoE semantics are malformed")
    if (
        not manifest.moe_runtime_semantics_digest
        or manifest.moe_runtime_semantics_digest != semantics.digest()
    ):
        raise ProofV3Error(
            "capture staging MoE semantics do not match the manifest"
        )

    semantic_layers = tuple(layer.layer_index for layer in semantics.layers)
    if (
        not semantic_layers
        or semantic_layers != tuple(range(len(semantic_layers)))
    ):
        raise ProofV3Error(
            "capture staging MoE layer inventory is not contiguous"
        )

    suffixes = manifest_projection_suffixes_v3() - {"gate_up", "down"}
    projections: dict[int, dict[str, object]] = {}
    for entry in manifest.entries:
        match = _LAYER_ENTRY.fullmatch(entry.name)
        if match is None or match.group(2) not in suffixes:
            continue
        layer = int(match.group(1))
        suffix = match.group(2)
        layer_entries = projections.setdefault(layer, {})
        if suffix in layer_entries:
            raise ProofV3Error(
                "capture staging manifest has duplicate layer projections"
            )
        layer_entries[suffix] = entry
    if tuple(sorted(projections)) != semantic_layers:
        raise ProofV3Error(
            "capture staging sparse shell does not match the MoE layers"
        )
    if any(not projections[layer] for layer in semantic_layers):
        raise ProofV3Error("capture staging sparse shell is empty")

    activation_bytes = moe_runtime_encoding_bytes_v3(
        semantics.runtime_encoding_id
    )
    router_bytes = moe_runtime_encoding_bytes_v3(
        semantics.router_encoding_id
    )
    shell_bytes = sum(
        (int(entry.in_dim) + int(entry.out_dim)) * activation_bytes
        for layer in semantic_layers
        for entry in projections[layer].values()
    )
    residual_bytes = (
        len(semantic_layers)
        * 3
        * semantics.hidden_size
        * activation_bytes
    )
    sparse_block_bytes = len(semantic_layers) * (
        2 * semantics.hidden_size * activation_bytes
        + semantics.num_experts * router_bytes
    )
    return shell_bytes + residual_bytes + sparse_block_bytes


def capture_bytes_per_row_v3(
    manifest: ProjectionManifestV3,
    *,
    moe_runtime_semantics: MoeRuntimeSemanticsV3 | None = None,
) -> int:
    """Dispatch capture sizing without changing the dense release path."""

    if not isinstance(manifest, ProjectionManifestV3):
        raise ProofV3Error("capture staging manifest is malformed")
    if manifest.moe_runtime_semantics_digest:
        if moe_runtime_semantics is None:
            raise ProofV3Error(
                "sparse capture staging requires MoE runtime semantics"
            )
        return sparse_moe_capture_bytes_per_row_v3(
            manifest,
            semantics=moe_runtime_semantics,
        )
    if moe_runtime_semantics is not None:
        raise ProofV3Error(
            "dense capture staging does not accept MoE runtime semantics"
        )
    return dense_capture_bytes_per_row_v3(manifest)


def recommended_capture_staging_rows_v3(
    manifest: ProjectionManifestV3,
    *,
    moe_runtime_semantics: MoeRuntimeSemanticsV3 | None = None,
    budget_bytes: int = CAPTURE_STAGING_BUDGET_BYTES_V3,
    maximum_rows: int = MAX_CAPTURE_STAGING_ROWS_V3,
) -> int:
    """Choose bounded staging rows for either dense or sparse manifests."""

    if (
        isinstance(budget_bytes, bool)
        or not isinstance(budget_bytes, int)
        or budget_bytes <= 0
        or isinstance(maximum_rows, bool)
        or not isinstance(maximum_rows, int)
        or maximum_rows <= 0
    ):
        raise ProofV3Error("capture staging budget is malformed")
    bytes_per_row = capture_bytes_per_row_v3(
        manifest,
        moe_runtime_semantics=moe_runtime_semantics,
    )
    affordable = min(maximum_rows, budget_bytes // bytes_per_row)
    if affordable < 1:
        raise ProofV3Error(
            "capture staging budget cannot hold one scheduler row"
        )
    return 1 << (affordable.bit_length() - 1)


def response_stamp_capture_rows_v3(
    *,
    max_requests: int,
    max_batched_tokens: int,
) -> int:
    """Bound the layer-zero stamp arena at scheduler admission capacity.

    Every co-batched request can contribute at most the canonical prompt-tail
    width in one prefill step. The scheduler token cap is a tighter bound when
    fewer total rows can run together. Unlike the full-model replay gather,
    this arena contains one layer-zero input only and therefore scales safely
    with admitted request concurrency.
    """

    if (
        isinstance(max_requests, bool)
        or not isinstance(max_requests, int)
        or max_requests <= 0
        or isinstance(max_batched_tokens, bool)
        or not isinstance(max_batched_tokens, int)
        or max_batched_tokens <= 0
    ):
        raise ProofV3Error("response-stamp scheduler capacity is malformed")
    return min(
        max_batched_tokens,
        max_requests * PROMPT_CANDIDATE_ROWS_V3,
    )
