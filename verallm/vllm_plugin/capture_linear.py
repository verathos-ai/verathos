"""Post-load wrappers for proof-bound dense linear operations.

``CaptureLinearWrapper`` is intentionally operation agnostic.  Its caller owns
the witness names and split-op kind identifiers, so one implementation can
capture the complete registered linear shell without inferring semantics from
module class names at execution time.
"""

import logging
from dataclasses import dataclass

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def qkv_kv_output_root_slice_v3(
    projection: nn.Module,
    output_dim: int,
) -> tuple[int, int]:
    """Return the local fused-QKV suffix containing exactly K and V."""

    try:
        partitions = tuple(
            int(value)
            for value in projection.output_partition_sizes
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(
            "QKV projection lacks canonical partition geometry"
        ) from exc
    if (
        isinstance(output_dim, bool)
        or not isinstance(output_dim, int)
        or output_dim <= 0
        or len(partitions) != 3
        or any(value <= 0 for value in partitions)
        or sum(partitions) != output_dim
    ):
        raise ValueError(
            "QKV projection has inconsistent partition geometry"
        )
    return partitions[0], sum(partitions)


@dataclass(frozen=True)
class DenseProjectionCaptureSpec:
    """One runtime projection and its canonical capture ABI identity."""

    operation_id: str
    parent: nn.Module
    module: nn.Module
    input_kind: int
    output_kind: int
    input_suffix: str
    output_suffix: str


@dataclass(frozen=True)
class _RuntimeRootBinding:
    """One graph-integrated runtime root stage awaiting shared storage."""

    stage_id: str
    row_width: int
    owner: nn.Module
    root_attribute: str
    staging_attribute: str
    dtype: torch.dtype
    history_attribute: str | None = None


def _deduplicate_decoder_boundary_root_bindings(
    bindings: tuple[_RuntimeRootBinding, ...],
) -> tuple[
    tuple[_RuntimeRootBinding, ...],
    dict[str, str],
]:
    """Share reducer slots for exact runtime tensor aliases."""

    by_stage = {item.stage_id: item for item in bindings}
    aliases: dict[str, str] = {}
    retained = []
    for item in bindings:
        layer_text, separator, suffix = item.stage_id.partition(".")
        source = None
        if (
            separator
            and suffix == "residual_in"
            and layer_text.startswith("l")
            and layer_text[1:].isdigit()
        ):
            layer = int(layer_text[1:])
            if layer > 0:
                candidate = by_stage.get(f"l{layer - 1}.residual_out")
                if (
                    candidate is not None
                    and candidate.row_width == item.row_width
                    and candidate.dtype == item.dtype
                ):
                    source = candidate.stage_id
        if source is None:
            retained.append(item)
        else:
            aliases[item.stage_id] = source
    return tuple(retained), aliases


def dense_execution_projection_specs(
    layer: nn.Module,
) -> tuple[DenseProjectionCaptureSpec, ...]:
    """Discover the exact dense Qwen hybrid linear shell for one layer.

    This performs structural discovery only.  The authenticated manifest is
    still the authority for which operation identities and dimensions the
    verifier accepts.
    """

    from verallm.proof_v2.layout import (
        FULL_OUTPUT_OPERATION_ID,
        FULL_QKV_OPERATION_ID,
        GDN_BA_OPERATION_ID,
        GDN_OUTPUT_OPERATION_ID,
        GDN_QKVZ_OPERATION_ID,
        MLP_DOWN_OPERATION_ID,
        MLP_GATE_UP_OPERATION_ID,
    )

    result = []

    def add(
        operation_id: str,
        parent: nn.Module | None,
        attribute: str,
        input_kind: int,
        output_kind: int,
        input_suffix: str,
        output_suffix: str,
    ) -> None:
        if parent is None:
            return
        module = getattr(parent, attribute, None)
        if module is None:
            return
        result.append(
            DenseProjectionCaptureSpec(
                operation_id,
                parent,
                module,
                input_kind,
                output_kind,
                input_suffix,
                output_suffix,
            )
        )

    mlp = getattr(layer, "mlp", None)
    add(
        MLP_GATE_UP_OPERATION_ID,
        mlp,
        "gate_up_proj",
        15,
        16,
        "mlp_gate_up_input",
        "mlp_gate_up_output",
    )
    add(
        MLP_DOWN_OPERATION_ID,
        mlp,
        "down_proj",
        17,
        18,
        "mlp_down_input",
        "mlp_down_output",
    )

    attention = getattr(layer, "self_attn", None)
    add(
        FULL_QKV_OPERATION_ID,
        attention,
        "qkv_proj",
        5,
        6,
        "attention_qkv_input",
        "attention_qkv_output",
    )
    add(
        FULL_OUTPUT_OPERATION_ID,
        attention,
        "o_proj",
        7,
        8,
        "attention_o_input",
        "attention_o_output",
    )

    gdn = getattr(layer, "linear_attn", None)
    add(
        GDN_QKVZ_OPERATION_ID,
        gdn,
        "in_proj_qkvz",
        9,
        10,
        "gdn_qkvz_input",
        "gdn_qkvz_output",
    )
    add(
        GDN_BA_OPERATION_ID,
        gdn,
        "in_proj_ba",
        11,
        12,
        "gdn_ba_input",
        "gdn_ba_output",
    )
    add(
        GDN_OUTPUT_OPERATION_ID,
        gdn,
        "out_proj",
        13,
        14,
        "gdn_o_input",
        "gdn_o_output",
    )
    return tuple(sorted(result, key=lambda item: item.operation_id))


class CaptureLinearWrapper(nn.Module):
    """Transparent wrapper that captures gate_proj input activations.

    Two modes:

    **Split mode** (``use_buffer=False``, default for MoE):
        Inserts a ``verallm::capture`` custom op that forces a piecewise
        CUDA graph split.  The op delegates to
        ``RequestActivationTracker.capture_at_split()`` for per-request
        row slicing.  Adds one graph split point per layer.

    **Buffer mode** (``use_buffer=True``, default for dense models):
        Pre-allocates a GPU buffer and uses native ``aten::copy_`` inside
        the CUDA graph — NO split point, NO custom op dispatch.
        The buffer is read after each engine step by
        ``RequestActivationTracker._readout_buffers()``.

    The ``__getattr__`` fallback delegates attribute lookups (e.g.
    ``.weight``, ``.qweight``, ``.scales``) to the wrapped module,
    making the wrapper transparent to quant-detection and weight-
    extraction code.
    """

    def __init__(self, original: nn.Module, layer_idx: int,
                 *, use_buffer: bool = False, max_tokens: int = 0,
                 hidden_dim: int = 0, dtype: torch.dtype = torch.bfloat16,
                 use_triton_copy: bool = False, output_dim: int = 0,
                 root_max_tokens: int = 0,
                 input_kind: int = 0, output_kind: int = 2,
                 input_suffix: str = "mlp_gate_input",
                 output_suffix: str = "mlp_gate_output",
                 output_root_suffix: str | None = None,
                 output_root_slice: tuple[int, int] | None = None,
                 row_indices: torch.Tensor | None = None,
                 root_stage_suffixes: frozenset[str] | None = None,
                 required_output_dtype: torch.dtype | None = None,
                 diagnostic_expose_root_tensors: bool = False):
        super().__init__()
        # NOTE: self.original is assigned at the END of __init__.  With it
        # set early, nn.Module.register_buffer's hasattr() probe would fall
        # through __getattr__ into a WRAPPED CaptureLinearWrapper's buffers
        # and raise "attribute already exists" -- nested wrapping (e.g. the
        # reduction-audit aux capture around a gather-mode trace wrapper)
        # must stay legal.
        self._layer_idx = layer_idx
        self._use_buffer = use_buffer
        self._use_triton_copy = use_triton_copy
        self._diagnostic_expose_root_tensors = bool(
            diagnostic_expose_root_tensors
        )
        self._capture_live_input = None
        self._capture_live_output = None
        if not isinstance(input_kind, int) or not isinstance(output_kind, int):
            raise TypeError("capture tensor kinds must be integers")
        for value, name in (
            (input_suffix, "capture input suffix"),
            (output_suffix, "capture output suffix"),
        ):
            if (
                not isinstance(value, str)
                or not value
                or not value.replace("_", "").isalnum()
            ):
                raise ValueError(f"{name} is invalid")
        self._capture_input_kind = input_kind
        self._capture_output_kind = output_kind
        self._capture_input_suffix = input_suffix
        self._capture_output_suffix = output_suffix
        if output_root_suffix is None:
            output_root_suffix = output_suffix
        if (
            not isinstance(output_root_suffix, str)
            or not output_root_suffix
            or not output_root_suffix.replace("_", "").isalnum()
        ):
            raise ValueError("capture output root suffix is invalid")
        self._capture_output_root_suffix = output_root_suffix
        if output_root_slice is None:
            output_root_slice = (0, output_dim)
        if (
            not isinstance(output_root_slice, tuple)
            or len(output_root_slice) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in output_root_slice
            )
            or not (
                (
                    output_dim == 0
                    and output_root_slice == (0, 0)
                )
                or 0 <= output_root_slice[0] < output_root_slice[1]
            )
            or output_root_slice[1] > output_dim
        ):
            raise ValueError("capture output root slice is invalid")
        self._capture_output_root_slice = output_root_slice
        self._capture_input_dim = int(hidden_dim)
        self._capture_output_root_dim = (
            output_root_slice[1] - output_root_slice[0]
        )
        self._capture_dtype = dtype
        if (
            required_output_dtype is not None
            and required_output_dtype
            not in (torch.float16, torch.bfloat16, torch.float32)
        ):
            raise ValueError("required capture output dtype is unsupported")
        self._capture_required_output_dtype = required_output_dtype
        self.register_buffer(
            "_capture_row_indices",
            row_indices,
            persistent=False,
        )
        # Root selection and raw selected-row retention are independent.
        # ``root_stage_suffixes`` limits the cheap always-on commitments, but
        # a post-nonce replay may select any registered projection stage.
        # CUDA-graph split callbacks do not execute again at replay time, so
        # every split projection needs the bounded gather buffers whenever a
        # shared row-index tensor is installed.
        retain_selected_input = row_indices is not None
        retain_selected_output = row_indices is not None
        if (
            (use_buffer or retain_selected_input)
            and max_tokens > 0
            and hidden_dim > 0
        ):
            # Pre-allocate ON THE SAME DEVICE as the wrapped module.
            # The hook creates us after load_model(), so the model is
            # already on CUDA — torch.zeros() defaults to CPU which
            # would break CUDA graph capture (CPU↔GPU copy unsupported).
            dev = next(original.parameters()).device
            self.register_buffer(
                '_capture_buf',
                torch.zeros(max_tokens, hidden_dim, dtype=dtype, device=dev),
            )
        else:
            self.register_buffer('_capture_buf', None)
        if (
            (use_buffer or retain_selected_output)
            and max_tokens > 0
            and output_dim > 0
        ):
            dev = next(original.parameters()).device
            self.register_buffer(
                '_capture_output_buf',
                torch.zeros(max_tokens, output_dim, dtype=dtype, device=dev),
            )
        else:
            self.register_buffer('_capture_output_buf', None)
        root_device = None
        if root_max_tokens > 0:
            root_device = next(original.parameters()).device
        for name in (
            "_capture_input_root_buf",
            "_capture_output_root_buf",
        ):
            is_input = name == "_capture_input_root_buf"
            dimension = (
                hidden_dim
                if is_input
                else output_root_slice[1] - output_root_slice[0]
            )
            suffix = (
                input_suffix if is_input else output_root_suffix
            )
            root_enabled = (
                root_stage_suffixes is None
                or suffix in root_stage_suffixes
            )
            self.register_buffer(
                name,
                torch.zeros(
                    root_max_tokens,
                    32,
                    dtype=torch.uint8,
                    device=root_device,
                )
                if (
                    root_device is not None
                    and dimension > 0
                    and root_enabled
                )
                else None,
                persistent=False,
            )
        for name in (
            "_capture_input_root_history_buf",
            "_capture_output_root_history_buf",
        ):
            self.register_buffer(name, None, persistent=False)
        self.register_buffer(
            "_capture_input_root_stage_buf",
            None,
            persistent=False,
        )
        self.register_buffer(
            "_capture_output_root_stage_buf",
            None,
            persistent=False,
        )
        self.register_buffer(
            "_capture_root_retained_values",
            None,
            persistent=False,
        )
        self.register_buffer(
            "_capture_root_retained_hashes",
            None,
            persistent=False,
        )
        self.register_buffer(
            "_capture_root_retention_slots",
            None,
            persistent=False,
        )
        self._capture_root_retention_indices: dict[str, int] = {}
        self.original = original

    def _capture_roots(
        self,
        destination: torch.Tensor | None,
        staging: torch.Tensor | None,
        source: torch.Tensor,
        suffix: str,
    ) -> None:
        if destination is None:
            return
        from verallm.proof_v3.execution_anchor import (
            execution_anchor_lane_bytes_v3,
        )

        lane_bytes = execution_anchor_lane_bytes_v3(
            f"l{self._layer_idx}.{suffix}"
        )
        if staging is None:
            torch.ops.verallm.activation_row_roots(
                destination,
                source,
                lane_bytes,
            )
        elif (
            suffix in self._capture_root_retention_indices
            and self._capture_root_retained_values is not None
            and self._capture_root_retained_hashes is not None
            and self._capture_root_retention_slots is not None
        ):
            torch.ops.verallm.activation_row_roots_or_stage_retain(
                destination,
                staging,
                self._capture_root_retained_values,
                self._capture_root_retained_hashes,
                self._capture_root_retention_slots,
                source,
                self._capture_root_retention_indices[suffix],
                lane_bytes,
            )
        else:
            torch.ops.verallm.activation_row_roots_or_stage(
                destination,
                staging,
                source,
                lane_bytes,
            )

    def _copy_capture_buffer(self, destination, source) -> None:
        if self._capture_row_indices is not None:
            if source.is_cuda:
                torch.ops.verallm.buffer_gather_rows(
                    destination,
                    source,
                    self._capture_row_indices,
                )
            else:
                valid = (
                    (self._capture_row_indices >= 0)
                    & (self._capture_row_indices < source.shape[0])
                )
                if bool(valid.any()):
                    positions = torch.nonzero(valid, as_tuple=False).flatten()
                    destination.index_copy_(
                        0,
                        positions,
                        source.index_select(
                            0,
                            self._capture_row_indices.index_select(0, positions),
                        ),
                    )
            return
        n = min(source.shape[0], destination.shape[0])
        if self._use_triton_copy:
            torch.ops.verallm.buffer_copy(destination, source, n)
        else:
            destination[:n].copy_(source[:n])

    @staticmethod
    def _canonical_capture_value(
        source: torch.Tensor,
        destination: torch.Tensor | None,
    ) -> torch.Tensor:
        """Encode roots and selected raw rows in the same signed dtype."""

        if destination is not None and source.dtype != destination.dtype:
            return source.to(dtype=destination.dtype)
        return source

    def __getattr__(self, name: str):
        # nn.Module.__getattr__ checks _parameters, _buffers, _modules.
        # This fallback delegates to the wrapped module for everything
        # else (.weight, .qweight, .weight_packed, .scales, etc.).
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name == "original":
                # During __init__ (buffers register before self.original
                # is assigned) delegation must fail plainly instead of
                # recursing on the missing attribute.
                raise
            return getattr(self.original, name)

    def forward(self, x):
        if self._diagnostic_expose_root_tensors:
            if self._capture_input_root_buf is not None:
                self._capture_live_input = x
            y = self.original(x)
            y_tensor = y[0] if isinstance(y, tuple) else y
            if not isinstance(y_tensor, torch.Tensor):
                raise TypeError("captured linear output is not a tensor")
            if (
                self._capture_required_output_dtype is not None
                and y_tensor.dtype != self._capture_required_output_dtype
            ):
                raise RuntimeError(
                    "captured operation output dtype does not match the "
                    "signed runtime contract"
                )
            if self._capture_output_root_buf is not None:
                root_start, root_stop = self._capture_output_root_slice
                self._capture_live_output = y_tensor[:, root_start:root_stop]
            return y
        capture_x = self._canonical_capture_value(
            x,
            (
                self._capture_buf
                if self._capture_buf is not None
                else self._capture_input_root_stage_buf
            ),
        )
        self._capture_roots(
            self._capture_input_root_buf,
            self._capture_input_root_stage_buf,
            capture_x,
            self._capture_input_suffix,
        )
        if self._capture_buf is not None:
            self._copy_capture_buffer(self._capture_buf, capture_x)
        if not self._use_buffer:
            # Split mode: custom op forces piecewise graph split.
            _ = torch.ops.verallm.capture(
                x,
                self._layer_idx,
                self._capture_input_kind,
            )
        y = self.original(x)
        y_tensor = y[0] if isinstance(y, tuple) else y
        if not isinstance(y_tensor, torch.Tensor):
            raise TypeError("captured linear output is not a tensor")
        if (
            self._capture_required_output_dtype is not None
            and y_tensor.dtype != self._capture_required_output_dtype
        ):
            raise RuntimeError(
                "captured operation output dtype does not match the signed "
                "runtime contract"
            )
        capture_y = self._canonical_capture_value(
            y_tensor,
            (
                self._capture_output_buf
                if self._capture_output_buf is not None
                else self._capture_output_root_stage_buf
            ),
        )
        root_start, root_stop = self._capture_output_root_slice
        self._capture_roots(
            self._capture_output_root_buf,
            self._capture_output_root_stage_buf,
            capture_y[:, root_start:root_stop],
            self._capture_output_root_suffix,
        )
        if self._capture_output_buf is not None:
            self._copy_capture_buffer(self._capture_output_buf, capture_y)
        if not self._use_buffer:
            _ = torch.ops.verallm.capture(
                y_tensor,
                self._layer_idx,
                self._capture_output_kind,
            )
        return y

    def proof_capture_root_buffers(self):
        return tuple(
            (
                self._layer_idx,
                suffix,
                (
                    history_buffer
                    if history_buffer is not None
                    else buffer
                ),
                dimension * self._capture_buf_element_size(),
            )
            for suffix, buffer, history_buffer, dimension in (
                (
                    self._capture_input_suffix,
                    self._capture_input_root_buf,
                    self._capture_input_root_history_buf,
                    self._capture_input_dim,
                ),
                (
                    self._capture_output_root_suffix,
                    self._capture_output_root_buf,
                    self._capture_output_root_history_buf,
                    self._capture_output_root_dim,
                ),
            )
            if buffer is not None
        )

    def _runtime_root_bindings(self) -> tuple[_RuntimeRootBinding, ...]:
        element_size = self._capture_buf_element_size()
        return tuple(
            _RuntimeRootBinding(
                stage_id=f"l{self._layer_idx}.{suffix}",
                row_width=dimension * element_size,
                owner=self,
                root_attribute=root_attribute,
                staging_attribute=staging_attribute,
                dtype=self._capture_dtype,
                history_attribute=history_attribute,
            )
            for (
                suffix,
                dimension,
                root_attribute,
                staging_attribute,
                history_attribute,
            ) in (
                (
                    self._capture_input_suffix,
                    self._capture_input_dim,
                    "_capture_input_root_buf",
                    "_capture_input_root_stage_buf",
                    "_capture_input_root_history_buf",
                ),
                (
                    self._capture_output_root_suffix,
                    self._capture_output_root_dim,
                    "_capture_output_root_buf",
                    "_capture_output_root_stage_buf",
                    "_capture_output_root_history_buf",
                ),
            )
            if getattr(self, root_attribute) is not None
        )

    def _install_runtime_root_retention(
        self,
        *,
        suffix: str,
        stage_index: int,
        values: torch.Tensor,
        hashes: torch.Tensor,
        slots: torch.Tensor,
    ) -> None:
        """Install shared graph-static storage for post-nonce lane capture."""

        self._capture_root_retained_values = values
        self._capture_root_retained_hashes = hashes
        self._capture_root_retention_slots = slots
        self._capture_root_retention_indices[suffix] = int(stage_index)

    def proof_capture_root_retention(self):
        return tuple(
            (
                f"l{self._layer_idx}.{suffix}",
                stage_index,
                self._capture_root_retained_values,
                self._capture_root_retained_hashes,
                self._capture_root_retention_slots,
            )
            for suffix, stage_index in sorted(
                self._capture_root_retention_indices.items()
            )
        )

    def proof_capture_split_stages(self):
        """Return raw stage identities emitted through capture_at_split()."""

        if self._use_buffer:
            return ()
        return (
            (self._layer_idx, self._capture_input_suffix),
            (self._layer_idx, self._capture_output_suffix),
        )

    def proof_capture_split_row_aliases(self):
        """Expose split-output slices that back distinct committed stages."""

        root_start, root_stop = self._capture_output_root_slice
        if (
            self._use_buffer
            or self._capture_output_root_suffix
            == self._capture_output_suffix
            or root_stop <= root_start
        ):
            return ()
        return (
            (
                self._layer_idx,
                self._capture_output_suffix,
                self._capture_output_root_suffix,
                root_start,
                root_stop,
            ),
        )

    def proof_capture_root_row_aliases(
        self,
        output_buffer: torch.Tensor | None = None,
    ):
        """Expose exact raw-row views for roots whose stage differs.

        Lean attention commits only the K/V suffix of the fused QKV output.
        The full output buffer already contains those graph-captured values,
        so replay retention can consume a view of that slice without another
        allocation or graph operation.  A whole-step outer capture may supply
        its buffer so the same view can retain post-nonce lanes across prefill.
        """

        root_start, root_stop = self._capture_output_root_slice
        source = (
            self._capture_output_buf
            if output_buffer is None
            else output_buffer
        )
        if (
            source is None
            or source.ndim != 2
            or int(source.shape[1]) < root_stop
            or self._capture_output_root_suffix
            == self._capture_output_suffix
            or root_stop <= root_start
        ):
            return ()
        return (
            (
                self._layer_idx,
                self._capture_output_root_suffix,
                source[:, root_start:root_stop],
            ),
        )

    def _capture_buf_element_size(self) -> int:
        return int(torch.empty((), dtype=self._capture_dtype).element_size())


class CaptureDecoderLayerWrapper(nn.Module):
    """Capture the residual-stream boundaries around one decoder layer."""

    def __init__(
        self,
        original: nn.Module,
        layer_idx: int,
        *,
        use_buffer: bool = False,
        max_tokens: int = 0,
        hidden_dim: int = 0,
        dtype: torch.dtype = torch.bfloat16,
        use_triton_copy: bool = False,
        root_max_tokens: int = 0,
        row_indices: torch.Tensor | None = None,
        root_stage_suffixes: frozenset[str] | None = None,
        response_stamp_max_tokens: int = 0,
        response_stamp_row_indices: torch.Tensor | None = None,
        response_stamp_only: bool = False,
        diagnostic_expose_root_tensors: bool = False,
    ):
        super().__init__()
        self.original = original
        self._layer_idx = layer_idx
        self._use_buffer = use_buffer
        self._use_triton_copy = use_triton_copy
        self._capture_hidden_dim = int(hidden_dim)
        self._capture_dtype = dtype
        self._response_stamp_only = bool(response_stamp_only)
        self._diagnostic_expose_root_tensors = bool(
            diagnostic_expose_root_tensors
        )
        if self._response_stamp_only and (
            layer_idx != 0
            or response_stamp_max_tokens <= 0
            or response_stamp_row_indices is None
            or use_buffer is False
        ):
            raise ValueError(
                "response-stamp-only capture requires a buffered layer-zero "
                "arena"
            )
        self._capture_live_residual_after_attention = None
        self._capture_live_residual_out_hidden = None
        self._capture_live_residual_out_residual = None
        self.register_buffer(
            "_capture_row_indices",
            row_indices,
            persistent=False,
        )
        self.register_buffer(
            "_capture_response_stamp_row_indices",
            response_stamp_row_indices,
            persistent=False,
        )
        capture_device = None
        if (
            max_tokens > 0
            or root_max_tokens > 0
            or response_stamp_max_tokens > 0
        ) and hidden_dim > 0:
            capture_device = next(original.parameters()).device
        buffer_device = None
        if (
            (use_buffer or row_indices is not None)
            and max_tokens > 0
            and hidden_dim > 0
        ):
            buffer_device = capture_device
        for name in (
            "_capture_residual_in_buf",
            "_capture_residual_after_attention_buf",
            "_capture_residual_out_buf",
        ):
            self.register_buffer(
                name,
                torch.zeros(
                    max_tokens,
                    hidden_dim,
                    dtype=dtype,
                    device=buffer_device,
                )
                if buffer_device is not None
                else None,
            )
        self.register_buffer(
            "_capture_response_stamp_buf",
            torch.zeros(
                response_stamp_max_tokens,
                hidden_dim,
                dtype=dtype,
                device=capture_device,
            )
            if (
                capture_device is not None
                and response_stamp_row_indices is not None
                and response_stamp_max_tokens > 0
                and hidden_dim > 0
            )
            else None,
            persistent=False,
        )
        for name in (
            "_capture_residual_in_root_buf",
            "_capture_residual_after_attention_root_buf",
            "_capture_residual_out_root_buf",
        ):
            suffix = name.removeprefix("_capture_").removesuffix(
                "_root_buf"
            )
            root_enabled = (
                root_stage_suffixes is None
                or suffix in root_stage_suffixes
            )
            self.register_buffer(
                name,
                torch.zeros(
                    root_max_tokens,
                    32,
                    dtype=torch.uint8,
                    device=capture_device,
                )
                if (
                    capture_device is not None
                    and root_max_tokens > 0
                    and root_enabled
                )
                else None,
                persistent=False,
            )
        for name in (
            "_capture_residual_in_root_stage_buf",
            "_capture_residual_after_attention_root_stage_buf",
            "_capture_residual_out_root_stage_buf",
            "_capture_root_batch_destination",
            "_capture_root_batch_scratch",
            "_capture_root_batch_staging",
            "_capture_root_batch_widths",
        ):
            self.register_buffer(name, None, persistent=False)
        for name in (
            "_capture_root_retained_values",
            "_capture_root_retained_hashes",
            "_capture_root_retention_slots",
        ):
            self.register_buffer(name, None, persistent=False)
        self._capture_root_retention_indices: dict[str, int] = {}
        self._capture_root_alias_sources: dict[str, str] = {}

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.original, name)

    def _capture(self, value: torch.Tensor, kind: int, buffer) -> None:
        if buffer is not None:
            if self._capture_row_indices is not None and value.is_cuda:
                torch.ops.verallm.buffer_gather_rows(
                    buffer,
                    value,
                    self._capture_row_indices,
                )
            elif self._capture_row_indices is not None:
                valid = (
                    (self._capture_row_indices >= 0)
                    & (self._capture_row_indices < value.shape[0])
                )
                if bool(valid.any()):
                    positions = torch.nonzero(valid, as_tuple=False).flatten()
                    buffer.index_copy_(
                        0,
                        positions,
                        value.index_select(
                            0,
                            self._capture_row_indices.index_select(0, positions),
                        ),
                    )
            elif self._use_triton_copy:
                count = min(int(value.shape[0]), int(buffer.shape[0]))
                torch.ops.verallm.buffer_copy(buffer, value, count)
            else:
                count = min(int(value.shape[0]), int(buffer.shape[0]))
                buffer[:count].copy_(value[:count])
        if not self._use_buffer:
            _ = torch.ops.verallm.capture(value, self._layer_idx, kind)

    def _capture_roots(
        self,
        value: torch.Tensor,
        suffix: str,
        buffer: torch.Tensor | None,
        staging: torch.Tensor | None,
    ) -> None:
        if buffer is None or suffix in self._capture_root_alias_sources:
            return
        from verallm.proof_v3.execution_anchor import (
            execution_anchor_lane_bytes_v3,
        )

        lane_bytes = execution_anchor_lane_bytes_v3(
            f"l{self._layer_idx}.{suffix}"
        )
        if staging is None:
            torch.ops.verallm.activation_row_roots(
                buffer,
                value,
                lane_bytes,
            )
        elif (
            suffix in self._capture_root_retention_indices
            and self._capture_root_retained_values is not None
            and self._capture_root_retained_hashes is not None
            and self._capture_root_retention_slots is not None
        ):
            torch.ops.verallm.activation_row_roots_or_stage_retain(
                buffer,
                staging,
                self._capture_root_retained_values,
                self._capture_root_retained_hashes,
                self._capture_root_retention_slots,
                value,
                self._capture_root_retention_indices[suffix],
                lane_bytes,
            )
        else:
            torch.ops.verallm.activation_row_roots_or_stage(
                buffer,
                staging,
                value,
                lane_bytes,
            )

    @staticmethod
    def _canonical_capture_value(
        value: torch.Tensor,
        buffer: torch.Tensor | None,
    ) -> torch.Tensor:
        """Encode roots and selected raw rows in the same signed dtype."""

        if buffer is not None and value.dtype != buffer.dtype:
            return value.to(dtype=buffer.dtype)
        return value

    def proof_capture_buffers(self):
        return tuple(
            (self._layer_idx, suffix, buffer)
            for suffix, buffer in (
                ("residual_in", self._capture_residual_in_buf),
                (
                    "residual_after_attention",
                    self._capture_residual_after_attention_buf,
                ),
                ("residual_out", self._capture_residual_out_buf),
            )
            if buffer is not None
        )

    def proof_capture_response_stamp_buffer(self):
        """Expose the concurrency-scaled layer-zero response-stamp arena."""

        if (
            self._capture_response_stamp_buf is None
            or self._capture_response_stamp_row_indices is None
        ):
            return ()
        return (
            (
                self._layer_idx,
                "residual_in",
                self._capture_response_stamp_buf,
                self._capture_response_stamp_row_indices,
            ),
        )

    def proof_capture_root_buffers(self):
        row_width = (
            self._capture_hidden_dim
            * int(torch.empty((), dtype=self._capture_dtype).element_size())
        )
        return tuple(
            (self._layer_idx, suffix, buffer, row_width)
            for suffix, buffer in (
                ("residual_in", self._capture_residual_in_root_buf),
                (
                    "residual_after_attention",
                    self._capture_residual_after_attention_root_buf,
                ),
                ("residual_out", self._capture_residual_out_root_buf),
            )
            if buffer is not None
        )

    def proof_capture_split_stages(self):
        """Return residual stages emitted through capture_at_split()."""

        if self._use_buffer or self._response_stamp_only:
            return ()
        return (
            (self._layer_idx, "residual_in"),
            (self._layer_idx, "residual_after_attention"),
            (self._layer_idx, "residual_out"),
        )

    def _runtime_root_bindings(self) -> tuple[_RuntimeRootBinding, ...]:
        row_width = (
            self._capture_hidden_dim
            * int(torch.empty((), dtype=self._capture_dtype).element_size())
        )
        return tuple(
            _RuntimeRootBinding(
                stage_id=f"l{self._layer_idx}.{suffix}",
                row_width=row_width,
                owner=self,
                root_attribute=root_attribute,
                staging_attribute=staging_attribute,
                dtype=self._capture_dtype,
            )
            for suffix, root_attribute, staging_attribute in (
                (
                    "residual_in",
                    "_capture_residual_in_root_buf",
                    "_capture_residual_in_root_stage_buf",
                ),
                (
                    "residual_after_attention",
                    "_capture_residual_after_attention_root_buf",
                    "_capture_residual_after_attention_root_stage_buf",
                ),
                (
                    "residual_out",
                    "_capture_residual_out_root_buf",
                    "_capture_residual_out_root_stage_buf",
                ),
            )
            if getattr(self, root_attribute) is not None
        )

    def _install_runtime_root_retention(
        self,
        *,
        suffix: str,
        stage_index: int,
        values: torch.Tensor,
        hashes: torch.Tensor,
        slots: torch.Tensor,
    ) -> None:
        self._capture_root_retained_values = values
        self._capture_root_retained_hashes = hashes
        self._capture_root_retention_slots = slots
        self._capture_root_retention_indices[suffix] = int(stage_index)

    def _install_runtime_root_alias(
        self,
        *,
        suffix: str,
        source_stage_id: str,
    ) -> None:
        self._capture_root_alias_sources[str(suffix)] = str(source_stage_id)

    def proof_capture_root_retention(self):
        return tuple(
            (
                f"l{self._layer_idx}.{suffix}",
                stage_index,
                self._capture_root_retained_values,
                self._capture_root_retained_hashes,
                self._capture_root_retention_slots,
            )
            for suffix, stage_index in sorted(
                self._capture_root_retention_indices.items()
            )
        )

    @staticmethod
    def _locate_hidden_residual(args, kwargs):
        """Find (hidden_states, residual) regardless of the decoder-layer
        call convention.  Older vLLM calls ``layer(hidden_states, residual)``;
        vLLM 0.19+ Qwen2 calls ``layer(positions, hidden_states, residual)``
        where ``positions`` is a 1-D int tensor.  We pick the 2-D floating
        tensors (hidden + residual), which is stable across both."""

        two_d = [
            a for a in args
            if isinstance(a, torch.Tensor) and a.dim() == 2 and a.is_floating_point()
        ]
        if two_d:
            hidden_states = two_d[0]
            residual = two_d[1] if len(two_d) > 1 else kwargs.get("residual")
        else:
            hidden_states = kwargs.get("hidden_states")
            residual = kwargs.get("residual")
        return hidden_states, residual

    def forward(self, *args, **kwargs):
        hidden_states, residual = self._locate_hidden_residual(args, kwargs)
        if self._response_stamp_only:
            if not isinstance(hidden_states, torch.Tensor):
                raise TypeError(
                    "response-stamp decoder layer has no hidden-state tensor"
                )
            residual_in = (
                hidden_states
                if residual is None
                else hidden_states + residual
            )
            self._capture_response_stamp(
                self._canonical_capture_value(
                    residual_in,
                    self._capture_response_stamp_buf,
                )
            )
            return self.original(*args, **kwargs)
        if self._diagnostic_expose_root_tensors:
            result = self.original(*args, **kwargs)
            if not isinstance(result, tuple) or len(result) < 2:
                raise TypeError(
                    "captured decoder layer must return hidden and residual"
                )
            hidden_out, residual_after_attention = result[:2]
            if not isinstance(hidden_out, torch.Tensor) or not isinstance(
                residual_after_attention,
                torch.Tensor,
            ):
                raise TypeError(
                    "captured decoder layer returned non-tensor state"
                )
            if self._capture_residual_after_attention_root_buf is not None:
                self._capture_live_residual_after_attention = (
                    residual_after_attention
                )
            if self._capture_residual_out_root_buf is not None:
                # Keep both already-computed result tensors live. The signed
                # residual-out sum is deliberately deferred until after graph
                # replay so this diagnostic adds no CUDA kernel to the model.
                self._capture_live_residual_out_hidden = hidden_out
                self._capture_live_residual_out_residual = (
                    residual_after_attention
                )
            return result
        residual_in_required = (
            self._capture_residual_in_buf is not None
            or "residual_in" not in self._capture_root_alias_sources
        )
        residual_in = None
        if residual_in_required:
            residual_in = (
                hidden_states
                if residual is None
                else hidden_states + residual
            )
        result = self.original(*args, **kwargs)
        if not isinstance(result, tuple) or len(result) < 2:
            raise TypeError("captured decoder layer must return hidden and residual")
        hidden_out, residual_after_attention = result[:2]
        if not isinstance(hidden_out, torch.Tensor) or not isinstance(
            residual_after_attention,
            torch.Tensor,
        ):
            raise TypeError("captured decoder layer returned non-tensor state")
        residual_out = hidden_out + residual_after_attention
        capture_residual_in = None
        if residual_in is not None:
            capture_residual_in = self._canonical_capture_value(
                residual_in,
                (
                    self._capture_residual_in_buf
                    if self._capture_residual_in_buf is not None
                    else self._capture_residual_in_root_stage_buf
                ),
            )
        capture_residual_after_attention = self._canonical_capture_value(
            residual_after_attention,
            (
                self._capture_residual_after_attention_buf
                if self._capture_residual_after_attention_buf is not None
                else self._capture_residual_after_attention_root_stage_buf
            ),
        )
        capture_residual_out = self._canonical_capture_value(
            residual_out,
            (
                self._capture_residual_out_buf
                if self._capture_residual_out_buf is not None
                else self._capture_residual_out_root_stage_buf
            ),
        )
        if capture_residual_in is not None:
            self._capture_roots(
                capture_residual_in,
                "residual_in",
                self._capture_residual_in_root_buf,
                self._capture_residual_in_root_stage_buf,
            )
        self._capture_roots(
            capture_residual_after_attention,
            "residual_after_attention",
            self._capture_residual_after_attention_root_buf,
            self._capture_residual_after_attention_root_stage_buf,
        )
        self._capture_roots(
            capture_residual_out,
            "residual_out",
            self._capture_residual_out_root_buf,
            self._capture_residual_out_root_stage_buf,
        )
        if capture_residual_in is not None:
            self._capture_response_stamp(capture_residual_in)
        if self._capture_root_batch_destination is not None:
            torch.ops.verallm.activation_staged_row_roots(
                self._capture_root_batch_destination,
                self._capture_root_batch_scratch,
                self._capture_root_batch_staging,
                self._capture_root_batch_widths,
                capture_residual_out,
            )
        if capture_residual_in is not None:
            self._capture(
                capture_residual_in,
                19,
                self._capture_residual_in_buf,
            )
        self._capture(
            capture_residual_after_attention,
            20,
            self._capture_residual_after_attention_buf,
        )
        self._capture(
            capture_residual_out,
            21,
            self._capture_residual_out_buf,
        )
        return result

    def _capture_response_stamp(self, residual_in: torch.Tensor) -> None:
        if self._capture_response_stamp_buf is None:
            return
        indices = self._capture_response_stamp_row_indices
        if indices is None:
            raise RuntimeError(
                "response-stamp capture indices are unavailable"
            )
        if residual_in.is_cuda:
            torch.ops.verallm.buffer_gather_rows(
                self._capture_response_stamp_buf,
                residual_in,
                indices,
            )
            return
        valid = (indices >= 0) & (indices < residual_in.shape[0])
        if bool(valid.any()):
            positions = torch.nonzero(valid, as_tuple=False).flatten()
            self._capture_response_stamp_buf.index_copy_(
                0,
                positions,
                residual_in.index_select(
                    0,
                    indices.index_select(0, positions),
                ),
            )


def enable_batched_runtime_root_capture_v3(
    model: nn.Module,
    *,
    max_decode_rows: int,
) -> int:
    """Batch normal decode-row commitments across the signed stage inventory.

    Large prefill steps keep the existing per-stage reducer.  Steps that fit
    within ``max_decode_rows`` stage each activation and execute one shared
    leaf/reduction launch after the final decoder layer.  The commitment bytes
    and public root-buffer layout are unchanged.
    """

    if (
        not isinstance(model, nn.Module)
        or isinstance(max_decode_rows, bool)
        or not isinstance(max_decode_rows, int)
        or max_decode_rows <= 0
    ):
        raise ValueError("batched runtime root geometry is malformed")

    bindings = tuple(
        binding
        for module in model.modules()
        if isinstance(
            module,
            (CaptureLinearWrapper, CaptureDecoderLayerWrapper),
        )
        for binding in module._runtime_root_bindings()
    )
    if not bindings:
        return 0
    bindings = tuple(sorted(bindings, key=lambda item: item.stage_id))
    if len({item.stage_id for item in bindings}) != len(bindings):
        raise RuntimeError("runtime root stage inventory contains duplicates")
    from verallm.proof_v3.execution_anchor import (
        execution_anchor_lane_bytes_v3,
    )

    # The shared native reducer is byte-identical for the canonical 2-KiB
    # execution-anchor lanes.  Compact 256-byte GDN lanes retain the existing
    # direct reducer until the shared ABI carries a per-stage lane width.
    bindings = tuple(
        item
        for item in bindings
        if execution_anchor_lane_bytes_v3(item.stage_id) == 2048
    )
    if not bindings:
        return 0

    # A signed sparse-MoE profile may use a router dtype different from the
    # surrounding execution trace. The native staged reducer owns one typed
    # rectangular arena, so only stages in the decoder's runtime dtype can
    # share it. Mixed-dtype stages retain their already-installed direct graph
    # reducer and exact root buffers.
    finalizers = tuple(
        module
        for module in model.modules()
        if isinstance(module, CaptureDecoderLayerWrapper)
    )
    if not finalizers:
        raise RuntimeError("runtime root batch lacks a decoder finalizer")
    finalizer = max(finalizers, key=lambda item: item._layer_idx)
    runtime_dtype = finalizer._capture_dtype
    direct_bindings = tuple(
        item for item in bindings if item.dtype != runtime_dtype
    )
    bindings = tuple(
        item for item in bindings if item.dtype == runtime_dtype
    )
    if not bindings:
        logger.info(
            "Kept %d mixed-dtype runtime root stage(s) on direct reducers",
            len(direct_bindings),
        )
        return 0

    logical_bindings = bindings
    bindings, boundary_aliases = _deduplicate_decoder_boundary_root_bindings(
        logical_bindings
    )
    root_tensors = tuple(
        getattr(item.owner, item.root_attribute)
        for item in bindings
    )
    first_root = root_tensors[0]
    if (
        not isinstance(first_root, torch.Tensor)
        or not first_root.is_cuda
        or first_root.dtype != torch.uint8
        or first_root.ndim != 2
        or int(first_root.shape[1]) != 32
    ):
        raise RuntimeError("runtime root storage is malformed")
    device = first_root.device
    root_capacity = int(first_root.shape[0])
    dtype = bindings[0].dtype
    if dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise RuntimeError("runtime root activation dtype is unsupported")
    element_size = int(torch.empty((), dtype=dtype).element_size())
    if any(
        not isinstance(root, torch.Tensor)
        or root.device != device
        or root.dtype != torch.uint8
        or root.ndim != 2
        or tuple(root.shape) != (root_capacity, 32)
        or item.dtype != dtype
        or item.row_width <= 0
        or item.row_width >= 1 << 24
        or item.row_width % element_size
        for item, root in zip(bindings, root_tensors)
    ):
        raise RuntimeError("runtime root stage geometry is inconsistent")

    from zkllm.cuda import zkllm_native

    if getattr(
        zkllm_native,
        "cuda_blake3_runtime_staged_row_roots_into",
        None,
    ) is None:
        raise RuntimeError("native staged runtime root reducer is unavailable")

    max_columns = max(
        item.row_width // element_size
        for item in bindings
    )
    max_lanes = max(
        (item.row_width + 2047) // 2048
        for item in bindings
    )
    if max_lanes > 256:
        raise RuntimeError("runtime root stage exceeds native lane capacity")

    retention_bindings = tuple(
        (index, item)
        for index, item in enumerate(bindings)
        if item.stage_id.endswith(
            (".attention_kv_output", ".attention_qkv_output")
        )
    )
    # Compact-v9 selects exactly two full-attention layers.  The shared slots
    # are graph-static but device-selected after the nonce, so no per-layer
    # raw activation arena is retained during ordinary serving.
    retention_slot_count = 2
    retained_values = None
    retained_hashes = None
    retention_slots = None
    if retention_bindings:
        retention_max_lanes = max(
            (item.row_width + 2047) // 2048
            for _index, item in retention_bindings
        )
        retained_values = torch.empty(
            (
                retention_slot_count,
                root_capacity,
                retention_max_lanes,
                2048,
            ),
            dtype=torch.uint8,
            device=device,
        )
        retained_hashes = torch.empty(
            (
                retention_slot_count,
                root_capacity,
                retention_max_lanes,
                32,
            ),
            dtype=torch.uint8,
            device=device,
        )
        retention_slots = torch.full(
            (len(bindings),),
            -1,
            dtype=torch.int32,
            device=device,
        )
        for stage_index, item in retention_bindings:
            suffix = item.stage_id.split(".", 1)[1]
            item.owner._install_runtime_root_retention(
                suffix=suffix,
                stage_index=stage_index,
                values=retained_values,
                hashes=retained_hashes,
                slots=retention_slots,
            )

    destination = torch.zeros(
        (len(bindings), root_capacity, 32),
        dtype=torch.uint8,
        device=device,
    )
    staging = torch.zeros(
        (len(bindings), max_decode_rows, max_columns),
        dtype=dtype,
        device=device,
    )
    scratch = torch.empty(
        (len(bindings), max_decode_rows, max_lanes, 32),
        dtype=torch.uint8,
        device=device,
    )
    row_widths = torch.tensor(
        tuple(item.row_width for item in bindings),
        dtype=torch.int32,
        device=device,
    )
    slot_by_stage = {
        item.stage_id: index for index, item in enumerate(bindings)
    }
    for item in logical_bindings:
        source_stage = boundary_aliases.get(item.stage_id, item.stage_id)
        index = slot_by_stage[source_stage]
        setattr(item.owner, item.root_attribute, destination[index])
        setattr(item.owner, item.staging_attribute, staging[index])
        if source_stage != item.stage_id:
            suffix = item.stage_id.split(".", 1)[1]
            if isinstance(item.owner, CaptureDecoderLayerWrapper):
                item.owner._install_runtime_root_alias(
                    suffix=suffix,
                    source_stage_id=source_stage,
                )
            else:
                raise RuntimeError(
                    "runtime root alias source is malformed"
                )

    finalizer._capture_root_batch_destination = destination
    finalizer._capture_root_batch_scratch = scratch
    finalizer._capture_root_batch_staging = staging
    finalizer._capture_root_batch_widths = row_widths
    logger.info(
        "Batched %d runtime root stage(s) in %d reducer slot(s) across at "
        "most %d decode row(s)",
        len(logical_bindings),
        len(bindings),
        max_decode_rows,
    )
    if direct_bindings:
        logger.info(
            "Kept %d mixed-dtype runtime root stage(s) on direct reducers",
            len(direct_bindings),
        )
    if retention_bindings:
        logger.info(
            "Bounded post-nonce root retention installed for %d stage(s) "
            "(%d shared slots, %d rows, %d lanes)",
            len(retention_bindings),
            retention_slot_count,
            root_capacity,
            int(retained_values.shape[2]),
        )
    return len(logical_bindings)


class CaptureLMHeadWrapper(nn.Module):
    """Capture lm_head input/output at graph split points.

    vLLM's ``LogitsProcessor`` accesses ``lm_head.quant_method``,
    ``lm_head.weight``, etc. directly — it does NOT call ``forward()``.
    ``__getattr__`` delegates unknown attribute lookups to the wrapped
    module so the wrapper is transparent.
    """

    def __init__(self, original: nn.Module):
        super().__init__()
        self.original = original

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.original, name)

    def forward(self, x):
        # Side-effect-only capture (discard returned clone).
        _ = torch.ops.verallm.capture(x, -1, 3)
        y = self.original(x)
        _ = torch.ops.verallm.capture(y, -1, 4)
        return y


def attach_capture_ops(
    model: nn.Module,
    layers: nn.ModuleList,
    is_moe: bool,
    get_mlp_fn,
    get_gate_proj_fn,
    is_moe_layer_fn,
    *,
    wrap_lm_head: bool = True,
) -> int:
    """Walk the model tree and attach capture ops to the right modules.

    For MoE models:
        CaptureFusedMoE (OOT) is already instantiated by vLLM.  We just
        need to set _layer_idx on each FusedMoE instance so they know
        which layer they belong to.

    For dense models:
        Wrap gate_proj with CaptureLinearWrapper to insert a capture op.

    Returns the number of layers instrumented.
    """
    instrumented = 0
    missing_moe_layers = []
    found_fused_modules = 0

    for idx, layer in enumerate(layers):
        mlp = get_mlp_fn(layer)
        if mlp is None:
            continue

        if is_moe and is_moe_layer_fn(layer):
            # MoE: set _layer_idx on the FusedMoE (or CaptureFusedMoE) instance.
            # Walk the MLP subtree to find FusedMoE modules.
            marked = _set_layer_idx_on_fused_moe(mlp, idx)
            if marked == 0:
                missing_moe_layers.append(idx)
            found_fused_modules += marked
            instrumented += 1
        else:
            # Dense: wrap gate_proj with CaptureLinearWrapper.
            gate_proj = get_gate_proj_fn(mlp)
            if gate_proj is not None:
                if isinstance(gate_proj, CaptureLinearWrapper):
                    # Already wrapped (e.g. by pre-construction load hook).
                    # Just ensure the layer_idx is correct.
                    gate_proj._layer_idx = idx
                else:
                    wrapper = CaptureLinearWrapper(gate_proj, idx)
                    _replace_module(mlp, gate_proj, wrapper)
                instrumented += 1

    lm_head_wrapped = _wrap_lm_head(model) if wrap_lm_head else False
    if lm_head_wrapped:
        instrumented += 1

    logger.info(
        "Attached capture ops to %d modules (is_moe=%s, lm_head=%s)",
        instrumented,
        is_moe,
        bool(lm_head_wrapped),
    )
    if is_moe and missing_moe_layers:
        logger.warning(
            "No FusedMoE modules found in %d MoE layers (first 8: %s)",
            len(missing_moe_layers),
            missing_moe_layers[:8],
        )
    if is_moe:
        logger.info("Marked _layer_idx on %d FusedMoE modules", found_fused_modules)
    return instrumented


def _set_layer_idx_on_fused_moe(module: nn.Module, layer_idx: int) -> int:
    """Find FusedMoE (or CaptureFusedMoE) in a module subtree and set _layer_idx."""
    try:
        from vllm.model_executor.layers.fused_moe.layer import FusedMoE
    except ImportError:
        return 0

    marked = 0
    for child in module.modules():
        if isinstance(child, FusedMoE):
            if marked == 0 and layer_idx == 0:
                logger.info(
                    "Layer 0 runtime layer_name=%s",
                    getattr(child, "layer_name", None),
                )
            # Robust across vLLM OOT dispatch styles:
            # - replaced class instances already define _layer_idx
            # - base FusedMoE instances can still carry this dynamic attribute
            setattr(child, "_layer_idx", layer_idx)
            marked += 1
    return marked


def _replace_module(parent: nn.Module, old_child: nn.Module, new_child: nn.Module) -> None:
    """Replace old_child with new_child in parent's attributes."""
    for name, child in parent.named_children():
        if child is old_child:
            setattr(parent, name, new_child)
            return
    # Fallback: search one level deeper (some architectures nest gate_proj)
    for name, child in parent.named_modules():
        if child is old_child and '.' not in name:
            setattr(parent, name, new_child)
            return


def _wrap_lm_head(model: nn.Module) -> bool:
    """Wrap lm_head/output projection with capture ops if present."""
    for attr in ("lm_head", "output", "embed_out"):
        head = getattr(model, attr, None)
        if head is None:
            continue
        if isinstance(head, CaptureLMHeadWrapper):
            return False
        setattr(model, attr, CaptureLMHeadWrapper(head))
        return True
    return False
