"""
CaptureFusedMoE — Out-of-tree replacement for vLLM's FusedMoE.

Inserts verallm::capture ops before the fused MoE forward pass to
capture hidden_states (input) and router_logits at graph split points.
The rest of the FusedMoE computation runs unchanged in CUDA graphs.

Registered via @CustomOp.register_oot for both "FusedMoE" and
"fused_moe" keys so OOT dispatch works across vLLM versions.
"""

import logging
import re
from collections.abc import Callable

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


_LAYER_NAME_PATTERNS = (
    re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)"),
    re.compile(r"(?:^|\.)h\.(\d+)(?:\.|$)"),
)

MOE_INPUT_CAPTURE_KIND_V3 = 22
MOE_ROUTER_LOGITS_CAPTURE_KIND_V3 = 23
MOE_AGGREGATE_OUTPUT_CAPTURE_KIND_V3 = 24

OPAQUE_MOE_RUNTIME_ROOT_SUFFIXES_V3 = (
    "moe_input",
    "moe_router_logits",
    "moe_aggregate_output",
)
QUALIFIED_OPAQUE_MOE_RUNTIME_ROOT_SUFFIXES_V3 = (
    *OPAQUE_MOE_RUNTIME_ROOT_SUFFIXES_V3,
    "residual_after_attention",
    "residual_out",
)


class OpaqueMoeRuntimeRootCapture(nn.Module):
    """Capture bounded roots inside a qualified vLLM MoE custom op."""

    def __init__(
        self,
        *,
        layer_idx: int,
        root_max_tokens: int,
        hidden_dim: int,
        router_dim: int,
        activation_dtype: torch.dtype,
        router_dtype: torch.dtype,
        device: torch.device,
        capture_decoder_residuals: bool = False,
    ) -> None:
        super().__init__()
        self._layer_idx = int(layer_idx)
        self._hidden_dim = int(hidden_dim)
        self._router_dim = int(router_dim)
        self._activation_dtype = activation_dtype
        self._router_dtype = router_dtype
        self._capture_decoder_residuals = bool(capture_decoder_residuals)
        self._batched_runtime_roots = False
        self._defer_runtime_root_finalization = False
        self._current_input: torch.Tensor | None = None
        self._current_router: torch.Tensor | None = None
        root_suffixes = (
            QUALIFIED_OPAQUE_MOE_RUNTIME_ROOT_SUFFIXES_V3
            if self._capture_decoder_residuals
            else OPAQUE_MOE_RUNTIME_ROOT_SUFFIXES_V3
        )
        for suffix in root_suffixes:
            self.register_buffer(
                f"_capture_{suffix}_root_buf",
                torch.zeros(
                    root_max_tokens,
                    32,
                    dtype=torch.uint8,
                    device=device,
                ),
                persistent=False,
            )
            self.register_buffer(
                f"_capture_{suffix}_root_stage_buf",
                None,
                persistent=False,
            )
            self.register_buffer(
                f"_capture_{suffix}_history_root_buf",
                None,
                persistent=False,
            )
        for name in (
            "_capture_root_batch_destination",
            "_capture_root_batch_scratch",
            "_capture_root_batch_staging",
            "_capture_root_batch_widths",
            "_capture_root_batch_lane_widths",
            "_capture_root_history_destination",
            "_capture_root_history_indices",
            "_capture_root_retained_values",
            "_capture_root_retained_hashes",
            "_capture_root_retention_slots",
        ):
            self.register_buffer(name, None, persistent=False)
        self._capture_root_retention_indices: dict[str, int] = {}
        self._capture_root_history_rows_per_slot = 0
        self._capture_root_history_slot_count = 0
        self._capture_root_history_max_decode_rows = 0

    def _write_root(self, suffix: str, value: torch.Tensor) -> None:
        from verallm.proof_v3.execution_anchor import (
            execution_anchor_lane_bytes_v3,
        )

        destination = getattr(self, f"_capture_{suffix}_root_buf")
        lane_bytes = execution_anchor_lane_bytes_v3(
            f"l{self._layer_idx}.{suffix}"
        )
        staging = getattr(self, f"_capture_{suffix}_root_stage_buf")
        if (
            suffix in self._capture_root_retention_indices
            and isinstance(staging, torch.Tensor)
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
                value,
                self._capture_root_retention_indices[suffix],
                lane_bytes,
            )
            return
        torch.ops.verallm.activation_row_roots(
            destination,
            value,
            lane_bytes,
        )

    def capture_input(self, value: torch.Tensor) -> None:
        if self._batched_runtime_roots or self._capture_decoder_residuals:
            self._current_input = value
            return
        self._write_root("moe_input", value)

    def capture_router(self, value: torch.Tensor) -> None:
        if value.dtype != self._router_dtype:
            raise RuntimeError(
                "opaque MoE router dtype does not match the signed runtime"
            )
        if self._batched_runtime_roots or self._capture_decoder_residuals:
            self._current_router = value
            return
        self._write_root("moe_router_logits", value)

    def capture_output(
        self,
        result: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        aggregate_parts: tuple[torch.Tensor, torch.Tensor] | None = None
        if isinstance(result, tuple):
            if (
                len(result) != 2
                or not isinstance(result[0], torch.Tensor)
                or not isinstance(result[1], torch.Tensor)
            ):
                raise RuntimeError("opaque MoE result tuple is malformed")
            aggregate_parts = (result[0], result[1])
            value = result[0]
        else:
            value = result
        if not isinstance(value, torch.Tensor):
            raise RuntimeError("opaque MoE result is not a tensor")
        if self._batched_runtime_roots:
            capture_input = self._current_input
            capture_router = self._current_router
            self._current_input = None
            self._current_router = None
            if (
                not isinstance(capture_input, torch.Tensor)
                or not isinstance(capture_router, torch.Tensor)
                or capture_input.shape[0] != value.shape[0]
                or capture_router.shape[0] != value.shape[0]
            ):
                raise RuntimeError(
                    "opaque MoE runtime tensors are unavailable"
                )
            input_staging = self._capture_moe_input_root_stage_buf
            router_staging = self._capture_moe_router_logits_root_stage_buf
            output_staging = self._capture_moe_aggregate_output_root_stage_buf
            if not all(
                isinstance(item, torch.Tensor)
                for item in (
                    input_staging,
                    router_staging,
                    output_staging,
                )
            ):
                raise RuntimeError("opaque MoE root staging is unavailable")
            if value.shape[0] <= input_staging.shape[0]:
                if aggregate_parts is None:
                    torch.ops.verallm.buffer_copy_three_rows_padded(
                        input_staging,
                        router_staging,
                        output_staging,
                        capture_input,
                        capture_router,
                        value,
                    )
                else:
                    torch.ops.verallm.buffer_copy_input_router_sum_rows_padded(
                        input_staging,
                        router_staging,
                        output_staging,
                        capture_input,
                        capture_router,
                        aggregate_parts[0],
                        aggregate_parts[1],
                    )
            else:
                self._write_root("moe_input", capture_input)
                self._write_root("moe_router_logits", capture_router)
                self._write_root(
                    "moe_aggregate_output",
                    (
                        value
                        if aggregate_parts is None
                        else aggregate_parts[0] + aggregate_parts[1]
                    ),
                )
            if not self._capture_decoder_residuals:
                self._finalize_runtime_roots(
                    value,
                    staging_row_capacity=int(input_staging.shape[0]),
                )
            return
        if self._capture_decoder_residuals:
            capture_input = self._current_input
            capture_router = self._current_router
            self._current_input = None
            self._current_router = None
            if (
                not isinstance(capture_input, torch.Tensor)
                or not isinstance(capture_router, torch.Tensor)
                or capture_input.shape[0] != value.shape[0]
                or capture_router.shape[0] != value.shape[0]
            ):
                raise RuntimeError(
                    "qualified opaque MoE runtime tensors are unavailable"
                )
            self._write_root("moe_input", capture_input)
            self._write_root("moe_router_logits", capture_router)
        self._write_root(
            "moe_aggregate_output",
            (
                value
                if aggregate_parts is None
                else aggregate_parts[0] + aggregate_parts[1]
            ),
        )

    def capture_decoder_output(
        self,
        hidden_out: torch.Tensor,
        residual_after_attention: torch.Tensor,
    ) -> None:
        """Capture the two signed residual boundaries after the sparse block."""

        if not self._capture_decoder_residuals:
            raise RuntimeError(
                "opaque MoE decoder residual capture is not enabled"
            )
        if (
            not isinstance(hidden_out, torch.Tensor)
            or not isinstance(residual_after_attention, torch.Tensor)
            or residual_after_attention.shape[0] != hidden_out.shape[0]
        ):
            raise RuntimeError(
                "qualified opaque MoE decoder tensors are unavailable"
            )
        if not self._batched_runtime_roots:
            self._write_root(
                "residual_after_attention",
                residual_after_attention,
            )
            self._write_root(
                "residual_out",
                hidden_out + residual_after_attention,
            )
            return
        staging = (
            self._capture_residual_after_attention_root_stage_buf,
            self._capture_residual_out_root_stage_buf,
        )
        if not all(isinstance(item, torch.Tensor) for item in staging):
            raise RuntimeError(
                "qualified opaque MoE root staging is unavailable"
            )
        residual_staging = staging[0]
        if hidden_out.shape[0] <= residual_staging.shape[0]:
            torch.ops.verallm.buffer_copy_residual_boundaries_rows_padded(
                *staging,
                residual_after_attention,
                hidden_out,
                residual_after_attention,
            )
        else:
            self._write_root(
                "residual_after_attention",
                residual_after_attention,
            )
            self._write_root(
                "residual_out",
                hidden_out + residual_after_attention,
            )
        if not self._defer_runtime_root_finalization:
            self._finalize_runtime_roots(
                hidden_out,
                staging_row_capacity=int(residual_staging.shape[0]),
            )

    def _finalize_runtime_roots(
        self,
        reference: torch.Tensor,
        *,
        staging_row_capacity: int,
    ) -> None:
        if self._capture_root_batch_destination is not None:
            if self._capture_root_history_destination is None:
                torch.ops.verallm.activation_staged_row_roots_lanes(
                    self._capture_root_batch_destination,
                    self._capture_root_batch_scratch,
                    self._capture_root_batch_staging,
                    self._capture_root_batch_widths,
                    self._capture_root_batch_lane_widths,
                    reference,
                )
            else:
                torch.ops.verallm.activation_staged_row_roots_lanes_history(
                    self._capture_root_history_destination,
                    self._capture_root_batch_scratch,
                    self._capture_root_batch_staging,
                    self._capture_root_batch_widths,
                    self._capture_root_batch_lane_widths,
                    self._capture_root_history_indices,
                    reference,
                )
        if (
            self._capture_root_history_destination is not None
            and reference.shape[0] > staging_row_capacity
        ):
            torch.ops.verallm.buffer_scatter_root_history(
                self._capture_root_history_destination,
                self._capture_root_batch_destination,
                self._capture_root_history_indices,
                reference,
            )

    def proof_capture_root_buffers(self):
        element_size = int(
            torch.empty((), dtype=self._activation_dtype).element_size()
        )
        router_element_size = int(
            torch.empty((), dtype=self._router_dtype).element_size()
        )
        suffixes = (
            QUALIFIED_OPAQUE_MOE_RUNTIME_ROOT_SUFFIXES_V3
            if self._capture_decoder_residuals
            else OPAQUE_MOE_RUNTIME_ROOT_SUFFIXES_V3
        )
        return tuple(
            (
                self._layer_idx,
                suffix,
                (
                    getattr(self, f"_capture_{suffix}_history_root_buf")
                    if getattr(
                        self,
                        f"_capture_{suffix}_history_root_buf",
                    ) is not None
                    else getattr(self, f"_capture_{suffix}_root_buf")
                ),
                (
                    self._router_dim * router_element_size
                    if suffix == "moe_router_logits"
                    else self._hidden_dim * element_size
                ),
            )
            for suffix in suffixes
        )

    def proof_capture_split_stages(self):
        """Return no callback-only stages for graph-native MoE roots.

        Opaque MoE capture always exposes graph-readable root buffers.  It
        participates in the common serving-capture wrapper inventory, but it
        never emits raw rows through ``capture_at_split()``.
        """

        return ()

    def proof_capture_root_history(self):
        if self._capture_root_history_destination is None:
            return ()
        return (
            (
                self._capture_root_history_indices,
                self._capture_root_history_rows_per_slot,
                self._capture_root_history_slot_count,
                self._capture_root_history_max_decode_rows,
            ),
        )

    def proof_capture_root_finalizer(self):
        """Expose the one qualified post-graph root finalizer."""

        return (
            (self._finalize_runtime_roots_after_graph,)
            if self._defer_runtime_root_finalization
            else ()
        )

    def _finalize_runtime_roots_after_graph(self, row_count: int) -> None:
        """Hash graph-staged rows after every compiled partition completes."""

        destination = self._capture_root_batch_destination
        staging = self._capture_root_batch_staging
        if (
            not self._defer_runtime_root_finalization
            or isinstance(row_count, bool)
            or not isinstance(row_count, int)
            or row_count <= 0
            or not isinstance(destination, torch.Tensor)
            or destination.ndim != 3
            or row_count > int(destination.shape[1])
            or not isinstance(staging, torch.Tensor)
            or staging.ndim != 3
        ):
            raise RuntimeError(
                "qualified post-graph root finalization is malformed"
            )
        self._finalize_runtime_roots(
            destination[0, :row_count],
            staging_row_capacity=int(staging.shape[1]),
        )

    def proof_capture_runtime_root_only_stages(self):
        suffixes = (
            QUALIFIED_OPAQUE_MOE_RUNTIME_ROOT_SUFFIXES_V3
            if self._capture_decoder_residuals
            else OPAQUE_MOE_RUNTIME_ROOT_SUFFIXES_V3
        )
        return tuple((self._layer_idx, suffix) for suffix in suffixes)

    def _install_runtime_root_retention(
        self,
        *,
        suffix: str,
        stage_index: int,
        values: torch.Tensor,
        hashes: torch.Tensor,
        slots: torch.Tensor,
    ) -> None:
        """Install the shared graph-static post-nonce prompt arena."""

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

    def _runtime_root_bindings(self):
        """Expose the exact staged rows used by the shared root reducer."""

        from verallm.vllm_plugin.capture_linear import _RuntimeRootBinding

        activation_element_size = int(
            torch.empty((), dtype=self._activation_dtype).element_size()
        )
        router_element_size = int(
            torch.empty((), dtype=self._router_dtype).element_size()
        )
        suffixes = (
            QUALIFIED_OPAQUE_MOE_RUNTIME_ROOT_SUFFIXES_V3
            if self._capture_decoder_residuals
            else OPAQUE_MOE_RUNTIME_ROOT_SUFFIXES_V3
        )
        return tuple(
            _RuntimeRootBinding(
                stage_id=f"l{self._layer_idx}.{suffix}",
                row_width=(
                    self._router_dim * router_element_size
                    if suffix == "moe_router_logits"
                    else self._hidden_dim * activation_element_size
                ),
                owner=self,
                root_attribute=f"_capture_{suffix}_root_buf",
                staging_attribute=f"_capture_{suffix}_root_stage_buf",
                dtype=(
                    self._router_dtype
                    if suffix == "moe_router_logits"
                    else self._activation_dtype
                ),
                history_attribute=(
                    f"_capture_{suffix}_history_root_buf"
                ),
            )
            for suffix in suffixes
        )


def enable_opaque_moe_batched_root_capture_v3(
    model: nn.Module,
    *,
    max_decode_rows: int,
    history_rows_per_slot: int = 0,
    history_slot_count: int = 0,
    retention_slot_count: int = 0,
) -> int:
    """Hash the qualified sparse tensors into bounded request history."""

    if (
        not isinstance(model, nn.Module)
        or isinstance(max_decode_rows, bool)
        or not isinstance(max_decode_rows, int)
        or max_decode_rows <= 0
        or isinstance(history_rows_per_slot, bool)
        or not isinstance(history_rows_per_slot, int)
        or history_rows_per_slot < 0
        or isinstance(history_slot_count, bool)
        or not isinstance(history_slot_count, int)
        or history_slot_count < 0
        or bool(history_rows_per_slot) != bool(history_slot_count)
        or isinstance(retention_slot_count, bool)
        or not isinstance(retention_slot_count, int)
        or retention_slot_count < 0
    ):
        raise ValueError("opaque MoE root-batch geometry is malformed")
    captures = tuple(
        sorted(
            (
                module
                for module in model.modules()
                if isinstance(module, OpaqueMoeRuntimeRootCapture)
            ),
            key=lambda module: module._layer_idx,
        )
    )
    if not captures:
        raise RuntimeError("opaque MoE root batch has no capture modules")
    if len({capture._layer_idx for capture in captures}) != len(captures):
        raise RuntimeError("opaque MoE root batch has duplicate layers")
    if len({capture._capture_decoder_residuals for capture in captures}) != 1:
        raise RuntimeError("opaque MoE root batch has mixed inventories")

    from verallm.proof_v3.execution_anchor import (
        execution_anchor_lane_bytes_v3,
    )
    from zkllm.cuda import zkllm_native

    if getattr(
        zkllm_native,
        "cuda_blake3_runtime_staged_row_roots_lanes_into",
        None,
    ) is None:
        raise RuntimeError(
            "native mixed-lane staged runtime root reducer is unavailable"
        )
    if history_rows_per_slot and getattr(
        zkllm_native,
        "cuda_blake3_runtime_staged_row_roots_lanes_history_into",
        None,
    ) is None:
        raise RuntimeError(
            "native persistent staged runtime root reducer is unavailable"
        )
    bindings = [
        binding
        for capture in captures
        for binding in capture._runtime_root_bindings()
    ]
    if captures[0]._capture_decoder_residuals:
        from verallm.vllm_plugin.capture_linear import CaptureLinearWrapper

        bindings.extend(
            binding
            for module in model.modules()
            if isinstance(module, CaptureLinearWrapper)
            for binding in module._runtime_root_bindings()
            if binding.stage_id.endswith(".attention_kv_output")
        )
    bindings = tuple(
        sorted(
            bindings,
            key=lambda item: (
                int(item.stage_id.split(".", 1)[0][1:]),
                item.stage_id.split(".", 1)[1],
            ),
        )
    )
    if len({binding.stage_id for binding in bindings}) != len(bindings):
        raise RuntimeError("opaque MoE root batch has duplicate stages")
    entries = tuple(
        (
            binding,
            execution_anchor_lane_bytes_v3(binding.stage_id),
        )
        for binding in bindings
    )
    dtype = entries[0][0].dtype
    if (
        dtype not in (torch.float16, torch.bfloat16, torch.float32)
        or any(binding.dtype != dtype for binding, _lane_width in entries)
        or any(lane_width not in (256, 2048) for _binding, lane_width in entries)
    ):
        raise RuntimeError("opaque MoE root batch has mixed runtime dtypes")
    roots = tuple(
        getattr(binding.owner, binding.root_attribute)
        for binding, _lane_width in entries
    )
    first_root = roots[0]
    if (
        not isinstance(first_root, torch.Tensor)
        or not first_root.is_cuda
        or first_root.dtype != torch.uint8
        or first_root.ndim != 2
        or first_root.shape[1] != 32
    ):
        raise RuntimeError("opaque MoE root storage is malformed")
    root_capacity = int(first_root.shape[0])
    device = first_root.device
    if any(
        not isinstance(root, torch.Tensor)
        or root.device != device
        or root.dtype != torch.uint8
        or tuple(root.shape) != (root_capacity, 32)
        for root in roots
    ):
        raise RuntimeError("opaque MoE root storage is inconsistent")
    element_size = int(torch.empty((), dtype=dtype).element_size())
    max_columns = max(
        binding.row_width // element_size
        for binding, _lane_width in entries
    )
    max_lanes = max(
        (binding.row_width + lane_width - 1) // lane_width
        for binding, lane_width in entries
    )
    if retention_slot_count > len(entries):
        raise RuntimeError(
            "opaque MoE retained-stage bound exceeds its inventory"
        )
    retained_values = None
    retained_hashes = None
    retention_slots = None
    if retention_slot_count:
        if any(lane_width != 2048 for _binding, lane_width in entries):
            raise RuntimeError(
                "opaque MoE retained roots require canonical 2-KiB lanes"
            )
        retained_values = torch.empty(
            (
                retention_slot_count,
                root_capacity,
                max_lanes,
                2048,
            ),
            dtype=torch.uint8,
            device=device,
        )
        retained_hashes = torch.empty(
            (
                retention_slot_count,
                root_capacity,
                max_lanes,
                32,
            ),
            dtype=torch.uint8,
            device=device,
        )
        retention_slots = torch.full(
            (len(entries),),
            -1,
            dtype=torch.int32,
            device=device,
        )
    destination = torch.zeros(
        (len(entries), root_capacity, 32),
        dtype=torch.uint8,
        device=device,
    )
    history_destination = None
    history_indices = None
    if history_rows_per_slot:
        history_capacity = history_rows_per_slot * history_slot_count
        if history_capacity <= 0 or history_capacity >= 1 << 31:
            raise RuntimeError(
                "opaque MoE root history capacity is malformed"
            )
        history_storage = torch.empty(
            (history_capacity, len(entries), 32),
            dtype=torch.uint8,
            device=device,
        )
        # Store chronological rows contiguously while exposing the logical
        # stage-major proof inventory ABI.
        history_destination = history_storage.permute(1, 0, 2)
        history_indices = torch.full(
            (root_capacity,),
            -1,
            dtype=torch.int64,
            device=device,
        )
    staging = torch.zeros(
        (len(entries), max_decode_rows, max_columns),
        dtype=dtype,
        device=device,
    )
    scratch = torch.empty(
        (len(entries), max_decode_rows, max_lanes, 32),
        dtype=torch.uint8,
        device=device,
    )
    row_widths = torch.tensor(
        tuple(binding.row_width for binding, _lane_width in entries),
        dtype=torch.int32,
        device=device,
    )
    lane_widths = torch.tensor(
        tuple(lane_width for _binding, lane_width in entries),
        dtype=torch.int32,
        device=device,
    )
    for index, (binding, _lane_width) in enumerate(entries):
        suffix = binding.stage_id.split(".", 1)[1]
        setattr(binding.owner, binding.root_attribute, destination[index])
        if history_destination is not None:
            if binding.history_attribute is None:
                raise RuntimeError(
                    "persistent hybrid root binding lacks history storage"
                )
            setattr(
                binding.owner,
                binding.history_attribute,
                history_destination[index],
            )
        setattr(
            binding.owner,
            binding.staging_attribute,
            staging[index],
        )
        if retained_values is not None:
            binding.owner._install_runtime_root_retention(
                suffix=suffix,
                stage_index=index,
                values=retained_values,
                hashes=retained_hashes,
                slots=retention_slots,
            )
        if isinstance(binding.owner, OpaqueMoeRuntimeRootCapture):
            binding.owner._batched_runtime_roots = True

    finalizer = captures[-1]
    finalizer._capture_root_batch_destination = destination
    finalizer._capture_root_batch_scratch = scratch
    finalizer._capture_root_batch_staging = staging
    finalizer._capture_root_batch_widths = row_widths
    finalizer._capture_root_batch_lane_widths = lane_widths
    finalizer._capture_root_history_destination = history_destination
    finalizer._capture_root_history_indices = history_indices
    finalizer._capture_root_history_rows_per_slot = history_rows_per_slot
    finalizer._capture_root_history_slot_count = history_slot_count
    finalizer._capture_root_history_max_decode_rows = max_decode_rows
    # Compiled hybrid decode can partition the per-layer staging writes from
    # this shared reducer. Finalize once at the model runner's post-step
    # boundary, after every partition has completed and before the tracker
    # reads either roots or raw rows. Ordinary and non-persistent adapters keep
    # their original graph-integrated finalization.
    finalizer._defer_runtime_root_finalization = bool(
        captures[0]._capture_decoder_residuals
    )
    return len(entries)


class OpaqueMoeDecoderResidualRootCapture(nn.Module):
    """Observe signed residual boundaries without changing decoder outputs."""

    def __init__(
        self,
        original: nn.Module,
        capture: Callable[[torch.Tensor, torch.Tensor], None],
    ) -> None:
        super().__init__()
        self.original = original
        self._capture_callback = capture

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.original, name)

    def forward(self, *args, **kwargs):
        result = self.original(*args, **kwargs)
        if (
            not isinstance(result, tuple)
            or len(result) < 2
            or not isinstance(result[0], torch.Tensor)
            or not isinstance(result[1], torch.Tensor)
        ):
            raise RuntimeError(
                "qualified opaque MoE decoder result is malformed"
            )
        self._capture_callback(result[0], result[1])
        return result


def wrap_opaque_moe_decoder_residual_root_capture_v3(
    *,
    layer: nn.Module,
    capture: OpaqueMoeRuntimeRootCapture,
) -> OpaqueMoeDecoderResidualRootCapture:
    """Wrap one qualified decoder layer for its signed residual roots."""

    if not capture._capture_decoder_residuals:
        raise RuntimeError(
            "qualified opaque MoE decoder residual capture is not enabled"
        )
    if isinstance(layer, OpaqueMoeDecoderResidualRootCapture):
        return layer
    return OpaqueMoeDecoderResidualRootCapture(
        layer,
        capture.capture_decoder_output,
    )


class _OpaqueMoeGateRootCapture(nn.Module):
    def __init__(
        self,
        original: nn.Module,
        capture: Callable[[torch.Tensor], None],
    ) -> None:
        super().__init__()
        self.original = original
        self._capture_callback = capture

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.original, name)

    def forward(self, value: torch.Tensor):
        result = self.original(value)
        output = result[0] if isinstance(result, tuple) else result
        if not isinstance(output, torch.Tensor):
            raise RuntimeError("opaque MoE gate returned a non-tensor")
        self._capture_callback(output)
        return result


def wrap_opaque_moe_runtime_root_capture_v3(
    *,
    layer: nn.Module,
    layer_idx: int,
    root_max_tokens: int,
    hidden_dim: int,
    activation_dtype: torch.dtype,
    router_dtype: torch.dtype,
    capture_decoder_residuals: bool = False,
) -> OpaqueMoeRuntimeRootCapture:
    """Add roots within the stock ``moe_forward`` custom-op implementation."""

    from verallm.introspection import get_mlp

    existing = getattr(layer, "_verathos_opaque_moe_root_capture", None)
    if isinstance(existing, OpaqueMoeRuntimeRootCapture):
        if existing._capture_decoder_residuals != bool(
            capture_decoder_residuals
        ):
            raise RuntimeError("opaque MoE capture inventory changed")
        return existing
    mlp = get_mlp(layer)
    experts = getattr(mlp, "experts", None)
    runner = getattr(experts, "runner", None)
    gate = getattr(runner, "gate", None)
    # vLLM 0.20 names the custom-op implementation ``_forward_impl`` while
    # the qualified 0.19 Ampere stack exposes the same boundary as
    # ``forward_impl``.  The custom op resolves the runner on every call, so
    # replacing the exact version-specific instance attribute keeps capture
    # inside the stock CUDA-graph path on both runtimes.
    implementation_name = (
        "_forward_impl"
        if callable(getattr(runner, "_forward_impl", None))
        else "forward_impl"
    )
    original_impl = getattr(runner, implementation_name, None)
    if (
        not isinstance(mlp, nn.Module)
        or not isinstance(experts, nn.Module)
        or runner is None
        or not isinstance(gate, nn.Module)
        or not callable(original_impl)
    ):
        raise RuntimeError("opaque MoE runtime boundary is unavailable")
    router_dim = _positive_module_dimension(
        gate,
        (
            "output_size_per_partition",
            "output_size",
            "out_features",
        ),
    )
    if router_dim <= 0:
        raise RuntimeError("opaque MoE router width is unavailable")
    device = next(experts.parameters()).device
    capture = OpaqueMoeRuntimeRootCapture(
        layer_idx=layer_idx,
        root_max_tokens=root_max_tokens,
        hidden_dim=hidden_dim,
        router_dim=router_dim,
        activation_dtype=activation_dtype,
        router_dtype=router_dtype,
        device=device,
        capture_decoder_residuals=capture_decoder_residuals,
    )
    runner.gate = _OpaqueMoeGateRootCapture(gate, capture.capture_router)

    def captured_forward_impl(
        moe_layer,
        hidden_states,
        router_logits,
        shared_experts_input,
        input_ids=None,
    ):
        capture.capture_input(hidden_states)
        if implementation_name == "_forward_impl":
            result = original_impl(
                moe_layer,
                hidden_states,
                router_logits,
                shared_experts_input,
                input_ids,
            )
        else:
            if input_ids is not None:
                raise RuntimeError(
                    "vLLM 0.19 MoE runtime received unsupported input IDs"
                )
            result = original_impl(
                moe_layer,
                hidden_states,
                router_logits,
                shared_experts_input,
            )
        capture.capture_output(result)
        return result

    setattr(runner, implementation_name, captured_forward_impl)
    layer.add_module("_verathos_opaque_moe_root_capture", capture)
    return capture


def _positive_module_dimension(
    module: nn.Module,
    attributes: tuple[str, ...],
) -> int:
    for attribute in attributes:
        try:
            value = int(getattr(module, attribute, 0))
        except (TypeError, ValueError):
            value = 0
        if value > 0:
            return value
    weight = getattr(module, "weight", None)
    if isinstance(weight, torch.Tensor) and weight.ndim == 2:
        if any(
            "output" in attribute or "out_" in attribute
            for attribute in attributes
        ):
            return int(weight.shape[0])
        return int(weight.shape[1])
    return 0


def wrap_sparse_moe_block_capture_v3(
    *,
    layer: nn.Module,
    layer_idx: int,
    max_tokens: int,
    hidden_dim: int,
    activation_dtype: torch.dtype,
    router_dtype: torch.dtype | None = None,
    use_buffer: bool,
    use_triton_copy: bool,
    root_max_tokens: int,
    row_indices: torch.Tensor | None,
    root_stage_suffixes: frozenset[str] | None,
) -> tuple[nn.Module, nn.Module]:
    """Capture one sparse block's exact input, router logits and aggregate.

    Qwen's ``SharedFusedMoE`` receives the hidden state twice when its router
    is internal; its public ``router_logits`` argument is therefore a dummy.
    The actual logits are produced by the gate owned by both the sparse block
    and the fused runner.  This wrapper replaces that one shared gate object
    at every live reference, then wraps the complete sparse block so its
    output is the exact routed-plus-shared aggregate used by the decoder
    residual chain.  A proof-v3 caller supplies the dtype pinned by the
    authenticated MoE semantics; legacy callers infer it from the live
    unquantized router parameter and still enforce the observed output.
    """

    from verallm.introspection import get_mlp
    from verallm.vllm_plugin.capture_linear import (
        CaptureLinearWrapper,
        _replace_module,
    )

    if (
        isinstance(layer_idx, bool)
        or not isinstance(layer_idx, int)
        or layer_idx < 0
        or isinstance(max_tokens, bool)
        or not isinstance(max_tokens, int)
        or max_tokens <= 0
        or isinstance(hidden_dim, bool)
        or not isinstance(hidden_dim, int)
        or hidden_dim <= 0
        or activation_dtype not in (torch.float16, torch.bfloat16)
        or router_dtype
        not in (None, torch.bfloat16, torch.float32)
        or isinstance(root_max_tokens, bool)
        or not isinstance(root_max_tokens, int)
        or root_max_tokens < 0
    ):
        raise ValueError("sparse-MoE capture geometry is malformed")

    mlp = get_mlp(layer)
    if (
        isinstance(mlp, CaptureLinearWrapper)
        and mlp._capture_input_suffix == "moe_input"
        and mlp._capture_output_suffix == "moe_aggregate_output"
    ):
        router = getattr(mlp.original, "gate", None)
        if not (
            isinstance(router, CaptureLinearWrapper)
            and router._capture_output_suffix == "moe_router_logits"
        ):
            raise RuntimeError("sparse-MoE router capture is missing")
        if (
            router_dtype is not None
            and router._capture_required_output_dtype != router_dtype
        ):
            raise RuntimeError("sparse-MoE router capture dtype does not match")
        return mlp, router
    experts = getattr(mlp, "experts", None)
    gate = getattr(mlp, "gate", None)
    if not isinstance(mlp, nn.Module) or not isinstance(experts, nn.Module):
        raise RuntimeError("sparse-MoE block is unavailable")
    if not isinstance(gate, nn.Module):
        raise RuntimeError("sparse-MoE router is unavailable")

    if router_dtype is None:
        candidates = (getattr(gate, "weight", None), *tuple(gate.parameters()))
        router_dtype = next(
            (
                value.dtype
                for value in candidates
                if isinstance(value, torch.Tensor)
                and value.dtype in (torch.bfloat16, torch.float32)
            ),
            None,
        )
    if router_dtype not in (torch.bfloat16, torch.float32):
        raise RuntimeError("sparse-MoE router dtype is unavailable")

    expert_count = _positive_module_dimension(
        gate,
        (
            "output_size_per_partition",
            "output_size",
            "out_features",
        ),
    )
    if expert_count <= 0:
        raise RuntimeError("sparse-MoE router width is unavailable")

    router_wrapper = CaptureLinearWrapper(
        gate,
        layer_idx,
        # The router executes inside vLLM's opaque fused-MoE custom op, where
        # a Python graph split cannot run on CUDA-graph replay.  Its explicit
        # fixed buffers and root reducer are therefore always graph-native.
        use_buffer=True,
        max_tokens=max_tokens,
        hidden_dim=0,
        dtype=router_dtype,
        use_triton_copy=use_triton_copy,
        output_dim=expert_count,
        root_max_tokens=root_max_tokens,
        input_kind=MOE_INPUT_CAPTURE_KIND_V3,
        output_kind=MOE_ROUTER_LOGITS_CAPTURE_KIND_V3,
        input_suffix="moe_input",
        output_suffix="moe_router_logits",
        row_indices=row_indices,
        root_stage_suffixes=root_stage_suffixes,
        required_output_dtype=router_dtype,
    )
    setattr(mlp, "gate", router_wrapper)

    # vLLM retains the same router under the FusedMoE and its runner.  Replace
    # only exact references to the original object; an unexpected independent
    # router is a different execution graph and must fail closed.
    references = []
    if hasattr(experts, "_gate"):
        references.append((experts, "_gate"))
    runner = getattr(experts, "runner", None)
    if runner is not None and hasattr(runner, "gate"):
        references.append((runner, "gate"))
    for owner, attribute in references:
        current = getattr(owner, attribute)
        if current is gate:
            setattr(owner, attribute, router_wrapper)
        elif current is not None and current is not router_wrapper:
            raise RuntimeError(
                "sparse-MoE runtime owns an independent router reference"
            )

    aggregate_wrapper = CaptureLinearWrapper(
        mlp,
        layer_idx,
        use_buffer=use_buffer,
        max_tokens=max_tokens,
        hidden_dim=hidden_dim,
        dtype=activation_dtype,
        use_triton_copy=use_triton_copy,
        output_dim=hidden_dim,
        root_max_tokens=root_max_tokens,
        input_kind=MOE_INPUT_CAPTURE_KIND_V3,
        output_kind=MOE_AGGREGATE_OUTPUT_CAPTURE_KIND_V3,
        input_suffix="moe_input",
        output_suffix="moe_aggregate_output",
        row_indices=row_indices,
        root_stage_suffixes=root_stage_suffixes,
    )
    _replace_module(layer, mlp, aggregate_wrapper)
    if get_mlp(layer) is not aggregate_wrapper:
        raise RuntimeError("sparse-MoE aggregate wrapper was not installed")
    return aggregate_wrapper, router_wrapper


def allocate_moe_capture_buffers(
    model: torch.nn.Module,
    max_tokens: int,
    dtype: torch.dtype = torch.bfloat16,
    force_buffer_mode: bool = False,
    use_triton_copy: bool = False,
) -> list[tuple[int, torch.Tensor]]:
    """Allocate pre-capture buffers on all CaptureFusedMoE modules.

    Walks the model tree, finds all FusedMoE instances (which are
    CaptureFusedMoE after OOT registration), and allocates a
    [max_tokens, hidden_size] buffer on the same CUDA device.

    SharedFusedMoE modules ALWAYS get ``_use_buffer=True`` because
    verallm::capture splitting ops inside SharedFusedMoE cause an AOT
    autograd tree-spec mismatch (torch.Size crosses graph split boundary).
    Regular FusedMoE modules only get ``_use_buffer=True`` when
    *force_buffer_mode* is set (preserving splitting-ops capture by default).

    Note: Router logits cannot be captured at this level because
    SharedFusedMoE in overlapped mode receives hidden_states as
    router_logits (dummy) — the real logits are computed internally
    by the gate inside forward_impl().

    Returns:
        List of (layer_idx, buffer_tensor) tuples for registration with
        ``RequestActivationTracker``.

    Args:
        model: The vLLM model (already loaded and on CUDA).
        max_tokens: Maximum concurrent decode tokens (= max_num_reqs).
        dtype: Buffer dtype (should match model compute dtype).
        force_buffer_mode: If True, set ``_use_buffer=True`` on ALL FusedMoE
            modules (not just SharedFusedMoE).
    """
    try:
        from vllm.model_executor.layers.fused_moe.layer import FusedMoE
    except ImportError:
        logger.warning("Cannot import FusedMoE; MoE buffer allocation skipped")
        return []

    try:
        from vllm.model_executor.layers.fused_moe.shared_fused_moe import (
            SharedFusedMoE,
        )
    except ImportError:
        SharedFusedMoE = None

    buffers: list[tuple[int, torch.Tensor]] = []
    total_bytes = 0
    n_shared = 0
    for child in model.modules():
        if not isinstance(child, FusedMoE):
            continue
        layer_idx = getattr(child, "_layer_idx", -1)
        if layer_idx < 0:
            continue
        hidden_size = getattr(child, "hidden_size", 0)
        if hidden_size <= 0:
            continue
        dev = next(child.parameters()).device
        buf = torch.zeros(max_tokens, hidden_size, dtype=dtype, device=dev)
        child._capture_buf = buf
        # SharedFusedMoE MUST use buffer mode (splitting ops crash AOT autograd).
        # Regular FusedMoE uses buffer mode only when explicitly requested.
        is_shared = SharedFusedMoE is not None and isinstance(child, SharedFusedMoE)
        child._use_buffer = is_shared or force_buffer_mode
        child._use_triton_copy = use_triton_copy
        if is_shared:
            n_shared += 1
        buffers.append((layer_idx, buf))
        total_bytes += buf.nelement() * buf.element_size()

    logger.info(
        "Allocated %d MoE capture buffers (%d shared, %.1f MB total, max_tokens=%d)",
        len(buffers), n_shared, total_bytes / 1e6, max_tokens,
    )
    return buffers


def _infer_layer_idx_from_name(layer_name: object) -> int:
    """Best-effort layer index extraction from vLLM layer_name prefix."""
    if not isinstance(layer_name, str):
        return -1
    for pattern in _LAYER_NAME_PATTERNS:
        match = pattern.search(layer_name)
        if match is not None:
            return int(match.group(1))
    return -1


# Populated by _register() — the cls.name values that custom_ops must
# enable with "+" for forward_cuda dispatch (where capture ops live).
ENABLED_OP_NAMES: list[str] = []


def _register():
    """Register CaptureFusedMoE as OOT replacement.

    Called by verallm.vllm_plugin.__init__.register_verathos_plugin().
    Separated into a function so the import-time side effect only happens
    when explicitly requested.
    """
    from vllm.model_executor.custom_op import CustomOp
    from vllm.model_executor.layers.fused_moe.layer import FusedMoE
    try:
        from vllm.model_executor.layers.fused_moe.shared_fused_moe import (
            SharedFusedMoE,
        )
    except Exception:
        SharedFusedMoE = None

    class _CaptureMoEMixin:
        """Common capture logic shared by FusedMoE and SharedFusedMoE OOT classes."""

        _layer_idx: int = -1
        _capture_buf: torch.Tensor | None = None
        _use_buffer: bool = False

        def _capture_inputs(
            self,
            hidden_states: torch.Tensor,
            router_logits: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            layer_idx = self._layer_idx
            if layer_idx < 0:
                layer_idx = _infer_layer_idx_from_name(getattr(self, "layer_name", None))
                if layer_idx >= 0:
                    self._layer_idx = layer_idx
            if layer_idx >= 0:
                if self._use_buffer and self._capture_buf is not None:
                    # Buffer mode: copy activation into pre-allocated buffer.
                    n = min(hidden_states.shape[0], self._capture_buf.shape[0])
                    if getattr(self, '_use_triton_copy', False):
                        # Ampere: static-grid Triton copy (avoids CUDA graph crash)
                        torch.ops.verallm.buffer_copy(self._capture_buf, hidden_states, n)
                    else:
                        # Ada/Hopper: native aten::copy_ inside CUDA graph
                        self._capture_buf[:n].copy_(hidden_states[:n])
                else:
                    # Split mode: custom op forces piecewise graph split.
                    # The op is a no-op when no tracker is active.
                    #
                    # IMPORTANT: Do NOT gate on get_active_tracker() here —
                    # it returns None during torch.compile tracing.  Gating
                    # would prevent the op from appearing in the graph.
                    _ = torch.ops.verallm.capture(hidden_states, layer_idx, 0)
            return hidden_states, router_logits

        def forward_native(
            self,
            hidden_states: torch.Tensor,
            router_logits: torch.Tensor,
        ):
            # Do not inject capture ops in forward_native. Under CUDA-graph
            # compilation this path can execute in compiled regions where
            # custom-op side effects perturb numerics for MoE models.
            # Splitting-ops capture is kept in forward_cuda.
            return super().forward_native(hidden_states, router_logits)

        def forward_cuda(
            self,
            hidden_states: torch.Tensor,
            router_logits: torch.Tensor,
        ):
            hidden_states, router_logits = self._capture_inputs(
                hidden_states, router_logits
            )
            return super().forward_cuda(hidden_states, router_logits)

    @CustomOp.register_oot(name="fused_moe")
    @CustomOp.register_oot(name="FusedMoE")
    class CaptureFusedMoE(_CaptureMoEMixin, FusedMoE):
        """FusedMoE with activation capture at graph split points."""

    registered_keys = ["FusedMoE", "fused_moe"]

    if SharedFusedMoE is not None:
        @CustomOp.register_oot(name="SharedFusedMoE")
        class CaptureSharedFusedMoE(_CaptureMoEMixin, SharedFusedMoE):
            """SharedFusedMoE with buffer-mode activation capture.

            Unlike CaptureFusedMoE (regular FusedMoE), SharedFusedMoE CANNOT
            use verallm::capture as a graph splitting op.  The calling code
            (e.g. QwenNextMoE.forward()) computes
            ``orig_shape = hidden_states.shape`` before calling us and uses it
            after.  A graph split inside our forward path forces that
            torch.Size to cross the piece boundary, which AOT autograd cannot
            serialise — causing a tree-spec mismatch assertion.

            Instead we use buffer-mode capture (aten::copy_) at the forward()
            level.  The buffer is allocated by the model-load hook (runs
            before torch.compile), so the copy IS in the compiled graph and
            replays inside CUDA graphs with zero graph splits.

            SharedFusedMoE is NOT added to ENABLED_OP_NAMES, so CustomOp
            dispatch goes through forward_native (not forward_cuda).  This
            means our mixin's forward_cuda capture path is never reached.
            """

            def forward(
                self,
                hidden_states: torch.Tensor,
                router_logits: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                # Buffer-mode capture at module entry — no graph split.
                # Uses aten::copy_ which compiles into the CUDA graph.
                # The buffer is read post-step by _readout_buffers_selective().
                layer_idx = self._layer_idx
                if layer_idx < 0:
                    layer_idx = _infer_layer_idx_from_name(
                        getattr(self, "layer_name", None)
                    )
                    if layer_idx >= 0:
                        self._layer_idx = layer_idx
                if layer_idx >= 0 and self._capture_buf is not None:
                    n = min(hidden_states.shape[0], self._capture_buf.shape[0])
                    if getattr(self, '_use_triton_copy', False):
                        torch.ops.verallm.buffer_copy(self._capture_buf, hidden_states, n)
                    else:
                        self._capture_buf[:n].copy_(hidden_states[:n])
                return SharedFusedMoE.forward(self, hidden_states, router_logits)

            def forward_cuda(
                self,
                hidden_states: torch.Tensor,
                router_logits: torch.Tensor,
            ):
                # Safety: if forward_cuda is ever called (shouldn't be — we
                # don't enable SharedFusedMoE as a custom op), skip capture
                # to avoid the graph-split tree-spec crash.
                return FusedMoE.forward_cuda(self, hidden_states, router_logits)

        registered_keys.append("SharedFusedMoE")

    # Collect the .name attribute of each OOT class — these are what
    # CustomOp.enabled() checks against custom_ops ("+name").
    # CaptureFusedMoE.name is the outermost decorator's name ("fused_moe").
    #
    # NOTE: SharedFusedMoE is intentionally EXCLUDED.  It uses buffer-mode
    # capture in forward() instead of splitting-ops capture in forward_cuda().
    # Enabling it as a custom op would route dispatch through forward_cuda,
    # injecting verallm::capture which splits the graph and causes a
    # torch.Size tree-spec mismatch in AOT autograd.
    _enabled = {CaptureFusedMoE.name}
    ENABLED_OP_NAMES.clear()
    ENABLED_OP_NAMES.extend(sorted(_enabled))

    logger.info(
        "Registered CaptureFusedMoE OOT replacements for keys: %s "
        "(custom_ops enables: %s)",
        ", ".join(registered_keys),
        ", ".join(ENABLED_OP_NAMES),
    )
    return CaptureFusedMoE
