"""Capture the live Qwen GDN cache at the prompt/decode boundary.

The wrapper is installed before vLLM graph construction but is inert unless a
request explicitly negotiated the supported GDN transition profile.  It never
alters the model result: it snapshots only the post-prefill cache entries that
vLLM already maintains for the request.
"""

from __future__ import annotations

import logging
import math
import types

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _gdn_cache_pair(module: nn.Module, context) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the two live GDN cache tensors across supported vLLM ABIs.

    vLLM 0.19 stores ``(conv, recurrent)`` directly on the module.  Older
    releases keep that pair inside a virtual-engine container.  Both layouts
    are read-only here and anything else is intentionally rejected.
    """

    cache = module.kv_cache
    if (
        isinstance(cache, (tuple, list))
        and len(cache) == 2
        and all(isinstance(item, torch.Tensor) for item in cache)
    ):
        return cache
    virtual_engine = getattr(context, "virtual_engine", None)
    if virtual_engine is None:
        raise RuntimeError("proof-v2 GDN cache has no virtual-engine selector")
    try:
        pair = cache[virtual_engine]
    except (IndexError, KeyError, TypeError) as exc:
        raise RuntimeError("proof-v2 GDN virtual-engine cache is unavailable") from exc
    if (
        not isinstance(pair, tuple)
        or len(pair) != 2
        or not all(isinstance(item, torch.Tensor) for item in pair)
    ):
        raise RuntimeError("proof-v2 GDN cache layout is unsupported")
    return pair


class CaptureGDNStateWrapper(nn.Module):
    """Transparent post-prefill state capture around one Qwen GDN module."""

    def __init__(self, original: nn.Module, layer_idx: int):
        super().__init__()
        self.original = original
        self._layer_idx = layer_idx
        self._graph_root_history_enabled = False
        self.add_module("_graph_root_history_capture", None)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.original, name)

    def forward(self, *args, **kwargs):
        result = self.original(*args, **kwargs)
        if not self._graph_root_history_enabled:
            self._capture_after_prefill()
        return result

    def install_graph_root_history_capture(
        self,
        capture: "OpaqueGDNRootHistoryCapture",
    ) -> None:
        """Record the finalizer inside vLLM's state-mutating GDN custom op."""

        original_core = getattr(self.original, "_forward_core", None)
        if (
            not callable(original_core)
            or self._graph_root_history_capture is not None
            or getattr(
                self.original,
                "_verathos_gdn_root_history_hooked",
                False,
            )
        ):
            raise RuntimeError(
                "qualified GDN root capture core hook is unavailable"
            )
        self.add_module("_graph_root_history_capture", capture)

        def hooked_core(_module, *args, **kwargs):
            result = original_core(*args, **kwargs)
            self._capture_graph_root_history()
            return result

        self.original._forward_core = types.MethodType(
            hooked_core,
            self.original,
        )
        self.original._verathos_gdn_root_history_hooked = True

    def _capture_graph_root_history(self) -> None:
        from vllm.forward_context import get_forward_context

        context = get_forward_context()
        cache = getattr(self.original, "kv_cache", None)
        if cache is None or (
            isinstance(cache, (tuple, list)) and not cache
        ):
            # vLLM profiles CUDA-graph memory before allocating its cache.
            # The core attention op is intentionally a no-op in that pass.
            return
        capture = self._graph_root_history_capture
        if not isinstance(capture, OpaqueGDNRootHistoryCapture):
            raise RuntimeError(
                "qualified GDN root capture finalizer is malformed"
            )
        capture.capture(context=context)

    def _capture_after_prefill(self) -> None:
        """Capture only a metadata layout that we can identify exactly."""

        try:
            from vllm.forward_context import get_forward_context
            from verallm.vllm_plugin.ops import get_active_tracker

            tracker = get_active_tracker()
            if tracker is None or not tracker._current_req_ids:
                return
            if not tracker.has_gdn_transition_capture():
                return
            context = get_forward_context()
            metadata = context.attn_metadata
            if not isinstance(metadata, dict):
                return
            metadata = metadata.get(self.original.prefix)
            if metadata is None:
                return
            # The transition witness currently supports only the ordinary
            # non-speculative cache.  Speculative layouts use multiple cache
            # positions per sequence and need their own authenticated ABI.
            if metadata.spec_sequence_masks is not None:
                return
            state_indices = metadata.non_spec_state_indices_tensor
            if state_indices is None or state_indices.ndim != 1:
                return
            request_count = len(tracker._current_req_ids)
            if int(state_indices.numel()) != request_count:
                return
            prefill_indices = tracker.gdn_transition_prefill_indices()
            if (
                not isinstance(prefill_indices, tuple)
                or not prefill_indices
                or tuple(sorted(set(prefill_indices))) != prefill_indices
                or any(
                    isinstance(index, bool)
                    or not isinstance(index, int)
                    or not 0 <= index < request_count
                    for index in prefill_indices
                )
            ):
                return
            selection = torch.tensor(
                prefill_indices,
                dtype=torch.long,
                device=state_indices.device,
            )
            slots = state_indices.index_select(0, selection)
            cache = _gdn_cache_pair(self.original, context)
            # vLLM stores this cache in its canonical runtime order
            # ``[slot, kernel - 1, conv_width]``.  The core temporarily
            # transposes it only for the causal-convolution kernel; the
            # verifier replay and signed ABI use the stored order.
            conv_cache = cache[0]
            recurrent_cache = cache[1]
            conv_states = conv_cache.index_select(0, slots)
            recurrent_states = recurrent_cache.index_select(0, slots)
            tracker.capture_gdn_prompt_boundary(
                layer_idx=self._layer_idx,
                request_indices=prefill_indices,
                conv_states=conv_states,
                recurrent_states=recurrent_states,
            )
        except Exception as exc:
            # A capture error must not silently become a valid transition
            # proof: the miner lacks the opening and the hard verifier rejects
            # it.  Keep serving behavior intact while preserving the detail in
            # logs for the unsupported runtime profile.
            logger.warning(
                "proof-v2 GDN boundary capture skipped for layer %d: %s",
                self._layer_idx,
                exc,
            )


def wrap_qwen_gdn_state_modules(
    layers,
    *,
    allow_uninitialized_cache: bool = False,
) -> int:
    """Wrap compatible Qwen GDN modules once their ordinary cache exists.

    The qualified opaque-MoE root path must install before vLLM allocates its
    cache so the native reducer is present during graph construction. Ordinary
    dense capture keeps the established post-allocation gate; wrapping its GDN
    module earlier would expose tracker-only Python methods to stock
    ``torch.compile`` on int4 Hopper/Blackwell runtimes.
    """

    count = 0
    for layer_idx, layer in enumerate(layers):
        # The dense residual wrapper is installed first.  Replace the module
        # on the underlying decoder, not on that transparent outer wrapper.
        owner = layer
        while isinstance(getattr(owner, "original", None), nn.Module):
            owner = owner.original
        original = getattr(owner, "linear_attn", None)
        if original is None or isinstance(original, CaptureGDNStateWrapper):
            continue
        if not (
            (allow_uninitialized_cache or hasattr(original, "kv_cache"))
            and hasattr(original, "prefix")
            and hasattr(original, "in_proj_qkvz")
            and hasattr(original, "in_proj_ba")
            and callable(getattr(original, "get_state_shape", None))
            and callable(getattr(original, "get_state_dtype", None))
        ):
            continue
        setattr(owner, "linear_attn", CaptureGDNStateWrapper(original, layer_idx))
        count += 1
    return count


class OpaqueGDNRootHistoryCapture(nn.Module):
    """Graph-integrated roots for the qualified opaque Qwen GDN cache."""

    _SUFFIXES = (
        "gdn_conv_decode_checkpoints",
        "gdn_recurrent_decode_checkpoints",
    )

    def __init__(
        self,
        *,
        sources: tuple[tuple[int, nn.Module], ...],
        max_decode_rows: int,
        history_rows_per_slot: int,
        history_slot_count: int,
        checkpoint_stride: int,
    ) -> None:
        super().__init__()
        if (
            not sources
            or isinstance(max_decode_rows, bool)
            or not isinstance(max_decode_rows, int)
            or max_decode_rows <= 0
            or isinstance(history_rows_per_slot, bool)
            or not isinstance(history_rows_per_slot, int)
            or history_rows_per_slot <= 0
            or isinstance(history_slot_count, bool)
            or not isinstance(history_slot_count, int)
            or history_slot_count <= 0
            or isinstance(checkpoint_stride, bool)
            or not isinstance(checkpoint_stride, int)
            or checkpoint_stride <= 0
        ):
            raise ValueError("qualified GDN root history geometry is malformed")
        layers = tuple(layer for layer, _module in sources)
        if layers != tuple(sorted(set(layers))):
            raise ValueError("qualified GDN root history layers are malformed")
        source_modules = tuple(module for _layer, module in sources)
        if any(
            not callable(getattr(module, "get_state_shape", None))
            or not callable(getattr(module, "get_state_dtype", None))
            for module in source_modules
        ):
            raise RuntimeError(
                "qualified GDN cache geometry is unavailable"
            )
        from zkllm.cuda import zkllm_native

        if getattr(
            zkllm_native,
            "cuda_blake3_runtime_tensor_indexed_row_roots_lanes_history_into",
            None,
        ) is None:
            raise RuntimeError(
                "native indexed GDN root reducer is unavailable"
            )
        try:
            device = next(source_modules[0].parameters()).device
        except StopIteration as exc:
            raise RuntimeError(
                "qualified GDN cache device is unavailable"
            ) from exc
        if not device.type == "cuda":
            raise RuntimeError("qualified GDN root history requires CUDA")

        # Keep these module references outside nn.Module's child registry. The
        # original decoder owns them; registering a second parent would alter
        # model traversal and state-dict structure.
        object.__setattr__(self, "_source_modules", source_modules)
        self._source_layers = layers
        self._history_rows_per_slot = history_rows_per_slot
        self._history_slot_count = history_slot_count
        self._max_decode_rows = max_decode_rows
        self._checkpoint_stride = checkpoint_stride
        history_capacity = history_rows_per_slot * history_slot_count
        if history_capacity >= 1 << 31:
            raise RuntimeError(
                "qualified GDN root history capacity is malformed"
            )
        self.register_buffer(
            "_destination_indices",
            torch.full(
                (max_decode_rows,),
                -1,
                dtype=torch.int64,
                device=device,
            ),
            persistent=False,
        )
        from verallm.proof_v3.execution_anchor import (
            execution_anchor_lane_bytes_v3,
        )

        source_records = []
        stage_records = []
        for layer, module in sources:
            shapes = module.get_state_shape()
            dtypes = module.get_state_dtype()
            if (
                not isinstance(shapes, tuple)
                or len(shapes) != 2
                or not isinstance(dtypes, tuple)
                or len(dtypes) != 2
            ):
                raise RuntimeError(
                    "qualified GDN cache geometry is malformed"
                )
            entries = []
            for cache_index, suffix in enumerate(self._SUFFIXES):
                shape = tuple(int(value) for value in shapes[cache_index])
                dtype = dtypes[cache_index]
                if (
                    not shape
                    or any(value <= 0 for value in shape)
                    or dtype
                    not in (torch.float16, torch.bfloat16, torch.float32)
                ):
                    raise RuntimeError(
                        "qualified GDN cache row geometry is malformed"
                    )
                row_width = math.prod(shape) * int(
                    torch.empty((), dtype=dtype).element_size()
                )
                stage_id = f"l{layer}.{suffix}"
                lane_width = execution_anchor_lane_bytes_v3(stage_id)
                entry = (stage_id, row_width, dtype, lane_width)
                entries.append(entry)
                stage_records.append(entry)
            source_records.append(tuple(entries))
        max_lanes = max(
            (row_width + lane_width - 1) // lane_width
            for _stage_id, row_width, _dtype, lane_width in stage_records
        )
        destination = torch.zeros(
            (len(stage_records), history_capacity, 32),
            dtype=torch.uint8,
            device=device,
        )
        # The final GDN hook hashes all layer-local cache rows in one launch.
        scratch = torch.empty(
            (len(stage_records), max_decode_rows, max_lanes, 32),
            dtype=torch.uint8,
            device=device,
        )
        row_widths = torch.tensor(
            tuple(
                row_width
                for _stage_id, row_width, _dtype, _lane_width
                in stage_records
            ),
            dtype=torch.int32,
            device=device,
        )
        lane_widths = torch.tensor(
            tuple(
                lane_width
                for _stage_id, _row_width, _dtype, lane_width
                in stage_records
            ),
            dtype=torch.int32,
            device=device,
        )
        self.register_buffer("_history", destination, persistent=False)
        self.register_buffer("_scratch", scratch, persistent=False)
        self.register_buffer("_row_widths", row_widths, persistent=False)
        self.register_buffer("_lane_widths", lane_widths, persistent=False)
        self._source_records = tuple(source_records)
        self._inventory = tuple(
            sorted(
                (
                    stage_id,
                    destination[index],
                    row_width,
                )
                for index, (
                    stage_id,
                    row_width,
                    _dtype,
                    _lane_width,
                ) in enumerate(stage_records)
            )
        )
        self._runtime_cache_geometry_armed = False

    def capture(self, *, context) -> None:
        metadata_inventory = context.attn_metadata
        if metadata_inventory is None:
            return
        if not isinstance(metadata_inventory, dict):
            raise RuntimeError(
                "qualified GDN root capture metadata is malformed"
            )
        if getattr(context, "virtual_engine", None) is None and any(
            not (
                isinstance(getattr(module, "kv_cache", None), (tuple, list))
                and len(module.kv_cache) == 2
                and all(
                    isinstance(item, torch.Tensor)
                    for item in module.kv_cache
                )
            )
            for module in self._source_modules
        ):
            # vLLM's CUDA-graph memory profile runs before it binds the
            # per-layer cache tensors. The later real graph capture records
            # the reducer after the complete cache inventory is installed.
            return

        runtime_sources = []
        runtime_source_indices = []
        invalid_geometry = None
        row_count = None
        for source_position, module in enumerate(self._source_modules):
            metadata = metadata_inventory.get(module.prefix)
            if metadata is None:
                invalid_geometry = "metadata"
                break
            if metadata.spec_sequence_masks is not None:
                raise RuntimeError(
                    "qualified GDN root capture does not support speculation"
                )
            source_indices = metadata.non_spec_state_indices_tensor
            if (
                not isinstance(source_indices, torch.Tensor)
                or source_indices.dtype != torch.int32
                or source_indices.ndim != 1
                or not source_indices.is_cuda
                or source_indices.device != self._destination_indices.device
            ):
                invalid_geometry = "indices"
                break
            source_row_count = int(source_indices.shape[0])
            if row_count is None:
                row_count = source_row_count
            if source_row_count != row_count:
                invalid_geometry = "indices"
                break
            try:
                cache_pair = _gdn_cache_pair(module, context)
            except RuntimeError:
                invalid_geometry = "source"
                break
            for cache_index, (
                _stage_id,
                row_width,
                dtype,
                _lane_width,
            ) in enumerate(self._source_records[source_position]):
                cache = cache_pair[cache_index]
                if (
                    cache.dtype != dtype
                    or not 2 <= cache.ndim <= 4
                    or not cache.is_cuda
                    or cache.device != source_indices.device
                ):
                    invalid_geometry = "source"
                    break
                inner_elements = 1
                inner_contiguous = True
                for dimension in range(cache.ndim - 1, 0, -1):
                    if int(cache.stride(dimension)) != inner_elements:
                        inner_contiguous = False
                        break
                    inner_elements *= int(cache.shape[dimension])
                row_bytes = inner_elements * int(cache.element_size())
                row_stride_bytes = (
                    int(cache.stride(0)) * int(cache.element_size())
                )
                if (
                    not inner_contiguous
                    or row_bytes != row_width
                    or row_stride_bytes < row_bytes
                ):
                    invalid_geometry = "row"
                    break
                runtime_sources.append(cache)
                runtime_source_indices.append(source_indices)
            if invalid_geometry is not None:
                break

        if invalid_geometry is not None:
            if self._runtime_cache_geometry_armed:
                raise RuntimeError(
                    "qualified GDN cache "
                    f"{invalid_geometry} changed after admission"
                )
            # vLLM profiles CUDA-graph memory with temporary cache tensors
            # before allocating the admitted cache inventory. Arm only after
            # the first complete, exact runtime inventory is visible.
            return
        if row_count is None or not 0 < row_count <= self._max_decode_rows:
            raise RuntimeError(
                "qualified GDN root history scheduler geometry is malformed"
            )

        self._runtime_cache_geometry_armed = True
        destination_indices = self._destination_indices[:row_count]
        torch.ops.verallm.activation_indexed_tensor_row_roots_lanes_history(
            self._history,
            self._scratch,
            runtime_sources,
            self._row_widths,
            self._lane_widths,
            runtime_source_indices,
            destination_indices,
        )

    def proof_capture_gdn_root_history(self):
        if not self._runtime_cache_geometry_armed:
            raise RuntimeError(
                "qualified GDN root history graph inventory is incomplete"
            )
        return (
            (
                tuple(self._inventory),
                self._destination_indices,
                self._history_rows_per_slot,
                self._history_slot_count,
                self._max_decode_rows,
                self._checkpoint_stride,
            ),
        )


def wrap_qwen_gdn_root_history_modules_v3(
    layers,
    *,
    max_decode_rows: int,
    history_rows_per_slot: int,
    history_slot_count: int,
    checkpoint_stride: int,
) -> tuple[int, int]:
    """Install the exact qualified graph-integrated GDN root inventory."""

    wrap_qwen_gdn_state_modules(
        layers,
        allow_uninitialized_cache=True,
    )
    wrappers = []
    for layer_idx, layer in enumerate(layers):
        owner = layer
        while isinstance(getattr(owner, "original", None), nn.Module):
            owner = owner.original
        wrapper = getattr(owner, "linear_attn", None)
        if isinstance(wrapper, CaptureGDNStateWrapper):
            wrappers.append((layer_idx, wrapper))
    if not wrappers:
        raise RuntimeError("qualified GDN root capture found no GDN layers")
    capture = OpaqueGDNRootHistoryCapture(
        sources=tuple(
            (layer_idx, wrapper.original)
            for layer_idx, wrapper in wrappers
        ),
        max_decode_rows=max_decode_rows,
        history_rows_per_slot=history_rows_per_slot,
        history_slot_count=history_slot_count,
        checkpoint_stride=checkpoint_stride,
    )
    for _layer_idx, wrapper in wrappers:
        wrapper._graph_root_history_enabled = True
    wrappers[-1][1].install_graph_root_history_capture(capture)
    return len(wrappers), len(capture._inventory)


__all__ = [
    "CaptureGDNStateWrapper",
    "OpaqueGDNRootHistoryCapture",
    "_gdn_cache_pair",
    "wrap_qwen_gdn_root_history_modules_v3",
    "wrap_qwen_gdn_state_modules",
]
