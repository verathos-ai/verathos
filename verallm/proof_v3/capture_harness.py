"""Reusable production-capture harness (GATE-qualified).

Wraps the exact production sequence proven in GATE.1-3: build a capture-
enabled VllmMiner (build-time CaptureLinearWrapper, CUDA graphs on) + the
RequestActivationTracker + BatchAwareEngine, and drive requests through the
real batch engine so captures are REAL (non-zero, Y==X@W, demux-clean).

NOT production serving code -- this is the qualification/audit driver that
stands up the real capture path outside the FastAPI server.  The capture
itself is the production tracker; only the request loop is synchronous.
"""

from __future__ import annotations

import os

__all__ = [
    "build_capture_miner",
    "serve_and_capture",
    "serve_reduction_requests",
    "CaptureMiner",
]


def _select_economic_capture_buffers(
    capture_buffers,
    root_row_aliases,
    *,
    suffix_filter,
    required_root_row_aliases=(),
):
    """Select raw witnesses by their canonical signed stage suffix."""

    candidates = list(tuple(capture_buffers) + tuple(root_row_aliases))
    candidate_keys = tuple(
        (int(item[0]), str(item[1])) for item in candidates
    )
    # Mirror the production server's convergence of graph-native reduction
    # aliases into the ordinary raw-witness inventory.  Split-mode attention
    # may expose its compact K/V row only through the whole-step reduction
    # buffer rather than the projection wrapper's own output storage.
    for item in required_root_row_aliases:
        key = (int(item[0]), str(item[1]))
        if suffix_filter is not None and key[1] not in suffix_filter:
            continue
        if key not in candidate_keys:
            candidates.append(item)
            candidate_keys += (key,)
    selected = tuple(
        item
        for item in candidates
        if suffix_filter is None or str(item[1]) in suffix_filter
    )
    if suffix_filter is not None and not selected:
        raise RuntimeError(
            "execution-anchor stage filter matched no capture buffers"
        )

    selected_keys = tuple(
        (int(item[0]), str(item[1])) for item in selected
    )
    for layer_idx, suffix, _buffer in required_root_row_aliases:
        key = (int(layer_idx), str(suffix))
        if suffix_filter is not None and key[1] not in suffix_filter:
            continue
        if selected_keys.count(key) != 1:
            raise RuntimeError(
                "economic K/V capture witness is missing or ambiguous"
            )
    return selected


def _select_qualification_economic_capture_buffers(
    capture_buffers,
    root_row_aliases,
    *,
    required_root_row_aliases=(),
):
    """Mirror production's complete build-time witness registration.

    Execution-anchor root selection and raw economic witness registration are
    separate inventories.  The request tracker applies the authenticated raw
    stage filter at capture time, so registering every graph-integrated raw
    buffer does not retain every buffer for a request.
    """

    return _select_economic_capture_buffers(
        capture_buffers,
        root_row_aliases,
        suffix_filter=None,
        required_root_row_aliases=required_root_row_aliases,
    )


def _reduction_capture_mode_v3(
    qkv,
    output,
    *,
    qkv_output_buffer=None,
    o_input_buffer=None,
    row_indices=None,
) -> str:
    """Classify reduction wrappers by their actual execution mode.

    Split-mode wrappers may still own compact selected-row buffers used by
    execution-root capture. Those auxiliary buffers do not make the
    reduction trace buffer-mode.
    """

    qkv_buffer_mode = bool(qkv._use_buffer)
    output_buffer_mode = bool(output._use_buffer)
    selected_row_capture = (
        qkv_output_buffer is not None or o_input_buffer is not None
    )
    if not qkv_buffer_mode and not output_buffer_mode:
        if not selected_row_capture:
            return "split"
        if (
            qkv_output_buffer is None
            or o_input_buffer is None
            or row_indices is None
        ):
            raise RuntimeError(
                "selected-row reduction capture is incomplete"
            )
        return "gather"
    if qkv_buffer_mode != output_buffer_mode:
        raise RuntimeError(
            "reduction capture wrappers mix split and buffer modes"
        )
    if qkv._capture_output_buf is None or output._capture_buf is None:
        raise RuntimeError(
            "buffer-mode reduction capture lacks canonical buffers"
        )
    # A backend may deliberately mix a whole-step QKV destination with a
    # selected-row o-projection destination.  The shared index merely
    # identifies gathered destinations; it does not make every registered
    # buffer a gather layout.  Preserve the wrappers' actual layouts exactly
    # as production server registration does, without GPU/quant branching.
    qkv_selected = getattr(qkv, "_capture_row_indices", None) is not None
    output_selected = (
        getattr(output, "_capture_row_indices", None) is not None
    )
    if row_indices is not None:
        if qkv_output_buffer is None or o_input_buffer is None:
            raise RuntimeError(
                "selected-row reduction capture is incomplete"
            )
        if not qkv_selected and not output_selected:
            raise RuntimeError(
                "selected-row reduction capture has no gathered buffer"
            )
        if qkv_selected and output_selected:
            return "gather"
        return "mixed"
    return "buffer"


def _reduction_wrapper_is_dedicated_buffer_v3(wrappers) -> bool:
    """Mirror production's base-inventory exclusion for reduction wrappers."""

    import torch

    row_indices = wrappers.get("row_indices")
    qkv = wrappers.get("qkv")
    output = wrappers.get("o")
    return (
        not isinstance(row_indices, torch.Tensor)
        or bool(getattr(qkv, "_use_buffer", False))
        or bool(getattr(output, "_use_buffer", False))
    )


class CaptureMiner:
    def __init__(self, miner, batch_engine, tracker, layers):
        self.miner = miner
        self.be = batch_engine
        self.tr = tracker
        self.layers = layers
        self.model = miner.model


def build_capture_miner(model_path: str, *, gpu_mem: float = 0.55,
                        max_model_len: int | None = 2048,
                        max_num_seqs: int | None = 8,
                        reduction_layers=None,
                        max_num_batched_tokens: int | None = None,
                        execution_anchor_stage_suffixes=None,
                        execution_anchor_checkpoint_stride: int = 1,
                        proof_v3_moe_router_encoding_id: str | None = None,
                        proof_v3_capture_adapter_id: str | None = None,
                        proof_v3_capture_selected_layer_count: int | None = None,
                        enable_prefix_caching: bool = False,
                        async_scheduling: bool | None = None,
                        enable_finished_cache_retention: bool = False,
                        quant: str = "fp16",
                        revision: str | None = None,
                        ) -> CaptureMiner:
    import torch

    suffix_filter = (
        None
        if execution_anchor_stage_suffixes is None
        else frozenset(
            str(suffix) for suffix in execution_anchor_stage_suffixes
        )
    )
    if suffix_filter is None:
        os.environ.pop("VERALLM_CAPTURE_ROOT_SUFFIXES", None)
    elif suffix_filter:
        os.environ["VERALLM_CAPTURE_ROOT_SUFFIXES"] = ",".join(
            sorted(suffix_filter)
        )
    else:
        raise ValueError("execution-anchor stage filter is empty")
    if (
        isinstance(execution_anchor_checkpoint_stride, bool)
        or not isinstance(execution_anchor_checkpoint_stride, int)
        or not 1 <= execution_anchor_checkpoint_stride <= 64
    ):
        raise ValueError("execution-anchor checkpoint stride is invalid")
    if execution_anchor_checkpoint_stride == 1:
        os.environ.pop(
            "VERALLM_CAPTURE_ROOT_CHECKPOINT_STRIDE",
            None,
        )
    else:
        os.environ["VERALLM_CAPTURE_ROOT_CHECKPOINT_STRIDE"] = str(
            execution_anchor_checkpoint_stride
        )
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.setdefault("VLLM_USE_AOT_COMPILE", "0")
    if reduction_layers:
        # picked up by the dense load hook: installs FULL-ROW aux capture
        # buffers (qkv output + o_proj input) on exactly these layers
        os.environ["VERALLM_REDUCTION_AUDIT_LAYERS"] = ",".join(
            str(int(layer)) for layer in reduction_layers)

    from verallm.miner.vllm_backend import VllmMiner
    from verallm.miner.batch_engine import BatchAwareEngine
    from verallm.miner.activation_tracker import RequestActivationTracker
    from verallm.vllm_plugin.capture_linear import (
        attach_capture_ops, CaptureLinearWrapper, CaptureDecoderLayerWrapper)
    from verallm.vllm_plugin.ops import set_active_tracker
    from verallm.types import ModelSpec

    temp_spec = ModelSpec(
        model_id=model_path, weight_merkle_root=b"\x00" * 32, num_layers=0,
        hidden_dim=0, num_heads=0, head_dim=0, intermediate_dim=0,
        vocab_size=0, activation="silu", norm_type="rmsnorm", attention_type="gqa")

    extra = {}
    if max_model_len is not None:
        extra["max_model_len"] = int(max_model_len)
    if max_num_seqs is not None:
        extra["max_num_seqs"] = int(max_num_seqs)
    if revision is not None:
        extra["revision"] = revision
    if max_num_batched_tokens is not None:
        extra["max_num_batched_tokens"] = int(max_num_batched_tokens)
    if enable_prefix_caching:
        extra["enable_prefix_caching"] = True
    elif (
        reduction_layers
        or os.environ.get("VERALLM_CAPTURE_FULL_ROWS", "") not in ("", "0")
        or os.environ.get("VERALLM_CAPTURE_ROOT_ROWS", "") not in ("", "0")
    ):
        # Audited serving REQUIRES a full prefill: a prefix-cache hit
        # skips recomputation of the cached prompt rows, so the
        # candidate-pool rows and streaming execution-anchor leaves for
        # those positions never exist.  The commits fail closed on an
        # incomplete request; disabling prefix caching is the production
        # configuration for audit-tracked serving.
        extra["enable_prefix_caching"] = False
    if async_scheduling is not None:
        extra["async_scheduling"] = bool(async_scheduling)
    if enable_prefix_caching or enable_finished_cache_retention:
        extra["scheduler_cls"] = (
            "verallm.miner.proof_cache_scheduler.ProofCacheScheduler"
        )
    miner = VllmMiner(model_path, temp_spec, None)
    miner.setup_vllm(quant=str(quant), gpu_memory_utilization=gpu_mem,
                     proof_v2_full_trace_capture=True,
                     proof_v3_moe_router_encoding_id=(
                         proof_v3_moe_router_encoding_id
                     ),
                     proof_v3_capture_adapter_id=(
                         proof_v3_capture_adapter_id
                     ),
                     proof_v3_capture_selected_layer_count=(
                         proof_v3_capture_selected_layer_count
                     ),
                     **extra)
    if not miner._use_cuda_graphs:
        raise RuntimeError("CUDA graphs not active - eager is not a production claim")

    be = BatchAwareEngine(miner.llm)
    layers = miner._get_layers()
    reduction_wrappers = getattr(
        be.model_runner, "_verathos_reduction_wrappers", None) or {}
    reduction_wrapper_ids = {
        id(wrapper)
        for wrappers in reduction_wrappers.values()
        if _reduction_wrapper_is_dedicated_buffer_v3(wrappers)
        for wrapper in (wrappers["qkv"], wrappers["o"])
    }
    tr = RequestActivationTracker(be.model_runner, backend="splitting_ops")
    tr.install_hooks(layers=layers, is_moe_layer_fn=lambda _: False,
                     get_mlp_fn=miner._get_mlp, get_gate_proj_fn=miner._get_gate_proj)
    gate0 = miner._get_gate_proj(miner._get_mlp(layers[0]))
    use_buffer = bool(getattr(gate0, "_use_buffer", False))
    attach_capture_ops(model=miner.model, layers=layers, is_moe=False,
                       get_mlp_fn=miner._get_mlp, get_gate_proj_fn=miner._get_gate_proj,
                       is_moe_layer_fn=lambda _: False, wrap_lm_head=not use_buffer)
    set_active_tracker(tr)
    bufs, seen = [], set()
    response_stamp_records = []
    split_economic_keys = set()
    root_wrappers, seen_root_wrappers = [], set()
    split_alias_wrappers, seen_split_alias_wrappers = [], set()
    runtime_root_only_keys = set()
    for L in layers:
        for m in L.modules():
            split_stage_source = getattr(
                m,
                "proof_capture_split_stages",
                None,
            )
            if split_stage_source is not None:
                split_economic_keys.update(split_stage_source())
            runtime_root_only_source = getattr(
                m,
                "proof_capture_runtime_root_only_stages",
                None,
            )
            if runtime_root_only_source is not None:
                runtime_root_only_keys.update(runtime_root_only_source())
            root_buffer_source = getattr(
                m,
                "proof_capture_root_buffers",
                None,
            )
            if (
                root_buffer_source is not None
                and id(m) not in seen_root_wrappers
                and root_buffer_source()
            ):
                seen_root_wrappers.add(id(m))
                root_wrappers.append(m)
            if (
                isinstance(m, CaptureLinearWrapper)
                and id(m) not in reduction_wrapper_ids
                and id(m) not in seen_split_alias_wrappers
                and m.proof_capture_split_row_aliases()
            ):
                seen_split_alias_wrappers.add(id(m))
                split_alias_wrappers.append(m)
            has_raw_capture = (
                (
                    isinstance(m, CaptureLinearWrapper)
                    and (
                        m._capture_buf is not None
                        or m._capture_output_buf is not None
                    )
                )
                or (
                    isinstance(m, CaptureDecoderLayerWrapper)
                    and bool(m.proof_capture_buffers())
                )
            )
            if (
                has_raw_capture
                and id(m) not in reduction_wrapper_ids
                and id(m) not in seen
            ):
                seen.add(id(m))
                bufs.append(m)
            response_stamp_source = getattr(
                type(m),
                "proof_capture_response_stamp_buffer",
                None,
            )
            if response_stamp_source is not None:
                response_stamp_records.extend(response_stamp_source(m))
    tr.register_capture_buffers(bufs)
    if response_stamp_records:
        if len(response_stamp_records) != 1:
            raise RuntimeError(
                "proof capture response-stamp inventory is ambiguous"
            )
        if tr._response_stamp_buffer is None:
            tr.register_response_stamp_capture(response_stamp_records[0])
    root_buffers = [
        item
        for wrapper in root_wrappers
        for item in wrapper.proof_capture_root_buffers()
        if suffix_filter is None or item[1] in suffix_filter
    ]
    root_row_aliases = [
        item
        for wrapper in bufs
        if isinstance(wrapper, CaptureLinearWrapper)
        for item in wrapper.proof_capture_root_row_aliases()
    ]
    split_root_row_aliases = [
        item
        # Qualified sparse QKV capture exposes its exact whole-step K/V alias
        # without allocating a duplicate root buffer.
        for wrapper in split_alias_wrappers
        for item in wrapper.proof_capture_split_row_aliases()
    ]
    if root_buffers:
        tr.register_execution_anchor_root_buffers(root_buffers)
        root_histories = tuple(
            item
            for wrapper in root_wrappers
            for history_source in (
                getattr(wrapper, "proof_capture_root_history", None),
            )
            if history_source is not None
            for item in history_source()
        )
        if root_histories:
            if len(root_histories) != 1:
                raise RuntimeError(
                    "proof capture root history inventory is ambiguous"
                )
            tr.register_execution_anchor_root_history(*root_histories[0])
        root_finalizers = tuple(
            item
            for wrapper in root_wrappers
            for finalizer_source in (
                getattr(wrapper, "proof_capture_root_finalizer", None),
            )
            if finalizer_source is not None
            for item in finalizer_source()
        )
        if root_finalizers:
            if len(root_finalizers) != 1:
                raise RuntimeError(
                    "proof capture root finalizer inventory is ambiguous"
                )
            tr.register_execution_anchor_root_post_graph_finalizer(
                root_finalizers[0]
            )
        root_staging_buffers = [
            (
                int(binding.stage_id.split(".", 1)[0][1:]),
                binding.stage_id.split(".", 1)[1],
                staging,
                binding.row_width,
            )
            for wrapper in root_wrappers
            for binding_source in (
                getattr(wrapper, "_runtime_root_bindings", None),
            )
            if binding_source is not None
            for binding in binding_source()
            for staging in (
                getattr(
                    binding.owner,
                    binding.staging_attribute,
                ),
            )
            if staging is not None
            and (
                suffix_filter is None
                or binding.stage_id.split(".", 1)[1] in suffix_filter
            )
        ]
        tr.register_execution_anchor_root_staging_buffers(
            root_staging_buffers
        )
        tr.register_execution_anchor_root_retention(
            tuple(
                item
                for wrapper in root_wrappers
                for retention_source in (
                    getattr(wrapper, "proof_capture_root_retention", None),
                )
                if retention_source is not None
                for item in retention_source()
            )
        )
        tr.register_split_execution_anchor_aliases(
            split_root_row_aliases
        )
    gdn_root_histories = tuple(
        item
        for layer in layers
        for module in layer.modules()
        for history_source in (
            getattr(module, "proof_capture_gdn_root_history", None),
        )
        if history_source is not None
        for item in history_source()
    )
    if gdn_root_histories:
        if len(gdn_root_histories) != 1:
            raise RuntimeError(
                "proof capture GDN root history inventory is ambiguous"
            )
        tr.register_gdn_decode_checkpoint_root_history(
            *gdn_root_histories[0]
        )
    gdn_modules = []
    for layer_index, layer in enumerate(layers):
        owner = layer
        while isinstance(getattr(owner, "original", None), torch.nn.Module):
            owner = owner.original
        module = getattr(owner, "linear_attn", None)
        while isinstance(getattr(module, "original", None), torch.nn.Module):
            module = module.original
        if isinstance(module, torch.nn.Module) and hasattr(module, "kv_cache"):
            gdn_modules.append((layer_index, module))
    tr.register_gdn_state_modules(gdn_modules)
    if enable_prefix_caching:
        gdn_layers = {layer for layer, _module in gdn_modules}
        attention_layers = tuple(
            layer for layer in range(len(layers)) if layer not in gdn_layers
        )
        if not attention_layers:
            raise RuntimeError(
                "prefix-cache qualification found no attention layers"
            )
        tr.register_prefix_cache_attention_layers(attention_layers)
    reduction_root_row_aliases = []
    if reduction_wrappers:
        reduction_buffers = []
        split_reduction_stages = []
        for layer, wrappers in sorted(reduction_wrappers.items()):
            qkv = wrappers["qkv"]
            output = wrappers.get(
                "qkv_output_buffer",
                qkv._capture_output_buf,
            )
            o_input = wrappers.get(
                "o_input_buffer",
                wrappers["o"]._capture_buf,
            )
            row_indices = wrappers.get(
                "row_indices",
                getattr(qkv, "_capture_row_indices", None),
            )
            mode = _reduction_capture_mode_v3(
                qkv,
                wrappers["o"],
                qkv_output_buffer=output,
                o_input_buffer=o_input,
                row_indices=row_indices,
            )
            if mode == "split":
                split_reduction_stages.extend(
                    (
                        (layer, "attention_qkv_output"),
                        (layer, "attention_o_input"),
                    )
                )
                continue
            if mode in ("gather", "mixed"):
                tr.register_capture_row_indices(row_indices)
            qkv_whole_step = (
                getattr(qkv, "_capture_row_indices", None) is None
            )
            o_whole_step = (
                getattr(wrappers["o"], "_capture_row_indices", None)
                is None
            )
            reduction_buffers.extend(
                (
                    (
                        layer,
                        "attention_qkv_output",
                        output,
                        qkv_whole_step,
                    ),
                    (
                        layer,
                        "attention_o_input",
                        o_input,
                        o_whole_step,
                    ),
                )
            )
            root_owner = qkv
            kv_aliases = ()
            while isinstance(root_owner, CaptureLinearWrapper):
                kv_aliases = root_owner.proof_capture_root_row_aliases(
                    output
                )
                if kv_aliases:
                    break
                root_owner = getattr(root_owner, "original", None)
            reduction_buffers.extend(
                (
                    alias_layer,
                    alias_suffix,
                    alias_buffer,
                    qkv_whole_step,
                )
                for alias_layer, alias_suffix, alias_buffer in kv_aliases
            )
            reduction_root_row_aliases.extend(kv_aliases)
        if reduction_buffers and split_reduction_stages:
            raise RuntimeError(
                "reduction capture cannot mix split and buffer modes"
            )
        if reduction_buffers:
            tr.register_reduction_buffers(reduction_buffers)
        else:
            tr.register_split_reduction_stages(split_reduction_stages)
    if (
        os.environ.get("VERALLM_CAPTURE_FULL_ROWS", "") not in ("", "0")
        or os.environ.get("VERALLM_CAPTURE_ROOT_ROWS", "") not in ("", "0")
    ):
        economic_buffers = _select_qualification_economic_capture_buffers(
            tr._capture_buffers,
            root_row_aliases,
            required_root_row_aliases=reduction_root_row_aliases,
        )
        if response_stamp_records and not any(
            (int(layer), str(suffix)) == (0, "residual_in")
            for layer, suffix, _buffer in economic_buffers
        ):
            economic_buffers += (response_stamp_records[0][:3],)
        root_keys = {
            (int(layer), str(suffix))
            for layer, suffix, _buffer, _width in root_buffers
        }
        raw_keys = {
            (int(layer), str(suffix))
            for layer, suffix, _buffer in economic_buffers
        }
        # Split-mode wrappers can still own graph-resident selected-row
        # buffers. Prefer those buffers; only stages without raw storage may
        # fall back to the eager split callback.
        split_economic_keys.difference_update(raw_keys)
        staging_keys = set(tr._capture_root_staging_buffers)
        missing_raw = tuple(
            sorted(
                root_keys
                - raw_keys
                - staging_keys
                - split_economic_keys
                - runtime_root_only_keys
            )
        )
        if missing_raw:
            diagnostics = []
            missing_set = set(missing_raw)
            for wrapper in root_wrappers:
                if not isinstance(wrapper, CaptureLinearWrapper):
                    continue
                wrapper_stages = (
                    (
                        int(wrapper._layer_idx),
                        str(wrapper._capture_input_suffix),
                        wrapper._capture_buf,
                    ),
                    (
                        int(wrapper._layer_idx),
                        str(wrapper._capture_output_root_suffix),
                        wrapper._capture_output_buf,
                    ),
                )
                for layer, suffix, raw_buffer in wrapper_stages:
                    if (layer, suffix) not in missing_set:
                        continue
                    diagnostics.append(
                        (
                            layer,
                            suffix,
                            wrapper._capture_row_indices is not None,
                            bool(wrapper._use_buffer),
                            int(wrapper._capture_input_dim),
                            int(wrapper._capture_output_root_dim),
                            (
                                None
                                if raw_buffer is None
                                else tuple(int(value) for value in raw_buffer.shape)
                            ),
                        )
                    )
            raise RuntimeError(
                "execution-anchor root inventory lacks selected-row "
                f"storage: {missing_raw!r}; "
                f"wrapper_diagnostics={tuple(diagnostics[:8])!r}"
            )
        tr.register_economic_pool_buffers(economic_buffers)
        tr.register_split_economic_pool_stages(
            sorted(split_economic_keys)
        )
    if tr.has_capture_buffers:
        be.set_step_output_callback(tr.snapshot_trace_step_buffers)
        be.set_finished_output_callback(tr.snapshot_capture_buffers)
    if hasattr(miner.model, "compute_logits"):
        tr.install_lm_head_hook(miner.model, capture_logits=False)
    return CaptureMiner(miner, be, tr, layers)


def serve_and_capture(cm: CaptureMiner, request_id: str, prompt: str,
                      *, max_tokens: int = 4, capture_logits: bool = True,
                      ignore_eos: bool = False,
                      capture_full_trace: bool = True,
                      reduction_pool: int | None = None,
                      reduction_layer_ids=None,
                      reduction_row_positions=None,
                      capture_prefill_rows: bool = False,
                      capture_economic_pool: bool = False,
                      economic_pool_stage_ids=None,
                      capture_execution_anchors: bool = False,
                      capture_gdn_transition: bool = False,
                      execution_anchor_reveal_rows=None,
                      execution_anchor_retain_lanes=None,
                      gdn_decode_checkpoint_stride: int = 0,
                      gdn_decode_checkpoint_target_forwarded_rows: int = 0,
                      gdn_decode_qualification_extra_forwarded_rows=()):
    """Serve one prompt through the real batch engine; return (captured, out).

    ``reduction_pool``: when set, the SAME request additionally carries
    the reduction-audit capture plane (full paged-cache K/V + pooled
    qkv/o rows on the engine's reduction layers) and the return value
    becomes ``(captured, out, reduction_material)`` -- the dual-plane
    serve the economic capture-kv attention embedding uses.  Requires
    the engine to have been built with ``reduction_layers``.
    ``reduction_layer_ids`` narrows retention to the validator-selected
    post-nonce subset; the graph-integrated wrappers remain installed for the
    signed universe so selection cannot change the serving graph.
    ``reduction_row_positions`` similarly retains the exact nonce-selected
    absolute prompt rows instead of the legacy prompt tail.

    ``capture_prefill_rows`` is the legacy full-sequence diagnostic mode.
    Production economic audits use ``capture_economic_pool`` instead: the
    graph-integrated whole-step buffers retain only the canonical bounded
    prompt/decode pool through RequestActivationTracker.  An optional
    ``economic_pool_stage_ids`` filter retains only those registered runtime
    stages without narrowing the full-sequence anchor inventory.

    ``execution_anchor_reveal_rows`` switches anchor capture into post-nonce
    replay mode.  It maps selected stage ids to absolute rows, retaining only
    their raw values plus one 32-byte leaf hash per sequence row.
    ``execution_anchor_retain_lanes`` uses the production bounded-lane path
    for coordinates whose exact row is selected only after a succinct
    transcript has been built.
    ``capture_gdn_transition`` additionally freezes the live post-prefill GDN
    cache boundary needed by the authenticated decode replay.
    ``gdn_decode_qualification_extra_forwarded_rows`` is reserved for offline
    qualification. It adds sparse state endpoints without changing the signed
    production checkpoint stride.
    """

    from vllm import SamplingParams

    extra = {}
    if reduction_pool or capture_economic_pool or capture_execution_anchors:
        tokenizer = cm.miner.llm.get_tokenizer()
        extra = dict(
            capture_reduction_audit=bool(reduction_pool),
            reduction_layer_ids=(
                tuple(reduction_layer_ids)
                if reduction_layer_ids is not None
                else None
            ),
            reduction_row_positions=(
                tuple(reduction_row_positions)
                if reduction_row_positions is not None
                else None
            ),
            capture_economic_pool=bool(capture_economic_pool),
            economic_pool_stage_ids=(
                economic_pool_stage_ids
                if capture_economic_pool
                else None
            ),
            reduction_pool=int(reduction_pool or 32),
            reduction_prompt_len=len(tokenizer(prompt)["input_ids"]))
    cm.tr.register_request(request_id, request_id,
                           capture_logits=capture_logits,
                           capture_full_trace=bool(capture_full_trace),
                           capture_prefill_rows=bool(capture_prefill_rows),
                           capture_gdn_transition=bool(
                               capture_gdn_transition),
                           capture_execution_anchors=bool(
                               capture_execution_anchors),
                           execution_anchor_reveal_rows=(
                               execution_anchor_reveal_rows
                           ),
                           execution_anchor_retain_lanes=(
                               execution_anchor_retain_lanes
                           ),
                           gdn_decode_checkpoint_stride=int(
                               gdn_decode_checkpoint_stride
                           ),
                           gdn_decode_checkpoint_target_forwarded_rows=int(
                               gdn_decode_checkpoint_target_forwarded_rows
                           ),
                           gdn_decode_qualification_extra_forwarded_rows=tuple(
                               int(value)
                               for value in
                               gdn_decode_qualification_extra_forwarded_rows
                           ),
                           **extra)
    queue = cm.be.add_request(request_id, prompt,
                              SamplingParams(
                                  temperature=0.0,
                                  max_tokens=max_tokens,
                                  min_tokens=(
                                      max_tokens if ignore_eos else 0
                                  ),
                                  ignore_eos=bool(ignore_eos),
                              ))
    while cm.be.has_active_requests():
        cm.be.step_and_distribute()
    captured = cm.tr.finalize_activations(request_id)
    material = (cm.tr.reduction_material(request_id)
                if reduction_pool else None)
    economic_material = None
    anchor_commitments = None
    if capture_economic_pool or capture_execution_anchors:
        queued_storage = getattr(queue, "queue", None)
        if queued_storage is None:
            queued_storage = getattr(queue, "_queue", None)
        queued = list(queued_storage or ())
        if not queued:
            raise RuntimeError(
                "economic pool capture produced no served output")
        decode_token_count = len(queued[-1].outputs[0].token_ids)
    if capture_economic_pool:
        economic_material = cm.tr.economic_pool_material(
            request_id, decode_token_count=decode_token_count)
    if capture_execution_anchors:
        tokenizer = cm.miner.llm.get_tokenizer()
        context_token_count = len(tokenizer(prompt)["input_ids"])
        sequence_token_count = context_token_count + decode_token_count - 1
        if execution_anchor_reveal_rows is None:
            anchor_commitments = cm.tr.execution_anchor_commitments(
                request_id,
                expected_row_count=sequence_token_count,
            )
        else:
            anchor_commitments = cm.tr.execution_anchor_replay_material(
                request_id,
                expected_row_count=sequence_token_count,
            )
    cm.tr.unregister_request(request_id)
    if reduction_pool and capture_economic_pool and capture_execution_anchors:
        return (
            captured,
            queue,
            material,
            economic_material,
            anchor_commitments,
        )
    if reduction_pool and capture_economic_pool:
        return captured, queue, material, economic_material
    if capture_economic_pool and capture_execution_anchors:
        return captured, queue, economic_material, anchor_commitments
    if reduction_pool and capture_execution_anchors:
        return captured, queue, material, anchor_commitments
    if capture_execution_anchors:
        return captured, queue, anchor_commitments
    if reduction_pool:
        return captured, queue, material
    if capture_economic_pool:
        return captured, queue, economic_material
    return captured, queue


def serve_reduction_requests(
    cm: CaptureMiner,
    requests,
    *,
    reduction_pool: int = 32,
    reduction_row_positions_by_request=None,
):
    """Serve prompts CONCURRENTLY with reduction-audit capture.

    ``requests``: iterable of ``(request_id, prompt, max_tokens)``. All
    requests are added before stepping, so the scheduler interleaves them
    (concurrency + chunked prefill are exercised by construction when the
    token budget forces it). ``reduction_row_positions_by_request`` may
    provide exact absolute prompt rows for offline qualification; omitted
    requests retain the legacy prompt tail. Returns
    ``{request_id: (material, prompt_len)}`` from the REAL tracker's
    ``reduction_material()``.
    """

    from vllm import SamplingParams

    tokenizer = cm.miner.llm.get_tokenizer()
    prompt_lens = {}
    for request_id, prompt, _max_tokens in requests:
        prompt_lens[request_id] = len(tokenizer(prompt)["input_ids"])
        cm.tr.register_request(
            request_id, request_id,
            capture_logits=False,
            capture_reduction_audit=True,
            reduction_row_positions=(
                tuple(reduction_row_positions_by_request[request_id])
                if (
                    reduction_row_positions_by_request is not None
                    and request_id in reduction_row_positions_by_request
                )
                else None
            ),
            reduction_pool=reduction_pool,
            reduction_prompt_len=prompt_lens[request_id])
    for request_id, prompt, max_tokens in requests:
        cm.be.add_request(
            request_id, prompt,
            SamplingParams(temperature=0.0, max_tokens=max_tokens))
    while cm.be.has_active_requests():
        cm.be.step_and_distribute()
    out = {}
    for request_id, _prompt, _max_tokens in requests:
        material = cm.tr.reduction_material(request_id)
        cm.tr.unregister_request(request_id)
        out[request_id] = (material, prompt_lens[request_id])
    return out
