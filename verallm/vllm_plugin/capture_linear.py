"""
CaptureLinearWrapper — Post-load wrapper for dense gate_proj layers.

Wraps a linear module to insert a verallm::capture op on its input.
Used for dense (non-MoE) models where gate_proj is a standard linear
layer.  Applied selectively after model loading — only to gate_proj /
gate_up_proj modules, NOT to attention q/k/v/o projections.

For MoE models, CaptureFusedMoE (OOT) handles capture instead.
"""

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


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
                 use_triton_copy: bool = False):
        super().__init__()
        self.original = original
        self._layer_idx = layer_idx
        self._use_buffer = use_buffer
        self._use_triton_copy = use_triton_copy
        if use_buffer and max_tokens > 0 and hidden_dim > 0:
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

    def __getattr__(self, name: str):
        # nn.Module.__getattr__ checks _parameters, _buffers, _modules.
        # This fallback delegates to the wrapped module for everything
        # else (.weight, .qweight, .weight_packed, .scales, etc.).
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.original, name)

    def forward(self, x):
        if self._use_buffer and self._capture_buf is not None:
            n = min(x.shape[0], self._capture_buf.shape[0])
            if self._use_triton_copy:
                # Ampere (sm_80-86): static-grid Triton copy avoids CUDA
                # illegal memory access that aten::copy_ causes during
                # graph capture on these GPUs.
                torch.ops.verallm.buffer_copy(self._capture_buf, x, n)
            else:
                # Ada/Hopper/Blackwell: native aten::copy_ compiles into
                # the CUDA graph with zero graph split points.
                self._capture_buf[:n].copy_(x[:n])
        elif not self._use_buffer:
            # Split mode: custom op forces piecewise graph split.
            _ = torch.ops.verallm.capture(x, self._layer_idx, 0)
        return self.original(x)


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
