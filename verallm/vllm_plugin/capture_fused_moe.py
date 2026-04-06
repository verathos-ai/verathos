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

import torch

logger = logging.getLogger(__name__)


_LAYER_NAME_PATTERNS = (
    re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)"),
    re.compile(r"(?:^|\.)h\.(\d+)(?:\.|$)"),
)


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
