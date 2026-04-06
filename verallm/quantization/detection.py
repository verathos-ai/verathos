"""
Quantization detection and weight extraction for VeraLLM INT8/INT4 native mode.

This module provides:
- Automatic detection of model quantization type (bitsandbytes, GPTQ, AWQ, FP16)
- Integer weight extraction from quantized layers
- Utilities for INT8 native mode proof generation
"""

import logging
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Tuple, TYPE_CHECKING
import torch

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from transformers import PreTrainedModel


class QuantizationType(Enum):
    """Detected quantization type for a model."""
    FP16 = "fp16"                    # Standard FP16/FP32, needs scaling
    BITSANDBYTES_INT8 = "bnb_int8"   # bitsandbytes load_in_8bit
    BITSANDBYTES_INT4 = "bnb_int4"   # bitsandbytes load_in_4bit (NF4/FP4)
    GPTQ = "gptq"                    # GPTQ INT4 quantization
    AWQ = "awq"                      # AWQ INT4 quantization
    FP8 = "fp8"                      # FP8 (e4m3fn) — Ada+ / vLLM fp8 kernel
    NVFP4 = "nvfp4"                  # NVIDIA FP4 (modelopt) — Blackwell (sm_120+)


@dataclass
class QuantizationInfo:
    """Detected quantization configuration for a model."""
    quant_type: QuantizationType
    bits: int  # 4, 8, or 16

    @property
    def is_exact_integer(self) -> bool:
        """Whether inference produces exact integer results (no FP rounding)."""
        return self.quant_type == QuantizationType.BITSANDBYTES_INT8

    @property
    def requires_y_recomputation(self) -> bool:
        """Whether Y must be recomputed for proof generation."""
        # Only INT8 native mode can skip Y recomputation
        # INT4 modes have dequantization that introduces FP operations
        return not self.is_exact_integer

    @property
    def quant_mode(self) -> str:
        """Proof-system quant mode derived from detected quantization.

        Returns "int8", "int4", "fp8", or "fp16". This controls how weights
        are extracted from layers for Merkle tree computation and proof
        generation.
        """
        if self.quant_type == QuantizationType.BITSANDBYTES_INT8:
            return "int8"
        elif self.quant_type in (QuantizationType.BITSANDBYTES_INT4,
                                  QuantizationType.GPTQ,
                                  QuantizationType.AWQ):
            return "int4"
        elif self.quant_type == QuantizationType.FP8:
            return "fp8"
        elif self.quant_type == QuantizationType.NVFP4:
            return "nvfp4"
        else:
            return "fp16"



def detect_quantization(model: "PreTrainedModel") -> QuantizationInfo:
    """
    Detect quantization type from a loaded model.

    Args:
        model: A loaded HuggingFace model

    Returns:
        QuantizationInfo with detected type and configuration

    Checks for:
    - bitsandbytes: config.quantization_config.load_in_8bit/load_in_4bit
    - GPTQ: modules with 'gptq' in class name or qweight attribute
    - AWQ: modules with 'awq' in class name
    """
    # Check for bitsandbytes quantization via config
    if hasattr(model, 'config'):
        config = model.config
        qc = getattr(config, 'quantization_config', None)

        if qc is not None:
            # bitsandbytes INT8
            if getattr(qc, 'load_in_8bit', False):
                return QuantizationInfo(
                    quant_type=QuantizationType.BITSANDBYTES_INT8,
                    bits=8
                )
            # bitsandbytes INT4 (NF4/FP4)
            if getattr(qc, 'load_in_4bit', False):
                return QuantizationInfo(
                    quant_type=QuantizationType.BITSANDBYTES_INT4,
                    bits=4
                )
            # vLLM stores quant config as dict with 'quant_method' key
            if isinstance(qc, dict):
                qm = qc.get('quant_method', '')
                qa = qc.get('quant_algo', '').upper()
                if qm == 'awq':
                    return QuantizationInfo(quant_type=QuantizationType.AWQ, bits=4)
                if qm == 'gptq':
                    return QuantizationInfo(quant_type=QuantizationType.GPTQ, bits=4)
                if qm == 'fp8':
                    return QuantizationInfo(quant_type=QuantizationType.FP8, bits=8)
                # modelopt NVFP4: quant_method='modelopt', quant_algo='NVFP4'
                if qm == 'modelopt' and qa == 'NVFP4':
                    return QuantizationInfo(quant_type=QuantizationType.NVFP4, bits=4)

    # Check for GPTQ/AWQ quantization via module inspection
    for name, module in model.named_modules():
        module_type = type(module).__name__.lower()

        # AWQ detection (must come before GPTQ - AWQ also has qweight attribute)
        # autoawq: WQLinear_GEMM with w_bit attribute
        # gptqmodel: AwqExllamaV2QuantLinear / AwqGEMMQuantLinear with 'awq' in class name
        if hasattr(module, 'w_bit') or 'wqlinear' in module_type or (
            'awq' in module_type and hasattr(module, 'qweight')
        ):
            return QuantizationInfo(
                quant_type=QuantizationType.AWQ,
                bits=getattr(module, 'bits', getattr(module, 'w_bit', 4))
            )

        # GPTQ detection
        if 'gptq' in module_type or hasattr(module, 'qweight'):
            return QuantizationInfo(
                quant_type=QuantizationType.GPTQ,
                bits=getattr(module, 'bits', 4)
            )

    # Check for bitsandbytes Linear8bitLt modules directly
    for name, module in model.named_modules():
        module_type = type(module).__name__
        if 'Linear8bitLt' in module_type:
            return QuantizationInfo(
                quant_type=QuantizationType.BITSANDBYTES_INT8,
                bits=8
            )
        if 'Linear4bit' in module_type:
            return QuantizationInfo(
                quant_type=QuantizationType.BITSANDBYTES_INT4,
                bits=4
            )

    # Check for vLLM bitsandbytes (uses bnb_quant_state prefix, not native bnb module types)
    for name, module in model.named_modules():
        weight = getattr(module, 'weight', None)
        if weight is not None and hasattr(weight, 'bnb_quant_state'):
            if getattr(weight, 'use_bitsandbytes_4bit', False):
                return QuantizationInfo(
                    quant_type=QuantizationType.BITSANDBYTES_INT4,
                    bits=4
                )
            if getattr(weight, 'use_bitsandbytes_8bit', False):
                return QuantizationInfo(
                    quant_type=QuantizationType.BITSANDBYTES_INT8,
                    bits=8
                )

    # Check for NVFP4 (modelopt FP4) — must come before FP8 check since
    # NVFP4 layers also contain FP8 scale tensors.
    # modelopt_fp4: weight is packed uint8 (2 FP4 values per byte) with
    # a weight_scale (FP8 block scales) and optional weight_scale_2.
    for name, module in model.named_modules():
        # Standard dense NVFP4: module.weight is uint8 + module.weight_scale
        weight = getattr(module, 'weight', None)
        if weight is not None:
            w = getattr(weight, 'data', weight)
            if w.dtype == torch.uint8 and hasattr(module, 'weight_scale'):
                return QuantizationInfo(
                    quant_type=QuantizationType.NVFP4,
                    bits=4
                )
        # vLLM FusedMoE NVFP4: w13_weight is uint8 + w13_weight_scale
        w13 = getattr(module, 'w13_weight', None)
        if w13 is not None:
            d = getattr(w13, 'data', w13)
            if d.dtype == torch.uint8 and hasattr(module, 'w13_weight_scale'):
                return QuantizationInfo(
                    quant_type=QuantizationType.NVFP4,
                    bits=4
                )
        # Also check class name for explicit modelopt markers
        cls_name = type(module).__name__.lower()
        if ('fp4' in cls_name or 'modeloptfp' in cls_name) and hasattr(module, 'weight'):
            return QuantizationInfo(
                quant_type=QuantizationType.NVFP4,
                bits=4
            )

    # Check for FP8 quantization via weight dtype (vLLM fp8 kernel)
    _fp8_dtype = getattr(torch, 'float8_e4m3fn', None)
    if _fp8_dtype is not None:
        for name, module in model.named_modules():
            weight = getattr(module, 'weight', None)
            if weight is not None and hasattr(weight, 'dtype') and weight.dtype == _fp8_dtype:
                return QuantizationInfo(
                    quant_type=QuantizationType.FP8,
                    bits=8
                )

    # No quantization detected - assume FP16/FP32
    return QuantizationInfo(
        quant_type=QuantizationType.FP16,
        bits=16
    )


def detect_layer_quant_mode(layer_module) -> str:
    """Detect quant mode from a single linear layer's attributes.

    Returns "int8", "int4", "fp8", or "fp16" based on what's stored in the layer.
    No model-level config needed — inspects the layer directly.
    """
    if layer_module is None:
        return "fp16"
    # NOTE: Do NOT unwrap CaptureLinearWrapper here.  The wrapper's
    # __getattr__ delegates attribute lookups to .original, so all
    # hasattr() checks work transparently.  Unwrapping would SKIP
    # attributes patched onto the wrapper itself (e.g. qweight added
    # by _patch_compressed_tensors_weights after the load hook wrapped
    # gate_proj).
    # bitsandbytes INT8: weight has CB attribute
    if hasattr(layer_module, 'weight') and hasattr(layer_module.weight, 'CB'):
        return "int8"
    # Also check module type name for INT8
    if 'Linear8bitLt' in type(layer_module).__name__:
        return "int8"
    # vLLM bitsandbytes: check bnb_quant_state + use_bitsandbytes flags
    if hasattr(layer_module, 'weight') and hasattr(layer_module.weight, 'bnb_quant_state'):
        if getattr(layer_module.weight, 'use_bitsandbytes_4bit', False):
            return "int4"
        if getattr(layer_module.weight, 'use_bitsandbytes_8bit', False):
            return "int8"
    # AWQ INT4: autoawq (w_bit/wqlinear) or gptqmodel ('awq' in class name)
    if hasattr(layer_module, 'w_bit') or 'wqlinear' in type(layer_module).__name__.lower():
        return "int4"
    if 'awq' in type(layer_module).__name__.lower() and hasattr(layer_module, 'qweight'):
        return "int4"
    # GPTQ INT4: has qweight attribute
    if hasattr(layer_module, 'qweight'):
        return "int4"
    # bitsandbytes INT4 (NF4): weight has quant_state
    if hasattr(layer_module, 'weight') and hasattr(layer_module.weight, 'quant_state'):
        return "int4"
    # Also check module type name for INT4
    if 'Linear4bit' in type(layer_module).__name__:
        return "int4"
    # NVFP4 (modelopt FP4): packed uint8 weight + FP8 weight_scale
    # Must come before FP8 check since NVFP4 layers also have FP8 scale tensors.
    if hasattr(layer_module, 'weight') and hasattr(layer_module, 'weight_scale'):
        w = getattr(layer_module.weight, 'data', layer_module.weight)
        if w.dtype == torch.uint8:
            return "nvfp4"
    # Also check class name for explicit modelopt markers
    cls_name = type(layer_module).__name__.lower()
    if ('fp4' in cls_name or 'modeloptfp' in cls_name) and hasattr(layer_module, 'weight'):
        return "nvfp4"
    # FP8: weight stored as float8_e4m3fn (vLLM fp8 quantization)
    _fp8_dtype = getattr(torch, 'float8_e4m3fn', None)
    if _fp8_dtype is not None and hasattr(layer_module, 'weight'):
        w = getattr(layer_module.weight, 'data', layer_module.weight)
        if hasattr(w, 'dtype') and w.dtype == _fp8_dtype:
            return "fp8"
    return "fp16"


def extract_int8_weights_bnb(layer) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract INT8 weights and scales from a bitsandbytes Linear8bitLt layer.

    bitsandbytes INT8 stores:
    - CB: Compressed bytes (int8 weights)
    - SCB: Scale factors per row

    The computation is: Y = (X_int8 @ W_int8) * scale

    Returns:
        (W_int8, scales) where W_int8 is [out_features, in_features]
    """
    if hasattr(layer, 'weight'):
        weight = layer.weight

        # bitsandbytes Linear8bitLt stores int8 weights in CB
        if hasattr(weight, 'CB') and weight.CB is not None:
            w_int8 = weight.CB
            scales = weight.SCB if hasattr(weight, 'SCB') else None
            return w_int8, scales

        # Direct int8 tensor (includes vLLM bitsandbytes INT8)
        if weight.dtype == torch.int8:
            return weight.data if hasattr(weight, 'data') else weight, None

    raise ValueError(f"Cannot extract INT8 weights from {type(layer).__name__}")


def get_int8_weights_gpu(layer) -> torch.Tensor:
    """
    Get INT8 weights directly on GPU (NO DMA transfer).

    This is the key optimization for INT8 native mode:
    - bitsandbytes stores INT8 weights on GPU
    - We convert to int64 ON GPU (fast, no DMA)
    - Return tensor stays on GPU for direct use in cuda_int64_matmul

    Args:
        layer: A bitsandbytes Linear8bitLt layer

    Returns:
        W_int64 on GPU, shape [in_features, out_features] for X @ W layout
    """
    if not hasattr(layer, 'weight'):
        raise ValueError(f"Layer {type(layer).__name__} has no weight attribute")

    weight = layer.weight

    # bitsandbytes Linear8bitLt stores int8 weights in CB
    if hasattr(weight, 'CB') and weight.CB is not None:
        W_int8 = weight.CB  # Already on GPU!
        # Convert to int64 ON GPU (fast, no DMA)
        W_int64 = W_int8.to(torch.int64)
        # bitsandbytes stores [out, in], we need [in, out] for X @ W
        if W_int64.shape[0] != layer.in_features:
            W_int64 = W_int64.T.contiguous()
        return W_int64  # Still on GPU!

    # Direct int8 tensor (less common)
    if weight.dtype == torch.int8:
        W_int64 = weight.to(torch.int64)
        if W_int64.shape[0] != getattr(layer, 'in_features', W_int64.shape[1]):
            W_int64 = W_int64.T.contiguous()
        return W_int64

    raise ValueError(f"Cannot get INT8 weights from {type(layer).__name__} - "
                     f"no CB attribute and dtype is {weight.dtype}")


def get_int8_weights_raw_gpu(layer) -> torch.Tensor:
    """
    Get INT8 weights directly from bitsandbytes - NO TYPE CONVERSION.

    This is the key optimization for true INT8 native mode:
    - bitsandbytes stores INT8 weights on GPU in weight.CB
    - We return them AS-IS (int8, not int64)
    - Zero memory overhead, zero conversion cost
    - Use with cuda_int8_matmul kernel for fastest performance

    Args:
        layer: A bitsandbytes Linear8bitLt layer

    Returns:
        W_int8 on GPU, shape [in_features, out_features], dtype=torch.int8
    """
    if not hasattr(layer, 'weight'):
        raise ValueError(f"Layer {type(layer).__name__} has no weight attribute")

    weight = layer.weight

    # bitsandbytes Linear8bitLt stores int8 weights in CB
    if hasattr(weight, 'CB') and weight.CB is not None:
        W_int8 = weight.CB  # Already int8 on GPU!
        # bitsandbytes stores [out, in], we need [in, out] for X @ W
        if W_int8.shape[0] != layer.in_features:
            W_int8 = W_int8.T.contiguous()
        return W_int8  # Still int8, still on GPU!

    # Direct int8 tensor (includes vLLM bitsandbytes INT8 which stores int8 in weight.data)
    if weight.dtype == torch.int8:
        W_int8 = weight.data if hasattr(weight, 'data') else weight
        in_feat = getattr(layer, 'in_features', getattr(layer, 'input_size', W_int8.shape[1]))
        if W_int8.shape[0] != in_feat:
            W_int8 = W_int8.T.contiguous()
        return W_int8

    raise ValueError(f"Layer has no INT8 weights (dtype={weight.dtype})")


# ============================================================================
# INT4 (GPTQ/AWQ) Weight Extraction
# ============================================================================

def extract_gptq_weights(layer) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Extract GPTQ weight components from a quantized layer.

    GPTQ stores weights as packed INT4 values in int32 tensors:
    - qweight: Packed INT4 weights, 8 values per int32. Shape [K/8, N]
    - scales: FP16 scaling factors per group. Shape [K/group_size, N]
    - qzeros: Packed zero-point values (optional, None for symmetric quant)
    - g_idx: Group index mapping (optional, for non-sequential groups)

    Args:
        layer: A GPTQ-quantized linear layer

    Returns:
        (qweight, scales, qzeros, g_idx) tuple
    """
    if not hasattr(layer, 'qweight') and not hasattr(layer, '_orig_qweight'):
        raise ValueError(f"Layer {type(layer).__name__} is not a GPTQ layer (no qweight)")

    # Prefer _orig_qweight (original GPTQ layout) over qweight (may be
    # Marlin-repacked by gptq_marlin and incompatible with standard unpacking).
    _oq = getattr(layer, '_orig_qweight', None)
    qweight = _oq if _oq is not None else layer.qweight
    _os = getattr(layer, '_orig_scales', None)
    scales = _os if _os is not None else layer.scales
    qzeros = getattr(layer, '_orig_qzeros', getattr(layer, 'qzeros', None))
    g_idx = getattr(layer, '_orig_g_idx', getattr(layer, 'g_idx', None))

    return qweight, scales, qzeros, g_idx


def unpack_int4_slice(qweight_slice: torch.Tensor) -> torch.Tensor:
    """
    Unpack a slice of INT4 values from packed int32 representation.

    GPTQ packs 8 INT4 values into each int32 using bit shifts:
    - Bit layout: [v7|v6|v5|v4|v3|v2|v1|v0] where each v is 4 bits
    - v0 is in bits 0-3, v1 is in bits 4-7, etc.

    This function operates on slices to minimize memory usage during
    proof generation. Works on GPU for fast vectorized operations.

    Args:
        qweight_slice: Packed slice [K/8, cols] as int32 on GPU

    Returns:
        Unpacked slice [K, cols] as int8 on GPU
    """
    K_packed, cols = qweight_slice.shape
    K = K_packed * 8
    device = qweight_slice.device

    # Vectorized unpack using bit shifts (fast on GPU)
    # Extract each of the 8 INT4 values
    unpacked = torch.stack([
        ((qweight_slice >> (i * 4)) & 0xF).to(torch.int8)
        for i in range(8)
    ], dim=0)  # [8, K/8, cols]

    # Reshape to interleaved [K, cols]
    # The original order is: v0 at row 0, v1 at row 1, etc.
    W_int8 = unpacked.permute(1, 0, 2).reshape(K, cols)

    # Convert unsigned [0,15] to signed [-8,7] if model uses signed representation
    # Most GPTQ models use unsigned [0,15], but some use signed
    # We keep as unsigned since Merkle tree just needs consistent representation
    return W_int8


def unpack_qzeros(qzeros: torch.Tensor) -> torch.Tensor:
    """
    Unpack zero-point values from packed int32 representation.

    Same packing as qweight: 8 values per int32.

    Args:
        qzeros: Packed zeros [num_groups, N/8] as int32

    Returns:
        Unpacked zeros [num_groups, N] as int8
    """
    num_groups, N_packed = qzeros.shape
    N = N_packed * 8
    device = qzeros.device

    unpacked = torch.stack([
        ((qzeros >> (i * 4)) & 0xF).to(torch.int8)
        for i in range(8)
    ], dim=0)  # [8, num_groups, N/8]

    # Reshape to [num_groups, N]
    zeros_int8 = unpacked.permute(1, 0, 2).reshape(num_groups, N)
    return zeros_int8


def dequant_int4_slice(
    w_int8: torch.Tensor,
    scales: torch.Tensor,
    qzeros: Optional[torch.Tensor],
    col_start: int,
    col_end: int,
    group_size: int = 128
) -> torch.Tensor:
    """
    Dequantize an INT4 slice to FP16.

    Formula: W_fp16 = (w_int4 - zeros) * scales

    Operates on GPU slice to minimize memory and maximize speed.
    Used for sumcheck computation (requires actual weight values).

    Args:
        w_int8: Unpacked INT4 weights as int8 [K, cols]
        scales: FP16 scale factors [K/group_size, N_full]
        qzeros: Packed zero points [K/group_size, N_full/8] or None
        col_start: Start column in full W matrix
        col_end: End column in full W matrix
        group_size: Group size for quantization (default 128)

    Returns:
        Dequantized weights as FP16 [K, cols] on same device
    """
    K, cols = w_int8.shape
    num_groups = K // group_size
    device = w_int8.device

    # Get scales for this column slice
    scales_slice = scales[:, col_start:col_end].to(device)  # [num_groups, cols]

    # Handle zeros
    if qzeros is not None:
        # Unpack zeros and get slice
        zeros_unpacked = unpack_qzeros(qzeros.to(device))  # [num_groups, N_full]
        zeros_slice = zeros_unpacked[:, col_start:col_end]  # [num_groups, cols]
    else:
        # Symmetric quantization: zero point = 8 (middle of [0,15])
        zeros_slice = torch.full((num_groups, cols), 8, dtype=torch.int8, device=device)

    # Expand scales and zeros to match weight dimensions
    # Each group covers group_size rows
    scales_expanded = scales_slice.repeat_interleave(group_size, dim=0)[:K]  # [K, cols]
    zeros_expanded = zeros_slice.repeat_interleave(group_size, dim=0)[:K]    # [K, cols]

    # Dequantize: W_fp16 = (w_int8 - zeros) * scales
    W_fp16 = (w_int8.float() - zeros_expanded.float()) * scales_expanded.float()
    return W_fp16.half()


def get_int4_weights_as_int8_gpu(layer, *, force_cpu: bool = False) -> torch.Tensor:
    """
    Get full INT4 weights as unpacked int8 tensor on GPU (or CPU if force_cpu).

    Used for Merkle tree building and proof generation.
    If qweight is on CPU (compressed-tensors models), transfers the packed
    tensor (K/8 × N, ~80MB) to GPU first so unpacking runs via fast GPU
    bit-shift ops.  This avoids slow CPU unpacking of the full K × N tensor
    (~160MB) and eliminates a large CPU→GPU transfer.

    When force_cpu=True, unpacking runs entirely on CPU (~10x slower but
    zero GPU memory).  Used by pre-warming when VRAM is too tight.

    Args:
        layer: A GPTQ-quantized linear layer
        force_cpu: If True, keep tensors on CPU for unpacking.

    Returns:
        W_int8, shape [in_features, out_features], dtype=torch.int8
    """
    qweight, scales, qzeros, g_idx = extract_gptq_weights(layer)

    # Fast path for compressed-tensors: qweight lives on CPU to save VRAM.
    # Move the *packed* tensor to GPU first (8× smaller than unpacked) so
    # vectorised bit-shift unpacking runs on GPU instead of CPU.
    if not force_cpu and not qweight.is_cuda and torch.cuda.is_available():
        qweight = qweight.cuda()

    # Unpack all weights
    W_int8 = unpack_int4_slice(qweight)  # [K, N] where K = in_features

    # Handle layout: GPTQ stores [K/8, N] packed, which unpacks to [K, N]
    # This should already be [in_features, out_features] for X @ W layout
    return W_int8


def get_int4_slice_for_proof(
    layer,
    col_start: int,
    col_end: int,
    group_size: int = 128
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get a column slice of INT4 weights for proof generation.

    This is the key function for Phase 6 (proof generation) - it:
    1. Slices the packed weights on GPU (no extra memory for full W)
    2. Unpacks only that slice on GPU (~0.1ms)
    3. Dequantizes only that slice on GPU (~0.1ms)
    4. Returns both int8 (for spot checks) and fp16 (for sumcheck)

    Args:
        layer: A GPTQ-quantized linear layer
        col_start: Start column index
        col_end: End column index
        group_size: Group size for dequantization

    Returns:
        (W_int8_slice, W_fp16_slice) both on GPU, shape [K, cols]
    """
    qweight, scales, qzeros, g_idx = extract_gptq_weights(layer)

    # Slice packed weights on GPU
    qweight_slice = qweight[:, col_start:col_end].contiguous()

    # Unpack slice
    W_int8_slice = unpack_int4_slice(qweight_slice)

    # Dequantize slice
    W_fp16_slice = dequant_int4_slice(
        W_int8_slice, scales, qzeros,
        col_start, col_end, group_size
    )

    return W_int8_slice, W_fp16_slice


# ============================================================================
# AWQ INT4 Weight Extraction
# ============================================================================

# AWQ packs 8 INT4 values per int32 in a non-sequential order.
# The packing order is [0, 2, 4, 6, 1, 3, 5, 7] — to reverse it we
# reorder with AWQ_REVERSE_ORDER after unpacking.
AWQ_REVERSE_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]


def is_awq_layer(layer) -> bool:
    """Check if a linear layer uses AWQ quantization (vs GPTQ).

    AWQ layers (autoawq WQLinear_GEMM) have a `w_bit` attribute,
    while GPTQ layers have `bits`. Both have `qweight`.
    AWQ packs columnwise: qweight [K, N/8] + scales [groups, N] → scales.shape[1] = 8*qweight.shape[1].
    GPTQ packs rowwise: qweight [K/8, N] + scales [groups, N] → scales.shape[1] = qweight.shape[1].
    """
    if hasattr(layer, 'w_bit'):
        return True
    if 'wqlinear' in type(layer).__name__.lower():
        return True
    if 'awq' in type(layer).__name__.lower():
        return True
    # Distinguish AWQ vs GPTQ by comparing scales and qweight shapes
    # AWQ columnwise: scales has full N, qweight has N/8
    if hasattr(layer, 'qweight') and hasattr(layer, 'scales'):
        qw_cols = layer.qweight.shape[1]
        scales_cols = layer.scales.shape[1]
        if scales_cols == qw_cols * 8:
            return True
    # After gptq_marlin repacking, the live qweight/scales shapes change.
    # Check original checkpoint shapes stored by _patch_gptq_marlin_weights().
    _oq = getattr(layer, '_orig_qweight', None)
    _os = getattr(layer, '_orig_scales', None)
    if _oq is not None and _os is not None:
        if _os.shape[1] == _oq.shape[1] * 8:
            return True
    return False


def unpack_awq_int4(qweight: torch.Tensor) -> torch.Tensor:
    """
    Unpack AWQ INT4 values from packed int32 representation.

    AWQ packs 8 INT4 values per int32 **columnwise** with a shuffled order:
    - qweight shape: [in_features, out_features // 8]
    - Packing order: [0, 2, 4, 6, 1, 3, 5, 7] (AWQ_ORDER)

    After unpacking and reordering, returns [in_features, out_features] int8.

    Args:
        qweight: Packed AWQ weights [K, N/8] as int32 on GPU

    Returns:
        Unpacked weights [K, N] as int8 on GPU, values in [0, 15]
    """
    K, N_packed = qweight.shape
    N = N_packed * 8
    device = qweight.device

    # Unpack columnwise: shift each packed int32 by [0, 4, 8, ..., 28] bits
    shifts = torch.arange(0, 32, 4, device=device)
    unpacked = (qweight[:, :, None] >> shifts[None, None, :]).to(torch.int8)
    # Shape: [K, N/8, 8]
    unpacked = unpacked.view(K, N)  # [K, N] but in AWQ shuffled order

    # Reverse AWQ order: reorder every group of 8 columns
    reverse_order = torch.arange(N, dtype=torch.int32, device=device)
    reverse_order = reverse_order.view(-1, 8)[:, AWQ_REVERSE_ORDER].view(-1)
    unpacked = unpacked[:, reverse_order]

    # Mask to clean 4-bit values [0, 15]
    unpacked = unpacked & 0xF

    return unpacked


def unpack_awq_qzeros(qzeros: torch.Tensor) -> torch.Tensor:
    """
    Unpack AWQ zero-point values from packed int32 representation.

    Same AWQ packing scheme as qweight (columnwise with shuffled order).

    Args:
        qzeros: Packed zeros [num_groups, N/8] as int32

    Returns:
        Unpacked zeros [num_groups, N] as int8, values in [0, 15]
    """
    num_groups, N_packed = qzeros.shape
    N = N_packed * 8
    device = qzeros.device

    shifts = torch.arange(0, 32, 4, device=device)
    unpacked = (qzeros[:, :, None] >> shifts[None, None, :]).to(torch.int8)
    unpacked = unpacked.view(num_groups, N)

    reverse_order = torch.arange(N, dtype=torch.int32, device=device)
    reverse_order = reverse_order.view(-1, 8)[:, AWQ_REVERSE_ORDER].view(-1)
    unpacked = unpacked[:, reverse_order]

    unpacked = unpacked & 0xF

    return unpacked


def get_awq_weights_as_int8_gpu(layer, *, force_cpu: bool = False) -> torch.Tensor:
    """
    Get full AWQ INT4 weights as unpacked int8 tensor on GPU (or CPU if force_cpu).

    Used for Merkle tree building (one-time during model registration).
    AWQ qweight is [in_features, out_features // 8], so after unpacking
    the result is already [in_features, out_features] — no transpose needed.

    When force_cpu=True, unpacking runs entirely on CPU (~10x slower but
    zero GPU memory).  Used by pre-warming when VRAM is too tight.

    Args:
        layer: An AWQ-quantized linear layer (WQLinear_GEMM)
        force_cpu: If True, keep tensors on CPU for unpacking.

    Returns:
        W_int8, shape [in_features, out_features], dtype=torch.int8
    """
    if not hasattr(layer, 'qweight') and not hasattr(layer, '_orig_qweight'):
        raise ValueError(f"Layer {type(layer).__name__} has no qweight (not AWQ)")

    # Prefer _orig_qweight (original AWQ layout) over qweight (may be
    # Marlin-repacked and incompatible with standard AWQ unpacking).
    _oq = getattr(layer, '_orig_qweight', None)
    qw = _oq if _oq is not None else layer.qweight

    # _orig_qweight is stored on CPU to save VRAM.  Move the *packed* tensor
    # to GPU first (8x smaller than unpacked) so bit-shift unpacking runs on
    # GPU instead of CPU — same pattern as get_int4_weights_as_int8_gpu().
    if not force_cpu and not qw.is_cuda and torch.cuda.is_available():
        qw = qw.cuda()

    W_int8 = unpack_awq_int4(qw)  # [K, N] = [in_features, out_features]
    return W_int8


def dequant_awq_int4_slice(
    w_int8: torch.Tensor,
    scales: torch.Tensor,
    qzeros: Optional[torch.Tensor],
    col_start: int,
    col_end: int,
    group_size: int = 128
) -> torch.Tensor:
    """
    Dequantize an AWQ INT4 slice to FP16.

    Formula: W_fp16 = (w_int4 - zeros) * scales

    Args:
        w_int8: Unpacked INT4 weights as int8 [K, cols]
        scales: FP16 scale factors [K/group_size, N_full]
        qzeros: Packed AWQ zero points [K/group_size, N_full/8] or None
        col_start: Start column in full W matrix
        col_end: End column in full W matrix
        group_size: Group size for quantization (default 128)

    Returns:
        Dequantized weights as FP16 [K, cols] on same device
    """
    K, cols = w_int8.shape
    device = w_int8.device

    # Get scales for this column slice
    scales_slice = scales[:, col_start:col_end].to(device)

    # Handle zeros
    if qzeros is not None:
        zeros_unpacked = unpack_awq_qzeros(qzeros.to(device))
        zeros_slice = zeros_unpacked[:, col_start:col_end]
    else:
        zeros_slice = torch.full(
            (scales_slice.shape[0], cols), 8, dtype=torch.int8, device=device
        )

    # Expand to match weight dimensions
    scales_expanded = scales_slice.repeat_interleave(group_size, dim=0)[:K]
    zeros_expanded = zeros_slice.repeat_interleave(group_size, dim=0)[:K]

    W_fp16 = (w_int8.float() - zeros_expanded.float()) * scales_expanded.float()
    return W_fp16.half()



# ============================================================================
# bitsandbytes INT4 (NF4/FP4) Weight Extraction
# ============================================================================

def extract_bnb_int4_weights(layer) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, Tuple[int, int]]:
    """
    Extract bitsandbytes INT4 (NF4/FP4) weight components from a Linear4bit layer.

    bitsandbytes NF4 stores weights as:
    - weight.data: Packed uint8 tensor (2 x 4-bit values per byte)
    - weight.quant_state.code: 16-value lookup table (FP32)
    - weight.quant_state.absmax: Per-block scale factors (FP32)
    - weight.quant_state.blocksize: Number of elements per block (typically 64)
    - weight.quant_state.shape: Original weight shape

    Args:
        layer: A bitsandbytes Linear4bit layer

    Returns:
        (packed_data, code, absmax, blocksize, original_shape) tuple
    """
    if not hasattr(layer, 'weight'):
        raise ValueError(f"Layer {type(layer).__name__} has no weight attribute")

    weight = layer.weight

    # Get quant_state from either transformers or vLLM naming convention
    quant_state = getattr(weight, 'quant_state', None)
    if quant_state is None:
        # vLLM uses bnb_quant_state (dict indexed by shard ID)
        bnb_qs = getattr(weight, 'bnb_quant_state', None)
        if bnb_qs is not None:
            if isinstance(bnb_qs, dict):
                # Single-GPU: use shard 0
                quant_state = bnb_qs[0] if 0 in bnb_qs else next(iter(bnb_qs.values()))
            else:
                quant_state = bnb_qs

    if quant_state is None:
        raise ValueError(f"Layer weight has no quant_state or bnb_quant_state - "
                         f"not a bitsandbytes INT4 layer (type: {type(layer).__name__})")

    # Handle nested quant_state (some versions wrap it)
    if hasattr(quant_state, 'quant_state'):
        quant_state = quant_state.quant_state

    # Extract components
    packed_data = weight.data  # uint8 tensor [N_packed]
    code = quant_state.code  # FP32 [16] - NF4 lookup table
    absmax = quant_state.absmax  # FP32 [num_blocks] - per-block scales
    blocksize = quant_state.blocksize  # int, typically 64
    original_shape = quant_state.shape  # (out_features, in_features) typically

    return packed_data, code, absmax, blocksize, original_shape


def unpack_bnb_int4(packed_data: torch.Tensor, num_elements: int) -> torch.Tensor:
    """
    Unpack bitsandbytes INT4 values from packed uint8 representation.

    bitsandbytes packs 2 INT4 values per uint8 byte:
    - Lower 4 bits: first value (& 0x0F)
    - Upper 4 bits: second value (>> 4)

    Args:
        packed_data: Packed uint8 tensor [num_elements / 2]
        num_elements: Total number of 4-bit values to unpack

    Returns:
        Unpacked tensor [num_elements] as int8 (values 0-15)
    """
    device = packed_data.device

    # Ensure packed_data is uint8
    packed = packed_data.view(torch.uint8)

    # Extract lower and upper nibbles
    lower = (packed & 0x0F).to(torch.int8)  # Lower 4 bits
    upper = (packed >> 4).to(torch.int8)    # Upper 4 bits

    # Interleave: [lo0, hi0, lo1, hi1, ...]
    unpacked = torch.stack([lower, upper], dim=-1).view(-1)[:num_elements]

    return unpacked


def dequant_bnb_int4(
    w_int8: torch.Tensor,
    code: torch.Tensor,
    absmax: torch.Tensor,
    blocksize: int,
    original_shape: Tuple[int, int]
) -> torch.Tensor:
    """
    Dequantize bitsandbytes INT4 (NF4) weights to FP16.

    NF4 dequantization:
    1. Use 4-bit index (0-15) to lookup value from code table
    2. Multiply by per-block absmax scale

    Formula: W_fp16[i] = code[w_int8[i]] * absmax[i // blocksize]

    Args:
        w_int8: Unpacked INT4 indices as int8 [N]
        code: NF4 lookup table FP32 [16]
        absmax: Per-block scale factors FP32 [num_blocks]
        blocksize: Elements per block (typically 64)
        original_shape: (out_features, in_features)

    Returns:
        Dequantized weights as FP16 [out_features, in_features]
    """
    device = w_int8.device
    out_features, in_features = original_shape
    num_elements = out_features * in_features

    # Lookup values from code table
    code_gpu = code.to(device)
    w_lookup = code_gpu[w_int8.long()]  # [N] FP32

    # Apply per-block absmax scaling
    absmax_gpu = absmax.to(device)
    num_blocks = (num_elements + blocksize - 1) // blocksize

    # Expand absmax to match weight dimensions
    # Each block of `blocksize` elements shares the same absmax
    absmax_expanded = absmax_gpu.repeat_interleave(blocksize)[:num_elements]

    # Scale and reshape
    W_fp16 = (w_lookup * absmax_expanded).view(out_features, in_features).half()

    return W_fp16


def get_bnb_int4_weights_as_int8_gpu(layer) -> torch.Tensor:
    """
    Get full bitsandbytes INT4 weights as unpacked int8 tensor on GPU.

    Used for Merkle tree building (one-time during model registration).
    The unpacked tensor contains the 4-bit indices (0-15) as int8 values.

    For verification, we commit to these discrete quantization indices,
    not the dequantized FP16 values. This ensures deterministic verification.

    Args:
        layer: A bitsandbytes Linear4bit layer

    Returns:
        W_int8 on GPU, shape [in_features, out_features], dtype=torch.int8
    """
    packed_data, code, absmax, blocksize, original_shape = extract_bnb_int4_weights(layer)

    out_features, in_features = original_shape
    num_elements = out_features * in_features

    # Unpack all weights
    W_int8_flat = unpack_bnb_int4(packed_data, num_elements)  # [N]

    # Reshape to [out_features, in_features]
    W_int8 = W_int8_flat.view(out_features, in_features)

    # Transpose to [in_features, out_features] for X @ W layout
    W_int8 = W_int8.T.contiguous()

    return W_int8


# ---------------------------------------------------------------------------
# FP8 (float8_e4m3fn) support
# ---------------------------------------------------------------------------

def is_fp8_layer(layer) -> bool:
    """Check if a linear layer uses FP8 quantization.

    FP8 layers in vLLM store weights as ``torch.float8_e4m3fn`` with a
    ``weight_scale`` attribute.  After ``process_weights_after_loading()``,
    the weight is transposed to ``[in_features, out_features]``.
    """
    _fp8_dtype = getattr(torch, 'float8_e4m3fn', None)
    if _fp8_dtype is None:
        return False
    weight = getattr(layer, 'weight', None)
    if weight is None:
        return False
    w = getattr(weight, 'data', weight)
    return hasattr(w, 'dtype') and w.dtype == _fp8_dtype


def get_fp8_weights_as_int8_gpu(layer) -> torch.Tensor:
    """Extract FP8 weights as int8 tensor via raw byte reinterpretation.

    FP8 (float8_e4m3fn) uses 1 byte per element, same as int8.  We reinterpret
    the raw bytes — no dequantization or scaling needed.  This is deterministic,
    lossless, and extremely fast (just a dtype view).

    After vLLM's ``process_weights_after_loading()``, the weight is already
    transposed to ``[in_features, out_features]`` — the correct layout for
    ``X @ W``.

    Args:
        layer: A vLLM FP8-quantized linear layer.

    Returns:
        W_int8 on GPU, shape [in_features, out_features], dtype=torch.int8
    """
    weight = getattr(layer, 'weight', None)
    if weight is None:
        raise ValueError(f"Layer {type(layer).__name__} has no weight attribute")
    w = weight.data
    _fp8_dtype = getattr(torch, 'float8_e4m3fn', None)
    if w.dtype != _fp8_dtype:
        raise ValueError(
            f"Expected float8_e4m3fn weight, got {w.dtype} "
            f"on {type(layer).__name__}"
        )
    # Reinterpret FP8 bytes as int8 — both are 1 byte per element.
    return w.view(torch.int8)


# ---------------------------------------------------------------------------
# NVFP4 (NVIDIA modelopt FP4) support
# ---------------------------------------------------------------------------

def is_nvfp4_layer(layer) -> bool:
    """Check if a linear layer uses NVFP4 (modelopt FP4) quantization.

    NVFP4 layers (via nvidia-modelopt / vLLM ``--quantization modelopt_fp4``)
    store weights as packed uint8 tensors (2 FP4 values per byte) with
    per-block FP8 scales in ``weight_scale``.

    Detection heuristics (any match → True):
    - weight.dtype == uint8 AND module has ``weight_scale`` attribute
    - Module class name contains 'fp4' or 'modeloptfp'
    """
    weight = getattr(layer, 'weight', None)
    if weight is None:
        return False
    w = getattr(weight, 'data', weight)
    # Primary: packed uint8 weights + FP8 block scale tensor
    if w.dtype == torch.uint8 and hasattr(layer, 'weight_scale'):
        return True
    # Secondary: class name hint from vLLM / modelopt
    cls_name = type(layer).__name__.lower()
    if ('fp4' in cls_name or 'modeloptfp' in cls_name):
        return True
    return False


def get_nvfp4_weights_as_int8_gpu(layer) -> torch.Tensor:
    """Extract NVFP4 weights as int8 tensor via raw byte reinterpretation.

    NVFP4 (nvidia-modelopt FP4) packs 2 FP4 values per uint8 byte.  Like
    FP8 extraction, we reinterpret the raw bytes as int8 — no dequantization
    or unpacking needed.  This is deterministic, lossless w.r.t. the
    quantized representation, and extremely fast (just a dtype view).

    The packed tensor's shape depends on the vLLM version and layer type:

    - Dense linear: ``[out_features, in_features // 2]`` (after vLLM
      ``process_weights_after_loading``).
    - 3D batched MoE experts: ``[num_experts, dim1, dim2 // 2]``.

    We preserve the shape as-is — the Merkle tree builder flattens before
    hashing, so the layout doesn't matter as long as it's deterministic
    across miner and verifier.

    .. note::

       This function commits to the **packed FP4 bytes**, not unpacked
       values.  The per-block FP8 scales (``weight_scale``) are NOT included
       in the commitment — they are deterministic given the checkpoint and
       are verified separately via the model identity hash.

       NVFP4 requires Blackwell (sm_120+) and vLLM >= 0.15.1 with
       ``--quantization modelopt_fp4``.  If running on pre-Blackwell
       hardware, this function will still work for hashing — it only
       requires the tensor bytes, not the FP4 compute kernels.

    Args:
        layer: A vLLM NVFP4-quantized linear layer.

    Returns:
        W_int8 on GPU, dtype=torch.int8.  Shape matches the packed layout
        (each int8 element encodes 2 FP4 values).
    """
    weight = getattr(layer, 'weight', None)
    if weight is None:
        raise ValueError(f"Layer {type(layer).__name__} has no weight attribute")
    w = weight.data
    if w.dtype != torch.uint8:
        raise ValueError(
            f"Expected uint8 packed NVFP4 weight, got {w.dtype} "
            f"on {type(layer).__name__}"
        )
    # Reinterpret packed FP4 bytes as int8 — both are 1 byte per element.
    # The packed representation is the canonical form we commit to.
    return w.view(torch.int8)


def extract_int_weights(layer, quant_info: QuantizationInfo,
                        scale_factor: int = 100) -> torch.Tensor:
    """
    Extract integer weights from a layer for proof generation.

    For INT8 native mode: Returns raw INT8 weights as int64
    For FP16 mode: Returns scaled integer weights

    Args:
        layer: A linear layer (nn.Linear, Linear8bitLt, etc.)
        quant_info: Detected quantization info
        scale_factor: Scale factor for FP16 -> int64 conversion

    Returns:
        W_int as torch.int64 tensor [in_features, out_features] (transposed for matmul)
    """
    if quant_info.quant_type == QuantizationType.BITSANDBYTES_INT8:
        # Extract raw INT8 weights
        w_int8, _ = extract_int8_weights_bnb(layer)
        # Convert to int64 and transpose for X @ W.T layout
        W_int = w_int8.to(torch.int64).cpu()
        # bitsandbytes stores [out, in], we need [in, out] for X @ W
        if W_int.shape[0] != layer.in_features:
            W_int = W_int.T.contiguous()
        return W_int

    elif quant_info.quant_type == QuantizationType.FP16:
        # FP16 - scale and convert to int64
        W_float = layer.weight.data.float().cpu()
        # Handle [out, in] vs [in, out] layout
        if W_float.shape[0] != getattr(layer, 'in_features', W_float.shape[1]):
            W_float = W_float.T.contiguous()
        W_int = (W_float * scale_factor).to(torch.int64)
        return W_int

    elif quant_info.quant_type == QuantizationType.GPTQ:
        # GPTQ INT4 - return unpacked int8 values for Merkle tree
        # These are the discrete quantization levels [0-15] stored as int8
        W_int8 = get_int4_weights_as_int8_gpu(layer)
        # Return as int64 for consistency, but values are in [0,15] range
        return W_int8.to(torch.int64).cpu()

    elif quant_info.quant_type == QuantizationType.BITSANDBYTES_INT4:
        # bitsandbytes INT4 (NF4/FP4) - return unpacked int8 values for Merkle tree
        # These are the discrete quantization indices [0-15] stored as int8
        W_int8 = get_bnb_int4_weights_as_int8_gpu(layer)
        # Return as int64 for consistency, but values are in [0,15] range
        return W_int8.to(torch.int64).cpu()

    elif quant_info.quant_type == QuantizationType.AWQ:
        # AWQ INT4 - two possible formats:
        # autoawq (WQLinear_GEMM): columnwise [K, N/8], AWQ-shuffled order
        # gptqmodel (AwqExllamaV2QuantLinear): rowwise [K/8, N], repacked as GPTQ
        if is_awq_layer(layer):
            # autoawq format - use AWQ-specific unpacking
            W_int8 = get_awq_weights_as_int8_gpu(layer)
        else:
            # gptqmodel repacked format - use GPTQ unpacking
            W_int8 = get_int4_weights_as_int8_gpu(layer)
        return W_int8.to(torch.int64).cpu()

    elif quant_info.quant_type == QuantizationType.NVFP4:
        # NVFP4 (modelopt FP4) - raw byte reinterpretation of packed uint8
        # Each byte holds 2 FP4 values; we commit to the packed bytes.
        W_int8 = get_nvfp4_weights_as_int8_gpu(layer)
        return W_int8.to(torch.int64).cpu()

    else:
        # Unknown quantization - fall back to FP16 scaled mode
        W_float = layer.weight.data.float().cpu()
        if W_float.shape[0] != getattr(layer, 'in_features', W_float.shape[1]):
            W_float = W_float.T.contiguous()
        W_int = (W_float * scale_factor).to(torch.int64)
        return W_int


def get_int8_matmul_result(layer, X: torch.Tensor, debug: bool = False) -> Optional[torch.Tensor]:
    """
    Get the integer matmul result from a bitsandbytes INT8 layer.

    This hooks into the internal computation to get Y_int = X_int8 @ W_int8
    before scaling is applied.

    Args:
        layer: A bitsandbytes Linear8bitLt layer
        X: Input activation tensor
        debug: Print debug info on failure

    Returns:
        Y_int as torch.int64, or None if not available
    """
    try:
        # For bitsandbytes, we need to intercept the integer matmul
        # This is complex because bitsandbytes fuses quant + matmul + dequant

        # Simpler approach: manually compute X_int @ W_int using extracted weights
        w_int8, _ = extract_int8_weights_bnb(layer)

        # Move everything to CPU for int64 matmul (CUDA doesn't support int64 matmul)
        X_cpu = X.detach().float().cpu()
        w_int8_cpu = w_int8.cpu()

        # Quantize input to INT8 (matching bitsandbytes absmax quantization)
        absmax = X_cpu.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-8)
        scale = absmax / 127.0
        X_int8 = (X_cpu / scale).round().clamp(-128, 127).to(torch.int8)

        # Compute integer matmul on CPU
        # X_int8: [batch, seq, in] @ W_int8: [out, in].T = [batch, seq, out]
        Y_int = torch.matmul(X_int8.to(torch.int64), w_int8_cpu.to(torch.int64).T)

        return Y_int

    except Exception as e:
        if debug:
            logger.debug("get_int8_matmul_result failed: %s", e)
            logger.debug("layer type: %s", type(layer).__name__)
            if hasattr(layer, 'weight'):
                w = layer.weight
                logger.debug("weight type: %s, dtype: %s", type(w).__name__, getattr(w, 'dtype', 'N/A'))
                logger.debug("has CB: %s, has SCB: %s", hasattr(w, 'CB'), hasattr(w, 'SCB'))
        return None
