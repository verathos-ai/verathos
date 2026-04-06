"""
INT8 quantization for deterministic computation.

Symmetric quantization: x_int8 = round(x / scale)
Dequantization: x_fp = x_int8 * scale

For GEMM: Y_fp = (X_int8 @ W_int8) * scale_x * scale_w
"""

from typing import Tuple
from dataclasses import dataclass

import torch


@dataclass
class QuantizedTensor:
    """A quantized tensor with its scale factor."""

    data: torch.Tensor  # INT8 tensor
    scale: float  # Scale factor for dequantization
    zero_point: int = 0  # For asymmetric quantization (0 for symmetric)

    def dequantize(self) -> torch.Tensor:
        """Convert back to floating point."""
        return (self.data.float() - self.zero_point) * self.scale

    @property
    def shape(self) -> torch.Size:
        return self.data.shape


def quantize_tensor(
    tensor: torch.Tensor,
    symmetric: bool = True,
    bits: int = 8,
) -> QuantizedTensor:
    """
    Quantize a floating point tensor to INT8.

    Args:
        tensor: Input FP tensor
        symmetric: If True, use symmetric quantization (zero_point = 0)
        bits: Number of bits (default 8)

    Returns:
        QuantizedTensor with INT8 data and scale
    """
    if symmetric:
        # Symmetric quantization: scale based on max absolute value
        max_val = tensor.abs().max().item()
        qmax = (1 << (bits - 1)) - 1  # 127 for int8

        if max_val == 0:
            scale = 1.0
        else:
            scale = max_val / qmax

        # Quantize
        data = torch.round(tensor / scale).clamp(-qmax, qmax).to(torch.int8)

        return QuantizedTensor(data=data, scale=scale, zero_point=0)

    else:
        # Asymmetric quantization
        min_val = tensor.min().item()
        max_val = tensor.max().item()
        qmin = -(1 << (bits - 1))  # -128
        qmax = (1 << (bits - 1)) - 1  # 127

        scale = (max_val - min_val) / (qmax - qmin)
        if scale == 0:
            scale = 1.0

        zero_point = int(round(qmin - min_val / scale))
        zero_point = max(qmin, min(qmax, zero_point))

        data = torch.round(tensor / scale + zero_point).clamp(qmin, qmax).to(torch.int8)

        return QuantizedTensor(data=data, scale=scale, zero_point=zero_point)


def dequantize_tensor(qtensor: QuantizedTensor) -> torch.Tensor:
    """Dequantize a QuantizedTensor to floating point."""
    return qtensor.dequantize()


def int8_matmul(
    x: QuantizedTensor,
    w: QuantizedTensor,
) -> Tuple[torch.Tensor, float]:
    """
    Perform INT8 matrix multiplication.

    Y = X @ W in INT32 accumulation, then dequantize.

    Args:
        x: Quantized input [M, K]
        w: Quantized weight [K, N]

    Returns:
        (Y_int32, combined_scale) where Y_fp = Y_int32 * combined_scale
    """
    # INT8 @ INT8 -> INT32 (use float for now, would use INT8 in production)
    x_int = x.data.to(torch.int32)
    w_int = w.data.to(torch.int32)

    y_int32 = torch.matmul(x_int, w_int)

    combined_scale = x.scale * w.scale

    return y_int32, combined_scale


def int8_matmul_dequant(
    x: QuantizedTensor,
    w: QuantizedTensor,
) -> torch.Tensor:
    """
    INT8 matmul with dequantization.

    Args:
        x: Quantized input
        w: Quantized weight

    Returns:
        Dequantized output tensor
    """
    y_int32, scale = int8_matmul(x, w)
    return y_int32.float() * scale


def quantize_model_weights(
    model: torch.nn.Module,
    symmetric: bool = True,
) -> dict:
    """
    Quantize all model weights to INT8.

    Args:
        model: PyTorch model
        symmetric: Use symmetric quantization

    Returns:
        Dict mapping parameter name to QuantizedTensor
    """
    quantized = {}

    for name, param in model.named_parameters():
        if param.requires_grad:  # Skip frozen params if any
            qtensor = quantize_tensor(param.data, symmetric=symmetric)
            quantized[name] = qtensor

    return quantized
