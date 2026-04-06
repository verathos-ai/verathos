"""
Quantization utilities for VeraLLM.

INT8 quantization provides deterministic computation
which is essential for verification.
"""

from verallm.quantization.int8 import (
    quantize_tensor,
    dequantize_tensor,
    int8_matmul,
    QuantizedTensor,
)

from verallm.quantization.detection import (
    detect_quantization,
    detect_layer_quant_mode,
    extract_int_weights,
    get_int8_weights_gpu,
    get_int8_weights_raw_gpu,
    get_int8_matmul_result,
    QuantizationType,
    QuantizationInfo,
    # INT4 (GPTQ) support
    extract_gptq_weights,
    unpack_int4_slice,
    dequant_int4_slice,
    get_int4_weights_as_int8_gpu,
    get_int4_slice_for_proof,
    # INT4 (AWQ) support
    is_awq_layer,
    unpack_awq_int4,
    dequant_awq_int4_slice,
    get_awq_weights_as_int8_gpu,
    # INT4 (bitsandbytes NF4/FP4) support
    extract_bnb_int4_weights,
    unpack_bnb_int4,
    dequant_bnb_int4,
    get_bnb_int4_weights_as_int8_gpu,
    # FP8 support
    is_fp8_layer,
    get_fp8_weights_as_int8_gpu,
    # NVFP4 (modelopt FP4) support
    is_nvfp4_layer,
    get_nvfp4_weights_as_int8_gpu,
)

__all__ = [
    # INT8 quantization
    "quantize_tensor",
    "dequantize_tensor",
    "int8_matmul",
    "QuantizedTensor",
    # Detection
    "detect_quantization",
    "detect_layer_quant_mode",
    "extract_int_weights",
    "get_int8_weights_gpu",
    "get_int8_weights_raw_gpu",
    "get_int8_matmul_result",
    "QuantizationType",
    "QuantizationInfo",
    # INT4 (GPTQ) support
    "extract_gptq_weights",
    "unpack_int4_slice",
    "dequant_int4_slice",
    "get_int4_weights_as_int8_gpu",
    "get_int4_slice_for_proof",
    # INT4 (AWQ) support
    "is_awq_layer",
    "unpack_awq_int4",
    "dequant_awq_int4_slice",
    "get_awq_weights_as_int8_gpu",
    # INT4 (bitsandbytes NF4/FP4) support
    "extract_bnb_int4_weights",
    "unpack_bnb_int4",
    "dequant_bnb_int4",
    "get_bnb_int4_weights_as_int8_gpu",
    # FP8 support
    "is_fp8_layer",
    "get_fp8_weights_as_int8_gpu",
    # NVFP4 (modelopt FP4) support
    "is_nvfp4_layer",
    "get_nvfp4_weights_as_int8_gpu",
]
