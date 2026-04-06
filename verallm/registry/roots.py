"""
Compute weight Merkle roots for on-chain model registration.

In production this is a ONE-TIME operation run by a trusted registrant
(subnet owner, DAO, or any auditor). The resulting ModelSpec — specifically
the weight_block_merkle_roots — is published on-chain (~1-4 KB).

Because the computation is deterministic (same model + same quantization
= identical roots), anyone can independently verify the published roots.
"""

import hashlib
import logging
import time

import torch

logger = logging.getLogger(__name__)

from verallm.introspection import (
    get_layers,
    get_mlp,
    get_gate_proj,
    get_text_config,
    get_embedding_module,
)
from verallm.quantization import (
    detect_quantization,
    detect_layer_quant_mode,
    get_int8_weights_raw_gpu,
    get_int4_weights_as_int8_gpu,
    is_awq_layer,
    get_awq_weights_as_int8_gpu,
    get_bnb_int4_weights_as_int8_gpu,
    get_fp8_weights_as_int8_gpu,
    get_nvfp4_weights_as_int8_gpu,
    unpack_awq_int4,
    unpack_int4_slice,
    extract_gptq_weights,
)
from verallm.crypto.merkle import compute_flat_weight_root, _HAS_CUDA_BLAKE3
from verallm.moe import (
    is_moe_layer,
    count_experts,
    has_3d_batched_experts,
    get_expert_weight_tensor,
    get_expert_proj_module,
    get_router,
)
from verallm.types import ModelSpec


def _compute_lm_head_root_cpu(lm_head_mod, hidden_size: int, chunk_size: int) -> bytes:
    """Compute lm_head weight root: unpack on CPU, hash on GPU via chunked transfer.

    Large-vocab models (131k+ tokens) create multi-GB intermediates during
    weight unpacking/quantization that exceed free GPU memory on 24GB cards.
    Moving the packed weights to CPU first avoids GPU OOM — system RAM is
    typically 32-128 GB.  ``compute_flat_weight_root`` automatically uses
    chunked GPU BLAKE3 hashing (256 MiB slices) for CPU int8 tensors.
    """
    mode = detect_layer_quant_mode(lm_head_mod)

    # AWQ int4: move packed qweight to CPU, unpack there.
    if mode == "int4" and is_awq_layer(lm_head_mod):
        _oq = getattr(lm_head_mod, '_orig_qweight', None)
        qw = _oq if _oq is not None else lm_head_mod.qweight
        W_int8 = unpack_awq_int4(qw.cpu())  # [K, N] int8 on CPU
        if W_int8.shape[0] == hidden_size:
            return compute_flat_weight_root(W_int8, chunk_size)

    # GPTQ int4: move packed qweight to CPU, unpack there.
    if mode == "int4" and hasattr(lm_head_mod, 'qweight'):
        qweight, _scales, _qzeros, _g_idx = extract_gptq_weights(lm_head_mod)
        W_int8 = unpack_int4_slice(qweight.cpu())  # [K, N] int8 on CPU
        if W_int8.shape[0] == hidden_size:
            return compute_flat_weight_root(W_int8, chunk_size)

    # FP16/BF16 fallback: move to CPU, quantize to int8.
    # Explicit del + gc.collect between steps to minimise peak RSS —
    # for 131k-vocab models the intermediates total ~10 GB.
    if hasattr(lm_head_mod, 'weight'):
        import gc
        W_raw = lm_head_mod.weight.data.cpu()
        if W_raw.shape[0] != hidden_size:
            W_trans = W_raw.T.contiguous()
            del W_raw
        else:
            W_trans = W_raw
        W_float = W_trans.float()
        del W_trans
        absmax = W_float.abs().max().clamp(min=1e-8)
        W_int8 = (W_float / absmax * 127).round().clamp(-128, 127).to(torch.int8)
        del W_float
        gc.collect()
        return compute_flat_weight_root(W_int8, chunk_size)

    raise RuntimeError("No extraction method available for lm_head on CPU")


def compute_model_roots(model, model_name: str, chunk_size: int = 128) -> ModelSpec:
    """
    Compute weight Merkle roots for a model and return a ModelSpec.

    Quantization format is auto-detected from the model — no need to specify.
    All modes produce int8 Merkle trees (1 byte per element).

    Args:
        model: Pre-loaded HuggingFace or vLLM-extracted model
        model_name: Model identifier string
        chunk_size: FlatWeightMerkle chunk size (elements per leaf)
    """
    model_config = model.config
    text_config = get_text_config(model)
    num_params = sum(p.numel() for p in model.parameters())
    hidden_size = getattr(text_config, "hidden_size", getattr(text_config, "n_embd", 768))
    intermediate_size = getattr(text_config, "intermediate_size",
                                getattr(text_config, "n_inner", None) or hidden_size * 4)
    num_layers = getattr(text_config, "num_hidden_layers", getattr(text_config, "n_layer", 12))
    num_heads = getattr(text_config, "num_attention_heads", getattr(text_config, "n_head", 12))

    # Auto-detect quantization from model
    quant_mode = detect_quantization(model).quant_mode
    mode_desc = {"int8": "native INT8", "int4": "unpacked INT4→int8", "fp8": "FP8 bytes→int8", "nvfp4": "NVFP4 packed→int8", "fp16": "FP16→int8"}.get(quant_mode, quant_mode)

    logger.info("Registry: Model has %s params", f"{num_params:,}")
    logger.info("Registry: Architecture: %d layers, hidden=%d, intermediate=%d",
                num_layers, hidden_size, intermediate_size)
    logger.info("Registry: Detected quantization: %s (%s)", quant_mode, mode_desc)

    # Model identity commitment (hash of parameter names + shapes)
    commitment_data = []
    for name, param in model.named_parameters():
        param_hash = hashlib.sha256(name.encode() + str(param.shape).encode()).digest()
        commitment_data.append(param_hash)
    model_commitment = hashlib.sha256(b"".join(commitment_data)).digest()

    def _compute_root_for_weight(gate_proj, W_raw_tensor=None):
        """Compute Merkle root for a weight tensor — auto-detects quant format."""
        mode = detect_layer_quant_mode(gate_proj) if gate_proj is not None else "fp16"
        W_int8 = None
        if gate_proj is not None:
            try:
                if mode == "int8":
                    W_int8 = get_int8_weights_raw_gpu(gate_proj).cpu()
                elif mode == "int4":
                    if hasattr(gate_proj, 'qweight'):
                        if is_awq_layer(gate_proj):
                            W_int8 = get_awq_weights_as_int8_gpu(gate_proj)
                        else:
                            W_int8 = get_int4_weights_as_int8_gpu(gate_proj)
                    else:
                        W_int8 = get_bnb_int4_weights_as_int8_gpu(gate_proj)
                elif mode == "fp8":
                    W_int8 = get_fp8_weights_as_int8_gpu(gate_proj).cpu()
                elif mode == "nvfp4":
                    W_int8 = get_nvfp4_weights_as_int8_gpu(gate_proj).cpu()
            except (ValueError, KeyError, AttributeError, RuntimeError):
                W_int8 = None
            # Validate: Marlin-repacked GPTQ weights have wrong layout
            # (e.g. 1024x256 instead of 2048x128) when _orig_qweight is missing.
            if W_int8 is not None and W_int8.dim() == 2 and W_int8.shape[0] == hidden_size:
                return compute_flat_weight_root(W_int8.cpu(), chunk_size)
            W_int8 = None
        # FP16 or fallback: quantize to int8 then hash.  For layer weights
        # (small), keep on GPU for speed.  The lm_head CPU fallback handles
        # large-vocab models separately via _compute_lm_head_root_cpu().
        if W_raw_tensor is not None:
            W_float = (W_raw_tensor.T if W_raw_tensor.shape[0] != hidden_size else W_raw_tensor).float()
        elif gate_proj is not None and hasattr(gate_proj, 'weight'):
            W_raw = gate_proj.weight.data
            W_float = (W_raw.T.contiguous() if W_raw.shape[0] != hidden_size else W_raw).float()
        else:
            W_float = None
        if W_float is not None:
            absmax = W_float.abs().max().clamp(min=1e-8)
            W_int8 = (W_float / absmax * 127).round().clamp(-128, 127).to(torch.int8)
            del W_float  # free float32 intermediate immediately
            if W_int8.is_cuda:
                W_int8 = W_int8.cpu()
            return compute_flat_weight_root(W_int8, chunk_size)
        # Identity-matrix fallback: forward() dequantizes any format (Marlin, etc.)
        if gate_proj is not None and hasattr(gate_proj, 'forward'):
            try:
                out_dim = getattr(gate_proj, 'output_size', 0)
                if out_dim <= 0:
                    out_dim = getattr(gate_proj, 'out_features', 0)
                if out_dim <= 0 and hasattr(gate_proj, 'output_partition_sizes'):
                    out_dim = sum(gate_proj.output_partition_sizes)
                if out_dim > 0:
                    device = next(gate_proj.parameters()).device
                    dtype = getattr(gate_proj, 'params_dtype', torch.float16)
                    eye = torch.eye(hidden_size, dtype=dtype, device=device)
                    with torch.no_grad():
                        result = gate_proj(eye)
                    W_fp16 = result[0] if isinstance(result, (tuple, list)) else result
                    W_float = W_fp16.float()
                    absmax = W_float.abs().max().clamp(min=1e-8)
                    W_int8 = (W_float / absmax * 127).round().clamp(-128, 127).to(torch.int8)
                    if W_int8.is_cuda:
                        W_int8 = W_int8.cpu()
                    return compute_flat_weight_root(W_int8, chunk_size)
            except Exception as e:
                logger.warning("Identity-matrix weight extraction failed: %s", e)
        return b'\x00' * 32

    # Quantized MoE: set up on-demand reader for per-expert weights from checkpoint.
    # AWQ, GPTQ, FP8, and MXFP4 all repack weights (Marlin format) making live
    # tensors unusable — the reader lazily loads original weights from checkpoint.
    gptq_moe_reader = None
    try:
        from verallm.miner.vllm_utils import (
            _detect_awq_moe_experts, _create_awq_moe_reader,
            _detect_gptq_moe_experts, _create_gptq_moe_reader,
            _detect_fp8_moe_experts, _create_fp8_moe_reader,
            _detect_mxfp4_moe_experts, _create_mxfp4_moe_reader,
        )
        if _detect_mxfp4_moe_experts(model_name):
            gptq_moe_reader = _create_mxfp4_moe_reader(model_name)
            if gptq_moe_reader is not None:
                logger.info("Registry: MXFP4 MoE detected — using checkpoint reader for expert roots")
        elif _detect_awq_moe_experts(model_name):
            gptq_moe_reader = _create_awq_moe_reader(model_name)
            if gptq_moe_reader is not None:
                logger.info("Registry: AWQ MoE detected — using checkpoint reader for expert roots")
        elif _detect_gptq_moe_experts(model_name):
            gptq_moe_reader = _create_gptq_moe_reader(model_name)
            if gptq_moe_reader is not None:
                logger.info("Registry: GPTQ MoE detected — using checkpoint reader for expert roots")
        elif _detect_fp8_moe_experts(model_name):
            gptq_moe_reader = _create_fp8_moe_reader(model_name)
            if gptq_moe_reader is not None:
                logger.info("Registry: FP8 MoE detected — using checkpoint reader for expert roots")
    except ImportError:
        pass

    # Compute per-layer weight Merkle roots
    logger.info(
        "Registry: Computing weight Merkle roots (%d layers, may take a few minutes). "
        "Cache at .model_root_cache/ is reusable — copy it to skip this on other instances.",
        num_layers)
    weight_block_merkle_roots = []
    expert_merkle_roots_map = {}  # layer_idx -> List[bytes] (per-expert roots)
    router_merkle_roots_map = {}  # layer_idx -> bytes (router/gate root)
    layers = get_layers(model)
    _root_t0 = time.perf_counter()

    for idx in range(num_layers):
        logger.debug("Registry: Layer %d/%d root...", idx + 1, num_layers)
        if idx < len(layers):
            layer = layers[idx]
            mlp = get_mlp(layer) if layer is not None else None

            # Check if this is an MoE layer
            if is_moe_layer(layer):
                n_experts = count_experts(layer)
                expert_roots = []
                router_root = b'\x00' * 32

                router_mod = get_router(layer)
                if router_mod is not None:
                    router_root = _compute_root_for_weight(router_mod)

                # Optimization: for 3D batched experts, quantize entire batch on GPU
                # Skip when GPTQ MoE reader is active — w13_weight is Marlin-packed garbage
                batch_int8_gpu = None
                batch_int8_cpu = None
                if has_3d_batched_experts(layer) and gptq_moe_reader is None:
                    mlp = get_mlp(layer) if layer is not None else None
                    experts_mod = getattr(mlp, 'experts', None) or getattr(mlp, 'routed_experts', None)
                    if experts_mod is not None:
                        for param_name in ['gate_up_proj', 'w1', 'w13_weight']:
                            if hasattr(experts_mod, param_name):
                                p = getattr(experts_mod, param_name)
                                if hasattr(p, 'device') and p.device.type == 'meta':
                                    break
                                gpu_3d = p.data if hasattr(p, 'data') else p
                                # NVFP4 packed uint8: reinterpret as int8 directly
                                if gpu_3d.dtype == torch.uint8:
                                    batch_q = gpu_3d.view(torch.int8)
                                    if _HAS_CUDA_BLAKE3 and batch_q.is_cuda:
                                        batch_int8_gpu = batch_q
                                    else:
                                        batch_int8_cpu = batch_q.cpu()
                                else:
                                    # Quantize on CPU to avoid GPU OOM on tight-VRAM
                                    # models (e.g. 22 GiB MoE on 32 GiB GPU).
                                    # The 3D batch [E, inter, hidden] as float32 can
                                    # exceed free VRAM; CPU quantization is fine for
                                    # a one-time startup cost.
                                    cpu_t = gpu_3d.permute(0, 2, 1).contiguous().cpu().float()
                                    absmax = cpu_t.abs().amax(dim=(1, 2), keepdim=True).clamp(min=1e-8)
                                    batch_q = (cpu_t / absmax * 127).round().clamp(-128, 127).to(torch.int8)
                                    del cpu_t, absmax
                                    batch_int8_cpu = batch_q
                                break

                for eidx in range(n_experts):
                    if batch_int8_gpu is not None:
                        root = compute_flat_weight_root(batch_int8_gpu[eidx], chunk_size)
                    elif batch_int8_cpu is not None:
                        root = compute_flat_weight_root(batch_int8_cpu[eidx], chunk_size)
                    else:
                        expert_proj = get_expert_proj_module(layer, eidx, "gate_up")
                        if expert_proj is not None:
                            root = _compute_root_for_weight(expert_proj)
                        elif gptq_moe_reader is not None:
                            # GPTQ MoE: read per-expert qweight from checkpoint
                            W_int8 = gptq_moe_reader.get_expert_int8(idx, eidx, "gate_up")
                            if W_int8 is not None:
                                root = compute_flat_weight_root(W_int8, chunk_size)
                            else:
                                root = b'\x00' * 32
                        else:
                            W_tensor = get_expert_weight_tensor(layer, eidx, "gate_up")
                            if W_tensor is not None:
                                W_raw = W_tensor.float().cpu()
                                W_t = W_raw.T.contiguous().float()
                                absmax = W_t.abs().max().clamp(min=1e-8)
                                W_int = (W_t / absmax * 127).round().clamp(-128, 127).to(torch.int8)
                                root = compute_flat_weight_root(W_int, chunk_size)
                            else:
                                root = b'\x00' * 32
                    expert_roots.append(root)
                    if (eidx + 1) % 16 == 0 or eidx == n_experts - 1:
                        logger.debug("Layer %d/%d: expert %d/%d...", idx, num_layers, eidx + 1, n_experts)

                h = hashlib.sha256(b"MOE_LAYER_V2")
                h.update(router_root)
                for er in expert_roots:
                    h.update(er)
                layer_root = h.digest()

                weight_block_merkle_roots.append(layer_root)
                expert_merkle_roots_map[idx] = expert_roots
                router_merkle_roots_map[idx] = router_root
                logger.info("Layer %d/%d: %d expert roots + router root -> hierarchical layer root",
                            idx, num_layers, n_experts)

            else:
                # Dense layer
                gate_proj_mod = get_gate_proj(mlp) if mlp else None
                has_weights = gate_proj_mod is not None and (
                    hasattr(gate_proj_mod, 'weight') or hasattr(gate_proj_mod, 'qweight')
                )

                if has_weights:
                    root = _compute_root_for_weight(gate_proj_mod)
                    weight_block_merkle_roots.append(root)
                else:
                    weight_block_merkle_roots.append(b'\x00' * 32)
        else:
            weight_block_merkle_roots.append(b'\x00' * 32)

    on_chain_bytes = len(weight_block_merkle_roots) * 32
    moe_expert_count = sum(len(roots) for roots in expert_merkle_roots_map.values())
    logger.info("Registry: Computed %d weight Merkle roots (%d bytes for on-chain storage)",
                len(weight_block_merkle_roots), on_chain_bytes)
    if moe_expert_count > 0:
        logger.info("Registry: MoE: %d expert roots across %d MoE layers",
                    moe_expert_count, len(expert_merkle_roots_map))

    router_top_k = (
        getattr(model_config, "num_experts_per_tok", 0)
        or getattr(text_config, "num_experts_per_tok", 0)
        or getattr(model_config, "num_selected_experts", 0)
        or getattr(text_config, "num_selected_experts", 0)
        or getattr(model_config, "top_k", 0)
        or getattr(text_config, "top_k", 0)
    )
    router_scoring = (
        getattr(model_config, "scoring_func", None)
        or getattr(text_config, "scoring_func", None)
        or "softmax"
    )

    # Detect actual W column count for dense layers with fused gate+up.
    # Use logical output size (output_size_per_partition or output_partition_sizes)
    # rather than raw tensor shape, which may be Marlin-packed.
    dense_w_cols = intermediate_size
    for layer_idx_tmp in range(min(num_layers, len(layers))):
        if not is_moe_layer(layers[layer_idx_tmp]):
            mlp_tmp = get_mlp(layers[layer_idx_tmp])
            gp_tmp = get_gate_proj(mlp_tmp) if mlp_tmp else None
            if gp_tmp is not None:
                # Prefer logical output size attributes (works for all formats
                # including Marlin-repacked AWQ/GPTQ/FP8).
                out_size = getattr(gp_tmp, 'output_size_per_partition', 0)
                if out_size <= 0 and hasattr(gp_tmp, 'output_partition_sizes'):
                    out_size = sum(gp_tmp.output_partition_sizes)
                if out_size > 0 and out_size != intermediate_size:
                    dense_w_cols = out_size
                elif out_size <= 0:
                    # Fallback to tensor shape for non-vLLM layers
                    w_data = None
                    if hasattr(gp_tmp, 'weight'):
                        w_data = gp_tmp.weight.data
                    elif hasattr(gp_tmp, 'qweight'):
                        w_data = gp_tmp.qweight.data
                    if w_data is not None:
                        actual_cols = max(w_data.shape)
                        if actual_cols != intermediate_size:
                            dense_w_cols = actual_cols
                break

    # Detect expert W column count for MoE models with fused gate+up (3D batched)
    # Checkpoint reader takes priority: Marlin-repacked GPU tensors have
    # different packed dimensions that don't reflect logical column count.
    expert_w_cols = 0
    if expert_merkle_roots_map:
        for layer_idx in expert_merkle_roots_map:
            if layer_idx < len(layers):
                layer = layers[layer_idx]
                if gptq_moe_reader is not None:
                    W_sample = gptq_moe_reader.get_expert_int8(layer_idx, 0, "gate_up")
                    if W_sample is not None:
                        expert_w_cols = W_sample.shape[1]  # [K, N] -> N = 2*intermediate
                        del W_sample
                elif has_3d_batched_experts(layer):
                    mlp_mod = get_mlp(layer)
                    experts_mod = getattr(mlp_mod, 'experts', None) or getattr(mlp_mod, 'routed_experts', None)
                    if experts_mod is not None:
                        for pn in ['gate_up_proj', 'w1', 'w13_weight']:
                            if hasattr(experts_mod, pn):
                                p = getattr(experts_mod, pn)
                                d = p.data if hasattr(p, 'data') else p
                                if not (hasattr(d, 'device') and d.device.type == 'meta'):
                                    # NVFP4 packed (uint8) weights are NOT transposed,
                                    # so per-expert shape is [shape[1], shape[2]].
                                    # Non-NVFP4 weights GET transposed (permute 0,2,1),
                                    # so per-expert shape after transpose is [shape[2], shape[1]].
                                    if d.dtype == torch.uint8:
                                        expert_w_cols = d.shape[2]  # no transpose
                                    else:
                                        expert_w_cols = d.shape[1]  # transposed
                                break
                break

    lm_head_root = b""
    lm_head_mod = (
        getattr(model, "lm_head", None)
        or getattr(model, "output", None)
        or getattr(model, "embed_out", None)
    )
    # Conditional-generation wrappers (e.g. Gemma3ForConditionalGeneration)
    # nest CausalLM inside model.language_model.
    if lm_head_mod is None:
        lm = getattr(model, "language_model", None)
        if lm is not None:
            lm_head_mod = (
                getattr(lm, "lm_head", None)
                or getattr(lm, "output", None)
                or getattr(lm, "embed_out", None)
            )
    if lm_head_mod is not None:
        import gc
        torch.cuda.empty_cache()
        try:
            lm_head_root = _compute_root_for_weight(lm_head_mod)
            logger.info("Registry: lm_head weight root: %s...", lm_head_root.hex()[:16])
        except Exception as e:
            # GPU OOM for large-vocab lm_heads (e.g. 131k Mistral/Devstral).
            # Clean up GPU memory from the failed extraction attempt before
            # CPU fallback — without this, leaked GPU temps from the OOM
            # prevent even the 256 MiB BLAKE3 chunked hashing slices.
            gc.collect()
            torch.cuda.empty_cache()
            logger.debug("Registry: lm_head GPU extraction failed (%s), falling back to CPU", type(e).__name__)
            try:
                lm_head_root = _compute_lm_head_root_cpu(
                    lm_head_mod, hidden_size, chunk_size
                )
                logger.info("Registry: lm_head weight root (CPU): %s...", lm_head_root.hex()[:16])
            except Exception as e2:
                logger.warning("lm_head root computation failed: %s", e2)
                lm_head_root = b""

    # Compute embedding table Merkle root (input binding proofs).
    # Embedding shape: [vocab_size, hidden_dim].  Quantize to int8 like
    # other weights, compute FlatWeightMerkle root.
    embedding_root = b""
    embed_mod = get_embedding_module(model)
    if embed_mod is not None and hasattr(embed_mod, 'weight'):
        import gc
        try:
            W_emb = embed_mod.weight.data.cpu().float()
            absmax = W_emb.abs().max().clamp(min=1e-8)
            W_emb_int8 = (W_emb / absmax * 127).round().clamp(-128, 127).to(torch.int8)
            del W_emb
            gc.collect()
            embedding_root = compute_flat_weight_root(W_emb_int8, chunk_size)
            logger.info("Registry: embedding weight root: %s... (shape %s)",
                        embedding_root.hex()[:16], embed_mod.weight.shape)
        except Exception as e:
            logger.warning("embedding root computation failed: %s", e)
            embedding_root = b""
    else:
        logger.info("Registry: embedding module not found (skipping embedding root)")

    # Per-layer expert count (all MoE layers have the same number of experts)
    num_experts_per_layer = 0
    if expert_merkle_roots_map:
        first_layer_roots = next(iter(expert_merkle_roots_map.values()), [])
        num_experts_per_layer = len(first_layer_roots)

    model_spec = ModelSpec(
        model_id=model_name,
        weight_merkle_root=model_commitment,
        num_layers=num_layers,
        hidden_dim=hidden_size,
        num_heads=num_heads,
        head_dim=hidden_size // num_heads,
        intermediate_dim=dense_w_cols,
        vocab_size=(
            getattr(model_config, "vocab_size", None)
            or getattr(getattr(model_config, "text_config", None), "vocab_size", None)
            or 50257
        ),
        activation="silu",
        norm_type="rmsnorm",
        attention_type="mha",
        weight_block_merkle_roots=weight_block_merkle_roots,
        w_merkle_chunk_size=chunk_size,
        expert_weight_merkle_roots=expert_merkle_roots_map,
        router_weight_merkle_roots=router_merkle_roots_map,
        lm_head_weight_merkle_root=lm_head_root,
        embedding_weight_merkle_root=embedding_root,
        quant_mode=quant_mode,
        expert_w_num_cols=expert_w_cols,
        num_experts=num_experts_per_layer,
        router_top_k=int(router_top_k) if router_top_k else 0,
        router_scoring=str(router_scoring),
    )

    # Compute tokenizer hash for on-chain anchoring (validator-side drift
    # detection).  Fail HARD at publish time — silently registering an empty
    # hash would permanently disable drift detection for this model.  The
    # only valid path to an empty hash is legacy specs registered before
    # this upgrade.
    from verallm.registry.tokenizer_hash import compute_tokenizer_hash
    model_spec.tokenizer_hash = compute_tokenizer_hash(model_name)
    logger.info("Registry: tokenizer hash: %s...",
                model_spec.tokenizer_hash.hex()[:16])

    return model_spec
