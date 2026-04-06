"""Proof of Inference — shared data and throughput baselines.

This module retains the expected throughput table and shared prompts used by
the epoch-based canary testing system.  The block-triggered PoI challenge
derivation (PoiChallenge, get_poi_execution_block, get_tested_models,
derive_poi_challenge) has been removed in favor of epoch-based canary
testing (see neurons/canary.py).
"""

from __future__ import annotations

import math

# Shared prompts — used as seed material for canary prompt generation.
SHARED_PROMPTS = [
    "Explain the concept of zero-knowledge proofs in simple terms.",
    "What is the difference between symmetric and asymmetric encryption?",
    "Describe how a hash function works and why it is useful.",
    "What are the main benefits of decentralized systems?",
    "Explain the Byzantine Generals Problem in distributed computing.",
    "How does a Merkle tree work and where is it used?",
    "What is the role of consensus mechanisms in blockchain?",
    "Describe the concept of differential privacy.",
    "What are homomorphic encryption schemes and their applications?",
    "Explain how public key infrastructure (PKI) works.",
    "What is the difference between proof of work and proof of stake?",
    "How do neural networks learn from data?",
    "Explain the concept of gradient descent in machine learning.",
    "What is tokenization in the context of large language models?",
    "Describe how attention mechanisms work in transformer models.",
    "What is the role of the softmax function in neural networks?",
    "Explain the difference between supervised and unsupervised learning.",
    "How does backpropagation work in training neural networks?",
    "What is the vanishing gradient problem and how is it addressed?",
    "Describe the architecture of a transformer model.",
    "What is transfer learning and why is it important?",
    "Explain the concept of embeddings in natural language processing.",
    "How do convolutional neural networks process images?",
    "What is the difference between batch normalization and layer normalization?",
    "Describe the concept of reinforcement learning from human feedback.",
    "What is quantization in the context of model deployment?",
    "Explain the difference between dense and mixture-of-experts models.",
    "How does key-value caching improve inference performance?",
    "What is the role of the residual connection in deep networks?",
    "Describe how beam search works in text generation.",
    "What is the difference between greedy decoding and sampling?",
    "Explain the concept of model distillation.",
    "How does rotary position embedding work?",
    "What are the trade-offs of different quantization methods?",
    "Describe the concept of speculative decoding.",
    "What is the relationship between perplexity and model quality?",
    "Explain how flash attention reduces memory usage.",
    "What is the role of the feed-forward network in a transformer layer?",
    "Describe the multi-head attention mechanism.",
    "What are the advantages of grouped-query attention?",
    "Explain the concept of context window length in language models.",
    "How does tensor parallelism work for large model inference?",
    "What is the difference between pipeline and tensor parallelism?",
    "Describe the concept of activation checkpointing.",
    "What is the role of the tokenizer in language model inference?",
    "Explain how top-k and top-p sampling work.",
    "What is the difference between causal and bidirectional attention?",
    "Describe the concept of model sharding across GPUs.",
    "How does mixed-precision training improve efficiency?",
    "What is the relationship between batch size and training stability?",
]

# Expected throughput (tok/s) for long-context inference (~80% context fill + 200 output tokens).
# Calibrated for single-GPU inference with the dominant quantization per size tier.
EXPECTED_TPS: dict[float, float] = {
    0.5: 67.0,
    1.0: 40.0,
    3.0: 20.0,
    7.0: 13.0,
    8.0: 13.0,
    12.0: 10.0,
    14.0: 8.0,
    24.0: 5.0,
    32.0: 4.0,
    70.0: 2.0,
}


def get_expected_tps(active_params_b: float) -> float:
    """Look up expected tok/s for a model size, interpolating if needed."""
    if active_params_b in EXPECTED_TPS:
        return EXPECTED_TPS[active_params_b]

    # Find nearest keys and interpolate
    sizes = sorted(EXPECTED_TPS.keys())
    if active_params_b <= sizes[0]:
        return EXPECTED_TPS[sizes[0]]
    if active_params_b >= sizes[-1]:
        return EXPECTED_TPS[sizes[-1]]

    for i in range(len(sizes) - 1):
        if sizes[i] <= active_params_b <= sizes[i + 1]:
            lo, hi = sizes[i], sizes[i + 1]
            # Log-linear interpolation (throughput scales roughly log-linearly)
            t = (math.log(active_params_b) - math.log(lo)) / (math.log(hi) - math.log(lo))
            return EXPECTED_TPS[lo] * (1 - t) + EXPECTED_TPS[hi] * t

    return 10.0  # fallback
