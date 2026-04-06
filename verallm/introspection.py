"""
Architecture-agnostic model introspection.

Standalone functions for accessing transformer layers, MLP modules, and
gate projections across different model architectures. Used by Miner,
Validator, and compute_model_roots() — eliminates triple duplication.
"""



def get_layers(model):
    """Get transformer layers from any supported architecture.

    Supports:
    - LLaMA/DeepSeek/Qwen/Mistral: model.model.layers
    - Gemma-3 multimodal: model.language_model.model.layers
    - GPT-2/GPT-Neo: model.transformer.h
    - Falcon: model.transformer.h
    - BLOOM: model.transformer.h
    - OPT: model.model.decoder.layers
    - GPT-NeoX: model.gpt_neox.layers
    """
    if model is None:
        return []
    # Multimodal models (Gemma-3, etc.): text model under language_model
    if hasattr(model, 'language_model'):
        lm = model.language_model
        if hasattr(lm, 'model') and hasattr(lm.model, 'layers'):
            return lm.model.layers
    if hasattr(model, 'model'):
        if hasattr(model.model, 'layers'):
            return model.model.layers  # LLaMA, DeepSeek, Qwen, Mistral
        if hasattr(model.model, 'decoder') and hasattr(model.model.decoder, 'layers'):
            return model.model.decoder.layers  # OPT
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        return model.transformer.h  # GPT-2, GPT-Neo, Falcon, BLOOM
    if hasattr(model, 'gpt_neox') and hasattr(model.gpt_neox, 'layers'):
        return model.gpt_neox.layers  # GPT-NeoX
    return []


def get_mlp(layer):
    """Get MLP module from a transformer layer.

    Supports various naming conventions:
    - block_sparse_moe (Mixtral MoE)
    - mlp (most models)
    - feed_forward, ff (some architectures)
    """
    for attr in ['block_sparse_moe', 'mlp', 'feed_forward', 'ff']:
        if hasattr(layer, attr):
            return getattr(layer, attr)
    return None


def get_gate_proj(mlp):
    """Get gate/up projection from MLP module.

    Supports:
    - gate_proj: LLaMA/DeepSeek/Qwen/Mistral (SwiGLU)
    - gate_up_proj: Phi-3/4 (fused gate+up, 2x intermediate cols)
    - c_fc: GPT-2 (Conv1D)
    - dense_h_to_4h: Falcon, GPT-NeoX
    - fc1: BLOOM, OPT
    - w1, up_proj: Other architectures
    """
    for attr in ['gate_proj', 'gate_up_proj', 'c_fc', 'dense_h_to_4h', 'fc1', 'w1', 'up_proj']:
        if hasattr(mlp, attr):
            return getattr(mlp, attr)
    return None


def get_embedding_module(model):
    """Get token embedding module from any supported architecture.

    Supports:
    - LLaMA/DeepSeek/Qwen/Mistral: model.model.embed_tokens
    - Gemma-3 multimodal: model.language_model.model.embed_tokens
    - GPT-2/GPT-Neo: model.transformer.wte
    - Falcon/BLOOM: model.transformer.word_embeddings
    - OPT: model.model.decoder.embed_tokens
    - GPT-NeoX: model.gpt_neox.embed_in

    Returns:
        The nn.Embedding module, or None if not found.
    """
    if model is None:
        return None
    # Multimodal models (Gemma-3, etc.)
    if hasattr(model, 'language_model'):
        lm = model.language_model
        if hasattr(lm, 'model') and hasattr(lm.model, 'embed_tokens'):
            return lm.model.embed_tokens
    if hasattr(model, 'model'):
        if hasattr(model.model, 'embed_tokens'):
            return model.model.embed_tokens  # LLaMA, DeepSeek, Qwen, Mistral
        if hasattr(model.model, 'decoder') and hasattr(model.model.decoder, 'embed_tokens'):
            return model.model.decoder.embed_tokens  # OPT
    if hasattr(model, 'transformer'):
        if hasattr(model.transformer, 'wte'):
            return model.transformer.wte  # GPT-2, GPT-Neo
        if hasattr(model.transformer, 'word_embeddings'):
            return model.transformer.word_embeddings  # Falcon, BLOOM
    if hasattr(model, 'gpt_neox') and hasattr(model.gpt_neox, 'embed_in'):
        return model.gpt_neox.embed_in  # GPT-NeoX
    return None


def get_text_config(model):
    """Get text model config, handling multimodal models with nested text_config."""
    config = model.config
    return getattr(config, "text_config", config)


def get_num_layers(model) -> int:
    """Get number of transformer layers."""
    config = get_text_config(model)
    return getattr(config, "num_hidden_layers", getattr(config, "n_layer", 12))


def get_hidden_dim(model) -> int:
    """Get hidden dimension."""
    config = get_text_config(model)
    return getattr(config, "hidden_size", getattr(config, "n_embd", 768))


def get_intermediate_dim(model) -> int:
    """Get intermediate (FFN) dimension."""
    config = get_text_config(model)
    hidden = get_hidden_dim(model)
    return getattr(config, "intermediate_size", getattr(config, "n_inner", None) or hidden * 4)
