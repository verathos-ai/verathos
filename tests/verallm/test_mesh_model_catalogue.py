

def test_mesh_chat_template_fallback_packaged_for_deepseek():
    """The DeepSeek V4 Flash base repo ships no chat template (the
    authoritative one lives only in the GGUF metadata); the registry
    packages a committed copy so validator-side prompt construction
    works without downloading the GGUF ."""
    from jinja2 import Environment

    from verallm.registry.models import mesh_chat_template_fallback

    template = mesh_chat_template_fallback("deepseek-v4-flash-0731-q2-k-xl")
    assert template and len(template) > 10_000
    rendered = Environment().from_string(template).render(
        messages=[{"role": "user", "content": "ping"}],
        add_generation_prompt=True,
        bos_token="<bos>",
    )
    assert "ping" in rendered

    # Families with a templated base repo need no fallback.
    assert mesh_chat_template_fallback("qwen3.6-27b-q4-k-m") is None
    assert mesh_chat_template_fallback("not-a-model") is None
