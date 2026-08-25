

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


def test_mesh_model_source_layers_prefer_variant_override():
    """Quants of one family can be cut from different base builds, so a
    variant-level layer count must win over the family entry's; without
    an override the entry value applies unchanged."""
    from verallm.registry.models import mesh_model_source

    # Override present: the dynamic-quant rebuild dropped a layer.
    assert mesh_model_source("qwen3.8-27b-q4-k-m")[3] == 64
    # No override: sibling quant inherits the family entry.
    assert mesh_model_source("qwen3.8-27b-q4-k-xl")[3] == 65


def test_registered_mesh_models_match_chain_committed_facts():
    """The catalogue must agree with what is registered on chain: the
    chain-bound launch refuses a mesh whose local GGUF layer count or
    package identity differs from the registered spec, so any drift
    here bricks fresh onboards of that model. These constants are the
    registered facts; a catalogue edit that breaks parity must fail
    loudly instead of failing at launch on an operator's box."""
    from verallm.registry.models import (
        mesh_model_manifest_root,
        mesh_model_source,
    )

    registered = (
        ("deepseek-v4-flash-0731-q2-k-xl", 43, 96_832_508_352,
         "f0cc75b069e590e73b7e2043aa008ace9f1696ec8e0b3928efc5540661144b86"),
        ("glm-5.2-iq2-m", 79, 238_577_580_768,
         "655f166d7fe2e3539ee607cf53f433aa153c33d486e977baca02d4b1d59a2a7a"),
        ("ornith-1.5-35b-q4-k-m", 40, 21_713_463_040,
         "4b0cecafbf3c9c3b0cdd287b322d2500359fb8e2ad8454ca71cd04ec9adae5a4"),
        ("qwen3.8-27b-q4-k-m", 64, 16_464_440_224,
         "de3d2acd111777f93044ca384eb09e89a4118115494a5cd6f0f734b2d2d80359"),
        ("qwen3.8-27b-q4-k-xl", 65, 17_923_394_624,
         "d079844f338dde253369fb768c24a3fb5cbf610e6323432e1a2bd8f7a514f529"),
        ("qwen3.8-27b-uncensored-q4-k-m", 65, 16_810_714_528,
         "dd4f8ff536844cb75fbab40b1f4c688478096b9ba6aa02374e13f35aaef13fdc"),
    )
    for mesh_model_id, layers, model_bytes, manifest_root in registered:
        source = mesh_model_source(mesh_model_id)
        assert source is not None, mesh_model_id
        _repo, _files, got_bytes, got_layers = source
        assert got_bytes == model_bytes, mesh_model_id
        assert got_layers == layers, mesh_model_id
        assert mesh_model_manifest_root(mesh_model_id) == manifest_root, (
            mesh_model_id
        )
