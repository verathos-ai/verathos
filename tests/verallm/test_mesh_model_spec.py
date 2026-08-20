from __future__ import annotations

import hashlib

import pytest

from verallm.mesh.model_spec import (
    GGUF_PACKAGE_HASH_DOMAIN,
    build_gguf_model_spec_from_manifest,
    gguf_package_hash,
    gguf_tokenizer_hash,
    model_spec_summary,
    verify_gguf_model_files,
)
from verallm.mesh.types import canonical_json_bytes


class _Field:
    def __init__(self, value):
        self.value = value

    def contents(self):
        return self.value


class _Reader:
    def __init__(self, **overrides):
        fields = {
            "general.architecture": "qwen35moe",
            "qwen35moe.block_count": 40,
            "qwen35moe.embedding_length": 2048,
            "qwen35moe.attention.head_count": 16,
            "qwen35moe.attention.head_count_kv": 2,
            "qwen35moe.attention.key_length": 256,
            "qwen35moe.attention.value_length": 256,
            "qwen35moe.rope.dimension_count": 128,
            "qwen35moe.expert_count": 256,
            "qwen35moe.expert_used_count": 8,
            "qwen35moe.expert_feed_forward_length": 512,
            "qwen35moe.attention.layer_norm_rms_epsilon": 1e-6,
            "tokenizer.ggml.model": "gpt2",
            "tokenizer.ggml.tokens": ["a", "b", "c"],
            "tokenizer.ggml.merges": ["a b"],
            "tokenizer.chat_template": "{{ messages }}",
        }
        fields.update(overrides)
        self.fields = {name: _Field(value) for name, value in fields.items()}


def _single_manifest(digest: str = "11" * 32) -> dict:
    return {
        "tensor_manifest_root": "22" * 32,
        "model_file_sha256": digest,
        "model_files": [
            {"index": 0, "n_bytes": 1234, "sha256": digest, "path": "/model.gguf"}
        ],
    }


_BASE_REPO_HASH = bytes.fromhex("ab" * 32)


def _patch_tokenizer_anchor(monkeypatch, *, source="org/base-repo"):
    """Route the chain tokenizer anchor through fakes (no network)."""

    import verallm.registry.models as registry_models
    import verallm.registry.tokenizer_hash as registry_tokenizer_hash

    monkeypatch.setattr(
        registry_models, "mesh_tokenizer_source", lambda model_id: source
    )
    monkeypatch.setattr(
        registry_tokenizer_hash,
        "compute_tokenizer_hash",
        lambda model_id: _BASE_REPO_HASH,
    )


def test_build_qwen35moe_model_spec_reuses_existing_contract_shape(monkeypatch):
    _patch_tokenizer_anchor(monkeypatch)
    manifest = _single_manifest()
    reader = _Reader()
    spec = build_gguf_model_spec_from_manifest(
        manifest,
        reader,
        model_id="Qwen/Qwen3.6-35B-A3B-GGUF-Q4_K_M",
        gguf_scheme="Q4_K_M",
    )

    assert spec.weight_merkle_root == bytes.fromhex("22" * 32)
    assert spec.weight_file_hash == bytes.fromhex("11" * 32)
    # The chain anchor is the validator-computable base-repo tokenizer hash,
    # never the GGUF-header hash (validators cannot recompute that without
    # downloading the full weights).
    assert spec.tokenizer_hash == _BASE_REPO_HASH
    assert spec.tokenizer_hash != gguf_tokenizer_hash(reader)
    assert spec.quant_mode == "gguf_q4_k_m"
    assert (spec.num_layers, spec.hidden_dim, spec.num_heads, spec.head_dim) == (
        40,
        2048,
        16,
        256,
    )
    assert (spec.num_experts, spec.router_top_k, spec.intermediate_dim) == (256, 8, 512)
    assert spec.weight_block_merkle_roots == []
    assert spec.expert_weight_merkle_roots == {}
    assert spec.norm_type == "rmsnorm"
    assert model_spec_summary(spec)["layer_root_count"] == 0


def test_build_refuses_model_without_registered_tokenizer_source(monkeypatch):
    import verallm.registry.models as registry_models

    monkeypatch.setattr(
        registry_models, "mesh_tokenizer_source", lambda model_id: None
    )
    with pytest.raises(ValueError, match="no tokenizer source"):
        build_gguf_model_spec_from_manifest(
            _single_manifest(),
            _Reader(),
            model_id="not-a-registered-mesh-model",
            gguf_scheme="Q4_K_M",
        )


def test_tokenizer_hash_commits_every_tokenizer_metadata_field():
    original = gguf_tokenizer_hash(_Reader())
    assert original != gguf_tokenizer_hash(
        _Reader(**{"tokenizer.chat_template": "different"})
    )
    assert original != gguf_tokenizer_hash(
        _Reader(**{"tokenizer.ggml.merges": ["b c"]})
    )


def test_sharded_package_hash_matches_manifest_canonical_aggregate():
    files = [
        {"index": 0, "n_bytes": 10, "sha256": "31" * 32},
        {"index": 1, "n_bytes": 20, "sha256": "32" * 32},
    ]
    aggregate = hashlib.sha256(
        GGUF_PACKAGE_HASH_DOMAIN + canonical_json_bytes(files)
    ).hexdigest()
    manifest = {"model_file_sha256": aggregate, "model_files": files}
    assert gguf_package_hash(manifest).hex() == aggregate

    manifest["model_file_sha256"] = "33" * 32
    with pytest.raises(ValueError, match="aggregate model file hash"):
        gguf_package_hash(manifest)


def test_model_file_verification_rejects_content_or_size_drift(tmp_path):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"committed bytes")
    digest = hashlib.sha256(model.read_bytes()).hexdigest()
    manifest = {
        "model_file_sha256": digest,
        "model_files": [
            {
                "index": 0,
                "path": str(model),
                "n_bytes": model.stat().st_size,
                "sha256": digest,
            }
        ],
    }
    verify_gguf_model_files(manifest)
    model.write_bytes(b"changed")
    with pytest.raises(ValueError, match="size mismatch"):
        verify_gguf_model_files(manifest)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("qwen35moe.block_count", True, "positive integer"),
        ("qwen35moe.attention.head_count_kv", 1.5, "must be an integer"),
        ("qwen35moe.expert_count", -1, "non-negative integer"),
        ("qwen35moe.expert_used_count", 0, "expert_used_count is invalid"),
    ],
)
def test_model_spec_rejects_ambiguous_architecture_metadata(field, value, message):
    with pytest.raises(ValueError, match=message):
        build_gguf_model_spec_from_manifest(
            _single_manifest(),
            _Reader(**{field: value}),
            model_id="mesh-model",
            gguf_scheme="Q4_K_M",
        )


def test_model_spec_requires_committed_tokenizer_vocabulary():
    reader = _Reader()
    del reader.fields["tokenizer.ggml.tokens"]
    with pytest.raises(ValueError, match="tokenizer vocabulary"):
        build_gguf_model_spec_from_manifest(
            _single_manifest(),
            reader,
            model_id="mesh-model",
            gguf_scheme="Q4_K_M",
        )


def test_nextn_mtp_blocks_excluded_from_registered_layer_count(monkeypatch):
    # Ornith-1.5 pattern: the GGUF block stack carries one trailing NextN/MTP
    # block (block_count=41, nextn_predict_layers=1) that the runtime never
    # computes. The registered layer count anchors the verification layer
    # universe, so it must be the 40-layer COMPUTE stack.
    _patch_tokenizer_anchor(monkeypatch)
    reader = _Reader(**{
        "qwen35moe.block_count": 41,
        "qwen35moe.nextn_predict_layers": 1,
    })
    spec = build_gguf_model_spec_from_manifest(
        _single_manifest(),
        reader,
        model_id="ornith-1.5-35b-q4-k-m",
        gguf_scheme="Q4_K_M",
    )
    assert spec.num_layers == 40


def test_invalid_nextn_predict_layers_is_refused(monkeypatch):
    _patch_tokenizer_anchor(monkeypatch)
    reader = _Reader(**{
        "qwen35moe.block_count": 41,
        "qwen35moe.nextn_predict_layers": 41,
    })
    import pytest
    with pytest.raises(ValueError, match="nextn_predict_layers"):
        build_gguf_model_spec_from_manifest(
            _single_manifest(),
            reader,
            model_id="ornith-1.5-35b-q4-k-m",
            gguf_scheme="Q4_K_M",
        )
