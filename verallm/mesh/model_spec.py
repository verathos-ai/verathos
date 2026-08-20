"""Build the existing on-chain ``ModelSpec`` from a committed GGUF package.

GGUF meshes do not need a second contract.  The existing ModelRegistry fields
carry runtime-specific commitments as follows:

* ``weight_merkle_root`` is the GGUF tensor-manifest root;
* ``weight_file_hash`` is the single-file SHA256, or a deterministic package
  hash for a sharded GGUF;
* ``tokenizer_hash`` is the validator-computable base-repo tokenizer hash
  (``compute_tokenizer_hash`` over the family's HF tokenizer artifacts) —
  NOT a GGUF-header hash: validators must be able to recompute the anchor
  without downloading model weights, and the GGUF-embedded tokenizer is
  already integrity-covered by ``weight_file_hash``/``weight_merkle_root``;
* ordinary architecture fields are read from the GGUF header.

The vLLM-only per-layer/per-expert root arrays remain empty.  Mesh proof
verification opens tensors against the aggregate manifest root instead.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from verallm.mesh.types import canonical_json_bytes
from verallm.types import ModelSpec


# Keep this byte-identical to gguf_manifest._aggregate_model_file_sha256().
# The manifest's aggregate is already the canonical identity used throughout
# mesh state and proof receipts; ModelSpec must not invent a second sharded
# package hash for the same files.
GGUF_PACKAGE_HASH_DOMAIN = b"VERATHOS_GGUF_MODEL_FILE_SET_V1"
GGUF_TOKENIZER_HASH_DOMAIN = b"VERATHOS_GGUF_TOKENIZER_V1"


def _hex_digest(value: Any, *, field_name: str) -> str:
    digest = str(value or "").strip().lower()
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise ValueError(f"{field_name} must be a lowercase 32-byte hex digest")
    if digest == "0" * 64:
        raise ValueError(f"{field_name} must not be zero")
    return digest


def _json_value(value: Any) -> Any:
    """Normalize GGUF-reader values into deterministic JSON primitives."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("GGUF metadata contains a non-finite float")
        return value
    if isinstance(value, bytes):
        return {"bytes_hex": value.hex()}
    if isinstance(value, Mapping):
        return {str(key): _json_value(child) for key, child in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_value(child) for child in value]
    item = getattr(value, "item", None)
    if callable(item):
        return _json_value(item())
    raise ValueError(f"unsupported GGUF metadata value: {type(value).__name__}")


def _field_value(reader: Any, name: str, *, required: bool = True) -> Any:
    fields = getattr(reader, "fields", None)
    if not isinstance(fields, Mapping):
        raise ValueError("GGUF reader does not expose a fields mapping")
    field = fields.get(name)
    if field is None:
        if required:
            raise ValueError(f"GGUF metadata field is required: {name}")
        return None
    contents = getattr(field, "contents", None)
    return contents() if callable(contents) else field


def gguf_tokenizer_hash(reader: Any) -> bytes:
    """Commit every tokenizer-related field embedded in the GGUF header."""

    fields = getattr(reader, "fields", None)
    if not isinstance(fields, Mapping):
        raise ValueError("GGUF reader does not expose a fields mapping")
    names = sorted(str(name) for name in fields if str(name).startswith("tokenizer."))
    if not names or "tokenizer.ggml.tokens" not in names:
        raise ValueError("GGUF does not contain a committed tokenizer vocabulary")
    payload = {name: _json_value(_field_value(reader, name)) for name in names}
    return hashlib.sha256(
        GGUF_TOKENIZER_HASH_DOMAIN + canonical_json_bytes(payload)
    ).digest()


def gguf_package_hash(manifest: Mapping[str, Any]) -> bytes:
    """Return a path-independent package identity from manifest file hashes."""

    raw_files = manifest.get("model_files")
    files: list[dict[str, Any]] = []
    if isinstance(raw_files, Sequence) and not isinstance(
        raw_files, (str, bytes, bytearray)
    ):
        for position, raw in enumerate(raw_files):
            if not isinstance(raw, Mapping):
                raise ValueError("manifest model_files entries must be objects")
            index = raw.get("index", position)
            n_bytes = raw.get("n_bytes")
            if type(index) is not int or index < 0:
                raise ValueError("manifest model file index must be non-negative")
            if type(n_bytes) is not int or n_bytes <= 0:
                raise ValueError("manifest model file n_bytes must be positive")
            files.append(
                {
                    "index": index,
                    "n_bytes": n_bytes,
                    "sha256": _hex_digest(
                        raw.get("sha256"),
                        field_name="manifest model file sha256",
                    ),
                }
            )
    expected = _hex_digest(
        manifest.get("model_file_sha256"),
        field_name="manifest model_file_sha256",
    )
    if not files:
        return bytes.fromhex(expected)
    files.sort(key=lambda item: item["index"])
    if [item["index"] for item in files] != list(range(len(files))):
        raise ValueError("manifest model file indexes must be contiguous from zero")
    if len(files) == 1:
        if expected != files[0]["sha256"]:
            raise ValueError("manifest model file hashes disagree")
        return bytes.fromhex(expected)
    computed = hashlib.sha256(
        GGUF_PACKAGE_HASH_DOMAIN + canonical_json_bytes(files)
    ).hexdigest()
    if computed != expected:
        raise ValueError("manifest aggregate model file hash is invalid")
    return bytes.fromhex(expected)


def verify_gguf_model_files(manifest: Mapping[str, Any]) -> None:
    """Hash every local GGUF file and match its manifest size and SHA256."""

    raw_files = manifest.get("model_files")
    if isinstance(raw_files, Sequence) and not isinstance(
        raw_files, (str, bytes, bytearray)
    ) and raw_files:
        records = list(raw_files)
    else:
        records = [
            {
                "path": manifest.get("model_file"),
                "n_bytes": manifest.get("model_file_n_bytes"),
                "sha256": manifest.get("model_file_sha256"),
            }
        ]
    for raw in records:
        if not isinstance(raw, Mapping):
            raise ValueError("manifest model file entry must be an object")
        path = Path(str(raw.get("path") or "")).expanduser()
        if not path.is_file():
            raise ValueError(f"GGUF model file is unavailable: {path}")
        expected_size = raw.get("n_bytes")
        if expected_size is not None and (
            type(expected_size) is not int or path.stat().st_size != expected_size
        ):
            raise ValueError(f"GGUF model file size mismatch: {path}")
        expected_hash = _hex_digest(
            raw.get("sha256"),
            field_name="manifest model file sha256",
        )
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected_hash:
            raise ValueError(f"GGUF model file hash mismatch: {path}")


def _positive_int(reader: Any, name: str) -> int:
    value = _field_value(reader, name)
    if type(value) is not int or value <= 0:
        raise ValueError(f"GGUF metadata {name} must be a positive integer")
    return value


def _optional_nonnegative_int(reader: Any, name: str, *, default: int = 0) -> int:
    value = _field_value(reader, name, required=False)
    if value is None:
        return default
    if type(value) is not int or value < 0:
        raise ValueError(f"GGUF metadata {name} must be a non-negative integer")
    return value


def build_gguf_model_spec_from_manifest(
    manifest: Mapping[str, Any],
    reader: Any,
    *,
    model_id: str,
    gguf_scheme: str,
    activation: str = "",
    norm_type: str = "",
    attention_type: str = "",
) -> ModelSpec:
    """Build a chain-ready ``ModelSpec`` from validated manifest/header data."""

    if not isinstance(manifest, Mapping):
        raise ValueError("GGUF manifest must be an object")
    model_id = str(model_id or "").strip()
    if not model_id or len(model_id) > 255:
        raise ValueError("model_id must be between 1 and 255 characters")
    scheme = str(gguf_scheme or "").strip().upper()
    if (
        not scheme
        or len(scheme) > 64
        or any(ch not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_" for ch in scheme)
    ):
        raise ValueError("gguf_scheme must be a bounded GGUF quantization token")

    manifest_root = _hex_digest(
        manifest.get("tensor_manifest_root"),
        field_name="tensor_manifest_root",
    )
    architecture = str(_field_value(reader, "general.architecture") or "").strip()
    if not architecture:
        raise ValueError("GGUF general.architecture is empty")
    prefix = architecture + "."
    num_layers = _positive_int(reader, prefix + "block_count")
    # NextN/MTP blocks ride at the tail of the GGUF block stack but the
    # runtime excludes them from the forward pass (it builds
    # block_count - nextn_predict_layers layers). The registered layer
    # count anchors the verification layer universe and stage planning,
    # so it must match the COMPUTE stack - counting MTP blocks would
    # mint layer challenges that no honest serve can ever fulfil.
    nextn_raw = _field_value(
        reader, prefix + "nextn_predict_layers", required=False
    )
    if nextn_raw is not None:
        if type(nextn_raw) is not int or nextn_raw < 0 or nextn_raw >= num_layers:
            raise ValueError("GGUF nextn_predict_layers is invalid")
        num_layers -= nextn_raw
    hidden_dim = _positive_int(reader, prefix + "embedding_length")
    num_heads = _positive_int(reader, prefix + "attention.head_count")
    kv_heads_raw = _field_value(
        reader,
        prefix + "attention.head_count_kv",
        required=False,
    )
    if kv_heads_raw is not None and type(kv_heads_raw) is not int:
        raise ValueError("GGUF attention.head_count_kv must be an integer")
    kv_heads = num_heads if kv_heads_raw is None else kv_heads_raw
    if kv_heads <= 0 or kv_heads > num_heads:
        raise ValueError("GGUF attention.head_count_kv is invalid")
    # Modern architectures can project attention heads wider than
    # hidden_dim / num_heads (Qwen3.6 is one example).  GGUF's key_length is
    # the architectural head width; rope.dimension_count is only the rotary
    # sub-dimension and must not be substituted for it.
    head_dim_raw = _field_value(
        reader,
        prefix + "attention.key_length",
        required=False,
    )
    if head_dim_raw is not None and type(head_dim_raw) is not int:
        raise ValueError("GGUF attention.key_length must be an integer")
    head_dim = head_dim_raw if head_dim_raw is not None else hidden_dim // num_heads
    if head_dim <= 0:
        raise ValueError("GGUF attention.key_length is invalid")
    value_dim_raw = _field_value(
        reader,
        prefix + "attention.value_length",
        required=False,
    )
    if value_dim_raw is not None and (
        type(value_dim_raw) is not int or value_dim_raw <= 0
    ):
        raise ValueError("GGUF attention.value_length must be a positive integer")
    rope_dim_raw = _field_value(
        reader,
        prefix + "rope.dimension_count",
        required=False,
    )
    if rope_dim_raw is not None and (
        type(rope_dim_raw) is not int
        or rope_dim_raw <= 0
        or rope_dim_raw > head_dim
    ):
        raise ValueError("GGUF rope.dimension_count is invalid")

    num_experts = _optional_nonnegative_int(reader, prefix + "expert_count")
    intermediate_name = (
        prefix + "expert_feed_forward_length"
        if num_experts
        else prefix + "feed_forward_length"
    )
    intermediate_dim = _positive_int(reader, intermediate_name)
    token_values = _field_value(
        reader,
        "tokenizer.ggml.tokens",
        required=False,
    )
    if not isinstance(token_values, Sequence) or isinstance(
        token_values, (str, bytes, bytearray)
    ):
        raise ValueError("GGUF tokenizer vocabulary must be an array")
    vocab_size = len(token_values)
    if vocab_size <= 0:
        raise ValueError("GGUF tokenizer vocabulary is empty")

    activation_value = str(activation or "").strip().lower()
    if not activation_value:
        if architecture.lower() in {
            "llama",
            "mistral",
            "qwen",
            "qwen2",
            "qwen2moe",
            "qwen35",
            "qwen35moe",
            "deepseek4",
        }:
            activation_value = "silu"
        else:
            raise ValueError(
                f"activation is required for unsupported GGUF architecture {architecture}"
            )
    if activation_value not in {"silu", "gelu", "relu"}:
        raise ValueError("activation must be silu, gelu, or relu")

    norm_value = str(norm_type or "").strip().lower()
    if not norm_value:
        has_rms = any(
            str(name).endswith("layer_norm_rms_epsilon")
            for name in getattr(reader, "fields", {})
        )
        norm_value = "rmsnorm" if has_rms else "layernorm"
    if norm_value not in {"rmsnorm", "layernorm"}:
        raise ValueError("norm_type must be rmsnorm or layernorm")

    attention_value = str(attention_type or "").strip().lower()
    if not attention_value:
        attention_value = "mha" if kv_heads == num_heads else "mqa" if kv_heads == 1 else "gqa"
    if attention_value not in {"mha", "gqa", "mqa"}:
        raise ValueError("attention_type must be mha, gqa, or mqa")

    router_top_k = _optional_nonnegative_int(
        reader,
        prefix + "expert_used_count",
    )
    if (not num_experts and router_top_k) or (
        num_experts and (router_top_k <= 0 or router_top_k > num_experts)
    ):
        raise ValueError("GGUF expert_used_count is invalid")

    # The chain anchor must be recomputable by every validator from a few-MB
    # tokenizer download; a GGUF-header hash would require the full weights.
    from verallm.registry.models import mesh_tokenizer_source
    from verallm.registry.tokenizer_hash import compute_tokenizer_hash

    if mesh_tokenizer_source(model_id) is None:
        raise ValueError(
            f"{model_id!r} has no tokenizer source in MESH_GGUF_MODELS; add "
            "the family (with tokenizer_hf_repo) before building its chain spec"
        )
    tokenizer_hash = compute_tokenizer_hash(model_id)

    spec = ModelSpec(
        model_id=model_id,
        weight_merkle_root=bytes.fromhex(manifest_root),
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        head_dim=head_dim,
        intermediate_dim=intermediate_dim,
        vocab_size=vocab_size,
        activation=activation_value,
        norm_type=norm_value,
        attention_type=attention_value,
        weight_block_merkle_roots=[],
        w_merkle_chunk_size=128,
        quant_mode="gguf_" + scheme.lower(),
        expert_w_num_cols=intermediate_dim if num_experts else 0,
        num_experts=num_experts,
        router_top_k=router_top_k,
        router_scoring="softmax",
        weight_file_hash=gguf_package_hash(manifest),
        tokenizer_hash=tokenizer_hash,
    )
    return spec


def build_gguf_model_spec(
    manifest_path: str | Path,
    *,
    model_id: str,
    gguf_scheme: str,
    activation: str = "",
    norm_type: str = "",
    attention_type: str = "",
    verify_model_files: bool = True,
    reader_factory: Callable[[Path], Any] | None = None,
) -> ModelSpec:
    """Load a GGUF manifest/header and produce a chain-ready ``ModelSpec``."""

    from verallm.mesh.gguf_manifest import load_gguf_tensor_manifest

    path = Path(manifest_path).expanduser()
    manifest = load_gguf_tensor_manifest(path)
    if verify_model_files:
        verify_gguf_model_files(manifest)
    model_file = Path(str(manifest.get("model_file") or "")).expanduser()
    if not model_file.is_file():
        raise ValueError(f"GGUF model file is unavailable: {model_file}")
    if reader_factory is None:
        try:
            import gguf
        except Exception as exc:  # pragma: no cover - dependency failure
            raise RuntimeError("building a GGUF ModelSpec requires the gguf package") from exc
        reader_factory = gguf.GGUFReader
    reader = reader_factory(model_file)
    return build_gguf_model_spec_from_manifest(
        manifest,
        reader,
        model_id=model_id,
        gguf_scheme=gguf_scheme,
        activation=activation,
        norm_type=norm_type,
        attention_type=attention_type,
    )


def model_spec_summary(spec: ModelSpec) -> dict[str, Any]:
    """Return a JSON-safe registration summary without local file paths."""

    return {
        "model_id": spec.model_id,
        "weight_merkle_root": spec.weight_merkle_root.hex(),
        "weight_file_hash": spec.weight_file_hash.hex(),
        "tokenizer_hash": spec.tokenizer_hash.hex(),
        "num_layers": int(spec.num_layers),
        "hidden_dim": int(spec.hidden_dim),
        "intermediate_dim": int(spec.intermediate_dim),
        "num_heads": int(spec.num_heads),
        "head_dim": int(spec.head_dim),
        "vocab_size": int(spec.vocab_size),
        "quant_mode": spec.quant_mode,
        "activation": spec.activation,
        "norm_type": spec.norm_type,
        "attention_type": spec.attention_type,
        "num_experts": int(spec.num_experts),
        "expert_w_num_cols": int(spec.expert_w_num_cols),
        "router_top_k": int(spec.router_top_k),
        "layer_root_count": len(spec.weight_block_merkle_roots),
    }


__all__ = [
    "GGUF_PACKAGE_HASH_DOMAIN",
    "GGUF_TOKENIZER_HASH_DOMAIN",
    "build_gguf_model_spec",
    "build_gguf_model_spec_from_manifest",
    "gguf_package_hash",
    "gguf_tokenizer_hash",
    "model_spec_summary",
    "verify_gguf_model_files",
]
