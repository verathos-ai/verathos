"""Compute the canonical tokenizer hash for a model.

The hash is anchored on-chain in ``ModelRegistry.ModelSpec.tokenizerHash``
so validators can detect drift between their local tokenizer files and
what the subnet owner registered.

The hash binds:
- ``tokenizer.json`` raw bytes (vocab, merges, special tokens, pre/post-processors)
- ``chat_template`` field from ``tokenizer_config.json`` (UTF-8)

Per-request commitments do NOT include this hash — token-level
correctness is enforced separately via ``input_commitment`` (the validator
independently re-tokenizes the messages and compares).  This hash is
purely a startup-time tripwire that catches the case where the
validator's local tokenizer files have drifted from canonical.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def compute_tokenizer_hash(model_id_or_path: str) -> bytes:
    """Compute a deterministic 32-byte hash of a model's tokenizer files.

    Hash format::

        SHA256("VERILLM_TOKENIZER_V1" || tokenizer.json_bytes || 0x00 || chat_template_utf8)

    The chat_template is read from ``tokenizer_config.json`` (or ``chat_template.jinja``
    if present).  Models without a chat_template default to empty string.

    Args:
        model_id_or_path: HF model ID (e.g. "Qwen/Qwen3.5-9B") or local
            directory path containing tokenizer.json and tokenizer_config.json.

    Returns:
        32-byte SHA256 digest.

    Raises:
        FileNotFoundError: if tokenizer.json cannot be located on disk.
        Exception: anything else (network failure, parse error) — fail
            closed so the caller knows there's a problem.
    """
    src = _resolve_tokenizer_dir(model_id_or_path)

    tokenizer_json = src / "tokenizer.json"
    tokenizer_config = src / "tokenizer_config.json"
    chat_template_jinja = src / "chat_template.jinja"

    if not tokenizer_json.exists():
        raise FileNotFoundError(
            f"tokenizer.json not found at {tokenizer_json} "
            f"for model {model_id_or_path}"
        )

    raw_tokenizer = tokenizer_json.read_bytes()

    chat_template = ""
    # Newer HF tokenizers may store the chat template as a separate
    # chat_template.jinja file rather than inside tokenizer_config.json.
    if chat_template_jinja.exists():
        chat_template = chat_template_jinja.read_text(encoding="utf-8")
    elif tokenizer_config.exists():
        cfg = json.loads(tokenizer_config.read_text(encoding="utf-8"))
        ct = cfg.get("chat_template", "") or ""
        if isinstance(ct, str):
            chat_template = ct
        elif isinstance(ct, list):
            # Some tokenizers expose multiple templates as a list of
            # {name, template} dicts.  Concatenate deterministically.
            parts = []
            for entry in ct:
                if isinstance(entry, dict):
                    parts.append(entry.get("name", ""))
                    parts.append(entry.get("template", ""))
            chat_template = "\x1e".join(parts)
        elif isinstance(ct, dict):
            parts = []
            for key in sorted(ct.keys()):
                parts.append(key)
                parts.append(ct[key] or "")
            chat_template = "\x1e".join(parts)

    h = hashlib.sha256(b"VERILLM_TOKENIZER_V1")
    h.update(raw_tokenizer)
    h.update(b"\x00")
    h.update(chat_template.encode("utf-8"))
    return h.digest()


def _resolve_tokenizer_dir(model_id_or_path: str) -> Path:
    """Resolve a model identifier to the directory containing its tokenizer files.

    Order of resolution:
    1. If the argument is an existing local directory, use it directly.
    2. Try the HuggingFace local cache (no network access).
    3. Fall back to ``snapshot_download`` with a strict ``allow_patterns``
       so we only fetch the small tokenizer files, NOT the multi-GB weights.
    """
    p = Path(model_id_or_path)
    if p.is_dir():
        return p

    # Use snapshot_download directly with allow_patterns to bound network use.
    # Setting local_files_only=True first to avoid any network call when cached.
    from huggingface_hub import snapshot_download

    allow = [
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "special_tokens_map.json",
        "added_tokens.json",
        # Legacy tokenizers (BPE merges + vocab):
        "vocab.json",
        "merges.txt",
    ]

    try:
        path = snapshot_download(
            model_id_or_path,
            allow_patterns=allow,
            local_files_only=True,
        )
    except Exception:
        # Not in cache — download only the tokenizer-related files.
        path = snapshot_download(
            model_id_or_path,
            allow_patterns=allow,
        )
    return Path(path)
