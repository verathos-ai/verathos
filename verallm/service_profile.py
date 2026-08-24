"""Service-bound request restrictions for qualified inference profiles."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


class UnsupportedModalityError(ValueError):
    """A request contains input outside the admitted service modality."""


def normalize_text_only_chat_messages(
    messages: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Copy chat messages, flatten text parts, and reject every other part."""

    normalized: list[dict[str, Any]] = []
    for message_index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            raise UnsupportedModalityError(f"message {message_index} is not an object")
        item = dict(message)
        content = item.get("content", "")
        if content is None or isinstance(content, str):
            normalized.append(item)
            continue
        if not isinstance(content, list):
            raise UnsupportedModalityError(
                f"message {message_index} content is not text"
            )

        text_parts: list[str] = []
        for part_index, part in enumerate(content):
            if not isinstance(part, Mapping) or part.get("type") != "text":
                raise UnsupportedModalityError(
                    "text-only model profile rejects non-text content at "
                    f"message {message_index} part {part_index}"
                )
            text = part.get("text")
            if not isinstance(text, str):
                raise UnsupportedModalityError(
                    "text-only model profile requires string text at "
                    f"message {message_index} part {part_index}"
                )
            text_parts.append(text)
        item["content"] = "".join(text_parts)
        normalized.append(item)
    return normalized


__all__ = ["UnsupportedModalityError", "normalize_text_only_chat_messages"]
