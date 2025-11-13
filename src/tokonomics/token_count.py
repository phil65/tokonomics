"""Token counting utilities with multiple fallback strategies."""

from __future__ import annotations

from functools import lru_cache
from importlib.util import find_spec
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Sequence


@lru_cache
def has_library(name: str) -> bool:
    """Check if a Python package is available."""
    return bool(find_spec(name))


def count_tokens(text: str | Sequence[str], model: str | None = None) -> int:
    """Count total number of tokens in text(s) with fallback strategies.

    Uses following fallback order:
    1. tiktoken (if available)
    2. transformers (if available)
    3. rough approximation

    Args:
        text: Single text or sequence of texts to count tokens for
        model: Optional model name for tiktoken (ignored in fallbacks)

    Returns:
        Total token count for all provided text(s)
    """
    # Try tiktoken first
    if has_library("tiktoken"):
        import tiktoken  # pyright: ignore[reportMissingImports]

        try:
            encoding = tiktoken.encoding_for_model(model or "gpt-3.5-turbo")
        except KeyError:
            encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

        if isinstance(text, str):
            return len(encoding.encode(text))

        return sum(len(encoding.encode(t)) for t in text)

    # Try transformers as fallback
    if has_library("transformers"):
        from transformers import AutoTokenizer  # pyright: ignore[reportMissingImports]

        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")

        if isinstance(text, str):
            return len(tokenizer.encode(text))

        return sum(len(tokenizer.encode(t)) for t in text)

    # Last resort: rough approximation
    if isinstance(text, str):
        return len(text.split()) + len(text) // 4

    return sum(len(t.split()) + len(t) // 4 for t in text)
