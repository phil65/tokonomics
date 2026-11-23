"""In-process cache of the model lists providers return.

A provider's response is cached on disk by the HTTP layer (``MODELS_CACHE_TTL``), but
parsing it is the expensive part: models.dev lists about 10k models, a few hundred ms
to parse. The parsed, unfiltered list is kept here for the same time, per provider and
request headers (a different API key may see different models), so repeated discovery
costs only the filtering.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from tokonomics.model_discovery.base import ModelProvider
    from tokonomics.model_discovery.model_info import ModelInfo


MODELS_CACHE_TTL = 86400
"""Seconds a provider's model list is reused, on disk and in memory (24 hours)."""

type _Key = tuple[str, tuple[tuple[str, str], ...]]

_parsed: dict[_Key, tuple[float, list[ModelInfo]]] = {}


def _key(name: str, provider: ModelProvider) -> _Key:
    return name, tuple(sorted(provider.headers.items()))


def get_cached(name: str, provider: ModelProvider) -> list[ModelInfo] | None:
    """The provider's parsed model list, if fetched within ``MODELS_CACHE_TTL``."""
    entry = _parsed.get(_key(name, provider))
    if entry is None or time.monotonic() - entry[0] > MODELS_CACHE_TTL:
        return None
    return entry[1]


def store(name: str, provider: ModelProvider, models: list[ModelInfo]) -> None:
    """Keep the provider's parsed model list (an empty one is not kept: a failed fetch)."""
    if models:
        _parsed[_key(name, provider)] = (time.monotonic(), models)


def clear_models_cache() -> None:
    """Drop the parsed model lists, so the next discovery parses again."""
    _parsed.clear()
