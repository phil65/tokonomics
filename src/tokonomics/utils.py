"""Utilities for calculating token costs using LiteLLM pricing data."""

from __future__ import annotations

import logging
import pathlib
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    import httpx


logger = logging.getLogger(__name__)

# Cache timeout in seconds (24 hours)
_CACHE_TIMEOUT = 86400

# Cache directory
CACHE_DIR = pathlib.Path("~/.cache/tokonomics/litellm").expanduser()
CACHE_DIR.mkdir(parents=True, exist_ok=True)


_TESTING = False  # Flag to disable caching during tests


async def make_request(url: str) -> httpx.Response:
    """Make an HTTP request with caching."""
    import hishel
    import httpx

    if _TESTING:
        # During tests, use a simple client without caching
        async with httpx.AsyncClient() as client:
            return await client.get(url)

    # In production, use hishel caching
    storage = hishel.AsyncFileStorage(
        base_path=CACHE_DIR,
        ttl=_CACHE_TIMEOUT,
    )
    controller = hishel.Controller(
        cacheable_methods=["GET"],
        cacheable_status_codes=[200],
        allow_stale=True,
    )
    transport = hishel.AsyncCacheTransport(
        transport=httpx.AsyncHTTPTransport(),
        storage=storage,
        controller=controller,
    )
    async with httpx.AsyncClient(transport=transport) as client:  # type: ignore[arg-type]
        return await client.get(url)
