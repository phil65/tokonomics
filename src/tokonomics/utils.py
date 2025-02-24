"""Utilities for calculating token costs using LiteLLM pricing data."""

from __future__ import annotations

import json
import logging
import pathlib
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    import httpx
    from respx.types import HeaderTypes


logger = logging.getLogger(__name__)

# Cache timeout in seconds (24 hours)
_CACHE_TIMEOUT = 86400

# Cache directory
CACHE_DIR = pathlib.Path("~/.cache/tokonomics/litellm").expanduser()
CACHE_DIR.mkdir(parents=True, exist_ok=True)


_TESTING = False  # Flag to disable caching during tests


def parse_json(data: str | bytes) -> Any:
    """Parse JSON data using the fastest available parser."""
    try:
        import orjson

        # orjson only accepts bytes or str and returns bytes
        if isinstance(data, str):
            data = data.encode()
        return orjson.loads(data)
    except ImportError:
        return json.loads(data)


class DownloadError(Exception):
    """Raised when a download fails."""


async def download_json(
    url: str,
    *,
    params: dict[str, Any] | None = None,
    headers: HeaderTypes | None = None,
    timeout: float = 10.0,
) -> Any:
    """Download and parse JSON from a URL.

    Args:
        url: URL to download from
        params: Optional query parameters
        headers: Optional HTTP headers
        timeout: Timeout in seconds

    Returns:
        Parsed JSON data

    Raises:
        DownloadError: If download or parsing fails
    """
    import httpx

    try:
        response = await make_request(url, params=params, headers=headers)
        response.raise_for_status()

        # Try to parse the response
        try:
            return parse_json(response.content)
        except (json.JSONDecodeError, ValueError) as e:
            msg = f"Invalid JSON response from {url}: {e}"
            logger.exception(msg)
            # Log a snippet of the response for debugging
            content_preview = response.text[:200]
            logger.debug("Response preview: %s...", content_preview)
            raise DownloadError(msg) from e

    except httpx.TimeoutException as e:
        msg = f"Timeout while downloading from {url}"
        logger.exception(msg)
        raise DownloadError(msg) from e
    except httpx.HTTPError as e:
        msg = f"HTTP error while downloading from {url}: {e}"
        logger.exception(msg)
        raise DownloadError(msg) from e


async def make_request(
    url: str,
    params: dict[str, Any] | None = None,
    headers: HeaderTypes | None = None,
) -> httpx.Response:
    """Make an HTTP request with caching."""
    import hishel
    import httpx

    if _TESTING:
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
        response = await client.get(url, params=params, headers=headers)
        logger.debug(
            "Response from %s - Status: %d, Content-Length: %s",
            url,
            response.status_code,
            response.headers.get("content-length"),
        )
        return response
