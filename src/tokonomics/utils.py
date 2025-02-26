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
    except (json.JSONDecodeError, ValueError) as e:
        msg = f"Invalid JSON response from {url}: {e}"
        logger.exception(msg)
        # Log a snippet of the response for debugging
        raise DownloadError(msg) from e

    except httpx.TimeoutException as e:
        msg = f"Timeout while downloading from {url}"
        logger.exception(msg)
        raise DownloadError(msg) from e
    except httpx.HTTPError as e:
        msg = f"HTTP error while downloading from {url}: {e}"
        logger.exception(msg)
        raise DownloadError(msg) from e

    try:
        return parse_json(response.content)
    except (json.JSONDecodeError, ValueError) as e:
        msg = f"Invalid JSON response from {url}: {e}"
        logger.exception(msg)
        # Log a snippet of the response for debugging
        content_preview = response.text[:200]
        logger.debug("Response preview: %s...", content_preview)
        raise DownloadError(msg) from e


async def make_request(
    url: str,
    params: dict[str, Any] | None = None,
    headers: HeaderTypes | None = None,
) -> httpx.Response:
    """Make an HTTP request with caching."""
    import hishel
    import httpx

    # Add standard headers if none provided
    if headers is None:
        headers = {
            b"User-Agent": b"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",  # noqa: E501
            b"Accept": b"application/json,*/*",
        }

    if _TESTING:
        async with httpx.AsyncClient() as client:
            return await client.get(url, params=params, headers=headers)

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
        return await client.get(
            url, params=params, headers=headers, follow_redirects=True
        )


def make_request_sync(
    url: str,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    """Make a synchronous HTTP request with caching."""
    import hishel
    import httpx

    storage = hishel.FileStorage(
        base_path=CACHE_DIR,
        ttl=86400,  # 24 hours
    )
    controller = hishel.Controller(
        cacheable_methods=["GET"],
        cacheable_status_codes=[200],
        allow_stale=True,
    )
    transport = hishel.CacheTransport(
        transport=httpx.HTTPTransport(),
        storage=storage,
        controller=controller,
    )
    with httpx.Client(transport=transport) as client:  # type: ignore[arg-type]
        return client.get(url, params=params, headers=headers, follow_redirects=True)
