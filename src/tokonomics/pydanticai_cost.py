"""Integration with pydantic-ai cost calculations."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol, overload, runtime_checkable

from tokonomics.core import calculate_token_cost


if TYPE_CHECKING:
    from tokonomics.toko_types import TokenCosts


logger = logging.getLogger(__name__)


@runtime_checkable
class Usage(Protocol):
    """Protocol matching pydantic-ai's Usage object structure."""

    total_tokens: int | None
    """Total number of tokens used"""
    request_tokens: int | None
    """Number of tokens in the request/prompt"""
    response_tokens: int | None
    """Number of tokens in the response/completion"""


@overload
async def calculate_pydantic_cost(
    model: str,
    usage: None,
    *,
    cache_timeout: int = 86400,
) -> None: ...


@overload
async def calculate_pydantic_cost(
    model: str,
    usage: Usage,
    *,
    cache_timeout: int = 86400,
) -> TokenCosts | None: ...


async def calculate_pydantic_cost(
    model: str,
    usage: Usage | None,
    *,
    cache_timeout: int = 86400,
) -> TokenCosts | None:
    """Calculate costs from a pydantic-ai Usage object.

    Args:
        model: Name of the model used (e.g. "gpt-4", "openai:gpt-3.5-turbo")
        usage: Token usage information from pydantic-ai
        cache_timeout: Number of seconds to keep prices in cache (default: 24 hours)

    Returns:
        TokenCosts | None: Detailed cost breakdown if pricing data available
    """
    if not usage:
        logger.debug("No usage information provided")
        return None

    return await calculate_token_cost(
        model=model,
        prompt_tokens=usage.request_tokens,
        completion_tokens=usage.response_tokens,
        cache_timeout=cache_timeout,
    )
