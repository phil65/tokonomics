"""Utilities for calculating token costs using LiteLLM pricing data."""

import asyncio
from dataclasses import dataclass
from decimal import Decimal
import os
import re
import threading
import time
from typing import Any, cast
import weakref

from anyenv import get_json

from tokonomics import log
from tokonomics.helpers import _is_numeric, _safe_numeric_convert
from tokonomics.toko_types import ModelCapabilities, ModelCosts, ModelCostTier, TokenCosts


logger = log.get_logger(__name__)


_CACHE_TIMEOUT = 86400
"""Seconds the price table is kept (in process and in the HTTP cache): 24 hours."""

LITELLM_PRICES_URL = "https://raw.githubusercontent.com/BerriAI/litellm/refs/heads/main/model_prices_and_context_window.json"

# Tier keys like "input_cost_per_token_above_200k_tokens" (not the _priority/_flex variants).
_TIER_RE = re.compile(
    r"^(input_cost_per_token|output_cost_per_token|cache_read_input_token_cost"
    r"|cache_creation_input_token_cost)_above_(\d+)k_tokens$"
)


@dataclass(slots=True)
class _PriceTable:
    loaded_at: float
    """``_now()`` (monotonic clock) of the download."""
    models: dict[str, dict[str, Any]]
    """Lower-cased model name -> LiteLLM entry."""


_now = time.monotonic
_table: _PriceTable | None = None
_inflight: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Task[_PriceTable]] = (
    weakref.WeakKeyDictionary()
)
_lock = threading.Lock()  # guards _table / _inflight (free-threading, several loops)


def _is_testing() -> bool:
    """Check if we're running in a test environment."""
    return bool(os.getenv("PYTEST_CURRENT_TEST"))


def reset_cache() -> None:
    """Clear all cached pricing and limit data."""
    global _table
    with _lock:
        _table = None


async def _download_table(cache_timeout: int) -> _PriceTable:
    logger.debug("Downloading pricing data from LiteLLM...")
    data = await get_json(
        LITELLM_PRICES_URL,
        cache=not _is_testing(),
        cache_ttl=cache_timeout,
        return_type=dict,
    )
    if not isinstance(data, dict):
        msg = f"Expected dict, got {type(data)}"
        raise TypeError(msg)
    models = {name.lower(): info for name, info in data.items() if isinstance(info, dict)}
    logger.debug("Loaded LiteLLM data for %d models", len(models))
    return _PriceTable(loaded_at=_now(), models=models)


async def _get_table(cache_timeout: int) -> _PriceTable:
    """Return the LiteLLM table, downloading it once per ``cache_timeout``.

    Concurrent callers on one loop share a single download. If a refresh fails while an
    older table exists, the old one is kept (and retried on the next call).
    """
    global _table
    table = _table
    if table is not None and _now() - table.loaded_at < cache_timeout:
        return table
    loop = asyncio.get_running_loop()
    with _lock:
        task = _inflight.get(loop)
        if task is None:
            task = _inflight[loop] = loop.create_task(_download_table(cache_timeout))
            task.add_done_callback(lambda _: _inflight.pop(loop, None))
    try:
        new_table = await asyncio.shield(task)
    except Exception:
        if table is None:
            raise
        logger.warning("Refreshing LiteLLM prices failed, keeping the old table", exc_info=True)
        return table
    with _lock:
        _table = new_table
    return new_table


def _name_candidates(model: str) -> list[str]:
    """Names to try for ``model``: as is, without ``provider:``, as ``provider/model``."""
    name = model.lower()
    if ":" not in name:
        return [name]
    provider, model_name = name.split(":", 1)
    return [name, model_name, f"{provider}/{model_name}"]


def _find_entry(table: _PriceTable, model: str) -> dict[str, Any] | None:
    for name in _name_candidates(model):
        if (info := table.models.get(name)) is not None:
            return info
    return None


def _decimal(value: Any) -> Decimal | None:
    return Decimal(str(value)) if _is_numeric(value) else None


def _parse_costs(info: dict[str, Any]) -> ModelCosts | None:
    input_cost = _decimal(info.get("input_cost_per_token"))
    output_cost = _decimal(info.get("output_cost_per_token"))
    if input_cost is None or output_cost is None:
        return None
    costs = ModelCosts(input_cost_per_token=input_cost, output_cost_per_token=output_cost)
    if (read := _decimal(info.get("cache_read_input_token_cost"))) is not None:
        costs["cache_read_input_token_cost"] = read
    if (write := _decimal(info.get("cache_creation_input_token_cost"))) is not None:
        costs["cache_creation_input_token_cost"] = write
    tiers: dict[int, ModelCostTier] = {}
    for key, value in info.items():
        if (match := _TIER_RE.match(key)) and (price := _decimal(value)) is not None:
            tier = tiers.setdefault(int(match[2]) * 1000, ModelCostTier())
            cast("dict[str, Decimal]", tier)[match[1]] = price
    if tiers:
        costs["tiers"] = dict(sorted(tiers.items()))
    return costs


async def get_model_costs(
    model: str,
    *,
    cache_timeout: int = _CACHE_TIMEOUT,
) -> ModelCosts | None:
    """Get cost information for a model from LiteLLM pricing data.

    The LiteLLM table is downloaded once and kept for ``cache_timeout`` seconds; lookups
    of unknown models don't download it again.

    Args:
        model: Name of the model to look up costs for
        cache_timeout: Number of seconds to keep prices in cache (default: 24 hours)

    Returns:
        ModelCosts | None: Model's cost information if found, None otherwise
    """
    try:
        table = await _get_table(cache_timeout)
    except Exception as e:  # noqa: BLE001
        logger.debug("Failed to get model costs for %r: %s", model, e)
        return None
    info = _find_entry(table, model)
    return _parse_costs(info) if info is not None else None


def _tier_prices(costs: ModelCosts, prompt_tokens: int) -> ModelCostTier:
    """Prices for a request with ``prompt_tokens`` input tokens.

    Like LiteLLM, a request above a tier's threshold is billed at that tier's prices
    for all its tokens; prices a tier doesn't list fall back to the base prices.
    """
    prices = ModelCostTier(
        input_cost_per_token=costs["input_cost_per_token"],
        output_cost_per_token=costs["output_cost_per_token"],
    )
    for key in ("cache_read_input_token_cost", "cache_creation_input_token_cost"):
        if key in costs:
            prices[key] = costs[key]
    for threshold, tier in costs.get("tiers", {}).items():
        if prompt_tokens > threshold:
            prices.update(tier)
    return prices


async def calculate_token_cost(
    model: str,
    input_tokens: int | None,
    output_tokens: int | None,
    *,
    cache_read_tokens: int | None = None,
    cache_write_tokens: int | None = None,
    cache_timeout: int = _CACHE_TIMEOUT,
) -> TokenCosts | None:
    """Calculate detailed costs for token usage based on model pricing.

    Combines input and output token counts with their respective costs to
    calculate the breakdown of costs. If a token count is None, it will
    be treated as 0 tokens.

    Args:
        model: Name of the model used (e.g. "gpt-4", "openai:gpt-3.5-turbo")
        input_tokens: Number of tokens in the prompt/input, or None. Includes the
            cache read and cache write tokens (as pydantic-ai counts them).
        output_tokens: Number of tokens in the completion/output, or None
        cache_read_tokens: Input tokens read from the prompt cache, billed at the
            model's cache read price (the input price if it has none)
        cache_write_tokens: Input tokens written to the prompt cache, billed at the
            model's cache creation price (the input price if it has none)
        cache_timeout: Number of seconds to keep prices in cache (default: 24 hours)

    Returns:
        TokenCosts | None: Detailed cost breakdown if pricing data available.
            ``input_cost`` includes the cache read and write costs.
    """
    costs = await get_model_costs(model, cache_timeout=cache_timeout)
    if not costs:
        logger.debug("No costs found for model %r", model)
        return None

    input_count = input_tokens or 0
    output_count = output_tokens or 0
    read_count = cache_read_tokens or 0
    write_count = cache_write_tokens or 0
    prices = _tier_prices(costs, input_count)
    input_price = prices["input_cost_per_token"]
    uncached = max(input_count - read_count - write_count, 0)
    input_cost = (
        uncached * input_price
        + read_count * prices.get("cache_read_input_token_cost", input_price)
        + write_count * prices.get("cache_creation_input_token_cost", input_price)
    )
    output_cost = output_count * prices["output_cost_per_token"]

    token_costs = TokenCosts(input_cost=input_cost, output_cost=output_cost)

    logger.debug(
        "Cost calculation - prompt: $%.6f, completion: $%.6f, total: $%.6f",
        token_costs.input_cost,
        token_costs.output_cost,
        token_costs.total_cost,
    )
    return token_costs


async def get_available_models(
    *,
    cache_timeout: int = _CACHE_TIMEOUT,
) -> list[str]:
    """Get a list of all available model names from LiteLLM pricing data.

    Args:
        cache_timeout: Number of seconds to keep data in cache (default: 24 hours)

    Returns:
        list[str]: List of available model names, sorted alphabetically
    """
    try:
        table = await _get_table(cache_timeout)
    except Exception as e:
        error_msg = "Failed to get available models"
        logger.exception(error_msg)
        raise ValueError(error_msg) from e
    return sorted(table.models)


def _parse_capabilities(info: dict[str, Any]) -> ModelCapabilities:
    # LiteLLM dropped the legacy "max_tokens" field (it mirrored the output limit).
    max_tokens_raw = info.get("max_tokens", info.get("max_output_tokens", 0))
    max_input_raw = info.get("max_input_tokens", max_tokens_raw)
    max_output_raw = info.get("max_output_tokens", max_tokens_raw)

    # Convert token values to integers, defaulting to 0
    try:
        max_tokens = int(_safe_numeric_convert(max_tokens_raw))
        max_input = int(_safe_numeric_convert(max_input_raw))
        max_output = int(_safe_numeric_convert(max_output_raw))
    except ValueError, TypeError:
        max_tokens = max_input = max_output = 0

    return ModelCapabilities(
        max_tokens=max_tokens,
        max_input_tokens=max_input,
        max_output_tokens=max_output,
        litellm_provider=info.get("litellm_provider"),
        mode=info.get("mode"),
        supports_function_calling=bool(info.get("supports_function_calling")),
        supports_parallel_function_calling=bool(info.get("supports_parallel_function_calling")),
        supports_vision=bool(info.get("supports_vision")),
        supports_audio_input=bool(info.get("supports_audio_input")),
        supports_audio_output=bool(info.get("supports_audio_output")),
        supports_prompt_caching=bool(info.get("supports_prompt_caching")),
        supports_response_schema=bool(info.get("supports_response_schema")),
        supports_system_messages=bool(info.get("supports_system_messages")),
    )


async def get_model_capabilities(
    model: str,
    *,
    cache_timeout: int = _CACHE_TIMEOUT,
) -> ModelCapabilities | None:
    """Get capabilities information for a model from LiteLLM data.

    Args:
        model: Name of the model to look up capabilities for
        cache_timeout: Number of seconds to keep data in cache (default: 24 hours)

    Returns:
        ModelCapabilities | None: Model's capabilities if found, None otherwise

    Raises:
        ValueError: If there's an error fetching the capabilities data
    """
    try:
        table = await _get_table(cache_timeout)
    except Exception as e:
        error_msg = f"Failed to get model capabilities for {model}: {e}"
        logger.exception(error_msg)
        raise ValueError(error_msg) from e
    info = _find_entry(table, model)
    return _parse_capabilities(info) if info is not None else None


if __name__ == "__main__":
    import asyncio

    async def main() -> None:
        # Example usage
        model_name = "openrouter:openai/gpt-5-nano"
        limits = await get_model_costs(model_name)
        print(limits)

    asyncio.run(main())
