"""Utilities for calculating token costs using LiteLLM pricing data."""

from __future__ import annotations

import logging
import pathlib
from typing import cast

import diskcache
import httpx
from platformdirs import user_data_dir

from tokonomics.toko_types import ModelCosts, TokenCosts, TokenLimits


logger = logging.getLogger(__name__)


# Cache cost data persistently
PRICING_DIR = pathlib.Path(user_data_dir("tokonomics", "tokonomics")) / "pricing"
PRICING_DIR.mkdir(parents=True, exist_ok=True)
_cost_cache = diskcache.Cache(directory=str(PRICING_DIR))

# Cache timeout in seconds (24 hours)
_CACHE_TIMEOUT = 86400

LITELLM_PRICES_URL = "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"


def find_litellm_model_name(model: str) -> str | None:
    """Find matching model name in LiteLLM pricing data.

    Attempts to match the input model name against cached LiteLLM pricing data
    by trying different formats (direct match, base name, provider format).

    Args:
        model: Input model name (e.g. "openai:gpt-4", "gpt-4")

    Returns:
        str | None: Matching LiteLLM model name if found in cache, None otherwise
    """
    logger.debug("Looking up model costs for: %s", model)

    # Normalize case
    model = model.lower()

    # First check direct match
    if model in _cost_cache:
        logger.debug("Found direct cache match for: %s", model)
        return model

    # For provider:model format, try both variants
    if ":" in model:
        provider, model_name = model.split(":", 1)
        # Try just model name (normalized)
        model_name = model_name.lower()
        if _cost_cache.get(model_name, None) is not None:
            logger.debug("Found cache match for base name: %s", model_name)
            return model_name
        # Try provider/model format (normalized)
        provider_format = f"{provider.lower()}/{model_name}"
        if _cost_cache.get(provider_format, None) is not None:
            logger.debug("Found cache match for provider format: %s", provider_format)
            return provider_format

    logger.debug("No cache match found for: %s", model)
    return None


async def get_model_costs(
    model: str,
    *,
    cache_timeout: int = _CACHE_TIMEOUT,
) -> ModelCosts | None:
    """Get cost information for a model from LiteLLM pricing data.

    Attempts to find model costs in cache first. If not found, downloads fresh
    pricing data from LiteLLM's GitHub repository and updates the cache.

    Args:
        model: Name of the model to look up costs for
        cache_timeout: Number of seconds to keep prices in cache (default: 24 hours)

    Returns:
        ModelCosts | None: Model's cost information if found, None otherwise
    """
    # Find matching model name in LiteLLM format
    if litellm_name := find_litellm_model_name(model):
        return _cost_cache.get(litellm_name)  # pyright: ignore

    # Not in cache, try to fetch
    try:
        logger.debug("Downloading pricing data from LiteLLM...")
        async with httpx.AsyncClient() as client:
            response = await client.get(LITELLM_PRICES_URL)
            response.raise_for_status()
            data = response.json()
        logger.debug("Successfully downloaded pricing data")

        # Extract just the cost information we need
        all_costs: dict[str, ModelCosts] = {}
        for name, info in data.items():
            if not isinstance(info, dict):  # Skip sample_spec
                continue
            if "input_cost_per_token" not in info or "output_cost_per_token" not in info:
                continue
            # Store with normalized case
            all_costs[name.lower()] = ModelCosts(
                input_cost_per_token=float(info["input_cost_per_token"]),
                output_cost_per_token=float(info["output_cost_per_token"]),
            )

        logger.debug("Extracted costs for %d models", len(all_costs))

        # Update cache with all costs
        for model_name, cost_info in all_costs.items():
            _cost_cache.set(model_name, cost_info, expire=cache_timeout)
        logger.debug("Updated cache with new pricing data")

        # Return costs for requested model
        if model in all_costs:
            logger.debug("Found costs for requested model: %s", model)
            return all_costs[model]
    except Exception as e:  # noqa: BLE001
        logger.debug("Failed to get model costs: %s", e)
        return None
    else:
        logger.debug("No costs found for model: %s", model)
        return None


async def calculate_token_cost(
    model: str,
    prompt_tokens: int | None,
    completion_tokens: int | None,
    *,
    cache_timeout: int = _CACHE_TIMEOUT,
) -> TokenCosts | None:
    """Calculate detailed costs for token usage based on model pricing.

    Combines input and output token counts with their respective costs to
    calculate the breakdown of costs. If either token count is None, it will
    be treated as 0 tokens.

    Args:
        model: Name of the model used (e.g. "gpt-4", "openai:gpt-3.5-turbo")
        prompt_tokens: Number of tokens in the prompt/input, or None
        completion_tokens: Number of tokens in the completion/output, or None
        cache_timeout: Number of seconds to keep prices in cache (default: 24 hours)

    Returns:
        TokenCosts | None: Detailed cost breakdown if pricing data available
    """
    costs = await get_model_costs(model, cache_timeout=cache_timeout)
    if not costs:
        logger.debug("No costs found for model")
        return None

    # Convert None values to 0
    prompt_count = prompt_tokens or 0
    completion_count = completion_tokens or 0

    prompt_cost = prompt_count * costs["input_cost_per_token"]
    completion_cost = completion_count * costs["output_cost_per_token"]

    token_costs = TokenCosts(
        prompt_cost=float(prompt_cost),
        completion_cost=float(completion_cost),
    )

    logger.debug(
        "Cost calculation - prompt: $%.6f, completion: $%.6f, total: $%.6f",
        token_costs.prompt_cost,
        token_costs.completion_cost,
        token_costs.total_cost,
    )
    return token_costs


async def get_model_limits(
    model: str,
    *,
    cache_timeout: int = _CACHE_TIMEOUT,
) -> TokenLimits | None:
    """Get token limit information for a model from LiteLLM data.

    Args:
        model: Name of the model to look up limits for
        cache_timeout: Number of seconds to keep limits in cache (default: 24 hours)

    Returns:
        TokenLimits | None: Model's token limits if found, None otherwise
    """
    # Normalize case for initial lookup
    normalized_model = model.lower()
    cache_key = f"{normalized_model}_limits"

    # Check cache first
    cached_limits = cast(TokenLimits | None, _cost_cache.get(cache_key))
    if cached_limits is not None:
        return cached_limits

    try:
        logger.debug("Downloading model data from LiteLLM...")
        async with httpx.AsyncClient() as client:
            response = await client.get(LITELLM_PRICES_URL)
            response.raise_for_status()
            data = response.json()
        logger.debug("Successfully downloaded model data")

        # Extract all model limits
        all_limits: dict[str, TokenLimits] = {}
        for name, info in data.items():
            if not isinstance(info, dict):  # Skip sample_spec
                continue

            # Get the token limits with fallbacks
            max_tokens = int(info.get("max_tokens", 0))
            max_input = int(info.get("max_input_tokens", max_tokens))
            max_output = int(info.get("max_output_tokens", max_tokens))

            if any((max_tokens, max_input, max_output)):
                # Store with normalized case
                all_limits[name.lower()] = TokenLimits(
                    total_tokens=max_tokens,
                    input_tokens=max_input,
                    output_tokens=max_output,
                )

        logger.debug("Extracted limits for %d models", len(all_limits))

        # Update cache with all limits
        for model_name, limit_info in all_limits.items():
            limit_cache_key = f"{model_name}_limits"
            _cost_cache.set(limit_cache_key, limit_info, expire=cache_timeout)
            # Also cache the model name for find_litellm_model_name
            _cost_cache.set(model_name, {}, expire=cache_timeout)
        logger.debug("Updated cache with new limit data")

        # Return limits for requested model
        if normalized_model in all_limits:
            logger.debug("Found limits for requested model: %s", model)
            return all_limits[normalized_model]
    except Exception as e:
        error_msg = "Failed to get model limits"
        logger.exception(error_msg)
        raise ValueError(error_msg) from e
    else:
        return None
