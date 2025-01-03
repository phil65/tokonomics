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


def reset_cache() -> None:
    """Clear all cached pricing and limit data."""
    _cost_cache.clear()


def _is_numeric(value: str | int | float) -> bool:  # noqa: PYI041
    """Check if a value can be converted to a number."""
    if isinstance(value, int | float):
        return True
    if not isinstance(value, str):
        return False
    try:
        float(value)
    except ValueError:
        return False
    else:
        return True


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
    normalized_model = model.lower()
    cache_key = f"{normalized_model}_costs"

    cached_costs = cast(ModelCosts | None, _cost_cache.get(cache_key))
    if cached_costs is not None:
        return cached_costs

    try:
        logger.debug("Downloading pricing data from LiteLLM...")
        async with httpx.AsyncClient() as client:
            response = await client.get(LITELLM_PRICES_URL)
            response.raise_for_status()
            data = response.json()
        logger.debug("Successfully downloaded pricing data")

        all_costs: dict[str, ModelCosts] = {}
        for name, info in data.items():
            if not isinstance(info, dict):  # Skip sample_spec
                continue

            # Get cost values safely
            input_cost = info.get("input_cost_per_token")
            output_cost = info.get("output_cost_per_token")

            # Skip if values are missing
            if input_cost is None or output_cost is None:
                logger.debug("Skipping model %s - missing cost data", name)
                continue

            # Skip if values aren't numeric
            if not (_is_numeric(input_cost) and _is_numeric(output_cost)):
                logger.debug("Skipping model %s - invalid cost data", name)
                continue

            # Store with normalized case
            all_costs[name.lower()] = ModelCosts(
                input_cost_per_token=float(input_cost),
                output_cost_per_token=float(output_cost),
            )

        logger.debug("Extracted costs for %d models", len(all_costs))

        # Update cache with all costs
        for model_name, cost_info in all_costs.items():
            cost_cache_key = f"{model_name}_costs"
            _cost_cache.set(cost_cache_key, cost_info, expire=cache_timeout)
        logger.debug("Updated cache with new pricing data")

        # Try finding model with different formats
        result = all_costs.get(normalized_model)
        if result is None and ":" in normalized_model:
            provider, model_name = normalized_model.split(":", 1)
            # Try base model name
            result = all_costs.get(model_name)
            if result is None:
                # Try provider/model format
                provider_format = f"{provider}/{model_name}"
                result = all_costs.get(provider_format)

        # Cache the result for this specific model name
        if result is not None:
            _cost_cache.set(cache_key, result, expire=cache_timeout)
    except Exception as e:  # noqa: BLE001
        logger.debug("Failed to get model costs: %s", e)
        return None
    else:
        return result


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
    normalized_model = model.lower()
    cache_key = f"{normalized_model}_limits"

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

        all_limits: dict[str, TokenLimits] = {}
        for name, info in data.items():
            if not isinstance(info, dict):  # Skip sample_spec
                continue

            # Check if values are numeric before parsing
            max_tokens_raw = info.get("max_tokens", 0)
            max_input_raw = info.get("max_input_tokens", max_tokens_raw)
            max_output_raw = info.get("max_output_tokens", max_tokens_raw)

            # Skip if values aren't numeric
            if not all(map(_is_numeric, (max_tokens_raw, max_input_raw, max_output_raw))):
                logger.debug("Skipping model %s - non-numeric limits", name)
                continue

            # Now safe to convert to int
            max_tokens = int(float(max_tokens_raw))
            max_input = int(float(max_input_raw))
            max_output = int(float(max_output_raw))

            if any((max_tokens, max_input, max_output)):
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
        error_msg = f"Failed to get model limits: {e}"
        logger.exception(error_msg)
        raise ValueError(error_msg) from e
    else:
        return None
