"""Utilities for calculating token costs using LiteLLM pricing data."""

from __future__ import annotations

from decimal import Decimal
import os
import pathlib
from typing import Any, cast

from anyenv import get_json
from platformdirs import user_data_dir

from tokonomics import log
from tokonomics.helpers import _is_numeric, _safe_numeric_convert
from tokonomics.toko_types import ModelCapabilities, ModelCosts, TokenCosts


logger = log.get_logger(__name__)


# Cache cost data persistently
PRICING_DIR = pathlib.Path(user_data_dir("tokonomics", "tokonomics")) / "pricing"
PRICING_DIR.mkdir(parents=True, exist_ok=True)
_cost_cache: dict[str, object] = {}

# Cache timeout in seconds (24 hours)
_CACHE_TIMEOUT = 86400


def _is_testing() -> bool:
    """Check if we're running in a test environment."""
    return bool(os.getenv("PYTEST_CURRENT_TEST"))


LITELLM_PRICES_URL = "https://raw.githubusercontent.com/BerriAI/litellm/refs/heads/main/model_prices_and_context_window.json"


def reset_cache() -> None:
    """Clear all cached pricing and limit data."""
    _cost_cache.clear()


def _find_litellm_model_name(model: str) -> str | None:
    """Find matching model name in LiteLLM pricing data.

    Attempts to match the input model name against cached LiteLLM pricing data
    by trying different formats (direct match, base name, provider format).

    Args:
        model: Input model name (e.g. "openai:gpt-4", "gpt-4")

    Returns:
        str | None: Matching LiteLLM model name if found in cache, None otherwise
    """
    logger.debug("Looking up model costs for: %s", model)
    model = model.lower()
    if model in _cost_cache:
        logger.debug("Found direct cache match for: %r", model)
        return model

    if ":" in model:  # For provider:model format, try both variants
        provider, model_name = model.split(":", 1)
        if _cost_cache.get(model_name) is not None:
            logger.debug("Found cache match for base name: %r", model_name)
            return model_name
        # Try provider/model  (normalized)
        if _cost_cache.get(key := f"{provider.lower()}/{model_name}") is not None:
            logger.debug("Found cache match for provider format: %r", key)
            return key
    logger.debug("No cache match found for: %r", model)
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
        data: dict[str, Any] = await get_json(
            LITELLM_PRICES_URL,
            cache=not _is_testing(),
            cache_ttl=_CACHE_TIMEOUT,
            return_type=dict,
        )
        assert isinstance(data, dict), f"Expected dict, got {type(data)}"
        logger.debug("Successfully downloaded pricing data.")

        all_costs: dict[str, ModelCosts] = {}
        for name, info in data.items():
            if not isinstance(info, dict):  # Skip sample_spec
                continue

            # Get cost values safely
            input_cost = info.get("input_cost_per_token")
            output_cost = info.get("output_cost_per_token")

            # Skip if values are missing
            if input_cost is None or output_cost is None:
                continue

            # Skip if values aren't numeric
            if not (_is_numeric(input_cost) and _is_numeric(output_cost)):
                continue

            # Store with normalized case
            all_costs[name.lower()] = ModelCosts(
                input_cost_per_token=Decimal(str(input_cost)),
                output_cost_per_token=Decimal(str(output_cost)),
            )

        logger.debug("Extracted costs for %d models", len(all_costs))

        # Update cache with all costs
        for model_name, cost_info in all_costs.items():
            cost_cache_key = f"{model_name}_costs"
            _cost_cache[cost_cache_key] = cost_info
        logger.debug("Updated cache with new pricing data")

        # Try finding model with different formats
        result = all_costs.get(normalized_model)
        if result is None and ":" in normalized_model:
            provider, model_name = normalized_model.split(":", 1)
            # Try base model name
            result = all_costs.get(model_name)
            if result is None:
                provider_format = f"{provider}/{model_name}"
                result = all_costs.get(provider_format)

        # Cache the result for this specific model name
        if result is not None:
            _cost_cache[cache_key] = result
    except Exception as e:  # noqa: BLE001
        logger.debug("Failed to get model costs for %r: %s", model, e)
        return None
    else:
        return result


async def calculate_token_cost(
    model: str,
    input_tokens: int | None,
    output_tokens: int | None,
    *,
    cache_timeout: int = _CACHE_TIMEOUT,
) -> TokenCosts | None:
    """Calculate detailed costs for token usage based on model pricing.

    Combines input and output token counts with their respective costs to
    calculate the breakdown of costs. If either token count is None, it will
    be treated as 0 tokens.

    Args:
        model: Name of the model used (e.g. "gpt-4", "openai:gpt-3.5-turbo")
        input_tokens: Number of tokens in the prompt/input, or None
        output_tokens: Number of tokens in the completion/output, or None
        cache_timeout: Number of seconds to keep prices in cache (default: 24 hours)

    Returns:
        TokenCosts | None: Detailed cost breakdown if pricing data available
    """
    costs = await get_model_costs(model, cache_timeout=cache_timeout)
    if not costs:
        logger.debug("No costs found for model %r", model)
        return None

    # Convert None values to 0
    input_count = input_tokens or 0
    output_count = output_tokens or 0

    input_cost = input_count * costs["input_cost_per_token"]
    output_cost = output_count * costs["output_cost_per_token"]

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
    cache_key = "available_models"

    cached_models = cast(list[str] | None, _cost_cache.get(cache_key))
    if cached_models is not None:
        return cached_models

    try:
        logger.debug("Downloading model data from LiteLLM...")
        data: dict[str, Any] = await get_json(
            LITELLM_PRICES_URL,
            cache=not _is_testing(),
            cache_ttl=_CACHE_TIMEOUT,
            return_type=dict,
        )
        assert isinstance(data, dict), f"Expected dict, got {type(data)}"
        # Filter out non-dictionary entries (like sample_spec) and collect model names
        model_names = sorted(
            name.lower() for name, info in data.items() if isinstance(info, dict)
        )

        logger.debug("Found %d available models", len(model_names))

        # Cache the results
        _cost_cache[cache_key] = model_names
    except Exception as e:
        error_msg = "Failed to get available models"
        logger.exception(error_msg)
        raise ValueError(error_msg) from e
    else:
        return model_names


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
    normalized_model = model.lower()
    cache_key = f"{normalized_model}_capabilities"

    cached_capabilities = cast(ModelCapabilities | None, _cost_cache.get(cache_key))
    if cached_capabilities is not None:
        return cached_capabilities

    try:
        logger.debug("Downloading model data from LiteLLM...")
        data: dict[str, Any] = await get_json(
            LITELLM_PRICES_URL,
            cache=not _is_testing(),
            cache_ttl=_CACHE_TIMEOUT,
            return_type=dict,
        )
        assert isinstance(data, dict), f"Expected dict, got {type(data)}"
        logger.debug("Successfully downloaded model data")

        all_capabilities: dict[str, ModelCapabilities] = {}
        for name, info in data.items():
            if not isinstance(info, dict):
                continue

            # Handle token limits with defaults
            max_tokens_raw = info.get("max_tokens", 0)
            max_input_raw = info.get("max_input_tokens", max_tokens_raw)
            max_output_raw = info.get("max_output_tokens", max_tokens_raw)

            # Convert token values to integers, defaulting to 0
            try:
                max_tokens = int(_safe_numeric_convert(max_tokens_raw))
                max_input = int(_safe_numeric_convert(max_input_raw))
                max_output = int(_safe_numeric_convert(max_output_raw))
            except (ValueError, TypeError):
                max_tokens = max_input = max_output = 0

            capabilities = ModelCapabilities(
                max_tokens=max_tokens,
                max_input_tokens=max_input,
                max_output_tokens=max_output,
                litellm_provider=info.get("litellm_provider"),
                mode=info.get("mode"),
                supports_function_calling=bool(info.get("supports_function_calling")),
                supports_parallel_function_calling=bool(
                    info.get("supports_parallel_function_calling")
                ),
                supports_vision=bool(info.get("supports_vision")),
                supports_audio_input=bool(info.get("supports_audio_input")),
                supports_audio_output=bool(info.get("supports_audio_output")),
                supports_prompt_caching=bool(info.get("supports_prompt_caching")),
                supports_response_schema=bool(info.get("supports_response_schema")),
                supports_system_messages=bool(info.get("supports_system_messages")),
            )
            all_capabilities[name.lower()] = capabilities

        logger.debug("Extracted capabilities for %d models", len(all_capabilities))

        # Update cache with all capabilities
        for model_name, model_capabilities in all_capabilities.items():
            cap_cache_key = f"{model_name}_capabilities"
            _cost_cache[cap_cache_key] = model_capabilities
        logger.debug("Updated cache with new capabilities data")

        # Try finding model with different formats
        result = all_capabilities.get(normalized_model)
        if result is None and ":" in normalized_model:
            provider, model_name = normalized_model.split(":", 1)
            # Try base model name
            result = all_capabilities.get(model_name)
            if result is None:
                # Try provider/model format
                provider_format = f"{provider}/{model_name}"
                result = all_capabilities.get(provider_format)

        if result is not None:
            _cost_cache[cache_key] = result
            return result

    except Exception as e:
        error_msg = f"Failed to get model capabilities for {model}: {e}"
        logger.exception(error_msg)
        raise ValueError(error_msg) from e

    return None


if __name__ == "__main__":
    import asyncio

    async def main():
        # Example usage
        model_name = "openrouter:openai/gpt-5-nano"
        limits = await get_model_costs(model_name)
        print(limits)

    asyncio.run(main())
