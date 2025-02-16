"""Model discovery package."""

from __future__ import annotations

import asyncio
import logging
from typing import Literal, TYPE_CHECKING

from tokonomics.model_discovery.anthropic_provider import AnthropicProvider
from tokonomics.model_discovery.openai_provider import OpenAIProvider
from tokonomics.model_discovery.groq_provider import GroqProvider
from tokonomics.model_discovery.mistral_provider import MistralProvider
from tokonomics.model_discovery.openrouter_provider import OpenRouterProvider
from tokonomics.model_discovery.base import ModelInfo, ModelPricing, ModelProvider

if TYPE_CHECKING:
    from collections.abc import Sequence


logger = logging.getLogger(__name__)


ProviderType = Literal[
    "anthropic",
    "groq",
    "mistral",
    "openai",
    "openrouter",
]


_PROVIDER_MAP: dict[ProviderType, type[ModelProvider]] = {
    "anthropic": AnthropicProvider,
    "groq": GroqProvider,
    "mistral": MistralProvider,
    "openai": OpenAIProvider,
    "openrouter": OpenRouterProvider,
}


async def get_all_models(
    *,
    providers: Sequence[ProviderType] | None = None,
) -> list[ModelInfo]:
    """Fetch models from selected providers in parallel.

    Args:
        providers: Sequence of provider names to use. Defaults to all providers.

    Returns:
        list[ModelInfo]: Combined list of models from all providers.
    """
    selected_providers = providers or list(_PROVIDER_MAP.keys())
    all_models: list[ModelInfo] = []

    async def fetch_provider_models(
        provider_name: ProviderType,
    ) -> list[ModelInfo] | None:
        """Fetch models from a single provider."""
        try:
            provider = _PROVIDER_MAP[provider_name]()
            return await provider.get_models()
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to fetch models from %s: %s", provider_name, str(e))
            return None

    # Fetch models from all providers in parallel
    results = await asyncio.gather(
        *(fetch_provider_models(provider) for provider in selected_providers),
        return_exceptions=False,
    )

    # Combine results, filtering out None values from failed providers
    for provider_models in results:
        if provider_models:
            all_models.extend(provider_models)

    return all_models


__all__ = [
    "AnthropicProvider",
    "GroqProvider",
    "MistralProvider",
    "ModelInfo",
    "ModelPricing",
    "ModelProvider",
    "OpenAIProvider",
    "OpenRouterProvider",
    "ProviderType",
    "get_all_models",
]
