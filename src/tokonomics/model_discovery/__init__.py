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
from tokonomics.model_discovery.base import ModelProvider
from tokonomics.model_discovery.model_info import ModelPricing, ModelInfo
from tokonomics.model_discovery.github_provider import GitHubProvider
from tokonomics.model_discovery.cerebras_provider import CerebrasProvider
from tokonomics.model_discovery.copilot_provider import CopilotProvider, token_manager
from tokonomics.model_discovery.gemini_provider import GeminiProvider
from tokonomics.model_discovery.cohere_provider import CohereProvider


if TYPE_CHECKING:
    from collections.abc import Sequence


logger = logging.getLogger(__name__)


ProviderType = Literal[
    "anthropic",
    "groq",
    "mistral",
    "openai",
    "openrouter",
    "github",
    "copilot",
    "cerebras",
    "gemini",
    "cohere",
]


_PROVIDER_MAP: dict[ProviderType, type[ModelProvider]] = {
    "anthropic": AnthropicProvider,
    "groq": GroqProvider,
    "mistral": MistralProvider,
    "openai": OpenAIProvider,
    "openrouter": OpenRouterProvider,
    "github": GitHubProvider,
    "copilot": CopilotProvider,
    "cerebras": CerebrasProvider,
    "gemini": GeminiProvider,
    "cohere": CohereProvider,
}


def get_all_models_sync(
    *,
    providers: Sequence[ProviderType] | None = None,
    max_workers: int | None = None,
    include_deprecated: bool = False,
) -> list[ModelInfo]:
    """Fetch models from selected providers in parallel using threads.

    Args:
        providers: Sequence of provider names to use. Defaults to all providers.
        max_workers: Maximum number of worker threads.
                     Defaults to min(32, os.cpu_count() * 5)
        include_deprecated: Whether to include deprecated models. Defaults to False.

    Returns:
        list[ModelInfo]: Combined list of models from all providers.
    """
    import concurrent.futures

    selected_providers = providers or list(_PROVIDER_MAP.keys())
    all_models: list[ModelInfo] = []

    def fetch_provider_models(provider_name: ProviderType) -> list[ModelInfo] | None:
        """Fetch models from a single provider."""
        import anyenv

        try:
            provider = _PROVIDER_MAP[provider_name]()
            models = anyenv.run_sync(provider.get_models())
            if not include_deprecated:
                models = [model for model in models if not model.is_deprecated]
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to fetch models from %s: %s", provider_name, str(e))
            return None
        else:
            return models

    # Use ThreadPoolExecutor for parallel execution
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_provider = {
            executor.submit(fetch_provider_models, provider): provider
            for provider in selected_providers
        }

        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_provider):
            provider_models = future.result()
            if provider_models:
                all_models.extend(provider_models)

    return all_models


async def get_all_models(
    *,
    providers: Sequence[ProviderType] | None = None,
    include_deprecated: bool = False,
) -> list[ModelInfo]:
    """Fetch models from selected providers in parallel.

    Args:
        providers: Sequence of provider names to use. Defaults to all providers.
        include_deprecated: Whether to include deprecated models. Defaults to False.

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
            models = await provider.get_models()
            if not include_deprecated:
                models = [model for model in models if not model.is_deprecated]
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to fetch models from %s: %s", provider_name, str(e))
            return None
        else:
            return models

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
    "CerebrasProvider",
    "CohereProvider",
    "CopilotProvider",
    "GeminiProvider",
    "GitHubProvider",
    "GroqProvider",
    "MistralProvider",
    "ModelInfo",
    "ModelPricing",
    "ModelProvider",
    "OpenAIProvider",
    "OpenRouterProvider",
    "ProviderType",
    "get_all_models",
    "get_all_models_sync",
    "token_manager",
]
