"""Model discovery package."""

from __future__ import annotations

import asyncio
import logging
from functools import partial
from typing import Literal, TYPE_CHECKING

from tokonomics.model_discovery.base import ModelProvider
from tokonomics.model_discovery.model_info import ModelPricing, ModelInfo
from tokonomics.model_discovery.groq_provider import GroqProvider
from tokonomics.model_discovery.mistral_provider import MistralProvider
from tokonomics.model_discovery.openrouter_provider import OpenRouterProvider
from tokonomics.model_discovery.github_provider import GitHubProvider
from tokonomics.model_discovery.copilot_provider import CopilotProvider, token_manager
from tokonomics.model_discovery.gemini_provider import GeminiProvider
from tokonomics.model_discovery.deepseek_provider import DeepSeekProvider
from tokonomics.model_discovery.requesty_provider import RequestyProvider
from tokonomics.model_discovery.xai_provider import XAIProvider
from tokonomics.model_discovery.comet_provider import CometProvider
from tokonomics.model_discovery.novita_provider import NovitaProvider
from tokonomics.model_discovery.vercel_gateway_provider import VercelGatewayProvider
from tokonomics.model_discovery.modelsdev_provider import ModelsDevProvider
from tokonomics.model_discovery.ollama_provider import OllamaProvider
# from tokonomics.model_discovery.cerebras_provider import CerebrasProvider
# from tokonomics.model_discovery.cohere_provider import CohereProvider

# Use ModelsDevProvider with pre-configured filters as drop-in replacements
AnthropicProvider = partial(ModelsDevProvider, provider="anthropic")
OpenAIProvider = partial(ModelsDevProvider, provider="openai")
CohereProvider = partial(ModelsDevProvider, provider="cohere")
CerebrasProvider = partial(ModelsDevProvider, provider="cerebras")
FireworksProvider = partial(ModelsDevProvider, provider="fireworks-ai")
AzureProvider = partial(ModelsDevProvider, provider="azure")
ChutesProvider = partial(ModelsDevProvider, provider="chutes")
CortecsProvider = partial(ModelsDevProvider, provider="cortecs")
MoonshotProvider = partial(ModelsDevProvider, provider="moonshotai")
ZAIProvider = partial(ModelsDevProvider, provider="zai")


if TYPE_CHECKING:
    from collections.abc import Callable
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
    "deepseek",
    "requesty",
    "xai",
    "comet",
    "novita",
    "vercel-gateway",
    "chutes",
    "cortecs",
    "azure",
    "fireworks-ai",
    "ollama",
    "moonshotai",
    "zai",
]


_PROVIDER_MAP: dict[ProviderType, Callable[[], ModelProvider]] = {
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
    "deepseek": DeepSeekProvider,
    "requesty": RequestyProvider,
    "xai": XAIProvider,
    "comet": CometProvider,
    "novita": NovitaProvider,
    "vercel-gateway": VercelGatewayProvider,
    "fireworks-ai": FireworksProvider,
    "azure": AzureProvider,
    "chutes": ChutesProvider,
    "cortecs": CortecsProvider,
    "ollama": OllamaProvider,
    "zai": ZAIProvider,
    "moonshotai": MoonshotProvider,
}


def get_all_models_sync(
    *,
    providers: Sequence[ProviderType] | None = None,
    max_workers: int | None = None,
    include_deprecated: bool = False,
) -> list[ModelInfo]:
    """Fetch models from selected providers in parallel using threads.

    Args:
        providers: Sequence of provider names to use. Defaults to available providers.
        max_workers: Maximum number of worker threads.
                     Defaults to min(32, os.cpu_count() * 5)
        include_deprecated: Whether to include deprecated models. Defaults to False.

    Returns:
        list[ModelInfo]: Combined list of models from all providers.
    """
    import concurrent.futures

    if providers is not None:
        selected_providers = providers
    else:
        # Only use available providers when no specific providers are requested
        selected_providers = []
        for provider_name, provider_class in _PROVIDER_MAP.items():
            try:
                provider = provider_class()
                if provider.is_available():
                    selected_providers.append(provider_name)
            except Exception:  # noqa: BLE001
                # Provider initialization failed, skip it
                continue
    all_models: list[ModelInfo] = []

    def fetch_provider_models(provider_name: ProviderType) -> list[ModelInfo] | None:
        """Fetch models from a single provider."""
        import anyenv

        try:
            provider = _PROVIDER_MAP[provider_name]()
            models = anyenv.run_sync(provider.get_models())
            if not include_deprecated:
                models = [model for model in models if not model.is_deprecated]
            models = [model for model in models if not model.is_embedding]
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to fetch models from %s: %s", provider_name, str(e))
            return None
        else:
            return models

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks and collect results as they complete
        fut = {executor.submit(fetch_provider_models, p): p for p in selected_providers}
        for future in concurrent.futures.as_completed(fut):
            provider_models = future.result()
            if provider_models:
                all_models.extend(provider_models)

    successful_providers = {model.provider for model in all_models}
    logger.info(
        "Fetched %d models from %d providers: %s",
        len(all_models),
        len(successful_providers),
        ", ".join(sorted(successful_providers)),
    )
    return all_models


async def get_all_models(
    *,
    providers: Sequence[ProviderType] | None = None,
    include_deprecated: bool = False,
) -> list[ModelInfo]:
    """Fetch models from selected providers in parallel.

    Args:
        providers: Sequence of provider names to use. Defaults to available providers.
        include_deprecated: Whether to include deprecated models. Defaults to False.

    Returns:
        list[ModelInfo]: Combined list of models from all providers.
    """
    if providers is not None:
        selected_providers = providers
    else:
        # Only use available providers when no specific providers are requested
        selected_providers = []
        for provider_name, provider_class in _PROVIDER_MAP.items():
            try:
                provider = provider_class()
                if provider.is_available():
                    selected_providers.append(provider_name)
            except Exception:  # noqa: BLE001
                # Provider initialization failed, skip it
                continue
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
            models = [model for model in models if not model.is_embedding]
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

    successful_providers = {model.provider for model in all_models}
    logger.info(
        "Fetched %d models from %d providers: %s",
        len(all_models),
        len(successful_providers),
        ", ".join(sorted(successful_providers)),
    )
    return all_models


__all__ = [
    "AnthropicProvider",
    "CerebrasProvider",
    "CohereProvider",
    "CometProvider",
    "CopilotProvider",
    "DeepSeekProvider",
    "GeminiProvider",
    "GitHubProvider",
    "GroqProvider",
    "MistralProvider",
    "ModelInfo",
    "ModelPricing",
    "ModelProvider",
    "NovitaProvider",
    "OllamaProvider",
    "OpenAIProvider",
    "OpenRouterProvider",
    "ProviderType",
    "RequestyProvider",
    "VercelGatewayProvider",
    "XAIProvider",
    "get_all_models",
    "get_all_models_sync",
    "token_manager",
]
