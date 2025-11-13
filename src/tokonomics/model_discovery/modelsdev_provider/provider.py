"""Models.dev provider - unified model discovery from models.dev API."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any, Literal

from tokonomics.model_discovery.base import ModelProvider
from tokonomics.model_discovery.model_info import ModelInfo, ModelPricing


if TYPE_CHECKING:
    from tokonomics.model_discovery.model_info import Modality


logger = logging.getLogger(__name__)


ModelsDevProviderType = Literal[
    "alibaba",
    "alibaba-cn",
    "amazon-bedrock",
    "anthropic",
    "azure",
    "baseten",
    "cerebras",
    "chutes",
    "cloudflare-workers-ai",
    "cohere",
    "cortecs",
    "deepinfra",
    "deepseek",
    "fastrouter",
    "fireworks-ai",
    "github-copilot",
    "github-models",
    "google",
    "google-vertex",
    "google-vertex-anthropic",
    "groq",
    "huggingface",
    "inception",
    "inference",
    "llama",
    "lmstudio",
    "lucidquery",
    "mistral",
    "modelscope",
    "moonshotai",
    "moonshotai-cn",
    "morph",
    "nvidia",
    "openai",
    "opencode",
    "openrouter",
    "perplexity",
    "requesty",
    "submodel",
    "synthetic",
    "togetherai",
    "upstage",
    "v0",
    "venice",
    "vercel",
    "wandb",
    "xai",
    "zai",
    "zai-coding-plan",
    "zhipuai",
    "zhipuai-coding-plan",
]


class ModelsDevProvider(ModelProvider):
    """Models.dev API provider - aggregates models from all providers."""

    def __init__(self, provider: ModelsDevProviderType | None = None):
        super().__init__()
        self.base_url = "https://models.dev"
        self.headers = {}
        self.params = {}
        self.provider_filter = provider

    def is_available(self) -> bool:  # noqa: PLR0911
        """Check whether the provider is available for use."""
        if self.provider_filter == "anthropic":
            return bool(os.environ.get("ANTHROPIC_API_KEY"))
        if self.provider_filter == "openai":
            return bool(os.environ.get("OPENAI_API_KEY"))
        if self.provider_filter == "cohere":
            return bool(os.environ.get("COHERE_API_KEY") or os.environ.get("CO_API_KEY"))
        if self.provider_filter == "cerebras":
            return bool(os.environ.get("CEREBRAS_API_KEY"))
        if self.provider_filter == "deepseek":
            return bool(os.environ.get("DEEPSEEK_API_KEY"))
        if self.provider_filter == "groq":
            return bool(os.environ.get("GROQ_API_KEY"))
        if self.provider_filter == "chutes":
            return bool(os.environ.get("CHUTES_API_KEY"))
        if self.provider_filter == "cloudflare-workers-ai":
            return bool(os.environ.get("CF_API_TOKEN"))
        if self.provider_filter == "cortecs":
            return bool(os.environ.get("CORTECS_API_KEY"))
        if self.provider_filter == "xai":
            return bool(os.environ.get("XAI_API_KEY"))
        if self.provider_filter == "azure":
            return bool(os.environ.get("AZURE_OPENAI_API_KEY"))
        if self.provider_filter == "fireworks-ai":
            return bool(os.environ.get("FIREWORKS_API_KEY"))
        if self.provider_filter == "mistral":
            return bool(os.environ.get("MISTRAL_API_KEY"))
        return True

    def _parse_model(self, data: dict[str, Any]) -> ModelInfo:
        """Parse models.dev API response into ModelInfo."""
        # Extract provider from context (set during parsing)
        provider_id = data.get("_provider_id", "unknown")

        # Extract pricing information
        pricing = None
        if "cost" in data:
            cost = data["cost"]
            pricing = ModelPricing(
                prompt=cost.get("input", 0) / 1_000_000 if "input" in cost else None,
                completion=cost.get("output", 0) / 1_000_000
                if "output" in cost
                else None,
                input_cache_read=cost.get("cache_read", 0) / 1_000_000
                if "cache_read" in cost
                else None,
                input_cache_write=cost.get("cache_write", 0) / 1_000_000
                if "cache_write" in cost
                else None,
            )

        # Extract modalities
        input_modalities: set[Modality] = {"text"}
        output_modalities: set[Modality] = {"text"}
        if "modalities" in data:
            modalities = data["modalities"]
            input_modalities = set(modalities.get("input", ["text"]))
            output_modalities = set(modalities.get("output", ["text"]))

        model_id = str(data["id"])

        # Determine if it's an embedding model (heuristic)
        is_embedding = "embedding" in model_id.lower() or "embed" in model_id.lower()

        return ModelInfo(
            id=model_id,
            name=str(data.get("name", model_id)),
            provider=provider_id,
            description=None,  # models.dev doesn't provide descriptions
            pricing=pricing,
            context_window=data.get("limit", {}).get("context"),
            max_output_tokens=data.get("limit", {}).get("output"),
            is_embedding=is_embedding,
            input_modalities=input_modalities,
            output_modalities=output_modalities,
            is_free=pricing is not None
            and pricing.prompt == 0
            and pricing.completion == 0,
            metadata={
                "attachment": data.get("attachment", False),
                "reasoning": data.get("reasoning", False),
                "temperature": data.get("temperature", True),
                "tool_call": data.get("tool_call", False),
                "knowledge": data.get("knowledge"),
                "release_date": data.get("release_date"),
                "last_updated": data.get("last_updated"),
                "open_weights": data.get("open_weights", False),
            },
        )

    async def get_models(self) -> list[ModelInfo]:
        """Fetch all models from models.dev API."""
        from anyenv import HttpError, get_json

        logger.debug("Fetching models from models.dev API")
        data = await get_json(
            f"{self.base_url}/api.json",
            headers=self.headers,
            params=self.params,
            cache=True,
            return_type=dict,
        )

        if not isinstance(data, dict):
            msg = "Invalid response format from models.dev"
            raise RuntimeError(msg)  # noqa: TRY004

        try:
            models = []
            for provider_id, provider_data in data.items():
                # Apply provider filter if specified
                if (
                    self.provider_filter is not None
                    and provider_id != self.provider_filter
                ):
                    continue

                if not isinstance(provider_data, dict) or "models" not in provider_data:
                    continue

                provider_models = provider_data["models"]
                if not isinstance(provider_models, dict):
                    continue

                for model_id, model_info in provider_models.items():
                    if not isinstance(model_info, dict):
                        continue

                    # Add provider context to model data
                    model_data = dict(model_info)
                    model_data["id"] = model_id
                    model_data["_provider_id"] = provider_id

                    try:
                        model = self._parse_model(model_data)
                        models.append(model)
                    except Exception as e:  # noqa: BLE001
                        logger.warning(
                            "Failed to parse model %s from provider %s: %s",
                            model_id,
                            provider_id,
                            e,
                        )
                        continue

            if self.provider_filter:
                logger.debug(
                    "Fetched %d models for provider %s via models.dev",
                    len(models),
                    self.provider_filter,
                )
            else:
                logger.info(
                    "Fetched %d models from %d providers via models.dev",
                    len(models),
                    len([
                        p
                        for p in data
                        if isinstance(data[p], dict) and "models" in data[p]
                    ]),
                )
        except HttpError as e:
            msg = f"Failed to fetch models from models.dev: {e}"
            raise RuntimeError(msg) from e
        else:
            return models


if __name__ == "__main__":
    import asyncio

    async def main():
        # Test all providers
        provider = ModelsDevProvider()
        models = await provider.get_models()
        print(f"All providers: {len(models)} models")

        # Test specific provider
        openai_provider = ModelsDevProvider(provider="openai")
        openai_models = await openai_provider.get_models()
        print(f"OpenAI only: {len(openai_models)} models")

        # Test another specific provider
        anthropic_provider = ModelsDevProvider(provider="anthropic")
        anthropic_models = await anthropic_provider.get_models()
        print(f"Anthropic only: {len(anthropic_models)} models")

        # Show sample models
        if openai_models:
            print("\nSample OpenAI model:")
            print(openai_models[0].format())

    asyncio.run(main())
