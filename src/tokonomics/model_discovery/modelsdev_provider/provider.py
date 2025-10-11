"""Models.dev provider - unified model discovery from models.dev API."""

from __future__ import annotations

import logging
from typing import Any

from tokonomics.model_discovery.base import ModelProvider
from tokonomics.model_discovery.model_info import ModelInfo, ModelPricing


logger = logging.getLogger(__name__)


class ModelsDevProvider(ModelProvider):
    """Models.dev API provider - aggregates models from all providers."""

    def __init__(self):
        super().__init__()
        self.base_url = "https://models.dev"
        self.headers = {}
        self.params = {}

    def is_available(self) -> bool:
        """Check whether the provider is available for use."""
        return True  # No API key required

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
        input_modalities = {"text"}
        output_modalities = {"text"}
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

        try:
            logger.debug("Fetching models from models.dev API")
            data = await get_json(
                f"{self.base_url}/api.json",
                headers=self.headers,
                params=self.params,
                cache=True,
                return_type=dict,
            )

            if not isinstance(data, dict):
                raise RuntimeError("Invalid response format from models.dev")

            models = []
            for provider_id, provider_data in data.items():
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
                    except Exception as e:
                        logger.warning(
                            "Failed to parse model %s from provider %s: %s",
                            model_id,
                            provider_id,
                            e,
                        )
                        continue

            logger.info(
                "Fetched %d models from %d providers via models.dev",
                len(models),
                len([
                    p
                    for p in data.keys()
                    if isinstance(data[p], dict) and "models" in data[p]
                ]),
            )
            return models

        except HttpError as e:
            msg = f"Failed to fetch models from models.dev: {e}"
            raise RuntimeError(msg) from e


if __name__ == "__main__":
    import asyncio

    async def main():
        provider = ModelsDevProvider()
        models = await provider.get_models()

        # Show summary
        providers = {}
        for model in models:
            providers[model.provider] = providers.get(model.provider, 0) + 1

        print(f"Total models: {len(models)}")
        print("Models by provider:")
        for provider_name, count in sorted(providers.items()):
            print(f"  {provider_name}: {count}")

        # Show sample models
        if models:
            print("\nSample model:")
            print(models[0].format())

    asyncio.run(main())
