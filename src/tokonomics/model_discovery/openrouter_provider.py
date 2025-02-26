"""OpenRouter provider."""

from __future__ import annotations

import os
from typing import Any

from tokonomics.model_discovery.base import ModelProvider
from tokonomics.model_discovery.model_info import ModelInfo, ModelPricing


class OpenRouterProvider(ModelProvider):
    """OpenRouter API provider."""

    def __init__(self, api_key: str | None = None):
        super().__init__()
        self.base_url = "https://openrouter.ai/api/v1"
        self.headers = {"HTTP-Referer": "https://github.com/phi-ai"}

        if api_key := (api_key or os.environ.get("OPENROUTER_API_KEY")):
            self.headers["Authorization"] = f"Bearer {api_key}"

        self.params = {}

    def _parse_model(self, data: dict[str, Any]) -> ModelInfo:
        """Parse OpenRouter API response into ModelInfo."""
        pricing = ModelPricing(
            prompt=float(data["pricing"]["prompt"]),
            completion=float(data["pricing"]["completion"]),
        )
        return ModelInfo(
            id=str(data["id"]),
            name=str(data["name"]),
            provider="openrouter",
            description=str(data.get("description")),
            pricing=pricing,
        )


if __name__ == "__main__":
    import asyncio

    provider = OpenRouterProvider(api_key="your_api_key")
    models = asyncio.run(provider.get_models())
    for model in models:
        print(model.format())
