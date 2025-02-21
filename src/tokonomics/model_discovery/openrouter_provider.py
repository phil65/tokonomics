"""OpenRouter provider."""

from __future__ import annotations

import json
import os
from typing import Any

import httpx

from tokonomics.model_discovery.base import ModelInfo, ModelPricing, ModelProvider
from tokonomics.utils import make_request


class OpenRouterProvider(ModelProvider):
    """OpenRouter API provider."""

    def __init__(self, api_key: str | None = None):
        super().__init__()
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self.base_url = "https://openrouter.ai/api/v1"

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

    async def get_models(self) -> list[ModelInfo]:
        """Fetch available models from OpenRouter."""
        url = f"{self.base_url}/models"

        try:
            headers = {"HTTP-Referer": "https://github.com/phi-ai"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            response = await make_request(url, headers=headers)
            response.raise_for_status()

            data = response.json()
            if not isinstance(data, dict) or "data" not in data:
                msg = "Invalid response format from OpenRouter API"
                raise RuntimeError(msg)

            return [self._parse_model(item) for item in data["data"]]

        except httpx.HTTPError as e:
            msg = f"Failed to fetch models from OpenRouter: {e}"
            raise RuntimeError(msg) from e
        except json.JSONDecodeError as e:
            msg = f"Invalid JSON response from OpenRouter: {e}"
            raise RuntimeError(msg) from e
        except (KeyError, ValueError) as e:
            msg = f"Invalid data in OpenRouter response: {e}"
            raise RuntimeError(msg) from e


if __name__ == "__main__":
    import asyncio

    provider = OpenRouterProvider(api_key="your_api_key")
    models = asyncio.run(provider.get_models())
    for model in models:
        print(model.format())
