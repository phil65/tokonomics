"""Anthropic provider."""

from __future__ import annotations

import json
import os
from typing import Any

import httpx

from tokonomics.model_discovery.base import ModelInfo, ModelProvider
from tokonomics.utils import make_request


class AnthropicProvider(ModelProvider):
    """Anthropic API provider."""

    def __init__(
        self,
        api_key: str | None = None,
        version: str = "2023-06-01",
    ):
        super().__init__()
        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        assert api_key, "API key not found"
        self.api_key = api_key
        if not self.api_key:
            msg = "Anthropic API key not found in parameters or ANTHROPIC_API_KEY env var"
            raise RuntimeError(msg)
        self.version = version
        self.base_url = "https://api.anthropic.com/v1"

    def _parse_model(self, data: dict[str, Any]) -> ModelInfo:
        """Parse Anthropic API response into ModelInfo."""
        return ModelInfo(
            id=str(data["id"]),
            name=str(data.get("name", data["id"])),
            provider="anthropic",
            description=str(data.get("description")) if "description" in data else None,
            context_window=(
                int(data["context_window"]) if "context_window" in data else None
            ),
        )

    async def get_models(self) -> list[ModelInfo]:
        """Fetch available models from Anthropic."""
        url = f"{self.base_url}/models"
        params = {"limit": 1000}
        headers = {"x-api-key": self.api_key, "anthropic-version": self.version}

        try:
            response = await make_request(url, params=params, headers=headers)
            response.raise_for_status()
            data = response.json()
            return [self._parse_model(item) for item in data.get("data", [])]

        except httpx.HTTPError as e:
            msg = f"Failed to fetch models from Anthropic: {e}"
            raise RuntimeError(msg) from e
        except json.JSONDecodeError as e:
            msg = f"Invalid JSON response from Anthropic: {e}"
            raise RuntimeError(msg) from e
        except (KeyError, ValueError) as e:
            msg = f"Invalid data in Anthropic response: {e}"
            raise RuntimeError(msg) from e


if __name__ == "__main__":
    import asyncio

    provider = AnthropicProvider()
    models = asyncio.run(provider.get_models())
    for model in models:
        print(model)
