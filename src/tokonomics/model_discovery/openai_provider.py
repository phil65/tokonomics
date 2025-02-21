"""OpenAI provider."""

from __future__ import annotations

import json
import os
from typing import Any

import httpx

from tokonomics.model_discovery.base import ModelInfo, ModelProvider
from tokonomics.utils import make_request


class OpenAIProvider(ModelProvider):
    """OpenAI API provider."""

    def __init__(self, api_key: str | None = None):
        super().__init__()
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            msg = "OpenAI API key not found in parameters or OPENAI_API_KEY env var"
            raise RuntimeError(msg)
        self.base_url = "https://api.openai.com/v1"

    def _parse_model(self, data: dict[str, Any]) -> ModelInfo:
        """Parse OpenAI API response into ModelInfo."""
        return ModelInfo(
            id=str(data["id"]),
            name=str(data.get("name", data["id"])),
            provider="openai",
            owned_by=str(data.get("owned_by")),
            description=str(data.get("description")) if "description" in data else None,
            context_window=(
                int(data["context_window"]) if "context_window" in data else None
            ),
        )

    async def get_models(self) -> list[ModelInfo]:
        """Fetch available models from OpenAI."""
        url = f"{self.base_url}/models"

        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            response = await make_request(url, headers=headers)
            response.raise_for_status()

            data = response.json()
            if not isinstance(data, dict) or "data" not in data:
                msg = "Invalid response format from OpenAI API"
                raise RuntimeError(msg)

            return [self._parse_model(item) for item in data["data"]]

        except httpx.HTTPError as e:
            msg = f"Failed to fetch models from OpenAI: {e}"
            raise RuntimeError(msg) from e
        except json.JSONDecodeError as e:
            msg = f"Invalid JSON response from OpenAI: {e}"
            raise RuntimeError(msg) from e
        except (KeyError, ValueError) as e:
            msg = f"Invalid data in OpenAI response: {e}"
            raise RuntimeError(msg) from e
