"""Mistral provider."""

from __future__ import annotations

import json
import os
from typing import Any

import httpx

from tokonomics.model_discovery.base import ModelInfo, ModelProvider
from tokonomics.utils import make_request


class MistralProvider(ModelProvider):
    """Mistral AI API provider."""

    def __init__(self, api_key: str | None = None):
        super().__init__()
        self.api_key = api_key or os.environ.get("MISTRAL_API_KEY")
        if not self.api_key:
            msg = "Mistral API key not found in parameters or MISTRAL_API_KEY env var"
            raise RuntimeError(msg)
        self.base_url = "https://api.mistral.ai/v1"

    def _parse_model(self, data: dict[str, Any]) -> ModelInfo:
        """Parse Mistral API response into ModelInfo."""
        return ModelInfo(
            id=str(data["id"]),
            name=str(data.get("name", data["id"])),
            provider="mistral",
            owned_by=str(data["owned_by"]),
            description=str(data.get("description")),
            context_window=int(data["max_context_length"]),
            # Model is deprecated if it has a deprecation date
            is_deprecated=bool(data.get("deprecation")),
        )

    async def get_models(self) -> list[ModelInfo]:
        """Fetch available models from Mistral."""
        url = f"{self.base_url}/models"

        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            response = await make_request(url, headers=headers)
            response.raise_for_status()

            data = response.json()
            if not isinstance(data, dict) or "data" not in data:
                msg = "Invalid response format from Mistral API"
                raise RuntimeError(msg)

            return [self._parse_model(item) for item in data["data"]]

        except httpx.HTTPError as e:
            msg = f"Failed to fetch models from Mistral: {e}"
            raise RuntimeError(msg) from e
        except json.JSONDecodeError as e:
            msg = f"Invalid JSON response from Mistral: {e}"
            raise RuntimeError(msg) from e
        except (KeyError, ValueError) as e:
            msg = f"Invalid data in Mistral response: {e}"
            raise RuntimeError(msg) from e
