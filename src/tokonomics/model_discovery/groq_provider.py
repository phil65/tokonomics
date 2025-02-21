"""Groq provider."""

from __future__ import annotations

import json
import os
from typing import Any

import httpx

from tokonomics.model_discovery.base import ModelInfo, ModelProvider
from tokonomics.utils import make_request


class GroqProvider(ModelProvider):
    """Groq API provider."""

    def __init__(self, api_key: str | None = None):
        super().__init__()
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            msg = "Groq API key not found in parameters or GROQ_API_KEY env var"
            raise RuntimeError(msg)
        self.base_url = "https://api.groq.com/openai/v1"

    def _parse_model(self, data: dict[str, Any]) -> ModelInfo:
        """Parse Groq API response into ModelInfo."""
        return ModelInfo(
            id=str(data["id"]),
            name=str(data.get("name", data["id"])),
            provider="groq",
            owned_by=str(data["owned_by"]),
            context_window=int(data["context_window"]),
            is_deprecated=not data.get("active", True),
        )

    async def get_models(self) -> list[ModelInfo]:
        """Fetch available models from Groq."""
        url = f"{self.base_url}/models"

        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            response = await make_request(url, headers=headers)
            response.raise_for_status()

            data = response.json()
            if not isinstance(data, dict) or "data" not in data:
                msg = "Invalid response format from Groq API"
                raise RuntimeError(msg)

            return [
                self._parse_model(item)
                for item in data["data"]
                if item.get("active", True)  # Only return active models
            ]

        except httpx.HTTPError as e:
            msg = f"Failed to fetch models from Groq: {e}"
            raise RuntimeError(msg) from e
        except json.JSONDecodeError as e:
            msg = f"Invalid JSON response from Groq: {e}"
            raise RuntimeError(msg) from e
        except (KeyError, ValueError) as e:
            msg = f"Invalid data in Groq response: {e}"
            raise RuntimeError(msg) from e
