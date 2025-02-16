"""Model discovery and information retrieval."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
import os
from typing import Any

import httpx
from llmling import LLMError


@dataclass
class ModelPricing:
    """Pricing information for a model."""

    prompt: float | None = None
    completion: float | None = None


@dataclass
class ModelInfo:
    """Unified model information from various providers."""

    id: str
    name: str
    provider: str
    description: str | None = None
    pricing: ModelPricing | None = None
    owned_by: str | None = None
    context_window: int | None = None
    is_deprecated: bool = False


class ModelProvider(ABC):
    """Base class for model providers."""

    @abstractmethod
    async def get_models(self) -> list[ModelInfo]:
        """Fetch available models from the provider."""


class AnthropicProvider(ModelProvider):
    """Anthropic API provider."""

    def __init__(
        self,
        api_key: str | None = None,
        version: str = "2023-06-01",
    ):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            msg = "Anthropic API key not found in parameters or ANTHROPIC_API_KEY env var"
            raise LLMError(msg)
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
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": self.version,
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, params=params, headers=headers)
                response.raise_for_status()

                data = response.json()
                return [self._parse_model(item) for item in data.get("data", [])]

        except httpx.HTTPError as e:
            msg = f"Failed to fetch models from Anthropic: {e}"
            raise LLMError(msg) from e
        except json.JSONDecodeError as e:
            msg = f"Invalid JSON response from Anthropic: {e}"
            raise LLMError(msg) from e
        except (KeyError, ValueError) as e:
            msg = f"Invalid data in Anthropic response: {e}"
            raise LLMError(msg) from e


class OpenRouterProvider(ModelProvider):
    """OpenRouter API provider."""

    def __init__(self, api_key: str | None = None):
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

            async with httpx.AsyncClient() as client:
                response = await client.get(url, headers=headers)
                response.raise_for_status()

                data = response.json()
                if not isinstance(data, dict) or "data" not in data:
                    msg = "Invalid response format from OpenRouter API"
                    raise LLMError(msg)

                return [self._parse_model(item) for item in data["data"]]

        except httpx.HTTPError as e:
            msg = f"Failed to fetch models from OpenRouter: {e}"
            raise LLMError(msg) from e
        except json.JSONDecodeError as e:
            msg = f"Invalid JSON response from OpenRouter: {e}"
            raise LLMError(msg) from e
        except (KeyError, ValueError) as e:
            msg = f"Invalid data in OpenRouter response: {e}"
            raise LLMError(msg) from e


class OpenAIProvider(ModelProvider):
    """OpenAI API provider."""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            msg = "OpenAI API key not found in parameters or OPENAI_API_KEY env var"
            raise LLMError(msg)
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
            async with httpx.AsyncClient() as client:
                response = await client.get(url, headers=headers)
                response.raise_for_status()

                data = response.json()
                if not isinstance(data, dict) or "data" not in data:
                    msg = "Invalid response format from OpenAI API"
                    raise LLMError(msg)

                return [self._parse_model(item) for item in data["data"]]

        except httpx.HTTPError as e:
            msg = f"Failed to fetch models from OpenAI: {e}"
            raise LLMError(msg) from e
        except json.JSONDecodeError as e:
            msg = f"Invalid JSON response from OpenAI: {e}"
            raise LLMError(msg) from e
        except (KeyError, ValueError) as e:
            msg = f"Invalid data in OpenAI response: {e}"
            raise LLMError(msg) from e


async def format_model_info(models: list[ModelInfo]) -> str:
    """Format model information into a readable string."""
    if not models:
        return "No models available"

    lines = ["Available Models:", "================"]

    for model in models:
        lines.extend([
            f"\nModel: {model.name}",
            f"ID: {model.id}",
            f"Provider: {model.provider}",
        ])

        if model.description:
            lines.append(f"Description: {model.description}")
        if model.owned_by:
            lines.append(f"Owner: {model.owned_by}")
        if model.context_window:
            lines.append(f"Context Window: {model.context_window:,} tokens")
        if model.pricing:
            lines.append("Pricing:")
            if model.pricing.prompt is not None:
                lines.append(f"  - Prompt: ${model.pricing.prompt:.4f} / 1K tokens")
            if model.pricing.completion is not None:
                lines.append(
                    f"  - Completion: ${model.pricing.completion:.4f} / 1K tokens"
                )

        lines.append("-" * 40)

    return "\n".join(lines)


async def main() -> None:
    """Test function to demonstrate usage."""
    providers: list[ModelProvider] = [
        OpenAIProvider(),
        OpenRouterProvider(),
        AnthropicProvider(),
    ]

    for provider in providers:
        print(f"\n=== {provider.__class__.__name__} Models ===")
        try:
            models = await provider.get_models()
            print(await format_model_info(models))
        except LLMError as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
