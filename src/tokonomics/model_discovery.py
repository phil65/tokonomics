"""Model discovery and information retrieval."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Any, Protocol

import httpx
from llmling import LLMError


@dataclass
class ModelPricing:
    """Pricing information for a model."""

    prompt: float | None = None
    completion: float | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelPricing:
        """Create pricing from API response dict."""
        return cls(
            prompt=float(data["prompt"]) if "prompt" in data else None,
            completion=float(data["completion"]) if "completion" in data else None,
        )


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

    @classmethod
    def from_openrouter_dict(cls, data: dict[str, Any]) -> ModelInfo:
        """Create model info from OpenRouter API response."""
        return cls(
            id=str(data["id"]),
            name=str(data["name"]),
            provider="openrouter",
            description=str(data.get("description")),
            pricing=ModelPricing.from_dict(data["pricing"]),
        )

    @classmethod
    def from_openai_dict(cls, data: dict[str, Any]) -> ModelInfo:
        """Create model info from OpenAI API response."""
        return cls(
            id=str(data["id"]),
            name=str(data.get("name", data["id"])),
            provider="openai",
            owned_by=str(data.get("owned_by")),
            # OpenAI specific fields that might be present
            description=str(data.get("description")) if "description" in data else None,
            context_window=int(data["context_window"])
            if "context_window" in data
            else None,
        )


class ModelProvider(Protocol):
    """Protocol for model providers."""

    async def get_models(self) -> list[ModelInfo]:
        """Fetch available models from the provider."""
        ...


class OpenRouterProvider:
    """OpenRouter API provider."""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.base_url = "https://openrouter.ai/api/v1"

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

                return [ModelInfo.from_openrouter_dict(item) for item in data["data"]]

        except httpx.HTTPError as e:
            msg = f"Failed to fetch models from OpenRouter: {e}"
            raise LLMError(msg) from e
        except json.JSONDecodeError as e:
            msg = f"Invalid JSON response from OpenRouter: {e}"
            raise LLMError(msg) from e
        except (KeyError, ValueError) as e:
            msg = f"Invalid data in OpenRouter response: {e}"
            raise LLMError(msg) from e


class OpenAIProvider:
    """OpenAI API provider."""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        assert self.api_key, "API key is required"
        self.base_url = "https://api.openai.com/v1"

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

                return [ModelInfo.from_openai_dict(item) for item in data["data"]]

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
                lines.append(f"  - Prompt: ${model.pricing.prompt:.6f} / 1K tokens")
            if model.pricing.completion is not None:
                lines.append(
                    f"  - Completion: ${model.pricing.completion:.6f} / 1K tokens"
                )

        lines.append("-" * 40)

    return "\n".join(lines)


async def main() -> None:
    """Test function to demonstrate usage."""
    openai_provider = OpenAIProvider()
    openrouter_provider = OpenRouterProvider()

    openai_models = await openai_provider.get_models()
    print("=== OpenAI Models ===")
    print(await format_model_info(openai_models))

    print("\n")

    openrouter_models = await openrouter_provider.get_models()
    print("=== OpenRouter Models ===")
    print(await format_model_info(openrouter_models))


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
