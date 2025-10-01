"""Cohere provider."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from tokonomics.model_discovery.base import ModelProvider
from tokonomics.model_discovery.model_info import ModelInfo


if TYPE_CHECKING:
    from tokonomics.model_discovery.model_info import Modality


class CohereProvider(ModelProvider):
    """Cohere API provider."""

    def __init__(self, api_key: str | None = None):
        super().__init__()
        self.api_key = (
            api_key or os.environ.get("COHERE_API_KEY") or os.environ.get("CO_API_KEY")
        )
        if not self.api_key:
            msg = "Cohere API key not found in parameters or COHERE_API_KEY env var"
            raise RuntimeError(msg)

        self.base_url = "https://api.cohere.com/v1"
        self.headers = {"Authorization": f"Bearer {self.api_key}"}
        self.params = {}

    def is_available(self) -> bool:
        """Check whether the provider is available for use."""
        return bool(self.api_key)

    def _parse_model(self, data: dict[str, Any]) -> ModelInfo:
        """Parse Cohere API response into ModelInfo."""
        # Extract model name and ID
        name = data.get("name", "")

        # Determine supported modalities
        input_modalities: set[Modality] = {"text"}
        if data.get("supports_vision"):
            input_modalities.add("image")

        # Extract context length if available
        context_window = None
        if isinstance(data.get("context_length"), int | float):
            context_window = int(data["context_length"])

        # Extract endpoints
        endpoints = data.get("endpoints", [])
        endpoints_str = ", ".join(endpoints) if endpoints else ""

        # Build description
        description = []
        if endpoints_str:
            description.append(f"Supported endpoints: {endpoints_str}")
        if data.get("tokenizer_url"):
            description.append(f"Tokenizer URL: {data['tokenizer_url']}")
        if data.get("finetuned"):
            description.append("This is a fine-tuned model")

        description_str = "\n".join(description) if description else None
        is_embedding = "embed" in data.get("endpoints", []) or "embed_image" in data.get(
            "endpoints", []
        )
        return ModelInfo(
            id=name,
            name=name,
            provider="cohere",
            description=description_str,
            context_window=context_window,
            is_embedding=is_embedding,
            input_modalities=input_modalities,
        )

    async def get_models(self) -> list[ModelInfo]:
        """Override the base method to handle Cohere's unique API structure."""
        from anyenv import HttpError, get_json

        url = f"{self.base_url}/models"

        try:
            data = await get_json(
                url,
                headers=self.headers,
                params=self.params,
                cache=True,
                return_type=dict,
            )

            if "models" not in data:
                msg = "Invalid response format from Cohere API"
                raise RuntimeError(msg)

            return [self._parse_model(item) for item in data["models"]]

        except HttpError as e:
            msg = f"Failed to fetch models from Cohere: {e}"
            raise RuntimeError(msg) from e


if __name__ == "__main__":
    import asyncio

    async def main():
        provider = CohereProvider()
        models = await provider.get_models()
        for model in models:
            print(model.format())

    asyncio.run(main())
