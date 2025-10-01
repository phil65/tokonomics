"""xAI provider."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from tokonomics.model_discovery.base import ModelProvider
from tokonomics.model_discovery.model_info import ModelInfo, ModelPricing


if TYPE_CHECKING:
    from tokonomics.model_discovery.model_info import Modality


class XAIProvider(ModelProvider):
    """xAI API provider."""

    def __init__(self, api_key: str | None = None):
        super().__init__()
        self.api_key = api_key or os.environ.get("XAI_API_KEY")
        if not self.api_key:
            msg = "xAI API key not found in parameters or XAI_API_KEY env var"
            raise RuntimeError(msg)

        self.base_url = "https://api.x.ai/v1"
        self.headers = {"Authorization": f"Bearer {self.api_key}"}
        self.params = {}

    def is_available(self) -> bool:
        """Check whether the provider is available for use."""
        return bool(self.api_key)

    def _parse_model(self, data: dict[str, Any]) -> ModelInfo:
        """Parse xAI API response into ModelInfo."""
        # Extract basic info
        model_id = str(data["id"])
        name = str(data.get("name", model_id))
        owned_by = str(data.get("owned_by"))

        # Parse modalities
        input_modalities: set[Modality] = set()
        output_modalities: set[Modality] = set()

        for modality in data.get("input_modalities", []):
            if modality in {"text", "image", "audio", "video"}:
                input_modalities.add(modality)

        for modality in data.get("output_modalities", []):
            if modality in {"text", "image", "audio", "video"}:
                output_modalities.add(modality)

        # Set defaults if empty
        if not input_modalities:
            input_modalities = {"text"}
        if not output_modalities:
            output_modalities = {"text"}

        # Parse pricing (prices appear to be in micro-units, convert to dollars per token)
        pricing = None
        if any(
            key in data
            for key in [
                "prompt_text_token_price",
                "completion_text_token_price",
                "prompt_image_token_price",
                "search_price",
            ]
        ):
            pricing = ModelPricing(
                prompt=data.get("prompt_text_token_price", 0) / 1_000_000
                if "prompt_text_token_price" in data
                else None,
                completion=data.get("completion_text_token_price", 0) / 1_000_000
                if "completion_text_token_price" in data
                else None,
                image=data.get("prompt_image_token_price", 0) / 1_000_000
                if "prompt_image_token_price" in data
                else None,
                input_cache_read=data.get("cached_prompt_text_token_price", 0) / 1_000_000
                if "cached_prompt_text_token_price" in data
                else None,
                web_search=data.get("search_price", 0) / 1_000_000
                if "search_price" in data
                else None,
            )

        # Build metadata
        metadata = {}
        if "fingerprint" in data:
            metadata["fingerprint"] = data["fingerprint"]
        if "version" in data:
            metadata["version"] = data["version"]
        if "created" in data:
            metadata["created_timestamp"] = data["created"]
        if data.get("aliases"):
            metadata["aliases"] = data["aliases"]

        return ModelInfo(
            id=model_id,
            name=name,
            provider="xai",
            owned_by=owned_by,
            pricing=pricing,
            input_modalities=input_modalities,
            output_modalities=output_modalities,
            metadata=metadata,
        )

    async def get_models(self) -> list[ModelInfo]:
        """Override the base method to handle xAI's unique API structure."""
        from anyenv import HttpError, get_json

        url = f"{self.base_url}/language-models"

        try:
            data = await get_json(
                url,
                headers=self.headers,
                params=self.params,
                cache=True,
                return_type=dict,
            )

            if "models" not in data:
                msg = "Invalid response format from xAI API"
                raise RuntimeError(msg)

            return [self._parse_model(item) for item in data["models"]]

        except HttpError as e:
            msg = f"Failed to fetch models from xAI: {e}"
            raise RuntimeError(msg) from e


if __name__ == "__main__":
    import asyncio

    async def main():
        provider = XAIProvider()
        models = await provider.get_models()
        for model in models:
            print(model.format())

    asyncio.run(main())
