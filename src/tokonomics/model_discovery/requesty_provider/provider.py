"""Requesty provider."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from tokonomics.model_discovery.base import ModelProvider
from tokonomics.model_discovery.model_info import ModelInfo, ModelPricing


if TYPE_CHECKING:
    from tokonomics.model_discovery.model_info import Modality


class RequestyProvider(ModelProvider):
    """Requesty API provider."""

    def __init__(self, api_key: str | None = None):
        super().__init__()
        self.api_key = api_key or os.environ.get("REQUESTY_API_KEY")

        self.base_url = "https://router.requesty.ai/v1"
        self.headers = {}
        if self.api_key:
            self.headers["Authorization"] = f"Bearer {self.api_key}"
        self.params = {}

    def is_available(self) -> bool:
        """Check whether the provider is available for use."""
        return bool(self.api_key)

    def _parse_model(self, data: dict[str, Any]) -> ModelInfo:
        """Parse Requesty API response into ModelInfo."""
        # Determine input modalities based on capabilities
        input_modalities: set[Modality] = {"text"}
        if data.get("supports_vision"):
            input_modalities.add("image")

        # Create pricing information
        pricing = ModelPricing(
            prompt=float(data["input_price"]) if data.get("input_price") else None,
            completion=float(data["output_price"]) if data.get("output_price") else None,
            input_cache_read=float(data["cached_price"])
            if data.get("cached_price")
            else None,
            input_cache_write=float(data["caching_price"])
            if data.get("caching_price")
            else None,
        )

        # Build metadata for additional capabilities
        metadata = {}
        if "supports_caching" in data:
            metadata["supports_caching"] = data["supports_caching"]
        if "supports_computer_use" in data:
            metadata["supports_computer_use"] = data["supports_computer_use"]
        if "supports_reasoning" in data:
            metadata["supports_reasoning"] = data["supports_reasoning"]
        if "created" in data:
            metadata["created_timestamp"] = data["created"]

        return ModelInfo(
            id=str(data["id"]),
            name=str(data["id"]),  # Requesty uses id as the model name
            provider="requesty",
            description=str(data.get("description")) if data.get("description") else None,
            pricing=pricing,
            owned_by=str(data.get("owned_by")) if data.get("owned_by") else None,
            context_window=int(data["context_window"])
            if data.get("context_window")
            else None,
            max_output_tokens=int(data["max_output_tokens"])
            if data.get("max_output_tokens")
            else None,
            input_modalities=input_modalities,
            metadata=metadata,
        )


if __name__ == "__main__":
    import asyncio

    provider = RequestyProvider()
    models = asyncio.run(provider.get_models())
    for model in models:
        print(model.format())
