"""Novita AI provider."""

from __future__ import annotations

import os
from typing import Any

from tokonomics.model_discovery.base import ModelProvider
from tokonomics.model_discovery.model_info import ModelInfo, ModelPricing


class NovitaProvider(ModelProvider):
    """Novita AI API provider."""

    def __init__(self, api_key: str | None = None):
        super().__init__()
        self.api_key = api_key or os.environ.get("NOVITA_API_KEY")
        if not self.api_key:
            msg = "Novita API key not found in parameters or NOVITA_API_KEY env var"
            raise RuntimeError(msg)

        self.base_url = "https://api.novita.ai/openai/v1"
        self.headers = {"Authorization": f"Bearer {self.api_key}"}
        self.params = {}

    def is_available(self) -> bool:
        """Check whether the provider is available for use."""
        return bool(self.api_key)

    def _parse_model(self, data: dict[str, Any]) -> ModelInfo:
        """Parse Novita API response into ModelInfo."""
        # Convert pricing from per million tokens to per token
        pricing = ModelPricing(
            prompt=float(data["input_token_price_per_m"]) / 1_000_000
            if data.get("input_token_price_per_m")
            else None,
            completion=float(data["output_token_price_per_m"]) / 1_000_000
            if data.get("output_token_price_per_m")
            else None,
        )

        # Build metadata for additional fields
        metadata = {}
        if "created" in data:
            metadata["created_timestamp"] = data["created"]
        if "object" in data:
            metadata["object_type"] = data["object"]

        return ModelInfo(
            id=str(data["id"]),
            name=str(data.get("title", data["id"])),
            provider="novita",
            description=str(data.get("description")) if data.get("description") else None,
            context_window=int(data["context_size"])
            if data.get("context_size")
            else None,
            pricing=pricing,
            metadata=metadata,
        )


if __name__ == "__main__":
    import asyncio

    async def main():
        provider = NovitaProvider()
        models = await provider.get_models()
        for model in models:
            print(model.format())

    asyncio.run(main())
