"""Vercel AI Gateway provider."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from tokonomics.model_discovery.base import ModelProvider
from tokonomics.model_discovery.model_info import ModelInfo, ModelPricing


if TYPE_CHECKING:
    from tokonomics.model_discovery.model_info import Modality


class VercelGatewayProvider(ModelProvider):
    """Vercel AI Gateway provider."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        headers: dict[str, str] | None = None,
    ):
        super().__init__()
        self.api_key = api_key or os.environ.get("AI_GATEWAY_API_KEY")
        if not self.api_key:
            msg = (
                "Vercel Gateway API key not found in parameters or "
                "AI_GATEWAY_API_KEY env var"
            )
            raise RuntimeError(msg)

        self.base_url = base_url or "https://ai-gateway.vercel.sh/v1/ai"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "ai-gateway-protocol-version": "0.0.1",
            "ai-gateway-auth-method": "api-key",
            "Content-Type": "application/json",
            **(headers or {}),
        }
        self.params = {}

    def is_available(self) -> bool:
        """Check whether the provider is available for use."""
        return bool(self.api_key)

    async def get_models(self) -> list[ModelInfo]:
        """Fetch available models from the Vercel Gateway."""
        from anyenv import HttpError, get_json

        url = f"{self.base_url}/config"

        try:
            data: dict[str, Any] = await get_json(
                url,
                headers=self.headers,
                params=self.params,
                cache=True,
                return_type=dict,
            )

            if not isinstance(data, dict) or "models" not in data:
                msg = f"Invalid response format from {self.__class__.__name__}"
                raise RuntimeError(msg)

            return [self._parse_model(item) for item in data["models"]]

        except HttpError as e:
            msg = f"Failed to fetch models from {self.__class__.__name__}: {e}"
            raise RuntimeError(msg) from e

    def _parse_model(self, data: dict[str, Any]) -> ModelInfo:
        """Parse Vercel Gateway API response into ModelInfo."""
        # Extract basic model information
        model_id = str(data["id"])
        name = str(data.get("name", model_id))
        description = data.get("description")

        # Extract specification details
        spec = data.get("specification", {})
        provider_name = spec.get("provider", "vercel-gateway")

        # Determine model type and modalities
        model_type = data.get("modelType")
        is_embedding = model_type == "embedding"

        # Set modalities based on model type
        input_modalities: set[Modality] = {"text"}
        output_modalities: set[Modality] = {"text"}

        if model_type == "image":
            output_modalities.add("image")
        elif "vision" in name.lower() or "gpt-4o" in model_id.lower():
            input_modalities.add("image")

        # Parse pricing information
        pricing = None
        if pricing_data := data.get("pricing"):
            pricing = ModelPricing(
                prompt=self._parse_price(pricing_data.get("input")),
                completion=self._parse_price(pricing_data.get("output")),
                input_cache_read=self._parse_price(pricing_data.get("cachedInputTokens")),
                input_cache_write=self._parse_price(
                    pricing_data.get("cacheCreationInputTokens")
                ),
            )

        # Build metadata dictionary
        metadata = {
            "gateway_provider": provider_name,
            "specification_version": spec.get("specificationVersion"),
            "model_type": model_type,
        }

        # Add any additional fields not covered by standard ModelInfo
        excluded_keys = {
            "id",
            "name",
            "description",
            "pricing",
            "specification",
            "modelType",
        }
        metadata.update({
            key: value
            for key, value in data.items()
            if key not in excluded_keys and value is not None
        })

        return ModelInfo(
            id=model_id,
            name=name,
            provider="vercel-gateway",
            description=description,
            pricing=pricing,
            is_embedding=is_embedding,
            input_modalities=input_modalities,
            output_modalities=output_modalities,
            is_free=pricing is None or (pricing.prompt == 0 and pricing.completion == 0),
            metadata=metadata,
        )

    def _parse_price(self, price_str: str | None) -> float | None:
        """Parse price string to float."""
        if not price_str:
            return None
        try:
            return float(price_str)
        except (ValueError, TypeError):
            return None


if __name__ == "__main__":
    import asyncio

    provider = VercelGatewayProvider()
    models = asyncio.run(provider.get_models())
    for model in models:
        print(model.format())
        print("-" * 80)
