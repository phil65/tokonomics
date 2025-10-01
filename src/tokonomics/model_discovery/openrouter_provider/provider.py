"""OpenRouter provider."""

from __future__ import annotations

import os
from typing import Any, cast

from tokonomics.model_discovery.base import ModelProvider
from tokonomics.model_discovery.model_info import Modality, ModelInfo, ModelPricing


class OpenRouterProvider(ModelProvider):
    """OpenRouter API provider."""

    def __init__(self, api_key: str | None = None):
        super().__init__()
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self.base_url = "https://openrouter.ai/api/v1"
        self.headers = {"HTTP-Referer": "https://github.com/phi-ai"}

        if self.api_key:
            self.headers["Authorization"] = f"Bearer {self.api_key}"

        self.params = {}

    def is_available(self) -> bool:
        """Check whether the provider is available for use."""
        return bool(self.api_key)

    def _parse_model(self, data: dict[str, Any]) -> ModelInfo:
        """Parse OpenRouter API response into ModelInfo."""
        pricing = ModelPricing(
            prompt=float(data["pricing"]["prompt"]),
            completion=float(data["pricing"]["completion"]),
            image=float(data["pricing"].get("image", 0))
            if "image" in data["pricing"]
            else None,
            request=float(data["pricing"].get("request", 0))
            if "request" in data["pricing"]
            else None,
            input_cache_read=float(data["pricing"].get("input_cache_read", 0))
            if "input_cache_read" in data["pricing"]
            else None,
            input_cache_write=float(data["pricing"].get("input_cache_write", 0))
            if "input_cache_write" in data["pricing"]
            else None,
            web_search=float(data["pricing"].get("web_search", 0))
            if "web_search" in data["pricing"]
            else None,
            internal_reasoning=float(data["pricing"].get("internal_reasoning", 0))
            if "internal_reasoning" in data["pricing"]
            else None,
        )
        model_id = str(data["id"])
        is_free = model_id.endswith(":free")

        # Extract modalities if available
        input_modalities: set[Modality] = {"text"}
        output_modalities: set[Modality] = {"text"}

        if architecture := data.get("architecture"):
            if input_mods := architecture.get("input_modalities"):
                input_modalities = {cast(Modality, m) for m in input_mods}
            if output_mods := architecture.get("output_modalities"):
                output_modalities = {cast(Modality, m) for m in output_mods}

        # Parse context length and created timestamp
        context_window = data.get("context_length")
        created_timestamp = data.get("created")

        # Extract additional fields
        hugging_face_id = data.get("hugging_face_id")

        # Check for top_provider info
        is_moderated = False
        if top_provider := data.get("top_provider"):
            is_moderated = top_provider.get("is_moderated", False)

        # Get supported parameters if available
        supported_parameters = data.get("supported_parameters", [])

        # Prepare metadata dictionary for OpenRouter-specific fields
        metadata = {
            "hugging_face_id": hugging_face_id,
            "created_timestamp": created_timestamp,
            "is_moderated": is_moderated,
            "supported_parameters": supported_parameters,
        }

        # Add per_request_limits if available
        if per_request_limits := data.get("per_request_limits"):
            metadata["per_request_limits"] = per_request_limits

        # Add tokenizer info if available
        if (architecture := data.get("architecture")) and (
            tokenizer := architecture.get("tokenizer")
        ):
            metadata["tokenizer"] = tokenizer

        return ModelInfo(
            id=str(data["id"]),
            name=str(data["name"]),
            provider="openrouter",
            description=str(data.get("description")),
            pricing=pricing,
            is_free=is_free,
            context_window=context_window,
            input_modalities=input_modalities,
            output_modalities=output_modalities,
            owned_by=hugging_face_id,  # Use hugging_face_id as the owner when available
            metadata=metadata,
        )


if __name__ == "__main__":
    import asyncio

    import devtools

    provider = OpenRouterProvider(api_key="your_api_key")
    models = asyncio.run(provider.get_models())
    for model in models:
        devtools.debug(model.input_modalities)
