"""Gemini provider."""

from __future__ import annotations

import os
from typing import Any

from tokonomics.model_discovery.base import ModelProvider
from tokonomics.model_discovery.model_info import Modality, ModelInfo


class GeminiProvider(ModelProvider):
    """Google Gemini API provider."""

    def __init__(self, api_key: str | None = None):
        super().__init__()
        api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            msg = "Gemini API key not found in parameters or GEMINI_API_KEY env var"
            raise RuntimeError(msg)

        self.base_url = "https://generativelanguage.googleapis.com/v1beta"
        self.params = {"key": api_key}
        self.headers = {}

    def _parse_model(self, data: dict[str, Any]) -> ModelInfo:
        """Parse Gemini API response into ModelInfo."""
        model_id = data.get("name", "")
        if model_id.startswith("models/"):
            model_id = model_id[7:]  # Remove 'models/' prefix

        input_modalities: set[Modality] = {"text"}
        output_modalities: set[Modality] = {"text"}
        model_name = data.get("displayName", "").lower()
        model_description = data.get("description", "").lower()
        if "vision" in model_name or "multimodal" in model_description:
            input_modalities.add("image")
        if "image generation" in model_name or "imagen" in model_id.lower():
            output_modalities.add("image")
        generation_methods = data.get("supportedGenerationMethods", [])
        if "predict" in generation_methods and "imagen" in model_id.lower():
            output_modalities.add("image")
        methods_str = ", ".join(generation_methods) if generation_methods else "None"
        description = data.get("description", "")
        description_parts = [description] if description else []
        if generation_methods:
            description_parts.append(f"Supported generation methods: {methods_str}")
        if "temperature" in data:
            description_parts.append(f"Default temperature: {data['temperature']}")
        if "maxTemperature" in data:
            description_parts.append(f"Max temperature: {data['maxTemperature']}")
        full_description = "\n".join(description_parts) if description_parts else None
        return ModelInfo(
            id=model_id,
            name=data.get("displayName", model_id),
            provider="gemini",
            description=full_description,
            owned_by="Google",
            context_window=data.get("inputTokenLimit"),
            max_output_tokens=data.get("outputTokenLimit"),
            input_modalities=input_modalities,
            output_modalities=output_modalities,
        )

    async def get_models(self) -> list[ModelInfo]:
        """Override the base method to handle Gemini's unique API structure."""
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
                msg = "Invalid response format from Gemini API"
                raise RuntimeError(msg)

            # Get all pages if nextPageToken is present
            models = data["models"]
            next_page_token = data.get("nextPageToken")

            while next_page_token:
                page_params = self.params.copy()
                page_params["pageToken"] = next_page_token

                next_data = await get_json(
                    url,
                    headers=self.headers,
                    params=page_params,
                    cache=True,
                    return_type=dict,
                )

                if "models" in next_data:
                    models.extend(next_data["models"])

                next_page_token = next_data.get("nextPageToken")

            return [self._parse_model(item) for item in models]

        except HttpError as e:
            msg = f"Failed to fetch models from Gemini: {e}"
            raise RuntimeError(msg) from e


if __name__ == "__main__":
    import asyncio

    async def main():
        provider = GeminiProvider()
        models = await provider.get_models()
        for model in models:
            print(model.format())

    asyncio.run(main())
