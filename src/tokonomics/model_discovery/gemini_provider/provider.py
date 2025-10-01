"""Gemini provider."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from tokonomics.model_discovery.base import ModelProvider
from tokonomics.model_discovery.model_info import ModelInfo


if TYPE_CHECKING:
    from tokonomics.model_discovery.model_info import Modality


class GeminiProvider(ModelProvider):
    """Google Gemini API provider."""

    def __init__(self, api_key: str | None = None):
        super().__init__()
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            msg = "Gemini API key not found in parameters or GEMINI_API_KEY env var"
            raise RuntimeError(msg)

        self.base_url = "https://generativelanguage.googleapis.com/v1beta"
        self.params = {"key": self.api_key}
        self.headers = {}

    def is_available(self) -> bool:
        """Check whether the provider is available for use."""
        return bool(self.api_key)

    def _parse_model(self, data: dict[str, Any]) -> ModelInfo:
        """Parse Gemini API response into ModelInfo."""
        model_id = str(data.get("name", ""))
        if model_id.startswith("models/"):
            model_id = model_id[7:]  # Remove 'models/' prefix

        display_name = str(data.get("displayName", model_id))

        # Check for embedding model indicators
        is_embedding = False
        model_id_lower = model_id.lower()
        display_name_lower = display_name.lower()
        description = data.get("description", "").lower()

        # Check name/ID for embedding indicators
        if "embed" in model_id_lower or "embed" in display_name_lower:
            is_embedding = True

        # Check generation methods for embedding indicators
        generation_methods = data.get("supportedGenerationMethods", [])
        if any(method in ("embedContent", "embedText") for method in generation_methods):
            is_embedding = True

        # Check output token limit (embedding models typically output only 1 token)
        if data.get("outputTokenLimit") == 1:
            is_embedding = True

        # Check description for embedding indicators
        if "distributed representation" in description:
            is_embedding = True

        # Determine input/output modalities
        input_modalities: set[Modality] = {"text"}
        output_modalities: set[Modality] = {"text"}

        if "vision" in display_name_lower or "multimodal" in description:
            input_modalities.add("image")

        # Check for image generation capabilities
        if "image generation" in display_name_lower or "imagen" in model_id_lower:
            output_modalities.add("image")

        if "predict" in generation_methods and "imagen" in model_id_lower:
            output_modalities.add("image")

        return ModelInfo(
            id=model_id,
            name=display_name,
            provider="gemini",
            description=data.get("description"),
            owned_by="Google",
            context_window=data.get("inputTokenLimit"),
            max_output_tokens=data.get("outputTokenLimit"),
            input_modalities=input_modalities,
            output_modalities=output_modalities,
            is_embedding=is_embedding,
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
