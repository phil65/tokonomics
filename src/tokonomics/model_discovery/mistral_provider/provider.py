"""Mistral provider."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from tokonomics.model_discovery.base import ModelProvider
from tokonomics.model_discovery.model_info import ModelInfo


if TYPE_CHECKING:
    from tokonomics.model_discovery.model_info import Modality


class MistralProvider(ModelProvider):
    """Mistral AI API provider."""

    def __init__(self, api_key: str | None = None):
        super().__init__()
        self.api_key = api_key or os.environ.get("MISTRAL_API_KEY")
        if not self.api_key:
            msg = "Mistral API key not found in parameters or MISTRAL_API_KEY env var"
            raise RuntimeError(msg)

        self.base_url = "https://api.mistral.ai/v1"
        self.headers = {"Authorization": f"Bearer {self.api_key}"}
        self.params = {}

    def is_available(self) -> bool:
        """Check whether the provider is available for use."""
        return bool(self.api_key)

    def _parse_model(self, data: dict[str, Any]) -> ModelInfo:
        """Parse Mistral API response into ModelInfo."""
        # Extract capabilities
        capabilities = data.get("capabilities", {})

        # Determine if model is an embedding model
        is_embedding = False
        model_id = str(data["id"]).lower()
        model_name = str(data.get("name", data["id"])).lower()

        if (
            "embed" in model_id
            or "embed" in model_name
            or (
                not capabilities.get("completion_chat", True)
                and not capabilities.get("function_calling", True)
            )
        ):
            is_embedding = True

        # Determine input/output modalities
        input_modalities: set[Modality] = {"text"}
        if capabilities.get("vision"):
            input_modalities.add("image")

        return ModelInfo(
            id=str(data["id"]),
            name=str(data.get("name", data["id"])),
            provider="mistral",
            owned_by=str(data.get("owned_by")),
            description=str(data.get("description")),
            context_window=int(data.get("max_context_length", 0)),
            is_deprecated=bool(data.get("deprecation")),
            is_embedding=is_embedding,
            input_modalities=input_modalities,
        )


if __name__ == "__main__":
    import asyncio

    provider = MistralProvider()
    models = asyncio.run(provider.get_models())
    for model in models:
        print(model.format())
