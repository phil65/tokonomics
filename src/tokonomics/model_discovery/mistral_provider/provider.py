"""Mistral provider."""

from __future__ import annotations

import os
from typing import Any

from tokonomics.model_discovery.base import ModelProvider
from tokonomics.model_discovery.model_info import ModelInfo


class MistralProvider(ModelProvider):
    """Mistral AI API provider."""

    def __init__(self, api_key: str | None = None):
        super().__init__()
        api_key = api_key or os.environ.get("MISTRAL_API_KEY")
        if not api_key:
            msg = "Mistral API key not found in parameters or MISTRAL_API_KEY env var"
            raise RuntimeError(msg)

        self.base_url = "https://api.mistral.ai/v1"
        self.headers = {"Authorization": f"Bearer {api_key}"}
        self.params = {}

    def _parse_model(self, data: dict[str, Any]) -> ModelInfo:
        """Parse Mistral API response into ModelInfo."""
        return ModelInfo(
            id=str(data["id"]),
            name=str(data.get("name", data["id"])),
            provider="mistral",
            owned_by=str(data["owned_by"]),
            description=str(data.get("description")),
            context_window=int(data["max_context_length"]),
            # Model is deprecated if it has a deprecation date
            is_deprecated=bool(data.get("deprecation")),
        )


if __name__ == "__main__":
    import asyncio

    provider = MistralProvider()
    models = asyncio.run(provider.get_models())
    for model in models:
        print(model.format())
