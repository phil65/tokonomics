"""OpenAI provider."""

from __future__ import annotations

import os
from typing import Any

from tokonomics.model_discovery.base import ModelProvider
from tokonomics.model_discovery.model_info import ModelInfo


class OpenAIProvider(ModelProvider):
    """OpenAI API provider."""

    def __init__(self, api_key: str | None = None):
        super().__init__()
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            msg = "OpenAI API key not found in parameters or OPENAI_API_KEY env var"
            raise RuntimeError(msg)

        self.base_url = "https://api.openai.com/v1"
        self.headers = {"Authorization": f"Bearer {api_key}"}
        self.params = {}

    def _parse_model(self, data: dict[str, Any]) -> ModelInfo:
        """Parse OpenAI API response into ModelInfo."""
        return ModelInfo(
            id=str(data["id"]),
            name=str(data.get("name", data["id"])),
            provider="openai",
            owned_by=str(data.get("owned_by")),
            is_embedding="embedding" in str(data.get("id", "")),
            description=str(data.get("description")) if "description" in data else None,
            context_window=(
                int(data["context_window"]) if "context_window" in data else None
            ),
        )


if __name__ == "__main__":
    import asyncio

    provider = OpenAIProvider()
    models = asyncio.run(provider.get_models())
    for model in models:
        print(model.format())
