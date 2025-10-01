"""DeepSeek provider."""

from __future__ import annotations

import os
from typing import Any

from tokonomics.model_discovery.base import ModelProvider
from tokonomics.model_discovery.model_info import ModelInfo


class DeepSeekProvider(ModelProvider):
    """DeepSeek API provider."""

    def __init__(self, api_key: str | None = None):
        super().__init__()
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not self.api_key:
            msg = "DeepSeek API key not found in parameters or DEEPSEEK_API_KEY env var"
            raise RuntimeError(msg)

        self.base_url = "https://api.deepseek.com/v1"
        self.headers = {"Authorization": f"Bearer {self.api_key}"}
        self.params = {}

    def is_available(self) -> bool:
        """Check whether the provider is available for use."""
        return bool(self.api_key)

    def _parse_model(self, data: dict[str, Any]) -> ModelInfo:
        """Parse DeepSeek API response into ModelInfo."""
        return ModelInfo(
            id=str(data["id"]),
            name=str(data.get("name", data["id"])),
            provider="deepseek",
            owned_by=str(data.get("owned_by")),
            description=str(data.get("description")) if "description" in data else None,
            context_window=(
                int(data["context_window"]) if "context_window" in data else None
            ),
        )


if __name__ == "__main__":
    import asyncio

    provider = DeepSeekProvider()
    models = asyncio.run(provider.get_models())
    for model in models:
        print(model.format())
