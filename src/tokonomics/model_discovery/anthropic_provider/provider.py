"""Anthropic provider."""

from __future__ import annotations

import os
from typing import Any

from tokonomics.model_discovery.base import ModelProvider
from tokonomics.model_discovery.model_info import ModelInfo


class AnthropicProvider(ModelProvider):
    """Anthropic API provider."""

    def __init__(self, api_key: str | None = None, version: str = "2023-06-01"):
        super().__init__()
        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        assert api_key, "API key not found"
        self.api_key = api_key
        if not self.api_key:
            msg = "Anthropic API key not found in parameters or ANTHROPIC_API_KEY env var"
            raise RuntimeError(msg)
        self.version = version
        self.base_url = "https://api.anthropic.com/v1"
        self.headers = {"x-api-key": api_key, "anthropic-version": version}
        self.params = {"limit": 1000}

    def is_available(self) -> bool:
        """Check whether the provider is available for use."""
        return bool(self.api_key)

    def _parse_model(self, data: dict[str, Any]) -> ModelInfo:
        """Parse Anthropic API response into ModelInfo."""
        return ModelInfo(
            id=str(data["id"]),
            name=str(data.get("name", data["id"])),
            provider="anthropic",
            description=str(data.get("description")) if "description" in data else None,
            context_window=(
                int(data["context_window"]) if "context_window" in data else None
            ),
        )


if __name__ == "__main__":
    import asyncio

    provider = AnthropicProvider()
    models = asyncio.run(provider.get_models())
    for model in models:
        print(model)
