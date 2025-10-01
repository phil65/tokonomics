"""Groq provider."""

from __future__ import annotations

import os
from typing import Any

from tokonomics.model_discovery.base import ModelProvider
from tokonomics.model_discovery.model_info import ModelInfo


class GroqProvider(ModelProvider):
    """Groq API provider."""

    def __init__(self, api_key: str | None = None):
        super().__init__()
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            msg = "Groq API key not found in parameters or GROQ_API_KEY env var"
            raise RuntimeError(msg)

        self.base_url = "https://api.groq.com/openai/v1"
        self.headers = {"Authorization": f"Bearer {self.api_key}"}
        self.params = {}

    def is_available(self) -> bool:
        """Check whether the provider is available for use."""
        return bool(self.api_key)

    def _parse_model(self, data: dict[str, Any]) -> ModelInfo:
        """Parse Groq API response into ModelInfo."""
        return ModelInfo(
            id=str(data["id"]),
            name=str(data.get("name", data["id"])),
            provider="groq",
            owned_by=str(data["owned_by"]),
            context_window=int(data["context_window"]),
            is_deprecated=not data.get("active", False),
        )


if __name__ == "__main__":

    async def main():
        provider = GroqProvider()
        models = await provider.get_models()
        print(models)

    import asyncio

    asyncio.run(main())
