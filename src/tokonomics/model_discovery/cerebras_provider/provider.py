"""Cerebras provider."""

from __future__ import annotations

import logging
import os
from typing import Any

from tokonomics.model_discovery.base import ModelProvider
from tokonomics.model_discovery.model_info import ModelInfo


logger = logging.getLogger(__name__)


class CerebrasProvider(ModelProvider):
    """Cerebras API provider."""

    def __init__(self, api_key: str | None = None):
        super().__init__()
        self.api_key = api_key or os.environ.get("CEREBRAS_API_KEY")
        if not self.api_key:
            msg = "Cerebras API key not found in parameters or CEREBRAS_API_KEY env var"
            raise RuntimeError(msg)

        self.base_url = "https://api.cerebras.ai/v1"
        self.headers = {"Authorization": f"Bearer {self.api_key}"}
        self.params = {}

    def is_available(self) -> bool:
        """Check whether the provider is available for use."""
        return bool(self.api_key)

    def _parse_model(self, data: dict[str, Any]) -> ModelInfo:
        """Parse Cerebras API response into ModelInfo."""
        # Extract model ID and creation time
        model_id = data.get("id", "")
        created_timestamp = data.get("created")

        # Extract owner information
        owned_by = data.get("owned_by")

        # Context window sizes based on known information about Llama models
        # These are estimates and should be updated with accurate info when available
        context_window = None
        if model_id == "llama3.1-8b":
            context_window = 8192  # Example value
        elif model_id == "llama-3.3-70b":
            context_window = 32768  # Example value

        # Create a description that includes creation timestamp if available
        description = None
        if created_timestamp:
            from datetime import datetime

            try:
                created_date = datetime.fromtimestamp(created_timestamp)
                formatted_date = created_date.strftime("%B %d, %Y")
                description = f"Model released: {formatted_date}"
            except (ValueError, TypeError, OverflowError) as e:
                logger.warning("Could not parse timestamp %s: %s", created_timestamp, e)

        return ModelInfo(
            id=model_id,
            name=model_id,
            provider="cerebras",
            description=description,
            owned_by=owned_by,
            context_window=context_window,
        )

    async def get_models(self) -> list[ModelInfo]:
        """Fetch available models from the Cerebras API."""
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

            if not isinstance(data, dict) or "data" not in data:
                msg = "Invalid response format from Cerebras API"
                raise RuntimeError(msg)

            return [self._parse_model(item) for item in data["data"]]

        except HttpError as e:
            msg = f"Failed to fetch models from Cerebras: {e}"
            raise RuntimeError(msg) from e


if __name__ == "__main__":
    import asyncio

    async def main():
        provider = CerebrasProvider()
        models = await provider.get_models()
        for model in models:
            print(model)

    asyncio.run(main())
