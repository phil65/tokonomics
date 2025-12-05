"""Comet API provider."""

from __future__ import annotations

import contextlib
from datetime import datetime
import os
from typing import Any

from tokonomics.model_discovery.base import ModelProvider
from tokonomics.model_discovery.model_info import ModelInfo


class CometProvider(ModelProvider):
    """Comet API provider."""

    def __init__(self, api_key: str | None = None):
        super().__init__()
        self.api_key = api_key or os.environ.get("COMET_API_KEY")
        if not self.api_key:
            msg = "Comet API key not found in parameters or COMET_API_KEY env var"
            raise RuntimeError(msg)

        self.base_url = "https://api.cometapi.com/v1"
        self.headers = {"Authorization": f"Bearer {self.api_key}"}
        self.params = {}

    def is_available(self) -> bool:
        """Check whether the provider is available for use."""
        return bool(self.api_key)

    def _parse_model(self, data: dict[str, Any]) -> ModelInfo:
        """Parse Comet API response into ModelInfo."""
        # Parse created timestamp
        created_at = None
        if created_timestamp := data.get("created"):
            with contextlib.suppress(ValueError, TypeError, OverflowError):
                created_at = datetime.fromtimestamp(created_timestamp)

        return ModelInfo(
            id=str(data["id"]),
            name=str(data.get("id")),  # Use id as name since no separate name field
            provider="comet",
            owned_by=str(data.get("owned_by")) if data.get("owned_by") else None,
            is_embedding="embedding" in str(data.get("id", "")).lower(),
            created_at=created_at,
            metadata={
                "object": data.get("object"),
                "root": data.get("root"),
                "parent": data.get("parent"),
                "permissions": data.get("permission", []),
            },
        )


if __name__ == "__main__":
    import asyncio

    provider = CometProvider()
    models = asyncio.run(provider.get_models())
    for model in models:
        print(model.format())
