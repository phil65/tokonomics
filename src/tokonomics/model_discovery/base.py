"""Model discovery and information retrieval."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from tokonomics.model_discovery.model_info import ModelInfo


class ModelProvider(ABC):
    """Base class for model providers."""

    def __init__(self) -> None:
        self.base_url: str
        self.headers: dict[str, str] = {}
        self.params: dict[str, Any] = {}

    @abstractmethod
    def _parse_model(self, data: dict[str, Any]) -> ModelInfo:
        """Parse provider-specific API response into ModelInfo."""

    def get_models_sync(self) -> list[ModelInfo]:
        """Fetch available models from the provider synchronously."""
        from anyenv import HttpError, get_json_sync

        url = f"{self.base_url}/models"

        try:
            data = get_json_sync(
                url,
                headers=self.headers,
                params=self.params,
                cache=True,
            )

            if not isinstance(data, dict) or "data" not in data:
                msg = f"Invalid response format from {self.__class__.__name__}"
                raise RuntimeError(msg)

            return [self._parse_model(item) for item in data["data"]]

        except HttpError as e:
            msg = f"Failed to fetch models from {self.__class__.__name__}: {e}"
            raise RuntimeError(msg) from e

    async def get_models(self) -> list[ModelInfo]:
        """Fetch available models from the provider asynchronously."""
        from anyenv import HttpError, get_json

        url = f"{self.base_url}/models"

        try:
            data = await get_json(
                url,
                headers=self.headers,
                params=self.params,
                cache=True,
            )

            if not isinstance(data, dict) or "data" not in data:
                msg = f"Invalid response format from {self.__class__.__name__}"
                raise RuntimeError(msg)

            return [self._parse_model(item) for item in data["data"]]

        except HttpError as e:
            msg = f"Failed to fetch models from {self.__class__.__name__}: {e}"
            raise RuntimeError(msg) from e
