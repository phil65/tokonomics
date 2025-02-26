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
        import json

        import httpx

        from tokonomics.utils import make_request_sync

        url = f"{self.base_url}/models"

        try:
            response = make_request_sync(url, headers=self.headers, params=self.params)
            response.raise_for_status()

            data = response.json()
            if not isinstance(data, dict) or "data" not in data:
                msg = f"Invalid response format from {self.__class__.__name__}"
                raise RuntimeError(msg)

            return [self._parse_model(item) for item in data["data"]]

        except httpx.HTTPError as e:
            msg = f"Failed to fetch models from {self.__class__.__name__}: {e}"
            raise RuntimeError(msg) from e
        except json.JSONDecodeError as e:
            msg = f"Invalid JSON response from {self.__class__.__name__}: {e}"
            raise RuntimeError(msg) from e
        except (KeyError, ValueError) as e:
            msg = f"Invalid data in {self.__class__.__name__} response: {e}"
            raise RuntimeError(msg) from e

    async def get_models(self) -> list[ModelInfo]:
        """Fetch available models from the provider asynchronously."""
        import json

        import httpx

        from tokonomics.utils import make_request

        url = f"{self.base_url}/models"

        try:
            response = await make_request(url, headers=self.headers, params=self.params)
            response.raise_for_status()

            data = response.json()
            if not isinstance(data, dict) or "data" not in data:
                msg = f"Invalid response format from {self.__class__.__name__}"
                raise RuntimeError(msg)

            return [self._parse_model(item) for item in data["data"]]

        except httpx.HTTPError as e:
            msg = f"Failed to fetch models from {self.__class__.__name__}: {e}"
            raise RuntimeError(msg) from e
        except json.JSONDecodeError as e:
            msg = f"Invalid JSON response from {self.__class__.__name__}: {e}"
            raise RuntimeError(msg) from e
        except (KeyError, ValueError) as e:
            msg = f"Invalid data in {self.__class__.__name__} response: {e}"
            raise RuntimeError(msg) from e
