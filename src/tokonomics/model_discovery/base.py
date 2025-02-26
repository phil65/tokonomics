"""Model discovery and information retrieval."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
import pathlib
from typing import Any


logger = logging.getLogger(__name__)

# Cache directory for model information
CACHE_DIR = pathlib.Path("~/.cache/tokonomics/models").expanduser()
CACHE_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class ModelPricing:
    """Pricing information for a model."""

    prompt: float | None = None
    completion: float | None = None


@dataclass
class ModelInfo:
    """Unified model information from various providers."""

    id: str
    name: str
    provider: str
    description: str | None = None
    pricing: ModelPricing | None = None
    owned_by: str | None = None
    context_window: int | None = None
    is_deprecated: bool = False

    def format(self) -> str:
        """Format model information as a human-readable string.

        Returns:
            str: Formatted model information
        """
        lines: list[str] = []

        # Basic info
        lines.append(f"Model: {self.name}")
        lines.append(f"Provider: {self.provider}")
        lines.append(f"ID: {self.id}")

        # Optional fields
        if self.owned_by:
            lines.append(f"Owned by: {self.owned_by}")

        if self.context_window:
            lines.append(f"Context window: {self.context_window:,} tokens")

        if self.pricing:
            if self.pricing.prompt is not None:
                lines.append(f"Prompt cost: ${self.pricing.prompt:.6f}/token")
            if self.pricing.completion is not None:
                lines.append(f"Completion cost: ${self.pricing.completion:.6f}/token")

        if self.description:
            lines.append("\nDescription:")
            lines.append(self.description)

        if self.is_deprecated:
            lines.append("\n⚠️ This model is deprecated")

        return "\n".join(lines)


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
