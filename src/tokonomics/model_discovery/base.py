"""Model discovery and information retrieval."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
import pathlib
from typing import ClassVar

import hishel
import httpx


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

    # Cache TTL in seconds (24 hours)
    CACHE_TTL: ClassVar[int] = 86400

    def __init__(self) -> None:
        """Initialize the provider with caching support."""
        # Set up filesystem cache storage
        storage = hishel.AsyncFileStorage(base_path=CACHE_DIR, ttl=self.CACHE_TTL)

        # Configure caching behavior
        controller = hishel.Controller(
            cacheable_methods=["GET"],
            cacheable_status_codes=[200],
            allow_stale=True,
        )

        # Create cached transport
        base_transport = httpx.AsyncHTTPTransport()
        transport = hishel.AsyncCacheTransport(
            transport=base_transport,
            storage=storage,
            controller=controller,
        )

        # Create cached client
        self.client = httpx.AsyncClient(transport=transport)  # type: ignore[arg-type]

    @abstractmethod
    async def get_models(self) -> list[ModelInfo]:
        """Fetch available models from the provider."""
