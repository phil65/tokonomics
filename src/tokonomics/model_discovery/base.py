"""Model discovery and information retrieval."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


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


class ModelProvider(ABC):
    """Base class for model providers."""

    @abstractmethod
    async def get_models(self) -> list[ModelInfo]:
        """Fetch available models from the provider."""
