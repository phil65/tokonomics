"""Model discovery and information retrieval."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class ModelPricing:
    """Pricing information for a model."""

    prompt: float | None = None
    """Cost per token for prompt inputs."""
    completion: float | None = None
    """Cost per token for completion outputs."""


Modality = Literal["text", "image", "audio", "video"]


@dataclass
class ModelInfo:
    """Unified model information from various providers."""

    id: str
    """Unique identifier for the model."""
    name: str
    """Display name of the model."""
    provider: str
    """Service provider name (e.g. OpenAI, Anthropic)."""
    description: str | None = None
    """Detailed description of the model's capabilities."""
    pricing: ModelPricing | None = None
    """Pricing information for using the model."""
    owned_by: str | None = None
    """Organization that owns/created the model."""
    context_window: int | None = None
    """Maximum number of tokens that can be processed in one request."""
    is_deprecated: bool = False
    """Whether this model version is deprecated."""
    max_output_tokens: int | None = None
    """Maximum number of tokens the model can generate in a response."""
    input_modalities: list[Modality] = field(default_factory=lambda: ["text"])
    """Supported input modalities (text, image, audio, video, etc.)."""
    output_modalities: list[Modality] = field(default_factory=lambda: ["text"])
    """Supported output modalities (text, image, audio, video, etc.)."""
    is_free: bool = False
    """Whether this model is free to use."""

    @property
    def pydantic_ai_id(self) -> str:
        """Unique pydantic-ai style identifier for the model."""
        return f"{self.provider}:{self.id}"

    @property
    def litellm_id(self) -> str:
        """Unique litellm style identifier for the model."""
        return f"{self.provider}/{self.id}"

    @property
    def iconify_icon(self) -> str | None:  # noqa: PLR0911
        """Iconify icon for the model."""
        if self.name.startswith("mistral"):
            return "logos:mistral-ai"
        if self.name.startswith("openai"):
            return "logos:openai"
        if self.name.startswith("claude"):
            return "logos:anthropic"
        if self.name.startswith("perplexity"):
            return "logos:perplexity"
        if self.name.startswith("hugging"):
            return "logos:hugging-face"
        if self.name.startswith("deepseek"):
            return "arcticons:deepseek"

        return None

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

        if self.max_output_tokens:
            lines.append(f"Max output tokens: {self.max_output_tokens:,}")

        if self.pricing:
            if self.pricing.prompt is not None:
                lines.append(f"Prompt cost: ${self.pricing.prompt:.6f}/token")
            if self.pricing.completion is not None:
                lines.append(f"Completion cost: ${self.pricing.completion:.6f}/token")

        if self.input_modalities and self.input_modalities != ["text"]:
            lines.append(f"Input modalities: {', '.join(self.input_modalities)}")

        if self.output_modalities and self.output_modalities != ["text"]:
            lines.append(f"Output modalities: {', '.join(self.output_modalities)}")

        if self.description:
            lines.append("\nDescription:")
            lines.append(self.description)

        if self.is_deprecated:
            lines.append("\n⚠️ This model is deprecated")

        return "\n".join(lines)
