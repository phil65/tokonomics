"""Model discovery and information retrieval."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


Modality = Literal["text", "image", "audio", "video"]


@dataclass
class ModelPricing:
    """Pricing information for a model."""

    prompt: float | None = None
    """Cost per token for prompt inputs."""
    completion: float | None = None
    """Cost per token for completion outputs."""
    image: float | None = None
    """Cost per image for input processing."""
    request: float | None = None
    """Cost per request."""
    input_cache_read: float | None = None
    """Cost for reading from input cache."""
    input_cache_write: float | None = None
    """Cost for writing to input cache."""
    web_search: float | None = None
    """Cost for web search functionality."""
    internal_reasoning: float | None = None
    """Cost for internal reasoning functionality."""


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
    is_embedding: bool = False
    """Whether this model is primarily used for creating embeddings."""
    max_output_tokens: int | None = None
    """Maximum number of tokens the model can generate in a response."""
    input_modalities: set[Modality] = field(default_factory=lambda: {"text"})
    """Supported input modalities (text, image, audio, video, etc.)."""
    output_modalities: set[Modality] = field(default_factory=lambda: {"text"})
    """Supported output modalities (text, image, audio, video, etc.)."""
    is_free: bool = False
    """Whether this model is free to use."""
    metadata: dict[str, Any] = field(default_factory=dict)
    """Provider-specific metadata that doesn't fit in standard fields."""

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
        name = self.name.lower()
        if name.startswith("mistral"):
            return "logos:mistral-ai"
        if name.startswith("openai"):
            return "logos:openai"
        if name.startswith("claude"):
            return "logos:anthropic"
        if name.startswith("perplexity"):
            return "logos:perplexity"
        if name.startswith("hugging"):
            return "logos:hugging-face"
        if name.startswith("deepseek"):
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
            if self.pricing.image is not None and self.pricing.image > 0:
                lines.append(f"Image cost: ${self.pricing.image:.6f}/image")
            if self.pricing.request is not None and self.pricing.request > 0:
                lines.append(f"Request cost: ${self.pricing.request:.6f}/request")
            if self.pricing.web_search is not None and self.pricing.web_search > 0:
                lines.append(f"Web search cost: ${self.pricing.web_search:.6f}")
            if (
                self.pricing.internal_reasoning is not None
                and self.pricing.internal_reasoning > 0
            ):
                lines.append(
                    f"Internal reasoning cost: ${self.pricing.internal_reasoning:.6f}"
                )

        if self.input_modalities and self.input_modalities != {"text"}:
            lines.append(f"Input modalities: {', '.join(self.input_modalities)}")

        if self.output_modalities and self.output_modalities != {"text"}:
            lines.append(f"Output modalities: {', '.join(self.output_modalities)}")

        if self.description:
            lines.append("\nDescription:")
            lines.append(self.description)

        if self.is_deprecated:
            lines.append("\n⚠️ This model is deprecated")

        # Add any relevant metadata
        if self.metadata and any(self.metadata.values()):
            lines.append("\nAdditional Information:")
            for key, value in self.metadata.items():
                if key == "supported_parameters" and value:
                    lines.append(
                        f"- Supported parameters: {', '.join(str(v) for v in value)}"
                    )
                elif key == "hugging_face_id" and value:
                    lines.append(f"- HuggingFace ID: {value}")
                elif key == "tokenizer" and value:
                    lines.append(f"- Tokenizer: {value}")
                elif key == "is_moderated" and value:
                    lines.append(f"- Moderated: {value}")
                elif key == "created_timestamp" and value:
                    from datetime import datetime

                    timestamp = datetime.fromtimestamp(value).strftime("%Y-%m-%d")
                    lines.append(f"- Created: {timestamp}")
                elif isinstance(value, list | set) and value:
                    vals = ", ".join(str(v) for v in value)
                    lines.append(f"- {key.replace('_', ' ').title()}: {vals}")
                elif value:
                    lines.append(f"- {key.replace('_', ' ').title()}: {value}")

        return "\n".join(lines)
