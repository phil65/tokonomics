"""Tokonomics types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any, TypedDict

from pydantic import BaseModel, Field


class ModelCosts(TypedDict):
    """Cost information for a model."""

    input_cost_per_token: float
    output_cost_per_token: float


class TokenUsage(TypedDict):
    """Token usage statistics from model responses."""

    total: int
    """Total tokens used"""
    prompt: int
    """Tokens used in the prompt"""
    completion: int
    """Tokens used in the completion"""


@dataclass(frozen=True, slots=True)
class TokenCosts:
    """Detailed breakdown of token costs."""

    prompt_cost: float
    """Cost for prompt tokens"""
    completion_cost: float
    """Cost for completion tokens"""

    @property
    def total_cost(self) -> float:
        """Calculate total cost as sum of prompt and completion costs."""
        return self.prompt_cost + self.completion_cost


@dataclass(frozen=True, slots=True)
class TokenLimits:
    """Token limits for a model."""

    total_tokens: int
    """Maximum total tokens (input + output) supported"""
    input_tokens: int
    """Maximum input/prompt tokens supported"""
    output_tokens: int
    """Maximum output/completion tokens supported"""


@dataclass(frozen=True, slots=True)
class ModelCapabilities:
    """Capabilities of a model."""

    max_tokens: int
    """Legacy parameter for maximum tokens"""
    max_input_tokens: int
    """Maximum input tokens supported"""
    max_output_tokens: int
    """Maximum output tokens supported"""
    litellm_provider: str | None
    """LiteLLM provider name"""
    mode: str | None
    """Model operation mode"""
    supports_function_calling: bool
    """Whether the model supports function calling"""
    supports_parallel_function_calling: bool
    """Whether the model supports parallel function calling"""
    supports_vision: bool
    """Whether the model supports vision/image input"""
    supports_audio_input: bool
    """Whether the model supports audio input"""
    supports_audio_output: bool
    """Whether the model supports audio output"""
    supports_prompt_caching: bool
    """Whether the model supports prompt caching"""
    supports_response_schema: bool
    """Whether the model supports response schema"""
    supports_system_messages: bool
    """Whether the model supports system messages"""


# Custom field types with 'field_type' metadata for UI rendering hints.


ModelIdentifier = Annotated[
    str,
    Field(
        json_schema_extra={"field_type": "model_identifier"},
        pattern=r"^[a-zA-Z0-9\-]+(/[a-zA-Z0-9\-]+)*(:[\w\-\.]+)?$",
    ),
]

Temperature = Annotated[
    float,
    Field(json_schema_extra={"field_type": "parameter", "step": 0.1}, ge=0.0, le=2.0),
]


# Helper function to extract field type metadata
def get_field_type(model: type[BaseModel], field_name: str) -> dict[str, Any]:
    """Extract field_type metadata from a model field."""
    field_info = model.model_fields[field_name]
    metadata = {}
    if field_info.json_schema_extra and isinstance(field_info.json_schema_extra, dict):
        metadata.update(field_info.json_schema_extra)

    return metadata


def render_field(model: type[BaseModel], field_name: str) -> str:
    """Example function demonstrating how to use field type metadata for UI rendering."""
    metadata = get_field_type(model, field_name)
    field_type = metadata.get("field_type", "text")
    if field_type == "model_identifier":
        provider = metadata.get("provider")
        if provider:
            return f"Model selector dropdown for {provider} provider"
        return "Generic model identifier selector"

    return "Default text input"


if __name__ == "__main__":

    class AIConfig(BaseModel):
        """AI Configuration with semantically typed fields."""

        model: ModelIdentifier = "gpt-4"

    config = AIConfig(model="gpt-4")

    print(type(config.model))
