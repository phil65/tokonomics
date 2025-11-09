"""Tokonomics types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypedDict


if TYPE_CHECKING:
    from decimal import Decimal


class ModelCosts(TypedDict):
    """Cost information for a model."""

    input_cost_per_token: Decimal
    output_cost_per_token: Decimal


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

    input_cost: Decimal
    """Cost for prompt tokens"""
    output_cost: Decimal
    """Cost for completion tokens"""

    @property
    def total_cost(self) -> Decimal:
        """Calculate total cost as sum of input and output costs."""
        return self.input_cost + self.output_cost


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
