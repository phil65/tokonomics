"""Tokonomics types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict


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
