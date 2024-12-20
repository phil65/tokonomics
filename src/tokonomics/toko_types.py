"""Tokonomics types."""

from __future__ import annotations

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
