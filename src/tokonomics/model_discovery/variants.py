"""Thinking/reasoning level configuration for different model providers.

This module provides functions to discover and configure thinking/reasoning levels
for models that support extended thinking capabilities.

Pydantic-ai uses provider-prefixed settings names:
- Anthropic: `anthropic_thinking`
- OpenAI: `openai_reasoning_effort`, `openai_reasoning_summary`
- Google: `google_thinking_config`
"""

from __future__ import annotations

from typing import Any, Literal


ReasoningLevel = Literal["off", "minimal", "low", "medium", "high", "max"]


def supports_reasoning(provider: str, model_id: str) -> bool:
    """Check if a model supports configurable reasoning/thinking levels.

    Args:
        provider: Provider name (e.g., "anthropic", "openai", "google")
        model_id: Model identifier

    Returns:
        True if the model supports configurable reasoning levels.
    """
    model_id_lower = model_id.lower()

    # Models with reasoning capability but no configurable variants
    if any(x in model_id_lower for x in ("deepseek", "minimax", "glm", "mistral")):
        return False

    # OpenAI o1-mini doesn't support reasoning effort
    if provider == "openai" and model_id_lower == "o1-mini":
        return False

    # Supported providers (pydantic-ai and models.dev provider names)
    supported_providers = {
        "anthropic",
        "openai",
        "azure",
        "google",
        "google-gla",
        "google-vertex",
        "bedrock",
        "amazon-bedrock",
    }

    return provider in supported_providers


def get_reasoning_levels(provider: str, model_id: str) -> list[str]:  # noqa: PLR0911
    """Get available reasoning levels for a model.

    Args:
        provider: Provider name (e.g., "anthropic", "openai", "google")
        model_id: Model identifier

    Returns:
        List of available level names, empty if reasoning not supported.
    """
    if not supports_reasoning(provider, model_id):
        return []

    model_id_lower = model_id.lower()

    # Anthropic models (native or via Bedrock)
    if provider == "anthropic":
        return ["off", "high", "max"]

    # AWS Bedrock with Anthropic models
    if provider in ("bedrock", "amazon-bedrock"):
        if "anthropic" in model_id_lower or "claude" in model_id_lower:
            return ["off", "high", "max"]
        return []

    # OpenAI models
    if provider in ("openai", "azure"):
        levels = ["off", "low", "medium", "high"]
        if "gpt-5" in model_id_lower:
            levels.insert(1, "minimal")  # after "off"
        return levels

    # Google models
    if provider in ("google", "google-gla", "google-vertex"):
        if "2.5" in model_id_lower or "2-5" in model_id_lower:
            return ["off", "high", "max"]
        return ["off", "low", "high"]

    return []


def get_reasoning_settings(provider: str, model_id: str, level: str) -> dict[str, Any]:  # noqa: PLR0911
    """Get pydantic-ai ModelSettings for a reasoning level.

    Args:
        provider: Provider name (e.g., "anthropic", "openai", "google")
        model_id: Model identifier
        level: Reasoning level (e.g., "high", "max", "low", "medium")

    Returns:
        Dict of pydantic-ai ModelSettings for the specified level.

    Raises:
        ValueError: If level is not valid for this provider/model.
    """
    available = get_reasoning_levels(provider, model_id)
    if not available:
        msg = f"Model {provider}:{model_id} does not support configurable reasoning"
        raise ValueError(msg)

    if level not in available:
        msg = f"Invalid level '{level}' for {provider}:{model_id}. Available: {available}"
        raise ValueError(msg)

    model_id_lower = model_id.lower()

    # Anthropic models (native or via Bedrock)
    if provider == "anthropic":
        if level == "off":
            return {"anthropic_thinking": {"type": "disabled"}}
        budgets = {"high": 16000, "max": 31999}
        return {"anthropic_thinking": {"type": "enabled", "budget_tokens": budgets[level]}}

    # AWS Bedrock with Anthropic models
    if provider in ("bedrock", "amazon-bedrock") and (
        "anthropic" in model_id_lower or "claude" in model_id_lower
    ):
        if level == "off":
            return {"anthropic_thinking": {"type": "disabled"}}
        budgets = {"high": 16000, "max": 31999}
        return {"anthropic_thinking": {"type": "enabled", "budget_tokens": budgets[level]}}

    # OpenAI models
    if provider in ("openai", "azure"):
        if level == "off":
            return {}  # No reasoning settings = disabled
        return {
            "openai_reasoning_effort": level,
            "openai_reasoning_summary": "auto",
        }

    # Google models
    if provider in ("google", "google-gla", "google-vertex"):
        if level == "off":
            return {"google_thinking_config": {"include_thoughts": False}}
        if "2.5" in model_id_lower or "2-5" in model_id_lower:
            budgets = {"high": 16000, "max": 24576}
            return {
                "google_thinking_config": {
                    "include_thoughts": True,
                    "thinking_budget": budgets[level],
                }
            }
        # Gemini 3+ just enables thinking, level doesn't map to budget
        return {"google_thinking_config": {"include_thoughts": True}}

    # Should not reach here if supports_reasoning is correct
    msg = f"No settings implementation for {provider}:{model_id}"
    raise ValueError(msg)


# Keep the old function for backwards compatibility with ModelInfo.pydantic_ai_variants
def get_pydantic_ai_variants(
    provider: str,
    model_id: str,
    supports_reasoning_flag: bool,
) -> dict[str, dict[str, Any]]:
    """Generate thinking level variants based on provider and model capabilities.

    Args:
        provider: Provider name (e.g., "anthropic", "openai", "google")
        model_id: Model identifier
        supports_reasoning_flag: Whether the model supports reasoning/thinking
            (from models.dev metadata)

    Returns:
        Dict mapping variant name to pydantic-ai ModelSettings.
        Empty dict if model doesn't support reasoning.
    """
    if not supports_reasoning_flag:
        return {}

    levels = get_reasoning_levels(provider, model_id)
    if not levels:
        return {}

    return {level: get_reasoning_settings(provider, model_id, level) for level in levels}
