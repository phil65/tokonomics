"""Thinking level variants for different model providers.

This module generates provider-specific model settings for different thinking/reasoning
effort levels. The variants are formatted for pydantic-ai's ModelSettings.

Based on OpenCode's ProviderTransform.variants() implementation.
"""

from __future__ import annotations

from typing import Any


# Common effort level sets
WIDELY_SUPPORTED_EFFORTS = ["low", "medium", "high"]
OPENAI_EFFORTS = ["low", "medium", "high"]  # Could add "none", "minimal", "xhigh" later


def get_pydantic_ai_variants(  # noqa: PLR0911
    provider: str,
    model_id: str,
    supports_reasoning: bool,
) -> dict[str, dict[str, Any]]:
    """Generate thinking level variants based on provider and model capabilities.

    Args:
        provider: Provider name (e.g., "anthropic", "openai", "google")
        model_id: Model identifier
        supports_reasoning: Whether the model supports reasoning/thinking

    Returns:
        Dict mapping variant name to pydantic-ai ModelSettings.
        Empty dict if model doesn't support reasoning.
    """
    if not supports_reasoning:
        return {}

    model_id_lower = model_id.lower()

    # Some models with reasoning=True don't actually support configurable variants
    if any(x in model_id_lower for x in ("deepseek", "minimax", "glm", "mistral")):
        return {}

    # Anthropic models (Claude with extended thinking)
    if provider == "anthropic":
        return {
            "high": {"thinking": {"type": "enabled", "budgetTokens": 16000}},
            "max": {"thinking": {"type": "enabled", "budgetTokens": 31999}},
        }

    # OpenAI models (o1, o3, gpt-5, etc.)
    if provider == "openai":
        if model_id_lower == "o1-mini":
            return {}
        efforts = list(OPENAI_EFFORTS)
        # GPT-5 models support minimal
        if "gpt-5" in model_id_lower:
            efforts.insert(0, "minimal")
        return {
            effort: {
                "reasoningEffort": effort,
                "reasoningSummary": "auto",
            }
            for effort in efforts
        }

    # Azure (same as OpenAI structure)
    if provider == "azure":
        if model_id_lower == "o1-mini":
            return {}
        efforts = list(OPENAI_EFFORTS)
        if "gpt-5" in model_id_lower:
            efforts.insert(0, "minimal")
        return {
            effort: {
                "reasoningEffort": effort,
                "reasoningSummary": "auto",
            }
            for effort in efforts
        }

    # Google models (Gemini with thinking)
    if provider in ("google", "google-vertex"):
        # Gemini 2.5 uses thinkingBudget
        if "2.5" in model_id_lower or "2-5" in model_id_lower:
            return {
                "high": {"thinkingConfig": {"includeThoughts": True, "thinkingBudget": 16000}},
                "max": {"thinkingConfig": {"includeThoughts": True, "thinkingBudget": 24576}},
            }
        # Gemini 3 uses thinkingLevel
        return {
            "low": {"includeThoughts": True, "thinkingLevel": "low"},
            "high": {"includeThoughts": True, "thinkingLevel": "high"},
        }

    # Amazon Bedrock
    if provider == "amazon-bedrock":
        # Anthropic models on Bedrock use reasoningConfig
        if "anthropic" in model_id_lower or "claude" in model_id_lower:
            return {
                "high": {"reasoningConfig": {"type": "enabled", "budgetTokens": 16000}},
                "max": {"reasoningConfig": {"type": "enabled", "budgetTokens": 31999}},
            }
        # Nova models use maxReasoningEffort
        return {
            effort: {"reasoningConfig": {"type": "enabled", "maxReasoningEffort": effort}}
            for effort in WIDELY_SUPPORTED_EFFORTS
        }

    # Google Vertex with Anthropic models
    if provider == "google-vertex-anthropic":
        return {
            "high": {"thinking": {"type": "enabled", "budgetTokens": 16000}},
            "max": {"thinking": {"type": "enabled", "budgetTokens": 31999}},
        }

    # xAI (Grok)
    if provider == "xai":
        return {effort: {"reasoningEffort": effort} for effort in WIDELY_SUPPORTED_EFFORTS}

    # Groq
    if provider == "groq":
        efforts = ["none", *WIDELY_SUPPORTED_EFFORTS]
        return {effort: {"includeThoughts": True, "thinkingLevel": effort} for effort in efforts}

    # Cerebras, TogetherAI, DeepInfra - use reasoningEffort
    if provider in ("cerebras", "togetherai", "deepinfra"):
        return {effort: {"reasoningEffort": effort} for effort in WIDELY_SUPPORTED_EFFORTS}

    # OpenRouter - needs special handling based on underlying model
    if provider == "openrouter":
        # Only some models support reasoning on OpenRouter
        if any(x in model_id_lower for x in ("gpt", "gemini-3", "grok")):
            return {effort: {"reasoning": {"effort": effort}} for effort in OPENAI_EFFORTS}
        return {}

    # Default: no variants for unknown providers
    return {}
