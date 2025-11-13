"""Tokonomics: main package.

Calcuate costs for LLM Usage based on token count.
"""

from __future__ import annotations

from importlib.metadata import version

__version__ = version("tokonomics")
__title__ = "Tokonomics"

__author__ = "Philipp Temminghoff"
__author_email__ = "philipptemminghoff@googlemail.com"
__copyright__ = "Copyright (c) 2024 Philipp Temminghoff"
__license__ = "MIT"
__url__ = "https://github.com/phil65/tokonomics"

from tokonomics.core import (
    get_model_costs,
    calculate_token_cost,
    get_model_capabilities,
    get_available_models,
    reset_cache,
)
from tokonomics.toko_types import ModelCosts, TokenCosts
from tokonomics.pydanticai_cost import calculate_pydantic_cost, Usage
from tokonomics.model_discovery import (
    AnthropicProvider,
    MistralProvider,
    OpenRouterProvider,
    OpenAIProvider,
    GroqProvider,
    ModelInfo,
    ModelPricing,
    ModelProvider,
    get_all_models,
)
from tokonomics.model_discovery.copilot_provider import CopilotTokenManager
from tokonomics.token_count import count_tokens
from tokonomics.model_names import ModelName


__all__ = [
    "AnthropicProvider",
    "CopilotTokenManager",
    "GroqProvider",
    "MistralProvider",
    "ModelCosts",
    "ModelInfo",
    "ModelName",
    "ModelPricing",
    "ModelProvider",
    "OpenAIProvider",
    "OpenRouterProvider",
    "TokenCosts",
    "Usage",
    "__version__",
    "calculate_pydantic_cost",
    "calculate_token_cost",
    "count_tokens",
    "get_all_models",
    "get_available_models",
    "get_model_capabilities",
    "get_model_costs",
    "reset_cache",
]
