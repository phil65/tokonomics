"""Claude Code provider with static model definitions."""

from __future__ import annotations

from typing import Any

from tokonomics.model_discovery.base import ModelProvider
from tokonomics.model_discovery.model_info import ModelInfo, ModelPricing


CLAUDE_CODE_MODELS: list[ModelInfo] = [
    ModelInfo(
        id="sonnet",
        name="Claude Sonnet (Claude Code)",
        provider="claude-code",
        description="Latest Claude Sonnet model - balanced performance and speed",
        owned_by="anthropic",
        context_window=200000,
        max_output_tokens=8192,
        input_modalities={"text", "image", "file"},
        output_modalities={"text"},
        pricing=ModelPricing(
            prompt=3.0 / 1_000_000,
            completion=15.0 / 1_000_000,
            input_cache_read=0.30 / 1_000_000,
            input_cache_write=3.75 / 1_000_000,
        ),
    ),
    ModelInfo(
        id="haiku",
        name="Claude Haiku (Claude Code)",
        provider="claude-code",
        description="Latest Claude Haiku model - fast and cost-effective",
        owned_by="anthropic",
        context_window=200000,
        max_output_tokens=8192,
        input_modalities={"text", "image", "file"},
        output_modalities={"text"},
        pricing=ModelPricing(
            prompt=0.80 / 1_000_000,
            completion=4.0 / 1_000_000,
            input_cache_read=0.08 / 1_000_000,
            input_cache_write=1.0 / 1_000_000,
        ),
    ),
    ModelInfo(
        id="opus",
        name="Claude Opus (Claude Code)",
        provider="claude-code",
        description="Latest Claude Opus model - most capable",
        owned_by="anthropic",
        context_window=200000,
        max_output_tokens=64000,
        input_modalities={"text", "image", "file"},
        output_modalities={"text"},
        pricing=ModelPricing(
            prompt=5.0 / 1_000_000,
            completion=25.0 / 1_000_000,
            input_cache_read=0.50 / 1_000_000,
            input_cache_write=6.25 / 1_000_000,
        ),
    ),
]


class ClaudeCodeProvider(ModelProvider):
    """Static provider for Claude Code models."""

    def __init__(self) -> None:
        super().__init__()
        self.base_url = ""
        self.headers = {}
        self.params = {}

    def is_available(self) -> bool:
        """Always available - no external dependencies."""
        return True

    def _parse_model(self, data: dict[str, Any]) -> ModelInfo:
        """Not used for static provider."""
        raise NotImplementedError

    async def get_models(self) -> list[ModelInfo]:
        """Return static list of Claude Code models."""
        return CLAUDE_CODE_MODELS.copy()
