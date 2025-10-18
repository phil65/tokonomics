"""Clean API models for chat completion models."""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

from schemez import Schema

from tokonomics.data_models import ModelMode, TieredPricingTier


if TYPE_CHECKING:
    from tokonomics.data_models import ChatCompletionModel


class ModelPricing(Schema):
    """Pricing information for chat models."""

    input_cost_per_token: Decimal | None = None
    """Standard input cost per token."""

    output_cost_per_token: Decimal | None = None
    """Standard output cost per token."""

    output_cost_per_reasoning_token: Decimal | None = None
    """Cost per reasoning output token for models like o1."""

    # Caching pricing
    cache_creation_input_token_cost: Decimal | None = None
    """Cost per token for cache creation."""

    cache_read_input_token_cost: Decimal | None = None
    """Cost per token for cache reads."""

    # Contextual pricing
    input_cost_above_128k: Decimal | None = None
    """Input cost per token above 128k context."""

    output_cost_above_128k: Decimal | None = None
    """Output cost per token above 128k context."""

    input_cost_above_200k: Decimal | None = None
    """Input cost per token above 200k context."""

    output_cost_above_200k: Decimal | None = None
    """Output cost per token above 200k context."""

    # Priority/flex pricing
    input_cost_priority: Decimal | None = None
    """Input cost per token for priority requests."""

    output_cost_priority: Decimal | None = None
    """Output cost per token for priority requests."""

    input_cost_flex: Decimal | None = None
    """Input cost per token for flex requests."""

    output_cost_flex: Decimal | None = None
    """Output cost per token for flex requests."""

    # Multimodal pricing
    input_cost_per_image: Decimal | None = None
    """Cost per image input."""

    input_cost_per_audio_second: Decimal | None = None
    """Cost per second of audio input."""

    input_cost_per_video_second: Decimal | None = None
    """Cost per second of video input."""

    # Tiered pricing
    tiered_pricing: list[TieredPricingTier] | None = None
    """Volume-based tiered pricing if available."""

    def calculate_cost(
        self,
        *,
        input_tokens: int = 0,
        cache_write_tokens: int = 0,
        cache_read_tokens: int = 0,
        input_audio_tokens: int = 0,
        cache_audio_read_tokens: int = 0,
        output_tokens: int = 0,
    ) -> Decimal | None:
        """Calculate cost based on usage mirroring pydantic-ai RunUsage.

        Uses contextual pricing if token count exceeds thresholds,
        otherwise uses standard input/output cost per token.

        Args:
            input_tokens: Total number of input/prompt tokens
            cache_write_tokens: Total number of tokens written to the cache
            cache_read_tokens: Total number of tokens read from the cache
            input_audio_tokens: Total number of audio input tokens
            cache_audio_read_tokens: Total number of audio tokens read from the cache
            output_tokens: Total number of output/completion tokens

        Returns:
            Total cost or None if pricing unavailable
        """
        total_cost = Decimal(0)

        # Calculate total input tokens for context threshold determination
        total_input = input_tokens + cache_write_tokens + input_audio_tokens

        # Calculate input cost (standard text tokens)
        if input_tokens > 0 and self.input_cost_per_token is not None:
            if total_input > 200_000 and self.input_cost_above_200k is not None:  # noqa: PLR2004
                input_cost = self.input_cost_above_200k
            elif total_input > 128_000 and self.input_cost_above_128k is not None:  # noqa: PLR2004
                input_cost = self.input_cost_above_128k
            else:
                input_cost = self.input_cost_per_token
            total_cost += input_cost * input_tokens

        # Calculate cache write cost
        if cache_write_tokens > 0:
            # Use cache creation pricing if available, otherwise standard input pricing
            cache_cost = (
                self.cache_creation_input_token_cost
                or self.input_cost_per_token
                or Decimal(0)
            )
            total_cost += cache_cost * cache_write_tokens

        # Calculate cache read cost
        if cache_read_tokens > 0:
            # Use actual cache read pricing if available, otherwise fallback to 10%
            if self.cache_read_input_token_cost is not None:
                cache_read_cost = self.cache_read_input_token_cost
                total_cost += cache_read_cost * cache_read_tokens
            else:
                # Fallback: assume 10% of input cost
                fallback_cost = self.input_cost_per_token or Decimal(0)
                total_cost += fallback_cost * cache_read_tokens * Decimal("0.1")

        # Calculate audio input cost
        if input_audio_tokens > 0:
            audio_cost = self.input_cost_per_token or Decimal(0)
            total_cost += audio_cost * input_audio_tokens

        # Calculate cache audio read cost
        if cache_audio_read_tokens > 0:
            audio_cache_cost = self.input_cost_per_token or Decimal(0)
            total_cost += audio_cache_cost * cache_audio_read_tokens * Decimal("0.1")

        # Calculate output cost
        if output_tokens > 0 and self.output_cost_per_token is not None:
            if total_input > 200_000 and self.output_cost_above_200k is not None:  # noqa: PLR2004
                output_cost = self.output_cost_above_200k
            elif total_input > 128_000 and self.output_cost_above_128k is not None:  # noqa: PLR2004
                output_cost = self.output_cost_above_128k
            else:
                output_cost = self.output_cost_per_token
            total_cost += output_cost * output_tokens

        # Note: tool_calls and requests are currently not charged by most providers

        return total_cost if total_cost > 0 else None


class ModelLimits(Schema):
    """Token and content limits for chat models."""

    max_input_tokens: int | None = None
    """Maximum input tokens supported."""

    max_output_tokens: int | None = None
    """Maximum output tokens supported."""

    max_images_per_prompt: int | None = None
    """Maximum images per prompt."""

    max_videos_per_prompt: int | None = None
    """Maximum videos per prompt."""

    max_video_length_hours: float | None = None
    """Maximum video length in hours."""

    max_audio_per_prompt: int | None = None
    """Maximum audio files per prompt."""

    max_audio_length_hours: float | None = None
    """Maximum audio length in hours."""

    max_pdf_size_mb: float | None = None
    """Maximum PDF size in MB."""


class ModelCapabilities(Schema):
    """Chat model capabilities and features."""

    supports_function_calling: bool = False
    """Whether model supports function calling."""

    supports_parallel_function_calling: bool = False
    """Whether model supports parallel function calling."""

    supports_tool_choice: bool = False
    """Whether model supports tool choice."""

    supports_response_schema: bool = False
    """Whether model supports response schema."""

    supports_system_messages: bool = False
    """Whether model supports system messages."""

    supports_assistant_prefill: bool = False
    """Whether model supports assistant prefill."""

    supports_vision: bool = False
    """Whether model supports vision/image input."""

    supports_pdf_input: bool = False
    """Whether model supports PDF input."""

    supports_audio_input: bool = False
    """Whether model supports audio input."""

    supports_audio_output: bool = False
    """Whether model supports audio output."""

    supports_video_input: bool = False
    """Whether model supports video input."""

    supports_url_context: bool = False
    """Whether model supports URL context."""

    supports_web_search: bool = False
    """Whether model supports web search."""

    supports_computer_use: bool = False
    """Whether model supports computer use."""

    supports_reasoning: bool = False
    """Whether model supports reasoning (like o1)."""


class ChatModel(Schema):
    """Clean API model for chat completion models."""

    name: str
    """Model identifier name."""

    provider: str
    """Model provider (e.g., 'openai', 'anthropic')."""

    mode: ModelMode
    """Model operation mode."""

    pricing: ModelPricing
    """Pricing information and cost calculation."""

    limits: ModelLimits
    """Token and content limits."""

    capabilities: ModelCapabilities
    """Model capabilities and features."""

    @classmethod
    def from_chat_completion_model(
        cls, name: str, model: ChatCompletionModel
    ) -> ChatModel:
        """Convert from internal ChatCompletionModel to clean API model."""
        # Extract pricing info
        pricing = ModelPricing(
            input_cost_per_token=model.input_cost_per_token,
            output_cost_per_token=model.output_cost_per_token,
            output_cost_per_reasoning_token=model.output_cost_per_reasoning_token,
            input_cost_above_128k=model.input_cost_per_token_above_128k_tokens,
            output_cost_above_128k=model.output_cost_per_token_above_128k_tokens,
            input_cost_above_200k=model.input_cost_per_token_above_200k_tokens,
            output_cost_above_200k=model.output_cost_per_token_above_200k_tokens,
            input_cost_priority=model.input_cost_per_token_priority,
            output_cost_priority=model.output_cost_per_token_priority,
            input_cost_flex=model.input_cost_per_token_flex,
            output_cost_flex=model.output_cost_per_token_flex,
            input_cost_per_image=model.input_cost_per_image,
            input_cost_per_audio_second=model.input_cost_per_audio_per_second,
            input_cost_per_video_second=model.input_cost_per_video_per_second,
            cache_creation_input_token_cost=model.cache_creation_input_token_cost,
            cache_read_input_token_cost=model.cache_read_input_token_cost,
            tiered_pricing=model.tiered_pricing,
        )

        # Extract limits
        limits = ModelLimits(
            max_input_tokens=model.max_input_tokens,
            max_output_tokens=model.max_output_tokens,
            max_images_per_prompt=model.max_images_per_prompt,
            max_videos_per_prompt=model.max_videos_per_prompt,
            max_video_length_hours=model.max_video_length,
            max_audio_per_prompt=model.max_audio_per_prompt,
            max_audio_length_hours=model.max_audio_length_hours,
            max_pdf_size_mb=model.max_pdf_size_mb,
        )

        # Extract capabilities
        capabilities = ModelCapabilities(
            supports_function_calling=model.supports_function_calling or False,
            supports_parallel_function_calling=model.supports_parallel_function_calling
            or False,
            supports_tool_choice=model.supports_tool_choice or False,
            supports_response_schema=model.supports_response_schema or False,
            supports_system_messages=model.supports_system_messages or False,
            supports_assistant_prefill=model.supports_assistant_prefill or False,
            supports_vision=model.supports_vision or False,
            supports_pdf_input=model.supports_pdf_input or False,
            supports_audio_input=model.supports_audio_input or False,
            supports_audio_output=model.supports_audio_output or False,
            supports_video_input=model.supports_video_input or False,
            supports_url_context=model.supports_url_context or False,
            supports_web_search=model.supports_web_search or False,
            supports_computer_use=model.supports_computer_use or False,
            supports_reasoning=model.supports_reasoning or False,
        )

        return cls(
            name=name,
            provider=model.litellm_provider,
            mode=model.mode,
            pricing=pricing,
            limits=limits,
            capabilities=capabilities,
        )
