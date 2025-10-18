"""Modern Pydantic models for LLM pricing configuration using discriminated unions."""

from __future__ import annotations

from decimal import Decimal
from enum import StrEnum
from typing import Any, Literal

from pydantic import Field
from schemez import Schema


# Literal type definitions for enum fields
SupportedRegion = Literal["global", "us-west2"]

SupportedEndpoint = Literal[
    "/v1/audio/speech",
    "/v1/audio/transcriptions",
    "/v1/batch",
    "/v1/chat/completions",
    "/v1/completions",
    "/v1/embeddings",
    "/v1/images/edits",
    "/v1/images/generations",
    "/v1/realtime",
    "/v1/responses",
]

SupportedModality = Literal["audio", "image", "text", "video"]

SupportedOutputModality = Literal["audio", "code", "image", "text", "video"]

LiteLLMProvider = Literal[
    "ai21",
    "aiml",
    "aleph_alpha",
    "anthropic",
    "anyscale",
    "assemblyai",
    "azure",
    "azure_ai",
    "azure_text",
    "bedrock",
    "bedrock_converse",
    "cerebras",
    "cloudflare",
    "codestral",
    "cohere",
    "cohere_chat",
    "dashscope",
    "databricks",
    "deepgram",
    "deepinfra",
    "deepseek",
    "elevenlabs",
    "featherless_ai",
    "fireworks_ai",
    "fireworks_ai-embedding-models",
    "friendliai",
    "gemini",
    "gradient_ai",
    "groq",
    "heroku",
    "hyperbolic",
    "jina_ai",
    "lambda_ai",
    "lemonade",
    "meta_llama",
    "mistral",
    "moonshot",
    "morph",
    "nlp_cloud",
    "nscale",
    "nvidia_nim",
    "oci",
    "ollama",
    "openai",
    "openrouter",
    "ovhcloud",
    "palm",
    "perplexity",
    "recraft",
    "replicate",
    "sagemaker",
    "sambanova",
    "snowflake",
    "text-completion-codestral",
    "text-completion-openai",
    "together_ai",
    "v0",
    "vercel_ai_gateway",
    "vertex_ai-ai21_models",
    "vertex_ai-anthropic_models",
    "vertex_ai-chat-models",
    "vertex_ai-code-chat-models",
    "vertex_ai-code-text-models",
    "vertex_ai-deepseek_models",
    "vertex_ai-embedding-models",
    "vertex_ai-image-models",
    "vertex_ai-language-models",
    "vertex_ai-llama_models",
    "vertex_ai-mistral_models",
    "vertex_ai-openai_models",
    "vertex_ai-qwen_models",
    "vertex_ai-text-models",
    "vertex_ai-video-models",
    "vertex_ai-vision-models",
    "volcengine",
    "voyage",
    "wandb",
    "watsonx",
    "xai",
]


class ModelMode(StrEnum):
    """Model operation modes."""

    CHAT = "chat"
    COMPLETION = "completion"
    EMBEDDING = "embedding"
    AUDIO_TRANSCRIPTION = "audio_transcription"
    AUDIO_SPEECH = "audio_speech"
    IMAGE_GENERATION = "image_generation"
    VIDEO_GENERATION = "video_generation"
    RERANK = "rerank"
    RESPONSES = "responses"
    MODERATION = "moderation"


class TieredPricingTier(Schema):
    """Single tier in a tiered pricing structure."""

    range: tuple[float, float]
    """Token range [start, end] for this tier."""

    input_cost_per_token: Decimal = Field(ge=0)
    """Input token cost for this tier."""

    output_cost_per_token: Decimal = Field(ge=0)
    """Output token cost for this tier."""

    output_cost_per_reasoning_token: Decimal | None = Field(None, ge=0)
    """Output cost per reasoning token for this tier."""

    cache_read_input_token_cost: Decimal | None = Field(None, ge=0)
    """Cache read input token cost for this tier."""


class SearchContextCost(Schema):
    """Search context pricing structure."""

    search_context_size_low: Decimal = Field(ge=0)
    """Cost for low search context size."""

    search_context_size_medium: Decimal = Field(ge=0)
    """Cost for medium search context size."""

    search_context_size_high: Decimal = Field(ge=0)
    """Cost for high search context size."""


class BaseModelConfig(Schema):
    """Base configuration shared by all model types."""

    # Core identification - always present
    litellm_provider: LiteLLMProvider | str
    """LiteLLM provider identifier."""

    # Common optional fields
    max_tokens: int | None = Field(None, ge=0)
    """Maximum total tokens."""

    source: str | None = None
    """Pricing source URL."""

    metadata: dict[str, Any] | None = None
    """Additional metadata."""

    deprecation_date: str | None = None
    """Model deprecation date."""

    # Rate limits
    rpm: int | None = Field(None, ge=0)
    """Requests per minute."""

    tpm: int | None = Field(None, ge=0)
    """Tokens per minute."""

    # Advanced pricing structures
    tiered_pricing: list[TieredPricingTier] | None = None
    """Volume-based pricing."""

    search_context_cost_per_query: SearchContextCost | None = None
    """Search context cost per query."""

    tool_use_system_prompt_tokens: int | None = Field(None, ge=0)
    """Tool use system prompt tokens."""


class TextGenerationModel(BaseModelConfig):
    """Base for text generation models (chat, completion, responses)."""

    # Token limits
    max_input_tokens: int | None = Field(None, ge=0)
    """Maximum input tokens."""

    max_output_tokens: int | None = Field(None, ge=0)
    """Maximum output tokens."""

    # Basic pricing
    input_cost_per_token: Decimal | None = Field(None, ge=0)
    """Cost per input token."""

    output_cost_per_token: Decimal | None = Field(None, ge=0)
    """Cost per output token."""

    # Batch processing pricing
    input_cost_per_token_batches: Decimal | None = Field(None, ge=0)
    """Input cost per token for batch processing."""

    output_cost_per_token_batches: Decimal | None = Field(None, ge=0)
    """Output cost per token for batch processing."""

    # Caching pricing
    cache_read_input_token_cost: Decimal | None = Field(None, ge=0)
    """Cache read input token cost."""

    # Core capabilities
    supports_function_calling: bool | None = None
    """Whether model supports function calling."""

    supports_parallel_function_calling: bool | None = None
    """Whether model supports parallel function calling."""

    supports_pdf_input: bool | None = None
    """Whether model supports PDF input."""

    supports_prompt_caching: bool | None = None
    """Whether model supports prompt caching."""

    supports_reasoning: bool | None = None
    """Whether model supports reasoning."""

    supports_response_schema: bool | None = None
    """Whether model supports response schema."""

    supports_system_messages: bool | None = None
    """Whether model supports system messages."""

    supports_tool_choice: bool | None = None
    """Whether model supports tool choice."""

    supports_vision: bool | None = None
    """Whether model supports vision/image input."""

    supports_native_streaming: bool | None = None
    """Whether model supports native streaming."""

    supports_web_search: bool | None = None
    """Whether model supports web search."""

    # API support
    supported_endpoints: list[SupportedEndpoint] | None = None
    """List of supported API endpoints."""

    supported_modalities: list[SupportedModality] | None = None
    """List of supported input modalities."""

    supported_output_modalities: list[SupportedOutputModality] | None = None
    """List of supported output modalities."""


class ChatCompletionModel(TextGenerationModel):
    """Chat and completion models with comprehensive pricing and capability support."""

    mode: Literal[ModelMode.CHAT, ModelMode.COMPLETION]
    """Model operation mode."""

    # Extended pricing options
    input_cost_per_token_above_200k_tokens: Decimal | None = Field(None, ge=0)
    """Input cost per token above 200k tokens."""

    output_cost_per_token_above_200k_tokens: Decimal | None = Field(None, ge=0)
    """Output cost per token above 200k tokens."""

    input_cost_per_token_above_1hr: Decimal | None = Field(None, ge=0)
    """Input cost per token above 1 hour."""

    input_cost_per_token_above_128k_tokens: Decimal | None = Field(None, ge=0)
    """Input cost per token above 128k tokens."""

    output_cost_per_token_above_128k_tokens: Decimal | None = Field(None, ge=0)
    """Output cost per token above 128k tokens."""

    output_cost_per_reasoning_token: Decimal | None = Field(None, ge=0)
    """Cost per reasoning output token."""

    # Priority pricing
    cache_read_input_token_cost_priority: Decimal | None = Field(None, ge=0)
    """Cache read input token cost for priority requests."""

    input_cost_per_token_priority: Decimal | None = Field(None, ge=0)
    """Input cost per token for priority requests."""

    output_cost_per_token_priority: Decimal | None = Field(None, ge=0)
    """Output cost per token for priority requests."""

    # Flex pricing
    cache_read_input_token_cost_flex: Decimal | None = Field(None, ge=0)
    """Cache read input token cost for flex requests."""

    input_cost_per_token_flex: Decimal | None = Field(None, ge=0)
    """Input cost per token for flex requests."""

    output_cost_per_token_flex: Decimal | None = Field(None, ge=0)
    """Output cost per token for flex requests."""

    # Image token pricing
    cache_read_input_image_token_cost: Decimal | None = Field(None, ge=0)
    """Cache read input image token cost."""

    input_cost_per_image_token: Decimal | None = Field(None, ge=0)
    """Input cost per image token."""

    output_cost_per_image_token: Decimal | None = Field(None, ge=0)
    """Output cost per image token."""

    # Request-based pricing
    input_cost_per_request: Decimal | None = Field(None, ge=0)
    """Input cost per request."""

    citation_cost_per_token: Decimal | None = Field(None, ge=0)
    """Citation cost per token."""

    # OpenAI service-specific costs
    code_interpreter_cost_per_session: Decimal | None = Field(None, ge=0)
    """Code interpreter cost per session."""

    computer_use_input_cost_per_1k_tokens: Decimal | None = Field(None, ge=0)
    """Computer use input cost per 1k tokens."""

    computer_use_output_cost_per_1k_tokens: Decimal | None = Field(None, ge=0)
    """Computer use output cost per 1k tokens."""

    file_search_cost_per_1k_calls: Decimal | None = Field(None, ge=0)
    """File search cost per 1k calls."""

    file_search_cost_per_gb_per_day: Decimal | None = Field(None, ge=0)
    """File search cost per GB per day."""

    vector_store_cost_per_gb_per_day: Decimal | None = Field(None, ge=0)
    """Vector store cost per GB per day."""

    # Caching pricing
    cache_creation_input_token_cost: Decimal | None = Field(None, ge=0)
    """Cache creation input token cost."""

    cache_creation_input_token_cost_above_200k_tokens: Decimal | None = Field(None, ge=0)
    """Cache creation input token cost above 200k tokens."""

    cache_creation_input_token_cost_above_1hr: Decimal | None = Field(None, ge=0)
    """Cache creation input token cost above 1 hour."""

    cache_read_input_token_cost_above_200k_tokens: Decimal | None = Field(None, ge=0)
    """Cache read input token cost above 200k tokens."""

    input_cost_per_token_cache_hit: Decimal | None = Field(None, ge=0)
    """Input cost per token for cache hits."""

    cache_creation_input_audio_token_cost: Decimal | None = Field(None, ge=0)
    """Cache creation input audio token cost."""

    cache_read_input_audio_token_cost: Decimal | None = Field(None, ge=0)
    """Cache read input audio token cost."""

    # Databricks DBU-based pricing
    input_dbu_cost_per_token: Decimal | None = Field(None, ge=0)
    """Input DBU cost per token."""

    output_dbu_cost_per_token: Decimal | None = Field(None, ge=0)
    """Output DBU cost per token."""

    output_db_cost_per_token: Decimal | None = Field(None, ge=0)
    """Output DB cost per token."""

    # Legacy character-based pricing
    input_cost_per_character: Decimal | None = Field(None, ge=0)
    """Input cost per character."""

    output_cost_per_character: Decimal | None = Field(None, ge=0)
    """Output cost per character."""

    input_cost_per_character_above_128k_tokens: Decimal | None = Field(None, ge=0)
    """Input cost per character above 128k tokens."""

    output_cost_per_character_above_128k_tokens: Decimal | None = Field(None, ge=0)
    """Output cost per character above 128k tokens."""

    # Multimodal pricing
    input_cost_per_image: Decimal | None = Field(None, ge=0)
    """Input cost per image."""

    input_cost_per_image_above_128k_tokens: Decimal | None = Field(None, ge=0)
    """Input cost per image above 128k tokens."""

    input_cost_per_video_per_second: Decimal | None = Field(None, ge=0)
    """Input cost per video per second."""

    input_cost_per_video_per_second_above_128k_tokens: Decimal | None = Field(None, ge=0)
    """Input cost per video per second above 128k tokens."""

    input_cost_per_audio_per_second: Decimal | None = Field(None, ge=0)
    """Input cost per audio per second."""

    input_cost_per_audio_per_second_above_128k_tokens: Decimal | None = Field(None, ge=0)
    """Input cost per audio per second above 128k tokens."""

    input_cost_per_audio_token: Decimal | None = Field(None, ge=0)
    """Input cost per audio token."""

    output_cost_per_audio_token: Decimal | None = Field(None, ge=0)
    """Output cost per audio token."""

    output_cost_per_image: Decimal | None = Field(None, ge=0)
    """Output cost per image."""

    # Time-based pricing (for some chat models)
    input_cost_per_second: Decimal | None = Field(None, ge=0)
    """Input cost per second."""

    output_cost_per_second: Decimal | None = Field(None, ge=0)
    """Output cost per second."""

    # Multimodal limits
    max_images_per_prompt: int | None = Field(None, gt=0)
    """Maximum images per prompt."""

    max_videos_per_prompt: int | None = Field(None, gt=0)
    """Maximum videos per prompt."""

    max_video_length: float | None = Field(None, gt=0)
    """Max video length in hours."""

    max_audio_per_prompt: int | None = Field(None, gt=0)
    """Maximum audio files per prompt."""

    max_audio_length_hours: float | None = Field(None, gt=0)
    """Maximum audio length in hours."""

    max_pdf_size_mb: float | None = Field(None, gt=0)
    """Maximum PDF size in MB."""

    # Core capabilities

    supports_assistant_prefill: bool | None = None
    """Whether model supports assistant prefill."""

    # Advanced capabilities

    supports_audio_input: bool | None = None
    """Whether model supports audio input."""

    supports_audio_output: bool | None = None
    """Whether model supports audio output."""

    supports_video_input: bool | None = None
    """Whether model supports video input."""

    supports_url_context: bool | None = None
    """Whether model supports URL context."""

    supports_computer_use: bool | None = None
    """Whether model supports computer use."""

    supports_service_tier: bool | None = None
    """Whether model supports service tier selection."""

    # API support

    supported_regions: list[SupportedRegion | str] | None = None
    """List of supported regions."""

    rpd: int | None = None
    """Rate per day."""


class EmbeddingModel(BaseModelConfig):
    """Embedding models with vector output capabilities."""

    mode: Literal[ModelMode.EMBEDDING]
    """Model operation mode."""

    # Token limits
    max_input_tokens: int | None = Field(None, ge=0)
    """Maximum input tokens."""

    # Pricing (output always free for embeddings)
    input_cost_per_token: Decimal | None = Field(None, ge=0)
    """Cost per input token."""

    output_cost_per_token: Decimal = Decimal("0.0")
    """Always 0 for embeddings."""

    # Multimodal embedding support
    input_cost_per_image: Decimal | None = Field(None, ge=0)
    """Input cost per image for multimodal embeddings."""

    input_cost_per_video_per_second: Decimal | None = Field(None, ge=0)
    """Input cost per video per second for multimodal embeddings."""

    input_cost_per_audio_per_second: Decimal | None = Field(None, ge=0)
    """Input cost per audio per second for multimodal embeddings."""

    supports_embedding_image_input: bool | None = None
    """Whether model supports embedding image input."""

    supports_image_input: bool | None = None
    """Deprecated field for image input support."""

    # Vector configuration
    max_output_tokens: int | None = Field(None, ge=0)
    """Maximum output tokens."""

    output_vector_size: int | None = Field(None, gt=0)
    """Embedding dimension."""

    # API support
    supported_endpoints: list[SupportedEndpoint] | None = None
    """List of supported API endpoints."""

    supported_modalities: list[SupportedModality] | None = None
    """List of supported input modalities."""

    # Databricks DBU-based pricing
    input_dbu_cost_per_token: Decimal | None = Field(None, ge=0)
    """Input DBU cost per token."""

    output_dbu_cost_per_token: Decimal | None = Field(None, ge=0)
    """Output DBU cost per token."""

    # Character-based pricing
    input_cost_per_character: Decimal | None = Field(None, ge=0)
    """Input cost per character."""

    # Video pricing tiers
    input_cost_per_video_per_second_above_15s_interval: Decimal | None = Field(None, ge=0)
    """Input cost per video per second above 15s interval."""

    input_cost_per_video_per_second_above_8s_interval: Decimal | None = Field(None, ge=0)
    """Input cost per video per second above 8s interval."""

    # Batch processing pricing
    input_cost_per_token_batches: Decimal | None = Field(None, ge=0)
    """Input cost per token for batch processing."""

    output_cost_per_token_batches: Decimal | None = Field(None, ge=0)
    """Output cost per token for batch processing."""

    input_cost_per_token_batch_requests: Decimal | None = Field(None, ge=0)
    """Input cost per token for batch requests."""


class AudioTranscriptionModel(BaseModelConfig):
    """Audio transcription models with time-based pricing."""

    mode: Literal[ModelMode.AUDIO_TRANSCRIPTION]
    """Model operation mode."""

    # Time-based pricing
    input_cost_per_second: Decimal | None = Field(None, ge=0)
    """Cost per second of audio."""

    output_cost_per_second: Decimal | None = Field(None, ge=0)
    """Output cost per second."""

    # Token-based pricing (for some audio transcription models)
    input_cost_per_audio_token: Decimal | None = Field(None, ge=0)
    """Input cost per audio token."""

    input_cost_per_token: Decimal | None = Field(None, ge=0)
    """Input cost per token."""

    output_cost_per_token: Decimal | None = Field(None, ge=0)
    """Output cost per token."""

    # Token limits
    max_input_tokens: int | None = Field(None, gt=0)
    """Maximum input tokens."""

    max_output_tokens: int | None = Field(None, gt=0)
    """Maximum output tokens."""

    # API support
    supported_endpoints: list[SupportedEndpoint] | None = None
    """List of supported API endpoints."""


class AudioSpeechModel(BaseModelConfig):
    """Audio speech synthesis models."""

    mode: Literal[ModelMode.AUDIO_SPEECH]
    """Model operation mode."""

    # Character-based pricing
    input_cost_per_character: Decimal | None = Field(None, ge=0)
    """Input cost per character."""

    output_cost_per_second: Decimal | None = Field(None, ge=0)
    """Output cost per second of generated audio."""

    # Token-based pricing (for some speech models)
    input_cost_per_token: Decimal | None = Field(None, ge=0)
    """Input cost per token."""

    output_cost_per_token: Decimal | None = Field(None, ge=0)
    """Output cost per token."""

    output_cost_per_audio_token: Decimal | None = Field(None, ge=0)
    """Output cost per audio token."""

    # API support
    supported_endpoints: list[SupportedEndpoint] | None = None
    """List of supported API endpoints."""

    supported_modalities: list[SupportedModality] | None = None
    """List of supported input modalities."""

    supported_output_modalities: list[SupportedOutputModality] | None = None
    """List of supported output modalities."""

    # Token limits
    max_input_tokens: int | None = Field(None, gt=0)
    """Maximum input tokens."""

    max_output_tokens: int | None = Field(None, gt=0)
    """Maximum output tokens."""


class ImageGenerationModel(BaseModelConfig):
    """Image generation models with per-image pricing."""

    mode: Literal[ModelMode.IMAGE_GENERATION]
    """Model operation mode."""

    # Token limits (for text prompts)
    max_input_tokens: int | None = Field(None, gt=0)
    """Maximum input tokens."""

    max_output_tokens: int | None = Field(None, gt=0)
    """Maximum output tokens."""

    # Pricing structure
    input_cost_per_token: Decimal | None = Field(None, ge=0)
    """Input cost per token."""

    output_cost_per_token: Decimal | None = Field(None, ge=0)
    """Output cost per token."""

    output_cost_per_image: Decimal | None = Field(None, ge=0)
    """Output cost per generated image."""

    output_cost_per_reasoning_token: Decimal | None = Field(None, ge=0)
    """Output cost per reasoning token."""

    # Cache and multimodal pricing
    cache_read_input_token_cost: Decimal | None = Field(None, ge=0)
    """Cache read input token cost."""

    input_cost_per_audio_token: Decimal | None = Field(None, ge=0)
    """Input cost per audio token."""

    # Pixel-based pricing for image generation
    input_cost_per_pixel: Decimal | None = Field(None, ge=0)
    """Input cost per pixel."""

    output_cost_per_pixel: Decimal | None = Field(None, ge=0)
    """Output cost per pixel."""

    # Alternative image-based pricing
    input_cost_per_image: Decimal | None = Field(None, ge=0)
    """Input cost per image."""

    # Multimodal limits (inherited from chat capabilities)
    max_images_per_prompt: int | None = Field(None, gt=0)
    """Maximum images per prompt."""

    max_videos_per_prompt: int | None = Field(None, gt=0)
    """Maximum videos per prompt."""

    max_video_length: float | None = Field(None, gt=0)
    """Maximum video length."""

    max_audio_per_prompt: int | None = Field(None, gt=0)
    """Maximum audio files per prompt."""

    max_audio_length_hours: float | None = Field(None, gt=0)
    """Maximum audio length in hours."""

    max_pdf_size_mb: float | None = Field(None, gt=0)
    """Maximum PDF size in MB."""

    # Capabilities (subset of chat model capabilities)
    supports_function_calling: bool | None = None
    """Whether model supports function calling."""

    supports_parallel_function_calling: bool | None = None
    """Whether model supports parallel function calling."""

    supports_tool_choice: bool | None = None
    """Whether model supports tool choice."""

    supports_response_schema: bool | None = None
    """Whether model supports response schema."""

    supports_system_messages: bool | None = None
    """Whether model supports system messages."""

    supports_vision: bool | None = None
    """Whether model supports vision/image input."""

    supports_pdf_input: bool | None = None
    """Whether model supports PDF input."""

    supports_prompt_caching: bool | None = None
    """Whether model supports prompt caching."""

    supports_url_context: bool | None = None
    """Whether model supports URL context."""

    supports_web_search: bool | None = None
    """Whether model supports web search."""

    supports_audio_output: bool | None = None
    """Whether model supports audio output."""

    # API support
    supported_endpoints: list[SupportedEndpoint] | None = None
    """List of supported API endpoints."""

    supported_modalities: list[SupportedModality] | None = None
    """List of supported input modalities."""

    supported_output_modalities: list[SupportedOutputModality] | None = None
    """List of supported output modalities."""


class VideoGenerationModel(BaseModelConfig):
    """Video generation models with per-video pricing."""

    mode: Literal[ModelMode.VIDEO_GENERATION]
    """Model operation mode."""

    # Token limits (for text prompts)
    max_input_tokens: int | None = Field(None, gt=0)
    """Maximum input tokens."""

    max_output_tokens: int | None = Field(None, gt=0)
    """Maximum output tokens."""

    # Pricing structure
    input_cost_per_token: Decimal | None = Field(None, ge=0)
    """Input cost per token."""

    output_cost_per_token: Decimal | None = Field(None, ge=0)
    """Output cost per token."""

    output_cost_per_video: Decimal | None = Field(None, ge=0)
    """Output cost per generated video."""

    output_cost_per_second: Decimal | None = Field(None, ge=0)
    """Output cost per second of generated video."""

    # Video limits
    max_video_length: float | None = Field(None, gt=0)
    """Maximum video length in seconds."""

    # Capabilities
    supports_function_calling: bool | None = None
    """Whether model supports function calling."""

    supports_system_messages: bool | None = None
    """Whether model supports system messages."""

    supports_vision: bool | None = None
    """Whether model supports vision/image input."""

    # API support
    supported_endpoints: list[SupportedEndpoint] | None = None
    """List of supported API endpoints."""

    supported_modalities: list[SupportedModality] | None = None
    """List of supported input modalities."""

    supported_output_modalities: list[SupportedOutputModality] | None = None
    """List of supported output modalities."""


class RerankModel(BaseModelConfig):
    """Document reranking models with query-based pricing."""

    mode: Literal[ModelMode.RERANK]
    """Model operation mode."""

    # Token limits
    max_input_tokens: int | None = Field(None, gt=0)
    """Maximum input tokens."""

    max_output_tokens: int | None = Field(None, gt=0)
    """Maximum output tokens."""

    max_query_tokens: int | None = Field(None, gt=0)
    """Maximum query tokens."""

    # Rerank-specific limits
    max_document_chunks_per_query: int | None = Field(None, gt=0)
    """Maximum document chunks per query."""

    max_tokens_per_document_chunk: int | None = Field(None, gt=0)
    """Maximum tokens per document chunk."""

    # Query-based pricing
    input_cost_per_query: Decimal | None = Field(None, ge=0)
    """Input cost per query."""

    input_cost_per_token: Decimal = Field(Decimal("0.0"), ge=0)
    """Input cost per token (usually 0)."""

    output_cost_per_token: Decimal = Field(Decimal("0.0"), ge=0)
    """Output cost per token (usually 0)."""


class ResponsesModel(TextGenerationModel):
    """Reasoning/responses models (like OpenAI o1 series)."""

    mode: Literal[ModelMode.RESPONSES]
    """Model operation mode."""

    # Override token limits with stricter validation for responses models
    max_input_tokens: int | None = Field(None, gt=0)
    """Maximum input tokens."""

    max_output_tokens: int | None = Field(None, gt=0)
    """Maximum output tokens."""


class ModerationModel(BaseModelConfig):
    """Content moderation models."""

    mode: Literal[ModelMode.MODERATION]
    """Model operation mode."""

    # Token limits
    max_input_tokens: int | None = Field(None, gt=0)
    """Maximum input tokens."""

    max_output_tokens: int | None = Field(None, ge=0)
    """Maximum output tokens (can be 0)."""

    # Pricing
    input_cost_per_token: Decimal | None = Field(None, ge=0)
    """Cost per input token."""

    output_cost_per_token: Decimal | None = Field(None, ge=0)
    """Cost per output token."""

    # API support
    supported_endpoints: list[SupportedEndpoint] | None = None
    """List of supported API endpoints."""

    supported_modalities: list[SupportedModality] | None = None
    """List of supported input modalities."""

    supported_output_modalities: list[SupportedOutputModality] | None = None
    """List of supported output modalities."""


# Union type for all model configs
ModelConfig = (
    ChatCompletionModel
    | EmbeddingModel
    | AudioTranscriptionModel
    | AudioSpeechModel
    | ImageGenerationModel
    | VideoGenerationModel
    | RerankModel
    | ResponsesModel
    | ModerationModel
)
