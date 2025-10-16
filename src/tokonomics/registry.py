"""Modern Pydantic models for LLM pricing configuration using discriminated unions."""

from __future__ import annotations

import json
import pathlib
from typing import TYPE_CHECKING, Any

from schemez import Schema

from tokonomics import log
from tokonomics.data_models import (
    AudioSpeechModel,
    AudioTranscriptionModel,
    ChatCompletionModel,
    EmbeddingModel,
    ImageGenerationModel,
    ModelConfig,
    ModerationModel,
    RerankModel,
    ResponsesModel,
    VideoGenerationModel,
)


logger = log.get_logger(__name__)

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from tokonomics.data_models import ModelMode


def iter_model_name_candidates(name: str) -> Iterator[str]:
    """Iter through model name candidates in LiteLLM pricing data.

    Iterates through name candidates for the LLM pricing data
    by trying different formats (direct match, base name, provider format).

    Args:
        name: Model name (e.g. "openai:gpt-4", "gpt-4")

    Yields:
        str: Name candidates for the LLM pricing data
    """
    name = name.lower()
    yield name
    if ":" in name:  # For provider:name format, try both variants
        provider, model_name = name.split(":", 1)
        yield model_name
        yield f"{provider.lower()}/{model_name}"


class ModelRegistry(Schema):
    """Registry containing all model configurations with query capabilities."""

    models: dict[str, ModelConfig]
    """Model name -> configuration mapping."""

    def get_model(self, name: str) -> ModelConfig | None:
        """Get model configuration by name."""
        return self.models.get(name)

    def get_models_by_provider(self, provider: str) -> dict[str, ModelConfig]:
        """Get all models for a specific provider."""
        return {
            name: config
            for name, config in self.models.items()
            if config.litellm_provider == provider
        }

    def get_models_by_mode(self, mode: ModelMode) -> dict[str, ModelConfig]:
        """Get all models for a specific mode."""
        return {
            name: config for name, config in self.models.items() if config.mode == mode
        }

    def get_chat_models(self) -> Mapping[str, ChatCompletionModel]:
        """Get all chat/completion models."""
        return {
            name: config
            for name, config in self.models.items()
            if isinstance(config, ChatCompletionModel)
        }

    def get_providers(self) -> set[str]:
        """Get all unique providers."""
        return {config.litellm_provider for config in self.models.values()}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelRegistry:
        """Create registry from dictionary data with automatic model discrimination."""
        models = {}
        failed_models = []
        dropped_models = []

        for name, config_data in data.items():
            try:
                # Skip models without mode field or sample_spec
                if "mode" not in config_data or name == "sample_spec":
                    dropped_models.append(name)
                    continue

                mode = config_data["mode"]
                match mode:
                    case "chat" | "completion":
                        model_config: ModelConfig = ChatCompletionModel.model_validate(
                            config_data
                        )
                    case "embedding":
                        model_config = EmbeddingModel.model_validate(config_data)
                    case "audio_transcription":
                        model_config = AudioTranscriptionModel.model_validate(config_data)
                    case "audio_speech":
                        model_config = AudioSpeechModel.model_validate(config_data)
                    case "image_generation":
                        model_config = ImageGenerationModel.model_validate(config_data)
                    case "video_generation":
                        model_config = VideoGenerationModel.model_validate(config_data)
                    case "rerank":
                        model_config = RerankModel.model_validate(config_data)
                    case "responses":
                        model_config = ResponsesModel.model_validate(config_data)
                    case "moderation":
                        model_config = ModerationModel.model_validate(config_data)
                    case _:
                        # Default to chat for unknown modes
                        model_config = ChatCompletionModel.model_validate(config_data)

                models[name] = model_config

            except Exception as e:  # noqa: BLE001
                failed_models.append((name, str(e)))
                # Only print first few failures to avoid spam
                if len(failed_models) <= 5:  # noqa: PLR2004
                    logger.debug("Failed to parse model %s: %s", name, e)

        if dropped_models:
            logger.debug("Dropped %s models without mode field", len(dropped_models))
        if failed_models:
            msg = "Failed to parse %s models out of %s"
            logger.debug(msg, len(failed_models), len(data))
        return cls(models=models)

    @classmethod
    def from_json_file(cls, file_path: str) -> ModelRegistry:
        """Load registry from JSON file."""
        with pathlib.Path(file_path).open() as f:
            data = json.load(f)
        return cls.from_dict(data)


def load_model_config(config_data: dict[str, Any]) -> ModelConfig:  # noqa: PLR0911
    """Load a single model configuration with manual discrimination."""
    mode = config_data.get("mode", "chat")

    match mode:
        case "chat" | "completion":
            return ChatCompletionModel.model_validate(config_data)
        case "embedding":
            return EmbeddingModel.model_validate(config_data)
        case "audio_transcription":
            return AudioTranscriptionModel.model_validate(config_data)
        case "audio_speech":
            return AudioSpeechModel.model_validate(config_data)
        case "image_generation":
            return ImageGenerationModel.model_validate(config_data)
        case "video_generation":
            return VideoGenerationModel.model_validate(config_data)
        case "rerank":
            return RerankModel.model_validate(config_data)
        case "responses":
            return ResponsesModel.model_validate(config_data)
        case "moderation":
            return ModerationModel.model_validate(config_data)
        case _:
            return ChatCompletionModel.model_validate(config_data)
