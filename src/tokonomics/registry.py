"""Modern Pydantic models for LLM pricing configuration using discriminated unions."""

from __future__ import annotations

from decimal import Decimal
import json
import pathlib
from typing import TYPE_CHECKING, Any

from schemez import Schema

from tokonomics.models import (
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


if TYPE_CHECKING:
    from tokonomics.models import (
        ModelMode,
    )


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

    def get_chat_models(self) -> dict[str, ChatCompletionModel]:
        """Get all chat/completion models."""
        return {
            name: config
            for name, config in self.models.items()
            if isinstance(config, ChatCompletionModel)
        }

    def get_embedding_models(self) -> dict[str, EmbeddingModel]:
        """Get all embedding models."""
        return {
            name: config
            for name, config in self.models.items()
            if isinstance(config, EmbeddingModel)
        }

    def get_providers(self) -> set[str]:
        """Get all unique providers."""
        return {config.litellm_provider for config in self.models.values()}

    def get_cheapest_model_by_mode(
        self, mode: ModelMode
    ) -> tuple[str, ModelConfig] | None:
        """Get the cheapest model for a given mode based on input token cost."""
        models = self.get_models_by_mode(mode)
        if not models:
            return None

        # Find model with lowest input cost, handling None values
        def get_cost(item):
            _, model = item
            if (
                hasattr(model, "input_cost_per_token")
                and model.input_cost_per_token is not None
            ):
                return model.input_cost_per_token
            if (
                isinstance(model, AudioTranscriptionModel)
                and model.input_cost_per_second is not None
            ):
                return model.input_cost_per_second
            return Decimal("inf")

        cheapest_name, cheapest_model = min(models.items(), key=get_cost)
        return cheapest_name, cheapest_model

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

                # Manual discrimination based on mode
                mode = config_data["mode"]

                if mode in ("chat", "completion"):
                    model_config = ChatCompletionModel.model_validate(config_data)
                elif mode == "embedding":
                    model_config = EmbeddingModel.model_validate(config_data)
                elif mode == "audio_transcription":
                    model_config = AudioTranscriptionModel.model_validate(config_data)
                elif mode == "audio_speech":
                    model_config = AudioSpeechModel.model_validate(config_data)
                elif mode == "image_generation":
                    model_config = ImageGenerationModel.model_validate(config_data)
                elif mode == "video_generation":
                    model_config = VideoGenerationModel.model_validate(config_data)
                elif mode == "rerank":
                    model_config = RerankModel.model_validate(config_data)
                elif mode == "responses":
                    model_config = ResponsesModel.model_validate(config_data)
                elif mode == "moderation":
                    model_config = ModerationModel.model_validate(config_data)
                else:
                    # Default to chat for unknown modes
                    model_config = ChatCompletionModel.model_validate(config_data)

                models[name] = model_config

            except Exception as e:  # noqa: BLE001
                failed_models.append((name, str(e)))
                # Only print first few failures to avoid spam
                if len(failed_models) <= 5:  # noqa: PLR2004
                    print(f"Failed to parse model {name}: {e}")

        if dropped_models:
            print(f"Dropped {len(dropped_models)} models without mode field")
        if failed_models:
            print(f"Failed to parse {len(failed_models)} models out of {len(data)}")

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

    if mode in ("chat", "completion"):
        return ChatCompletionModel.model_validate(config_data)
    if mode == "embedding":
        return EmbeddingModel.model_validate(config_data)
    if mode == "audio_transcription":
        return AudioTranscriptionModel.model_validate(config_data)
    if mode == "audio_speech":
        return AudioSpeechModel.model_validate(config_data)
    if mode == "image_generation":
        return ImageGenerationModel.model_validate(config_data)
    if mode == "video_generation":
        return VideoGenerationModel.model_validate(config_data)
    if mode == "rerank":
        return RerankModel.model_validate(config_data)
    if mode == "responses":
        return ResponsesModel.model_validate(config_data)
    if mode == "moderation":
        return ModerationModel.model_validate(config_data)
    return ChatCompletionModel.model_validate(config_data)
