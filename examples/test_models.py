"""Test script to demonstrate loading and analyzing LLM pricing models."""

from collections import Counter
from pathlib import Path

from tokonomics.data_models import (
    AudioTranscriptionModel,
    ChatCompletionModel,
    EmbeddingModel,
    ResponsesModel,
)
from tokonomics.registry import ModelRegistry


def main():
    """Load and analyze the LLM pricing data."""
    # Load the data
    json_path = Path("model_prices_and_context_window.json")
    if not json_path.exists():
        print(f"Please download the data file: {json_path}")
        return

    print("Loading model registry...")
    registry = ModelRegistry.from_json_file(str(json_path))

    print(f"Successfully loaded {len(registry.models)} models")

    # Basic analysis
    print("\n" + "=" * 50)
    print("BASIC ANALYSIS")
    print("=" * 50)

    # Count by mode
    mode_counts = Counter(model.mode for model in registry.models.values())
    print("\nModels by type:")
    for mode, count in mode_counts.most_common():
        print(f"  {mode.value:20} {count:4d}")

    # Count by provider
    provider_counts = Counter(
        model.litellm_provider for model in registry.models.values()
    )
    print("\nTop 10 providers:")
    for provider, count in provider_counts.most_common(10):
        print(f"  {provider:25} {count:4d}")

    # Demonstrate discriminated union working
    print("\n" + "=" * 50)
    print("DISCRIMINATED UNION EXAMPLES")
    print("=" * 50)

    # Show different model types
    examples = {
        "Chat": "gpt-4o",
        "Embedding": "text-embedding-3-small",
        "Audio": "whisper-1",
        "Responses": "o1-preview",
    }

    for model_type, model_name in examples.items():
        if model_config := registry.get_model(model_name):
            print(f"\n{model_type} model: {model_name}")
            print(f"  Type: {type(model_config).__name__}")
            print(f"  Mode: {model_config.mode}")
            print(f"  Provider: {model_config.litellm_provider}")

            # Show type-specific fields
            if isinstance(
                model_config, (ChatCompletionModel, EmbeddingModel, ResponsesModel)
            ):
                if model_config.input_cost_per_token is not None:
                    print(f"  Input cost: ${model_config.input_cost_per_token:.8f}/token")
                if model_config.output_cost_per_token is not None:
                    print(
                        f"  Output cost: ${model_config.output_cost_per_token:.8f}/token"
                    )
            elif isinstance(model_config, AudioTranscriptionModel):
                if model_config.input_cost_per_second is not None:
                    print(
                        f"  Input cost: ${model_config.input_cost_per_second:.6f}/second"
                    )

            if isinstance(model_config, ChatCompletionModel):
                print(f"  Function calling: {model_config.supports_function_calling}")

    # Capability analysis
    print("\n" + "=" * 50)
    print("CAPABILITY ANALYSIS")
    print("=" * 50)

    chat_models = registry.get_chat_models()

    # Count models with specific capabilities
    capabilities = [
        "supports_function_calling",
        "supports_vision",
        "supports_reasoning",
        "supports_prompt_caching",
    ]

    capability_counts = {
        "supports_function_calling": sum(
            1 for m in chat_models.values() if m.supports_function_calling
        ),
        "supports_vision": sum(1 for m in chat_models.values() if m.supports_vision),
        "supports_reasoning": sum(
            1 for m in chat_models.values() if m.supports_reasoning
        ),
        "supports_prompt_caching": sum(
            1 for m in chat_models.values() if m.supports_prompt_caching
        ),
    }

    for capability, count in capability_counts.items():
        print(f"{capability:25}: {count:4d}/{len(chat_models)} models")

    # Provider analysis
    print("\n" + "=" * 50)
    print("PROVIDER SPOTLIGHT")
    print("=" * 50)

    # OpenAI models
    openai_models = registry.get_models_by_provider("openai")
    print(f"\nOpenAI models ({len(openai_models)}):")
    for name, model in list(openai_models.items())[:5]:  # Show first 5
        if isinstance(model, (ChatCompletionModel, EmbeddingModel, ResponsesModel)):
            print(f"  {name:30} ${model.input_cost_per_token:.8f}/token")
        else:
            print(f"  {name:30} No token pricing")

    # Anthropic models
    anthropic_models = registry.get_models_by_provider("anthropic")
    print(f"\nAnthropic models ({len(anthropic_models)}):")
    for name, model in list(anthropic_models.items())[:5]:  # Show first 5
        if isinstance(model, (ChatCompletionModel, EmbeddingModel, ResponsesModel)):
            print(f"  {name:30} ${model.input_cost_per_token:.8f}/token")
        else:
            print(f"  {name:30} No token pricing")

    # Show model with complex pricing
    print("\n" + "=" * 50)
    print("COMPLEX PRICING EXAMPLE")
    print("=" * 50)

    # Look for a model with tiered pricing
    for name, model in registry.models.items():
        if model.tiered_pricing is not None and len(model.tiered_pricing) > 0:
            print(f"\n{name} has tiered pricing:")
            for i, tier in enumerate(model.tiered_pricing):
                print(f"  Tier {i + 1}: {tier.range[0]:,}-{tier.range[1]:,} tokens")
                print(f"    Input:  ${tier.input_cost_per_token:.8f}/token")
                print(f"    Output: ${tier.output_cost_per_token:.8f}/token")
            break

    # Show multimodal capabilities
    print("\n" + "=" * 50)
    print("MULTIMODAL CAPABILITIES")
    print("=" * 50)

    multimodal_models = []
    for name, model in chat_models.items():
        if (
            model.supports_vision
            or model.supports_audio_input
            or model.max_images_per_prompt is not None
        ):
            multimodal_models.append((name, model))

    print(f"Found {len(multimodal_models)} multimodal models:")
    for name, model in multimodal_models[:10]:  # Show first 10
        capabilities = []
        if model.supports_vision:
            capabilities.append("Vision")
        if model.supports_audio_input:
            capabilities.append("Audio")
        if model.max_images_per_prompt is not None:
            capabilities.append(f"Images({model.max_images_per_prompt})")

        print(f"  {name:30} {', '.join(capabilities)}")


if __name__ == "__main__":
    main()
