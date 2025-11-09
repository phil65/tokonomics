"""Test suite for model discovery providers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tokonomics.model_discovery import get_all_models


if TYPE_CHECKING:
    from tokonomics.model_discovery import ProviderType


@pytest.mark.parametrize(
    "provider",
    [
        "anthropic",
        "groq",
        "mistral",
        "openai",
        "openrouter",
        "github",
        "copilot",
        "cerebras",
        "gemini",
        "cohere",
        "deepseek",
        "requesty",
        "xai",
        "novita",
        "vercel-gateway",
        "ollama",
    ],
)
async def test_provider_model_fetch(provider: ProviderType):
    """Test that each provider can fetch models without errors.

    This test ensures we get notified if provider APIs change unexpectedly.
    It doesn't validate the content, just that the fetch succeeds.
    """
    from tokonomics.model_discovery import _PROVIDER_MAP

    # Check if provider is available before attempting to fetch
    try:
        provider_class = _PROVIDER_MAP[provider]
        provider_instance = provider_class()

        if not provider_instance.is_available():
            pytest.skip(
                f"Provider {provider} not available (no API key or other requirement)"
            )

    except KeyError:
        pytest.skip(f"Provider {provider} not found in provider map")
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"Provider {provider} unavailable: {e}")

    try:
        models = await get_all_models(providers=[provider])

        # Basic validation - should return a list
        assert isinstance(models, list)

        # If models are returned, they should have basic required fields
        if models:
            sample_model = models[0]
            assert sample_model.id is not None
            assert sample_model.name is not None
            assert sample_model.provider is not None

    except Exception as e:
        # If provider is not available (missing API key), that's expected
        if "API key" in str(e) or "not found" in str(e).lower():
            pytest.skip(f"Provider {provider} not available: {e}")
        else:
            # Unexpected error - this should fail the test
            raise


async def test_modelsdev_provider_specific():
    """Test ModelsDevProvider specifically since it's our new unified provider."""
    from tokonomics.model_discovery.modelsdev_provider import ModelsDevProvider

    provider = ModelsDevProvider()
    models = await provider.get_models()

    # Should return models from multiple providers
    assert len(models) > 0

    # Should have models from different providers
    providers_found = {model.provider for model in models}
    assert len(providers_found) > 1

    # Should include OpenAI and Anthropic
    assert "openai" in providers_found
    assert "anthropic" in providers_found

    # Validate rich data structure
    openai_models = [m for m in models if m.provider == "openai"]
    if openai_models:
        sample = openai_models[0]
        # Should have pricing information
        assert sample.pricing is not None
        # Should have context window
        assert sample.context_window is not None
        # Should have metadata
        assert sample.metadata is not None


async def test_openai_anthropic_partials():
    """Test that OpenAI and Anthropic partials work correctly."""
    # Test that they return only their respective models
    openai_models = await get_all_models(providers=["openai"])
    anthropic_models = await get_all_models(providers=["anthropic"])

    # All OpenAI models should have openai provider
    assert all(m.provider == "openai" for m in openai_models)

    # All Anthropic models should have anthropic provider
    assert all(m.provider == "anthropic" for m in anthropic_models)

    # Should have some models from each
    assert len(openai_models) > 0
    assert len(anthropic_models) > 0


if __name__ == "__main__":
    pytest.main(["-v", __file__])
