from __future__ import annotations

import pytest

from tokonomics.core import get_model_capabilities


@pytest.mark.asyncio
async def test_get_model_capabilities():
    """Test model capabilities fetching."""
    # Test with a well-known model that should have stable capabilities
    capabilities = await get_model_capabilities("gpt-4")
    assert capabilities is not None

    # Test core attributes that should always be present
    assert capabilities.max_tokens > 0
    assert capabilities.max_input_tokens > 0
    assert capabilities.max_output_tokens > 0
    assert capabilities.litellm_provider == "openai"
    assert capabilities.mode == "chat"

    # Test boolean flags that should be stable for GPT-4
    assert capabilities.supports_function_calling is True
    assert capabilities.supports_system_messages is True

    # Verify types of all fields
    assert isinstance(capabilities.max_tokens, int)
    assert isinstance(capabilities.max_input_tokens, int)
    assert isinstance(capabilities.max_output_tokens, int)
    assert isinstance(capabilities.litellm_provider, str | type(None))
    assert isinstance(capabilities.mode, str | type(None))
    assert isinstance(capabilities.supports_function_calling, bool)
    assert isinstance(capabilities.supports_parallel_function_calling, bool)
    assert isinstance(capabilities.supports_vision, bool)
    assert isinstance(capabilities.supports_audio_input, bool)
    assert isinstance(capabilities.supports_audio_output, bool)
    assert isinstance(capabilities.supports_prompt_caching, bool)
    assert isinstance(capabilities.supports_response_schema, bool)
    assert isinstance(capabilities.supports_system_messages, bool)
