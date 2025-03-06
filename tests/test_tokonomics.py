"""Test suite for tokonomics core functionality."""

from __future__ import annotations

import httpx
import pytest
import respx

from tokonomics import calculate_token_cost, core, get_model_costs
from tokonomics.core import get_model_limits


SAMPLE_PRICING_DATA = {
    "gpt-4": {
        "input_cost_per_token": 0.03,
        "output_cost_per_token": 0.06,
        "max_tokens": 8192,
        "max_input_tokens": 6144,
        "max_output_tokens": 2048,
    },
    "gpt-3.5-turbo": {
        "input_cost_per_token": 0.001,
        "output_cost_per_token": 0.002,
        "max_tokens": 4096,
    },
}


@pytest.fixture(autouse=True)
def setup_teardown():
    """Setup and teardown for each test."""
    import tokonomics.core

    # Disable caching during tests
    tokonomics.core._TESTING = True

    # Clear in-memory cache
    tokonomics.core._cost_cache.clear()

    yield

    # Reset after test
    tokonomics.core._TESTING = False
    tokonomics.core._cost_cache.clear()


@pytest.fixture
def mock_litellm_api():
    """Mock LiteLLM API responses."""
    with respx.mock(assert_all_mocked=True) as respx_mock:
        route = respx_mock.get(core.LITELLM_PRICES_URL)
        route.mock(return_value=httpx.Response(200, json=SAMPLE_PRICING_DATA))
        yield respx_mock


@pytest.mark.asyncio
async def test_get_model_costs_success(mock_litellm_api):
    """Test successful model cost retrieval."""
    costs = await get_model_costs("gpt-4", cache_timeout=1)
    assert costs is not None
    assert costs["input_cost_per_token"] == 0.03  # noqa: PLR2004
    assert costs["output_cost_per_token"] == 0.06  # noqa: PLR2004


@pytest.mark.asyncio
async def test_get_model_costs_case_insensitive(mock_litellm_api):
    """Test that model name matching is case insensitive."""
    # First call to populate cache
    await get_model_costs("gpt-4", cache_timeout=1)
    # Second call with different case
    costs = await get_model_costs("GPT-4", cache_timeout=1)
    assert costs is not None
    assert costs["input_cost_per_token"] == 0.03  # noqa: PLR2004


@pytest.mark.asyncio
async def test_get_model_costs_provider_format(mock_litellm_api):
    """Test that provider:model format works."""
    # First call to populate cache
    await get_model_costs("gpt-4", cache_timeout=1)
    # Second call with provider format
    costs = await get_model_costs("openai:gpt-4", cache_timeout=1)
    assert costs is not None
    assert costs["input_cost_per_token"] == 0.03  # noqa: PLR2004


@pytest.mark.asyncio
async def test_get_model_costs_unknown_model(mock_litellm_api):
    """Test behavior with unknown model."""
    costs = await get_model_costs("unknown-model", cache_timeout=1)
    assert costs is None


@pytest.mark.asyncio
async def test_calculate_token_cost_success(mock_litellm_api):
    """Test successful token cost calculation."""
    costs = await calculate_token_cost(
        model="gpt-4",
        prompt_tokens=10,
        completion_tokens=20,
        cache_timeout=1,
    )
    assert costs is not None
    assert costs.prompt_cost == 0.3  # 10 tokens * 0.03  # noqa: PLR2004
    assert costs.completion_cost == 1.2  # 20 tokens * 0.06  # noqa: PLR2004
    assert costs.total_cost == 1.5  # 0.3 + 1.2  # noqa: PLR2004


@pytest.mark.asyncio
async def test_calculate_token_cost_with_none(mock_litellm_api):
    """Test token cost calculation with None values."""
    costs = await calculate_token_cost(
        model="gpt-4",
        prompt_tokens=None,
        completion_tokens=20,
        cache_timeout=1,
    )
    assert costs is not None
    assert costs.prompt_cost == 0.0
    assert costs.completion_cost == 1.2  # 20 tokens * 0.06  # noqa: PLR2004
    assert costs.total_cost == 1.2  # noqa: PLR2004


@pytest.mark.asyncio
async def test_calculate_token_cost_unknown_model(mock_litellm_api):
    """Test token cost calculation with unknown model."""
    costs = await calculate_token_cost(
        model="unknown-model",
        prompt_tokens=10,
        completion_tokens=20,
        cache_timeout=1,
    )
    assert costs is None


@pytest.mark.asyncio
async def test_api_error(mock_litellm_api):
    """Test behavior when API request fails."""
    mock_litellm_api.get(core.LITELLM_PRICES_URL).mock(return_value=httpx.Response(500))
    costs = await get_model_costs("gpt-4", cache_timeout=1)
    assert costs is None


@pytest.mark.asyncio
async def test_get_model_limits_success(mock_litellm_api):
    """Test successful model limit retrieval."""
    limits = await get_model_limits("gpt-4", cache_timeout=1)
    assert limits is not None
    assert limits.total_tokens == 8192  # noqa: PLR2004
    assert limits.input_tokens == 6144  # noqa: PLR2004
    assert limits.output_tokens == 2048  # noqa: PLR2004


@pytest.mark.asyncio
async def test_get_model_limits_fallback(mock_litellm_api):
    """Test limits fallback to max_tokens when specific limits aren't provided."""
    limits = await get_model_limits("gpt-3.5-turbo", cache_timeout=1)
    assert limits is not None
    assert limits.total_tokens == 4096  # noqa: PLR2004
    assert limits.input_tokens == 4096  # Fallback to max_tokens  # noqa: PLR2004
    assert limits.output_tokens == 4096  # Fallback to max_tokens  # noqa: PLR2004


if __name__ == "__main__":
    pytest.main(["-v", __file__])
