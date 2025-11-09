"""Test suite for tokonomics core functionality."""

from __future__ import annotations

from decimal import Decimal

import httpx
import pytest
import respx

from tokonomics import calculate_token_cost, core, get_model_costs
import tokonomics.core


SAMPLE_PRICING_DATA = {
    "gpt-4": {
        "input_cost_per_token": 0.00003,
        "output_cost_per_token": 0.00006,
        "max_tokens": 8192,
        "max_input_tokens": 6144,
        "max_output_tokens": 2048,
    },
    "gpt-3.5-turbo": {
        "input_cost_per_token": 0.00001,
        "output_cost_per_token": 0.00002,
        "max_tokens": 4096,
    },
}


@pytest.fixture(autouse=True)
def setup_teardown():
    """Setup and teardown for each test."""
    # Clear in-memory cache
    tokonomics.core._cost_cache.clear()

    yield

    # Reset after test
    tokonomics.core._cost_cache.clear()


@pytest.fixture
def mock_litellm_api():
    """Mock LiteLLM API responses."""
    respx_mock = respx.mock(assert_all_called=False)
    respx_mock.start()
    route = respx_mock.get(core.LITELLM_PRICES_URL)
    route.mock(return_value=httpx.Response(200, json=SAMPLE_PRICING_DATA))
    yield respx_mock
    respx_mock.stop()


async def test_get_model_costs_success(mock_litellm_api):
    """Test successful model cost retrieval."""
    costs = await get_model_costs("gpt-4", cache_timeout=1)
    assert costs is not None
    assert costs["input_cost_per_token"] == Decimal("0.00003")
    assert costs["output_cost_per_token"] == Decimal("0.00006")


async def test_get_model_costs_case_insensitive(mock_litellm_api):
    """Test that model name matching is case insensitive."""
    # First call to populate cache
    await get_model_costs("gpt-4", cache_timeout=1)
    # Second call with different case
    costs = await get_model_costs("GPT-4", cache_timeout=1)
    assert costs is not None
    assert costs["input_cost_per_token"] == Decimal("0.00003")


async def test_get_model_costs_provider_format(mock_litellm_api):
    """Test that provider:model format works."""
    # First call to populate cache
    await get_model_costs("gpt-4", cache_timeout=1)
    # Second call with provider format
    costs = await get_model_costs("openai:gpt-4", cache_timeout=1)
    assert costs is not None
    assert costs["input_cost_per_token"] == Decimal("0.00003")


async def test_get_model_costs_unknown_model(mock_litellm_api):
    """Test behavior with unknown model."""
    costs = await get_model_costs("unknown-model", cache_timeout=1)
    assert costs is None


async def test_calculate_token_cost_success(mock_litellm_api):
    """Test successful token cost calculation."""
    costs = await calculate_token_cost(
        model="gpt-4",
        input_tokens=10,
        output_tokens=20,
        cache_timeout=1,
    )
    assert costs is not None
    assert costs.input_cost == Decimal("0.0003")  # 10 tokens * 0.00003
    assert costs.output_cost == Decimal("0.0012")  # 20 tokens * 0.00006
    assert costs.total_cost == Decimal("0.0015")  # 0.0003 + 0.0012


async def test_calculate_token_cost_with_none(mock_litellm_api):
    """Test token cost calculation with None values."""
    costs = await calculate_token_cost(
        model="gpt-4",
        input_tokens=None,
        output_tokens=20,
        cache_timeout=1,
    )
    assert costs is not None
    assert costs.input_cost == Decimal("0")
    assert costs.output_cost == Decimal("0.0012")  # 20 tokens * 0.00006
    assert costs.total_cost == Decimal("0.0012")


async def test_calculate_token_cost_unknown_model(mock_litellm_api):
    """Test token cost calculation with unknown model."""
    costs = await calculate_token_cost(
        model="unknown-model",
        input_tokens=10,
        output_tokens=20,
        cache_timeout=1,
    )
    assert costs is None


async def test_api_error(mock_litellm_api):
    """Test behavior when API request fails."""
    # Clear the route and add a new one that returns error
    mock_litellm_api.routes.clear()
    mock_litellm_api.get(core.LITELLM_PRICES_URL).mock(return_value=httpx.Response(500))
    # Clear specific cache entries for this test
    cache_key = "gpt-4_costs"
    if cache_key in tokonomics.core._cost_cache:
        del tokonomics.core._cost_cache[cache_key]
    costs = await get_model_costs("gpt-4", cache_timeout=1)
    assert costs is None


if __name__ == "__main__":
    pytest.main(["-v", __file__])
