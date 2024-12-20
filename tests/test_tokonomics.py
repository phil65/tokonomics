"""Test suite for tokonomics core functionality."""

from __future__ import annotations

from pathlib import Path
import shutil

import diskcache
import httpx
from platformdirs import user_data_dir
import pytest
import respx

from tokonomics import calculate_token_cost, get_model_costs
from tokonomics.toko_types import TokenUsage


SAMPLE_PRICING_DATA = {
    "gpt-4": {
        "input_cost_per_token": 0.03,
        "output_cost_per_token": 0.06,
    },
    "gpt-3.5-turbo": {
        "input_cost_per_token": 0.001,
        "output_cost_per_token": 0.002,
    },
}

SAMPLE_TOKEN_USAGE = TokenUsage(
    prompt=10,
    completion=20,
    total=30,
)

# Use a test-specific cache directory
TEST_CACHE_DIR = Path(user_data_dir("tokonomics", "tokonomics")) / "pricing"


@pytest.fixture(autouse=True)
def setup_teardown():
    """Setup and teardown for each test."""
    # Setup: ensure clean cache directory
    import tokonomics.core

    if hasattr(tokonomics.core, "_cost_cache"):
        tokonomics.core._cost_cache.close()

    if TEST_CACHE_DIR.exists():
        shutil.rmtree(TEST_CACHE_DIR)
    TEST_CACHE_DIR.mkdir(parents=True)

    # Point the cache to our test directory
    tokonomics.core._cost_cache = diskcache.Cache(str(TEST_CACHE_DIR))

    yield

    # Teardown: clean up
    tokonomics.core._cost_cache.close()
    if TEST_CACHE_DIR.exists():
        shutil.rmtree(TEST_CACHE_DIR)


@pytest.fixture
def mock_litellm_api():
    """Mock LiteLLM API responses."""
    with respx.mock(assert_all_mocked=True, assert_all_called=True) as respx_mock:
        route = respx_mock.get(
            "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"
        )
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
    cost = await calculate_token_cost("gpt-4", SAMPLE_TOKEN_USAGE, cache_timeout=1)
    assert cost is not None
    # 10 tokens * 0.03 + 20 tokens * 0.06 = 1.5
    assert cost == 1.5  # noqa: PLR2004


@pytest.mark.asyncio
async def test_calculate_token_cost_unknown_model(mock_litellm_api):
    """Test token cost calculation with unknown model."""
    cost = await calculate_token_cost(
        "unknown-model", SAMPLE_TOKEN_USAGE, cache_timeout=1
    )
    assert cost is None


@pytest.mark.asyncio
async def test_api_error(mock_litellm_api):
    """Test behavior when API request fails."""
    mock_litellm_api.get(
        "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"
    ).mock(return_value=httpx.Response(500))
    costs = await get_model_costs("gpt-4", cache_timeout=1)
    assert costs is None
