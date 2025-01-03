"""Tests for core functionality."""

from __future__ import annotations

from httpx import AsyncClient
import pytest

from tokonomics.core import TokenLimits, get_model_limits


@pytest.mark.asyncio
async def test_get_model_limits_handles_non_numeric_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test handling of non-numeric values in LiteLLM data."""
    # Mock response with mix of valid and invalid data
    mock_data = {
        "sample_spec": "some value",  # Should be skipped
        "valid-model": {
            "max_tokens": "32000",
            "max_input_tokens": 24000,
            "max_output_tokens": "8000",
        },
        "broken-model": {
            "max_tokens": (
                "set to max_output_tokens if provider specifies it. "
                "IF not set to max_tokens provider specifies"
            ),
            "max_input_tokens": "not a number",
            "max_output_tokens": "description instead of value",
        },
        "float-model": {
            "max_tokens": "32000.0",
            "max_input_tokens": 24000.5,
            "max_output_tokens": "8000.9",
        },
    }

    async def mock_get(*args: object, **kwargs: object) -> MockResponse:
        return MockResponse(mock_data)  # type: ignore

    # Patch httpx.AsyncClient.get
    monkeypatch.setattr(AsyncClient, "get", mock_get)

    # Test valid model
    valid_limits = await get_model_limits("valid-model")
    assert valid_limits == TokenLimits(
        total_tokens=32000,
        input_tokens=24000,
        output_tokens=8000,
    )

    # Test model with non-numeric values
    broken_limits = await get_model_limits("broken-model")
    assert broken_limits is None

    # Test model with float values
    float_limits = await get_model_limits("float-model")
    assert float_limits == TokenLimits(
        total_tokens=32000,
        input_tokens=24000,
        output_tokens=8000,
    )


class MockResponse:
    """Mock httpx response."""

    def __init__(self, json_data: dict[str, object]) -> None:
        """Initialize with JSON data."""
        self.json_data = json_data

    def json(self) -> dict[str, object]:
        """Return JSON data."""
        return self.json_data

    def raise_for_status(self) -> None:
        """Do nothing for successful response."""
