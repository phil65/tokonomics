"""Tests for core functionality."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from tokonomics.core import TokenLimits, get_model_costs, get_model_limits
from tokonomics.toko_types import ModelCosts


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

    mock_get_json = AsyncMock(return_value=mock_data)
    with patch("tokonomics.core.get_json", mock_get_json):
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


@pytest.mark.asyncio
async def test_get_model_costs_handles_non_numeric_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test handling of non-numeric values in cost data."""
    mock_data = {
        "sample_spec": "skip me",
        "valid-model": {
            "input_cost_per_token": "0.001",
            "output_cost_per_token": 0.002,
        },
        "broken-model": {
            "input_cost_per_token": "contact sales for pricing",
            "output_cost_per_token": "varies by usage",
        },
        "float-model": {
            "input_cost_per_token": "0.001500",
            "output_cost_per_token": "0.002000",
        },
        "missing-fields": {
            "some_other_field": "value",
        },
    }

    mock_get_json = AsyncMock(return_value=mock_data)
    with patch("tokonomics.core.get_json", mock_get_json):
        # Test valid model
        valid_costs = await get_model_costs("valid-model")
        assert valid_costs == ModelCosts(
            input_cost_per_token=0.001,
            output_cost_per_token=0.002,
        )

        # Test model with non-numeric values
        broken_costs = await get_model_costs("broken-model")
        assert broken_costs is None

        # Test model with longer float values
        float_costs = await get_model_costs("float-model")
        assert float_costs == ModelCosts(
            input_cost_per_token=0.0015,
            output_cost_per_token=0.002,
        )

        # Test model with missing cost fields
        missing_costs = await get_model_costs("missing-fields")
        assert missing_costs is None


if __name__ == "__main__":
    pytest.main([__file__, "-vv"])
