"""Tests for core functionality."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock, patch

import pytest

from tokonomics.core import get_model_costs
from tokonomics.toko_types import ModelCosts


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
            input_cost_per_token=Decimal("0.001"),
            output_cost_per_token=Decimal("0.002"),
        )

        # Test model with non-numeric values
        broken_costs = await get_model_costs("broken-model")
        assert broken_costs is None

        # Test model with longer float values
        float_costs = await get_model_costs("float-model")
        assert float_costs == ModelCosts(
            input_cost_per_token=Decimal("0.001500"),
            output_cost_per_token=Decimal("0.002000"),
        )

        # Test model with missing cost fields
        missing_costs = await get_model_costs("missing-fields")
        assert missing_costs is None


if __name__ == "__main__":
    pytest.main([__file__, "-vv"])
