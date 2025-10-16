"""Helpers for calculating token costs using LiteLLM pricing data."""

from __future__ import annotations

from typing import Any


def _is_numeric(value: str | int | float | None) -> bool:  # noqa: PYI041
    """Check if a value can be converted to a number."""
    if value is None:
        return False
    if isinstance(value, int | float):
        return True
    if not isinstance(value, str):
        return False
    try:
        float(value)
    except ValueError:
        return False
    else:
        return True


def _safe_numeric_convert(value: Any) -> float:
    """Safely convert a value to float, returning 0 if not possible."""
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0
