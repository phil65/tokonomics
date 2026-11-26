"""Tests for the LiteLLM price table: cached-input pricing, caching, single flight."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

import pytest

from tokonomics import core
from tokonomics.pydanticai_cost import calculate_pydantic_cost


PRICES = {
    "sample_spec": "not a model",
    "gpt-4o": {
        "input_cost_per_token": 2.5e-06,
        "output_cost_per_token": 1e-05,
        "cache_read_input_token_cost": 1.25e-06,
    },
    "claude-x": {
        "input_cost_per_token": 3e-06,
        "output_cost_per_token": 1.5e-05,
        "cache_read_input_token_cost": 3e-07,
        "cache_creation_input_token_cost": 3.75e-06,
    },
    "gemini-x": {
        "input_cost_per_token": 1e-06,
        "output_cost_per_token": 2e-06,
        "input_cost_per_token_above_200k_tokens": 2e-06,
        "output_cost_per_token_above_200k_tokens": 4e-06,
        "input_cost_per_token_above_200k_tokens_priority": 9e-06,
    },
}


@pytest.fixture
def fetches(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    calls: list[int] = []

    async def fake_get_json(url: str, **kwargs: Any) -> dict[str, Any]:
        assert url == core.LITELLM_PRICES_URL
        calls.append(1)
        await asyncio.sleep(0.01)
        return PRICES

    core.reset_cache()
    monkeypatch.setattr(core, "get_json", fake_get_json)
    yield calls
    core.reset_cache()


async def test_cache_read_tokens_billed_at_cache_price(fetches):
    # 100k input of which 90k cache reads: 10k * 2.5e-6 + 90k * 1.25e-6 = 0.1375
    costs = await core.calculate_token_cost("gpt-4o", 100_000, 0, cache_read_tokens=90_000)
    assert costs is not None
    assert costs.input_cost == Decimal("0.1375")


async def test_cache_write_tokens_billed_at_creation_price(fetches):
    # 10k uncached * 3e-6 + 20k reads * 3e-7 + 70k writes * 3.75e-6
    costs = await core.calculate_token_cost(
        "claude-x", 100_000, 1000, cache_read_tokens=20_000, cache_write_tokens=70_000
    )
    assert costs is not None
    assert costs.input_cost == Decimal("0.03") + Decimal("0.006") + Decimal("0.2625")
    assert costs.output_cost == Decimal("0.015")


async def test_cache_tokens_without_cache_price_use_input_price(fetches):
    costs = await core.calculate_token_cost("gemini-x", 1000, 0, cache_read_tokens=500)
    assert costs is not None
    assert costs.input_cost == Decimal("0.001")


async def test_tiered_price_above_threshold(fetches):
    low = await core.calculate_token_cost("gemini-x", 200_000, 1000)
    high = await core.calculate_token_cost("gemini-x", 200_001, 1000)
    assert low is not None
    assert high is not None
    assert low.input_cost == Decimal("0.2")
    assert low.output_cost == Decimal("0.002")
    assert high.input_cost == Decimal("200001") * Decimal("2e-06")
    assert high.output_cost == Decimal("0.004")


async def test_pydantic_usage_cache_tokens(fetches):
    @dataclass
    class FakeUsage:
        input_tokens: int
        output_tokens: int
        cache_read_tokens: int = 0
        cache_write_tokens: int = 0

        @property
        def total_tokens(self) -> int:
            return self.input_tokens + self.output_tokens

    costs = await calculate_pydantic_cost("gpt-4o", FakeUsage(100_000, 0, 90_000))
    assert costs is not None
    assert costs.input_cost == Decimal("0.1375")


async def test_unknown_models_do_not_refetch(fetches):
    for _ in range(5):
        assert await core.get_model_costs("no-such-model") is None
    assert await core.get_model_capabilities("no-such-model") is None
    assert await core.get_model_costs("openai:gpt-4o") is not None
    assert await core.get_available_models() == ["claude-x", "gemini-x", "gpt-4o"]
    assert len(fetches) == 1


async def test_concurrent_first_lookups_fetch_once(fetches):
    results = await asyncio.gather(*(core.get_model_costs("gpt-4o") for _ in range(10)))
    assert all(r is not None for r in results)
    assert len(fetches) == 1


async def test_table_expires_after_cache_timeout(fetches, monkeypatch: pytest.MonkeyPatch):
    now = [1000.0]
    monkeypatch.setattr(core, "_now", lambda: now[0])
    await core.get_model_costs("gpt-4o", cache_timeout=60)
    now[0] += 30
    await core.get_model_costs("gpt-4o", cache_timeout=60)
    assert len(fetches) == 1
    now[0] += 31
    await core.get_model_costs("gpt-4o", cache_timeout=60)
    assert len(fetches) == 2  # noqa: PLR2004


async def test_failed_refresh_keeps_stale_table(fetches, monkeypatch: pytest.MonkeyPatch):
    now = [1000.0]
    monkeypatch.setattr(core, "_now", lambda: now[0])
    assert await core.get_model_costs("gpt-4o", cache_timeout=60) is not None

    async def failing(url: str, **kwargs: Any) -> dict[str, Any]:
        raise OSError("offline")

    monkeypatch.setattr(core, "get_json", failing)
    now[0] += 120
    assert await core.get_model_costs("gpt-4o", cache_timeout=60) is not None
