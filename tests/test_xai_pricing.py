"""xAI's language-models prices are USD cents per 100M tokens."""

from __future__ import annotations

import pytest

from tokonomics.model_discovery.xai_provider.provider import XAIProvider


# From GET https://api.x.ai/v1/language-models (2026-11-26); LiteLLM and models.dev list
# grok-4.3 at $1.25/M input, $2.50/M output, $0.20/M cache reads.
GROK_4_3 = {
    "id": "grok-4.3",
    "input_modalities": ["text", "image"],
    "output_modalities": ["text"],
    "prompt_text_token_price": 12500,
    "cached_prompt_text_token_price": 2000,
    "prompt_image_token_price": 12500,
    "completion_text_token_price": 25000,
    "search_price": 0,
}


def test_prices_per_token():
    model = XAIProvider(api_key="test")._parse_model(GROK_4_3)
    assert model.pricing is not None
    assert model.pricing.prompt == pytest.approx(1.25e-06)
    assert model.pricing.completion == pytest.approx(2.5e-06)
    assert model.pricing.input_cache_read == pytest.approx(2e-07)
    assert model.pricing.image == pytest.approx(1.25e-06)
