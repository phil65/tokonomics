# Tokonomics

[![PyPI License](https://img.shields.io/pypi/l/tokonomics.svg)](https://pypi.org/project/tokonomics/)
[![Package status](https://img.shields.io/pypi/status/tokonomics.svg)](https://pypi.org/project/tokonomics/)
[![Monthly downloads](https://img.shields.io/pypi/dm/tokonomics.svg)](https://pypi.org/project/tokonomics/)
[![Distribution format](https://img.shields.io/pypi/format/tokonomics.svg)](https://pypi.org/project/tokonomics/)
[![Wheel availability](https://img.shields.io/pypi/wheel/tokonomics.svg)](https://pypi.org/project/tokonomics/)
[![Python version](https://img.shields.io/pypi/pyversions/tokonomics.svg)](https://pypi.org/project/tokonomics/)
[![Implementation](https://img.shields.io/pypi/implementation/tokonomics.svg)](https://pypi.org/project/tokonomics/)
[![Releases](https://img.shields.io/github/downloads/phil65/tokonomics/total.svg)](https://github.com/phil65/tokonomics/releases)
[![Github Contributors](https://img.shields.io/github/contributors/phil65/tokonomics)](https://github.com/phil65/tokonomics/graphs/contributors)
[![Github Forks](https://img.shields.io/github/forks/phil65/tokonomics)](https://github.com/phil65/tokonomics/forks)
[![Github Issues](https://img.shields.io/github/issues/phil65/tokonomics)](https://github.com/phil65/tokonomics/issues)
[![Github Issues](https://img.shields.io/github/issues-pr/phil65/tokonomics)](https://github.com/phil65/tokonomics/pulls)
[![Github Watchers](https://img.shields.io/github/watchers/phil65/tokonomics)](https://github.com/phil65/tokonomics/watchers)
[![Github Stars](https://img.shields.io/github/stars/phil65/tokonomics)](https://github.com/phil65/tokonomics/stars)
[![Github Repository size](https://img.shields.io/github/repo-size/phil65/tokonomics)](https://github.com/phil65/tokonomics)
[![Github last commit](https://img.shields.io/github/last-commit/phil65/tokonomics)](https://github.com/phil65/tokonomics/commits)
[![Github release date](https://img.shields.io/github/release-date/phil65/tokonomics)](https://github.com/phil65/tokonomics/releases)
[![Github language count](https://img.shields.io/github/languages/count/phil65/tokonomics)](https://github.com/phil65/tokonomics)
[![Github commits this month](https://img.shields.io/github/commit-activity/m/phil65/tokonomics)](https://github.com/phil65/tokonomics)
[![Package status](https://codecov.io/gh/phil65/tokonomics/branch/main/graph/badge.svg)](https://codecov.io/gh/phil65/tokonomics/)
[![PyUp](https://pyup.io/repos/github/phil65/tokonomics/shield.svg)](https://pyup.io/repos/github/phil65/tokonomics/)

[Read the documentation!](https://phil65.github.io/tokonomics/)


Calculate LLM usage costs from token counts using [LiteLLM's pricing data](https://github.com/BerriAI/litellm),
and discover available models (pricing, context window, modalities) across many providers.

## Installation

```bash
pip install tokonomics
# optional: exact token counting via tiktoken
pip install "tokonomics[tiktoken]"
```

## Quickstart

```python
import asyncio

from tokonomics import calculate_token_cost


async def main():
    costs = await calculate_token_cost(
        model="gpt-4o",
        input_tokens=100,  # tokens used in the prompt
        output_tokens=50,  # tokens used in the completion
    )
    if costs:
        print(f"Input cost: ${costs.input_cost:.6f}")
        print(f"Output cost: ${costs.output_cost:.6f}")
        print(f"Total cost: ${costs.total_cost:.6f}")
    else:
        print("Could not determine cost for model")


asyncio.run(main())
```

## Features

- Cost calculation for thousands of models from LiteLLM's pricing JSON
  (`Decimal` input/output/total breakdown)
- Flexible model names: `"gpt-4o"`, `"openai:gpt-4o"`, `"openai/gpt-4o"` (case-insensitive)
- Model capabilities: context limits, function calling, vision, audio, prompt caching
- Model discovery across providers (OpenAI, Anthropic, OpenRouter, Groq, Mistral, Gemini,
  GitHub Models, Copilot, DeepSeek, xAI, Ollama, models.dev, ...)
- Token counting with tiktoken, transformers or a heuristic fallback
- pydantic-ai integration via `calculate_pydantic_cost`
- Generated `Literal` types of known model ids (`tokonomics.ModelName`)
- Async API, fully typed, pricing data cached locally

## Usage

### Model costs and capabilities

```python
from tokonomics import get_model_capabilities, get_model_costs, reset_cache

# Per-token prices; cache_timeout controls how long pricing data is cached (seconds)
costs = await get_model_costs("openai:gpt-4o", cache_timeout=3600)
if costs:
    print(f"Input cost per token: ${costs['input_cost_per_token']}")
    print(f"Output cost per token: ${costs['output_cost_per_token']}")

caps = await get_model_capabilities("gpt-4o")
if caps:
    print(caps.max_input_tokens, caps.supports_vision, caps.supports_function_calling)

reset_cache()  # clear the in-memory cache
```

### Pydantic-AI integration

```python
from tokonomics import calculate_pydantic_cost

# `usage` is a pydantic-ai usage object (anything with input_tokens / output_tokens)
costs = await calculate_pydantic_cost(model="gpt-4o", usage=result.usage())
if costs:
    print(f"Total cost: ${costs.total_cost:.6f}")
```

### Model discovery

```python
from tokonomics import get_all_models

# Without `providers`, all providers whose API keys are set are queried.
models = await get_all_models(providers=["openai", "anthropic", "openrouter"])
for model in models[:5]:
    print(model.pydantic_ai_id, model.context_window, model.pricing)
```

Providers read their API keys from the usual environment variables
(`OPENROUTER_API_KEY`, `GROQ_API_KEY`, `MISTRAL_API_KEY`, `GEMINI_API_KEY`, ...).
A synchronous variant is available as `tokonomics.model_discovery.get_all_models_sync`.

### Token counting

```python
from tokonomics import count_tokens

count_tokens("Hello world, how are you?", model="gpt-4o")
```

Uses tiktoken if installed, then transformers, then a rough approximation.

## Data sources

Pricing and capability data come from
[LiteLLM's model_prices_and_context_window.json](https://github.com/BerriAI/litellm);
model discovery uses the providers' model APIs and [models.dev](https://models.dev).
Responses are cached locally (24 hours by default).

## Requirements

- Python 3.13+

## Documentation

Full API reference: https://phil65.github.io/tokonomics/

## License

This project is licensed under the MIT License - see the LICENSE file for details.
