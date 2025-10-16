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
[![Github Discussions](https://img.shields.io/github/discussions/phil65/tokonomics)](https://github.com/phil65/tokonomics/discussions)
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


Calculate costs for LLM usage based on token counts using LiteLLM's pricing data.

## Installation

```bash
pip install tokonomics
```

## Features

- Automatic cost calculation for various LLM models
- Detailed cost breakdown (prompt, completion, and total costs)
- Caches pricing data locally (24-hour default cache duration)
- Supports multiple model name formats (e.g., "gpt-4", "openai:gpt-4")
- Asynchronous API
- Fully typed with runtime type checking
- Zero configuration required

## Usage

```python
import asyncio
from tokonomics import calculate_token_cost

async def main():
    # Calculate cost with token counts
    costs = await calculate_token_cost(
        model="gpt-4",
        input_tokens=100,    # tokens used in the prompt
        output_tokens=50,  # tokens used in the completion
    )

    if costs:
        print(f"Prompt cost: ${costs.input_cost:.6f}")
        print(f"Completion cost: ${costs.output_cost:.6f}")
        print(f"Total cost: ${costs.total_cost:.6f}")
    else:
        print("Could not determine cost for model")

asyncio.run(main())
```

You can customize the cache timeout:

```python
from tokonomics import get_model_costs, clear_cache

# Get model costs with custom cache duration (e.g., 1 hour)
costs = await get_model_costs("gpt-4", cache_timeout=3600)
if costs:
    print(f"Input cost per token: ${costs['input_cost_per_token']}")
    print(f"Output cost per token: ${costs['output_cost_per_token']}")

clear_cache()
```


### Pydantic-AI Integration

If you're using pydantic-ai, you can directly calculate costs from its Usage objects:

```python
from tokonomics import calculate_pydantic_cost

# Assuming you have a pydantic-ai Usage object
costs = await calculate_pydantic_cost(
    model="gpt-4",
    usage=usage_object,
)

if costs:
    print(f"Prompt cost: ${costs.input_cost:.6f}")
    print(f"Completion cost: ${costs.output_cost:.6f}")
    print(f"Total cost: ${costs.total_cost:.6f}")
```

## Model Name Support

The library supports multiple formats for model names:
- Direct model names: `"gpt-4"`
- Provider-prefixed: `"openai:gpt-4"`
- Provider-path style: `"openai/gpt-4"`

Names are matched case-insensitively.

## Data Source

Pricing data is sourced from [LiteLLM's pricing repository](https://github.com/BerriAI/litellm) and is automatically cached locally using `hishel`. The cache is updated when pricing data is not found or has expired.

## Requirements

- Python 3.12+
- `httpx`
- `platformdirs`
- `upath`
- `pydantic` (≥ 2.0)

## License

This project is licensed under the MIT License - see the LICENSE file for details.
