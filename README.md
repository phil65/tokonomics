# Tokonomics

[![PyPI License](https://img.shields.io/pypi/l/tokonomics.svg)](https://pypi.org/project/tokonomics/)
[![Package status](https://img.shields.io/pypi/status/tokonomics.svg)](https://pypi.org/project/tokonomics/)
[![Daily downloads](https://img.shields.io/pypi/dd/tokonomics.svg)](https://pypi.org/project/tokonomics/)
[![Weekly downloads](https://img.shields.io/pypi/dw/tokonomics.svg)](https://pypi.org/project/tokonomics/)
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
[![Github commits this week](https://img.shields.io/github/commit-activity/w/phil65/tokonomics)](https://github.com/phil65/tokonomics)
[![Github commits this month](https://img.shields.io/github/commit-activity/m/phil65/tokonomics)](https://github.com/phil65/tokonomics)
[![Github commits this year](https://img.shields.io/github/commit-activity/y/phil65/tokonomics)](https://github.com/phil65/tokonomics)
[![Package status](https://codecov.io/gh/phil65/tokonomics/branch/main/graph/badge.svg)](https://codecov.io/gh/phil65/tokonomics/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PyUp](https://pyup.io/repos/github/phil65/tokonomics/shield.svg)](https://pyup.io/repos/github/phil65/tokonomics/)

[Read the documentation!](https://phil65.github.io/tokonomics/)



Calculate costs for LLM usage based on token counts using LiteLLM's pricing data.

## Installation

```bash
pip install tokonomics
```

## Features

- Automatic cost calculation for various LLM models
- Caches pricing data locally (24-hour default cache duration)
- Supports multiple model name formats (e.g., "gpt-4", "openai:gpt-4")
- Asynchronous API
- Fully typed with runtime type checking
- Zero configuration required

## Usage

```python
import asyncio
from tokonomics import calculate_token_cost, TokenUsage

async def main():
    # Define your token usage
    usage = TokenUsage(
        prompt=100,      # tokens used in the prompt
        completion=50,   # tokens used in the completion
        total=150        # total tokens used
    )

    # Calculate cost
    cost = await calculate_token_cost("gpt-4", usage)
    if cost:
        print(f"Total cost: ${cost:.6f}")
    else:
        print("Could not determine cost for model")

asyncio.run(main())
```

You can customize the cache timeout:

```python
from tokonomics import get_model_costs

# Get model costs with custom cache duration (e.g., 1 hour)
costs = await get_model_costs("gpt-4", cache_timeout=3600)
```

## Model Name Support

The library supports multiple formats for model names:
- Direct model names: `"gpt-4"`
- Provider-prefixed: `"openai:gpt-4"`
- Provider-path style: `"openai/gpt-4"`

Names are matched case-insensitively.

## Data Source

Pricing data is sourced from [LiteLLM's pricing repository](https://github.com/BerriAI/litellm) and is automatically cached locally using `diskcache`. The cache is updated when pricing data is not found or has expired.

## Requirements

- Python 3.12+
- `httpx`
- `diskcache`
- `platformdirs`

## License

This project is licensed under the MIT License - see the LICENSE file for details.
