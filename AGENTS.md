# AGENTS.md

## Overview

`tokonomics` calculates LLM usage costs from token counts using LiteLLM's public
pricing JSON, and discovers available models (with pricing, context window and
modalities) from many provider APIs and the models.dev aggregator. It is an async-first
library with no CLI. It is a dependency of `llmling-models` and `agentpool`, so treat
the names exported from `src/tokonomics/__init__.py` as public API and keep them stable.

## Architecture

- `src/tokonomics/core.py`: LiteLLM pricing lookups. `get_model_costs`,
  `calculate_token_cost`, `get_available_models`, `get_model_capabilities`,
  `reset_cache`. Downloads `LITELLM_PRICES_URL` via `anyenv.get_json` (HTTP cache),
  plus an in-process dict cache (`_cost_cache`). Model-name matching (case-insensitive,
  `provider:model`, `provider/model`) lives in `_find_litellm_model_name`.
- `src/tokonomics/toko_types.py`: lightweight result types (`ModelCosts` TypedDict,
  `TokenCosts` and `ModelCapabilities` dataclasses).
- `src/tokonomics/pydanticai_cost.py`: `calculate_pydantic_cost` plus a `Usage`
  Protocol (structural, no pydantic-ai import).
- `src/tokonomics/token_count.py`: `count_tokens`, falling back from tiktoken (extra) to
  transformers (not a dependency) to a rough heuristic.
- `src/tokonomics/model_discovery/`: model discovery.
  - `base.py`: `ModelProvider` ABC (`is_available`, `_parse_model`, `get_models`;
    default `get_models` GETs `{base_url}/models` and parses `data[]`).
  - `model_info.py`: `ModelInfo` / `ModelPricing` dataclasses (with `pydantic_ai_id`,
    `litellm_id`, `pydantic_ai_variants`, `format()`).
  - `*_provider/provider.py`: one provider per directory (OpenRouter, Groq, Mistral,
    GitHub, Copilot, Gemini, DeepSeek, Requesty, xAI, Comet, Novita, Vercel Gateway,
    Ollama, models.dev).
  - `__init__.py`: `_PROVIDER_MAP`, `ProviderType` literal, `get_all_models` /
    `get_all_models_sync`. They keep each provider's parsed list in memory
    (`cache.py`, `MODELS_CACHE_TTL`, per provider and headers) and filter it per call;
    discovery requests are HTTP-cached for the same 24 h. Anthropic/OpenAI/Cohere/Cerebras/... are
    `partial(ModelsDevProvider, provider=...)`, not their own classes.
  - `variants.py`: reasoning/thinking level settings for pydantic-ai.
  - `copilot_provider/token_manager.py`: `CopilotTokenManager` (GitHub OAuth to Copilot token).
- `src/tokonomics/data_models.py` + `registry.py`: pydantic/schemez models for the full
  LiteLLM JSON (discriminated union per `mode`), `ModelRegistry` for querying it.
  `src/tokonomics/models/chat_model.py`: cleaned-up `ChatModel` view.
- `src/tokonomics/model_names/`: **generated** `Literal` types of model ids per provider
  (`ModelName`, `ModelId`). Regenerate with `scripts/generate_model_names.py`; excluded from ruff.
- `src/tokonomics/resources/`: static reference data (Artificial Analysis, LMArena URL).

## Where to look

| Task | Location |
| --- | --- |
| Cost lookup / name matching | `src/tokonomics/core.py` |
| Add or fix a provider | `src/tokonomics/model_discovery/<name>_provider/`, register in `model_discovery/__init__.py` |
| Unified model fields | `src/tokonomics/model_discovery/model_info.py` |
| LiteLLM schema models | `src/tokonomics/data_models.py`, `src/tokonomics/registry.py` |
| Generated model name literals | `scripts/generate_model_names.py` -> `src/tokonomics/model_names/` |
| Public exports | `src/tokonomics/__init__.py` |
| Tests | `tests/` (`test_tokonomics.py` mocks HTTP with `httpx2.MockTransport`; others hit the network) |

## Commands

```bash
uv sync --all-extras                      # setup (default groups: dev, lint, docs)
uv run pytest                             # tests
uvx ruff check . && uvx ruff format --check .
uv run ty check                           # type check (checks src/ only)
uv run deptry src                         # dependency hygiene
uv run mknodes build                      # docs (serve: uv run mknodes serve)
uv build
uv run python scripts/generate_model_names.py   # regenerate model_names/ (needs network/API keys)
```

`duties.py` wraps these (`uvx duty lint`, `uvx duty test`, `uvx duty build`); `duty`
itself is not in any dependency group. CI (`.github/workflows/build.yml`): ruff check/format, `ty check`, pytest with coverage on
Python 3.13/3.14 x Linux/macOS/Windows (3.14 is `continue-on-error`).

## Conventions

- Python >= 3.13, `from __future__ import annotations`, Google-style docstrings, line length 100.
- Network calls go through `anyenv` (`get_json`, `post`, ...), not httpx2 directly.
- Heavy or optional imports are done inside functions; keep `import tokonomics` cheap and
  free of optional deps.
- Prices: LiteLLM data is per token; `ModelPricing` in model discovery is also per token.
- pytest runs with `asyncio_mode = "auto"`; async tests need no decorator.

## Specs

Data sources and model-list endpoints. All track latest (no versioned schema upstream).

| Spec | URL | Targets | Code | Update check |
|---|---|---|---|---|
| LiteLLM model prices and context windows (JSON) | https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json | unpinned (tracks `main`; `LITELLM_PRICES_URL` in `src/tokonomics/core.py`) | `src/tokonomics/core.py`, `src/tokonomics/data_models.py`, `src/tokonomics/registry.py` | same URL (file history) |
| models.dev catalog (`/api.json`) | https://github.com/anomalyco/models.dev | unpinned (tracks latest) | `src/tokonomics/model_discovery/modelsdev_provider/` | same URL |
| OpenAI-compatible `GET /models` (OpenRouter, Groq, Mistral, DeepSeek, Requesty, Comet, Novita, Vercel AI Gateway, GitHub, Copilot) | https://platform.openai.com/docs/api-reference/models/list | unpinned (each provider's `/v1` base URL in its `provider.py`) | `src/tokonomics/model_discovery/base.py`, `src/tokonomics/model_discovery/*_provider/` | provider docs, e.g. https://openrouter.ai/docs/api-reference/models/get-models |
| Gemini API `models.list` | https://ai.google.dev/api/models | API v1beta (`https://generativelanguage.googleapis.com/v1beta`) | `src/tokonomics/model_discovery/gemini_provider/` | same URL |
| xAI language models | https://docs.x.ai/docs/api-reference | API v1 (`https://api.x.ai/v1/language-models`) | `src/tokonomics/model_discovery/xai_provider/` | same URL |
| Ollama `/api/tags`, `/api/show` | https://docs.ollama.com/api/tags | unpinned (tracks latest) | `src/tokonomics/model_discovery/ollama_provider/` | https://github.com/ollama/ollama/releases |

## Pitfalls

- Copier-managed: `.copier-answers.yml`, workflows, `.pre-commit-config.yaml`, `duties.py`
  and most of `pyproject.toml` come from the `copier-phil65` template. Don't hand-edit
  template-owned parts; add dependencies below the `# Only add below (Copier)` markers.
- Many tests fetch live data (LiteLLM JSON, models.dev, provider APIs). Provider tests
  `pytest.skip` when API keys are missing; don't delete them, skip them.
- `core._is_testing()` disables the HTTP cache under pytest (`PYTEST_CURRENT_TEST`).
- The anyenv HTTP cache lives in the user cache dir (`anyenv_cache_hishel2.db`) and is
  shared with other projects. If you see `JsonLoadError: expected value at line 1 column 1`
  on cached fetches, a stale/compressed cache entry is the likely cause, not this code.
- Provider API keys come from env vars (`OPENROUTER_API_KEY`, `GROQ_API_KEY`,
  `MISTRAL_API_KEY`, `GEMINI_API_KEY`, `GITHUB_TOKEN`, `GITHUB_COPILOT_API_KEY`, ...);
  see each provider's `__init__`.
- LiteLLM's JSON schema drifts (e.g. it dropped `max_tokens`); parse defensively.
- `uv.lock` is gitignored.
