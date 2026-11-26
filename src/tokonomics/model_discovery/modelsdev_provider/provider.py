"""Models.dev provider - unified model discovery from models.dev API."""

from __future__ import annotations

import asyncio
import contextlib
from datetime import datetime
import logging
import os
import threading
import time
from typing import TYPE_CHECKING, Any, Literal

from tokonomics.model_discovery.base import ModelProvider
from tokonomics.model_discovery.cache import MODELS_CACHE_TTL
from tokonomics.model_discovery.model_info import ModelInfo, ModelPricing


if TYPE_CHECKING:
    from tokonomics.model_discovery.model_info import Modality


logger = logging.getLogger(__name__)


# Mapping from models.dev provider names to pydantic-ai provider names
_PROVIDER_NAME_MAP: dict[str, str] = {
    "amazon-bedrock": "bedrock",
    "fireworks-ai": "fireworks",
    "google": "google-gla",
    "togetherai": "together",
    "github-models": "github",
    "xai": "grok",
}


ModelsDevProviderType = Literal[
    "alibaba",
    "alibaba-cn",
    "amazon-bedrock",
    "anthropic",
    "azure",
    "baseten",
    "cerebras",
    "chutes",
    "cloudflare-workers-ai",
    "cohere",
    "cortecs",
    "deepinfra",
    "deepseek",
    "fastrouter",
    "fireworks-ai",
    "github-copilot",
    "github-models",
    "google",
    "google-vertex",
    "google-vertex-anthropic",
    "groq",
    "huggingface",
    "inception",
    "inference",
    "llama",
    "lmstudio",
    "lucidquery",
    "mistral",
    "modelscope",
    "moonshotai",
    "moonshotai-cn",
    "morph",
    "nvidia",
    "openai",
    "opencode",
    "openrouter",
    "perplexity",
    "requesty",
    "submodel",
    "synthetic",
    "togetherai",
    "upstage",
    "v0",
    "venice",
    "vercel",
    "wandb",
    "xai",
    "zai",
    "zai-coding-plan",
    "zhipuai",
    "zhipuai-coding-plan",
]


_API_URL = "https://models.dev/api.json"
# The parsed api.json, shared by all ModelsDevProvider partials (anthropic, zai, ...):
# one download per MODELS_CACHE_TTL, concurrent callers on one loop share it.
_api_data: tuple[float, dict[str, Any]] | None = None
_api_inflight: dict[asyncio.AbstractEventLoop, asyncio.Task[dict[str, Any]]] = {}
_api_lock = threading.Lock()


def clear_api_cache() -> None:
    """Forget the downloaded models.dev catalog."""
    global _api_data
    with _api_lock:
        _api_data = None


async def _download_api() -> dict[str, Any]:
    from anyenv import get_json

    logger.debug("Fetching models from models.dev API")
    data = await get_json(_API_URL, cache=True, cache_ttl=MODELS_CACHE_TTL, return_type=dict)
    if not isinstance(data, dict):
        msg = "Invalid response format from models.dev"
        raise RuntimeError(msg)  # noqa: TRY004
    return data


async def _get_api_data() -> dict[str, Any]:
    global _api_data
    entry = _api_data
    if entry is not None and time.monotonic() - entry[0] < MODELS_CACHE_TTL:
        return entry[1]
    loop = asyncio.get_running_loop()
    with _api_lock:
        task = _api_inflight.get(loop)
        if task is None:
            task = _api_inflight[loop] = loop.create_task(_download_api())
            task.add_done_callback(lambda _: _api_inflight.pop(loop, None))
    data = await asyncio.shield(task)
    with _api_lock:
        _api_data = (time.monotonic(), data)
    return data


_API_KEY_VARS: dict[str, tuple[str, ...]] = {
    "anthropic": ("ANTHROPIC_API_KEY",),
    "openai": ("OPENAI_API_KEY",),
    "cohere": ("COHERE_API_KEY", "CO_API_KEY"),
    "cerebras": ("CEREBRAS_API_KEY",),
    "deepseek": ("DEEPSEEK_API_KEY",),
    "groq": ("GROQ_API_KEY",),
    "chutes": ("CHUTES_API_KEY",),
    "cloudflare-workers-ai": ("CF_API_TOKEN",),
    "cortecs": ("CORTECS_API_KEY",),
    "xai": ("XAI_API_KEY",),
    "azure": ("AZURE_OPENAI_API_KEY",),
    "fireworks-ai": ("FIREWORKS_API_KEY",),
    "mistral": ("MISTRAL_API_KEY",),
    "zai": ("ZAI_API_KEY", "ZHIPU_API_KEY"),
    "moonshotai": ("MOONSHOT_API_KEY",),
}


class ModelsDevProvider(ModelProvider):
    """Models.dev API provider - aggregates models from all providers."""

    def __init__(self, provider: ModelsDevProviderType | None = None) -> None:
        super().__init__()
        self.base_url = "https://models.dev"
        self.headers = {}
        self.params = {}
        self.provider_filter = provider

    def is_available(self) -> bool:
        """Whether the models can be used: the whole catalog always, a provider with its key.

        A provider filter without a known key variable counts as unavailable.
        """
        if self.provider_filter is None:
            return True
        env_vars = _API_KEY_VARS.get(self.provider_filter, ())
        return any(os.environ.get(var) for var in env_vars)

    def _parse_model(self, data: dict[str, Any]) -> ModelInfo:
        """Parse models.dev API response into ModelInfo."""
        # Extract provider from context (set during parsing)
        # Map to pydantic-ai provider names
        raw_provider = data.get("_provider_id", "unknown")
        provider_id = _PROVIDER_NAME_MAP.get(raw_provider, raw_provider)

        # Extract pricing information
        pricing = None
        if "cost" in data:
            cost = data["cost"]
            pricing = ModelPricing(
                prompt=cost.get("input", 0) / 1_000_000 if "input" in cost else None,
                completion=cost.get("output", 0) / 1_000_000 if "output" in cost else None,
                input_cache_read=cost.get("cache_read", 0) / 1_000_000
                if "cache_read" in cost
                else None,
                input_cache_write=cost.get("cache_write", 0) / 1_000_000
                if "cache_write" in cost
                else None,
            )

        # Extract modalities (map 'pdf' -> 'file' for consistency)
        input_modalities: set[Modality] = {"text"}
        output_modalities: set[Modality] = {"text"}
        if "modalities" in data:
            modalities = data["modalities"]
            raw_input = modalities.get("input", ["text"])
            raw_output = modalities.get("output", ["text"])
            # Map 'pdf' to 'file' for consistency with other providers
            input_modalities = {"file" if m == "pdf" else m for m in raw_input}
            output_modalities = {"file" if m == "pdf" else m for m in raw_output}

        # Parse release_date (format: "YYYY-MM-DD")
        created_at = None
        if release_date := data.get("release_date"):
            with contextlib.suppress(ValueError, TypeError):
                created_at = datetime.strptime(release_date, "%Y-%m-%d")

        model_id = str(data["id"])

        # Determine if it's an embedding model (heuristic)
        is_embedding = "embedding" in model_id.lower() or "embed" in model_id.lower()

        return ModelInfo(
            id=model_id,
            name=str(data.get("name", model_id)),
            provider=provider_id,
            description=None,  # models.dev doesn't provide descriptions
            pricing=pricing,
            context_window=data.get("limit", {}).get("context"),
            max_output_tokens=data.get("limit", {}).get("output"),
            is_embedding=is_embedding,
            input_modalities=input_modalities,
            output_modalities=output_modalities,
            is_free=pricing is not None and pricing.prompt == 0 and pricing.completion == 0,
            created_at=created_at,
            metadata={
                "attachment": data.get("attachment", False),
                "reasoning": data.get("reasoning", False),
                "temperature": data.get("temperature", True),
                "tool_call": data.get("tool_call", False),
                "knowledge": data.get("knowledge"),
                "last_updated": data.get("last_updated"),
                "open_weights": data.get("open_weights", False),
            },
        )

    async def get_models(self) -> list[ModelInfo]:
        """Fetch all models from models.dev API."""
        from anyenv import HttpError

        try:
            data = await _get_api_data()
        except HttpError as e:
            msg = f"Failed to fetch models from models.dev: {e}"
            raise RuntimeError(msg) from e
        if self.provider_filter is not None:
            return self._parse_models(data)
        # The whole catalog (~10k models, ~150 ms): parse off the event loop.
        return await asyncio.to_thread(self._parse_models, data)

    def _parse_models(self, data: dict[str, Any]) -> list[ModelInfo]:
        models = []
        for provider_id, provider_data in data.items():
            if self.provider_filter is not None and provider_id != self.provider_filter:
                continue
            if not isinstance(provider_data, dict) or "models" not in provider_data:
                continue
            provider_models = provider_data["models"]
            if not isinstance(provider_models, dict):
                continue
            for model_id, model_info in provider_models.items():
                if not isinstance(model_info, dict):
                    continue
                # Add provider context to model data
                model_data = dict(model_info)
                model_data["id"] = model_id
                model_data["_provider_id"] = provider_id
                try:
                    models.append(self._parse_model(model_data))
                except Exception as e:  # noqa: BLE001
                    logger.warning(
                        "Failed to parse model %s from provider %s: %s", model_id, provider_id, e
                    )
        logger.debug(
            "Parsed %d models from models.dev (filter: %s)", len(models), self.provider_filter
        )
        return models


if __name__ == "__main__":
    import asyncio

    async def main() -> None:
        # Test all providers
        provider = ModelsDevProvider()
        models = await provider.get_models()
        print(f"All providers: {len(models)} models")

        # Test specific provider
        openai_provider = ModelsDevProvider(provider="openai")
        openai_models = await openai_provider.get_models()
        print(f"OpenAI only: {len(openai_models)} models")

        # Test another specific provider
        anthropic_provider = ModelsDevProvider(provider="anthropic")
        anthropic_models = await anthropic_provider.get_models()
        print(f"Anthropic only: {len(anthropic_models)} models")

        # Show sample models
        if openai_models:
            print("\nSample OpenAI model:")
            print(openai_models[0].format())

    asyncio.run(main())
