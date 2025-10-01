"""GitHub models provider."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

from tokonomics.model_discovery.base import ModelProvider
from tokonomics.model_discovery.model_info import ModelInfo


if TYPE_CHECKING:
    from tokonomics.model_discovery.model_info import Modality


logger = logging.getLogger(__name__)


def get_token_from_gh_cli() -> str | None:
    """Get GitHub token from gh CLI."""
    import subprocess

    try:
        result = subprocess.run(
            ["gh", "auth", "token"], capture_output=True, text=True, check=True
        )
        token = result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError) as e:
        logger.debug("Failed to get GitHub token from gh CLI: %s", e)
        return None
    else:
        return token if token else None


class GitHubProvider(ModelProvider):
    """GitHub AI models API provider."""

    def __init__(self, token: str | None = None):
        super().__init__()
        self.token = token or os.environ.get("GITHUB_TOKEN")
        if not self.token:
            self.token = get_token_from_gh_cli()

        if not self.token:
            msg = "GitHub token not found in parameters, GITHUB_TOKEN env var, or gh CLI"
            raise RuntimeError(msg)

        self.base_url = "https://api.catalog.azureml.ms"
        self.models_url = f"{self.base_url}/asset-gallery/v1.0/models"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.token}",
        }

    def is_available(self) -> bool:
        """Check whether the provider is available for use."""
        return bool(self.token)

    async def _fetch_models(self, is_free: bool) -> list[ModelInfo]:
        """Fetch models with specific freePlayground value."""
        from anyenv import post

        params = {
            "filters": [
                {
                    "field": "freePlayground",
                    "values": ["true" if is_free else "false"],
                    "operator": "eq",
                },
                {"field": "labels", "values": ["latest"], "operator": "eq"},
            ],
            "order": [{"field": "displayName", "direction": "asc"}],
        }

        try:
            response = await post(
                self.models_url,
                json=params,
                headers=self.headers,
                cache=True,
            )
            data = await response.json()
            if not isinstance(data, dict) or "summaries" not in data:
                msg = "Invalid response format from GitHub Models API"
                raise RuntimeError(msg)  # noqa: TRY301

            # Inject is_free into each model's data
            summaries = []
            for item in data["summaries"]:
                item_with_free = item.copy()  # Make a copy to not modify the original
                item_with_free["is_free"] = is_free
                summaries.append(item_with_free)

            return [self._parse_model(item) for item in summaries]
        except Exception as e:
            msg = f"Failed to fetch models from GitHub: {e}"
            logger.exception(msg)
            raise RuntimeError(msg) from e

    def _parse_model(self, data: dict[str, Any]) -> ModelInfo:
        """Parse GitHub models API response into ModelInfo."""
        inference_task = ""
        if data.get("task"):
            inference_task = data["task"]
        elif data.get("inferenceTasks") and isinstance(data["inferenceTasks"], list):
            inference_task = data["inferenceTasks"][0] if data["inferenceTasks"] else ""

        # Combine summary and task information for description
        description = data.get("summary", "")
        if inference_task:
            description = (
                f"{description}\nTask: {inference_task}"
                if description
                else f"Task: {inference_task}"
            )

        # Extract context window and output tokens
        context_window = None
        max_output_tokens = None
        limits = data.get("modelLimits")
        if isinstance(limits, dict) and isinstance(limits.get("textLimits"), dict):
            text_limits = limits["textLimits"]
            context_window = text_limits.get("inputContextWindow")
            max_output_tokens = text_limits.get("maxOutputTokens")

        input_modalities: set[Modality] = {"text"}
        output_modalities: set[Modality] = {"text"}
        if isinstance(limits, dict):
            if limits.get("supportedInputModalities"):
                input_modalities = set(limits["supportedInputModalities"])
            if limits.get("supportedOutputModalities"):
                output_modalities = set(limits["supportedOutputModalities"])

        # Use name as ID as it's more consistent with other providers
        model_id = data.get("name", "")
        model_name = (
            data.get("friendly_name", "") or data.get("displayName", "") or model_id
        )

        return ModelInfo(
            id=model_id,
            name=model_name,
            provider="github",
            description=description,
            owned_by=data.get("publisher", "GitHub"),
            context_window=context_window,
            max_output_tokens=max_output_tokens,
            input_modalities=input_modalities,
            output_modalities=output_modalities,
            is_free=data.get("is_free", True),
        )

    async def get_models(self) -> list[ModelInfo]:
        """Get both free and paid models through separate API calls."""
        free_models = await self._fetch_models(is_free=True)
        paid_models = await self._fetch_models(is_free=False)
        return free_models + paid_models


if __name__ == "__main__":
    import asyncio

    async def main():
        provider = GitHubProvider()
        models = await provider.get_models()
        for model in models:
            print(model.format())

    asyncio.run(main())
