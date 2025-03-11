"""GitHub models provider."""

from __future__ import annotations

import logging
import os
from typing import Any

from tokonomics.model_discovery.base import ModelProvider
from tokonomics.model_discovery.model_info import Modality, ModelInfo


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
        self.params = {
            "filters": [
                {"field": "freePlayground", "values": ["true"], "operator": "eq"},
                {"field": "labels", "values": ["latest"], "operator": "eq"},
            ],
            "order": [{"field": "displayName", "direction": "asc"}],
        }

    def _parse_model(self, data: dict[str, Any]) -> ModelInfo:
        """Parse GitHub models API response into ModelInfo."""
        # Extract task and inferenceTasks
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
        if isinstance(data.get("modelLimits"), dict) and isinstance(
            data["modelLimits"].get("textLimits"), dict
        ):
            text_limits = data["modelLimits"]["textLimits"]
            context_window = text_limits.get("inputContextWindow")
            max_output_tokens = text_limits.get("maxOutputTokens")

        # Extract modalities
        input_modalities: list[Modality] = ["text"]  # Default
        output_modalities: list[Modality] = ["text"]  # Default
        if isinstance(data.get("modelLimits"), dict):
            if data["modelLimits"].get("supportedInputModalities"):
                input_modalities = data["modelLimits"]["supportedInputModalities"]
            if data["modelLimits"].get("supportedOutputModalities"):
                output_modalities = data["modelLimits"]["supportedOutputModalities"]

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
        )

    async def get_models(self) -> list[ModelInfo]:
        """Override the base method to handle GitHub's unique API structure."""
        from anyenv import post

        try:
            response = await post(
                self.models_url,
                json=self.params,
                headers=self.headers,
                cache=True,
            )
            data = await response.json()
            if not isinstance(data, dict) or "summaries" not in data:
                msg = "Invalid response format from GitHub Models API"
                raise RuntimeError(msg)  # noqa: TRY301

            # Process summaries directly - they contain more data than our mapping
            models = [self._parse_model(item) for item in data["summaries"]]
        except Exception as e:
            msg = f"Failed to fetch models from GitHub: {e}"
            logger.exception(msg)
            raise RuntimeError(msg) from e
        else:
            return models

    def get_models_sync(self) -> list[ModelInfo]:
        """Override the base method to handle GitHub's unique API structure."""
        import httpx

        try:
            response = httpx.post(self.models_url, json=self.params, headers=self.headers)

            if response.status_code != 200:  # noqa: PLR2004
                msg = f"Failed to fetch GitHub models: {response.status_code} - {response.text}"  # noqa: E501
                raise RuntimeError(msg)  # noqa: TRY301

            data = response.json()

            if not isinstance(data, dict) or "summaries" not in data:
                msg = "Invalid response format from GitHub Models API"
                raise RuntimeError(msg)  # noqa: TRY301

            # Process summaries directly - they contain more data than our mapping
            models = [self._parse_model(item) for item in data["summaries"]]

        except Exception as e:
            msg = f"Failed to fetch models from GitHub: {e}"
            logger.exception(msg)
            raise RuntimeError(msg) from e
        else:
            return models


if __name__ == "__main__":
    import asyncio

    async def main():
        provider = GitHubProvider()
        models = await provider.get_models()
        for model in models:
            print(model.format())

    asyncio.run(main())
