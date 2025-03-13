"""GitHub Copilot provider for model discovery."""

from __future__ import annotations

from datetime import datetime, timedelta
import logging
import os
import threading
from typing import Any

import anyenv
import httpx

from tokonomics.model_discovery.base import ModelProvider
from tokonomics.model_discovery.model_info import Modality, ModelInfo


logger = logging.getLogger(__name__)

# Models to exclude from the provider
COPILOT_EXCLUDED_MODELS = set[str]()  # "o1"

# Constants for token management
EDITOR_VERSION = "Neovim/0.6.1"
EDITOR_PLUGIN_VERSION = "copilot.vim/1.16.0"
USER_AGENT = "GithubCopilot/1.155.0"
TOKEN_EXPIRY_BUFFER_SECONDS = 120  # Refresh token 2 minutes before expiry


class CopilotTokenManager:
    """Manager for GitHub Copilot API tokens."""

    def __init__(self):
        # Get the GitHub OAuth token from environment
        self._github_oauth_token = os.environ.get("GITHUB_COPILOT_API_KEY")
        if not self._github_oauth_token:
            msg = "GitHub OAuth token not found in GITHUB_COPILOT_API_KEY env var"
            raise RuntimeError(msg)

        # This will store the short-lived Copilot token
        self._copilot_token = None
        self._token_expires_at = datetime.now()
        self._token_lock = threading.Lock()
        self._api_endpoint = "https://api.githubcopilot.com"

        # Initial token fetch will happen when first needed

    def get_token(self) -> str:
        """Get a valid Copilot token, refreshing if needed."""
        with self._token_lock:
            # If token is missing or expires in less than buffer time, refresh it
            now = datetime.now()
            if self._copilot_token is None or now > self._token_expires_at - timedelta(
                seconds=TOKEN_EXPIRY_BUFFER_SECONDS
            ):
                self._refresh_token()
            assert self._copilot_token, "Copilot token is missing"
            return self._copilot_token

    def _refresh_token(self) -> None:
        """Refresh the Copilot token using the GitHub OAuth token."""
        try:
            logger.debug("Fetching fresh GitHub Copilot token")
            response = httpx.get(
                "https://api.github.com/copilot_internal/v2/token",
                headers={
                    "authorization": f"token {self._github_oauth_token}",
                    "editor-version": EDITOR_VERSION,
                    "editor-plugin-version": EDITOR_PLUGIN_VERSION,
                    "user-agent": USER_AGENT,
                },
                timeout=30,
            )

            if response.status_code != 200:  # noqa: PLR2004
                logger.error(
                    "Failed to get Copilot token. Status: %d, Response: %s",
                    response.status_code,
                    response.text,
                )
                msg = f"Failed to get Copilot token: HTTP {response.status_code}"
                raise RuntimeError(msg)  # noqa: TRY301

            data = response.json()

            # Extract the Copilot token
            self._copilot_token = data.get("token")
            if not self._copilot_token:
                msg = "No token found in response from Copilot API"
                raise RuntimeError(msg)  # noqa: TRY301

            # Extract expiration time
            expires_at = data.get("expires_at")
            if expires_at is not None:
                self._token_expires_at = datetime.fromtimestamp(expires_at)
            else:
                # Default expiry: 25 minutes if not specified
                self._token_expires_at = datetime.now() + timedelta(minutes=25)

            # Update API endpoint if provided
            endpoints = data.get("endpoints", {})
            if "api" in endpoints:
                self._api_endpoint = endpoints["api"]

            logger.debug(
                "Copilot token refreshed, valid until: %s",
                self._token_expires_at.isoformat(),
            )
        except Exception as e:
            logger.exception("Failed to refresh GitHub Copilot token")
            if not self._copilot_token:
                msg = "Failed to obtain GitHub Copilot token"
                raise RuntimeError(msg) from e

    def generate_headers(self) -> dict[str, str]:
        """Generate headers for GitHub Copilot API requests."""
        return {
            "Authorization": f"Bearer {self.get_token()}",
            "editor-version": "Neovim/0.9.0",
            "Copilot-Integration-Id": "vscode-chat",
        }


class CopilotProvider(ModelProvider):
    """GitHub Copilot API provider."""

    def __init__(self):
        super().__init__()
        self._token_manager = CopilotTokenManager()
        self.base_url = self._token_manager._api_endpoint
        self.params = {}

    def _get_headers(self) -> dict[str, str]:
        """Get headers with a current valid token."""
        return self._token_manager.generate_headers()

    def _parse_model(self, data: dict[str, Any]) -> ModelInfo:
        """Parse Copilot API response into ModelInfo."""
        # Extract capabilities and limits
        capabilities = data.get("capabilities", {})
        limits = capabilities.get("limits", {})

        # Extract context window, input and output tokens
        context_window = limits.get("max_context_window_tokens")
        _max_input_tokens = limits.get("max_prompt_tokens")
        max_output_tokens = limits.get("max_output_tokens")

        # Determine modalities
        input_modalities: list[Modality] = ["text"]
        output_modalities: list[Modality] = ["text"]

        # Add vision capability if supported
        if capabilities.get("supports", {}).get("vision"):
            input_modalities.append("image")

        # Extract model family and version info
        model_family = capabilities.get("family", "")
        model_version = data.get("version", "")

        # Create description
        description = ""
        if model_family:
            description += f"Model family: {model_family}\n"
        if model_version:
            description += f"Version: {model_version}\n"
        if vendor := data.get("vendor"):
            description += f"Vendor: {vendor}\n"

        # Add capabilities to description
        supports = capabilities.get("supports", {})
        support_features = []
        if supports.get("tool_calls"):
            support_features.append("tool calls")
        if supports.get("parallel_tool_calls"):
            support_features.append("parallel tool calls")
        if supports.get("vision"):
            support_features.append("vision")
        if supports.get("streaming"):
            support_features.append("streaming")

        if support_features:
            description += f"Supports: {', '.join(support_features)}"

        return ModelInfo(
            id=str(data["id"]),
            name=str(data.get("name", data["id"])),
            provider="copilot",
            description=description.strip() or None,
            context_window=context_window,
            max_output_tokens=max_output_tokens,
            is_deprecated=not data.get("model_picker_enabled", True),
            owned_by=data.get("vendor"),
            input_modalities=input_modalities,
            output_modalities=output_modalities,
        )

    async def get_models(self) -> list[ModelInfo]:
        """Override the standard get_models to use Copilot's specific endpoint."""
        try:
            headers = self._get_headers()
            url = f"{self.base_url}/models"
            response = await anyenv.get(url, headers=headers, timeout=30)
            data = await response.json()
            if not isinstance(data, dict) or "data" not in data:
                msg = f"Invalid response format from Copilot API: {data}"
                raise RuntimeError(msg)  # noqa: TRY301
            models = []
            for model in data["data"]:
                # Skip models without model picker enabled
                if not model.get("model_picker_enabled", False):
                    continue
                capabilities = model.get("capabilities", {})
                if (
                    capabilities.get("type") != "chat"
                    or not capabilities.get("supports", {}).get("tool_calls", False)
                    or model["id"] in COPILOT_EXCLUDED_MODELS
                ):
                    continue

                models.append(self._parse_model(model))
        except Exception as e:
            msg = f"Failed to fetch models from Copilot: {e}"
            logger.exception(msg)
            raise RuntimeError(msg) from e
        else:
            return models

    def get_models_sync(self) -> list[ModelInfo]:
        """Synchronous version of get_models."""
        try:
            headers = self._get_headers()
            url = f"{self.base_url}/models"
            data = anyenv.get_json_sync(
                url, headers=headers, timeout=30, return_type=dict
            )
            if "data" not in data:
                msg = f"Invalid response format from Copilot API: {data}"
                raise RuntimeError(msg)  # noqa: TRY301

            models = []
            for model in data["data"]:
                # Skip models without model picker enabled
                if not model.get("model_picker_enabled", False):
                    continue

                # Get capabilities
                capabilities = model.get("capabilities", {})

                # Skip models that don't support chat or tool calls
                if (
                    capabilities.get("type") != "chat"
                    or not capabilities.get("supports", {}).get("tool_calls", False)
                    or model["id"] in COPILOT_EXCLUDED_MODELS
                ):
                    continue

                models.append(self._parse_model(model))
        except Exception as e:
            msg = f"Failed to fetch models from Copilot: {e}"
            logger.exception(msg)
            raise RuntimeError(msg) from e
        else:
            return models


if __name__ == "__main__":
    import asyncio

    # Set up logging
    logging.basicConfig(
        level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    async def test_provider():
        try:
            # Create provider
            provider = CopilotProvider()
            print(f"Provider initialized: {provider.base_url}")
            print("\nFetching models asynchronously...")
            models = await provider.get_models()
            print(f"Found {len(models)} models from Copilot API:")
            for i, model in enumerate(models, 1):
                print(f"\n{i}. {model.name} ({model.id})")
                print(f"   Context window: {model.context_window}")
            print("\nTesting token refresh mechanism...")
            # Force token refresh by setting expiry to now
            provider._token_manager._token_expires_at = datetime.now()
            models = await provider.get_models()
            print("Token refresh successful, found models:", len(models))

        except Exception as e:  # noqa: BLE001
            print(f"Error: {e}")

    # Run the async test function
    asyncio.run(test_provider())
