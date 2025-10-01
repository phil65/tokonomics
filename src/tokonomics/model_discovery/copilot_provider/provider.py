"""GitHub Copilot provider for model discovery."""

from __future__ import annotations

from datetime import datetime
import logging
from typing import TYPE_CHECKING, Any

from tokonomics.model_discovery.base import ModelProvider
from tokonomics.model_discovery.copilot_provider.token_manager import token_manager
from tokonomics.model_discovery.model_info import ModelInfo


if TYPE_CHECKING:
    from tokonomics.model_discovery.model_info import Modality


logger = logging.getLogger(__name__)

# Models to exclude from the provider
COPILOT_EXCLUDED_MODELS = set[str]()  # "o1"


class CopilotProvider(ModelProvider):
    """GitHub Copilot API provider."""

    def __init__(self):
        super().__init__()
        self._token_manager = token_manager
        self.base_url = self._token_manager._api_endpoint
        self.params = {}

    def is_available(self) -> bool:
        """Check whether the provider is available for use."""
        return self._token_manager.is_available()

    def _parse_model(self, data: dict[str, Any]) -> ModelInfo:
        """Parse Copilot API response into ModelInfo."""
        # Extract capabilities and limits
        capabilities = data.get("capabilities", {})
        limits = capabilities.get("limits", {})

        # Extract context window, input and output tokens
        context_window = limits.get("max_context_window_tokens")
        _max_input_tokens = limits.get("max_input_tokens")
        max_output_tokens = limits.get("max_output_tokens")

        # Determine modalities
        input_modalities: set[Modality] = {"text"}
        output_modalities: set[Modality] = {"text"}

        # Add vision capability if supported
        if capabilities.get("supports", {}).get("vision"):
            input_modalities.add("image")

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
        import anyenv

        try:
            headers = await self._token_manager.generate_headers()
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


if __name__ == "__main__":
    import anyenv

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
    anyenv.run_sync(test_provider())
