"""Ollama provider."""

from __future__ import annotations

import socket
from typing import TYPE_CHECKING, Any

from tokonomics.model_discovery.base import ModelProvider
from tokonomics.model_discovery.model_info import ModelInfo, ModelPricing


if TYPE_CHECKING:
    from tokonomics.model_discovery.model_info import Modality


class OllamaProvider(ModelProvider):
    """Ollama local API provider."""

    def __init__(self, base_url: str = "http://localhost:11434"):
        super().__init__()
        self.base_url = base_url
        self.headers = {}
        self.params = {}

    def is_available(self) -> bool:
        """Check whether Ollama is running and available."""
        try:
            # Extract host and port from base_url
            if "://" in self.base_url:
                host_port = self.base_url.split("://")[1]
            else:
                host_port = self.base_url

            if ":" in host_port:
                host, port_str = host_port.split(":", 1)
                port = int(port_str.split("/")[0])  # Handle paths after port
            else:
                host = host_port
                port = 11434  # Default Ollama port

            # Try to connect to the Ollama server
            with socket.create_connection((host, port), timeout=2):
                return True
        except (ValueError, OSError):
            return False

    async def get_models(self) -> list[ModelInfo]:
        """Fetch available models from Ollama using the /api/tags endpoint."""
        from anyenv import HttpError, get_json, post_json

        try:
            data = await get_json(
                f"{self.base_url}/api/tags",
                headers=self.headers,
                params=self.params,
                cache=True,
                return_type=dict,
            )

            if not isinstance(data, dict) or "models" not in data:
                msg = f"Invalid response format from {self.__class__.__name__}"
                raise RuntimeError(msg)

            # Fetch detailed info for each model
            models = []
            for item in data["models"]:
                try:
                    # Get detailed model info using /api/show
                    detail_data = await post_json(
                        f"{self.base_url}/api/show",
                        json_data={"name": item["name"]},
                        headers=self.headers,
                        cache=True,
                        return_type=dict,
                    )
                    models.append(self._parse_model(item, detail_data))
                except HttpError:
                    # Fallback to basic info if detailed fetch fails
                    models.append(self._parse_model(item))
        except HttpError as e:
            msg = f"Failed to fetch models from {self.__class__.__name__}: {e}"
            raise RuntimeError(msg) from e
        else:
            return models

    def _parse_model(
        self, data: dict[str, Any], detail_data: dict[str, Any] | None = None
    ) -> ModelInfo:
        """Parse Ollama API response into ModelInfo."""
        name = str(data["name"])

        # Extract model details from basic info
        details = data.get("details", {})
        family = details.get("family", "")
        parameter_size = details.get("parameter_size", "")
        quantization = details.get("quantization_level", "")

        # Extract additional info from detailed response if available
        context_window = None
        exact_param_count = None
        model_parameters = {}

        if detail_data:
            model_info = detail_data.get("model_info", {})

            # Get actual context window from model_info
            for key, value in model_info.items():
                if key.endswith(".context_length"):
                    context_window = int(value)
                    break

            # Get exact parameter count
            exact_param_count = model_info.get("general.parameter_count")

            # Parse model parameters (temperature, top_k, etc.)
            params_str = detail_data.get("parameters", "")
            if params_str:
                for line in params_str.strip().split("\n"):
                    if line.strip():
                        parts = line.strip().split(None, 1)
                        if len(parts) == 2:  # noqa: PLR2004
                            param_name, param_value = parts
                            # Convert numeric values
                            try:
                                if "." in param_value.strip('"'):
                                    model_parameters[param_name] = float(
                                        param_value.strip('"')
                                    )
                                elif param_value.strip('"').isdigit():
                                    model_parameters[param_name] = int(
                                        param_value.strip('"')
                                    )
                                else:
                                    model_parameters[param_name] = param_value.strip('"')
                            except ValueError:
                                model_parameters[param_name] = param_value.strip('"')

        # Fallback to estimation if no detailed context window found
        if not context_window:
            context_window = self._estimate_context_window(
                parameter_size, data.get("size", 0)
            )

        # Build description from available details
        description_parts = []
        if family:
            description_parts.append(f"Family: {family}")
        if parameter_size:
            description_parts.append(f"Parameters: {parameter_size}")
        if exact_param_count and exact_param_count != parameter_size:
            description_parts.append(f"Exact count: {exact_param_count:,}")
        if quantization:
            description_parts.append(f"Quantization: {quantization}")

        description = " | ".join(description_parts) if description_parts else None

        # Determine modalities based on family
        input_modalities: set[Modality] = {"text"}
        output_modalities: set[Modality] = {"text"}

        if (family and "llava" in family.lower()) or (
            family
            and any(
                vision in family.lower() for vision in ["vision", "visual", "multimodal"]
            )
        ):
            input_modalities.add("image")

        # Build comprehensive metadata
        metadata = {
            "size_bytes": data.get("size", 0),
            "digest": data.get("digest"),
            "modified_at": data.get("modified_at"),
            "format": details.get("format"),
            "families": details.get("families", []),
            "quantization_level": quantization,
            "parameter_size": parameter_size,
            "parent_model": details.get("parent_model"),
        }

        # Add detailed info if available
        if detail_data:
            metadata.update({
                "exact_parameter_count": exact_param_count,
                "model_parameters": model_parameters,
                "has_template": bool(detail_data.get("template")),
                "license_type": "Apache License 2.0"
                if "Apache License" in detail_data.get("license", "")
                else None,
            })

            # Add model architecture details from model_info
            if model_info := detail_data.get("model_info", {}):
                arch_info = {}
                for key, value in model_info.items():
                    if any(
                        key.startswith(prefix) for prefix in ["general.", f"{family}."]
                    ):
                        clean_key = key.split(".", 1)[1] if "." in key else key
                        arch_info[clean_key] = value
                if arch_info:
                    metadata["architecture"] = arch_info

        return ModelInfo(
            id=name,
            name=name,
            provider="ollama",
            description=description,
            pricing=ModelPricing(),  # Local models are free
            owned_by=family if family else None,
            context_window=context_window,
            is_free=True,  # Local Ollama models are free to use
            input_modalities=input_modalities,
            output_modalities=output_modalities,
            metadata=metadata,
        )

    def _estimate_context_window(
        self, parameter_size: str, size_bytes: int
    ) -> int | None:
        """Estimate context window based on parameter size and model family."""
        if not parameter_size:
            return None

        param_size_lower = parameter_size.lower()

        # Common context windows for different model sizes
        if (
            "1b" in param_size_lower
            or "1.1b" in param_size_lower
            or "3b" in param_size_lower
            or "7b" in param_size_lower
            or "8b" in param_size_lower
        ):
            return 8192
        if (
            "13b" in param_size_lower
            or "14b" in param_size_lower
            or "30b" in param_size_lower
            or "34b" in param_size_lower
            or "70b" in param_size_lower
        ):
            return 4096
        if "405b" in param_size_lower:
            return 8192

        # Default fallback
        return 4096


if __name__ == "__main__":
    import asyncio

    provider = OllamaProvider()
    if provider.is_available():
        models = asyncio.run(provider.get_models())
        for model in models:
            print(model.format())
            print("-" * 50)
    else:
        print("Ollama is not running or not available")
