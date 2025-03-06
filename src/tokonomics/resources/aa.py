from __future__ import annotations


ARTIFICIAL_ANALYSIS = {
    "o1": 90.0,
    "o3-mini": 89.0,
    "o1-preview": 85.0,  # From Microsoft Azure provider
    "o1-mini": 84.0,  # Taking the higher score from Microsoft Azure
    "gpt-4o-2024-08-06": 78.0,
    "gpt-4o-2024-05-13": 78.0,
    "gpt-4o": 75.0,  # Taking the higher score from Microsoft Azure
    "gpt-4o-mini": 73.0,
    "llama-3-3-instruct-70b": 75.0,  # Using the highest score (Together.ai Turbo)
    "llama-3-3-instruct-70b-vision": 68.0,  # Using the highest score
    "llama-3-1-instruct-405b": 75.0,  # Using the highest score (Hyperbolic)
    "llama-3-1-instruct-70b": 69.0,  # Using the highest score (Hyperbolic)
    "llama-3-2-instruct-90b-vision": 68.0,  # Using the highest score (Google Vertex)
    "llama-3-2-instruct-11b-vision": 54.0,  # Using the highest score
    "llama-3-1-instruct-8b": 54.0,  # Using the highest score
    "llama-3-2-instruct-3b": 50.0,  # Using the highest score (Fireworks)
    "llama-3-2-instruct-1b": 26.0,
    "gemini-2-0-flash-experimental": 82.0,
    "gemini-1-5-pro": 80.0,
    "gemini-1-5-flash": 74.0,
    "gemini-1-5-pro-may-2024": 72.0,
    "gemma-2-27b": 61.0,
    "gemma-2-9b": 55.0,
    "gemini-1-5-flash-8b": 47.0,
    "gemini-1-5-flash-may-2024": 60.0,
    "gemini-experimental-dec-2024": 0.0,
    "claude-35-sonnet": 80.0,
    "claude-35-sonnet-june-24": 76.0,
    "claude-3-opus": 70.0,
    "claude-3-5-haiku": 68.0,
    "claude-3-haiku": 55.0,
    "pixtral-large-2411": 74.0,
    "mistral-large-2": 74.0,
    "mistral-large-2407": 74.0,
    "mistral-small-3": 72.0,
    "mistral-small": 61.0,
    "mistral-8x22b-instruct": 62.0,  # Using the highest score (Mistral)
    "pixtral": 57.0,  # Using the highest score (Hyperbolic)
    "ministral-8b": 56.0,
    "mistral-nemo": 54.0,  # Using the highest score (Deepinfra)
    "ministral-3b": 53.0,
    "mixtral-8x7b-instruct": 43.0,  # Using the highest score (Fireworks)
    "codestral-mamba": 33.0,
    "codestral": 0.0,
    "command-r-plus": 55.0,
    "command-r-plus-04-2024": 47.0,  # Using the highest score (Cohere)
    "command-r-03-2024": 37.0,  # Using the highest score (Cohere)
    "command-r": 51.0,
    "aya-expanse-32b": 0.0,
    "aya-expanse-8b": 0.0,
    "grok-beta": 72.0,
    "nova-pro": 75.0,
    "nova-lite": 70.0,
    "nova-micro": 65.0,
    "phi-4": 76.0,
    "phi-3-medium": 0.0,
    "dbrx": 50.0,  # Using the highest score (Databricks)
    "llama-3-1-nemotron-instruct-70b": 72.0,
    "jamba-1-5-large": 64.0,
    "jamba-1-5-mini": 46.0,
    "deepseek-r1": 89.0,
    "deepseek-v3": 80.0,  # Using the highest score (DeepSeek)
    "deepseek-llm-67b-chat": 47.0,
    "deepseek-r1-distill-llama-70b": 0.0,
    "deepseek-v2-5-sep-2024": 0.0,
    "qwen-2-5-max": 79.0,
    "qwen2-5-72b-instruct": 78.0,  # Using the highest score (Deepinfra)
    "qwen2-5-coder-32b-instruct": 72.0,
    "qwen-turbo": 71.0,
    "qwen2-72b-instruct": 68.0,
    "QwQ-32B-Preview": 0.0,
    "qwen2.5-32b-instruct": 0.0,
    "yi-large": 62.0,
    "gpt-4-turbo": 75.0,
    "gpt-4": 0.0,
    "llama-3-instruct-70b": 62.0,  # Using the highest scores (Hyperbolic/Novita)
    "llama-3-instruct-8b": 53.0,  # Using the highest score (Novita)
    "llama-2-chat-7b": 0.0,
    "gemini-pro": 0.0,
    "claude-3-sonnet": 57.0,
    "claude-21": 0.0,
    "claude-2": 0.0,
    "mistral-small-2402": 59.0,
    "mistral-large": 57.0,  # Using the highest score (Mistral)
    "mistral-7b-instruct": 33.0,  # Using the highest score (Novita)
    "codestral-2405": 0.0,
    "mistral-medium": 0.0,
    "sonar-3-1-small-chat": 0.0,
    "sonar-3-1-large-chat": 0.0,
    "openchat-35": 44.0,
    "jamba-instruct": 28.0,
}
