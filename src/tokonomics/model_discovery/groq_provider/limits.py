from __future__ import annotations

from dataclasses import dataclass


# Data here is copied from groq developer console (no API access right now)
# https://console.groq.com/dashboard/limits


@dataclass
class ChatCompletionModel:
    """Represents rate limits for a chat completion model."""

    id: str
    requests_per_minute: int
    requests_per_day: int
    tokens_per_minute: int
    tokens_per_day: int | None = None  # None represents "No limit"


@dataclass
class SpeechToTextModel:
    """Represents rate limits for a speech-to-text model."""

    id: str
    requests_per_minute: int
    requests_per_day: int
    audio_seconds_per_hour: int
    audio_seconds_per_day: int


# Chat completion models
chat_models: dict[str, ChatCompletionModel] = {}
chat_model_data = [
    ChatCompletionModel("allam-2-7b", 30, 7000, 6000, None),
    ChatCompletionModel("deepseek-r1-distill-llama-70b", 30, 1000, 6000, None),
    ChatCompletionModel("deepseek-r1-distill-qwen-32b", 30, 1000, 6000, None),
    ChatCompletionModel("gemma2-9b-it", 30, 14400, 15000, 500000),
    ChatCompletionModel("llama-3.1-8b-instant", 30, 14400, 6000, 500000),
    ChatCompletionModel("llama-3.2-11b-vision-preview", 30, 7000, 7000, 500000),
    ChatCompletionModel("llama-3.2-1b-preview", 30, 7000, 7000, 500000),
    ChatCompletionModel("llama-3.2-3b-preview", 30, 7000, 7000, 500000),
    ChatCompletionModel("llama-3.2-90b-vision-preview", 15, 3500, 7000, 250000),
    ChatCompletionModel("llama-3.3-70b-specdec", 30, 1000, 6000, 100000),
    ChatCompletionModel("llama-3.3-70b-versatile", 30, 1000, 6000, 100000),
    ChatCompletionModel("llama-guard-3-8b", 30, 14400, 15000, 500000),
    ChatCompletionModel("llama3-70b-8192", 30, 14400, 6000, 500000),
    ChatCompletionModel("llama3-8b-8192", 30, 14400, 6000, 500000),
    ChatCompletionModel("mistral-saba-24b", 30, 1000, 6000, 500000),
    ChatCompletionModel("mixtral-8x7b-32768", 30, 14400, 5000, 500000),
    ChatCompletionModel("qwen-2.5-32b", 30, 1000, 6000, None),
    ChatCompletionModel("qwen-2.5-coder-32b", 30, 1000, 6000, None),
    ChatCompletionModel("qwen-qwq-32b", 30, 1000, 6000, None),
]

for chat_model in chat_model_data:
    chat_models[chat_model.id] = chat_model

# Speech-to-text models
speech_models: dict[str, SpeechToTextModel] = {}
speech_model_data = [
    SpeechToTextModel("distil-whisper-large-v3-en", 20, 2000, 7200, 28800),
    SpeechToTextModel("whisper-large-v3", 20, 2000, 7200, 28800),
    SpeechToTextModel("whisper-large-v3-turbo", 20, 2000, 7200, 28800),
]

for speech_model in speech_model_data:
    speech_models[speech_model.id] = speech_model
