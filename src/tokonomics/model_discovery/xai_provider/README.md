# xAI Provider

Provider for xAI's Grok models via their API.

## Setup

1. Get an API key from [xAI Console](https://console.x.ai/)
2. Set the environment variable:
   ```bash
   export XAI_API_KEY="your-api-key-here"
   ```

## Usage

```python
from tokonomics.model_discovery.xai_provider import XAIProvider

# Using environment variable
provider = XAIProvider()

# Or pass API key directly
provider = XAIProvider(api_key="your-api-key")

# Get all models
models = await provider.get_models()
```

## API Endpoint

The provider uses the xAI Language Models API:
- **Endpoint**: `https://api.x.ai/v1/language-models`
- **Authentication**: Bearer token in `Authorization` header
- **Response Format**: Custom format with `models` array (not standard OpenAI format)

## Pricing

xAI provides pricing in micro-units that are automatically converted to dollars per token:
- API values are divided by 1,000,000 to get USD per token
- Supports prompt, completion, image, cached prompt, and web search pricing

## Supported Models

The provider automatically discovers all available xAI models, including:
- Grok-3 (text)
- Grok-3 Mini (text) 
- Grok-2 Vision (text + image)

## Features

- **Modalities**: Automatically detects text and image input/output support
- **Pricing**: Full pricing information including caching and web search costs
- **Metadata**: Model fingerprints, versions, creation timestamps, and aliases
- **Error Handling**: Robust error handling with descriptive messages

## Notes

⚠️ **Pricing Units**: The xAI API returns pricing in micro-units. This provider converts them to standard dollars per token, but the resulting values may seem high compared to other providers. Verify pricing with xAI's official documentation.