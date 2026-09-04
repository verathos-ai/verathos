# Verathos + Hermes Agent

Use [Verathos](https://verathos.ai) verified LLM inference with [Hermes Agent](https://github.com/NousResearch/hermes-agent) — no plugin needed.

Hermes Agent uses the OpenAI Python SDK under the hood, so Verathos works out of the box as a custom provider.

## Setup

### Option A: Config file (recommended)

Add to `~/.hermes/config.yaml`:

```yaml
model:
  provider: custom:verathos
  default: "auto"
  base_url: "https://api.verathos.ai/v1"
  api_key: "your-api-key"
```

Then start Hermes:

```bash
hermes
```

### Option B: Environment variables

```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_BASE_URL="https://api.verathos.ai/v1"
hermes
```

### Option C: CLI flag

```bash
hermes --model-provider custom --base-url https://api.verathos.ai/v1 --api-key your-api-key
```

## Model selection

Use `"auto"` (default) to let Verathos pick the best available model by miner score, or specify a model explicitly:

```yaml
model:
  default: "auto"           # best available (recommended)
  # default: "MODEL_ID_FROM_V1_MODELS"  # specific model
```

Available models: `curl https://api.verathos.ai/v1/models`

## What you get

Successful inference responses include authenticated Gleipnir verification
metadata. Ordinary requests bind the request and output to a frozen execution
commitment, while validators run unpredictable execution audits.

## Getting an API key

1. Go to [verathos.ai](https://verathos.ai) and create an account
2. Deposit TAO or USDC
3. Generate an API key

## Smart model routing

Hermes Agent's built-in smart routing (cheap model for simple tasks, strong model for complex ones) works with Verathos. Set different Verathos models for each tier:

```yaml
model:
  provider: custom:verathos
  default: "MODEL_ID_FROM_V1_MODELS"
  base_url: "https://api.verathos.ai/v1"
  api_key: "your-api-key"

smart_routing:
  enabled: true
  cheap_model: "MODEL_ID_FROM_V1_MODELS"
  strong_model: "MODEL_ID_FROM_V1_MODELS"
```
