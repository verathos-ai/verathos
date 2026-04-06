# litellm-verathos

LiteLLM custom provider for [Verathos](https://verathos.ai) -- verified LLM inference on the Bittensor network.

Every response from Verathos is backed by cryptographic proofs (ZK sumcheck + Merkle commitments) that guarantee the output was produced by the declared model. No output substitution is possible.

## Installation

```bash
pip install litellm-verathos
```

## Quick Start

```python
import litellm
from litellm_verathos import VerathosProvider

# Register the provider (once, at startup)
VerathosProvider.register()

# Use "verathos/auto" for automatic best-model selection
response = litellm.completion(
    model="verathos/auto",
    messages=[{"role": "user", "content": "Explain zero-knowledge proofs"}],
    api_key="vrt_sk_...",  # your Verathos API key
)
print(response.choices[0].message.content)
```

## Model Names

Use the `verathos/` prefix followed by any model identifier:

| Model string | What happens |
|---|---|
| `verathos/auto` | Verathos picks the best available model for you |
| `verathos/Qwen/Qwen3-30B-A3B` | Routes to a specific model |
| `verathos/meta-llama/Llama-3.3-70B-Instruct` | Routes to a specific model |

To discover available models, query the Verathos API directly:

```bash
curl https://api.verathos.ai/v1/models -H "Authorization: Bearer $VERATHOS_API_KEY"
```

## Authentication

Pass the API key in any of these ways (checked in order):

1. `api_key=` parameter on each call
2. `VERATHOS_API_KEY` environment variable

```bash
export VERATHOS_API_KEY="vrt_sk_..."
```

```python
# Then no api_key= needed
response = litellm.completion(
    model="verathos/auto",
    messages=[{"role": "user", "content": "Hello!"}],
)
```

### Getting an API Key

1. Visit [verathos.ai](https://verathos.ai) and sign up
2. Fund your account with TAO or USDC deposits
3. Generate an API key from the dashboard

## Streaming

Streaming works out of the box:

```python
response = litellm.completion(
    model="verathos/auto",
    messages=[{"role": "user", "content": "Write a haiku about cryptography"}],
    api_key="vrt_sk_...",
    stream=True,
)
for chunk in response:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

## Async

```python
import asyncio
import litellm
from litellm_verathos import VerathosProvider

VerathosProvider.register()

async def main():
    response = await litellm.acompletion(
        model="verathos/auto",
        messages=[{"role": "user", "content": "Hello!"}],
        api_key="vrt_sk_...",
    )
    print(response.choices[0].message.content)

asyncio.run(main())
```

## All Standard Parameters

Verathos is OpenAI-compatible, so all standard chat completion parameters work:

```python
response = litellm.completion(
    model="verathos/auto",
    messages=[{"role": "user", "content": "Solve x^2 - 4 = 0"}],
    api_key="vrt_sk_...",
    temperature=0.0,
    max_tokens=512,
    top_p=0.95,
    stop=["\n\n"],
    seed=42,
)
```

## Custom API Base

For self-hosted Verathos validators or local development:

```python
response = litellm.completion(
    model="verathos/auto",
    messages=[{"role": "user", "content": "Hello!"}],
    api_key="vrt_sk_...",
    api_base="http://localhost:8080/v1",
)
```

Or via environment variable:

```bash
export VERATHOS_API_BASE="http://localhost:8080/v1"
```

## LiteLLM Proxy (config.yaml)

You can also use the Verathos provider with the LiteLLM proxy server. Add to your `config.yaml`:

```yaml
model_list:
  - model_name: verathos-auto
    litellm_params:
      model: openai/auto
      api_base: https://api.verathos.ai/v1
      api_key: os.environ/VERATHOS_API_KEY

  - model_name: verathos-qwen
    litellm_params:
      model: openai/Qwen/Qwen3-30B-A3B
      api_base: https://api.verathos.ai/v1
      api_key: os.environ/VERATHOS_API_KEY
```

> **Note:** The LiteLLM proxy's `config.yaml` uses the `openai/` prefix since it handles
> OpenAI-compatible endpoints natively. The `litellm-verathos` Python package is
> for programmatic usage where you want the cleaner `verathos/` prefix.

## x402: Pay-Per-Request with USDC (No API Key Needed)

Verathos supports [x402](https://www.x402.org/) -- a protocol for HTTP-native micropayments. With x402, you pay per request using USDC on Base L2, with no account or API key required.

x402 works at the HTTP level (the server returns HTTP 402 with payment instructions, you sign a USDC payment and resend). Since this operates below the LiteLLM abstraction layer, use the [x402 client SDK](https://docs.cdp.coinbase.com/x402/docs/quickstart) directly for pay-per-request:

```python
# x402 example (direct, not through LiteLLM)
from openai import OpenAI
from x402.client import create_x402_client
import httpx

x402_client = create_x402_client(
    httpx.Client(),
    wallet,  # your Base wallet with USDC
)
client = OpenAI(
    base_url="https://api.verathos.ai/v1",
    api_key="x402",  # any placeholder
    http_client=x402_client,
)
response = client.chat.completions.create(
    model="auto",
    messages=[{"role": "user", "content": "Hello!"}],
)
```

## Why Verathos?

Traditional LLM APIs are a black box -- you have no way to verify that the provider actually ran your prompt through the model they claim. Verathos changes this:

- **Verified inference**: Every response includes cryptographic proofs that the output was computed by the declared model
- **No output substitution**: SHA256 output commitments + Fiat-Shamir binding prevent response tampering
- **Decentralized**: Runs on Bittensor's incentive network -- miners compete to serve models, validators verify proofs
- **OpenAI-compatible**: Drop-in replacement for any OpenAI-compatible client

## License

MIT
