# Quickstart

Go from zero to your first verified API call in under 2 minutes.

> **Preview mode:** Accounts, API keys, and deposits are not yet publicly available. You can try verified inference in the [chat](https://verathos.ai/chat) without an account. The steps below will work once public API access launches.

## 1. Create an account

```bash
curl -X POST https://api.verathos.ai/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email": "you@example.com", "password": "your-password"}'
```

You'll receive an API key (`vrt_sk_...`) and $1.00 free credit, enough for thousands of tokens.

## 2. Make your first request

```bash
curl https://api.verathos.ai/v1/chat/completions \
  -H "Authorization: Bearer vrt_sk_YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "max_tokens": 256
  }'
```

Use `"auto"` to let Verathos pick the best available model, or specify one explicitly (e.g. `"qwen3.5-9b"`). The response is OpenAI-compatible (same format, same fields). Every response includes a cryptographic proof that the model weights weren't tampered with.

## 3. Use with any OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://api.verathos.ai/v1",
    api_key="vrt_sk_YOUR_KEY",
)

response = client.chat.completions.create(
    model="auto",
    messages=[{"role": "user", "content": "Explain ZK proofs in one sentence."}],
    max_tokens=256,
)
print(response.choices[0].message.content)
```

Works with Python, TypeScript, Go, Rust, and any language with an OpenAI-compatible client. For AI agent frameworks, see [integrations](integrations.md) (LiteLLM, LangChain, elizaOS, OpenClaw, and more).

## 4. Check your balance

```bash
curl https://api.verathos.ai/v1/balance \
  -H "Authorization: Bearer vrt_sk_YOUR_KEY"
```

## Available models

```bash
curl https://api.verathos.ai/v1/models \
  -H "Authorization: Bearer vrt_sk_YOUR_KEY"
```

The subnet maintains a curated model registry with verified weight commitments. Miners choose which registered models to serve, and the available list changes as miners join and leave the network.

## Alternative: Pay with USDC (no account needed)

Use [x402](https://docs.cdp.coinbase.com/x402) to pay per request with USDC on Base, with no registration and no API key. Install `pip install x402 eth-account` and see `examples/x402_client.py` for a working example. See the [User Guide](user_guide.md#x402-pay-per-request-usdc) for details.

## Example scripts

Ready-to-run Python scripts in the `examples/` directory:

- **`openai_client.py`**: Basic chat with the OpenAI SDK (simplest integration)
- **`streaming_client.py`**: Token-by-token streaming output
- **`x402_client.py`**: Pay with USDC on Base (no API key needed)

All support `--model`, `--quant` (e.g. `int4`, `fp16`), and `--gateway` flags.

## What's next?

- **[User Guide](user_guide.md)**: Deposits (TAO + USDC), withdrawals, x402, API keys
- **[API Reference](api.md)**: Full endpoint documentation
- **[What is Verathos?](intro.md)**: How verification works under the hood
- **[Verified Chat](https://verathos.ai/chat)**: Try it in the browser with no setup
