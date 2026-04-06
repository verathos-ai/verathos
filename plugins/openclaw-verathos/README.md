# Verathos Provider

OpenClaw provider plugin for [Verathos](https://verathos.ai) — verified LLM inference on Bittensor.

Every response from Verathos is backed by cryptographic proofs (ZK sumcheck + Merkle commitments), ensuring the model actually ran the computation it claims.

## Features

- OpenAI-compatible chat completions (streaming and non-streaming)
- Automatic model selection with `model: "auto"`
- Live model discovery from the Verathos API
- API key authentication
- Custom endpoint support for self-hosted proxies

## Setup

### 1. Get an API key

Sign up at [verathos.ai](https://verathos.ai) and create an API key, or deposit TAO/USDC to receive credits automatically.

### 2. Install the plugin

Copy this directory into your OpenClaw `extensions/` folder, or add it to your workspace plugins.

### 3. Configure

**Interactive (recommended):**

Run the OpenClaw setup wizard and select "Verathos" when prompted for a provider.

**Environment variable:**

```bash
export VERATHOS_API_KEY="vrt_sk_your_key_here"
```

**CLI flag:**

```bash
openclaw --verathos-api-key "vrt_sk_your_key_here"
```

## Models

Verathos serves multiple models across its miner network. Use `"auto"` to let the network pick the best available model, or specify a model explicitly:

| Model ref | Description |
|-----------|-------------|
| `verathos/auto` | Best available model (recommended) |
| `verathos/<model-id>` | Specific model from `/v1/models` |

The plugin fetches available models from the Verathos API at runtime, so the model list stays current as miners join or leave the network.

## Custom endpoint

For validators running their own proxy, use the custom endpoint auth method during setup. This lets you point to any Verathos-compatible API:

```
Base URL: https://your-proxy.example.com/v1
```

## How it works

Verathos is a subnet on the Bittensor network where miners serve LLM inference with cryptographic proofs:

1. You send a chat completion request through the Verathos proxy
2. The proxy routes your request to a score-weighted miner
3. The miner runs inference and generates a ZK proof of correct computation
4. Validators verify proofs and set on-chain weights
5. You receive the response with proof verification metadata

All of this happens transparently behind the standard OpenAI-compatible API.

## Payment options

- **API key + credits**: Deposit TAO or USDC, pay per token
- **x402 pay-per-request**: Sign a USDC payment per request (no account needed)

## Links

- Website: [verathos.ai](https://verathos.ai)
- API base URL: `https://api.verathos.ai/v1`
- Documentation: [docs.verathos.ai](https://docs.verathos.ai)
