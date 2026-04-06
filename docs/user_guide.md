# Verathos User Guide

How to use Verathos as an end user, from getting an API key to making inference requests.

> **Preview mode:** Accounts, API keys, and deposits are not yet publicly available. You can try verified inference in the [chat](https://verathos.ai/chat) without an account. The guide below describes the full system that will be available once public API access launches.

## Overview

1. Create an API key (email or wallet)
2. Add funds (TAO, USDC on Base, or x402 pay-per-request)
3. Send inference requests (OpenAI-compatible API)

## 1. Create an API Key

### Option A: Email/Password (quickest)

No wallet needed:

```bash
curl -X POST https://api.verathos.ai/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email": "you@example.com", "password": "pick-a-password"}'
# Returns: {"api_key": "vrt_sk_...", "user_id": "uuid-..."}
```

Already registered? Login:

```bash
curl -X POST https://api.verathos.ai/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email": "you@example.com", "password": "pick-a-password"}'
```

### Option B: EVM Wallet Signature

For crypto-native users with an EVM wallet (MetaMask, etc.):

```bash
# 1. Get a challenge
curl https://api.verathos.ai/v1/auth/challenge
# Returns: {"challenge": "vrt_auth_abc123_1710000000"}

# 2. Sign with your wallet (EIP-191 personal_sign), then:
curl -X POST https://api.verathos.ai/v1/auth/wallet \
  -H "Content-Type: application/json" \
  -d '{
    "address": "0xYourAddress",
    "challenge": "vrt_auth_abc123_1710000000",
    "signature": "0x...",
    "name": "my-app"
  }'
# Returns: {"api_key": "vrt_sk_...", "user_id": "uuid-..."}
```

**Save the API key.** It is shown once and never stored in plaintext.

### Managing API Keys

```bash
# List your keys
curl https://api.verathos.ai/v1/api-keys \
  -H "Authorization: Bearer vrt_sk_..."

# Revoke a key
curl -X DELETE https://api.verathos.ai/v1/api-keys/<key_hash> \
  -H "Authorization: Bearer vrt_sk_..."
```

## 2. Add Funds

You need credits to pay for inference. Three options:

- **TAO**: deposit TAO on Bittensor
- **USDC**: deposit USDC on Base L2
- **x402**: pay per request with USDC, no deposit needed (see below)

Each user gets unique deposit addresses for both TAO and USDC:

```bash
curl https://api.verathos.ai/v1/user/deposit-address \
  -H "Authorization: Bearer vrt_sk_..."
```

Deposits are detected within 30 seconds. Your balance updates automatically. Balances are tracked separately; inference deducts from USD first, then TAO.

```bash
curl https://api.verathos.ai/v1/balance \
  -H "Authorization: Bearer vrt_sk_..."
# Returns: {"balance_tao": 0.1, "balance_usd": 10.0, "tao_usd": 500.0, "total_balance_usd": 60000000, ...}

# Include withdrawable amounts (requires chain lookup):
curl "https://api.verathos.ai/v1/balance?withdrawable=1" \
  -H "Authorization: Bearer vrt_sk_..."
# Additional fields: "withdrawable_tao": 0.095, "withdrawable_usd": 9.99
```

### TAO

Send TAO to the `tao.address` (SS58) returned above using any Bittensor wallet:

```bash
btcli wallet transfer --dest 5G6xBZHQ... --amount 0.1 --wallet default
```

Withdraw unconsumed TAO at any time. Supports both EVM (`0x...`) and SS58 (`5...`) destinations (SS58 uses `ISubtensorBalanceTransfer` precompile internally):

```bash
curl -X POST https://api.verathos.ai/v1/user/withdraw \
  -H "Authorization: Bearer vrt_sk_..." \
  -H "Content-Type: application/json" \
  -d '{"amount_tao": 0.05, "destination": "0xYourEVMAddress"}'
# Returns: {"tx_hash": "0x...", "amount_tao": 0.05, "destination_type": "evm", "chain": "bittensor", "status": "completed"}
```

### USDC (Base L2)

Send USDC to the `base.address` returned above using any wallet on the Base network. Credited 1:1 as USD balance.

> Only USDC is supported. Native ETH sent to deposit addresses will not be credited.

Withdraw unconsumed USDC at any time. Destination must be a Base EVM address (validator auto-funds gas):

```bash
curl -X POST https://api.verathos.ai/v1/user/withdraw \
  -H "Authorization: Bearer vrt_sk_..." \
  -H "Content-Type: application/json" \
  -d '{"amount_usdc": 5.0, "destination": "0xYourBaseAddress"}'
# Returns: {"tx_hash": "0x...", "amount_usdc": 5.0, "destination_type": "evm", "chain": "base", "status": "completed"}
```

Withdrawals are rate limited to one per 5 minutes. Minimum: 0.01 TAO or $1 USDC. Only on-chain balance minus gas is withdrawable.

## 3. Send Inference Requests

The API is OpenAI-compatible. Use any OpenAI SDK or HTTP client.

### List available models

```bash
curl https://api.verathos.ai/v1/models \
  -H "Authorization: Bearer vrt_sk_..."
```

Each model shows `available_quants` (e.g. `["fp16", "gptq_int4"]`) and `qualified_ids` for requesting a specific quantization.

### Chat completions

```bash
curl -X POST https://api.verathos.ai/v1/chat/completions \
  -H "Authorization: Bearer vrt_sk_..." \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

### Model selection

Use `"auto"` to let Verathos pick the best available model. It selects the highest-scored healthy miner across all models. If that miner fails, it automatically falls through to the next-best endpoint regardless of model. This is the recommended default for most use cases.

You can also specify a model explicitly:

```bash
# Auto - best available model and miner (recommended)
"model": "auto"

# Specific model
"model": "qwen3-8b"

# Specific model + quantization
"model": "qwen3-8b:fp16"
"model": "qwen3-8b:gptq_int4"
"model": "qwen3-8b:fp8"
```

Check `/v1/models` for available models and quantizations.

### Python (OpenAI SDK)

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://api.verathos.ai/v1",
    api_key="vrt_sk_...",
)

response = client.chat.completions.create(
    model="qwen3-8b:fp16",  # or just "qwen3-8b" for auto
    messages=[{"role": "user", "content": "Explain zero-knowledge proofs"}],
    stream=True,
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

Works with any OpenAI-compatible tool. We provide dedicated plugins for [LiteLLM, LangChain, elizaOS, OpenClaw, Hermes, and AutoGen](integrations.md) with features like proof metadata, model discovery, and x402 payments.

### Example scripts

Ready-to-run Python scripts in the `examples/` directory:

- **`openai_client.py`**: Basic chat completion
- **`streaming_client.py`**: Token-by-token streaming
- **`x402_client.py`**: Pay with USDC (no API key)

All support `--model`, `--quant`, and `--gateway` flags.

## x402 Pay-Per-Request (USDC)

Pay per request with USDC on Base, with no deposit, no API key, and no account needed. Uses the [Coinbase x402 protocol](https://docs.cdp.coinbase.com/x402). Built for autonomous agents: machine-native payment at inference time.

**How it works**: Send a request without auth → gateway returns HTTP 402 with payment requirements → the x402 SDK signs a USDC payment on Base and retransmits → USDC transfers on-chain → inference proceeds.

**Setup**:
- `pip install x402 eth-account`
- Use the `x402HttpxClient` with your EVM wallet private key
- The SDK handles the 402 → sign → retry flow automatically
- See `examples/x402_client.py` in the repo for a working example

**Requirements**: EVM wallet with USDC + ETH (for gas) on Base. Testnet: Base Sepolia.

**Cost**: x402 uses the `exact` scheme, charging upfront based on `max_tokens` (worst case). Each request is an on-chain transfer with ~$0.001 gas on Base mainnet. For small requests, gas dominates; use API key + deposit instead. Set `max_tokens` conservatively to avoid overpaying.

## TEE Inference (Trusted Execution Environments)

> **Not yet available on mainnet.** TEE support will be enabled once reproducible builds are validated across hardware platforms. Available on testnet.

TEE inference runs models inside hardware-isolated enclaves (Intel TDX, AMD SEV-SNP, NVIDIA Confidential Computing). The miner operator cannot access model inputs or outputs at the hardware level. There are two ways to use TEE, depending on your trust model. Authentication and billing work the same as standard inference.

### Find TEE-enabled models

Models with TEE-enabled miners show `:tee` qualified IDs in the model list:

```bash
curl https://api.verathos.ai/v1/models -H "Authorization: Bearer vrt_sk_..."
# qualified_ids like "qwen3-8b:gptq_int4:tee" indicate TEE availability
```

If no `:tee` variants appear, no TEE-enabled miners are currently online.

### Option 1: TEE-verified inference (OpenAI-compatible)

Append `:tee` to any model qualifier to route to a TEE-enabled miner. The API stays OpenAI-compatible, so no code changes are required beyond the model name. The gateway sees the plaintext, but the miner operator cannot (hardware isolation).

```bash
curl -X POST https://api.verathos.ai/v1/chat/completions \
  -H "Authorization: Bearer vrt_sk_..." \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-8b:gptq_int4:tee",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

```python
from openai import OpenAI

client = OpenAI(base_url="https://api.verathos.ai/v1", api_key="vrt_sk_...")
response = client.chat.completions.create(
    model="qwen3-8b:gptq_int4:tee",
    messages=[{"role": "user", "content": "Explain zero-knowledge proofs"}],
    stream=True,
)
for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

The response includes `proof_mode: "attestation"` and TEE details (enclave key, attestation hash, platform) instead of cryptographic proof data.

### Option 2: End-to-end encrypted inference (TEEClient)

For privacy-sensitive workloads where neither the gateway nor the miner operator should see plaintext. Prompts are encrypted on your machine and decrypted only inside the miner's enclave.

```bash
pip install verathos  # includes pynacl for X25519 + XSalsa20
```

**Non-streaming:**

```python
from verallm.tee.client import TEEClient

# Specify a model - the gateway assigns a TEE miner and returns its enclave key
client = TEEClient.from_miner("https://api.verathos.ai", model="qwen3-8b")

# Verify hardware attestation (genuine TEE, correct model weights)
assert client.verify_attestation()

# Encrypt, send, decrypt - all handled internally
result = client.chat(
    messages=[{"role": "user", "content": "Explain zero-knowledge proofs"}],
    max_tokens=512,
)
print(result["output_text"])
```

**Streaming:**

```python
with TEEClient.from_miner("https://api.verathos.ai", model="qwen3-8b") as client:
    for event in client.chat_stream(
        messages=[{"role": "user", "content": "Hello!"}],
        max_tokens=256,
    ):
        if event["type"] == "token":
            print(event["text"], end="", flush=True)
        elif event["type"] == "done":
            print(f"\nProof verified: {event['proof_verified']}")
```

Use `model="auto"` to let the gateway pick the best available TEE miner across all models. The client handles key exchange, encryption, decryption, and automatic retry if a miner goes offline.

See [Inference Protocol: TEE Verification](inference_protocol.md#tee-verification-trusted-execution-environments) for the full technical details on attestation, cryptography, and trust model.

## Pricing

Pricing is in USD per million tokens, with four tiers based on model size (Small, Medium, Large, XL). Every request is verified, and verification cost is included in the price. TAO-paying users are charged at the live market rate.

Use `GET /v1/price` to compute exact costs before sending, or `GET /v1/models` to see per-model pricing. See the [pricing table](intro.md#pricing-usd-per-1m-tokens) for full details.

## Rate Limits

All endpoints are rate-limited per-IP (public) or per-API-key (authenticated). Inference is limited to 300 requests/min, auth endpoints to 10/min, withdrawals to 1 per 5 minutes. Exceeding limits returns HTTP 429 with a `Retry-After` header. All responses include `X-RateLimit-Limit` and `X-RateLimit-Remaining` headers.
