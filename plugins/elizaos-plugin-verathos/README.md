# @elizaos/plugin-verathos

Verified LLM inference for elizaOS agents via [Verathos](https://verathos.ai) on [Bittensor](https://bittensor.com).

Every inference response is backed by cryptographic proofs (ZK sumcheck + Merkle commitments) — your agent can prove it ran the exact model it claims, with no output substitution.

## Quick Start

```bash
bun add @elizaos/plugin-verathos
```

### Option A: API Key (prepaid credits)

```json
{
  "name": "my-agent",
  "plugins": ["@elizaos/plugin-verathos"],
  "settings": {
    "VERATHOS_API_KEY": "your-api-key"
  }
}
```

### Option B + C: x402 modes — temporarily unavailable

> **Notice:** the x402 paths below (CDP wallet and raw key) are
> **disabled in this plugin** as of Verathos v0.1.10 because the gateway
> migrated to the x402 `upto` scheme (Permit2-based, post-inference
> settlement), and the upstream **TypeScript** x402 SDK does not yet
> support `upto` (only `exact`).  Calling x402 mode from this plugin
> will raise a clear error pointing you here.
>
> **Workarounds available today:**
> - Use Option A (API key + USDC deposit) above.
> - Or call `api.verathos.ai` directly using the **Python** x402 SDK,
>   which does support upto.  See
>   [`examples/x402_client.py`](https://github.com/verathos-ai/verathos/blob/main/examples/x402_client.py)
>   for a complete working reference (including the session-pass
>   aggregation pattern that lets agents pay per-token instead of
>   hitting the per-call gas floor).
>
> The config below is preserved for reference and will become valid
> again once the TypeScript SDK lands upto support.

### Option B: x402 with Coinbase CDP Wallet (production) — disabled, see notice above

Signing keys live in a Coinbase MPC TEE — never exposed to your application.

```json
{
  "name": "my-agent",
  "plugins": ["@elizaos/plugin-verathos"],
  "settings": {
    "CDP_API_KEY_ID": "your-cdp-api-key-id",
    "CDP_API_KEY_SECRET": "your-cdp-api-key-secret",
    "CDP_WALLET_SECRET": "your-cdp-wallet-secret"
  }
}
```

Requires `@coinbase/cdp-sdk`: `bun add @coinbase/cdp-sdk`

### Option C: x402 with Raw Key (dev/testnet) — disabled, see notice above

```json
{
  "name": "my-agent",
  "plugins": ["@elizaos/plugin-verathos"],
  "settings": {
    "VERATHOS_X402_PRIVATE_KEY": "0x...",
    "VERATHOS_X402_TESTNET": "true"
  }
}
```

Use a dedicated hot wallet with limited funds. Never use your main wallet.

## Configuration

### General

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `VERATHOS_API_URL` | No | `https://api.verathos.ai/v1` | Verathos API base URL |
| `VERATHOS_SMALL_MODEL` | No | `auto` | Model for `TEXT_SMALL` requests |
| `VERATHOS_LARGE_MODEL` | No | `auto` | Model for `TEXT_LARGE` requests |
| `VERATHOS_EMBEDDING_MODEL` | No | `auto` | Model for `TEXT_EMBEDDING` requests |

### Auth (choose one)

| Variable | Mode | Description |
|----------|------|-------------|
| `VERATHOS_API_KEY` | API key | Prepaid credits from Verathos |
| `CDP_API_KEY_ID` | x402 CDP | Coinbase CDP API key ID |
| `CDP_API_KEY_SECRET` | x402 CDP | Coinbase CDP API key secret |
| `CDP_WALLET_SECRET` | x402 CDP | CDP wallet secret (TEE auth) |
| `VERATHOS_X402_CDP_ACCOUNT` | x402 CDP | Account name (default: `verathos-agent`) |
| `VERATHOS_X402_PRIVATE_KEY` | x402 raw | EVM private key (dev/testnet only) |
| `VERATHOS_X402_TESTNET` | x402 | Use Base Sepolia instead of mainnet |

**Priority:** CDP wallet > raw private key > API key.

When model is set to `auto` (default), the API selects the best available model based on current miner scores.

## Auth Modes

### 1. API Key Mode

Traditional prepaid credits. Register at [verathos.ai](https://verathos.ai), deposit TAO or USDC, generate an API key.

```bash
export VERATHOS_API_KEY="your-api-key"
```

### 2. x402 CDP Mode (recommended for production)

Uses [Coinbase Developer Platform](https://docs.cdp.coinbase.com) MPC wallets. Signing keys live in an AWS Nitro Enclave TEE — even if your server is compromised, the attacker cannot extract the private key.

```bash
export CDP_API_KEY_ID="..."
export CDP_API_KEY_SECRET="..."
export CDP_WALLET_SECRET="..."
# Optional
export VERATHOS_X402_CDP_ACCOUNT="my-agent-wallet"
```

Get credentials from the [CDP Portal](https://portal.cdp.coinbase.com):
1. Create a project and generate **Secret API Keys** (API key ID + secret)
2. Go to Products > Server Wallets > **Generate Wallet Secret**
3. Fund the wallet address with USDC on Base

The plugin auto-creates a persistent CDP account named `verathos-agent` (or your custom name). Same name = same address across restarts.

### 3. x402 Raw Key Mode (dev/testnet)

Direct private key — fast to set up, suitable for development and testnet.

```bash
export VERATHOS_X402_PRIVATE_KEY="0x..."
export VERATHOS_X402_TESTNET="true"
```

**Security:** Use a dedicated hot wallet funded with only what you need. The key is held in process memory — same risk model as any crypto bot or trading agent.

### How x402 Works

1. Agent sends inference request (no auth header)
2. Verathos responds with HTTP 402 + USDC payment requirements
3. Plugin auto-signs a USDC payment authorization (CDP TEE or local key)
4. Retries the request with the signed payment header
5. Verathos settles the USDC on-chain via Coinbase x402 facilitator
6. Inference result returned

Fully automatic — the agent just calls `runtime.useModel()` and gets a response.

**Scheme:** x402 [`upto`](https://github.com/coinbase/x402/blob/main/specs/schemes/upto/scheme_upto.md) — the agent signs an authorisation for a session cap and the gateway settles for the actual cost after inference. Agents only pay for what they consume. For high-frequency callers, reuse the same `X-PAYMENT` header across requests within the signature's 10-minute deadline to aggregate consumption into a single on-chain settlement.

## Supported Model Types

| elizaOS Model Type | Supported | Notes |
|-------------------|-----------|-------|
| `TEXT_SMALL` | Yes | Fast inference (e.g., Qwen3-8B) |
| `TEXT_LARGE` | Yes | High-quality inference (e.g., Llama-3.3-70B) |
| `OBJECT_SMALL` | Yes | Structured JSON output (small model) |
| `OBJECT_LARGE` | Yes | Structured JSON output (large model) |
| `TEXT_TOKENIZER_ENCODE` | Yes | cl100k_base tokenizer |
| `TEXT_TOKENIZER_DECODE` | Yes | cl100k_base tokenizer |
| `TEXT_EMBEDDING` | Planned | |
| `IMAGE` | No | Verathos is text-only |

## How It Works

```
elizaOS Agent
    |
    v
plugin-verathos
    |  (OpenAI-compatible API)
    |  (API key / x402 CDP / x402 raw key)
    v
Verathos API (api.verathos.ai)
    |  (score-weighted routing)
    v
Bittensor Miners (GPU inference + ZK proofs)
```

1. Your agent calls `runtime.useModel(ModelType.TEXT_LARGE, { prompt: "..." })`
2. The plugin routes the request to the Verathos API
3. Verathos selects the best miner based on scores (throughput, latency, proof success rate)
4. The miner runs inference and generates a cryptographic proof
5. The validator verifies the proof and returns the response
6. Your agent gets verified, tamper-proof inference

## Why Verathos?

- **Cryptographic verification** — ZK proofs guarantee the response came from the claimed model
- **Decentralized** — Bittensor network, no single point of failure
- **Competitive pricing** — miners compete on price/performance
- **x402 native** — agents pay per-request with USDC, no accounts needed
- **MPC wallet support** — production-grade key security via Coinbase CDP
- **OpenAI-compatible** — drop-in replacement, same API format

## Optional Dependencies

The core plugin works with just an API key. For x402 payment modes, install:

```bash
# x402 SDK (recommended for correct payment signing)
bun add x402

# Coinbase CDP MPC wallet (production x402)
bun add @coinbase/cdp-sdk
```

## Development

```bash
bun install
bun run build
bun run dev      # watch mode
bun run test     # requires valid credentials
```

## License

MIT
