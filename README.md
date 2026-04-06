<p align="center">
  <strong>Verathos</strong><br>
  Cryptographically verified AI compute on <a href="https://bittensor.com">Bittensor</a>
</p>

<p align="center">
  <a href="https://verathos.ai">Website</a> &middot;
  <a href="https://verathos.ai/docs">Docs</a> &middot;
  <a href="https://verathos.ai/chat">Try It</a> &middot;
  <a href="https://verathos.ai/docs?page=setup">Setup Guide</a>
</p>

---

Verathos is a decentralized compute network on Bittensor (Subnet 96) where any tensor operation – in inference or training – can be cryptographically proven via ZK-inspired **sumcheck-based verification** over Merkle-committed weights, anchored on-chain. Validators verify proofs on CPU in milliseconds and set weights accordingly. The result is a permissionless network where compute is verifiable, not trusted.

### Verified Inference – Live

A proof plugin integrates directly into production [vLLM](https://github.com/vllm-project/vllm) serving. It generates sumcheck proofs for GEMM operations in parallel during CUDA graph execution – no challenge-response round trip, single-digit percent overhead. The network exposes an OpenAI-compatible API with score-weighted routing across all miners.

### Verified Training – In Development

The same proof system extends to training. The training prover verifies forward pass, backward pass (gradient GEMM), and optimizer step for full fine-tuning and LoRA (AdamW, SGD, Muon). A training job produces proofs that the correct base model was fine-tuned with the claimed data and optimizer. The protocol is implemented and tested but not yet active on the network.

## What Gets Proven

| Guarantee | How |
|-----------|-----|
| **Correct weights** | Merkle root of quantized weights committed on-chain. Proof checks layer outputs against committed weights. Wrong model = caught. |
| **Correct computation** | Sumcheck protocol: prover and verifier agree on GEMM results via Fiat-Shamir transform. Covers inference forward pass and training backward pass. |
| **Output integrity** | SHA-256 commitment over full output, bound to the proof via Fiat-Shamir. Tampering invalidates it. |
| **Probabilistic coverage** | k random layers challenged per request. Detection approaches 100% over multiple queries. |

## Architecture

```
                                  Bittensor EVM
                             ┌─────────────────────┐
                             │  ModelRegistry      │  model specs + Merkle roots
                             │  MinerRegistry      │  endpoints, heartbeats
                             │  PaymentGateway     │  deposits, staking
                             └──────────┬──────────┘
                                        │
        ┌───────────────────────────────┼───────────────────────────────┐
        │                               │                               │
   ┌────▼─────┐                   ┌─────▼──────┐                 ┌──────▼─────┐
   │  Miner   │ ◄── canary ────   │ Validator  │                 │  Gateway   │
   │  (GPU)   │  ── receipt ──►   │  (CPU)     │──shared_state──►│  (API)     │
   │          │                   │            │                 │            │
   │ vLLM +   │ ◄── inference ─────────────────────────────────  │ OpenAI-    │
   │ proofs   │  ── response ─────────────────────────────────►  │ compatible │
   └──────────┘                   └────────────┘                 └────────────┘
```

**Miner** – Serves models, generates proofs, registers on Bittensor EVM. Multiple models per hotkey.

**Validator** – Epoch-based canary testing (~72 min cycles), proof verification, scoring (throughput x latency x proof), weight setting. Proof failure = instant score zero.

**Gateway** – OpenAI-compatible API. Score-weighted routing to miners. Payments via TAO, USDC on Base, or [x402](https://www.x402.org/) pay-per-request.

## Getting Started

### As a User

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://api.verathos.ai/v1",
    api_key="vrt_sk_YOUR_KEY",
)

response = client.chat.completions.create(
    model="auto",
    messages=[{"role": "user", "content": "Hello!"}],
)
```

Every response includes cryptographic proof metadata. Works with LiteLLM, LangChain, elizaOS, and any OpenAI-compatible client. See the [Quickstart](docs/quickstart.md) and [Integrations](docs/integrations.md).

### As a Miner

Requires NVIDIA GPU with 24 GB+ VRAM (RTX 4090, A100, H100, etc.).

**Quick start** (guided wizard handles wallet, registration, funding, HTTPS, PM2):
```bash
curl -fsSL https://verathos.ai/install.sh | bash   # or: git clone ... && cd verathos && bash scripts/setup_miner.sh
verathos setup                                       # interactive setup wizard
verathos start                                       # start mining
```

**Manual step-by-step:**
```bash
git clone https://github.com/verathos-ai/verathos && cd verathos
bash scripts/setup_miner.sh          # creates venv, installs deps, builds CUDA ext
```

Then create a wallet, register on subnet, fund your EVM address, and start:

```bash
python -m neurons.miner \
    --wallet miner --hotkey default \
    --model-id auto \
    --netuid 96 \
    --subtensor-network finney \
    --endpoint https://YOUR-PUBLIC-IP-OR-DOMAIN
```

`--model-id auto` detects your GPU and picks the optimal model. See the [Setup Guide](docs/setup.md) for wallet creation, EVM funding, model selection, and production deployment with PM2.

### As a Validator

No GPU required.

**Quick start:**
```bash
curl -fsSL https://verathos.ai/install.sh | bash -s – --validator   # or: git clone ... && bash scripts/setup_validator.sh
verathos setup validator                                              # interactive setup wizard
verathos start validator                                              # start validating
```

**Manual step-by-step:**
```bash
git clone https://github.com/verathos-ai/verathos && cd verathos
bash scripts/setup_validator.sh

python -m neurons.validator \
    --wallet validator --hotkey default \
    --netuid 96 \
    --subtensor-network finney
```

See the [Setup Guide](docs/setup.md) for wallet creation, EVM funding, auto-update, and running the gateway proxy.

## Repository Structure

```
verallm/        Verified inference – vLLM proof plugin, chain integration, model registry
neurons/        Bittensor subnet – miner, validator, gateway, scoring, credits
contracts/      Smart contracts (Foundry/Solidity) – UUPS proxies on Bittensor EVM
plugins/        Framework plugins – LiteLLM, LangChain, elizaOS, OpenClaw
scripts/        Setup scripts – setup_miner.sh, setup_validator.sh
examples/       Client examples – OpenAI, streaming, x402
dist/           Pre-built wheels – zkllm (CUDA kernels)
docs/           Documentation
```

`zkllm` (cryptographic primitives, field arithmetic, Merkle trees, sumcheck, CUDA kernels) is distributed as a pre-built wheel in `dist/`. The setup scripts install it automatically.

## Documentation

- **[What is Verathos?](docs/intro.md)** – Overview, proof guarantees, and architecture
- **[Quickstart](docs/quickstart.md)** – First API call in 2 minutes
- **[Setup Guide](docs/setup.md)** – Hardware requirements, miner and validator setup
- **[User Guide](docs/user_guide.md)** – API keys, deposits, inference, withdrawals
- **[Integrations](docs/integrations.md)** – LiteLLM, LangChain, elizaOS, and more
- **[API Reference](docs/api.md)** – Full HTTP API reference
- **[Inference Protocol](docs/inference_protocol.md)** – Deep dive into sumcheck-based verification
- **[Bittensor Integration](docs/bittensor_integration.md)** – Epoch lifecycle, scoring, contracts
- **[Economic Model](docs/economic_model.md)** – Tokenomics, alpha staking, pricing
- **[Active Research](docs/research.md)** – Future directions

## License

MIT
