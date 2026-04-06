# Integrations

Verathos works with any OpenAI-compatible client. For popular AI agent frameworks, we provide dedicated plugins with deeper integration (model discovery, proof metadata, x402 payments).

## Quick Start

Every framework below can use Verathos with just two settings:

```
base_url: https://api.verathos.ai/v1
api_key:  your-api-key
model:    auto
```

Set `model` to `"auto"` or a specific model ID (e.g. `"qwen3.5-9b"`, `"minimax-2.5"`). Use `GET /v1/models` to list what's available.

With `"auto"`, Verathos pools all nodes across all models and selects using score-weighted routing, factoring in node score, health status, and current load. On failure, retries fall through to the next-best healthy endpoint regardless of model.

## Framework Plugins

| Framework | Plugin | Language | What it adds |
|-----------|--------|----------|-------------|
| [LiteLLM](https://github.com/BerriAI/litellm) | `litellm-verathos` | Python | `verathos/` model prefix, works with CrewAI/Letta/Swarms |
| [LangChain](https://python.langchain.com) | `langchain-verathos` | Python | `ChatVerathos` with proof metadata in `response_metadata` |
| [elizaOS](https://github.com/elizaOS/eliza) | `@elizaos/plugin-verathos` | TypeScript | Full model provider with x402 + CDP wallet support |
| [OpenClaw](https://openclaw.ai) | `openclaw-verathos` | TypeScript | Provider plugin with model discovery |
| [Hermes Agent](https://github.com/NousResearch/hermes-agent) | Config only | - | Just set `base_url` in config |
| [AutoGen](https://github.com/microsoft/autogen) | Config only | - | `OpenAIChatCompletionClient(base_url=...)` |

### LiteLLM

Also works with frameworks that use LiteLLM under the hood (CrewAI, Letta, Swarms).

```python
pip install litellm-verathos
```

```python
import litellm
from litellm_verathos import VerathosProvider

VerathosProvider.register()

response = litellm.completion(
    model="verathos/auto",
    messages=[{"role": "user", "content": "Hello"}],
    api_key="your-key",
)
```

### LangChain

Proof verification metadata on every response.

```python
pip install langchain-verathos
```

```python
from langchain_verathos import ChatVerathos

llm = ChatVerathos(api_key="your-key")
msg = llm.invoke("Explain ZK proofs.")
print(msg.response_metadata["proof_verified"])  # True
```

### elizaOS

For autonomous AI agents with x402 USDC pay-per-request.

```bash
bun add @elizaos/plugin-verathos
```

```json
{
  "plugins": ["@elizaos/plugin-verathos"],
  "settings": { "VERATHOS_API_KEY": "your-key" }
}
```

### OpenClaw

```bash
openclaw plugins install --link /path/to/openclaw-verathos
```

### Hermes Agent

No plugin needed, just config:

```yaml
model:
  provider: custom:verathos
  default: "auto"
  base_url: "https://api.verathos.ai/v1"
  api_key: "your-key"
```

### AutoGen

No plugin needed:

```python
from autogen_ext.models.openai import OpenAIChatCompletionClient

client = OpenAIChatCompletionClient(
    model="auto",
    base_url="https://api.verathos.ai/v1",
    api_key="your-key",
    model_info={"vision": False, "function_calling": False,
                "json_output": False, "family": "unknown",
                "structured_output": False},
)
```

### Any OpenAI-Compatible Client

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://api.verathos.ai/v1",
    api_key="your-key",
)
response = client.chat.completions.create(
    model="auto",
    messages=[{"role": "user", "content": "Hello"}],
)
```

## Authentication

All integrations support three auth modes:

| Mode | How | Best for |
|------|-----|----------|
| API key | `api_key="your-key"` | Most use cases |
| x402 raw key | EVM private key signs USDC payments per-request | Dev/testnet |
| x402 CDP wallet | Coinbase MPC wallet (keys in TEE) | Production agents |

See the [user guide](user_guide.md) for account setup and deposits.
