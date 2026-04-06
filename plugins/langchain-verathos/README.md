# langchain-verathos

LangChain integration for [Verathos](https://verathos.ai) — verified LLM inference on [Bittensor](https://bittensor.com).

Every inference response from Verathos is backed by cryptographic proofs (ZK sumcheck + Merkle commitments). Your LangChain chains can verify that the declared model was executed faithfully — no output substitution, no bait-and-switch.

## Installation

```bash
pip install langchain-verathos
```

## Quick Start

```python
from langchain_verathos import ChatVerathos

# Set your API key (or export VERATHOS_API_KEY=...)
llm = ChatVerathos(api_key="vrt_sk_...")

# "auto" picks the best available model (this is the default)
llm = ChatVerathos(model="auto")

msg = llm.invoke("Explain zero-knowledge proofs in one paragraph.")
print(msg.content)

# Proof verification metadata is on every response
print(msg.response_metadata["proof_verified"])  # True
print(msg.response_metadata["timing"])          # {"inference_ms": ..., "prove_ms": ...}
print(msg.response_metadata["proof_details"])   # {"challenged_layers": [...], ...}
```

## Why not just `ChatOpenAI(base_url=...)`?

The Verathos API is OpenAI-compatible, so you *can* use plain `ChatOpenAI`:

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="https://api.verathos.ai/v1",
    api_key="vrt_sk_...",
    model="auto",
)
msg = llm.invoke("Hello!")
print(msg.content)  # works fine
```

This works for basic chat, but you lose the proof metadata. The OpenAI SDK parses responses into typed Pydantic models, and while it preserves extra fields internally, `ChatOpenAI` doesn't extract them into `response_metadata`.

`ChatVerathos` adds:

| Feature | `ChatOpenAI` | `ChatVerathos` |
|---------|:---:|:---:|
| Chat completions | Yes | Yes |
| Streaming | Yes | Yes |
| Tool calling | Yes | Yes |
| Structured output | Yes | Yes |
| Proof metadata in `response_metadata` | No | Yes |
| Proof metadata in `additional_kwargs` | No | Yes |
| `VERATHOS_API_KEY` env var | No | Yes |
| `model="auto"` default | No | Yes |
| `list_models()` discovery | No | Yes |
| LangSmith provider tag `verathos` | No | Yes |

## Features

### Automatic model selection

Verathos supports `model="auto"` which automatically selects the best available model from the network. This is the default for `ChatVerathos`.

```python
llm = ChatVerathos()  # model="auto" by default
```

### Proof verification metadata

Every response includes cryptographic proof metadata:

```python
msg = llm.invoke("What is 2+2?")

# Top-level proof status
msg.response_metadata["proof_verified"]  # True or False

# Detailed timing breakdown
msg.response_metadata["timing"]
# {
#     "inference_ms": 1234.5,
#     "prove_ms": 567.8,
#     "verify_ms": 89.0,
#     ...
# }

# Detailed proof information (when include_proof=True, the default)
msg.response_metadata["proof_details"]
# {
#     "challenged_layers": [3, 17, 28, 41],
#     "total_layers": 48,
#     "beacon_valid": True,
#     "detection_prob_single": 0.0816,
#     "detection_prob_10": 0.5765,
#     "detection_prob_100": 1.0,
#     "proof_size_kb": 12.4,
#     "sampling_active": True,
#     "moe_info": {"is_moe": True, ...},
#     ...
# }
```

The same metadata is also available in `msg.additional_kwargs` for convenient programmatic access:

```python
if msg.additional_kwargs.get("proof_verified"):
    print("Response is cryptographically verified!")
```

### Streaming with proof metadata

In streaming mode, proof metadata arrives on the final chunk:

```python
full = None
for chunk in llm.stream("Write a haiku about verification."):
    print(chunk.text, end="", flush=True)
    full = chunk if full is None else full + chunk

print()
# Proof metadata is available after accumulation
print(full.response_metadata.get("proof_verified"))
```

### Model discovery

```python
# List all available models
models = ChatVerathos.list_models()
for m in models:
    print(f"{m['id']:40s}  {m.get('owned_by', '')}")

# Just the model IDs
ids = ChatVerathos.list_model_ids()
# ['Qwen/Qwen3-30B-A3B', 'meta-llama/Llama-3.3-70B-Instruct', ...]
```

### Async support

All async methods work out of the box:

```python
msg = await llm.ainvoke("Hello!")

async for chunk in llm.astream("Hello!"):
    print(chunk.text, end="")
```

### Tool calling and structured output

Since `ChatVerathos` extends `ChatOpenAI`, all tool calling and structured output features work:

```python
from pydantic import BaseModel

class Answer(BaseModel):
    reasoning: str
    answer: int

structured_llm = llm.with_structured_output(Answer)
result = structured_llm.invoke("What is 15 * 23?")
print(result.answer)  # 345
```

### Disabling proof details

To save bandwidth, you can disable the detailed proof metadata (you'll still get `proof_verified` and `timing`):

```python
llm = ChatVerathos(include_proof=False)
```

## Configuration

### Environment variables

| Variable | Description |
|----------|-------------|
| `VERATHOS_API_KEY` | API key for authentication |

### Constructor parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model` | `"auto"` | Model name or `"auto"` for automatic selection |
| `api_key` | `VERATHOS_API_KEY` env | API key |
| `base_url` | `https://api.verathos.ai/v1` | API base URL |
| `include_proof` | `True` | Include detailed proof metadata |
| `temperature` | `None` | Sampling temperature |
| `max_tokens` | `None` | Maximum tokens to generate |
| `streaming` | `False` | Enable streaming by default |

All standard `ChatOpenAI` parameters (timeout, max_retries, etc.) are also supported.

## Getting an API key

1. Visit [verathos.ai](https://verathos.ai) to create an account
2. Deposit TAO or USDC to get credits
3. Generate an API key from the dashboard

Alternatively, Verathos supports [x402](https://www.x402.org/) pay-per-request with USDC on Base — no API key or deposit needed.

## Architecture

Verathos runs on [Bittensor](https://bittensor.com), a decentralized AI network. Miners serve open-weight models with cryptographic proofs. Validators verify proofs and set reputation scores. The Verathos proxy routes your requests to the highest-scoring miners.

```
Your App -> ChatVerathos -> api.verathos.ai -> Validator Proxy -> Miner (GPU)
                                                    |
                                              Proof Verification
                                              (ZK sumcheck + Merkle)
```

Every response proves:
- **Model integrity**: The exact declared model weights were used (Merkle commitment against on-chain root)
- **Computation integrity**: Matrix multiplications were executed correctly (ZK sumcheck protocol)
- **Output binding**: The returned text matches the proven computation (SHA-256 commitment)
- **Input binding**: The prompt you sent is the prompt that was executed (embedding proof)
