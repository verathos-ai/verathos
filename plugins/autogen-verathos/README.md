# Verathos + Microsoft AutoGen

Use [Verathos](https://verathos.ai) verified LLM inference with [AutoGen](https://github.com/microsoft/autogen) — no plugin needed.

AutoGen's `OpenAIChatCompletionClient` accepts a custom `base_url`, so Verathos works out of the box.

## Setup

```bash
pip install autogen-agentchat autogen-ext[openai]
```

## Usage

```python
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

# Point AutoGen at Verathos
model_client = OpenAIChatCompletionClient(
    model="auto",  # best available model by score
    base_url="https://api.verathos.ai/v1",
    api_key="your-api-key",
)

agent = AssistantAgent(
    name="verathos_agent",
    model_client=model_client,
    system_message="You are a helpful assistant.",
)

# Single turn
response = await agent.run(task="What is quantum computing?")
print(response.messages[-1].content)
```

## Multi-agent example

```python
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_ext.models.openai import OpenAIChatCompletionClient

client = OpenAIChatCompletionClient(
    model="auto",
    base_url="https://api.verathos.ai/v1",
    api_key="your-api-key",
)

researcher = AssistantAgent("researcher", model_client=client,
    system_message="You research topics thoroughly.")
writer = AssistantAgent("writer", model_client=client,
    system_message="You write concise summaries. Say DONE when finished.")

team = RoundRobinGroupChat(
    [researcher, writer],
    termination_condition=TextMentionTermination("DONE"),
)

result = await team.run(task="Research and summarize ZK proofs in 3 sentences.")
```

## Model selection

Use `"auto"` to let Verathos pick the best available model, or specify one:

```python
# Auto — best available (recommended)
model_client = OpenAIChatCompletionClient(model="auto", ...)

# Specific model
model_client = OpenAIChatCompletionClient(model="qwen3.5-9b", ...)

# Specific model + quantization
model_client = OpenAIChatCompletionClient(model="qwen3-8b:int4", ...)
```

## What you get

Every response is backed by cryptographic proofs (ZK sumcheck + Merkle commitments). Verified inference for multi-agent systems that need trustworthy outputs.

## Getting an API key

1. Go to [verathos.ai](https://verathos.ai) and create an account
2. Deposit TAO or USDC
3. Generate an API key
