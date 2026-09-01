<p align="center">
  <strong>Verathos</strong><br>
  The verifiable serving layer for Bittensor intelligence
</p>

<p align="center">
  <a href="https://verathos.ai">Website</a> &middot;
  <a href="https://verathos.ai/docs">Docs</a> &middot;
  <a href="https://verathos.ai/chat">Chat</a> &middot;
  <a href="https://verathos.ai/docs?page=setup">Operator setup</a>
</p>

---

Verathos is Bittensor Subnet 96. It turns models into open, verifiable inference
services.

A state-of-the-art open model, a model trained by another Bittensor subnet, or
a fine-tuned community model should not remain tied to the machine, team, or
platform that created it. Verathos separates **model creation** from **model
serving**: model builders publish an exact model identity, independent
operators serve it, applications reach it through one API, and validators
verify the work before rewards are distributed.

The verification system is what makes that portability possible. Operators can
serve a model through any hardware and runtime lane for which a qualified
verification profile exists and network policy enables participation. The
application does not need a different integration for each model, operator, or
hardware architecture.

The result is a shared serving layer where Bittensor intelligence can move from
training and fine-tuning into real usage, while operators compete on useful
models, speed, capacity, reliability, and price.

## Why Verathos

- **Model portability** — frontier open models, subnet-trained models, and
  fine-tuned derivatives can enter one serving network under an exact identity.
- **Hardware reach** — qualified verification profiles let the same model be
  served across different operator hardware and runtime designs.
- **Bittensor-native distribution** — subnets and model builders can turn
  produced intelligence into a service without operating one centralized
  inference fleet.
- **Open supply** — independent operators contribute compute instead of serving
  through one centralized platform.
- **One developer surface** — applications use an OpenAI-compatible API while
  the network handles discovery and routing.
- **Verifiable service** — responses carry proof metadata and operators face
  unpredictable audits of execution and available capacity.
- **Performance-based rewards** — eligible endpoints earn through measured
  utility, latency, throughput, reliability, and verified operation.
- **Model choice** — the network can support multiple registered models and
  serving architectures without forcing applications into separate APIs.
- **Open access** — wallet credit, conviction-based daily inference allowance,
  and x402 pay-per-request serve users, builders, and autonomous agents.

## Who the network is for

### Builders

Use one API to reach a score-ranked pool of independent model providers. Select
a specific qualified model or let the network choose an eligible route.

### Model creators and subnets

Move trained or fine-tuned models into a permissionless serving market. A
registered model identity and its verification profile travel with the model,
while independent operators provide the execution capacity.

### Compute operators

Turn compatible GPU infrastructure into a scored inference service. Verathos
supports conventional vLLM endpoints and coordinated GGUF mesh pools that can
combine multiple CUDA workers behind one service.

### Validators

Measure serving quality, verify network evidence, audit capacity, and translate
endpoint performance into Bittensor weights.

### Wallet holders

Use paid inference credit or qualify for a daily inference allowance through an
eligible SN96 conviction lock. Accountless clients can pay per request through
x402.

## How value moves through Verathos

1. A model creator publishes a qualified model identity and verification
   profile.
2. Operators deploy that model on eligible hardware and register endpoints.
3. Applications send standard chat-completion requests through one API.
4. The network routes traffic using current health and score evidence.
5. Operators return inference together with verification metadata.
6. Validators test performance and run unpredictable integrity and capacity
   audits.
7. Bittensor rewards flow toward models and operators that deliver useful,
   available, and verified compute.

The active network policy determines which registered models and runtime
families are currently eligible for emissions. The
[Gleipnir protocol guide](docs/proof_protocol.md) and
[Economic Model](docs/economic_model.md) describe the verification and scoring
mechanisms in detail.

## Build with Verathos

Verathos exposes an OpenAI-compatible API:

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

print(response.choices[0].message.content)
```

Start with the [Quickstart](docs/quickstart.md), then see the
[API Reference](docs/api.md) and [Integrations](docs/integrations.md).

## Contribute compute

### vLLM miner

The guided installer prepares the supported inference and proof environment,
then walks through wallet, endpoint, and process setup:

```bash
curl -fsSL https://verathos.ai/install.sh | bash
verathos setup
verathos start
```

### GGUF mesh pool

Start a coordinator:

```bash
curl -fsSL https://verathos.ai/install.sh \
  | bash -s -- --mesh-coordinator
```

Join each CUDA worker with the token printed by the coordinator:

```bash
curl -fsSL https://verathos.ai/install.sh \
  | bash -s -- --mesh-worker --token vtpool_...
```

The [Mesh Quickstart](docs/mesh_quickstart.md) covers pool formation,
qualification, registration, worker lifecycle, and private API access.

### Validator

```bash
curl -fsSL https://verathos.ai/install.sh \
  | bash -s -- --validator
verathos setup validator
verathos start validator
```

The [Setup Guide](docs/setup.md) covers hardware, HTTPS, wallets, model
selection, auto-update, multi-endpoint operation, and validator roles.

## Repository

```text
verallm/        Inference, proof, chain, registry, and mesh runtimes
neurons/        Miner, validator, routing, scoring, and receipt services
contracts/      Bittensor EVM registry and account contracts
scripts/        Installers, setup tools, and operational utilities
patches/        Mesh-runtime integration patches
dist/           Shipped proof, verifier, and capacity-audit artifacts
tests/          Public installation and security test subset
docs/           User, operator, API, protocol, and economic documentation
plugins/        Framework integrations
examples/       OpenAI-compatible client examples
```

## Documentation

- [What is Verathos?](docs/intro.md)
- [Quickstart](docs/quickstart.md)
- [User Guide](docs/user_guide.md)
- [Setup Guide](docs/setup.md)
- [Inference verification](docs/inference_protocol.md)
- [Gleipnir proof protocol v3](docs/proof_protocol.md)
- [Mesh Quickstart](docs/mesh_quickstart.md)
- [Mesh Operator Flow](docs/mesh_operator_flow.md)
- [Bittensor integration](docs/bittensor_integration.md)
- [Economic model](docs/economic_model.md)
- [API Reference](docs/api.md)

## License

MIT
