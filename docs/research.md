# Active Research

Verathos is not just an inference network. It is a long-term research effort toward building a decentralized cognitive architecture where a modular network of specialized models collectively outperforms any single frontier model. Below is a summary of active research directions beyond the current production system.

## Intelligent Routing

**Status: Implementation in progress**

The current system routes requests to miners based on throughput and model utility scores. The next step is intelligent routing: a learned router that selects the best model and miner for each query based on content, complexity, and domain.

### Why this matters

Recent research demonstrates that networks of open models, when properly routed, match or exceed frontier monolithic models:

- **Mixture-of-Agents** (Wang et al., 2024): 6 open models orchestrated in a 3-layer pipeline score 65.1% on AlpacaEval 2.0 LC, beating GPT-4-Omni (57.5%) by +7.6 percentage points. The critical finding: 6 *different* models (61.3%) clearly outperform the *same* model called 6 times (56.7%). Diversity of the pool is the lever, not parallelism alone. ([arXiv:2406.04692](https://arxiv.org/abs/2406.04692))

- **Avengers** (2025): 10 open ~8B models with zero neural training beat GPT-4.1 by +1.95% on aggregate across 15 datasets, with +37% on AIME and +32% on MBPP. The entire method is training-free: cluster queries by embedding similarity, route to the best model per cluster, and vote via self-consistency. ([arXiv:2505.19797](https://arxiv.org/abs/2505.19797))

- **RouteLLM** (Ong et al., 2024): A lightweight router trained on preference data achieves 95% of GPT-4 quality while cutting costs by 85% on MT-Bench, by dynamically selecting between strong and weak models per query. The router generalizes across model pairs without retraining. ([arXiv:2406.18665](https://arxiv.org/abs/2406.18665))

These results validate the core thesis: a decentralized network with diverse specialized miners and intelligent routing can compete with centralized frontier providers, not by building one massive model, but by composing many specialized ones.

### How it works on Verathos

The router operates in three stages, each using a disjoint signal source:

1. **Trusted-metadata filter** (microseconds): on-chain registry data (model architecture, quantization, context length, TEE capability, probation status) filters the candidate pool. No self-reported fields enter this stage.

2. **Per-prompt ranker** (10-50ms): a lightweight classifier evaluates the incoming prompt and ranks expert classes by expected quality for that query type. Approaches range from embedding-based k-NN (matching or beating 7 learned routers with zero training) to small SFT models like Arch-Router-1.5B (93% accuracy, 51ms, new model classes added via natural-language description with no retraining).

3. **Score-weighted resolution** (microseconds): within the selected expert class, the existing EMA score system picks the best live miner, incorporating throughput, latency, and proof track record.

This naturally extends to multi-expert composition: decomposing complex queries across multiple specialized miners and synthesizing their outputs, as demonstrated by MoA and Avengers.

### Implementation path

**Phase 1: Baseline router.** The subnet owner team trains and deploys an initial router model on the validator side, establishing a baseline for routing quality and providing a working template. This router learns from accumulated traffic data which models excel at which query types and provides immediate quality improvements over the current throughput-only routing.

**Phase 2: Incentivized router and expert training.** A second incentive mechanism on the same subnet (alongside the existing inference verification) rewards miners for two contributions: (a) training better routers that improve overall benchmark scores across the network, and (b) training domain-specialized expert models that fill capability gaps identified by the router. Miners compete to produce the routing model and expert adapters that maximize collective network quality on standardized benchmarks.

This creates a self-improving loop that is uniquely possible on Bittensor: the network identifies its own weaknesses (domains where no current expert performs well), incentivizes miners to fill those gaps (train a specialist), and rewards the router improvements that best leverage the new experts. The network adapts and improves itself through incentive alone, without central coordination.

### Autonomous expert discovery

Coevolutionary expert discovery (CycleQD, Sakana AI 2024; AC/DC, ICLR 2026) demonstrates that this self-improvement loop works in practice. CycleQD uses MAP-Elites quality-diversity search over merged Llama-3-8B models: the resulting archive of 8B specialists collectively reaches GPT-3.5-Turbo parity without any explicit benchmark optimization. AC/DC extends this by coevolving the tasks alongside the models, where synthetic data generation creates new evaluation niches, reaching claimed GPT-4o-level coverage.

The pattern maps directly onto Verathos: miners running coevolutionary merging with the network's EMA score as the fitness function, producing increasingly specialized and diverse expert populations autonomously. The router atlas provides the routing layer; the incentive mechanism provides the optimization pressure; Bittensor provides the economic substrate.

## Verified Training

**Status: Implemented, testing on testnet**

The same proof system that verifies inference extends to training. The training prover verifies the forward pass, backward pass (gradient GEMM), and optimizer step. Supported methods include full fine-tuning and LoRA, with AdamW, SGD, and Muon optimizers.

Once live, verified training enables miners to fine-tune LoRA adapters on domain-specific data (code, math, legal, medical, etc.) with cryptographic proofs that the correct base model was used with the claimed data and optimizer. Combined with intelligent routing, this creates domain-specialized miners that the router can dispatch to based on query content. Verified training is the infrastructure that makes the incentivized expert training described above trustless: the network can verify that a miner actually trained the adapter it claims to have trained.

## Network-Level MoE

**Status: Feasibility analysis complete**

Standard MoE models (e.g., Qwen3-30B-A3B with 128 experts) run entirely on a single miner. An alternative approach distributes the MoE architecture across the network itself: backbone nodes hold attention layers and the router, while individual miners each hold a subset of expert FFN blocks.

This enables the network to collectively serve models far larger than any single node can hold. A cluster of RTX 4090s (24 GB each) could serve a model that would normally require an H100 (80 GB), where each miner holds only a few experts (~2 GB) while the backbone handles shared computation.

The primary challenge is latency: every MoE layer requires a network round-trip to dispatch tokens to expert miners and collect results. This makes the approach viable within datacenters (<1ms RTT) but challenging across the open internet. Active research focuses on latency-hiding techniques, speculative expert prefetching, and asynchronous dispatch patterns.

## Neural Transfer Protocol (NTP)

**Status: Research**

In the initial network architecture, expert-to-expert communication happens through text; the orchestrator passes natural language between miners. This is effective for chaining domain experts but inherently lossy: a vision expert must describe what it sees in words rather than passing its internal representation directly.

The Neural Transfer Protocol replaces text communication between experts with tensor-level communication. Experts exchange activations, embeddings, or latent representations through a learned universal embedding space. Projection adapters translate between different model architectures, enabling a Qwen expert and a Llama expert to exchange internal states without either needing to serialize through language.

The primitives for this exist in current research: cross-model activation transfer (ULD), byte-level distillation, Cross-LoRA, CKA matching, tokenizer alignment (ZeTT, TokAlign), and soft-token coprocessors. What does not yet exist is a *verifiable* protocol for latent handoff, where the receiving model can prove it processed the transferred activations correctly. This is the open research gap that Verathos's sumcheck-based verification is positioned to address.

NTP is the infrastructure layer that enables the capabilities below. Without it, expert composition is limited to text; with it, the network can compose at the representation level, enabling multimodal fusion, distributed attention, and latent-space reasoning across experts.

## Toward a Cognitive Network

**Status: Research**

The long-term vision is a network where intelligence lives in the topology and composition of experts, not in any single model's weights. Several research threads contribute to this:

**Persistent memory.** Dedicated memory experts maintain per-user and per-session state (long-term preferences, episodic memory, working context). The orchestrator queries memory on every request and stores new knowledge after each interaction. Memory is verified like any other inference call.

**Multimodal grounding.** Vision, audio, and sensor experts ground the network's language understanding in non-textual reality. NTP enables these experts to pass representations directly: a vision expert sends embeddings, not text descriptions of what it sees.

**Predictive planning.** World-model experts that predict future states and plan action sequences. The orchestrator uses these predictions to evaluate strategies before committing, enabling multi-step reasoning about consequences.

**Continuous adaptation.** Moving from batch training cycles (train, evaluate, deploy) to continuous online learning with versioned commitment windows. The network evolves its experts in response to real traffic patterns, not just periodic benchmark runs.

**Self-supervised learning from interaction.** Experts learn from the outcomes of orchestration: which expert chains produced good results, which routing decisions led to user satisfaction. This feedback loop drives expert evolution without explicit human-curated training data.

## Purpose-Driven Autonomous Research

**Status: Long-term vision**

The final research direction transforms the network from a reactive tool (answers queries) into a proactive agent (pursues goals). A human-governed goal register (on-chain, transparent, auditable, revocable) defines objectives the network pursues autonomously. A research orchestrator, distinct from the serving orchestrator, runs continuous perceive-plan-execute-evaluate loops independent of user queries.

This is an explicit alignment mechanism: the network has no hidden drives or emergent goals. It pursues only the objectives explicitly approved by validator quorum. Goals are chosen by humanity, not evolved.

---

These research directions build on each other: intelligent routing improves inference quality immediately with the existing model pool, verified training enables domain specialization, NTP enables representation-level composition, grounding and memory enable persistent intelligence, and the goal register provides alignment. Each layer extends the network's capabilities without redesigning the foundation. The same verified inference primitives that run in production today serve as the substrate for everything above.

## References

- Wang et al. (2024). *Together for Multi-Agent Collaboration: Mixture-of-Agents Surpasses GPT-4o.* [arXiv:2406.04692](https://arxiv.org/abs/2406.04692)
- Ong et al. (2024). *RouteLLM: Learning to Route LLMs with Preference Data.* [arXiv:2406.18665](https://arxiv.org/abs/2406.18665)
- Huang et al. (2025). *Avengers: Training-Free Routing for Open Model Composition.* [arXiv:2505.19797](https://arxiv.org/abs/2505.19797)
- Srivatsa et al. (2025). *Arch-Router: Learned Routing for Multi-Model Architectures.* [arXiv:2503.15421](https://arxiv.org/abs/2503.15421)
- Cemri et al. (2025). *Why Do Multi-Agent LLM Systems Fail?* [arXiv:2503.13657](https://arxiv.org/abs/2503.13657)
- Sakana AI (2024). *CycleQD: MAP-Elites for LLM Expert Discovery.* [arXiv:2410.14735](https://arxiv.org/abs/2410.14735)
- Sakana AI (2026). *AC/DC: Discovering Novel LLM Experts via Task-Capability Coevolution.* ICLR 2026.
