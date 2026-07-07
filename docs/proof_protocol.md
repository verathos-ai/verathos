# Proof Protocol Whitepaper

Protocol version: Verifiable inference proof line, July 6, 2026.

This document specifies the proof protocol used by Verathos validators to verify
LLM inference from remote miners without downloading the model or running a GPU.
It is written as a citable technical description of the production proof line.
Operational scoring and hot-capacity audits are separate systems that build on
this verification layer.

## Abstract

Verathos uses a non-interactive probabilistic proof protocol for neural
inference. A miner serves an LLM response, commits to the prompt, generated token
IDs, model activations, and decode metadata, then derives verifier challenges
with Fiat-Shamir from a validator nonce and the miner commitment. The proof
opens a random subset of transformer layers, GEMM operations, matrix blocks,
input embedding rows, decode positions, and MoE router decisions. Each opened
operation is checked against on-chain model commitments with Merkle paths and
sumcheck proofs over an integer field.

The validator verifies the proof on CPU. It does not need model weights, a GPU,
or private state. A failed proof makes the receipt invalid and enters the
subnet's scoring and probation logic.

## Goals

The protocol is designed for a permissionless inference subnet with adversarial
miners and lightweight validators.

It provides:

- Weight binding: the miner must use the model weights registered on-chain.
- Prompt binding: the proof is tied to the validator's request.
- Decode binding: the returned token IDs are committed into the transcript.
- Compute binding: challenged linear operations are proven as matrix products.
- Sampling binding: generation parameters and sampled decode positions are
  bound to the commitment.
- Lightweight verification: validators verify with Merkle roots and CPU work.
- Cumulative security: repeated independent requests drive detection
  probability toward one.

It does not try to prove every floating-point operation of every token in every
request. Instead, it combines cryptographic spot verification with economic
penalties and validator-controlled canaries. This keeps serving practical on
commodity GPUs while making cheaper-than-honest serving unprofitable.

## Actors

**Subnet owner.** Registers supported models and their verification metadata.

**Miner.** Runs the model, serves inference, builds commitments, and produces
proof bundles.

**Validator.** Sends requests and nonces, verifies proof bundles, signs
receipts, scores miners, and applies probation/zero-score policy.

**User or proxy.** Sends OpenAI-compatible chat requests through a validator or
gateway. The user-facing API does not need to know proof internals.

## Public Model Commitments

Each supported model has a public `ModelSpec` anchored on-chain. The exact
schema can evolve, but the verification trust anchor contains these fields:

- `model_id`, architecture, layer count, dimensions, vocabulary size, context
  length, and quantization mode.
- Overall model weight commitment.
- Per-layer or hierarchical layer weight roots.
- Flat chunk Merkle roots for lightweight per-weight openings.
- Embedding weight root for input-token binding.
- `lm_head` weight root for decode/sampling verification.
- Tokenizer and chat-template hash.
- MoE router and expert roots for mixture-of-experts models.
- Quantization/proof metadata needed to map served weights into the proof
  integer domain.

The subnet owner computes these roots from the canonical artifacts. Miners
cannot choose their own roots for a registered model. Validators read the roots
from chain and use them as the verification anchor.

## Proof Line

At a high level, every verified inference follows this line:

```text
request + validator_nonce
  -> miner inference
  -> inference commitment
  -> beacon = H(commitment_hash, validator_nonce)
  -> Fiat-Shamir challenges
  -> proof bundle
  -> validator verification
  -> signed receipt
  -> scoring / routing / probation
```

There is no challenge-response round trip after inference. The validator sends a
fresh nonce with the request. The miner cannot know the final challenge set until
after it has committed to the inference trace.

## Request Binding

For each request the validator sends:

- prompt or canonical chat messages
- generation parameters
- sampling verification rate
- requested maximum output length
- a fresh 32-byte `validator_nonce`

The miner canonicalizes the prompt with the model's tokenizer/chat template and
computes:

$$
\mathrm{prompt\_hash}
= \operatorname{SHA256}(\mathrm{canonical\_prompt\_or\_message\_json})
$$

The miner also commits generation parameters:

$$
\begin{aligned}
\mathrm{sampler\_config\_hash}
= \operatorname{SHA256}(&\text{"SAMPLER\_CONFIG\_V2"} \,\Vert\,
\mathrm{top\_k} \,\Vert\, \mathrm{top\_p} \,\Vert\, \mathrm{min\_p} \,\Vert\\
&\mathrm{presence\_penalty})
\end{aligned}
$$

The validator recomputes the expected hashes from the request it sent. A mismatch
rejects the proof.

## Inference Commitment

During inference the miner records enough data to bind the served response to an
execution trace:

- session and model identity
- model weight commitment
- input commitment
- output token commitment
- per-layer activation commitments
- embedding output commitment
- layer transition hashes
- decode hidden-row root
- optional decode logits-row root for high-assurance sampling checks
- output token count
- `do_sample`, temperature, and sampling verification rate
- prompt hash
- sampler config hash
- optional sampling seed commitment
- optional MoE router commitment hash

The generated token IDs are committed as:

$$
\mathrm{output\_commitment}
= \operatorname{SHA256}(\operatorname{int64\_le}(\mathrm{output\_token\_ids}))
$$

The full inference commitment serializes these fields with domain separators.
The commitment hash is:

$$
C = \operatorname{SHA256}(\operatorname{serialize}(\mathrm{InferenceCommitment}))
$$

The validator later checks that the proof bundle's `output_token_ids` match
`output_commitment`, so the miner cannot prove one output and return another.

## Beacon

The beacon is derived from the miner commitment and the validator nonce:

$$
\mathrm{beacon}
= \operatorname{SHA256}(\text{"VERILLM\_BEACON\_V2"} \,\Vert\, C \,\Vert\,
\mathrm{validator\_nonce})
$$

Security depends on two facts:

1. The validator nonce is fixed before inference and unpredictable to the miner.
2. The commitment depends on the actual inference trace.

The miner cannot choose a favorable challenge set unless it can also construct a
valid commitment and proofs for the chosen output.

## Challenge Derivation

All challenges are deterministic functions of the beacon and commitment. The
validator recomputes them independently.

For dense transformer layers:

$$
\begin{aligned}
\mathrm{challenge\_seed}
= \operatorname{SHA256}(&\text{"VERILLM\_CHALLENGE\_V1"} \,\Vert\,
\mathrm{beacon} \,\Vert\, \mathrm{model\_commitment} \,\Vert\\
&\mathrm{input\_commitment} \,\Vert\, \mathrm{output\_commitment} \,\Vert\,
\mathrm{layer\_commitments})
\end{aligned}
$$

The challenge samples:

- `k_layers` unique transformer layers.
- `k_gemms_per_layer` GEMM operations inside each challenged layer.
- `k_blocks_per_gemm` matrix-output blocks inside each challenged GEMM.
- Fiat-Shamir-derived spot positions inside each opened block.

Current implementation defaults use an auto layer rate targeting roughly 6.25%
per-request layer coverage, with at least one layer and a cap at half the model.
For a 64-layer dense model this yields 4 challenged layers. The proof line is
parameterized; validators can increase challenge rates for canaries or future
policy.

## GEMM Proof

The core arithmetic proof verifies a challenged matrix multiplication:

$$
Y = XW
$$

where:

- `X` is a committed activation block.
- `W` is a committed weight block rooted in the on-chain `ModelSpec`.
- `Y` is the committed output block for that operation.

The proof operates over the Mersenne prime field:

$$
p = 2^{61} - 1
$$

Floating or quantized serving formats are mapped into an integer proof domain.
INT8/INT4/FP8/MXFP4 paths all resolve to deterministic integer representations
for proof generation, avoiding floating-point nondeterminism in verification.

For a challenged output block, write:

$$
X_{\mathrm{block}} \in \mathbb{F}^{m \times K}, \qquad
W_{\mathrm{block}} \in \mathbb{F}^{K \times n}, \qquad
Y_{\mathrm{block}} \in \mathbb{F}^{m \times n}.
$$

where `m` and `n` are the output block dimensions and `K` is the inner model
dimension. The tensors are padded to powers of two. Let:

$$
\ell_m = \lceil \log_2 m \rceil,\qquad
\ell_n = \lceil \log_2 n \rceil,\qquad
\ell_K = \lceil \log_2 K \rceil.
$$

$$
\widetilde{X}: \mathbb{F}^{\ell_m + \ell_K} \to \mathbb{F},\qquad
\widetilde{W}: \mathbb{F}^{\ell_K + \ell_n} \to \mathbb{F},\qquad
\widetilde{Y}: \mathbb{F}^{\ell_m + \ell_n} \to \mathbb{F}
$$

be the multilinear extensions of the padded tables. The verifier derives random
row/column points `r_i` and `r_j` from the transcript and checks:

$$
\widetilde{Y}(r_i, r_j)
= \sum_{z \in \{0,1\}^{\ell_K}}
\widetilde{X}(r_i, z)\,\widetilde{W}(z, r_j)
$$

The right side is a Boolean-hypercube sum of the polynomial:

$$
g(z) = \widetilde{X}(r_i, z)\,\widetilde{W}(z, r_j)
$$

`g` has degree at most 2 in each variable because it is the product of two
multilinear polynomials. This is exactly the setting where the sumcheck
protocol is efficient: the verifier checks a length-`K` inner product with only
`logK` rounds and constant-size work per round.

For each challenged block the miner sends:

- the output block Merkle path under the `Y` root
- a sumcheck transcript proving the multilinear extension identity for `X * W`
- the prover's final `A` and `B` field evaluations
- Merkle-opened spot checks for `X`
- Merkle-opened spot checks for `W`

The verifier checks:

1. The output block leaf belongs to the committed `Y` root.
2. The sumcheck transcript is valid under the verifier's Fiat-Shamir transcript.
3. The final claim equals `final_A * final_B` in the proof field.
4. Opened `X` values belong to the per-request activation commitment.
5. Opened `W` values belong to the on-chain weight Merkle root.

In lightweight mode, validators never hold the full model. Weight checks are
done only through Merkle openings against on-chain roots.

## Sumcheck Protocol

The sumcheck subprotocol proves a claim of the form:

$$
S = \sum_{z \in \{0,1\}^{\ell}} g(z)
$$

where $\ell = \lceil \log_2 K \rceil$. In the GEMM proof, $S$ is the
multilinear evaluation $\widetilde{Y}(r_i, r_j)$ and
$g(z) = \widetilde{X}(r_i,z)\,\widetilde{W}(z,r_j)$.

At round `t`, the prover sends three field elements:

$$
h_t(0),\quad h_t(1),\quad h_t(2)
$$

These define a degree-2 univariate polynomial:

$$
h_t(u)
= \sum_{z_{t+1},\ldots,z_{\ell} \in \{0,1\}}
g(r_1,\ldots,r_{t-1},u,z_{t+1},\ldots,z_{\ell})
$$

The verifier checks:

$$
h_t(0) + h_t(1) = \mathrm{current\_claim}
$$

Then it derives the next challenge:

$$
r_t =
H(\mathrm{transcript} \,\Vert\, h_t(0) \,\Vert\, h_t(1) \,\Vert\, h_t(2))
$$

and updates:

$$
\mathrm{current\_claim} = h_t(r_t)
$$

where `h_t(r_t)` is reconstructed from `h_t(0)`, `h_t(1)`, and `h_t(2)` by
degree-2 interpolation. After $\ell$ rounds the verifier checks the final reduced
claim:

$$
\mathrm{current\_claim} = \mathrm{final\_A} \cdot \mathrm{final\_B}
$$

where `final_A` and `final_B` are the prover's final folded evaluations of the
$\widetilde{X}(r_i,z)$ and $\widetilde{W}(z,r_j)$ tables at the Fiat-Shamir
point $(r_1,\ldots,r_{\ell})$.

### Soundness

If the prover's polynomial is wrong, the equality checked in some round becomes
a non-zero low-degree polynomial identity. By the Schwartz-Zippel lemma, a
non-zero degree-$d$ polynomial over $\mathbb{F}_p$ evaluates to zero at a
random point with probability at most $d/p$.

Here each sumcheck round uses degree at most 2, so the arithmetic soundness error
for one $\ell$-round GEMM block is bounded by approximately:

$$
\varepsilon_{\mathrm{sumcheck}} \le \frac{2\ell}{p}
$$

With $p = 2^{61} - 1$, even $\ell = 16$ gives:

$$
\varepsilon_{\mathrm{sumcheck}}
\le \frac{32}{2^{61}-1}
\approx 1.4 \times 10^{-17}
$$

This is the soundness of the arithmetic reduction once the challenged layer,
GEMM, and block have been selected. The separate probabilistic part is the
sampling of which layers, GEMMs, blocks, decode positions, and spot openings are
audited for a request.

The Merkle openings are what bind the algebraic tables to the committed
inference trace:

- `Y` block openings bind the claimed output block to the `Y` root.
- `X` spot openings bind activation values to the per-request activation root.
- `W` spot openings bind weight values to the on-chain model root.

So the proof is not only "there exists some matrix product"; it is a matrix
product tied to the committed request trace and registered model weights at
unpredictable positions chosen after the commitment is fixed.

## Activation and Layer Binding

Each transformer layer emits an activation commitment. Challenged GEMM proofs
open `X` values under the layer's activation root. Layer transition hashes bind
neighboring layer commitments and the embedding output root into the inference
commitment.

This prevents a miner from presenting unrelated witness tensors for the
challenged operation: the opened activation values must match the same
commitment that determines the beacon and challenge set.

## Input Binding

Prompt substitution is prevented by three linked checks:

1. The validator recomputes `prompt_hash`.
2. The miner opens sampled input token positions against the embedding weight
   root.
3. The embedding output root is tied into the layer transition hash chain.

The embedding challenge is derived as:

$$
\operatorname{SHA256}(\text{"VERILLM\_EMBEDDING\_CHALLENGE\_V1"}
\,\Vert\, \mathrm{beacon}
\,\Vert\, \mathrm{input\_commitment}
\,\Vert\, \mathrm{num\_input\_tokens})
$$

Production validators sample up to five input token positions per proof. This
binds the prompt/token IDs to the first model activation without requiring the
validator to run the embedding layer for the full prompt.

## Decode and Sampling Binding

The proof commits to generated token IDs, output length, decoding mode, and
sampling parameters. The validator checks:

- `output_token_ids` hash to `output_commitment`
- `output_token_count` matches the number of committed token IDs
- `do_sample` and temperature match validator expectations when required
- `sampler_config_hash` matches requested generation parameters

For decode-integrity verification, a Fiat-Shamir gate decides whether the
request includes a sampling challenge:

$$
\operatorname{SHA256}(\text{"VERILLM\_SAMPLING\_GATE\_V1"} \,\Vert\,
\mathrm{beacon})
$$

If active, the challenge samples output positions from:

$$
\operatorname{SHA256}(\text{"VERILLM\_SAMPLING\_CHALLENGE\_V1"} \,\Vert\,
\mathrm{beacon} \,\Vert\, \mathrm{decode\_hidden\_row\_root} \,\Vert\,
\mathrm{output\_commitment} \,\Vert\, \mathrm{vocab\_size})
$$

At each challenged decode position the miner opens:

- hidden row under `decode_hidden_row_root`
- optional fp16 logits row under `decode_logits_row_root`
- `lm_head` weight root
- `lm_head` GEMM proof for the opened row
- committed token ID for that decode step

For greedy decoding (`temperature=0`), the verifier checks that the committed
token is the argmax of the proven logits when strict mode applies. For sampled
decoding, the miner can commit a sampling seed before inference; the validator
replays the canonical sampler on opened logits and checks the chosen token.

Decode checks are sampled because proving every `lm_head` row for every output
token would dominate inference cost. Validator canaries can raise the sampling
rate when higher assurance is needed.

## MoE Proofs

Mixture-of-experts models add router and expert commitments.

For each challenged MoE layer, the challenge samples token positions and expert
openings. The proof verifies:

- router commitment hash matches the inference commitment
- selected experts match opened router rows
- expert roots hash into the registered layer root
- selected expert GEMMs prove their gate/up/down projections
- shared experts, when present, are challenged independently

The validator does not need all expert weights. It verifies expert weight spots
through Merkle openings rooted in the `ModelSpec`.

## Verification Algorithm

Validator verification is deterministic:

```text
verify(request, response, proof_bundle):
  load ModelSpec(model_id) from chain
  recompute prompt_hash and sampler_config_hash
  check output_token_ids hash to output_commitment
  C = SHA256(serialize(commitment))
  beacon = SHA256("VERILLM_BEACON_V2" || C || validator_nonce)
  derive dense or MoE challenges
  verify embedding proof if active
  for each challenged layer:
      for each challenged GEMM:
          verify Y Merkle opening
          verify sumcheck transcript
          verify final field claim
          verify X Merkle spots
          verify W Merkle spots against on-chain root
  verify MoE router/expert openings if active
  verify decode/sampling openings if active
  accept only if every check passes
```

An accepted proof produces a signed validator receipt containing the commitment
hash, proof result, locally measured timing metrics, and token counts. A failed
proof is not counted as valid work.

## Detection Probability

For a miner cheating in a specific transformer layer, detection probability per
request is dominated by whether that layer is challenged:

$$
P_{\mathrm{detect}}^{(1)}
= \frac{k_{\mathrm{layers}}}{N_{\mathrm{layers}}}
$$

After $M$ independent proof requests:

$$
P_{\mathrm{detect}}^{(M)}
= 1 - \left(1 - \frac{k_{\mathrm{layers}}}{N_{\mathrm{layers}}}\right)^M
$$

Examples:

| Layers | Challenged | One request | 36 requests | 72 requests |
|---:|---:|---:|---:|---:|
| 32 | 2 | 6.25% | 90.2% | 99.0% |
| 64 | 4 | 6.25% | 90.2% | 99.0% |
| 80 | 5 | 6.25% | 90.2% | 99.0% |

If the miner serves the wrong model, many or all layer weights differ, so any
challenged layer reveals the mismatch. If the miner corrupts only a small part
of a single layer, detection is still cumulative across independent requests.

## Security Assumptions

The protocol assumes:

- SHA-256 is collision resistant.
- Merkle paths bind leaves to published roots.
- Fiat-Shamir challenges are unpredictable before the commitment is fixed.
- The sumcheck protocol is sound over the selected field.
- Validators generate fresh nonces and verify against the correct on-chain
  `ModelSpec`.
- The canonical tokenizer/chat-template hash used by validators matches the
  registered model metadata.

The protocol is probabilistic, not a full SNARK of every operation. Its security
comes from unpredictable challenge sampling plus economic penalties for failed
proofs.

## Operational Controls

The proof protocol is one layer of the subnet's integrity system. Validators
also use:

- canary prompts with forced proof requests
- full-context canaries
- decoded-output sanity checks
- receipt-level timing and throughput checks measured by the validator
- probation and score-zeroing after failures
- hot-capacity audits for real hardware capacity under load

These controls cover behavior that is not purely an arithmetic proof question:
serving availability, response completeness, malformed output, shared-capacity
endpoints, or routing quality.

## What Is Publicly Verifiable

Given a proof bundle, validator nonce, response token IDs, and chain `ModelSpec`,
a third party can verify:

- the beacon was derived correctly
- the challenge set was derived correctly
- opened weights match registered roots
- opened activations match the miner's commitment
- challenged matrix multiplications are valid
- output token IDs match the commitment
- sampled decode checks match the committed tokens

They cannot infer private user prompts beyond what is present in the request,
and they do not need private validator state.

## Summary

The Verathos proof protocol binds each served response to registered model
weights, the validator's prompt, generated token IDs, sampled decode positions,
and challenged neural-network operations. The validator verifies all opened
claims against on-chain commitments without running the model. Because
challenges are unpredictable and repeated across validator traffic, dishonest
serving is detected with rapidly increasing probability while honest serving
keeps overhead low enough for production inference.

## See Also

- [Inference Protocol](inference_protocol.md) - product-level overview of
  verified inference
- [Bittensor Integration](bittensor_integration.md) - scoring, epochs, and
  validator operations
- [API Reference](api.md) - OpenAI-compatible HTTP API
