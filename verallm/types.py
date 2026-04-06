"""
Core types for VeraLLM.

Shared ZK proof types are re-exported from zkllm.types.
Inference- and MoE-specific types are defined here.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import hashlib
import os
import struct
import time
import uuid

# Re-export core ZK proof types from zkllm
from zkllm.types import (  # noqa: F401
    SpotCheck,
    SpotCheckWithProof,
    SumcheckRound,
    SumcheckProof,
    MerklePath,
    GEMMBlockProof,
    GEMMProof,
    VerificationResult,
)


@dataclass
class NonLinearWitness:
    """Witness for non-linear operation (for recomputation)."""

    op_type: str  # "softmax", "gelu", "silu", "rmsnorm", "layernorm"
    input_commitment: bytes
    output_commitment: bytes
    # For challenged verification, prover provides actual tensors
    # These are stored separately in WitnessStore


@dataclass
class LayerProof:
    """Proof for one transformer layer."""

    layer_idx: int
    gemm_proofs: List[GEMMProof]  # Proofs for challenged GEMMs
    nonlinear_witnesses: List[NonLinearWitness]  # Witnesses for non-linear ops


@dataclass
class InferenceCommitment:
    """Commitment to one inference run."""

    session_id: str
    model_id: str
    model_commitment: bytes  # Merkle root of model weights
    input_commitment: bytes  # Hash of input
    output_commitment: bytes  # Hash of output
    layer_commitments: List[bytes]  # Merkle root per layer
    router_commitment_hash: Optional[bytes] = None  # Hash of MoE routing decisions
    # Decode-integrity commitment (temperature=0 path):
    # Merkle root over per-step lm_head hidden rows (fp16 row bytes).
    decode_hidden_row_root: bytes = b""
    # High-assurance: Merkle root over per-step fp16 logits rows.
    decode_logits_row_root: bytes = b""
    # Validator-requested sampling verification rate in basis points [0,10000].
    sampling_verification_bps: int = 0
    # Number of generated output tokens (decode steps).
    output_token_count: int = 0
    # Decoding mode metadata bound into commitment hash.
    do_sample: bool = False
    # Quantized temperature in milli-units (e.g. 0.7 -> 700).
    temperature_milli: int = 0
    # Presence penalty in milli-units (e.g. 1.5 -> 1500).
    # Bound into commitment so verifier knows whether argmax can diverge.
    presence_penalty_milli: int = 0
    # Merkle root of embedding output tensor (binds embedding lookup
    # to actual activations entering layer 0).
    embedding_output_root: bytes = b""
    # Hash chain binding consecutive layer activations.
    # transition_hash[i] = SHA256(layer_i_commitment || layer_{i+1}_commitment || embedding_output_root)
    # Committed before challenges, verified for proven layer boundaries.
    layer_transition_hashes: List[bytes] = field(default_factory=list)
    # SHA256 of the raw prompt messages JSON (before chat template / tokenization).
    # Verified by the proxy on every request to catch prompt substitution.
    prompt_hash: bytes = b""
    # SHA256("SAMPLER_CONFIG_V2" || top_k || top_p || min_p || presence_penalty)
    # — binds sampling parameters so the verifier can detect substitution.
    # V2 adds presence_penalty; previously it was committed separately but
    # omitted from the expected-hash check (one-way binding).
    sampler_config_hash: bytes = b""
    # SHA256(sampling_seed) — committed before challenge, opened on audit
    # for canonical sampler replay when do_sample=True.
    sampling_seed_commitment: bytes = b""
    timestamp: float = field(default_factory=time.time)

    @classmethod
    def empty(cls) -> "InferenceCommitment":
        """Create an empty commitment (TEE mode — no ZK commitment)."""
        return cls(
            session_id="", model_id="", model_commitment=b"",
            input_commitment=b"", output_commitment=b"",
            layer_commitments=[],
        )

    def to_bytes(self) -> bytes:
        """Serialize commitment for hashing."""
        parts = [
            b"VERILLM_COMMITMENT_V1",
            self.session_id.encode(),
            self.model_id.encode(),
            self.model_commitment,
            self.input_commitment,
            self.output_commitment,
        ]
        parts.extend(self.layer_commitments)
        if self.router_commitment_hash is not None:
            parts.append(b"ROUTER_V1")
            parts.append(self.router_commitment_hash)
        parts.append(b"DECODE_V1")
        parts.append(
            struct.pack(
                "<IHH?",
                int(self.output_token_count),
                int(self.sampling_verification_bps),
                int(self.temperature_milli),
                bool(self.do_sample),
            )
        )
        has_hidden_root = bool(self.decode_hidden_row_root)
        parts.append(struct.pack("<?", has_hidden_root))
        if has_hidden_root:
            parts.append(self.decode_hidden_row_root)
        has_logits_root = bool(self.decode_logits_row_root)
        parts.append(struct.pack("<?", has_logits_root))
        if has_logits_root:
            parts.append(self.decode_logits_row_root)
        parts.append(struct.pack("<d", self.timestamp))
        # V2 extension: presence_penalty binding
        if self.presence_penalty_milli != 0:
            parts.append(b"SAMPLING_V2")
            parts.append(struct.pack("<H", int(self.presence_penalty_milli)))
        # V3 extension: embedding output + layer transition binding
        if self.embedding_output_root:
            parts.append(b"EMBEDDING_OUTPUT_V1")
            parts.append(self.embedding_output_root)
        if self.layer_transition_hashes:
            parts.append(b"TRANSITION_V1")
            parts.append(struct.pack("<I", len(self.layer_transition_hashes)))
            for h in self.layer_transition_hashes:
                parts.append(h)
        # V4 extension: prompt hash binding (input integrity on every request)
        if self.prompt_hash:
            parts.append(b"PROMPT_HASH_V1")
            parts.append(self.prompt_hash)
        # V5 extension: sampler config binding (top_k/top_p/min_p)
        if self.sampler_config_hash:
            parts.append(b"SAMPLER_CONFIG_V1")
            parts.append(self.sampler_config_hash)
        # V6 extension: committed sampling seed for do_sample=True replay
        if self.sampling_seed_commitment:
            parts.append(b"SAMPLING_SEED_V1")
            parts.append(self.sampling_seed_commitment)
        return b"".join(parts)

    @staticmethod
    def compute_router_hash(
        router_commitments: "Dict[int, RouterCommitment]",
    ) -> Optional[bytes]:
        """Compute hash of all router commitments for Fiat-Shamir binding."""
        if not router_commitments:
            return None
        h = hashlib.sha256(b"ROUTER_COMMITMENTS_V3")
        for layer_idx in sorted(router_commitments.keys()):
            rc = router_commitments[layer_idx]
            h.update(struct.pack("<Q", layer_idx))
            h.update(rc.router_logits_hash)
            h.update(rc.router_logits_row_root)
            h.update(struct.pack(
                "<III",
                int(rc.seq_len),
                int(rc.num_experts),
                int(rc.top_k),
            ))
            scoring_bytes = (rc.scoring_func or "").encode("utf-8")
            h.update(struct.pack("<I", len(scoring_bytes)))
            h.update(scoring_bytes)

            # Hash selected_experts with explicit lengths to avoid ambiguity.
            h.update(struct.pack("<I", len(rc.selected_experts)))
            for row in rc.selected_experts:
                h.update(struct.pack("<I", len(row)))
                for expert_idx in row:
                    h.update(struct.pack("<i", int(expert_idx)))

            # Hash proof-domain selected experts (quantized X_int @ W_int top-k).
            h.update(struct.pack("<I", len(rc.proof_selected_experts)))
            for row in rc.proof_selected_experts:
                h.update(struct.pack("<I", len(row)))
                for expert_idx in row:
                    h.update(struct.pack("<i", int(expert_idx)))

            # Hash routing_weights (field elements) with explicit lengths.
            h.update(struct.pack("<I", len(rc.routing_weights)))
            for row in rc.routing_weights:
                h.update(struct.pack("<I", len(row)))
                for weight in row:
                    h.update(struct.pack("<Q", int(weight) & ((1 << 64) - 1)))
        return h.digest()

    def commitment_hash(self) -> bytes:
        """Get hash of this commitment."""
        return hashlib.sha256(self.to_bytes()).digest()


@dataclass
class GEMMChallenge:
    """Challenge for one GEMM operation."""

    gemm_idx: int  # Which GEMM in the layer (0-5 typical)
    block_indices: List[Tuple[int, int]]  # (bi, bj) pairs to verify


@dataclass
class LayerChallenge:
    """Challenge for one layer."""

    layer_idx: int
    gemm_challenges: List[GEMMChallenge]


@dataclass
class EmbeddingChallenge:
    """Challenge for embedding input binding verification."""

    token_positions: List[int]  # Positions in input sequence to verify


@dataclass
class SamplingChallenge:
    """Challenge for decode-integrity verification (temperature=0)."""

    decode_positions: List[int]  # output token positions to verify
    lm_head_block_indices: List[Tuple[int, int]]  # GEMM blocks for lm_head proof
    high_assurance: bool = False  # fp16 logits row opening + exact argmax


@dataclass
class ChallengeSet:
    """Complete set of challenges for one inference."""

    beacon: bytes
    layer_challenges: List[LayerChallenge]
    sampling_challenge: Optional[SamplingChallenge] = None
    embedding_challenge: Optional[EmbeddingChallenge] = None

    @property
    def layer_indices(self) -> List[int]:
        """Get list of challenged layer indices."""
        return [lc.layer_idx for lc in self.layer_challenges]


@dataclass
class InferenceProofBundle:
    """Complete proof bundle for challenged inference.

    The validator_nonce is included for auditability: any third party can
    verify that beacon = SHA256(commitment_hash || validator_nonce).
    """

    commitment: InferenceCommitment
    beacon: bytes
    layer_proofs: List[LayerProof]
    validator_nonce: bytes = b""  # Optional: included for auditability
    router_commitments: Dict[int, "RouterCommitment"] = field(default_factory=dict)  # MoE only
    router_layer_proofs: Dict[int, "RouterLayerProof"] = field(default_factory=dict)  # MoE routing openings
    # Per-layer expert Merkle roots for chain-mode verification.
    # Populated by the miner for challenged MoE layers.  The verifier checks
    # SHA256("MOE_LAYER_V2" || router_root || roots...) == on-chain layer root.
    expert_roots: Dict[int, List[bytes]] = field(default_factory=dict)
    # Generated output token ids (required to bind served output to commitment).
    output_token_ids: List[int] = field(default_factory=list)
    # Optional decode-integrity proofs for challenged output positions.
    sampling_proofs: List["SamplingProof"] = field(default_factory=list)
    # Embedding input binding proof.
    embedding_proof: Optional["EmbeddingProof"] = None

    @classmethod
    def empty(cls) -> "InferenceProofBundle":
        """Create an empty proof bundle (TEE mode — no ZK proofs)."""
        return cls(
            commitment=InferenceCommitment.empty(),
            beacon=b"",
            layer_proofs=[],
        )

    def to_bytes(self) -> bytes:
        """Serialize proof bundle."""
        # Simplified serialization - full implementation would be more complete
        return self.commitment.to_bytes() + self.beacon + self.validator_nonce

    def verify_beacon(self) -> bool:
        """Verify that beacon was correctly derived from commitment + nonce.

        Returns True if beacon = SHA256("VERILLM_BEACON_V2" || commitment_hash || nonce)
        """
        if not self.validator_nonce:
            return True  # No nonce stored, can't verify (legacy format)

        import hashlib
        h = hashlib.sha256()
        h.update(b"VERILLM_BEACON_V2")
        h.update(self.commitment.commitment_hash())
        h.update(self.validator_nonce)
        expected_beacon = h.digest()
        return self.beacon == expected_beacon


@dataclass
class ModelSpec:
    """Specification of a model for verification."""

    model_id: str
    weight_merkle_root: bytes
    num_layers: int
    hidden_dim: int
    num_heads: int
    head_dim: int
    intermediate_dim: int  # FFN hidden size
    vocab_size: int
    activation: str  # "gelu" | "silu" | "relu"
    norm_type: str  # "layernorm" | "rmsnorm"
    attention_type: str  # "mha" | "gqa" | "mqa"

    # Per-layer weight commitments for efficient verification
    layer_weight_commitments: List[bytes] = field(default_factory=list)

    # Per-layer weight Merkle roots for lightweight verification
    weight_block_merkle_roots: List[bytes] = field(default_factory=list)

    # Chunk size used for flat weight Merkle trees (elements per leaf)
    w_merkle_chunk_size: int = 128

    # Per-expert weight Merkle roots for MoE layers
    expert_weight_merkle_roots: Dict[int, List[bytes]] = field(default_factory=dict)
    # Per-layer router weight Merkle roots for MoE layers
    router_weight_merkle_roots: Dict[int, bytes] = field(default_factory=dict)
    # Dedicated lm_head weight Merkle root (decode-integrity proofs).
    lm_head_weight_merkle_root: bytes = b""
    # Embedding table Merkle root (input binding proofs).
    embedding_weight_merkle_root: bytes = b""

    # Flat SHA256(safetensors bytes) — cheap model identity for TEE verification.
    # Not used by ZK path; only needed for TEE attestation binding.
    weight_file_hash: bytes = b""

    # On-chain anchor for validator-side tokenizer drift detection.
    # SHA256("VERILLM_TOKENIZER_V1" || tokenizer.json bytes || 0x00 || chat_template UTF-8).
    # The validator computes its local tokenizer hash at epoch start and
    # compares — on mismatch, the model is marked as drifted and the
    # validator refuses to verify proofs (attributed as validator-side, NOT
    # a miner failure).  Per-request commitment is unchanged.
    tokenizer_hash: bytes = b""

    # Auto-detected from model (informational)
    quant_mode: str = "fp16"

    # Group size for INT4 (GPTQ/AWQ) quantization
    int4_group_size: int = 128

    # Number of columns in the W matrix for MoE expert GEMMs
    expert_w_num_cols: int = 0

    # MoE expert count (0 = dense model). Stored on-chain for chain-mode
    # verification where expert_weight_merkle_roots is not available.
    num_experts: int = 0
    # Global router metadata for lightweight routing checks.
    router_top_k: int = 0
    router_scoring: str = "softmax"

    # Pickle-safe defaults: pickle bypasses __init__/__post_init__, so old
    # cached objects may lack fields added after they were serialized.
    # __getattr__ is only called when normal attribute lookup fails.
    _PICKLE_DEFAULTS = {
        "router_weight_merkle_roots": dict,
        "expert_weight_merkle_roots": dict,
        "layer_weight_commitments": list,
        "weight_block_merkle_roots": list,
        "num_experts": lambda: 0,
        "router_top_k": lambda: 0,
        "router_scoring": lambda: "softmax",
        "expert_w_num_cols": lambda: 0,
        "int4_group_size": lambda: 128,
        "w_merkle_chunk_size": lambda: 128,
        "quant_mode": lambda: "fp16",
        "lm_head_weight_merkle_root": lambda: b"",
        "embedding_weight_merkle_root": lambda: b"",
        "weight_file_hash": lambda: b"",
        "tokenizer_hash": lambda: b"",
    }

    def __getattr__(self, name: str):
        factory = ModelSpec._PICKLE_DEFAULTS.get(name)
        if factory is not None:
            value = factory()
            object.__setattr__(self, name, value)
            return value
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")


# ============================================================================
# Protocol Request/Timing Types
# ============================================================================


@dataclass
class InferenceRequest:
    """Request from validator to miner.

    The validator_nonce is critical for security: it's random bytes generated
    by the validator and sent WITH the request, BEFORE the miner runs inference.

    The beacon is derived as: SHA256(commitment_hash || validator_nonce)

    This prevents the miner from grinding through commitment variations to find
    favorable challenges, because:
    1. The nonce is fixed before inference starts
    2. The commitment depends on actual activations (miner can't predict it)
    3. Both values are needed to compute the beacon

    The miner cannot:
    - Change the nonce (validator controls it)
    - Predict the commitment before running inference
    - Grind for favorable beacons (would need to re-run inference each attempt)
    """
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    prompt: str = ""
    validator_nonce: bytes = field(default_factory=lambda: os.urandom(32))
    max_new_tokens: int = 50
    do_sample: bool = False
    temperature: float = 1.0
    sampling_verification_bps: int = 0
    # Sampling params bound into sampler_config_hash.
    top_k: int = -1
    top_p: float = 1.0
    min_p: float = 0.0
    presence_penalty: float = 0.0

    def size_bytes(self) -> int:
        return len(self.prompt.encode()) + len(self.validator_nonce) + 100  # Approximate


@dataclass
class PhaseTimings:
    """Timing information for each protocol phase."""
    setup_ms: float = 0.0
    request_ms: float = 0.0
    inference_ms: float = 0.0
    commitment_ms: float = 0.0
    beacon_ms: float = 0.0
    challenge_ms: float = 0.0
    prove_ms: float = 0.0
    verify_ms: float = 0.0

    def miner_overhead_ms(self) -> float:
        """Miner-side overhead: commitment + beacon + challenge derivation + proving."""
        return self.commitment_ms + self.beacon_ms + self.challenge_ms + self.prove_ms

    def validator_overhead_ms(self) -> float:
        """Validator-side overhead: verification only."""
        return self.verify_ms

    def total_overhead_ms(self) -> float:
        """Total protocol overhead (miner + validator)."""
        return self.miner_overhead_ms() + self.validator_overhead_ms()


# ============================================================================
# MoE (Mixture of Experts) Types
# ============================================================================


@dataclass
class MoELayerSpec:
    """Specification for one MoE layer's expert configuration."""

    layer_idx: int
    num_routed_experts: int
    num_shared_experts: int
    top_k: int  # Number of experts selected per token
    expert_weight_roots: List[bytes]  # Per-expert Merkle roots
    shared_expert_weight_roots: List[bytes]  # Shared expert roots
    router_weight_root: bytes  # Router/gate weight commitment


@dataclass
class ExpertChallenge:
    """Challenge for a specific expert within an MoE layer."""

    expert_idx: int
    gemm_idx: int  # 0=gate/w1, 1=up/w2, 2=down/w3
    block_indices: List[Tuple[int, int]]  # (bi, bj) pairs to verify


@dataclass
class MoELayerChallenge:
    """Challenge for an MoE layer (extends LayerChallenge concept)."""

    layer_idx: int
    expert_challenges: List[ExpertChallenge]  # Challenges for routed experts
    shared_expert_challenges: List[ExpertChallenge]  # Challenges for shared experts
    verify_routing: bool  # Whether to verify router decisions
    sampled_token_indices: List[int]  # Which tokens were sampled for challenge


@dataclass
class RouterCommitment:
    """Commitment to routing decisions for one MoE layer."""

    router_logits_hash: bytes  # Hash of full router logits
    selected_experts: List[List[int]]  # [token][rank] -> expert_id
    routing_weights: List[List[int]]  # [token][rank] -> weight (field element)
    # Quantized proof-domain top-k (same domain as router GEMM proof: X_int @ W_int).
    # This field enables quantization-agnostic exact checks between committed
    # routing and proved router outputs.
    proof_selected_experts: List[List[int]] = field(default_factory=list)
    router_logits_row_root: bytes = b""  # Merkle root over per-token logits rows
    seq_len: int = 0
    num_experts: int = 0
    top_k: int = 0
    scoring_func: str = "softmax"


@dataclass
class RouterLogitsOpening:
    """Merkle opening for one sampled token's router logits row."""

    token_idx: int
    logits: List[float]  # Full logits row [num_experts]
    merkle_path: MerklePath


@dataclass
class RouterOutputRow:
    """Proven router output row for one sampled token."""

    token_idx: int
    logits_int: List[int]  # Quantized/proven router output row [num_experts]


@dataclass
class RouterLayerProof:
    """Routing proof payload for one MoE layer in lightweight mode."""

    layer_idx: int
    router_weight_root: bytes
    logits_openings: List[RouterLogitsOpening]
    router_gemm_proof: Optional[GEMMProof] = None
    proved_output_rows: List[RouterOutputRow] = field(default_factory=list)


@dataclass
class SamplingProof:
    """Proof payload for one challenged decode position.

    When positions are batched into a single GEMM proof, only the first
    SamplingProof carries ``lm_head_gemm_proof`` and ``logits_merkle_path``;
    subsequent proofs set them to ``None``.  The verifier reconstructs the
    batched X/Y from individual rows and verifies the single proof.
    """

    decode_step: int
    token_id: int
    hidden_row: bytes  # fp16 row bytes [hidden_dim]
    hidden_merkle_path: MerklePath
    # Proven lm_head output row serialized as little-endian int32 [vocab_size].
    proved_logits_i32: bytes
    lm_head_weight_root: bytes
    # Batched GEMM proof — only on the first SamplingProof per challenge.
    lm_head_gemm_proof: Optional[GEMMProof] = None
    # Path for proved_logits_i32 under lm_head_gemm_proof.output_root.
    logits_merkle_path: Optional[MerklePath] = None
    # High-assurance: actual fp16 logits row from model output.
    fp16_logits_row: Optional[bytes] = None
    # Merkle path for fp16_logits_row under decode_logits_row_root.
    fp16_logits_merkle_path: Optional[MerklePath] = None
    # Opened sampling seed for do_sample=True canonical replay.
    sampling_seed: Optional[bytes] = None


@dataclass
class EmbeddingRowOpening:
    """Merkle opening for one embedding table row (proves correct lookup)."""

    token_position: int  # Position in the input sequence
    token_id: int  # Token ID at this position
    leaf_data: bytes  # Raw leaf bytes from the embedding weight Merkle tree
    merkle_path: MerklePath  # Path to the embedding_weight_merkle_root


@dataclass
class EmbeddingOutputOpening:
    """Merkle opening for one position in the embedding output tensor.

    Proves that the actual embedding output at position p matches the
    embedding weight row for token_id_p (embedding output binding).
    """

    token_position: int  # Position in the input sequence
    leaf_data: bytes  # Raw leaf bytes from the embedding output Merkle tree
    merkle_path: MerklePath  # Path to the embedding_output_root


@dataclass
class EmbeddingProof:
    """Proof that the miner used the correct embedding rows for the input tokens.

    The validator sends a prompt, the miner tokenizes it, and commits
    input_commitment = SHA256(input_token_ids).  The embedding proof
    cryptographically binds those token IDs to the on-chain embedding
    weight Merkle root by opening rows of the embedding table at
    beacon-derived random positions.

    row_openings prove token IDs exist in the on-chain embedding table.
    output_openings prove the embedding output matches those rows
    (binds proven rows to actual layer 0 input).
    """

    input_token_ids: List[int]  # Full input token ID sequence
    row_openings: List[EmbeddingRowOpening]  # weight Merkle proofs
    output_openings: List[EmbeddingOutputOpening] = field(default_factory=list)  # output binding proofs


@dataclass
class RoutingProof:
    """Proof for router decision verification."""

    spot_checks: List[SpotCheck]  # Spot checks on router output
    top_k_consistency: bool  # Whether top-k matches commitment


@dataclass
class ExpertProof:
    """Proof for a single expert's GEMM computation."""

    expert_idx: int
    gemm_idx: int  # Which GEMM in the expert (gate/up/down)
    gemm_proof: GEMMProof  # Standard GEMM proof
    expert_merkle_path: MerklePath  # Path from expert root to layer root


@dataclass
class MoELayerProof:
    """Complete proof for one MoE layer."""

    layer_idx: int
    expert_proofs: List[ExpertProof]  # Proofs for challenged experts
    shared_expert_proofs: List[ExpertProof]  # Proofs for shared experts
    routing_proof: Optional[RoutingProof]  # Router verification (if challenged)
    router_commitment: RouterCommitment  # Committed routing decisions


@dataclass
class MoEModelSpec:
    """Extended model specification for MoE models."""

    # Base model info (mirrors ModelSpec)
    model_id: str
    weight_merkle_root: bytes
    num_layers: int
    hidden_dim: int
    num_heads: int
    head_dim: int
    intermediate_dim: int
    vocab_size: int
    activation: str  # "gelu" | "silu" | "swiglu" | "relu"
    norm_type: str  # "layernorm" | "rmsnorm"
    attention_type: str  # "mha" | "gqa" | "mqa" | "mla"

    # MoE-specific fields
    is_moe: bool = True
    moe_layer_indices: List[int] = field(default_factory=list)
    moe_layer_specs: List[MoELayerSpec] = field(default_factory=list)

    # Quantization (same as ModelSpec)
    layer_weight_commitments: List[bytes] = field(default_factory=list)
    weight_block_merkle_roots: List[bytes] = field(default_factory=list)
    w_merkle_chunk_size: int = 128
    quant_mode: str = "fp16"
    int4_group_size: int = 128
