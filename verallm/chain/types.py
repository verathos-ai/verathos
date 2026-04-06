"""On-chain type bridge: Solidity structs <-> Python dataclasses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from verallm.types import ModelSpec


@dataclass
class OnChainModelSpec:
    """Mirror of ModelRegistry.ModelSpec Solidity struct."""

    model_id: str
    weight_merkle_root: bytes  # 32 bytes
    layer_roots: List[bytes]  # 32 bytes each
    num_layers: int
    hidden_dim: int
    intermediate_dim: int
    num_heads: int
    head_dim: int
    vocab_size: int
    quant_mode: str
    merkle_chunk_size: int
    activation: str
    norm_type: str
    attention_type: str
    num_experts: int = 0  # MoE expert count (0 = dense)
    expert_w_num_cols: int = 0  # columns per expert W matrix (0 = use intermediate_dim)
    lm_head_root: bytes = b""  # 32 bytes — Merkle root of lm_head weights
    embedding_root: bytes = b""  # 32 bytes — Merkle root of embedding table weights
    weight_file_hash: bytes = b""  # 32 bytes — SHA256(safetensors) for TEE verification
    tokenizer_hash: bytes = b""  # 32 bytes — anchor for validator-side tokenizer drift detection


@dataclass
class OnChainMinerModel:
    """Mirror of MinerRegistry.MinerModel Solidity struct."""

    model_id: str
    endpoint: str
    model_spec_ref: bytes  # 32 bytes (keccak256 of ModelSpec)
    quant: str
    max_context_len: int
    expires_at: int  # unix timestamp
    active: bool


@dataclass
class OnChainValidatorInfo:
    """Mirror of ValidatorRegistry.ValidatorInfo Solidity struct."""

    proxy_endpoint: str
    uid: int
    registered_at: int  # unix timestamp
    updated_at: int  # unix timestamp
    active: bool


@dataclass
class OnChainTEECapability:
    """Mirror of MinerRegistry.TEECapability Solidity struct."""

    enabled: bool = False
    platform: str = ""  # "tdx", "sev-snp", "mock"
    enclave_pub_key: bytes = b""  # 32-byte X25519 public key
    attestation_hash: bytes = b""  # keccak256(full attestation report)
    attested_at: int = 0  # block number when attested
    model_weight_hash: bytes = b""  # SHA256(safetensors) for on-chain cross-check
    code_measurement: bytes = b""  # keccak256(mr_td) for TDX, keccak256(launch_digest) for SEV-SNP


@dataclass
class ScoringParams:
    """Mirror of SubnetConfig.ScoringParams Solidity struct.

    All bps fields: divide by 10000 to get the float value.
    E.g. emaAlphaBps=2000 → ema_alpha=0.20, throughputPowerBps=20000 → throughput_power=2.0.
    """

    tee_bonus_bps: int = 1000                # 10% TEE bonus
    ema_alpha_bps: int = 2000                # 0.20
    throughput_power_bps: int = 20000        # 2.0
    proof_sample_rate_bps: int = 3000        # 30%
    probation_required_passes: int = 3       # 3 consecutive passes
    demand_bonus_max_bps: int = 2000         # 20%
    emission_burn_bps: int = 5000            # 50%

    @property
    def tee_bonus(self) -> float:
        return 1.0 + self.tee_bonus_bps / 10000.0

    @property
    def ema_alpha(self) -> float:
        return self.ema_alpha_bps / 10000.0

    @property
    def throughput_power(self) -> float:
        return self.throughput_power_bps / 10000.0

    @property
    def proof_sample_rate(self) -> float:
        return self.proof_sample_rate_bps / 10000.0

    @property
    def demand_bonus_max(self) -> float:
        return self.demand_bonus_max_bps / 10000.0

    @property
    def emission_burn(self) -> float:
        return self.emission_burn_bps / 10000.0


def on_chain_to_model_spec(oc: OnChainModelSpec) -> ModelSpec:
    """Convert an on-chain ModelSpec to the Python ModelSpec used by verifiers.

    Expert weight Merkle roots are NOT stored on-chain — they are verified
    via hierarchical hashing. New registrations use:
    layer_root = SHA256("MOE_LAYER_V2" || router_root || expert_roots...).
    Legacy registrations may use MOE_LAYER_V1 without router_root.
    The verifier checks expert roots from the proof bundle against the on-chain layer root.
    """
    return ModelSpec(
        model_id=oc.model_id,
        weight_merkle_root=oc.weight_merkle_root,
        num_layers=oc.num_layers,
        hidden_dim=oc.hidden_dim,
        num_heads=oc.num_heads,
        head_dim=oc.head_dim,
        intermediate_dim=oc.intermediate_dim,
        vocab_size=oc.vocab_size,
        activation=oc.activation,
        norm_type=oc.norm_type,
        attention_type=oc.attention_type,
        weight_block_merkle_roots=list(oc.layer_roots),
        w_merkle_chunk_size=oc.merkle_chunk_size,
        quant_mode=oc.quant_mode,
        expert_w_num_cols=oc.expert_w_num_cols,
        num_experts=oc.num_experts,
        lm_head_weight_merkle_root=oc.lm_head_root if oc.lm_head_root else b"",
        embedding_weight_merkle_root=oc.embedding_root if oc.embedding_root else b"",
        weight_file_hash=oc.weight_file_hash if oc.weight_file_hash else b"",
        tokenizer_hash=oc.tokenizer_hash if oc.tokenizer_hash else b"",
    )


def model_spec_to_on_chain(spec: ModelSpec) -> OnChainModelSpec:
    """Convert a Python ModelSpec to the on-chain representation for registration."""
    # Infer num_experts from expert_weight_merkle_roots if present
    num_experts = 0
    if spec.expert_weight_merkle_roots:
        first_layer_roots = next(iter(spec.expert_weight_merkle_roots.values()), [])
        num_experts = len(first_layer_roots)

    return OnChainModelSpec(
        model_id=spec.model_id,
        weight_merkle_root=spec.weight_merkle_root,
        layer_roots=list(spec.weight_block_merkle_roots),
        num_layers=spec.num_layers,
        hidden_dim=spec.hidden_dim,
        intermediate_dim=spec.intermediate_dim,
        num_heads=spec.num_heads,
        head_dim=spec.head_dim,
        vocab_size=spec.vocab_size,
        quant_mode=spec.quant_mode,
        merkle_chunk_size=spec.w_merkle_chunk_size,
        activation=spec.activation,
        norm_type=spec.norm_type,
        attention_type=spec.attention_type,
        num_experts=num_experts,
        expert_w_num_cols=spec.expert_w_num_cols,
        lm_head_root=spec.lm_head_weight_merkle_root if spec.lm_head_weight_merkle_root else b"\x00" * 32,
        embedding_root=spec.embedding_weight_merkle_root if spec.embedding_weight_merkle_root else b"\x00" * 32,
        weight_file_hash=getattr(spec, "weight_file_hash", b"") or b"\x00" * 32,
        tokenizer_hash=getattr(spec, "tokenizer_hash", b"") or b"\x00" * 32,
    )
