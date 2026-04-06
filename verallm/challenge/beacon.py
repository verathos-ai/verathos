"""
Beacon-based challenge derivation.

Deterministically derives which layers/operations to verify
from a random beacon and the inference commitment.
"""

from typing import List, Tuple, Optional
import hashlib
import struct
import random

from verallm.types import (
    InferenceCommitment,
    SamplingChallenge,
    EmbeddingChallenge,
    ChallengeSet,
    LayerChallenge,
    GEMMChallenge,
)
from verallm.sampling import clamp_sampling_bps, HIGH_ASSURANCE_BPS
from verallm.config import get_config

# Re-export pure functions from zkllm
from zkllm.challenge.beacon import (  # noqa: F401
    derive_beacon,
    derive_beacon_from_nonce,
    compute_detection_probability,
)


def derive_challenges(
    beacon: bytes,
    commitment: InferenceCommitment,
    k_layers: Optional[int] = None,
    k_gemms_per_layer: int = 2,
    k_blocks_per_gemm: int = 4,
) -> ChallengeSet:
    """
    Derive deterministic challenges from beacon and commitment.

    This is reproducible by both prover and verifier.

    Args:
        beacon: Random 32-byte beacon
        commitment: The inference commitment
        k_layers: Number of layers to challenge (default from config)
        k_gemms_per_layer: Number of GEMMs per layer to verify
        k_blocks_per_gemm: Number of blocks per GEMM

    Returns:
        ChallengeSet with deterministic challenges
    """
    config = get_config()
    k_layers = k_layers or config.k_layers

    # Derive seed from beacon + commitment
    seed_material = (
        b"VERILLM_CHALLENGE_V1"
        + beacon
        + commitment.model_commitment
        + commitment.input_commitment
        + commitment.output_commitment
    )
    for lc in commitment.layer_commitments:
        seed_material += lc

    seed = hashlib.sha256(seed_material).digest()

    # Use seed for deterministic sampling
    rng = random.Random(int.from_bytes(seed[:8], "little"))

    num_layers = len(commitment.layer_commitments)
    if num_layers == 0:
        return ChallengeSet(beacon=beacon, layer_challenges=[])

    # Sample k_layers unique layer indices
    k_actual = min(k_layers, num_layers)
    layer_indices = rng.sample(range(num_layers), k_actual)

    # For each layer, derive GEMM and block challenges
    layer_challenges: List[LayerChallenge] = []

    for layer_idx in layer_indices:
        # Derive layer-specific seed
        layer_seed = hashlib.sha256(seed + struct.pack("<Q", layer_idx)).digest()
        layer_rng = random.Random(int.from_bytes(layer_seed[:8], "little"))

        # Typical transformer has 6 major GEMMs per layer
        num_gemms = 6
        gemm_indices = layer_rng.sample(range(num_gemms), min(k_gemms_per_layer, num_gemms))

        gemm_challenges: List[GEMMChallenge] = []

        for gemm_idx in gemm_indices:
            gemm_seed = hashlib.sha256(
                layer_seed + struct.pack("<Q", gemm_idx)
            ).digest()
            gemm_rng = random.Random(int.from_bytes(gemm_seed[:8], "little"))

            max_blocks = 16
            num_blocks = min(k_blocks_per_gemm, max_blocks * max_blocks)

            block_indices: List[Tuple[int, int]] = []
            all_blocks = [(i, j) for i in range(max_blocks) for j in range(max_blocks)]

            if len(all_blocks) <= num_blocks:
                block_indices = all_blocks
            else:
                block_indices = gemm_rng.sample(all_blocks, num_blocks)

            gemm_challenges.append(
                GEMMChallenge(
                    gemm_idx=gemm_idx,
                    block_indices=block_indices,
                )
            )

        layer_challenges.append(
            LayerChallenge(
                layer_idx=layer_idx,
                gemm_challenges=gemm_challenges,
            )
        )

    return ChallengeSet(
        beacon=beacon,
        layer_challenges=layer_challenges,
    )


def derive_embedding_challenge(
    beacon: bytes,
    commitment: InferenceCommitment,
    num_input_tokens: int,
    k_positions: int = 5,
) -> Optional[EmbeddingChallenge]:
    """Derive which input token positions to verify for embedding binding.

    Selects k random positions from the input sequence. The miner must
    provide Merkle inclusion proofs for the corresponding embedding rows.

    Args:
        beacon: Random 32-byte beacon
        commitment: The inference commitment
        num_input_tokens: Length of the input token sequence
        k_positions: Number of positions to challenge (default 5)

    Returns:
        EmbeddingChallenge, or None if num_input_tokens is 0
    """
    if num_input_tokens <= 0:
        return None

    seed = hashlib.sha256(
        b"VERILLM_EMBEDDING_CHALLENGE_V1"
        + beacon
        + commitment.input_commitment
        + struct.pack("<I", num_input_tokens)
    ).digest()
    rng = random.Random(int.from_bytes(seed[:8], "little"))

    k_actual = min(k_positions, num_input_tokens)
    positions = sorted(rng.sample(range(num_input_tokens), k_actual))

    return EmbeddingChallenge(token_positions=positions)


def should_challenge_sampling(beacon: bytes, sampling_verification_bps: int) -> bool:
    """Fiat-Shamir gate for decode-integrity sampling verification."""
    bps = clamp_sampling_bps(sampling_verification_bps)
    if bps <= 0:
        return False
    if bps >= 10_000:
        return True
    h = hashlib.sha256(b"VERILLM_SAMPLING_GATE_V1" + beacon).digest()
    draw = int.from_bytes(h[:2], "little")  # [0, 65535]
    threshold = (bps * 65536) // 10_000
    return draw < threshold


def _compute_k_positions(num_output_tokens: int) -> int:
    """Scale challenged decode positions with output length.

    One verified position is sufficient to catch a wrong lm_head — the
    challenged position is Fiat-Shamir derived and unpredictable.  Extra
    positions only add redundancy, so we scale conservatively to keep
    proof overhead low (each position costs one full lm_head GEMM).
    """
    if num_output_tokens <= 1024:
        return 1
    elif num_output_tokens <= 4096:
        return 2
    else:
        return 3


def derive_sampling_challenge(
    beacon: bytes,
    commitment: InferenceCommitment,
    vocab_size: int,
    k_positions: Optional[int] = None,
) -> Optional[SamplingChallenge]:
    """Derive decode positions to verify sampling correctness.

    For temperature=0 (greedy): argmax verification against proved logits.
    For do_sample=True (temperature > 0): canonical sampler replay
    verification when ``sampling_seed_commitment`` is present.

    Args:
        k_positions: Override number of challenged positions.
            None = auto-scale with output length.
    """
    # Allow sampling challenge for do_sample=True when seed is committed.
    is_greedy = (not commitment.do_sample) and (commitment.temperature_milli == 0)
    is_sampled_with_seed = (
        commitment.do_sample
        and commitment.sampling_seed_commitment
    )
    if not is_greedy and not is_sampled_with_seed:
        return None
    if not commitment.decode_hidden_row_root:
        return None
    num_steps = int(commitment.output_token_count or 0)
    if num_steps <= 0:
        return None
    if not should_challenge_sampling(beacon, commitment.sampling_verification_bps):
        return None

    if k_positions is None:
        k_positions = _compute_k_positions(num_steps)

    seed = hashlib.sha256(
        b"VERILLM_SAMPLING_CHALLENGE_V1"
        + beacon
        + commitment.decode_hidden_row_root
        + commitment.output_commitment
        + struct.pack("<I", int(vocab_size))
    ).digest()
    rng = random.Random(int.from_bytes(seed[:8], "little"))

    k_actual = min(max(1, int(k_positions)), num_steps)
    decode_positions = sorted(rng.sample(range(num_steps), k_actual))

    # One-row lm_head proofs use a single Y block spanning vocab.
    block_indices = [(0, 0)]

    # High-assurance is determined solely by the validator-requested bps.
    # The verifier enforces that decode_logits_row_root is present when
    # high_assurance=True — a miner cannot downgrade by omitting it.
    bps = clamp_sampling_bps(commitment.sampling_verification_bps)
    high_assurance = bps >= HIGH_ASSURANCE_BPS

    return SamplingChallenge(
        decode_positions=decode_positions,
        lm_head_block_indices=block_indices,
        high_assurance=high_assurance,
    )
