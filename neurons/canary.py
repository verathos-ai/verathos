"""Epoch canary testing — indistinguishable-from-real-traffic tests for miners.

Validators send canary tests through the normal inference pipeline (POST /inference),
making them indistinguishable from real user requests.  Each epoch (~72 min), every
miner is tested with a mix of small + full-context canaries.

Small canaries: 500-2000 input tokens, 100-300 output tokens, varied temperature.
Full-context canaries: ~80% of max_context_len, 200 output tokens, temperature 0.0.

Test scheduling uses (epoch_number, validator_hotkey, miner_address, test_index)
plus a random per-epoch salt.  The validator hotkey ensures different validators
produce different schedules; the random salt prevents miners from precomputing
timing even if they know the hash formula.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple

from neurons.poi import SHARED_PROMPTS


# ── Canary prompt templates (diverse, realistic-looking queries) ────────

_CANARY_TEMPLATES = [
    "Write a detailed explanation of {topic}.",
    "Compare and contrast {topic_a} with {topic_b}. Include examples.",
    "What are the main advantages and disadvantages of {topic}?",
    "Explain {topic} as if you were teaching a graduate student.",
    "Provide a step-by-step guide for {topic}.",
    "What are the latest developments in {topic}?",
    "Analyze the trade-offs involved in {topic}.",
    "How does {topic} work under the hood?",
    "Write a comprehensive overview of {topic} for a technical blog post.",
    "What are common misconceptions about {topic}?",
    "Describe the historical evolution of {topic}.",
    "What practical applications does {topic} have in industry?",
]

_CANARY_TOPICS = [
    "zero-knowledge proofs",
    "transformer attention mechanisms",
    "distributed consensus algorithms",
    "gradient descent optimization",
    "model quantization techniques",
    "key-value caching in LLMs",
    "mixture-of-experts architectures",
    "flash attention implementations",
    "tensor parallelism for inference",
    "neural network pruning",
    "speculative decoding",
    "rotary position embeddings",
    "reinforcement learning from human feedback",
    "differential privacy in machine learning",
    "federated learning protocols",
    "homomorphic encryption",
    "blockchain consensus mechanisms",
    "cryptographic hash functions",
    "public key infrastructure",
    "Byzantine fault tolerance",
    "memory-efficient fine-tuning (LoRA)",
    "activation checkpointing",
    "continuous batching for inference",
    "knowledge distillation",
]


@dataclass
class CanaryTest:
    """A single canary test to execute against a miner."""

    miner_address: str
    miner_endpoint: str
    model_id: str
    model_index: int
    max_context_len: int

    target_block: int  # block at which to dispatch this test
    test_index: int  # index within this miner's epoch tests
    test_type: str  # "small" | "full_context"

    prompt: str
    max_new_tokens: int
    temperature: float
    verify_proof: bool  # whether to verify ZK proof for this test
    verify_tee: bool = False  # whether to verify TEE attestation for this test (replaces ZK for TEE miners)
    enable_thinking: bool = True  # chain-of-thought; randomized 50/50 so miners can't fingerprint canaries

    # When enable_thinking=False, use clean sampling params so argmax is strictly binding
    presence_penalty: Optional[float] = None  # None = use model default; 0.0 = explicit off
    top_k: Optional[int] = None
    top_p: Optional[float] = None


@dataclass
class CanaryScheduler:
    """Plans and dispatches canary tests across an epoch.

    Tests are spread across the epoch with target blocks derived from a
    hash of (epoch_number, validator_hotkey, miner_address, test_index)
    plus a random per-epoch salt.  The validator hotkey ensures different
    validators produce different schedules; the random salt ensures miners
    cannot precompute the schedule even if they know the hash formula.
    """

    epoch_number: int
    epoch_start_block: int
    epoch_blocks: int  # total blocks in epoch (e.g. 360)
    validator_hotkey: str = ""  # SS58 or hex address — differentiates validators
    small_count: int = 12
    full_context_count: int = 1
    proof_sample_rate: float = 0.30
    probation_entries: Set[Tuple[str, int]] = field(default_factory=set)
    tests: List[CanaryTest] = field(default_factory=list)
    _epoch_salt: bytes = field(default_factory=lambda: os.urandom(16), repr=False)

    def plan_epoch(
        self,
        miners: list,
    ) -> List[CanaryTest]:
        """Schedule all canary tests for the epoch.

        Args:
            miners: List of ActiveMiner entries to test.

        Returns:
            Sorted list of CanaryTest, ordered by target_block.
        """
        self.tests = []

        for miner in miners:
            test_idx = 0
            is_tee = getattr(miner, "tee_enabled", False)

            # Small canary tests
            for i in range(self.small_count):
                target_block = self._target_block(miner.address, test_idx)
                prompt, max_new_tokens, temperature = generate_small_canary_prompt(
                    self.epoch_number, miner.address, i,
                )
                # TEE miners: verify attestation instead of ZK proof
                verify = False if is_tee else self._should_verify_proof(
                    miner.address, miner.model_index, test_idx,
                )

                enable_thinking = self._should_enable_thinking(
                    miner.address, test_idx,
                )
                # Greedy canaries (temp=0) require enable_thinking=False so
                # the existing argmax sampling proof path is reachable.  The
                # thinking logits processor runs *after* compute_logits, so
                # the captured pre-mask logits diverge from the output token
                # and the miner's strict alignment check refuses to build
                # decode_hidden_row_root → no sampling proof.  Sampled
                # canaries (temp>0) keep randomized thinking — canonical
                # canonical replay handles thinking transparently because it
                # bypasses the alignment check by design.
                if temperature == 0.0:
                    enable_thinking = False

                # Always use clean sampling params for canary tests so argmax
                # check is strictly binding.  The server applies a default
                # presence_penalty when None, which alters post-logits sampling
                # and causes divergence with the captured pre-penalty logits.
                pp = 0.0
                tk = 0
                tp = 1.0

                self.tests.append(CanaryTest(
                    miner_address=miner.address,
                    miner_endpoint=miner.endpoint,
                    model_id=miner.model_id,
                    model_index=miner.model_index,
                    max_context_len=miner.max_context_len,
                    target_block=target_block,
                    test_index=test_idx,
                    test_type="small",
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    verify_proof=verify,
                    verify_tee=is_tee,
                    enable_thinking=enable_thinking,
                    presence_penalty=pp,
                    top_k=tk,
                    top_p=tp,
                ))
                test_idx += 1

            # Full-context canary tests (always verify proof or TEE)
            for i in range(self.full_context_count):
                target_block = self._target_block(miner.address, test_idx)
                prompt, max_new_tokens, temperature = generate_full_context_canary_prompt(
                    self.epoch_number, miner.address, i, miner.max_context_len,
                )

                # Full-context canaries are always temp=0.0 (greedy).  Force
                # enable_thinking=False so the strict argmax sampling proof
                # path is reachable.  See comment in the small-canary loop.
                enable_thinking = False

                pp = 0.0
                tk = 0
                tp = 1.0

                self.tests.append(CanaryTest(
                    miner_address=miner.address,
                    miner_endpoint=miner.endpoint,
                    model_id=miner.model_id,
                    model_index=miner.model_index,
                    max_context_len=miner.max_context_len,
                    target_block=target_block,
                    test_index=test_idx,
                    test_type="full_context",
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    verify_proof=not is_tee,  # ZK proof for non-TEE, skip for TEE
                    verify_tee=is_tee,
                    enable_thinking=enable_thinking,
                    presence_penalty=pp,
                    top_k=tk,
                    top_p=tp,
                ))
                test_idx += 1

        # Sort by target block for ordered dispatch
        self.tests.sort(key=lambda t: (t.target_block, t.miner_address))
        return self.tests

    def get_pending_tests(self, current_block: int) -> List[CanaryTest]:
        """Return tests whose target_block <= current_block (not yet dispatched).

        Removes returned tests from the internal list to avoid re-dispatch.
        """
        pending = [t for t in self.tests if t.target_block <= current_block]
        self.tests = [t for t in self.tests if t.target_block > current_block]
        return pending

    def _target_block(self, miner_address: str, test_index: int) -> int:
        """Derive target block for a test, spread across the epoch.

        Uses a hash of (epoch, validator_hotkey, miner, test_index, random salt)
        to spread tests uniformly across the epoch.  The validator hotkey ensures
        different validators get different schedules; the random salt prevents
        miners from precomputing timing.
        """
        seed = hashlib.sha256(
            b"VERATHOS_CANARY_SCHED_V2"
            + self.epoch_number.to_bytes(4, "big")
            + self.validator_hotkey.encode()
            + miner_address.encode()
            + test_index.to_bytes(2, "big")
            + self._epoch_salt
        ).digest()

        # Spread across 90% of the epoch (leave tail for grace period)
        usable_blocks = int(self.epoch_blocks * 0.90)
        offset = int.from_bytes(seed[:4], "big") % max(usable_blocks, 1)
        return self.epoch_start_block + offset

    def _should_enable_thinking(
        self, miner_address: str, test_index: int,
    ) -> bool:
        """~50/50 coin flip for enable_thinking.

        Randomized so miners cannot fingerprint canary tests by observing
        that thinking is always on or always off.  When thinking is disabled,
        greedy argmax checks become strictly binding (no logits-processor
        divergence excuse).
        """
        seed = hashlib.sha256(
            b"VERATHOS_CANARY_THINKING_V2"
            + self.epoch_number.to_bytes(4, "big")
            + self.validator_hotkey.encode()
            + miner_address.encode()
            + test_index.to_bytes(2, "big")
            + self._epoch_salt
        ).digest()
        return seed[0] < 128  # ~50% True, ~50% False

    def _should_verify_proof(
        self, miner_address: str, model_index: int, test_index: int,
    ) -> bool:
        """Deterministically decide whether to verify ZK proof for a small canary.

        Miners on probation ALWAYS get proof verification (100% rate).
        """
        # Probation entries: always verify proof
        if (miner_address, model_index) in self.probation_entries:
            return True

        seed = hashlib.sha256(
            b"VERATHOS_CANARY_PROOF_V2"
            + self.epoch_number.to_bytes(4, "big")
            + self.validator_hotkey.encode()
            + miner_address.encode()
            + test_index.to_bytes(2, "big")
            + self._epoch_salt
        ).digest()
        # Compare first byte against threshold (0-255)
        threshold = int(self.proof_sample_rate * 256)
        return seed[0] < threshold


# ── Prompt generation ─────────────────────────────────────────────────

def generate_small_canary_prompt(
    epoch_number: int,
    miner_address: str,
    test_index: int,
) -> tuple[str, int, float]:
    """Generate a small canary prompt that looks like a real user query.

    Returns:
        (prompt, max_new_tokens, temperature)
    """
    seed = hashlib.sha256(
        b"VERATHOS_SMALL_CANARY_V1"
        + epoch_number.to_bytes(4, "big")
        + miner_address.encode()
        + test_index.to_bytes(2, "big")
    ).digest()

    # Select template and topics
    template_idx = int.from_bytes(seed[:4], "big") % len(_CANARY_TEMPLATES)
    topic_idx = int.from_bytes(seed[4:8], "big") % len(_CANARY_TOPICS)
    topic_b_idx = int.from_bytes(seed[8:12], "big") % len(_CANARY_TOPICS)
    # Avoid same topic for A/B comparison
    if topic_b_idx == topic_idx:
        topic_b_idx = (topic_b_idx + 1) % len(_CANARY_TOPICS)

    template = _CANARY_TEMPLATES[template_idx]
    prompt = template.format(
        topic=_CANARY_TOPICS[topic_idx],
        topic_a=_CANARY_TOPICS[topic_idx],
        topic_b=_CANARY_TOPICS[topic_b_idx],
    )

    # Add context from SHARED_PROMPTS to increase prompt length
    num_context = 2 + (int.from_bytes(seed[12:14], "big") % 4)  # 2-5 extra questions
    context_parts = []
    rng = seed
    for i in range(num_context):
        rng = hashlib.sha256(b"CANARY_CTX" + rng).digest()
        idx = int.from_bytes(rng[:4], "big") % len(SHARED_PROMPTS)
        context_parts.append(SHARED_PROMPTS[idx])

    full_prompt = (
        prompt + "\n\nFor context, also consider these related questions:\n"
        + "\n".join(f"- {p}" for p in context_parts)
    )

    # max_new_tokens: 100-300 (deterministic)
    max_new_tokens = 100 + (int.from_bytes(seed[14:16], "big") % 201)

    # temperature: 0.0, 0.3, 0.5, 0.7, or 1.0
    temp_options = [0.0, 0.3, 0.5, 0.7, 1.0]
    temperature = temp_options[int.from_bytes(seed[16:18], "big") % len(temp_options)]

    return full_prompt, max_new_tokens, temperature


def generate_full_context_canary_prompt(
    epoch_number: int,
    miner_address: str,
    test_index: int,
    max_context_len: int,
) -> tuple[str, int, float]:
    """Generate a full-context canary prompt (~80% of max_context_len).

    Returns:
        (prompt, max_new_tokens, temperature)
    """
    seed = hashlib.sha256(
        b"VERATHOS_FULL_CANARY_V1"
        + epoch_number.to_bytes(4, "big")
        + miner_address.encode()
        + test_index.to_bytes(2, "big")
    ).digest()

    fill_tokens = int(max_context_len * 0.8)
    target_chars = fill_tokens * 4  # ~4 chars per token (conservative)

    # Build a long prompt from SHARED_PROMPTS + context
    parts = [
        "You are a knowledgeable AI assistant. Please provide detailed, "
        "accurate responses to the following questions. Each answer should "
        "be thorough and well-structured.\n\n"
    ]
    total_chars = len(parts[0])

    question_num = 1
    rng = seed
    while total_chars < target_chars:
        rng = hashlib.sha256(b"CANARY_RNG" + rng).digest()
        topic_idx = int.from_bytes(rng[:4], "big") % len(_CANARY_TOPICS)
        template_idx = int.from_bytes(rng[4:8], "big") % len(_CANARY_TEMPLATES)
        prompt_idx = int.from_bytes(rng[8:12], "big") % len(SHARED_PROMPTS)

        entry = (
            f"Question {question_num}: "
            + _CANARY_TEMPLATES[template_idx].format(
                topic=_CANARY_TOPICS[topic_idx],
                topic_a=_CANARY_TOPICS[topic_idx],
                topic_b=_CANARY_TOPICS[(topic_idx + 1) % len(_CANARY_TOPICS)],
            )
            + f" Additionally, {SHARED_PROMPTS[prompt_idx]}\n\n"
        )
        parts.append(entry)
        total_chars += len(entry)
        question_num += 1

    result = "".join(parts)
    # Trim to approximate target (don't cut mid-word), leaving room for nonce
    nonce_suffix = f"\n[verification nonce: {seed.hex()[:32]}]"
    trim_target = target_chars - len(nonce_suffix)
    if len(result) > trim_target:
        result = result[:trim_target].rsplit(" ", 1)[0]

    # Nonce suffix — unique per epoch/miner, prevents caching across epochs
    result += nonce_suffix

    return result, 200, 0.0
