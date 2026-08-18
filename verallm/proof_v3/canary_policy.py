"""Authority-signed network policy for proof-v3 canary obligations.

Execution profiles authenticate model arithmetic.  This separate, small
artifact authenticates the network policy that decides how often those
profiles must be exercised.  Keeping the policy separate avoids resigning
large model artifacts when the subnet changes only scheduling policy.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from verallm.proof_v3.errors import (
    ProofV3DocumentError,
    ProofV3SignatureError,
    ProofV3VerificationError,
)


LEGACY_CANARY_POLICY_ABI_V3 = "proof_v3.canary_policy.v5"
UNIFORM_CANARY_POLICY_ABI_V3 = "proof_v3.canary_policy.v6"
CANARY_POLICY_ABI_V3 = "proof_v3.canary_policy.v7"
CANARY_PROMPT_RECIPE_ABI_V3 = "natural_composite.secret_seeded.long_form.v3"
CANARY_CONTEXT_CONSTRUCTION_ABI_V3 = "validator_tokenizer.near_advertised.v2"
CANARY_BUSY_POLICY_ABI_V3 = "validator_receipt_interval.one_epoch_debt.v1"
CANARY_SCHEDULE_SELECTION_ABI_V3 = (
    "validator_secret.marked_bernoulli_per_block.v1"
)
CANARY_HARD_DECODE_SELECTION_ABI_V3 = (
    "secret_seeded.independent_common_anchors_log_uniform_tail.v2"
)
CANARY_REPEAT_PREFIX_ABI_V3 = "validator_secret.shared_prefix_groups.v1"
LEGACY_CANARY_OWNER_CONTEXT_SIZING_ABI_V3 = (
    "secret_seeded.min_heavy_log_uniform_prompt.bernoulli_max.v1"
)
CANARY_OWNER_CONTEXT_SIZING_ABI_V3 = (
    "secret_seeded.min_heavy_geometric_prompt.bernoulli_max.v2"
)
MIN_CANARY_OWNER_FULL_MAX_DRAW_BPS_V3 = 100
LEGACY_CANARY_OWNER_FULL_MIN_PROMPT_BPS_V3 = 1_000
DEFAULT_CANARY_OWNER_FULL_MIN_PROMPT_BPS_V3 = 500
DEFAULT_CANARY_OWNER_FULL_MAX_DRAW_BPS_V3 = 100
CANARY_PROMPT_MIN_TOKEN_TOLERANCE_V3 = 64
CANARY_HARD_DECODE_ANCHORS_V3 = (512, 1_024, 2_048, 4_096, 8_192)
MIN_CANARY_HARD_DECODE_ANCHOR_BPS_V3 = 2_500
MIN_CANARY_HARD_DECODE_TAIL_BPS_V3 = 1_000
MIN_CANARY_LATE_DECODE_OUTPUT_BPS_V3 = 8_000
DEFAULT_CANARY_CANDIDATE_TARGET_PER_EPOCH_V3 = 2
DEFAULT_CANARY_CANDIDATE_HARD_BPS_V3 = 5_000
DEFAULT_CANARY_MAX_HARD_DROUGHT_EPOCHS_V3 = 2
DEFAULT_CANARY_REPEAT_PREFIX_TARGET_BPS_V3 = 5_000
DEFAULT_CANARY_REPEAT_PREFIX_MIN_TOKENS_V3 = 256
# Retained only for decoding pre-v4 in-memory test objects. The v4 signed
# policy never creates paired requests.
MAX_CANARY_FULL_PAIR_HOLD_SECONDS_V3 = 930
MAX_CANARY_POLICY_DOCUMENT_BYTES = 1 << 18
MAX_CANARY_POLICY_SIGNATURES = 32
MAX_CANARY_POLICY_MODELS = 1 << 14
_POLICY_DOMAIN = b"VERATHOS/PROOF_V3/CANARY_POLICY/V3/SHA256"
_SECP256K1_ORDER = (
    0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
)
_SECP256K1_HALF_ORDER = _SECP256K1_ORDER // 2


def canary_prompt_token_tolerance_v3(target_tokens: int) -> int:
    """Return the canonical under-target tokenizer tolerance for one canary."""

    if type(target_tokens) is not int or target_tokens <= 0:
        raise ProofV3DocumentError(
            "canary target prompt tokens must be a positive integer"
        )
    return max(
        CANARY_PROMPT_MIN_TOKEN_TOLERANCE_V3,
        target_tokens // 500,
    )


def _identifier(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 128
        or any(not (ch.isalnum() or ch in "._:-/") for ch in value)
    ):
        raise ProofV3DocumentError(f"{name} is malformed")
    return value


def _u32(value: object, name: str, *, minimum: int = 0) -> int:
    if (
        type(value) is not int
        or value < minimum
        or value >= 1 << 32
    ):
        raise ProofV3DocumentError(f"{name} is out of range")
    return value


def _hex32(value: object, name: str) -> bytes:
    if not isinstance(value, str) or len(value) != 64 or value.lower() != value:
        raise ProofV3DocumentError(f"{name} must be lowercase hexadecimal")
    try:
        decoded = bytes.fromhex(value)
    except ValueError as exc:
        raise ProofV3DocumentError(f"{name} must be hexadecimal") from exc
    if len(decoded) != 32:
        raise ProofV3DocumentError(f"{name} must be exactly 32 bytes")
    return decoded


def _canonical_quant(value: object) -> str:
    quant = _identifier(value, "allowed quantization").lower()
    if len(quant) > 32:
        raise ProofV3DocumentError("allowed quantization is too long")
    return quant


def _signature(value: object) -> bytes:
    if isinstance(value, bytes):
        signature = value
    elif isinstance(value, str):
        encoded = value[2:] if value.startswith("0x") else value
        try:
            signature = bytes.fromhex(encoded)
        except ValueError as exc:
            raise ProofV3SignatureError(
                "canary policy signature must be hexadecimal"
            ) from exc
    else:
        raise ProofV3SignatureError(
            "canary policy signature must be bytes or hexadecimal"
        )
    if len(signature) != 65:
        raise ProofV3SignatureError(
            "canary policy signature must be exactly 65 bytes"
        )
    r = int.from_bytes(signature[:32], "big")
    s = int.from_bytes(signature[32:64], "big")
    if (
        not 0 < r < _SECP256K1_ORDER
        or not 0 < s <= _SECP256K1_HALF_ORDER
        or signature[64] not in (27, 28)
    ):
        raise ProofV3SignatureError("canary policy signature is not canonical")
    return signature


def _address(value: object, name: str) -> bytes:
    if isinstance(value, bytes):
        address = value
    elif isinstance(value, str) and value.startswith("0x") and len(value) == 42:
        try:
            address = bytes.fromhex(value[2:])
        except ValueError as exc:
            raise ProofV3SignatureError(f"{name} is not hexadecimal") from exc
    else:
        raise ProofV3SignatureError(f"{name} is not a 20-byte address")
    if len(address) != 20 or address == bytes(20):
        raise ProofV3SignatureError(f"{name} is not a nonzero 20-byte address")
    return address


@dataclass(frozen=True, slots=True)
class CanaryModelPolicyV3:
    """One signed model/profile/advertised-quantization qualification."""

    model_id: str
    execution_profile_digest: bytes
    allowed_quantizations: tuple[str, ...]
    max_hard_audit_decode_tokens: int

    def __post_init__(self) -> None:
        model_id = _identifier(self.model_id, "model_id")
        digest = bytes(self.execution_profile_digest)
        if len(digest) != 32 or digest == bytes(32):
            raise ProofV3DocumentError(
                "execution_profile_digest must be a nonzero 32-byte digest"
            )
        quants = tuple(sorted(_canonical_quant(v) for v in self.allowed_quantizations))
        if not quants or len(quants) > 32 or len(quants) != len(set(quants)):
            raise ProofV3DocumentError(
                "allowed_quantizations must be a nonempty unique inventory"
            )
        max_hard_decode = _u32(
            self.max_hard_audit_decode_tokens,
            "max_hard_audit_decode_tokens",
            minimum=1,
        )
        object.__setattr__(self, "model_id", model_id)
        object.__setattr__(self, "execution_profile_digest", digest)
        object.__setattr__(self, "allowed_quantizations", quants)
        object.__setattr__(
            self,
            "max_hard_audit_decode_tokens",
            max_hard_decode,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "allowed_quantizations": list(self.allowed_quantizations),
            "execution_profile_digest": self.execution_profile_digest.hex(),
            "max_hard_audit_decode_tokens": (
                self.max_hard_audit_decode_tokens
            ),
            "model_id": self.model_id,
        }

    @classmethod
    def from_dict(cls, value: object) -> "CanaryModelPolicyV3":
        if not isinstance(value, dict) or set(value) != {
            "allowed_quantizations",
            "execution_profile_digest",
            "max_hard_audit_decode_tokens",
            "model_id",
        }:
            raise ProofV3DocumentError(
                "canary model policy fields do not match the canonical schema"
            )
        quants = value["allowed_quantizations"]
        if not isinstance(quants, list):
            raise ProofV3DocumentError("allowed_quantizations must be a list")
        return cls(
            model_id=_identifier(value["model_id"], "model_id"),
            execution_profile_digest=_hex32(
                value["execution_profile_digest"],
                "execution_profile_digest",
            ),
            allowed_quantizations=tuple(quants),
            max_hard_audit_decode_tokens=_u32(
                value["max_hard_audit_decode_tokens"],
                "max_hard_audit_decode_tokens",
                minimum=1,
            ),
        )


@dataclass(frozen=True, slots=True)
class CanaryPolicyV3:
    """Signed obligations for the active hard auditor and light validators.

    External validators schedule only the external low-context inventory.
    The subnet-configured hard auditor schedules one advertised-context light
    obligation plus a validator-secret marked process of qualified-context
    candidates. Candidate count and hard/light marks are not public-epoch
    counters, so observing one hard reveal never proves that the remainder of
    the epoch is safe.
    """

    model_policies: tuple[CanaryModelPolicyV3, ...]
    minimum_low_context_canaries: int = 2
    minimum_advertised_context_light_canaries: int = 1
    hard_auditor_candidate_target_per_epoch: int = (
        DEFAULT_CANARY_CANDIDATE_TARGET_PER_EPOCH_V3
    )
    hard_auditor_candidate_hard_bps: int = (
        DEFAULT_CANARY_CANDIDATE_HARD_BPS_V3
    )
    low_context_min_tokens: int = 512
    low_context_max_tokens: int = 2_048
    low_context_max_decode_tokens: int = 192
    full_context_decode_reserve_tokens: int = 256
    full_context_max_decode_tokens: int = 96
    full_context_max_attempts: int = 4
    advertised_context_target_bps: int = 9_000
    hard_decode_anchor_bps: int = MIN_CANARY_HARD_DECODE_ANCHOR_BPS_V3
    hard_decode_tail_bps: int = MIN_CANARY_HARD_DECODE_TAIL_BPS_V3
    late_decode_min_output_bps: int = 9_000
    repeat_prefix_target_bps: int = (
        DEFAULT_CANARY_REPEAT_PREFIX_TARGET_BPS_V3
    )
    repeat_prefix_min_tokens: int = (
        DEFAULT_CANARY_REPEAT_PREFIX_MIN_TOKENS_V3
    )
    owner_full_context_min_prompt_bps: int = (
        DEFAULT_CANARY_OWNER_FULL_MIN_PROMPT_BPS_V3
    )
    owner_full_context_max_draw_bps: int = (
        DEFAULT_CANARY_OWNER_FULL_MAX_DRAW_BPS_V3
    )
    owner_context_sizing_abi_id: str = CANARY_OWNER_CONTEXT_SIZING_ABI_V3
    max_full_context_deferral_epochs: int = 1
    max_hard_audit_drought_epochs: int = (
        DEFAULT_CANARY_MAX_HARD_DROUGHT_EPOCHS_V3
    )
    policy_version: int = 3
    protocol_version: int = 3
    policy_abi_id: str = CANARY_POLICY_ABI_V3
    prompt_recipe_abi_id: str = CANARY_PROMPT_RECIPE_ABI_V3
    context_construction_abi_id: str = CANARY_CONTEXT_CONSTRUCTION_ABI_V3
    busy_policy_abi_id: str = CANARY_BUSY_POLICY_ABI_V3
    schedule_selection_abi_id: str = CANARY_SCHEDULE_SELECTION_ABI_V3
    hard_decode_selection_abi_id: str = (
        CANARY_HARD_DECODE_SELECTION_ABI_V3
    )
    repeat_prefix_abi_id: str = CANARY_REPEAT_PREFIX_ABI_V3

    def __post_init__(self) -> None:
        models = tuple(sorted(self.model_policies, key=lambda item: item.model_id))
        if (
            not models
            or len(models) > MAX_CANARY_POLICY_MODELS
            or len({item.model_id for item in models}) != len(models)
        ):
            raise ProofV3DocumentError(
                "canary policy model inventory is empty, duplicated, or oversized"
            )
        if _u32(self.minimum_low_context_canaries, "minimum_low_context_canaries") < 2:
            raise ProofV3DocumentError(
                "canary policy must require at least two external low-context "
                "canaries"
            )
        if (
            _u32(
                self.minimum_advertised_context_light_canaries,
                "minimum_advertised_context_light_canaries",
            )
            != 1
        ):
            raise ProofV3DocumentError(
                "canary policy must require exactly one advertised-context "
                "light canary"
            )
        candidate_target = _u32(
            self.hard_auditor_candidate_target_per_epoch,
            "hard_auditor_candidate_target_per_epoch",
            minimum=1,
        )
        if candidate_target > 32:
            raise ProofV3DocumentError(
                "hard-auditor candidate target exceeds its protocol bound"
            )
        candidate_hard_bps = _u32(
            self.hard_auditor_candidate_hard_bps,
            "hard_auditor_candidate_hard_bps",
            minimum=1,
        )
        if candidate_hard_bps >= 10_000:
            raise ProofV3DocumentError(
                "hard-auditor candidate hard rate must leave light decoys"
            )
        low_min = _u32(
            self.low_context_min_tokens,
            "low_context_min_tokens",
            minimum=1,
        )
        low_max = _u32(
            self.low_context_max_tokens,
            "low_context_max_tokens",
            minimum=low_min,
        )
        if low_max < low_min:
            raise ProofV3DocumentError(
                "low-context token range is inverted"
            )
        low_decode = _u32(
            self.low_context_max_decode_tokens,
            "low_context_max_decode_tokens",
            minimum=1,
        )
        full_reserve = _u32(
            self.full_context_decode_reserve_tokens,
            "full_context_decode_reserve_tokens",
            minimum=1,
        )
        full_decode = _u32(
            self.full_context_max_decode_tokens,
            "full_context_max_decode_tokens",
            minimum=1,
        )
        if full_decode > full_reserve:
            raise ProofV3DocumentError(
                "full-context decode length exceeds its reserved context"
            )
        attempts = _u32(
            self.full_context_max_attempts,
            "full_context_max_attempts",
            minimum=2,
        )
        if attempts > 32:
            raise ProofV3DocumentError(
                "full_context_max_attempts exceeds its protocol bound"
            )
        target_bps = _u32(
            self.advertised_context_target_bps,
            "advertised_context_target_bps",
            minimum=1,
        )
        if target_bps > 10_000:
            raise ProofV3DocumentError(
                "advertised_context_target_bps exceeds 10000"
            )
        tail_bps = _u32(
            self.hard_decode_tail_bps,
            "hard_decode_tail_bps",
            minimum=MIN_CANARY_HARD_DECODE_TAIL_BPS_V3,
        )
        anchor_bps = _u32(
            self.hard_decode_anchor_bps,
            "hard_decode_anchor_bps",
            minimum=MIN_CANARY_HARD_DECODE_ANCHOR_BPS_V3,
        )
        if anchor_bps > 10_000 or tail_bps > 10_000:
            raise ProofV3DocumentError(
                "hard-decode selection rate exceeds 10000"
            )
        if anchor_bps + tail_bps >= 10_000:
            raise ProofV3DocumentError(
                "hard-decode distribution must retain its common range"
            )
        late_output_bps = _u32(
            self.late_decode_min_output_bps,
            "late_decode_min_output_bps",
            minimum=MIN_CANARY_LATE_DECODE_OUTPUT_BPS_V3,
        )
        if late_output_bps > 10_000:
            raise ProofV3DocumentError(
                "late_decode_min_output_bps exceeds 10000"
            )
        repeat_prefix_bps = _u32(
            self.repeat_prefix_target_bps,
            "repeat_prefix_target_bps",
            minimum=1,
        )
        if repeat_prefix_bps >= 10_000:
            raise ProofV3DocumentError(
                "repeat_prefix_target_bps must leave a request-specific tail"
            )
        repeat_prefix_min = _u32(
            self.repeat_prefix_min_tokens,
            "repeat_prefix_min_tokens",
            minimum=64,
        )
        if repeat_prefix_min > low_min // 2:
            raise ProofV3DocumentError(
                "repeat_prefix_min_tokens exceeds half the minimum low context"
            )
        owner_min_prompt_bps = _u32(
            self.owner_full_context_min_prompt_bps,
            "owner_full_context_min_prompt_bps",
            minimum=1,
        )
        if owner_min_prompt_bps >= 10_000:
            raise ProofV3DocumentError(
                "owner_full_context_min_prompt_bps must leave draw range "
                "below the safe maximum"
            )
        owner_max_draw_bps = _u32(
            self.owner_full_context_max_draw_bps,
            "owner_full_context_max_draw_bps",
            minimum=MIN_CANARY_OWNER_FULL_MAX_DRAW_BPS_V3,
        )
        if owner_max_draw_bps > 10_000:
            raise ProofV3DocumentError(
                "owner_full_context_max_draw_bps exceeds 10000"
            )
        if self.owner_context_sizing_abi_id not in {
            LEGACY_CANARY_OWNER_CONTEXT_SIZING_ABI_V3,
            CANARY_OWNER_CONTEXT_SIZING_ABI_V3,
        }:
            raise ProofV3DocumentError(
                "canary owner context-sizing ABI is unsupported"
            )
        if self.max_full_context_deferral_epochs != 1:
            raise ProofV3DocumentError(
                "proof-v3 permits exactly one epoch of full-context deferral"
            )
        drought_epochs = _u32(
            self.max_hard_audit_drought_epochs,
            "max_hard_audit_drought_epochs",
            minimum=1,
        )
        if drought_epochs != DEFAULT_CANARY_MAX_HARD_DROUGHT_EPOCHS_V3:
            raise ProofV3DocumentError(
                "proof-v3 hard-audit drought window is unsupported"
            )
        if self.policy_version != 3 or self.protocol_version != 3:
            raise ProofV3DocumentError(
                "canary policy version is unsupported"
            )
        if self.policy_abi_id not in {
            LEGACY_CANARY_POLICY_ABI_V3,
            UNIFORM_CANARY_POLICY_ABI_V3,
            CANARY_POLICY_ABI_V3,
        }:
            raise ProofV3DocumentError("canary policy ABI is unsupported")
        if (
            self.policy_abi_id == LEGACY_CANARY_POLICY_ABI_V3
            and (
                owner_min_prompt_bps
                != LEGACY_CANARY_OWNER_FULL_MIN_PROMPT_BPS_V3
                or owner_max_draw_bps != 10_000
            )
        ):
            raise ProofV3DocumentError(
                "legacy canary policy must retain fixed context sizing"
            )
        if (
            self.policy_abi_id == UNIFORM_CANARY_POLICY_ABI_V3
            and self.owner_context_sizing_abi_id
            != LEGACY_CANARY_OWNER_CONTEXT_SIZING_ABI_V3
        ):
            raise ProofV3DocumentError(
                "canary policy v6 must retain uniform owner context sizing"
            )
        if (
            self.policy_abi_id == CANARY_POLICY_ABI_V3
            and self.owner_context_sizing_abi_id
            != CANARY_OWNER_CONTEXT_SIZING_ABI_V3
        ):
            raise ProofV3DocumentError(
                "canary policy v7 must use geometric owner context sizing"
            )
        if self.prompt_recipe_abi_id != CANARY_PROMPT_RECIPE_ABI_V3:
            raise ProofV3DocumentError("canary prompt recipe ABI is unsupported")
        if (
            self.context_construction_abi_id
            != CANARY_CONTEXT_CONSTRUCTION_ABI_V3
        ):
            raise ProofV3DocumentError(
                "canary context construction ABI is unsupported"
            )
        if self.busy_policy_abi_id != CANARY_BUSY_POLICY_ABI_V3:
            raise ProofV3DocumentError("canary busy policy ABI is unsupported")
        if self.schedule_selection_abi_id != CANARY_SCHEDULE_SELECTION_ABI_V3:
            raise ProofV3DocumentError(
                "canary schedule-selection ABI is unsupported"
            )
        if (
            self.hard_decode_selection_abi_id
            != CANARY_HARD_DECODE_SELECTION_ABI_V3
        ):
            raise ProofV3DocumentError(
                "canary hard-decode selection ABI is unsupported"
            )
        if self.repeat_prefix_abi_id != CANARY_REPEAT_PREFIX_ABI_V3:
            raise ProofV3DocumentError(
                "canary repeat-prefix ABI is unsupported"
            )
        for value, name in (
            (self.policy_abi_id, "policy_abi_id"),
            (self.prompt_recipe_abi_id, "prompt_recipe_abi_id"),
            (self.context_construction_abi_id, "context_construction_abi_id"),
            (self.busy_policy_abi_id, "busy_policy_abi_id"),
            (
                self.schedule_selection_abi_id,
                "schedule_selection_abi_id",
            ),
            (
                self.hard_decode_selection_abi_id,
                "hard_decode_selection_abi_id",
            ),
            (
                self.owner_context_sizing_abi_id,
                "owner_context_sizing_abi_id",
            ),
            (self.repeat_prefix_abi_id, "repeat_prefix_abi_id"),
        ):
            _identifier(value, name)
        object.__setattr__(self, "model_policies", models)
        object.__setattr__(self, "low_context_min_tokens", low_min)
        object.__setattr__(self, "low_context_max_tokens", low_max)
        object.__setattr__(self, "low_context_max_decode_tokens", low_decode)
        object.__setattr__(
            self,
            "full_context_decode_reserve_tokens",
            full_reserve,
        )
        object.__setattr__(
            self,
            "full_context_max_decode_tokens",
            full_decode,
        )
        object.__setattr__(
            self,
            "hard_auditor_candidate_target_per_epoch",
            candidate_target,
        )
        object.__setattr__(
            self,
            "hard_auditor_candidate_hard_bps",
            candidate_hard_bps,
        )
        object.__setattr__(
            self,
            "advertised_context_target_bps",
            target_bps,
        )
        object.__setattr__(self, "hard_decode_tail_bps", tail_bps)
        object.__setattr__(self, "hard_decode_anchor_bps", anchor_bps)
        object.__setattr__(
            self,
            "late_decode_min_output_bps",
            late_output_bps,
        )
        object.__setattr__(
            self,
            "repeat_prefix_target_bps",
            repeat_prefix_bps,
        )
        object.__setattr__(
            self,
            "repeat_prefix_min_tokens",
            repeat_prefix_min,
        )
        object.__setattr__(
            self,
            "owner_full_context_min_prompt_bps",
            owner_min_prompt_bps,
        )
        object.__setattr__(
            self,
            "owner_full_context_max_draw_bps",
            owner_max_draw_bps,
        )

    def to_dict(self) -> dict[str, object]:
        result = {
            "advertised_context_target_bps": (
                self.advertised_context_target_bps
            ),
            "busy_policy_abi_id": self.busy_policy_abi_id,
            "context_construction_abi_id": self.context_construction_abi_id,
            "full_context_decode_reserve_tokens": (
                self.full_context_decode_reserve_tokens
            ),
            "full_context_max_attempts": self.full_context_max_attempts,
            "full_context_max_decode_tokens": self.full_context_max_decode_tokens,
            "hard_auditor_candidate_hard_bps": (
                self.hard_auditor_candidate_hard_bps
            ),
            "hard_auditor_candidate_target_per_epoch": (
                self.hard_auditor_candidate_target_per_epoch
            ),
            "hard_decode_anchor_bps": self.hard_decode_anchor_bps,
            "hard_decode_selection_abi_id": (
                self.hard_decode_selection_abi_id
            ),
            "hard_decode_tail_bps": self.hard_decode_tail_bps,
            "late_decode_min_output_bps": self.late_decode_min_output_bps,
            "low_context_max_decode_tokens": self.low_context_max_decode_tokens,
            "low_context_max_tokens": self.low_context_max_tokens,
            "low_context_min_tokens": self.low_context_min_tokens,
            "max_full_context_deferral_epochs": (
                self.max_full_context_deferral_epochs
            ),
            "max_hard_audit_drought_epochs": (
                self.max_hard_audit_drought_epochs
            ),
            "minimum_advertised_context_light_canaries": (
                self.minimum_advertised_context_light_canaries
            ),
            "minimum_low_context_canaries": self.minimum_low_context_canaries,
            "model_policies": [item.to_dict() for item in self.model_policies],
            "policy_abi_id": self.policy_abi_id,
            "policy_version": self.policy_version,
            "prompt_recipe_abi_id": self.prompt_recipe_abi_id,
            "protocol_version": self.protocol_version,
            "repeat_prefix_abi_id": self.repeat_prefix_abi_id,
            "repeat_prefix_min_tokens": self.repeat_prefix_min_tokens,
            "repeat_prefix_target_bps": self.repeat_prefix_target_bps,
            "schedule_selection_abi_id": self.schedule_selection_abi_id,
        }
        if self.policy_abi_id in {
            UNIFORM_CANARY_POLICY_ABI_V3,
            CANARY_POLICY_ABI_V3,
        }:
            result.update(
                {
                    "owner_context_sizing_abi_id": (
                        self.owner_context_sizing_abi_id
                    ),
                    "owner_full_context_max_draw_bps": (
                        self.owner_full_context_max_draw_bps
                    ),
                    "owner_full_context_min_prompt_bps": (
                        self.owner_full_context_min_prompt_bps
                    ),
                }
            )
        return result

    def canonical_bytes(self) -> bytes:
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")

    def digest(self) -> bytes:
        return hashlib.sha256(_POLICY_DOMAIN + self.canonical_bytes()).digest()

    def model_policy(self, model_id: str) -> CanaryModelPolicyV3 | None:
        for item in self.model_policies:
            if item.model_id == model_id:
                return item
        return None

    @classmethod
    def from_dict(cls, value: object) -> "CanaryPolicyV3":
        legacy_keys = {
            "advertised_context_target_bps",
            "busy_policy_abi_id",
            "context_construction_abi_id",
            "full_context_decode_reserve_tokens",
            "full_context_max_attempts",
            "full_context_max_decode_tokens",
            "hard_auditor_candidate_hard_bps",
            "hard_auditor_candidate_target_per_epoch",
            "hard_decode_anchor_bps",
            "hard_decode_selection_abi_id",
            "hard_decode_tail_bps",
            "late_decode_min_output_bps",
            "low_context_max_decode_tokens",
            "low_context_max_tokens",
            "low_context_min_tokens",
            "max_full_context_deferral_epochs",
            "max_hard_audit_drought_epochs",
            "minimum_advertised_context_light_canaries",
            "minimum_low_context_canaries",
            "model_policies",
            "policy_abi_id",
            "policy_version",
            "prompt_recipe_abi_id",
            "protocol_version",
            "repeat_prefix_abi_id",
            "repeat_prefix_min_tokens",
            "repeat_prefix_target_bps",
            "schedule_selection_abi_id",
        }
        if not isinstance(value, dict):
            raise ProofV3DocumentError(
                "canary policy fields do not match the canonical schema"
            )
        policy_abi_id = value.get("policy_abi_id")
        if policy_abi_id == LEGACY_CANARY_POLICY_ABI_V3:
            expected_keys = legacy_keys
        elif policy_abi_id in {
            UNIFORM_CANARY_POLICY_ABI_V3,
            CANARY_POLICY_ABI_V3,
        }:
            expected_keys = legacy_keys | {
                "owner_context_sizing_abi_id",
                "owner_full_context_max_draw_bps",
                "owner_full_context_min_prompt_bps",
            }
        else:
            raise ProofV3DocumentError("canary policy ABI is unsupported")
        if set(value) != expected_keys:
            raise ProofV3DocumentError(
                "canary policy fields do not match the canonical schema"
            )
        models = value["model_policies"]
        if not isinstance(models, list):
            raise ProofV3DocumentError("model_policies must be a list")
        return cls(
            model_policies=tuple(
                CanaryModelPolicyV3.from_dict(item) for item in models
            ),
            minimum_low_context_canaries=_u32(
                value["minimum_low_context_canaries"],
                "minimum_low_context_canaries",
            ),
            minimum_advertised_context_light_canaries=_u32(
                value["minimum_advertised_context_light_canaries"],
                "minimum_advertised_context_light_canaries",
            ),
            hard_auditor_candidate_target_per_epoch=_u32(
                value["hard_auditor_candidate_target_per_epoch"],
                "hard_auditor_candidate_target_per_epoch",
            ),
            hard_auditor_candidate_hard_bps=_u32(
                value["hard_auditor_candidate_hard_bps"],
                "hard_auditor_candidate_hard_bps",
            ),
            low_context_min_tokens=_u32(
                value["low_context_min_tokens"],
                "low_context_min_tokens",
            ),
            low_context_max_tokens=_u32(
                value["low_context_max_tokens"],
                "low_context_max_tokens",
            ),
            low_context_max_decode_tokens=_u32(
                value["low_context_max_decode_tokens"],
                "low_context_max_decode_tokens",
            ),
            full_context_decode_reserve_tokens=_u32(
                value["full_context_decode_reserve_tokens"],
                "full_context_decode_reserve_tokens",
            ),
            full_context_max_decode_tokens=_u32(
                value["full_context_max_decode_tokens"],
                "full_context_max_decode_tokens",
            ),
            hard_decode_selection_abi_id=_identifier(
                value["hard_decode_selection_abi_id"],
                "hard_decode_selection_abi_id",
            ),
            hard_decode_anchor_bps=_u32(
                value["hard_decode_anchor_bps"],
                "hard_decode_anchor_bps",
            ),
            hard_decode_tail_bps=_u32(
                value["hard_decode_tail_bps"],
                "hard_decode_tail_bps",
            ),
            late_decode_min_output_bps=_u32(
                value["late_decode_min_output_bps"],
                "late_decode_min_output_bps",
            ),
            repeat_prefix_target_bps=_u32(
                value["repeat_prefix_target_bps"],
                "repeat_prefix_target_bps",
            ),
            repeat_prefix_min_tokens=_u32(
                value["repeat_prefix_min_tokens"],
                "repeat_prefix_min_tokens",
            ),
            full_context_max_attempts=_u32(
                value["full_context_max_attempts"],
                "full_context_max_attempts",
            ),
            advertised_context_target_bps=_u32(
                value["advertised_context_target_bps"],
                "advertised_context_target_bps",
            ),
            max_full_context_deferral_epochs=_u32(
                value["max_full_context_deferral_epochs"],
                "max_full_context_deferral_epochs",
            ),
            max_hard_audit_drought_epochs=_u32(
                value["max_hard_audit_drought_epochs"],
                "max_hard_audit_drought_epochs",
            ),
            owner_full_context_min_prompt_bps=(
                _u32(
                    value["owner_full_context_min_prompt_bps"],
                    "owner_full_context_min_prompt_bps",
                )
                if policy_abi_id
                in {UNIFORM_CANARY_POLICY_ABI_V3, CANARY_POLICY_ABI_V3}
                else LEGACY_CANARY_OWNER_FULL_MIN_PROMPT_BPS_V3
            ),
            owner_full_context_max_draw_bps=(
                _u32(
                    value["owner_full_context_max_draw_bps"],
                    "owner_full_context_max_draw_bps",
                )
                if policy_abi_id
                in {UNIFORM_CANARY_POLICY_ABI_V3, CANARY_POLICY_ABI_V3}
                else 10_000
            ),
            owner_context_sizing_abi_id=(
                _identifier(
                    value["owner_context_sizing_abi_id"],
                    "owner_context_sizing_abi_id",
                )
                if policy_abi_id
                in {UNIFORM_CANARY_POLICY_ABI_V3, CANARY_POLICY_ABI_V3}
                else CANARY_OWNER_CONTEXT_SIZING_ABI_V3
            ),
            policy_version=_u32(value["policy_version"], "policy_version"),
            protocol_version=_u32(
                value["protocol_version"],
                "protocol_version",
            ),
            policy_abi_id=_identifier(value["policy_abi_id"], "policy_abi_id"),
            prompt_recipe_abi_id=_identifier(
                value["prompt_recipe_abi_id"],
                "prompt_recipe_abi_id",
            ),
            context_construction_abi_id=_identifier(
                value["context_construction_abi_id"],
                "context_construction_abi_id",
            ),
            busy_policy_abi_id=_identifier(
                value["busy_policy_abi_id"],
                "busy_policy_abi_id",
            ),
            schedule_selection_abi_id=_identifier(
                value["schedule_selection_abi_id"],
                "schedule_selection_abi_id",
            ),
            repeat_prefix_abi_id=_identifier(
                value["repeat_prefix_abi_id"],
                "repeat_prefix_abi_id",
            ),
        )


def _unique_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ProofV3DocumentError("JSON object contains duplicate keys")
        result[key] = value
    return result


def _reject_constant(_value: str) -> None:
    raise ProofV3DocumentError("JSON constants are not allowed")


@dataclass(frozen=True, slots=True)
class SignedCanaryPolicyDocumentV3:
    policy: CanaryPolicyV3
    signatures: tuple[bytes, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.policy, CanaryPolicyV3):
            raise ProofV3DocumentError("canary policy has an unexpected type")
        try:
            signatures = tuple(sorted(_signature(value) for value in self.signatures))
        except ProofV3SignatureError as exc:
            raise ProofV3DocumentError(
                "canary policy signatures are not canonical"
            ) from exc
        if (
            not signatures
            or len(signatures) > MAX_CANARY_POLICY_SIGNATURES
            or len(signatures) != len(set(signatures))
        ):
            raise ProofV3DocumentError(
                "canary policy signature inventory is invalid"
            )
        object.__setattr__(self, "signatures", signatures)

    def to_dict(self) -> dict[str, object]:
        return {
            "policy": self.policy.to_dict(),
            "signatures": [value.hex() for value in self.signatures],
        }

    def canonical_json_bytes(self) -> bytes:
        encoded = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
        if len(encoded) > MAX_CANARY_POLICY_DOCUMENT_BYTES:
            raise ProofV3DocumentError(
                "canary policy document exceeds its protocol limit"
            )
        return encoded

    @classmethod
    def from_json_bytes(cls, encoded: bytes) -> "SignedCanaryPolicyDocumentV3":
        if (
            not isinstance(encoded, bytes)
            or not encoded
            or len(encoded) > MAX_CANARY_POLICY_DOCUMENT_BYTES
        ):
            raise ProofV3DocumentError(
                "canary policy document length is out of range"
            )
        try:
            value = json.loads(
                encoded.decode("ascii"),
                object_pairs_hook=_unique_pairs,
                parse_constant=_reject_constant,
            )
        except ProofV3DocumentError:
            raise
        except (RecursionError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProofV3DocumentError(
                "canary policy document is not valid JSON"
            ) from exc
        if not isinstance(value, dict) or set(value) != {"policy", "signatures"}:
            raise ProofV3DocumentError(
                "canary policy document fields do not match the canonical schema"
            )
        signatures = value["signatures"]
        if not isinstance(signatures, list):
            raise ProofV3DocumentError("canary policy signatures must be a list")
        result = cls(
            policy=CanaryPolicyV3.from_dict(value["policy"]),
            signatures=tuple(_signature(item) for item in signatures),
        )
        if result.canonical_json_bytes() != encoded:
            raise ProofV3DocumentError(
                "canary policy document is not canonical JSON"
            )
        return result


def load_signed_canary_policy_document_v3(
    path: str | Path,
) -> SignedCanaryPolicyDocumentV3:
    source = Path(path).expanduser().resolve()
    try:
        size = source.stat().st_size
        if not 0 < size <= MAX_CANARY_POLICY_DOCUMENT_BYTES:
            raise ProofV3DocumentError(
                "canary policy document file size is out of range"
            )
        encoded = source.read_bytes()
    except ProofV3DocumentError:
        raise
    except OSError as exc:
        raise ProofV3DocumentError(
            f"cannot read canary policy document: {source}"
        ) from exc
    if len(encoded) != size:
        raise ProofV3DocumentError(
            "canary policy document file size changed while loading"
        )
    return SignedCanaryPolicyDocumentV3.from_json_bytes(encoded)


def verify_signed_canary_policy_v3(
    document: SignedCanaryPolicyDocumentV3,
    *,
    expected_authority_signers: Collection[str | bytes],
    authority_threshold: int,
) -> tuple[str, ...]:
    if not isinstance(document, SignedCanaryPolicyDocumentV3):
        raise ProofV3VerificationError(
            "signed canary policy has an unexpected type"
        )
    expected_signers = tuple(expected_authority_signers)
    authorities = frozenset(
        _address(value, "expected canary policy authority")
        for value in expected_signers
    )
    if (
        not authorities
        or len(authorities) != len(expected_signers)
        or type(authority_threshold) is not int
        or not 1 <= authority_threshold <= len(authorities)
    ):
        raise ProofV3SignatureError(
            "canary policy authority set or threshold is invalid"
        )
    try:
        from eth_account import Account
        from eth_account.messages import encode_defunct
    except ImportError as exc:
        raise ProofV3DocumentError(
            "canary policy signature verification requires chain dependencies"
        ) from exc
    signable = encode_defunct(primitive=document.policy.digest())
    recovered: set[bytes] = set()
    for value in document.signatures:
        try:
            signer = _address(
                Account.recover_message(signable, signature=value),
                "recovered canary policy authority",
            )
        except Exception as exc:
            raise ProofV3SignatureError(
                "canary policy signature recovery failed"
            ) from exc
        if signer not in authorities or signer in recovered:
            raise ProofV3SignatureError(
                "canary policy signature authority is unexpected or duplicated"
            )
        recovered.add(signer)
    if len(recovered) < authority_threshold:
        raise ProofV3SignatureError(
            "canary policy authority threshold was not met"
        )
    return tuple(f"0x{value.hex()}" for value in sorted(recovered))


def qualify_canary_policy_v3(
    document: SignedCanaryPolicyDocumentV3,
    *,
    qualified_releases: Mapping[str, object],
    expected_authority_signers: Collection[str | bytes],
    authority_threshold: int,
) -> CanaryPolicyV3:
    """Authenticate the policy and bind every configured release profile."""

    verify_signed_canary_policy_v3(
        document,
        expected_authority_signers=expected_authority_signers,
        authority_threshold=authority_threshold,
    )
    for model_id, release in qualified_releases.items():
        item = document.policy.model_policy(model_id)
        if item is None:
            raise ProofV3VerificationError(
                f"canary policy has no rule for configured model {model_id}"
            )
        try:
            digest = release.qualified_profile.profile.digest()
        except Exception as exc:
            raise ProofV3VerificationError(
                "configured proof-v3 release has an invalid profile"
            ) from exc
        if item.execution_profile_digest != digest:
            raise ProofV3VerificationError(
                f"canary policy profile does not match configured model {model_id}"
            )
        try:
            profile = release.qualified_profile.profile
            profile_decode_cap = int(profile.max_verified_decode_tokens)
            profile_context_cap = int(profile.max_verified_context_tokens)
        except Exception as exc:
            raise ProofV3VerificationError(
                "configured proof-v3 release has invalid decode limits"
            ) from exc
        if item.max_hard_audit_decode_tokens != profile_decode_cap:
            raise ProofV3VerificationError(
                f"canary policy hard-decode cap does not match configured "
                f"model {model_id}"
            )
        if item.max_hard_audit_decode_tokens >= profile_context_cap:
            raise ProofV3VerificationError(
                f"canary policy hard-decode cap leaves no prompt capacity for "
                f"configured model {model_id}"
            )
    return document.policy


__all__ = [
    "CANARY_BUSY_POLICY_ABI_V3",
    "CANARY_CONTEXT_CONSTRUCTION_ABI_V3",
    "CANARY_HARD_DECODE_ANCHORS_V3",
    "MAX_CANARY_FULL_PAIR_HOLD_SECONDS_V3",
    "CANARY_HARD_DECODE_SELECTION_ABI_V3",
    "CANARY_OWNER_CONTEXT_SIZING_ABI_V3",
    "CANARY_POLICY_ABI_V3",
    "CANARY_PROMPT_RECIPE_ABI_V3",
    "CANARY_REPEAT_PREFIX_ABI_V3",
    "DEFAULT_CANARY_MAX_HARD_DROUGHT_EPOCHS_V3",
    "DEFAULT_CANARY_OWNER_FULL_MAX_DRAW_BPS_V3",
    "DEFAULT_CANARY_OWNER_FULL_MIN_PROMPT_BPS_V3",
    "DEFAULT_CANARY_REPEAT_PREFIX_MIN_TOKENS_V3",
    "DEFAULT_CANARY_REPEAT_PREFIX_TARGET_BPS_V3",
    "MIN_CANARY_HARD_DECODE_ANCHOR_BPS_V3",
    "MIN_CANARY_HARD_DECODE_TAIL_BPS_V3",
    "MIN_CANARY_LATE_DECODE_OUTPUT_BPS_V3",
    "MIN_CANARY_OWNER_FULL_MAX_DRAW_BPS_V3",
    "LEGACY_CANARY_POLICY_ABI_V3",
    "LEGACY_CANARY_OWNER_CONTEXT_SIZING_ABI_V3",
    "UNIFORM_CANARY_POLICY_ABI_V3",
    "CanaryModelPolicyV3",
    "CanaryPolicyV3",
    "SignedCanaryPolicyDocumentV3",
    "load_signed_canary_policy_document_v3",
    "qualify_canary_policy_v3",
    "verify_signed_canary_policy_v3",
]
