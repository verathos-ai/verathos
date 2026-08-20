"""Nonce-selected bounded GDN replay corridors along the decode axis."""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass

from verallm.proof_v3.errors import ProofV3Error

GDN_DECODE_CORRIDOR_ABI_V3 = (
    "gdn.decode_corridor.checkpoints.projection_sample.v2"
)
GDN_PROJECTION_BINDING_ROWS_PER_WINDOW_V3 = 4

_SELECTION_DOMAIN = b"VERATHOS/PROOF_V3/GDN_DECODE_CORRIDOR/SELECT/V1"
_PROJECTION_BINDING_DOMAIN = (
    b"VERATHOS/PROOF_V3/GDN_DECODE_CORRIDOR/PROJECTION_BINDING/V1"
)
_MAX_DECODE_TOKENS = (1 << 31) - 1
_MAX_STRIDE = 4096

__all__ = [
    "GDN_DECODE_CORRIDOR_ABI_V3",
    "GDN_PROJECTION_BINDING_ROWS_PER_WINDOW_V3",
    "GdnDecodeCorridorPlanV3",
    "derive_gdn_decode_corridor_from_selection_seed_v3",
    "derive_gdn_decode_corridor_for_challenge_v3",
    "derive_gdn_decode_corridor_plan_v3",
    "derive_gdn_projection_binding_positions_v3",
    "gdn_decode_checkpoint_offsets_v3",
]


def _positive_int(value: object, name: str, maximum: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 < value <= maximum
    ):
        raise ProofV3Error(f"GDN decode-corridor {name} is out of range")
    return value


def gdn_decode_checkpoint_offsets_v3(
    *,
    decode_token_count: int,
    checkpoint_stride: int,
) -> tuple[int, ...]:
    """Return state-boundary offsets in forwarded decode rows.

    Prefill returns the first output token, so a response with ``N`` output
    tokens forwards ``N - 1`` decode rows.  Offset zero is the authenticated
    post-prefill state.  The final state is always included, even when the
    suffix is shorter than or not divisible by the signed stride.
    """

    decode = _positive_int(
        decode_token_count,
        "token count",
        _MAX_DECODE_TOKENS,
    )
    stride = _positive_int(
        checkpoint_stride,
        "checkpoint stride",
        _MAX_STRIDE,
    )
    forwarded = decode - 1
    if forwarded == 0:
        return (0,)
    offsets = list(range(0, forwarded + 1, stride))
    if offsets[-1] != forwarded:
        offsets.append(forwarded)
    return tuple(offsets)


def _unbiased_index(seed: bytes, population: int) -> int:
    bound = (1 << 64) - ((1 << 64) % population)
    for counter in range(1 << 16):
        digest = hashlib.sha256(
            seed + struct.pack("<I", counter)
        ).digest()
        for offset in range(0, len(digest), 8):
            value = int.from_bytes(digest[offset : offset + 8], "little")
            if value < bound:
                return value % population
    raise ProofV3Error("GDN decode-corridor selection was exhausted")


def derive_gdn_projection_binding_positions_v3(
    *,
    selection_seed: bytes,
    layer_index: int,
    sequence_positions,
    count: int = GDN_PROJECTION_BINDING_ROWS_PER_WINDOW_V3,
) -> tuple[int, ...]:
    """Select rows that bind a full replay window to registered projections.

    The recurrence consumes every committed row between authenticated state
    checkpoints. Projection equations are required only at this independent
    post-commitment sample. A miner cannot target the sample before freezing
    the runtime roots, while proof cost stays independent of checkpoint
    stride.
    """

    if (
        not isinstance(selection_seed, bytes)
        or len(selection_seed) != 32
        or isinstance(layer_index, bool)
        or not isinstance(layer_index, int)
        or layer_index < 0
        or layer_index >= 1 << 32
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count <= 0
        or count > _MAX_STRIDE
    ):
        raise ProofV3Error(
            "GDN projection-binding selection is malformed"
        )
    try:
        positions = tuple(int(position) for position in sequence_positions)
    except (TypeError, ValueError) as exc:
        raise ProofV3Error(
            "GDN projection-binding positions are malformed"
        ) from exc
    if (
        not positions
        or positions != tuple(sorted(set(positions)))
        or any(
            position < 0 or position > _MAX_DECODE_TOKENS
            for position in positions
        )
    ):
        raise ProofV3Error(
            "GDN projection-binding positions are malformed"
        )
    sample_count = min(count, len(positions))
    if sample_count == len(positions):
        return positions
    seed = hashlib.sha256(
        _PROJECTION_BINDING_DOMAIN
        + selection_seed
        + struct.pack("<III", layer_index, sample_count, len(positions))
        + b"".join(struct.pack("<I", position) for position in positions)
    ).digest()
    bound = (1 << 64) - ((1 << 64) % len(positions))
    chosen: set[int] = set()
    for counter in range(1 << 16):
        digest = hashlib.sha256(
            seed + struct.pack("<I", counter)
        ).digest()
        for offset in range(0, len(digest), 8):
            value = int.from_bytes(digest[offset : offset + 8], "little")
            if value < bound:
                chosen.add(positions[value % len(positions)])
                if len(chosen) == sample_count:
                    return tuple(sorted(chosen))
    raise ProofV3Error(
        "GDN projection-binding selection was exhausted"
    )


@dataclass(frozen=True, slots=True)
class GdnDecodeCorridorPlanV3:
    """One post-commitment replay window selected from committed boundaries."""

    checkpoint_offsets: tuple[int, ...]
    window_index: int
    context_token_count: int

    def __post_init__(self) -> None:
        offsets = tuple(self.checkpoint_offsets)
        context = _positive_int(
            self.context_token_count,
            "context token count",
            _MAX_DECODE_TOKENS,
        )
        if (
            len(offsets) < 2
            or offsets[0] != 0
            or offsets != tuple(sorted(set(offsets)))
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= value <= _MAX_DECODE_TOKENS
                for value in offsets
            )
            or isinstance(self.window_index, bool)
            or not isinstance(self.window_index, int)
            or not 0 <= self.window_index < len(offsets) - 1
        ):
            raise ProofV3Error("GDN decode-corridor plan is malformed")
        object.__setattr__(self, "checkpoint_offsets", offsets)
        object.__setattr__(self, "context_token_count", context)

    @property
    def start_checkpoint_row(self) -> int:
        return self.window_index

    @property
    def end_checkpoint_row(self) -> int:
        return self.window_index + 1

    @property
    def start_forwarded_row(self) -> int:
        return self.checkpoint_offsets[self.window_index]

    @property
    def end_forwarded_row(self) -> int:
        return self.checkpoint_offsets[self.window_index + 1]

    @property
    def forwarded_row_count(self) -> int:
        return self.end_forwarded_row - self.start_forwarded_row

    @property
    def sequence_positions(self) -> tuple[int, ...]:
        return tuple(
            range(
                self.context_token_count + self.start_forwarded_row,
                self.context_token_count + self.end_forwarded_row,
            )
        )


def derive_gdn_decode_corridor_from_selection_seed_v3(
    *,
    selection_seed: bytes,
    context_token_count: int,
    decode_token_count: int,
    checkpoint_stride: int,
    committed_checkpoint_offsets: tuple[int, ...],
) -> GdnDecodeCorridorPlanV3:
    """Derive the exact window from an already transcript-bound seed."""

    if not isinstance(selection_seed, bytes) or len(selection_seed) != 32:
        raise ProofV3Error(
            "GDN decode-corridor selection seed is malformed"
        )
    context = _positive_int(
        context_token_count,
        "context token count",
        _MAX_DECODE_TOKENS,
    )
    decode = _positive_int(
        decode_token_count,
        "token count",
        _MAX_DECODE_TOKENS,
    )
    stride = _positive_int(
        checkpoint_stride,
        "checkpoint stride",
        _MAX_STRIDE,
    )
    expected = gdn_decode_checkpoint_offsets_v3(
        decode_token_count=decode,
        checkpoint_stride=stride,
    )
    try:
        supplied = tuple(committed_checkpoint_offsets)
    except TypeError as exc:
        raise ProofV3Error(
            "GDN decode-corridor checkpoint inventory is malformed"
        ) from exc
    if supplied != expected:
        raise ProofV3Error(
            "GDN decode-corridor checkpoint inventory is not canonical"
        )
    if len(expected) < 2:
        raise ProofV3Error(
            "GDN decode-corridor hard audit requires a forwarded row"
        )
    seed = hashlib.sha256(
        _SELECTION_DOMAIN
        + selection_seed
        + struct.pack("<III", context, decode, stride)
        + struct.pack("<I", len(expected))
        + b"".join(struct.pack("<I", value) for value in expected)
    ).digest()
    return GdnDecodeCorridorPlanV3(
        checkpoint_offsets=expected,
        window_index=_unbiased_index(seed, len(expected) - 1),
        context_token_count=context,
    )


def derive_gdn_decode_corridor_plan_v3(
    *,
    validator_nonce: bytes,
    capture_chain_digest: bytes,
    profile_digest: bytes,
    context_token_count: int,
    decode_token_count: int,
    checkpoint_stride: int,
    committed_checkpoint_offsets: tuple[int, ...],
) -> GdnDecodeCorridorPlanV3:
    """Replay the validator's unbiased window draw after roots are frozen."""

    for value, name in (
        (validator_nonce, "validator nonce"),
        (capture_chain_digest, "capture-chain digest"),
        (profile_digest, "profile digest"),
    ):
        if not isinstance(value, bytes) or len(value) != 32:
            raise ProofV3Error(f"GDN decode-corridor {name} is malformed")
    selection_seed = hashlib.sha256(
        _SELECTION_DOMAIN
        + validator_nonce
        + capture_chain_digest
        + profile_digest
    ).digest()
    return derive_gdn_decode_corridor_from_selection_seed_v3(
        selection_seed=selection_seed,
        context_token_count=context_token_count,
        decode_token_count=decode_token_count,
        checkpoint_stride=checkpoint_stride,
        committed_checkpoint_offsets=committed_checkpoint_offsets,
    )


def derive_gdn_decode_corridor_for_challenge_v3(
    *,
    challenge,
    semantics,
) -> GdnDecodeCorridorPlanV3 | None:
    """Return the signed checkpoint window, or ``None`` for legacy semantics."""

    stride = int(getattr(semantics, "decode_checkpoint_stride", 0))
    if stride == 0:
        return None
    try:
        context = int(challenge.context_token_count)
        decode = int(challenge.decode_token_count)
        seed = challenge.selection_seed
    except (AttributeError, TypeError, ValueError) as exc:
        raise ProofV3Error(
            "GDN decode-corridor challenge is malformed"
        ) from exc
    offsets = gdn_decode_checkpoint_offsets_v3(
        decode_token_count=decode,
        checkpoint_stride=stride,
    )
    return derive_gdn_decode_corridor_from_selection_seed_v3(
        selection_seed=seed,
        context_token_count=context,
        decode_token_count=decode,
        checkpoint_stride=stride,
        committed_checkpoint_offsets=offsets,
    )
