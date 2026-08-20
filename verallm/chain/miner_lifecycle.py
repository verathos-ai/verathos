"""MinerRegistry slot-lifecycle planning and the GGUF mesh eligibility gate.

Pure contract-semantics logic shared by the operator registration script and
the mesh deploy pipeline. Lives under verallm/chain (which ships in the
public release tree) so GGUF-mesh miners can register on-chain; the historic
home, scripts/register_miner_onchain.py, is excluded from the public repo
and now re-exports from here.

Registration is idempotent for an exact active entry, understands that the
contract reactivates an expired/inactive matching tuple in place, and
renewal requires both an index and the expected full tuple so a stale index
is never renewed blindly.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Sequence


class LifecycleRefusal(ValueError):
    """Raised when a registry write would be ambiguous or unsafe."""


MESH_QUANT_PREFIX = "gguf_mesh"
MAX_MESH_MODEL_LAYERS = 1_000_000


@dataclass(frozen=True)
class RegistrationTarget:
    model_id: str
    endpoint: str
    model_spec_ref: bytes
    quant: str
    max_context_len: int


@dataclass(frozen=True)
class LifecyclePlan:
    action: str
    predicted_index: int
    reason: str


def is_mesh_target(target: RegistrationTarget) -> bool:
    """Return whether the MinerRegistry quant marks a private GGUF mesh."""

    return str(target.quant or "").strip().lower().startswith(MESH_QUANT_PREFIX)


def expected_mesh_model_quant(target: RegistrationTarget) -> str:
    """Map ``gguf_mesh_<scheme>`` to the ModelSpec ``gguf_<scheme>`` token."""

    marker = str(target.quant or "").strip().lower()
    suffix = marker[len(MESH_QUANT_PREFIX) :]
    if not suffix.startswith("_") or len(suffix) == 1:
        raise LifecycleRefusal(
            "mesh quant must include an explicit GGUF scheme, for example "
            "gguf_mesh_q4_k_m"
        )
    return "gguf" + suffix


def require_mesh_commitment(model_spec, field_name: str) -> bytes:
    """Return one non-zero bytes32 verifier trust anchor."""

    try:
        digest = bytes(getattr(model_spec, field_name))
    except Exception as exc:
        raise LifecycleRefusal(
            f"ModelRegistry {field_name} is not a bytes32 commitment"
        ) from exc
    if len(digest) != 32:
        raise LifecycleRefusal(
            f"ModelRegistry {field_name} must be exactly 32 bytes"
        )
    if digest == b"\x00" * 32:
        raise LifecycleRefusal(
            f"ModelRegistry {field_name} must be a non-zero commitment"
        )
    return digest


def validate_mesh_model_registry_eligibility(
    target: RegistrationTarget,
    model_spec,
) -> str:
    """Validate the on-chain trust anchors required by mesh validators.

    This intentionally applies only to ``gguf_mesh_*`` MinerRegistry targets.
    Legacy vLLM registrations retain their existing compatibility behavior.
    The returned value is the exact GGUF ModelSpec quantization token.
    """

    if not is_mesh_target(target):
        raise LifecycleRefusal("GGUF mesh eligibility requires a mesh target")
    if model_spec is None:
        raise LifecycleRefusal(
            f"model {target.model_id!r} is absent from ModelRegistry; "
            "validators cannot verify this mesh"
        )
    if str(getattr(model_spec, "model_id", "")) != target.model_id:
        raise LifecycleRefusal(
            "ModelRegistry returned a ModelSpec whose model_id does not match "
            "the MinerRegistry target"
        )

    expected_quant = expected_mesh_model_quant(target)
    actual_quant = str(getattr(model_spec, "quant_mode", "") or "").strip().lower()
    if actual_quant != expected_quant:
        raise LifecycleRefusal(
            "ModelRegistry quant_mode does not match the MinerRegistry GGUF "
            f"scheme (expected {expected_quant!r}, found {actual_quant!r})"
        )

    require_mesh_commitment(model_spec, "weight_file_hash")
    require_mesh_commitment(model_spec, "weight_merkle_root")
    require_mesh_commitment(model_spec, "tokenizer_hash")

    layer_roots = getattr(model_spec, "weight_block_merkle_roots", None)
    if not isinstance(layer_roots, (list, tuple)):
        raise LifecycleRefusal(
            "ModelRegistry GGUF layer roots must be an explicit empty sequence"
        )
    if layer_roots:
        raise LifecycleRefusal(
            "ModelRegistry GGUF layer roots must be empty; the tensor-manifest "
            "root is the mesh weight trust anchor"
        )

    num_layers = getattr(model_spec, "num_layers", None)
    if (
        type(num_layers) is not int
        or not 1 <= num_layers <= MAX_MESH_MODEL_LAYERS
    ):
        raise LifecycleRefusal(
            f"ModelRegistry num_layers must be in [1, {MAX_MESH_MODEL_LAYERS}]"
        )
    if (
        type(target.max_context_len) is not int
        or not 1 <= target.max_context_len < 2**32
    ):
        raise LifecycleRefusal(
            "mesh max_context_len must be a positive uint32"
        )
    return expected_quant


def identity_matches(entry, target: RegistrationTarget) -> bool:
    """Match the tuple used by MinerRegistry duplicate/reactivation logic."""
    return (
        entry.model_id == target.model_id
        and entry.endpoint == target.endpoint
        and entry.quant == target.quant
    )


def configuration_matches(entry, target: RegistrationTarget) -> bool:
    return (
        identity_matches(entry, target)
        and bytes(entry.model_spec_ref) == bytes(target.model_spec_ref)
        and int(entry.max_context_len) == target.max_context_len
    )


def is_live(entry, now_unix: int) -> bool:
    return bool(entry.active) and int(entry.expires_at) > now_unix


def plan_registration(
    entries: Sequence[object],
    target: RegistrationTarget,
    *,
    now_unix: int | None = None,
) -> LifecyclePlan:
    """Plan registerModel() using the contract's exact slot semantics."""
    now = int(time.time()) if now_unix is None else int(now_unix)
    matches = [
        (index, entry)
        for index, entry in enumerate(entries)
        if identity_matches(entry, target)
    ]
    if len(matches) > 1:
        indices = ", ".join(str(index) for index, _entry in matches)
        raise LifecycleRefusal(
            "multiple entries have the requested (model, endpoint, quant) tuple "
            f"at indices {indices}; refusing to guess which slot is authoritative"
        )

    if matches:
        index, entry = matches[0]
        if is_live(entry, now):
            if not configuration_matches(entry, target):
                # A changed contract (re-measured maxContextLen beyond the
                # jitter band, or a re-anchored ModelSpec) must UPDATE the
                # existing slot, never append a new index: the contract's
                # registerModel reactivates a deactivated matching tuple
                # in place, preserving the per-(address, model_index)
                # score history. registerModel on an ACTIVE duplicate
                # would revert, so the refresh deactivates first.
                return LifecyclePlan(
                    action="refresh",
                    predicted_index=index,
                    reason=(
                        f"active tuple at index {index} carries a different "
                        "modelSpecRef or maxContextLen; deactivate + "
                        "register updates the slot in place"
                    ),
                )
            return LifecyclePlan(
                action="reuse-active",
                predicted_index=index,
                reason="exact active registration already exists; no transaction needed",
            )

        state = "inactive" if not entry.active else "expired"
        return LifecyclePlan(
            action="reactivate",
            predicted_index=index,
            reason=(
                f"exact {state} tuple will be reactivated in place; "
                "registerModel updates modelSpecRef, maxContextLen, and lease"
            ),
        )

    return LifecyclePlan(
        action="append",
        predicted_index=len(entries),
        reason="no matching tuple exists; contract will append if the preflight state holds",
    )


def plan_renewal(
    entries: Sequence[object],
    target: RegistrationTarget,
    index: int,
    *,
    now_unix: int | None = None,
) -> LifecyclePlan:
    """Validate an index and full tuple before calling renewModel()."""
    now = int(time.time()) if now_unix is None else int(now_unix)
    if index < 0 or index >= len(entries):
        raise LifecycleRefusal(
            f"renew index {index} is outside the current entry range "
            f"0..{max(len(entries) - 1, 0)}"
        )

    entry = entries[index]
    if not configuration_matches(entry, target):
        raise LifecycleRefusal(
            f"entry {index} does not match the expected model, endpoint, quant, "
            "modelSpecRef, and maxContextLen; refusing index-only renewal"
        )
    if int(entry.expires_at) <= now:
        raise LifecycleRefusal(
            f"entry {index} is expired; renewModel would revert. Run registration "
            "mode with the same tuple to reactivate the slot in place"
        )

    return LifecyclePlan(
        action="renew-active" if entry.active else "renew-inactive",
        predicted_index=index,
        reason=(
            "lease will be extended"
            if entry.active
            else "unexpired inactive entry will be reactivated and its lease extended"
        ),
    )


def resolve_registered_index(
    entries: Sequence[object],
    target: RegistrationTarget,
    *,
    now_unix: int | None = None,
) -> int:
    """Resolve the actual live index after a confirmed registry transaction."""
    now = int(time.time()) if now_unix is None else int(now_unix)
    identity_hits = [
        (index, entry)
        for index, entry in enumerate(entries)
        if identity_matches(entry, target)
    ]
    if len(identity_hits) != 1:
        indices = ", ".join(str(index) for index, _entry in identity_hits) or "none"
        raise LifecycleRefusal(
            "post-transaction registry state is ambiguous for the requested tuple "
            f"(matching indices: {indices})"
        )

    index, entry = identity_hits[0]
    if not configuration_matches(entry, target):
        raise LifecycleRefusal(
            f"post-transaction entry {index} has unexpected modelSpecRef or maxContextLen"
        )
    if not is_live(entry, now):
        raise LifecycleRefusal(
            f"post-transaction entry {index} is not active with an unexpired lease"
        )
    return index


def target_from_values(
    *,
    model_id: str,
    endpoint: str,
    quant: str,
    max_context_len: int,
) -> RegistrationTarget:
    from web3 import Web3

    return RegistrationTarget(
        model_id=model_id,
        endpoint=endpoint,
        model_spec_ref=bytes(Web3.solidity_keccak(["string"], [model_id])),
        quant=quant,
        max_context_len=int(max_context_len),
    )
