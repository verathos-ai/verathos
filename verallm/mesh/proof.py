"""Proof transcript primitives for llama.cpp-backed Verathos meshes."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, ClassVar, Iterable, Mapping

from verallm.mesh.receipt_signing import (
    sign_stage_proof_receipt_body_hash,
    verify_stage_proof_receipt_signature,
)
from verallm.mesh.types import MeshSpec, StageRange, canonical_json_bytes


RECEIPT_ONLY_PROOF_MODE = "receipt_only"
LLAMA_CPP_RPC_RECEIPT_PROOF_MODE = "llama_cpp_rpc_receipt_v1"
VERATHOS_GGML_TRACE_PROOF_MODE = "verathos_ggml_trace_v1"
VERATHOS_GGML_GEMM_PROOF_MODE = "verathos_ggml_gemm_v1"
# Light tier: openings-only, no weight materialization, no sumcheck. Proves
# structural membership + activation chaining + streamed-token binding, NOT
# weight execution (that is the hard tier). See mesh_hardening_progress.md.
VERATHOS_GGML_LIGHT_PROOF_MODE = "verathos_ggml_light_v1"
VERATHOS_GGUF_DECODE_AUDIT_MODE = "verathos_gguf_decode_audit_v1"
VERATHOS_GGUF_DECODE_AUDIT_TOP_K = 8
# Teacher-forced proof probes run through the same bounded eager prompt-tail
# path as verified serving. The native capture filters, proof assembler, and
# verifier must share this ceiling: a recurrent model can legitimately expose
# the whole eager tail as one GEMM instead of a one-row decode GEMM.
SLOT_VIEW_PROBE_MAX_ROWS = 32
# Near-tie acceptance for the f32 argmax equality check: split meshes flip
# near-tied argmaxes between the batched serve path and the single-row f32
# replay (different reduction orders). A served token inside the audited
# top-k whose hash-bound replay-row logit sits within this tolerance of the
# replay argmax is an honest cross-path flip, not a substitution; degenerate
# spam sits orders of magnitude outside it. Mirrors the runtime's shipped
# GEMM proof tolerances (abs 0.08 / rel 0.04).
VERATHOS_GGUF_DECODE_AUDIT_NEAR_TIE_ABS = 0.08
VERATHOS_GGUF_DECODE_AUDIT_NEAR_TIE_REL = 0.04
VALIDATOR_POSTCOMMIT_CHALLENGE_KIND = "validator_postcommit_v1"
PROOF_SAMPLE_BPS_DENOMINATOR = 10_000


# Marker for failures where the coordinator broke a commitment it had already
# made, rather than merely failing a check.  A shrunken challenge universe, a
# weight root that does not match the receipt, a refusal to serve the pinned
# snapshot: none of these happen to an honest operator having a bad epoch, so
# they are priced harder than an ordinary proof failure.  The scorer matches
# on this prefix, so it is a deliberate contract between the verifier and the
# scorer and must stay stable.
MESH_BINDING_VIOLATION_PREFIX = "mesh binding violation"

# Staged rollout for the stage-boundary activation chain.
#
# The verifier can enforce the chain today, but the capture side is a
# llama.cpp patch whose byte streams have not yet been confirmed to line up
# across a live RPC edge.  Requiring the roots before that is confirmed would
# reject every honest multi-stage mesh, so presence is opt-in and *consistency
# is always enforced*: whatever roots a mesh does report must chain.  Flip this
# on once a real multi-worker run shows matching roots, and the absence of a
# root stops being tolerated.
MESH_REQUIRE_BOUNDARY_CHAIN_ENV = "VERATHOS_MESH_REQUIRE_BOUNDARY_CHAIN"


def mesh_boundary_chain_required() -> bool:
    """Return whether missing activation boundary roots are a failure.

    Default ON because captured roots survive the receipt path end to end
    and chain across stages. Set the env to ``0`` only to debug a
    capture-less build.
    """

    import os

    return os.environ.get(MESH_REQUIRE_BOUNDARY_CHAIN_ENV, "1") == "1"


def mesh_binding_violation(detail: str) -> RuntimeError:
    """Return a proof failure tagged as a broken commitment."""

    return RuntimeError(f"{MESH_BINDING_VIOLATION_PREFIX}: {detail}")


def mesh_binding_violation_reason(detail: str) -> str:
    """Tag an existing failure reason as a broken commitment, idempotently."""

    text = str(detail or "")
    if is_mesh_binding_violation(text):
        return text
    return f"{MESH_BINDING_VIOLATION_PREFIX}: {text}"


def is_mesh_binding_violation(reason: Any) -> bool:
    """Return whether a failure reason describes a broken commitment."""

    return MESH_BINDING_VIOLATION_PREFIX in str(reason or "")


def normalize_proof_sample_bps(value: int) -> int:
    """Return a valid basis-point proof challenge rate."""

    bps = int(value)
    if bps < 0 or bps > PROOF_SAMPLE_BPS_DENOMINATOR:
        raise ValueError("proof sample bps must be between 0 and 10000")
    return bps


REPLAY_SEED_MODE_DERIVED = "derived_request_id_v1"
REPLAY_SEED_MODE_CLIENT = "client"


def derive_mesh_replay_seed(request_id: str) -> int:
    """Derive a deterministic llama.cpp sampler seed from the request id.

    Organic traffic keeps client sampler controls, so without a pinned seed a
    deferred-audit replay cannot reproduce the committed completion. The
    request id is already committed in the receipt, so a seed derived from it
    is replayable by anyone holding the receipt while still varying across
    requests. Constrained to [1, 2^31-1]: llama.cpp treats negative/UINT32_MAX
    seeds as "randomize" and 0 is kept clear of the verified sampler profile.
    """

    digest = hashlib.sha256(
        b"VERATHOS_MESH_REPLAY_SEED_V1" + str(request_id).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:4], "little") % (2**31 - 1) + 1


def normalize_validator_nonce(value: str) -> bytes:
    """Parse a validator nonce in the same 32-byte hex form as the vLLM path."""

    raw = str(value or "").strip()
    if raw.startswith("0x"):
        raw = raw[2:]
    if len(raw) != 64:
        raise ValueError("validator_nonce must be 32 bytes encoded as hex")
    try:
        nonce = bytes.fromhex(raw)
    except ValueError as exc:
        raise ValueError("validator_nonce must be hex") from exc
    if len(nonce) != 32:
        raise ValueError("validator_nonce must be 32 bytes")
    return nonce


def normalize_validator_challenge_nonce(value: str | bytes) -> bytes:
    """Parse the secret nonce revealed only after the origin receipt exists."""

    if isinstance(value, bytes):
        nonce = value
    else:
        try:
            nonce = normalize_validator_nonce(value)
        except ValueError as exc:
            raise ValueError(
                "validator challenge nonce must be 32 bytes encoded as hex"
            ) from exc
    if len(nonce) != 32:
        raise ValueError("validator challenge nonce must be 32 bytes")
    return nonce


def mesh_validator_challenge_nonce_commitment(
    challenge_nonce: str | bytes,
    *,
    validator_request_id: str,
    verification_snapshot_hash: str,
) -> str:
    """Commit a hidden validator challenge to one request and snapshot."""

    _require_hex_digest("validator_request_id", validator_request_id)
    _require_hex_digest(
        "verification_snapshot_hash",
        verification_snapshot_hash,
    )
    nonce = normalize_validator_challenge_nonce(challenge_nonce)
    return hashlib.sha256(
        b"VERATHOS_MESH_VALIDATOR_CHALLENGE_NONCE_COMMITMENT_V1"
        + nonce
        + bytes.fromhex(validator_request_id)
        + bytes.fromhex(verification_snapshot_hash)
    ).hexdigest()


def derive_mesh_postcommit_proof_beacon(
    *,
    origin_receipt_hash: str,
    proof_gate_hash: str,
    challenge_nonce: str | bytes,
) -> bytes:
    """Derive proof selection after response and stage roots are signed."""

    _require_hex_digest("origin_receipt_hash", origin_receipt_hash)
    _require_hex_digest("proof_gate_hash", proof_gate_hash)
    nonce = normalize_validator_challenge_nonce(challenge_nonce)
    return hashlib.sha256(
        b"VERATHOS_MESH_POSTCOMMIT_PROOF_BEACON_V1"
        + bytes.fromhex(origin_receipt_hash)
        + bytes.fromhex(proof_gate_hash)
        + nonce
    ).digest()


def _mesh_stable_sampling_commitment(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Return the non-grindable inputs shared by mesh sampling domains.

    Routing, member assignment, and trace/manifest commitments are deliberately
    absent.  They remain signed receipt and proof-transcript bindings, but a
    coordinator can choose or serialize them and therefore must not be able to
    use them to steer whether an expensive audit is selected or which decode
    position is opened.

    The signed snapshot hash commits the approved mesh identity, model anchors,
    stage set, and proof policy.  Request/response and token commitments bind
    the actual inference.  The remaining explicit policy fields distinguish
    the validator-approved organic and canary profiles.
    """

    prompt_token_ids_hash = str(receipt.get("prompt_token_ids_hash", "") or "")
    completion_token_ids_hash = str(
        receipt.get("completion_token_ids_hash", "") or ""
    )
    return {
        "version": 2,
        "request_id": str(receipt.get("request_id", "") or ""),
        "request_hash": str(receipt.get("request_hash", "") or ""),
        "semantic_response_hash": str(
            receipt.get("semantic_response_hash", "") or ""
        ),
        "mesh_id": str(receipt.get("mesh_id", "") or ""),
        "verification_snapshot_hash": str(
            receipt.get("verification_snapshot_hash", "") or ""
        ),
        "model_package_hash": str(
            receipt.get("model_package_hash", "") or ""
        ),
        "model_tensor_manifest_root": str(
            receipt.get("model_tensor_manifest_root", "") or ""
        ),
        "prompt_token_ids_hash": prompt_token_ids_hash,
        "prompt_token_count": (
            int(receipt.get("prompt_token_count", 0))
            if prompt_token_ids_hash
            else 0
        ),
        "completion_token_ids_hash": completion_token_ids_hash,
        "completion_token_count": (
            int(receipt.get("completion_token_count", 0))
            if completion_token_ids_hash
            else 0
        ),
        "proof_policy_profile": str(
            receipt.get("proof_policy_profile", "") or ""
        ),
        "proof_trace_manifest_format": str(
            receipt.get("proof_trace_manifest_format", "") or ""
        ),
        "proof_sample_bps": int(receipt.get("proof_sample_bps", 0)),
        "proof_sample_denominator": int(
            receipt.get(
                "proof_sample_denominator",
                PROOF_SAMPLE_BPS_DENOMINATOR,
            )
        ),
        "proof_ops_per_request": int(
            receipt.get("proof_ops_per_request", 0)
        ),
        "proof_trace_candidates_per_request": int(
            receipt.get("proof_trace_candidates_per_request", 0)
        ),
        "proof_challenge_kind": str(
            receipt.get("proof_challenge_kind", "") or ""
        ),
        "proof_challenge_nonce_commitment": str(
            receipt.get("proof_challenge_nonce_commitment", "") or ""
        ),
        "proof_validator_hotkey": str(
            receipt.get("proof_validator_hotkey", "") or ""
        ),
        "proof_postcommit": bool(receipt.get("proof_postcommit", False)),
        "proof_deferred": bool(receipt.get("proof_deferred", False)),
        "verified_sampler_mode": str(
            receipt.get("verified_sampler_mode", "") or ""
        ),
        "verified_sampler_controls_hash": str(
            receipt.get("verified_sampler_controls_hash", "") or ""
        ),
        "decode_audit_mode": str(
            receipt.get("decode_audit_mode", "") or ""
        ),
        "decode_audit_bps": int(receipt.get("decode_audit_bps", 0)),
        "decode_audit_top_k": int(receipt.get("decode_audit_top_k", 0)),
    }


def _mesh_deferred_replay_commitment(
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Freeze coordinator-selected replay state before future randomness.

    Inline proof/decode sampling must exclude these fields because the
    coordinator already knows the validator nonce and could otherwise grind
    cheap routing or trace variants.  Deferred audits have the opposite
    requirement: their randomness is not available until after the origin
    receipt is fixed, so the commitment must bind the exact topology, trace
    view, and deterministic replay layout that the later audit will use.
    """

    capture_window = str(receipt.get("proof_capture_window", "") or "")
    return {
        "version": 2,
        "receipt_version": int(receipt.get("version", 0)),
        "mesh_spec_hash": str(receipt.get("mesh_spec_hash", "") or ""),
        "stage_assignment_hash": str(
            receipt.get("stage_assignment_hash", "") or ""
        ),
        "rpc_plan_hash": str(receipt.get("rpc_plan_hash", "") or ""),
        "runtime": str(receipt.get("runtime", "") or ""),
        "uid": int(receipt.get("uid", 0)),
        "hotkey": str(receipt.get("hotkey", "") or ""),
        "endpoint": str(receipt.get("endpoint", "") or ""),
        "stage_index": int(receipt.get("stage_index", 0)),
        "layer_start": int(receipt.get("layer_start", 0)),
        "layer_end": int(receipt.get("layer_end", 0)),
        "response_hash": str(receipt.get("response_hash", "") or ""),
        "verification_snapshot_generation": int(
            receipt.get("verification_snapshot_generation", 0)
        ),
        "verification_snapshot_epoch": int(
            receipt.get("verification_snapshot_epoch", 0)
        ),
        "model_index": int(receipt.get("model_index", 0)),
        "model_total_layers": int(receipt.get("model_total_layers", 0)),
        "prompt_token_source": str(
            receipt.get("prompt_token_source", "") or ""
        ),
        "prompt_template_hash": str(
            receipt.get("prompt_template_hash", "") or ""
        ),
        "completion_token_source": str(
            receipt.get("completion_token_source", "") or ""
        ),
        "proof_trace_commitment_root": str(
            receipt.get("proof_trace_commitment_root", "") or ""
        ),
        "proof_trace_commitment_count": int(
            receipt.get("proof_trace_commitment_count", 0)
        ),
        "proof_trace_scope": str(
            receipt.get("proof_trace_scope", "") or ""
        ),
        "proof_op_manifest_root": str(
            receipt.get("proof_op_manifest_root", "") or ""
        ),
        "proof_op_manifest_count": int(
            receipt.get("proof_op_manifest_count", 0)
        ),
        "proof_op_manifest_scope": str(
            receipt.get("proof_op_manifest_scope", "") or ""
        ),
        "proof_slot_id": int(receipt.get("proof_slot_id", -1)),
        "proof_capture_window_hash": hashlib.sha256(
            b"VERATHOS_MESH_CAPTURE_WINDOW_V1"
            + capture_window.encode("utf-8")
        ).hexdigest(),
        "proof_runtime_ubatch_size": int(
            receipt.get("proof_runtime_ubatch_size", 0)
        ),
        "proof_replay_seed_mode": str(
            receipt.get("proof_replay_seed_mode", "") or ""
        ),
        "proof_replay_seed": int(receipt.get("proof_replay_seed", 0)),
        "decode_audit_commitment_hash": str(
            receipt.get("decode_audit_commitment_hash", "") or ""
        ),
        "decode_audit_stage_index": int(
            receipt.get("decode_audit_stage_index", -1)
        ),
    }


def mesh_proof_gate_hash(receipt: Mapping[str, Any]) -> str:
    """Commit stable validator/snapshot/model/request/response sampling inputs."""

    return hashlib.sha256(
        b"VERATHOS_MESH_PROOF_GATE_COMMITMENT_V2"
        + canonical_json_bytes(_mesh_stable_sampling_commitment(receipt))
    ).hexdigest()


def derive_mesh_proof_beacon(gate_hash: str, validator_nonce: str | bytes) -> bytes:
    """Derive the mesh proof beacon from a post-inference gate commitment."""

    _require_hex_digest("gate_hash", gate_hash)
    nonce = validator_nonce if isinstance(validator_nonce, bytes) else normalize_validator_nonce(validator_nonce)
    if len(nonce) != 32:
        raise ValueError("validator_nonce must be 32 bytes")
    h = hashlib.sha256()
    h.update(b"VERATHOS_MESH_PROOF_BEACON_V1")
    h.update(bytes.fromhex(gate_hash))
    h.update(nonce)
    return h.digest()


def normalize_deferred_randomness(value: str | bytes) -> bytes:
    """Parse future validator randomness as a 32-byte value."""

    if isinstance(value, bytes):
        randomness = value
    else:
        raw = str(value or "").strip()
        if raw.startswith("0x"):
            raw = raw[2:]
        try:
            randomness = bytes.fromhex(raw)
        except ValueError as exc:
            raise ValueError("deferred randomness must be hex") from exc
    if len(randomness) != 32:
        raise ValueError("deferred randomness must be 32 bytes")
    return randomness


def mesh_deferred_audit_sample_commitment_hash(
    receipt: Mapping[str, Any],
) -> str:
    """Commit only non-grindable inputs used by future-randomness sampling."""

    body = {
        "version": 2,
        "stable_sampling_commitment": _mesh_stable_sampling_commitment(receipt),
        "proof_gate_hash": str(receipt.get("proof_gate_hash", "") or ""),
        "proof_deferred_obligation": bool(
            receipt.get("proof_deferred_obligation", False)
        ),
        "proof_deferred_required": bool(
            receipt.get("proof_deferred_required", False)
        ),
        "proof_deferred_mode": str(
            receipt.get("proof_deferred_mode", "future_randomness_v1") or ""
        ),
        "proof_deferred_audit_bps": int(
            receipt.get("proof_deferred_audit_bps", 0)
        ),
        "proof_deferred_randomness_round": str(
            receipt.get("proof_deferred_randomness_round", "") or ""
        ),
    }
    return hashlib.sha256(
        b"VERATHOS_MESH_DEFERRED_AUDIT_SAMPLE_COMMITMENT_V2"
        + canonical_json_bytes(body)
    ).hexdigest()


def mesh_deferred_audit_commitment_hash(receipt: Mapping[str, Any]) -> str:
    """Freeze the exact deferred proof obligation before future randomness.

    The separate sample commitment excludes coordinator-controlled choices.
    This obligation commitment binds that stable sample domain to the concrete
    topology, trace universe, slot view, and replay parameters that a later
    sampled audit must prove.
    """

    body = {
        "version": 3,
        "sample_commitment_hash": (
            mesh_deferred_audit_sample_commitment_hash(receipt)
        ),
        "deferred_replay_commitment": _mesh_deferred_replay_commitment(receipt),
    }
    return hashlib.sha256(
        b"VERATHOS_MESH_DEFERRED_AUDIT_OBLIGATION_V3"
        + canonical_json_bytes(body)
    ).hexdigest()


def derive_mesh_deferred_audit_beacon(
    deferred_sample_commitment_hash: str,
    randomness: str | bytes,
    *,
    randomness_round: str = "",
) -> bytes:
    """Derive the later audit beacon from the stable sample commitment."""

    _require_hex_digest(
        "deferred_sample_commitment_hash",
        deferred_sample_commitment_hash,
    )
    h = hashlib.sha256()
    h.update(b"VERATHOS_MESH_DEFERRED_AUDIT_BEACON_V2")
    h.update(bytes.fromhex(deferred_sample_commitment_hash))
    h.update(normalize_deferred_randomness(randomness))
    h.update(str(randomness_round or "").encode("utf-8"))
    return h.digest()


def mesh_deferred_audit_sample_value(
    beacon: bytes,
    *,
    denominator: int = PROOF_SAMPLE_BPS_DENOMINATOR,
) -> int:
    """Map a future-randomness audit beacon to a basis-point sample bucket."""

    if len(beacon) != 32:
        raise ValueError("beacon must be 32 bytes")
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    digest = hashlib.sha256(b"VERATHOS_MESH_DEFERRED_AUDIT_SAMPLE_GATE_V1" + beacon).digest()
    return int.from_bytes(digest[:8], "little") % int(denominator)


def should_sample_mesh_deferred_audit(
    *,
    beacon: bytes,
    sample_bps: int,
    denominator: int = PROOF_SAMPLE_BPS_DENOMINATOR,
) -> bool:
    """Return whether future randomness selected a receipt for audit."""

    bps = normalize_proof_sample_bps(sample_bps)
    if bps <= 0:
        return False
    if bps >= denominator:
        return True
    return mesh_deferred_audit_sample_value(beacon, denominator=denominator) < bps


def mesh_proof_sample_value(
    beacon: bytes,
    *,
    denominator: int = PROOF_SAMPLE_BPS_DENOMINATOR,
) -> int:
    """Map a mesh proof beacon to a basis-point sample bucket."""

    if len(beacon) != 32:
        raise ValueError("beacon must be 32 bytes")
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    digest = hashlib.sha256(b"VERATHOS_MESH_PROOF_SAMPLE_GATE_V1" + beacon).digest()
    return int.from_bytes(digest[:8], "little") % int(denominator)


def should_sample_mesh_proof(
    *,
    beacon: bytes,
    sample_bps: int,
    denominator: int = PROOF_SAMPLE_BPS_DENOMINATOR,
) -> bool:
    """Fiat-Shamir proof gate equivalent to the vLLM sampling bps gate."""

    bps = normalize_proof_sample_bps(sample_bps)
    if bps <= 0:
        return False
    if bps >= denominator:
        return True
    return mesh_proof_sample_value(beacon, denominator=denominator) < bps


def mesh_decode_audit_commitment_hash(receipt: Mapping[str, Any]) -> str:
    """Commit stable inputs used for sampled decode-position selection."""

    return hashlib.sha256(
        b"VERATHOS_MESH_DECODE_AUDIT_COMMITMENT_V2"
        + canonical_json_bytes(_mesh_stable_sampling_commitment(receipt))
    ).hexdigest()


def mesh_decode_audit_sample_value(
    beacon: bytes,
    *,
    denominator: int = PROOF_SAMPLE_BPS_DENOMINATOR,
) -> int:
    """Map a mesh proof beacon to an independent decode-audit sample bucket."""

    if len(beacon) != 32:
        raise ValueError("beacon must be 32 bytes")
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    digest = hashlib.sha256(b"VERATHOS_MESH_DECODE_AUDIT_SAMPLE_GATE_V1" + beacon).digest()
    return int.from_bytes(digest[:8], "little") % int(denominator)


def should_sample_mesh_decode_audit(
    *,
    beacon: bytes,
    sample_bps: int,
    denominator: int = PROOF_SAMPLE_BPS_DENOMINATOR,
) -> bool:
    """Return whether the bound request is sampled for decode/logit audit."""

    bps = normalize_proof_sample_bps(sample_bps)
    if bps <= 0:
        return False
    if bps >= denominator:
        return True
    return mesh_decode_audit_sample_value(beacon, denominator=denominator) < bps


# Decode audits draw from the trailing candidate window only (GLEIPNIR
# semantics). Bounding the pool is what makes the teacher-forced audit
# replay a bounded window: recurrent/hybrid models keep rolling state
# checkpoints only near the sequence end, so a position deep in the
# completion would force a replay from the prompt boundary that scales
# with the completion instead of the window. The final position is always
# audited, so end-of-stream substitution is caught deterministically;
# tokens outside the window stay bound by completion_token_ids_hash and
# condition the audited trailing rows through teacher forcing.
DECODE_AUDIT_CANDIDATE_ROWS = 32


def derive_mesh_decode_audit_positions(
    *,
    beacon: bytes | str,
    decode_commitment_hash: str,
    completion_token_count: int,
    max_positions: int | None = None,
) -> list[int]:
    """Select generated-token positions for a sampled GGUF decode audit."""

    count = int(completion_token_count)
    if count <= 0:
        return []
    _require_hex_digest("decode_commitment_hash", decode_commitment_hash)
    beacon_bytes = bytes.fromhex(beacon) if isinstance(beacon, str) else beacon
    if len(beacon_bytes) != 32:
        raise ValueError("decode audit beacon must be 32 bytes")
    if max_positions is None:
        if count <= 1024:
            limit = 1
        elif count <= 4096:
            limit = 2
        else:
            limit = 3
    else:
        limit = int(max_positions)
    window = min(int(DECODE_AUDIT_CANDIDATE_ROWS), count)
    window_start = count - window
    limit = min(max(1, limit), window)
    # The audit anchors at the SECOND-to-last completion position, and the
    # random candidates exclude the final position too. The final token's
    # next-token distribution is the flattest of the whole completion
    # (sentence-final punctuation, EOS-adjacent mass), and its top-k
    # membership does not survive replay-vs-serve numerics on split
    # meshes: with the anchor at count-1 EVERY audit hit that position,
    # and an honest 4-GPU mesh cycled through probation on false
    # attributions for a full night. Position count-2 still tail-binds the completion and still
    # catches decode degeneration; the full fix (opening the SERVED
    # trace's column instead of a replay reproduction) is tracked. Both
    # sides derive positions - deploy validator and workers together.
    anchor = count - 2 if count >= 2 else count - 1
    draw_span = max(1, window - 1) if count >= 2 else window
    selected: list[int] = [anchor]
    counter = 0
    while len(selected) < limit:
        digest = hashlib.sha256(
            b"VERATHOS_MESH_DECODE_AUDIT_POSITION_V2"
            + beacon_bytes
            + bytes.fromhex(decode_commitment_hash)
            + int(count).to_bytes(8, "little", signed=False)
            + counter.to_bytes(4, "little", signed=False)
        ).digest()
        candidate = (
            window_start + int.from_bytes(digest[:8], "little") % draw_span
        )
        if candidate not in selected:
            selected.append(candidate)
        counter += 1
    return sorted(selected)


def _sha256_tagged(tag: bytes, *parts: bytes) -> bytes:
    h = hashlib.sha256(tag)
    for part in parts:
        h.update(part)
    return h.digest()


def _is_hex_digest(value: str, *, bytes_len: int = 32) -> bool:
    if len(value) != bytes_len * 2:
        return False
    try:
        bytes.fromhex(value)
    except ValueError:
        return False
    return True


def _require_hex_digest(name: str, value: str, *, allow_empty: bool = False) -> None:
    if allow_empty and not value:
        return
    if not _is_hex_digest(value):
        raise ValueError(f"{name} must be a {32 * 2}-character hex SHA-256 digest")


@dataclass(frozen=True)
class LlamaGraphOpReceipt:
    """One proof transcript commitment for a GGML graph operation.

    This is deliberately independent from Python/PyTorch witness capture. A
    proof-capable llama.cpp RPC runtime can emit the same JSON shape while
    staying wire-compatible with upstream llama.cpp RPC.
    """

    request_id: str
    mesh_id: str
    mesh_spec_hash: str
    stage_assignment_hash: str
    rpc_plan_hash: str
    model_package_hash: str
    model_tensor_manifest_root: str
    uid: int
    hotkey: str
    endpoint: str
    stage_index: int
    layer_start: int
    layer_end: int
    request_hash: str
    response_hash: str
    graph_id: str
    op_index: int
    op_type: str
    layer_index: int
    tensor_name: str
    input_root: str
    weight_root: str
    output_root: str
    quantization: str
    backend: str
    device: str
    proof_kind: str
    proof_commitment_hash: str
    # Activation handoff commitments for the whole stage, captured at the RPC
    # boundary rather than around one sampled op.  Empty on a mesh with no
    # handoff to chain; see MeshStageProofReceipt for the chain rule.
    input_boundary_root: str = ""
    output_boundary_root: str = ""
    signature: str = ""
    version: int = 1

    def validate(self) -> None:
        if self.version != 1:
            raise ValueError("unsupported LlamaGraphOpReceipt version")
        if not self.request_id:
            raise ValueError("request_id is required")
        if not self.mesh_id:
            raise ValueError("mesh_id is required")
        if self.uid < 0:
            raise ValueError("uid must be >= 0")
        if not self.hotkey:
            raise ValueError("hotkey is required")
        if not self.endpoint:
            raise ValueError("endpoint is required")
        if self.stage_index < 0:
            raise ValueError("stage_index must be >= 0")
        StageRange(self.layer_start, self.layer_end).validate()
        if self.layer_index < -1:
            raise ValueError("layer_index must be >= -1")
        if self.op_index < 0:
            raise ValueError("op_index must be >= 0")
        if not self.graph_id:
            raise ValueError("graph_id is required")
        if not self.op_type:
            raise ValueError("op_type is required")
        if not self.proof_kind:
            raise ValueError("proof_kind is required")
        _require_hex_digest("mesh_spec_hash", self.mesh_spec_hash)
        _require_hex_digest("stage_assignment_hash", self.stage_assignment_hash)
        _require_hex_digest("rpc_plan_hash", self.rpc_plan_hash, allow_empty=True)
        _require_hex_digest("model_package_hash", self.model_package_hash, allow_empty=True)
        _require_hex_digest(
            "model_tensor_manifest_root",
            self.model_tensor_manifest_root,
            allow_empty=True,
        )
        _require_hex_digest("request_hash", self.request_hash)
        _require_hex_digest("response_hash", self.response_hash)
        _require_hex_digest("input_root", self.input_root)
        _require_hex_digest("weight_root", self.weight_root)
        _require_hex_digest("output_root", self.output_root)
        # Empty is permitted only so a single-stage mesh, which has no
        # handoff to chain, stays representable.  The chain check rejects an
        # empty root wherever a neighbouring stage exists.
        _require_hex_digest(
            "input_boundary_root", self.input_boundary_root, allow_empty=True,
        )
        _require_hex_digest(
            "output_boundary_root", self.output_boundary_root, allow_empty=True,
        )
        _require_hex_digest("proof_commitment_hash", self.proof_commitment_hash)

    def to_dict(self, *, include_signature: bool = True) -> dict[str, Any]:
        data = {
            "version": int(self.version),
            "request_id": self.request_id,
            "mesh_id": self.mesh_id,
            "mesh_spec_hash": self.mesh_spec_hash,
            "stage_assignment_hash": self.stage_assignment_hash,
            "rpc_plan_hash": self.rpc_plan_hash,
            "model_package_hash": self.model_package_hash,
            "model_tensor_manifest_root": self.model_tensor_manifest_root,
            "uid": int(self.uid),
            "hotkey": self.hotkey,
            "endpoint": self.endpoint,
            "stage_index": int(self.stage_index),
            "layer_start": int(self.layer_start),
            "layer_end": int(self.layer_end),
            "request_hash": self.request_hash,
            "response_hash": self.response_hash,
            "graph_id": self.graph_id,
            "op_index": int(self.op_index),
            "op_type": self.op_type,
            "layer_index": int(self.layer_index),
            "tensor_name": self.tensor_name,
            "input_root": self.input_root,
            "weight_root": self.weight_root,
            "output_root": self.output_root,
            "input_boundary_root": self.input_boundary_root,
            "output_boundary_root": self.output_boundary_root,
            "quantization": self.quantization,
            "backend": self.backend,
            "device": self.device,
            "proof_kind": self.proof_kind,
            "proof_commitment_hash": self.proof_commitment_hash,
        }
        if include_signature:
            data["signature"] = self.signature
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LlamaGraphOpReceipt":
        receipt = cls(
            version=int(data.get("version", 1)),
            request_id=str(data["request_id"]),
            mesh_id=str(data["mesh_id"]),
            mesh_spec_hash=str(data["mesh_spec_hash"]),
            stage_assignment_hash=str(data["stage_assignment_hash"]),
            rpc_plan_hash=str(data.get("rpc_plan_hash", "")),
            model_package_hash=str(data.get("model_package_hash", "")),
            model_tensor_manifest_root=str(data.get("model_tensor_manifest_root", "")),
            uid=int(data["uid"]),
            hotkey=str(data["hotkey"]),
            endpoint=str(data["endpoint"]),
            stage_index=int(data["stage_index"]),
            layer_start=int(data["layer_start"]),
            layer_end=int(data["layer_end"]),
            request_hash=str(data["request_hash"]),
            response_hash=str(data["response_hash"]),
            graph_id=str(data["graph_id"]),
            op_index=int(data["op_index"]),
            op_type=str(data["op_type"]),
            layer_index=int(data.get("layer_index", -1)),
            tensor_name=str(data.get("tensor_name", "")),
            input_root=str(data["input_root"]),
            weight_root=str(data["weight_root"]),
            output_root=str(data["output_root"]),
            quantization=str(data.get("quantization", "unknown")),
            backend=str(data.get("backend", "")),
            device=str(data.get("device", "")),
            proof_kind=str(data["proof_kind"]),
            proof_commitment_hash=str(data["proof_commitment_hash"]),
            input_boundary_root=str(data.get("input_boundary_root", "")),
            output_boundary_root=str(data.get("output_boundary_root", "")),
            signature=str(data.get("signature", "")),
        )
        receipt.validate()
        return receipt

    def body_hash(self) -> bytes:
        return _sha256_tagged(
            b"VERATHOS_LLAMA_GRAPH_OP_RECEIPT_BODY_V1",
            canonical_json_bytes(self.to_dict(include_signature=False)),
        )

    def receipt_hash(self) -> bytes:
        return _sha256_tagged(
            b"VERATHOS_LLAMA_GRAPH_OP_RECEIPT_V1",
            canonical_json_bytes(self.to_dict(include_signature=True)),
        )

    def receipt_hash_hex(self) -> str:
        return self.receipt_hash().hex()


@dataclass(frozen=True)
class MeshStageProofReceipt:
    """Endpoint-free public receipt for one private mesh stage proof.

    A proof-producing worker converts its private
    :class:`LlamaGraphOpReceipt` to this public form and signs it before
    returning it.  Coordinators only aggregate already-signed receipts, so
    worker identity and routing fields never cross the public boundary.
    """

    request_id: str
    mesh_id: str
    mesh_spec_hash: str
    stage_assignment_hash: str
    rpc_plan_hash: str
    model_package_hash: str
    model_tensor_manifest_root: str
    model_index: int
    model_total_layers: int
    verification_snapshot_hash: str
    proof_gate_hash: str
    stage_proof_commitment: str
    stage_id: str
    stage_index: int
    layer_start: int
    layer_end: int
    request_hash: str
    response_hash: str
    graph_id: str
    op_index: int
    op_type: str
    layer_index: int
    tensor_name: str
    input_root: str
    weight_root: str
    output_root: str
    # Hash chain over the activations that actually crossed the RPC boundary
    # into and out of this stage.  input_root/output_root above commit the
    # operands of one sampled op; these commit the whole stage handoff, so a
    # stage cannot be proved in isolation over an X of its own choosing.
    input_boundary_root: str
    output_boundary_root: str
    quantization: str
    backend: str
    device: str
    proof_kind: str
    proof_commitment_hash: str
    signature: str = ""
    version: int = 3

    _FIELDS: ClassVar[set[str]] = {
        "version",
        "request_id",
        "mesh_id",
        "mesh_spec_hash",
        "stage_assignment_hash",
        "rpc_plan_hash",
        "model_package_hash",
        "model_tensor_manifest_root",
        "model_index",
        "model_total_layers",
        "verification_snapshot_hash",
        "proof_gate_hash",
        "stage_proof_commitment",
        "stage_id",
        "stage_index",
        "layer_start",
        "layer_end",
        "request_hash",
        "response_hash",
        "graph_id",
        "op_index",
        "op_type",
        "layer_index",
        "tensor_name",
        "input_root",
        "weight_root",
        "output_root",
        "input_boundary_root",
        "output_boundary_root",
        "quantization",
        "backend",
        "device",
        "proof_kind",
        "proof_commitment_hash",
        "signature",
    }

    def validate(self, *, require_signature: bool = False) -> None:
        if type(self.version) is not int or self.version != 3:
            raise ValueError("unsupported MeshStageProofReceipt version")
        string_fields = {
            "request_id": self.request_id,
            "mesh_id": self.mesh_id,
            "mesh_spec_hash": self.mesh_spec_hash,
            "stage_assignment_hash": self.stage_assignment_hash,
            "rpc_plan_hash": self.rpc_plan_hash,
            "model_package_hash": self.model_package_hash,
            "model_tensor_manifest_root": self.model_tensor_manifest_root,
            "verification_snapshot_hash": self.verification_snapshot_hash,
            "proof_gate_hash": self.proof_gate_hash,
            "stage_proof_commitment": self.stage_proof_commitment,
            "stage_id": self.stage_id,
            "request_hash": self.request_hash,
            "response_hash": self.response_hash,
            "graph_id": self.graph_id,
            "op_type": self.op_type,
            "tensor_name": self.tensor_name,
            "input_root": self.input_root,
            "weight_root": self.weight_root,
            "output_root": self.output_root,
            "input_boundary_root": self.input_boundary_root,
            "output_boundary_root": self.output_boundary_root,
            "quantization": self.quantization,
            "backend": self.backend,
            "device": self.device,
            "proof_kind": self.proof_kind,
            "proof_commitment_hash": self.proof_commitment_hash,
            "signature": self.signature,
        }
        for field_name, value in string_fields.items():
            if not isinstance(value, str):
                raise ValueError(f"{field_name} must be a string")
        if not self.request_id:
            raise ValueError("request_id is required")
        if not self.mesh_id:
            raise ValueError("mesh_id is required")
        if not (
            self.stage_id.startswith("stg_")
            and len(self.stage_id) == 36
            and all(ch in "0123456789abcdef" for ch in self.stage_id[4:])
        ):
            raise ValueError("stage_id must be an opaque 128-bit stage identifier")
        for field_name, value in (
            ("stage_index", self.stage_index),
            ("model_index", self.model_index),
            ("model_total_layers", self.model_total_layers),
            ("layer_start", self.layer_start),
            ("layer_end", self.layer_end),
            ("op_index", self.op_index),
            ("layer_index", self.layer_index),
        ):
            if type(value) is not int:
                raise ValueError(f"{field_name} must be an integer")
        if self.stage_index < 0:
            raise ValueError("stage_index must be >= 0")
        if self.model_index < 0:
            raise ValueError("model_index must be >= 0")
        if self.model_total_layers <= 0:
            raise ValueError("model_total_layers must be > 0")
        StageRange(self.layer_start, self.layer_end).validate()
        if self.layer_index < -1:
            raise ValueError("layer_index must be >= -1")
        if self.op_index < 0:
            raise ValueError("op_index must be >= 0")
        if not self.graph_id:
            raise ValueError("graph_id is required")
        if not self.op_type:
            raise ValueError("op_type is required")
        if not self.proof_kind:
            raise ValueError("proof_kind is required")
        _require_hex_digest("mesh_spec_hash", self.mesh_spec_hash)
        _require_hex_digest("stage_assignment_hash", self.stage_assignment_hash)
        _require_hex_digest("rpc_plan_hash", self.rpc_plan_hash, allow_empty=True)
        _require_hex_digest("model_package_hash", self.model_package_hash, allow_empty=True)
        _require_hex_digest(
            "model_tensor_manifest_root",
            self.model_tensor_manifest_root,
            allow_empty=True,
        )
        _require_hex_digest(
            "verification_snapshot_hash",
            self.verification_snapshot_hash,
        )
        _require_hex_digest("proof_gate_hash", self.proof_gate_hash)
        _require_hex_digest(
            "stage_proof_commitment",
            self.stage_proof_commitment,
        )
        _require_hex_digest("request_hash", self.request_hash)
        _require_hex_digest("response_hash", self.response_hash)
        _require_hex_digest("input_root", self.input_root)
        _require_hex_digest("weight_root", self.weight_root)
        _require_hex_digest("output_root", self.output_root)
        # Empty is permitted only so a single-stage mesh, which has no handoff
        # to chain, stays representable. The chain check rejects an empty root
        # wherever a neighbouring stage exists.
        _require_hex_digest(
            "input_boundary_root", self.input_boundary_root, allow_empty=True,
        )
        _require_hex_digest(
            "output_boundary_root", self.output_boundary_root, allow_empty=True,
        )
        _require_hex_digest("proof_commitment_hash", self.proof_commitment_hash)
        if require_signature and not self.signature:
            raise ValueError("stage proof receipt signature is required")
        if self.signature and not (
            len(self.signature) == 128
            and all(ch in "0123456789abcdef" for ch in self.signature)
        ):
            raise ValueError(
                "signature must be a lowercase 64-byte hex Sr25519 signature"
            )

    def to_dict(self, *, include_signature: bool = True) -> dict[str, Any]:
        data = {
            "version": int(self.version),
            "request_id": self.request_id,
            "mesh_id": self.mesh_id,
            "mesh_spec_hash": self.mesh_spec_hash,
            "stage_assignment_hash": self.stage_assignment_hash,
            "rpc_plan_hash": self.rpc_plan_hash,
            "model_package_hash": self.model_package_hash,
            "model_tensor_manifest_root": self.model_tensor_manifest_root,
            "model_index": int(self.model_index),
            "model_total_layers": int(self.model_total_layers),
            "verification_snapshot_hash": self.verification_snapshot_hash,
            "proof_gate_hash": self.proof_gate_hash,
            "stage_proof_commitment": self.stage_proof_commitment,
            "stage_id": self.stage_id,
            "stage_index": int(self.stage_index),
            "layer_start": int(self.layer_start),
            "layer_end": int(self.layer_end),
            "request_hash": self.request_hash,
            "response_hash": self.response_hash,
            "graph_id": self.graph_id,
            "op_index": int(self.op_index),
            "op_type": self.op_type,
            "layer_index": int(self.layer_index),
            "tensor_name": self.tensor_name,
            "input_root": self.input_root,
            "weight_root": self.weight_root,
            "output_root": self.output_root,
            "input_boundary_root": self.input_boundary_root,
            "output_boundary_root": self.output_boundary_root,
            "quantization": self.quantization,
            "backend": self.backend,
            "device": self.device,
            "proof_kind": self.proof_kind,
            "proof_commitment_hash": self.proof_commitment_hash,
        }
        if include_signature:
            data["signature"] = self.signature
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MeshStageProofReceipt":
        if not isinstance(data, Mapping):
            raise ValueError("MeshStageProofReceipt must be a mapping")
        if set(data) != cls._FIELDS:
            missing = sorted(cls._FIELDS - set(data))
            unknown = sorted(set(data) - cls._FIELDS)
            detail = []
            if missing:
                detail.append("missing " + ",".join(missing))
            if unknown:
                detail.append("unknown " + ",".join(unknown))
            raise ValueError(
                "invalid MeshStageProofReceipt fields: " + "; ".join(detail)
            )
        receipt = cls(
            version=data["version"],
            request_id=data["request_id"],
            mesh_id=data["mesh_id"],
            mesh_spec_hash=data["mesh_spec_hash"],
            stage_assignment_hash=data["stage_assignment_hash"],
            rpc_plan_hash=data["rpc_plan_hash"],
            model_package_hash=data["model_package_hash"],
            model_tensor_manifest_root=data["model_tensor_manifest_root"],
            model_index=data["model_index"],
            model_total_layers=data["model_total_layers"],
            verification_snapshot_hash=data["verification_snapshot_hash"],
            proof_gate_hash=data["proof_gate_hash"],
            stage_proof_commitment=data["stage_proof_commitment"],
            stage_id=data["stage_id"],
            stage_index=data["stage_index"],
            layer_start=data["layer_start"],
            layer_end=data["layer_end"],
            request_hash=data["request_hash"],
            response_hash=data["response_hash"],
            graph_id=data["graph_id"],
            op_index=data["op_index"],
            op_type=data["op_type"],
            layer_index=data["layer_index"],
            tensor_name=data["tensor_name"],
            input_root=data["input_root"],
            weight_root=data["weight_root"],
            output_root=data["output_root"],
            input_boundary_root=data["input_boundary_root"],
            output_boundary_root=data["output_boundary_root"],
            quantization=data["quantization"],
            backend=data["backend"],
            device=data["device"],
            proof_kind=data["proof_kind"],
            proof_commitment_hash=data["proof_commitment_hash"],
            signature=data["signature"],
        )
        receipt.validate(require_signature=True)
        return receipt

    @classmethod
    def from_private_receipt(
        cls,
        receipt: LlamaGraphOpReceipt,
        *,
        stage_id: str,
        model_index: int,
        model_total_layers: int,
        verification_snapshot_hash: str,
        proof_gate_hash: str,
        stage_proof_commitment: str,
    ) -> "MeshStageProofReceipt":
        """Build the unsigned public form on the proof-producing worker.

        The caller must immediately sign the result with the snapshot stage
        key.  A coordinator must never use this conversion to impersonate a
        worker after collecting private receipts.
        """

        receipt.validate()
        public_receipt = cls(
            request_id=receipt.request_id,
            mesh_id=receipt.mesh_id,
            mesh_spec_hash=receipt.mesh_spec_hash,
            stage_assignment_hash=receipt.stage_assignment_hash,
            rpc_plan_hash=receipt.rpc_plan_hash,
            model_package_hash=receipt.model_package_hash,
            model_tensor_manifest_root=receipt.model_tensor_manifest_root,
            model_index=model_index,
            model_total_layers=model_total_layers,
            verification_snapshot_hash=verification_snapshot_hash,
            proof_gate_hash=proof_gate_hash,
            stage_proof_commitment=stage_proof_commitment,
            stage_id=stage_id,
            stage_index=receipt.stage_index,
            layer_start=receipt.layer_start,
            layer_end=receipt.layer_end,
            request_hash=receipt.request_hash,
            response_hash=receipt.response_hash,
            graph_id=receipt.graph_id,
            op_index=receipt.op_index,
            op_type=receipt.op_type,
            layer_index=receipt.layer_index,
            tensor_name=receipt.tensor_name,
            input_root=receipt.input_root,
            weight_root=receipt.weight_root,
            output_root=receipt.output_root,
            input_boundary_root=receipt.input_boundary_root,
            output_boundary_root=receipt.output_boundary_root,
            quantization=receipt.quantization,
            backend=receipt.backend,
            device=receipt.device,
            proof_kind=receipt.proof_kind,
            proof_commitment_hash=receipt.proof_commitment_hash,
        )
        public_receipt.validate(require_signature=False)
        return public_receipt

    def body_hash(self) -> bytes:
        return _sha256_tagged(
            b"VERATHOS_MESH_STAGE_PROOF_RECEIPT_BODY_V2",
            canonical_json_bytes(self.to_dict(include_signature=False)),
        )

    def receipt_hash(self) -> bytes:
        return _sha256_tagged(
            b"VERATHOS_MESH_STAGE_PROOF_RECEIPT_V2",
            canonical_json_bytes(self.to_dict(include_signature=True)),
        )


def sign_mesh_stage_proof_receipt(
    receipt: MeshStageProofReceipt,
    keypair: Any,
    *,
    expected_proof_key: str,
    proof_key_scheme: str,
) -> MeshStageProofReceipt:
    """Sign an endpoint-free receipt with its snapshot-declared stage key."""

    if not isinstance(receipt, MeshStageProofReceipt):
        raise ValueError("receipt must be a MeshStageProofReceipt")
    receipt.validate(require_signature=False)
    signature = sign_stage_proof_receipt_body_hash(
        receipt.body_hash().hex(),
        keypair,
        expected_proof_key=expected_proof_key,
        proof_key_scheme=proof_key_scheme,
    )
    signed = replace(receipt, signature=signature)
    signed.validate(require_signature=True)
    return signed


def mesh_stage_proof_receipt_root(
    receipts: Iterable[MeshStageProofReceipt | Mapping[str, Any]],
) -> bytes:
    """Commit an endpoint-free list of public stage proof receipts."""

    parsed = [
        item
        if isinstance(item, MeshStageProofReceipt)
        else MeshStageProofReceipt.from_dict(item)
        for item in receipts
    ]
    ordered = sorted(
        parsed,
        key=lambda item: (
            item.request_id,
            item.stage_index,
            item.layer_index,
            item.op_index,
            item.stage_id,
            item.proof_commitment_hash,
        ),
    )
    if not ordered:
        return b""
    h = hashlib.sha256(b"VERATHOS_MESH_STAGE_PROOF_RECEIPT_ROOT_V2")
    h.update(len(ordered).to_bytes(4, "little"))
    for receipt in ordered:
        receipt.validate(require_signature=True)
        h.update(receipt.receipt_hash())
    return h.digest()


def mesh_stage_proof_receipt_root_hex(
    receipts: Iterable[MeshStageProofReceipt | Mapping[str, Any]],
) -> str:
    return mesh_stage_proof_receipt_root(receipts).hex()


def verify_mesh_stage_proof_receipts_for_snapshot(
    receipt: Mapping[str, Any],
    proof_receipts: Iterable[MeshStageProofReceipt | Mapping[str, Any]],
    snapshot: Any,
    *,
    require_complete_coverage: bool = True,
) -> list[MeshStageProofReceipt]:
    """Verify endpoint-free proof receipts against a signed stage snapshot."""

    parsed = [
        item
        if isinstance(item, MeshStageProofReceipt)
        else MeshStageProofReceipt.from_dict(item)
        for item in proof_receipts
    ]
    if not parsed:
        raise RuntimeError("proof receipts are required")
    seen_receipt_bodies: set[bytes] = set()
    for proof in parsed:
        body_hash = proof.body_hash()
        if body_hash in seen_receipt_bodies:
            raise RuntimeError("duplicate stage proof receipt")
        seen_receipt_bodies.add(body_hash)
    if type(receipt.get("proof_receipt_count")) is not int or int(
        receipt.get("proof_receipt_count", -1)
    ) != len(parsed):
        raise RuntimeError("proof_receipt_count mismatch")
    expected_root = mesh_stage_proof_receipt_root_hex(parsed)
    if str(receipt.get("proof_receipt_root", "")) != expected_root:
        raise RuntimeError("proof_receipt_root mismatch")
    if str(receipt.get("proof_receipt_format", "")) != "opaque_stage_v2":
        raise RuntimeError("proof receipt format mismatch")

    snapshot.validate(require_signature=True)
    snapshot_hash = snapshot.snapshot_hash_hex()
    if str(receipt.get("verification_snapshot_hash", "")) != snapshot_hash:
        raise RuntimeError("proof receipt snapshot hash mismatch")
    if str(receipt.get("mesh_id", "")) != snapshot.mesh_id:
        raise RuntimeError("proof receipt snapshot mesh_id mismatch")
    if str(receipt.get("model_package_hash", "")) != snapshot.model.model_package_hash:
        raise RuntimeError("proof receipt snapshot model_package_hash mismatch")
    if str(receipt.get("model_tensor_manifest_root", "")) != (
        snapshot.model.model_tensor_manifest_root
    ):
        raise RuntimeError("proof receipt snapshot tensor manifest mismatch")

    snapshot_by_range = {
        (int(stage.layer_start), int(stage.layer_end)): stage
        for stage in snapshot.stages
    }
    if len(snapshot_by_range) != len(snapshot.stages):
        raise RuntimeError("snapshot contains duplicate stage ranges")
    common = {
        "request_id": str(receipt.get("request_id", "")),
        "mesh_id": snapshot.mesh_id,
        "mesh_spec_hash": str(receipt.get("mesh_spec_hash", "")),
        "stage_assignment_hash": str(receipt.get("stage_assignment_hash", "")),
        "rpc_plan_hash": str(receipt.get("rpc_plan_hash", "")),
        "model_package_hash": snapshot.model.model_package_hash,
        "model_tensor_manifest_root": snapshot.model.model_tensor_manifest_root,
        "model_index": snapshot.coordinator.model_index,
        "model_total_layers": snapshot.model.total_layers,
        "request_hash": str(receipt.get("request_hash", "")),
        "response_hash": str(receipt.get("response_hash", "")),
        "proof_gate_hash": str(receipt.get("proof_gate_hash", "")),
    }
    covered_stage_ids: set[str] = set()
    stage_index_to_id: dict[int, str] = {}
    stage_id_to_index: dict[str, int] = {}
    for proof in parsed:
        data = proof.to_dict()
        for field, expected in common.items():
            if data.get(field) != expected:
                raise RuntimeError(f"proof receipt {field} mismatch")
        stage = snapshot_by_range.get((proof.layer_start, proof.layer_end))
        if stage is None or stage.stage_id != proof.stage_id:
            raise RuntimeError("proof receipt opaque stage binding mismatch")
        if proof.verification_snapshot_hash != snapshot_hash:
            raise RuntimeError("proof receipt verification snapshot hash mismatch")
        if proof.stage_proof_commitment != stage.proof_commitment:
            raise RuntimeError("proof receipt stage proof commitment mismatch")
        previous_id = stage_index_to_id.setdefault(proof.stage_index, proof.stage_id)
        if previous_id != proof.stage_id:
            raise RuntimeError("proof receipt stage index maps to multiple stages")
        previous_index = stage_id_to_index.setdefault(
            proof.stage_id,
            proof.stage_index,
        )
        if previous_index != proof.stage_index:
            raise RuntimeError("proof receipt opaque stage maps to multiple indexes")
        if not verify_stage_proof_receipt_signature(
            proof.body_hash().hex(),
            proof.signature,
            stage.proof_key,
            stage.proof_key_scheme,
        ):
            raise RuntimeError("proof receipt stage signature invalid")
        covered_stage_ids.add(proof.stage_id)

    if require_complete_coverage:
        expected_ids = {stage.stage_id for stage in snapshot.stages}
        missing = sorted(expected_ids - covered_stage_ids)
        if missing:
            raise RuntimeError(
                "missing proof receipts for snapshot stages: " + ",".join(missing)
            )
    _require_stage_boundary_chain(parsed, stage_count=len(snapshot.stages))
    return parsed


def _require_stage_boundary_chain(
    receipts: Iterable["MeshStageProofReceipt"],
    *,
    stage_count: int,
) -> None:
    """Require each stage's output activations to be the next stage's input.

    Without this every stage proves an isolated statement over an activation
    of its own choosing, so a coordinator can run one cheap stage, fabricate a
    plausible hidden state, and have the remaining stages prove correct
    arithmetic on it.  Chaining the boundaries makes a forgery global: the
    whole activation path from the first stage to the last has to be
    consistent, and the roots are frozen before the challenge nonce is
    revealed.

    A single-stage mesh has no handoff and is exempt.  Everywhere else an
    empty root is a missing commitment, not an excused one.
    """

    by_index: dict[int, "MeshStageProofReceipt"] = {}
    for receipt in receipts:
        index = int(receipt.stage_index)
        seen = by_index.get(index)
        if seen is None:
            by_index[index] = receipt
            continue
        if (
            seen.input_boundary_root != receipt.input_boundary_root
            or seen.output_boundary_root != receipt.output_boundary_root
        ):
            raise mesh_binding_violation(
                f"stage {index} reported two different activation boundaries"
            )
    if stage_count <= 1 or len(by_index) <= 1:
        return

    ordered = sorted(by_index)
    if mesh_boundary_chain_required():
        for index in ordered:
            receipt = by_index[index]
            # The first stage has no upstream handoff and the last has no
            # downstream one, so those two ends are allowed to be open.
            needs_input = index != ordered[0]
            needs_output = index != ordered[-1]
            if needs_input and not receipt.input_boundary_root:
                raise mesh_binding_violation(
                    f"stage {index} is missing its input activation boundary"
                )
            if needs_output and not receipt.output_boundary_root:
                raise mesh_binding_violation(
                    f"stage {index} is missing its output activation boundary"
                )
    for previous, current in zip(ordered, ordered[1:]):
        # Only adjacent stages share a handoff.  When coverage is partial the
        # present indexes can have gaps, and stitching across a gap would
        # reject an honest mesh.
        if current != previous + 1:
            continue
        upstream = by_index[previous].output_boundary_root
        downstream = by_index[current].input_boundary_root
        # Consistency is enforced whether or not presence is required: a mesh
        # that reports roots must have them chain. Only the both-absent case
        # is tolerated during the staged rollout, and the presence loop above
        # closes that once the chain is required.
        if not upstream and not downstream:
            continue
        if upstream != downstream:
            raise mesh_binding_violation(
                f"stage {previous} output activations do not match stage "
                f"{current} input activations"
            )


def llama_graph_receipt_root(receipts: Iterable[LlamaGraphOpReceipt]) -> bytes:
    """Compute a deterministic root over GGML graph proof receipts."""

    ordered = sorted(
        receipts,
        key=lambda item: (
            item.request_id,
            item.stage_index,
            item.layer_index,
            item.op_index,
            item.uid,
            item.endpoint,
        ),
    )
    if not ordered:
        return b""
    h = hashlib.sha256(b"VERATHOS_LLAMA_GRAPH_RECEIPT_ROOT_V1")
    h.update(len(ordered).to_bytes(4, "little"))
    for receipt in ordered:
        receipt.validate()
        h.update(receipt.receipt_hash())
    return h.digest()


def llama_graph_receipt_root_hex(receipts: Iterable[LlamaGraphOpReceipt]) -> str:
    return llama_graph_receipt_root(receipts).hex()


def mesh_response_commitment_hash(receipt: Mapping[str, Any]) -> str:
    """Hash the request/response and mesh assignment fields from a receipt."""

    body = {
        "mesh_id": receipt["mesh_id"],
        "mesh_spec_hash": receipt["mesh_spec_hash"],
        "stage_assignment_hash": receipt["stage_assignment_hash"],
        "rpc_plan_hash": receipt["rpc_plan_hash"],
        "model_package_hash": receipt.get("model_package_hash", ""),
        "model_tensor_manifest_root": receipt.get("model_tensor_manifest_root", ""),
        "uid": receipt["uid"],
        "hotkey": receipt["hotkey"],
        "endpoint": receipt["endpoint"],
        "stage_index": receipt["stage_index"],
        "layer_start": receipt["layer_start"],
        "layer_end": receipt["layer_end"],
        "request_hash": receipt["request_hash"],
        "response_hash": receipt["response_hash"],
    }
    if receipt.get("proof_receipt_root"):
        body["proof_mode"] = receipt.get("proof_mode", "")
        body["proof_receipt_count"] = int(receipt.get("proof_receipt_count", 0))
        body["proof_receipt_root"] = receipt["proof_receipt_root"]
    if receipt.get("proof_postcommit"):
        body["proof_challenge_kind"] = receipt.get(
            "proof_challenge_kind",
            "",
        )
        body["proof_challenge_nonce_commitment"] = receipt.get(
            "proof_challenge_nonce_commitment",
            "",
        )
        body["proof_postcommit_origin_receipt_hash"] = receipt.get(
            "proof_postcommit_origin_receipt_hash",
            "",
        )
    if receipt.get("verified_sampler_mode"):
        body["verified_sampler_mode"] = receipt.get("verified_sampler_mode", "")
        body["verified_sampler_controls_hash"] = receipt.get(
            "verified_sampler_controls_hash",
            "",
        )
    if receipt.get("prompt_token_source"):
        body["prompt_token_ids_hash"] = receipt.get("prompt_token_ids_hash", "")
        body["prompt_token_count"] = int(receipt.get("prompt_token_count", 0))
        body["prompt_token_source"] = receipt.get("prompt_token_source", "")
        body["prompt_template_hash"] = receipt.get("prompt_template_hash", "")
    if receipt.get("completion_token_source"):
        body["completion_token_ids_hash"] = receipt.get("completion_token_ids_hash", "")
        body["completion_token_count"] = int(receipt.get("completion_token_count", 0))
        body["completion_token_source"] = receipt.get("completion_token_source", "")
    if receipt.get("decode_audit_completion_token_ids_hash"):
        body["decode_audit_completion_token_ids_hash"] = receipt.get(
            "decode_audit_completion_token_ids_hash",
            "",
        )
        body["decode_audit_completion_token_count"] = int(
            receipt.get("decode_audit_completion_token_count", 0)
        )
        body["decode_audit_completion_token_source"] = receipt.get(
            "decode_audit_completion_token_source",
            "",
        )
    return hashlib.sha256(
        b"VERATHOS_MESH_RESPONSE_COMMITMENT_V1"
        + json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def mesh_receipt_hash(receipt: Mapping[str, Any]) -> str:
    """Hash the canonical unsigned mesh receipt body.

    ``receipt_hash`` and ``signature`` are envelope fields: the hash is written
    into the former and the latter signs that hash.  Excluding both keeps hash
    recomputation stable after a signature is attached while every substantive
    receipt field remains covered.
    """

    body = dict(receipt)
    body.pop("receipt_hash", None)
    body.pop("signature", None)
    return hashlib.sha256(
        b"VERATHOS_MESH_RECEIPT_V1"
        + json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def verify_llama_graph_proof_receipts(
    receipt: Mapping[str, Any],
    proof_receipts: Iterable[LlamaGraphOpReceipt | Mapping[str, Any]],
) -> None:
    """Verify graph proof receipts are structurally bound to a mesh receipt.

    This verifies transcript identity and Merkle-root binding. It does not
    perform GGML GEMM mathematics; that verifier plugs in behind the same root.
    """

    parsed = [
        item if isinstance(item, LlamaGraphOpReceipt) else LlamaGraphOpReceipt.from_dict(item)
        for item in proof_receipts
    ]
    if not parsed:
        raise RuntimeError("proof receipts are required")
    expected_root = llama_graph_receipt_root_hex(parsed)
    if receipt.get("proof_receipt_root") != expected_root:
        raise RuntimeError("proof_receipt_root mismatch")
    expected = {
        "request_id": receipt.get("request_id", ""),
        "mesh_id": receipt.get("mesh_id", ""),
        "mesh_spec_hash": receipt.get("mesh_spec_hash", ""),
        "stage_assignment_hash": receipt.get("stage_assignment_hash", ""),
        "rpc_plan_hash": receipt.get("rpc_plan_hash", ""),
        "model_package_hash": receipt.get("model_package_hash", ""),
        "model_tensor_manifest_root": receipt.get("model_tensor_manifest_root", ""),
        "uid": receipt.get("uid"),
        "hotkey": receipt.get("hotkey", ""),
        "endpoint": receipt.get("endpoint", ""),
        "stage_index": receipt.get("stage_index"),
        "layer_start": receipt.get("layer_start"),
        "layer_end": receipt.get("layer_end"),
        "request_hash": receipt.get("request_hash", ""),
        "response_hash": receipt.get("response_hash", ""),
    }
    for proof in parsed:
        data = proof.to_dict()
        for field, value in expected.items():
            if data.get(field) != value:
                raise RuntimeError(f"proof receipt {field} mismatch")


def verify_llama_graph_proof_receipts_for_mesh(
    receipt: Mapping[str, Any],
    proof_receipts: Iterable[LlamaGraphOpReceipt | Mapping[str, Any]],
    spec: MeshSpec,
    *,
    required_stage_indexes: Iterable[int] = (),
) -> None:
    """Verify graph proof receipts from any member in a mesh assignment."""

    parsed = [
        item if isinstance(item, LlamaGraphOpReceipt) else LlamaGraphOpReceipt.from_dict(item)
        for item in proof_receipts
    ]
    if not parsed:
        raise RuntimeError("proof receipts are required")
    expected_root = llama_graph_receipt_root_hex(parsed)
    if receipt.get("proof_receipt_root") != expected_root:
        raise RuntimeError("proof_receipt_root mismatch")

    common = {
        "request_id": receipt.get("request_id", ""),
        "mesh_id": spec.mesh_id,
        "mesh_spec_hash": spec.spec_hash_hex(),
        "stage_assignment_hash": spec.stage_assignment_hash_hex(),
        "rpc_plan_hash": receipt.get("rpc_plan_hash", ""),
        "model_package_hash": spec.model_package_hash,
        "model_tensor_manifest_root": spec.model_tensor_manifest_root,
        "request_hash": receipt.get("request_hash", ""),
        "response_hash": receipt.get("response_hash", ""),
    }
    allowed_members = {
        (
            member.uid,
            member.hotkey,
            member.endpoint,
            member.stage_index,
            member.layers.start,
            member.layers.end,
        )
        for member in spec.members
    }
    covered: set[int] = set()
    for proof in parsed:
        data = proof.to_dict()
        for field, value in common.items():
            if data.get(field) != value:
                raise RuntimeError(f"proof receipt {field} mismatch")
        member_key = (
            int(data.get("uid")),
            str(data.get("hotkey", "")),
            str(data.get("endpoint", "")),
            int(data.get("stage_index")),
            int(data.get("layer_start")),
            int(data.get("layer_end")),
        )
        if member_key not in allowed_members:
            raise RuntimeError("proof receipt member assignment mismatch")
        covered.add(int(data.get("stage_index")))

    required = {int(item) for item in required_stage_indexes}
    missing = sorted(required - covered)
    if missing:
        raise RuntimeError(
            "missing proof receipts for mesh stage indexes: "
            + ",".join(str(item) for item in missing)
        )


def is_proof_capable_rpc_worker_binary(binary: str) -> bool:
    """Return true for Verathos RPC binaries that can emit proof transcripts."""

    name = Path(binary).name.lower()
    if name.endswith(".exe"):
        name = name[:-4]
    return name == "verathos-rpc-server" or name.startswith("verathos-rpc-server-")
