"""Endpoint-free, signed verification snapshots for private GGUF meshes.

The runtime :class:`~verallm.mesh.types.MeshSpec` contains private routing
addresses because the coordinator needs them to drive workers.  Validators do
not need those addresses.  This module defines the smaller, immutable protocol
surface a coordinator can sign and a validator can cache without disclosing
the worker topology.

The schema is deliberately closed: serializers emit only the fields declared
here and deserializers reject unknown fields.  Stages are represented by
opaque identifiers, layer ranges, public proof keys, and commitments.  There
is no metadata escape hatch in which an endpoint can be hidden.
"""

from __future__ import annotations

import hashlib
import hmac
import re
import secrets
import time
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

from verallm.mesh.receipt_signing import (
    STAGE_PROOF_KEY_SCHEME,
    _keypair_from_ss58,
    load_hotkey_keypair,
    validate_stage_proof_public_key,
)
from verallm.mesh.types import (
    SUPPORTED_TRACE_MANIFEST_FORMATS,
    MeshSpec,
    canonical_json_bytes,
)


SNAPSHOT_VERSION = 1
SNAPSHOT_SIGNATURE_SCHEME = "sr25519"
SNAPSHOT_BODY_DOMAIN = b"VERATHOS_MESH_VERIFICATION_SNAPSHOT_BODY_V1"
SNAPSHOT_ENVELOPE_DOMAIN = b"VERATHOS_MESH_VERIFICATION_SNAPSHOT_V1"
SNAPSHOT_SIGNATURE_DOMAIN = b"verathos-mesh-verification-snapshot-v1:"
STAGE_COVERAGE_DOMAIN = b"VERATHOS_MESH_VERIFICATION_STAGE_COVERAGE_V1"
STAGE_ID_DOMAIN = b"VERATHOS_MESH_VERIFICATION_STAGE_ID_V1"
STAGE_IDENTITY_COMMITMENT_DOMAIN = (
    b"VERATHOS_MESH_SNAPSHOT_STAGE_IDENTITY_V1"
)
STAGE_PROOF_COMMITMENT_DOMAIN = (
    b"VERATHOS_MESH_SNAPSHOT_PROOF_COMMITMENT_V1"
)

MAX_STAGES = 4096
MAX_TOTAL_LAYERS = 1_000_000
MAX_CONTEXT_LENGTH = 2**32 - 1
MAX_PROOF_OPS_PER_REQUEST = 4096
MIN_SECURE_TRACE_CANDIDATES_PER_REQUEST = 1024
# Producer default for the postcommit hard-audit tier draw. ZERO by
# policy: organic traffic is NEVER hard-audited - light stays light, and
# an operator or user who selected the light tier gets the light tier,
# always. Hard coverage comes exclusively from explicit demands: the
# validator's signed audit_tier=hard canary slots, the deploy probe
# gate, and an operator explicitly choosing hard. (The ambient
# nonce-derived organic draw this rate used to enable made a sampled
# fraction of user chats pay multi-second hard-relation latency.)
MESH_POSTCOMMIT_HARD_AUDIT_BPS = 0
SUPPORTED_STAGE_PROOF_KEY_SCHEMES = frozenset({STAGE_PROOF_KEY_SCHEME})
SUPPORTED_CHALLENGE_SCHEMES = frozenset({"validator_postcommit_v1"})

_HEX_32_RE = re.compile(r"^[0-9a-f]{64}$")
_HEX_SIGNATURE_RE = re.compile(r"^[0-9a-f]{128}$")
_EVM_ADDRESS_RE = re.compile(r"^0x[0-9a-f]{40}$")
_SS58_RE = re.compile(r"^[1-9A-HJ-NP-Za-km-z]{40,64}$")
_OPAQUE_STAGE_ID_RE = re.compile(r"^stg_[0-9a-f]{32}$")
_SAFE_TOKEN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$")
_SAFE_MESH_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
_SAFE_MODEL_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,254}$")
_URL_SCHEME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")
_HOST_PORT_RE = re.compile(
    r"^(?:localhost|[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+|"
    r"(?:\d{1,3}\.){3}\d{1,3}):\d{1,5}$",
    re.IGNORECASE,
)
_IP_LITERAL_RE = re.compile(r"^(?:(?:\d{1,3}\.){3}\d{1,3}|\[[0-9A-Fa-f:]+\])$")

_FORBIDDEN_LOCATION_KEYS = frozenset(
    {
        "endpoint",
        "endpoints",
        "host",
        "hostname",
        "ip",
        "ip_address",
        "port",
        "url",
        "uri",
    }
)
_FORBIDDEN_LOCATION_SUFFIXES = (
    "_endpoint",
    "_endpoints",
    "_host",
    "_hostname",
    "_ip",
    "_port",
    "_url",
    "_uri",
)


def new_opaque_stage_id() -> str:
    """Return a random 128-bit stage identifier with no topology semantics."""

    return "stg_" + secrets.token_hex(16)


def _is_int(value: Any) -> bool:
    """Return true only for JSON integers, never booleans or coercible values."""

    return type(value) is int


def _require_int(value: Any, *, field_name: str) -> int:
    if not _is_int(value):
        raise ValueError(f"{field_name} must be an integer")
    return value


def _require_string(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    return value


def _require_bool(value: Any, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be boolean")
    return value


def _require_exact_keys(
    data: Mapping[str, Any], required: set[str], *, context: str
) -> None:
    if any(not isinstance(key, str) for key in data):
        raise ValueError(f"{context} keys must be strings")
    actual = set(data)
    missing = sorted(required - actual)
    unknown = sorted(actual - required)
    if missing:
        raise ValueError(f"{context} missing fields: {', '.join(missing)}")
    if unknown:
        raise ValueError(f"{context} contains unknown fields: {', '.join(unknown)}")


def _require_hex_32(value: str, *, field_name: str) -> None:
    _require_string(value, field_name=field_name)
    if not _HEX_32_RE.fullmatch(value):
        raise ValueError(f"{field_name} must be a lowercase 32-byte hex digest")


def _require_safe_token(value: str, *, field_name: str) -> None:
    _require_string(value, field_name=field_name)
    if not _SAFE_TOKEN_RE.fullmatch(value):
        raise ValueError(f"{field_name} must be a bounded protocol token")


def derive_opaque_stage_id(
    *,
    mesh_id: str,
    stage_index: int,
    layer_start: int,
    layer_end: int,
    stage_identity_commitment: str,
) -> str:
    """Derive a stable endpoint-free stage ID from committed public facts.

    ``stage_identity_commitment`` must be a canonical 32-byte digest of a
    non-network worker identifier.  Requiring a digest keeps hostnames and
    addresses out of both the snapshot and its deterministic identifiers.
    The 128-bit ID is only an opaque lookup key; the full commitment remains
    in the stage record and provides the cryptographic binding.
    """

    _require_string(mesh_id, field_name="mesh_id")
    if not _SAFE_MESH_ID_RE.fullmatch(mesh_id):
        raise ValueError("mesh_id must be a bounded opaque identifier")
    _require_int(stage_index, field_name="stage_index")
    _require_int(layer_start, field_name="layer_start")
    _require_int(layer_end, field_name="layer_end")
    if not 0 <= stage_index < MAX_STAGES:
        raise ValueError(f"stage_index must be in [0, {MAX_STAGES})")
    if layer_start < 0 or layer_end <= layer_start:
        raise ValueError("stage layers must be a non-empty half-open range")
    if layer_end > MAX_TOTAL_LAYERS:
        raise ValueError(f"layer_end must be <= {MAX_TOTAL_LAYERS}")
    _require_hex_32(
        stage_identity_commitment,
        field_name="stage_identity_commitment",
    )
    material = canonical_json_bytes(
        {
            "mesh_id": mesh_id,
            "stage_index": stage_index,
            "layers": {"start": layer_start, "end": layer_end},
            "stage_identity_commitment": stage_identity_commitment,
        }
    )
    return "stg_" + hashlib.sha256(STAGE_ID_DOMAIN + material).hexdigest()[:32]


def _looks_like_network_location(value: str) -> bool:
    candidate = value.strip()
    return bool(
        _URL_SCHEME_RE.match(candidate)
        or _HOST_PORT_RE.fullmatch(candidate)
        or _IP_LITERAL_RE.fullmatch(candidate)
    )


def assert_endpoint_free_payload(value: Any, *, path: str = "snapshot") -> None:
    """Reject endpoint-bearing keys and obvious network locator values.

    The typed schema already prevents endpoint fields.  This recursive guard is
    also run over every serialized payload so a future schema extension cannot
    accidentally add a worker URL without changing tests and protocol code.
    """

    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            normalized = key.lower()
            if normalized in _FORBIDDEN_LOCATION_KEYS or normalized.endswith(
                _FORBIDDEN_LOCATION_SUFFIXES
            ):
                raise ValueError(f"{path}.{key} is a forbidden network-location field")
            assert_endpoint_free_payload(child, path=f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            assert_endpoint_free_payload(child, path=f"{path}[{index}]")
        return
    if isinstance(value, str) and _looks_like_network_location(value):
        raise ValueError(f"{path} must not contain a network location")


@dataclass(frozen=True)
class MeshVerificationStageBinding:
    """Public proof material supplied for one runtime compute stage.

    The identity commitment must be computed outside this module from a
    non-network stage identity.  The builder deliberately accepts the digest,
    rather than a worker record, so an endpoint cannot accidentally become
    input to an opaque public stage identifier.
    """

    stage_index: int
    stage_identity_commitment: str
    proof_key_scheme: str
    proof_key: str
    proof_commitment: str

    def validate(self) -> None:
        _require_int(self.stage_index, field_name="binding.stage_index")
        if not 0 <= self.stage_index < MAX_STAGES:
            raise ValueError(f"binding.stage_index must be in [0, {MAX_STAGES})")
        _require_hex_32(
            self.stage_identity_commitment,
            field_name="binding.stage_identity_commitment",
        )
        _require_string(
            self.proof_key_scheme,
            field_name="binding.proof_key_scheme",
        )
        if self.proof_key_scheme not in SUPPORTED_STAGE_PROOF_KEY_SCHEMES:
            raise ValueError(
                "unsupported binding proof key scheme: " f"{self.proof_key_scheme}"
            )
        _require_string(self.proof_key, field_name="binding.proof_key")
        validate_stage_proof_public_key(
            self.proof_key,
            self.proof_key_scheme,
        )
        _require_hex_32(
            self.proof_commitment,
            field_name="binding.proof_commitment",
        )
        assert_endpoint_free_payload(
            {
                "stage_index": self.stage_index,
                "stage_identity_commitment": self.stage_identity_commitment,
                "proof_key_scheme": self.proof_key_scheme,
                "proof_key": self.proof_key,
                "proof_commitment": self.proof_commitment,
            },
            path="stage_binding",
        )


def derive_mesh_verification_stage_bindings(
    mesh_spec: MeshSpec,
    *,
    internal_auth_secret: str | bytes,
) -> tuple[MeshVerificationStageBinding, ...]:
    """Derive endpoint-free commitments for every compute stage.

    The operational mesh spec contains private endpoints.  This helper
    deliberately commits only public model/stage facts plus each dedicated
    stage proof key, using the private mesh control secret as the HMAC key.
    Both the standalone snapshot CLI and the pool finalizer use this one
    implementation so their signed public schema cannot drift.
    """

    if not isinstance(mesh_spec, MeshSpec):
        raise ValueError("mesh_spec must be a MeshSpec")
    mesh_spec.validate()
    secret = (
        internal_auth_secret.encode("utf-8")
        if isinstance(internal_auth_secret, str)
        else bytes(internal_auth_secret)
    )
    if not secret:
        raise ValueError("mesh internal auth secret is required for snapshot bindings")

    def commitment(domain: bytes, facts: Mapping[str, Any]) -> str:
        assert_endpoint_free_payload(facts, path="stage_binding_facts")
        return hmac.new(
            secret,
            domain + b":" + canonical_json_bytes(facts),
            hashlib.sha256,
        ).hexdigest()

    bindings: list[MeshVerificationStageBinding] = []
    for member in sorted(mesh_spec.members, key=lambda item: item.stage_index):
        if member.layers.end <= member.layers.start:
            continue
        proof_key = str(member.proof_key or member.hotkey).strip()
        public_stage = {
            "mesh_id": mesh_spec.mesh_id,
            "stage_index": int(member.stage_index),
            "layers": member.layers.to_dict(),
            "proof_key_scheme": STAGE_PROOF_KEY_SCHEME,
            "proof_key": proof_key,
            "model_id": mesh_spec.model_id,
            "model_package_hash": mesh_spec.model_package_hash,
            "model_tensor_manifest_root": mesh_spec.model_tensor_manifest_root,
        }
        proof_facts = {
            "proof_key_scheme": STAGE_PROOF_KEY_SCHEME,
            "proof_key": proof_key,
            "model_package_hash": mesh_spec.model_package_hash,
            "model_tensor_manifest_root": mesh_spec.model_tensor_manifest_root,
            "layers": member.layers.to_dict(),
        }
        binding = MeshVerificationStageBinding(
            stage_index=member.stage_index,
            stage_identity_commitment=commitment(
                STAGE_IDENTITY_COMMITMENT_DOMAIN,
                public_stage,
            ),
            proof_key_scheme=STAGE_PROOF_KEY_SCHEME,
            proof_key=proof_key,
            proof_commitment=commitment(
                STAGE_PROOF_COMMITMENT_DOMAIN,
                proof_facts,
            ),
        )
        binding.validate()
        bindings.append(binding)
    return tuple(bindings)


@dataclass(frozen=True)
class MeshVerificationStage:
    """One expected proof-producing stage, without worker identity or address.

    ``proof_commitment`` is stage-specific snapshot context.  Every public
    proof receipt must repeat it inside the stage-key signature, proving that
    the stage acknowledged the exact signed snapshot rather than an unbound
    coordinator assertion.
    """

    stage_id: str
    layer_start: int
    layer_end: int
    proof_key_scheme: str
    proof_key: str
    proof_commitment: str

    _FIELDS = {
        "stage_id",
        "layers",
        "proof_key_scheme",
        "proof_key",
        "proof_commitment",
    }

    def validate(self, *, total_layers: int | None = None) -> None:
        _require_string(self.stage_id, field_name="stage_id")
        if not _OPAQUE_STAGE_ID_RE.fullmatch(self.stage_id):
            raise ValueError(
                "stage_id must be stg_ followed by 128 bits of lowercase hex"
            )
        _require_int(self.layer_start, field_name="stage.layers.start")
        _require_int(self.layer_end, field_name="stage.layers.end")
        if self.layer_start < 0 or self.layer_end <= self.layer_start:
            raise ValueError("stage layers must be a non-empty half-open range")
        if self.layer_end > MAX_TOTAL_LAYERS:
            raise ValueError(f"stage layer end must be <= {MAX_TOTAL_LAYERS}")
        if total_layers is not None and self.layer_end > total_layers:
            raise ValueError("stage layer range exceeds total_layers")
        _require_string(self.proof_key_scheme, field_name="proof_key_scheme")
        if self.proof_key_scheme not in SUPPORTED_STAGE_PROOF_KEY_SCHEMES:
            raise ValueError(
                f"unsupported stage proof key scheme: {self.proof_key_scheme}"
            )
        _require_string(self.proof_key, field_name="proof_key")
        validate_stage_proof_public_key(
            self.proof_key,
            self.proof_key_scheme,
        )
        _require_hex_32(self.proof_commitment, field_name="proof_commitment")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        payload = {
            "stage_id": self.stage_id,
            "layers": {"start": int(self.layer_start), "end": int(self.layer_end)},
            "proof_key_scheme": self.proof_key_scheme,
            "proof_key": self.proof_key,
            "proof_commitment": self.proof_commitment,
        }
        assert_endpoint_free_payload(payload, path="stage")
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MeshVerificationStage":
        _require_exact_keys(data, cls._FIELDS, context="stage")
        layers = data["layers"]
        if not isinstance(layers, Mapping):
            raise ValueError("stage.layers must be an object")
        _require_exact_keys(layers, {"start", "end"}, context="stage.layers")
        stage = cls(
            stage_id=_require_string(data["stage_id"], field_name="stage_id"),
            layer_start=_require_int(layers["start"], field_name="stage.layers.start"),
            layer_end=_require_int(layers["end"], field_name="stage.layers.end"),
            proof_key_scheme=_require_string(
                data["proof_key_scheme"], field_name="proof_key_scheme"
            ),
            proof_key=_require_string(data["proof_key"], field_name="proof_key"),
            proof_commitment=_require_string(
                data["proof_commitment"], field_name="proof_commitment"
            ),
        )
        stage.validate()
        return stage


@dataclass(frozen=True)
class MeshModelAnchors:
    """Model commitments that every stage proof must ultimately bind to."""

    model_id: str
    model_package_hash: str
    model_tensor_manifest_root: str
    tokenizer_hash: str
    total_layers: int
    max_context_len: int
    quantization_scheme: str
    activation_dtype: str

    _FIELDS = {
        "model_id",
        "model_package_hash",
        "model_tensor_manifest_root",
        "tokenizer_hash",
        "total_layers",
        "max_context_len",
        "quantization_scheme",
        "activation_dtype",
    }

    def validate(self) -> None:
        _require_string(self.model_id, field_name="model_id")
        if not _SAFE_MODEL_ID_RE.fullmatch(self.model_id):
            raise ValueError("model_id must be a bounded registry model identifier")
        if "//" in self.model_id or ".." in self.model_id:
            raise ValueError("model_id contains a forbidden path sequence")
        if _looks_like_network_location(self.model_id):
            raise ValueError("model_id must not be a network location")
        _require_hex_32(self.model_package_hash, field_name="model_package_hash")
        _require_hex_32(
            self.model_tensor_manifest_root,
            field_name="model_tensor_manifest_root",
        )
        _require_hex_32(self.tokenizer_hash, field_name="tokenizer_hash")
        _require_int(self.total_layers, field_name="total_layers")
        if not 1 <= self.total_layers <= MAX_TOTAL_LAYERS:
            raise ValueError(f"total_layers must be in [1, {MAX_TOTAL_LAYERS}]")
        _require_int(self.max_context_len, field_name="max_context_len")
        if not 1 <= self.max_context_len <= MAX_CONTEXT_LENGTH:
            raise ValueError(
                f"max_context_len must be in [1, {MAX_CONTEXT_LENGTH}]"
            )
        _require_safe_token(self.quantization_scheme, field_name="quantization_scheme")
        _require_safe_token(self.activation_dtype, field_name="activation_dtype")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        payload = {
            "model_id": self.model_id,
            "model_package_hash": self.model_package_hash,
            "model_tensor_manifest_root": self.model_tensor_manifest_root,
            "tokenizer_hash": self.tokenizer_hash,
            "total_layers": int(self.total_layers),
            "max_context_len": int(self.max_context_len),
            "quantization_scheme": self.quantization_scheme,
            "activation_dtype": self.activation_dtype,
        }
        assert_endpoint_free_payload(payload, path="model")
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MeshModelAnchors":
        _require_exact_keys(data, cls._FIELDS, context="model")
        anchors = cls(
            model_id=_require_string(data["model_id"], field_name="model_id"),
            model_package_hash=_require_string(
                data["model_package_hash"], field_name="model_package_hash"
            ),
            model_tensor_manifest_root=_require_string(
                data["model_tensor_manifest_root"],
                field_name="model_tensor_manifest_root",
            ),
            tokenizer_hash=_require_string(
                data["tokenizer_hash"], field_name="tokenizer_hash"
            ),
            total_layers=_require_int(data["total_layers"], field_name="total_layers"),
            max_context_len=_require_int(
                data["max_context_len"],
                field_name="max_context_len",
            ),
            quantization_scheme=_require_string(
                data["quantization_scheme"], field_name="quantization_scheme"
            ),
            activation_dtype=_require_string(
                data["activation_dtype"], field_name="activation_dtype"
            ),
        )
        anchors.validate()
        return anchors


@dataclass(frozen=True)
class MeshVerificationPolicy:
    """Proof profile and sampling policy committed by the coordinator."""

    profile: str
    trace_manifest_format: str
    base_proof_sample_bps: int
    organic_decode_sample_bps: int
    canary_decode_sample_bps: int
    proof_ops_per_request: int
    deferred_proof_enabled: bool
    proof_trace_candidates_per_request: int = (
        MIN_SECURE_TRACE_CANDIDATES_PER_REQUEST
    )
    challenge_scheme: str = "validator_postcommit_v1"
    # Post-commit HARD-audit rate for the tier draw. The nonce-derived beacon
    # decides hard vs light against this rate only AFTER the origin receipt
    # froze the response, so the rate carries no phase-one signal. 10000
    # reproduces the pre-tiering behavior (every postcommit audit is hard);
    # producers use MESH_POSTCOMMIT_HARD_AUDIT_BPS. The validator can always
    # be stricter per request via the signed audit_tier=hard demand.
    postcommit_hard_audit_bps: int = 10_000
    version: int = 2

    _FIELDS = {
        "version",
        "profile",
        "trace_manifest_format",
        "base_proof_sample_bps",
        "organic_decode_sample_bps",
        "canary_decode_sample_bps",
        "proof_ops_per_request",
        "proof_trace_candidates_per_request",
        "deferred_proof_enabled",
        "challenge_scheme",
        "postcommit_hard_audit_bps",
    }

    def validate(self) -> None:
        _require_int(self.version, field_name="policy.version")
        if self.version != 2:
            raise ValueError("unsupported mesh verification policy version")
        _require_safe_token(self.profile, field_name="policy.profile")
        _require_safe_token(
            self.trace_manifest_format,
            field_name="policy.trace_manifest_format",
        )
        if self.trace_manifest_format not in SUPPORTED_TRACE_MANIFEST_FORMATS:
            raise ValueError(
                "unsupported policy.trace_manifest_format: "
                f"{self.trace_manifest_format}"
            )
        _require_safe_token(
            self.challenge_scheme,
            field_name="policy.challenge_scheme",
        )
        if self.challenge_scheme not in SUPPORTED_CHALLENGE_SCHEMES:
            raise ValueError(
                "unsupported policy.challenge_scheme: "
                f"{self.challenge_scheme}"
            )
        for field_name, value in (
            ("base_proof_sample_bps", self.base_proof_sample_bps),
            ("organic_decode_sample_bps", self.organic_decode_sample_bps),
            ("canary_decode_sample_bps", self.canary_decode_sample_bps),
            ("postcommit_hard_audit_bps", self.postcommit_hard_audit_bps),
        ):
            _require_int(value, field_name=f"policy.{field_name}")
            if not 0 <= value <= 10_000:
                raise ValueError(f"{field_name} must be in [0, 10000]")
        _require_int(
            self.proof_ops_per_request,
            field_name="policy.proof_ops_per_request",
        )
        if not 1 <= self.proof_ops_per_request <= MAX_PROOF_OPS_PER_REQUEST:
            raise ValueError(
                f"proof_ops_per_request must be in [1, {MAX_PROOF_OPS_PER_REQUEST}]"
            )
        _require_int(
            self.proof_trace_candidates_per_request,
            field_name="policy.proof_trace_candidates_per_request",
        )
        if not (
            MIN_SECURE_TRACE_CANDIDATES_PER_REQUEST
            <= self.proof_trace_candidates_per_request
            <= MAX_PROOF_OPS_PER_REQUEST
        ):
            raise ValueError(
                "proof_trace_candidates_per_request must be in "
                f"[{MIN_SECURE_TRACE_CANDIDATES_PER_REQUEST}, "
                f"{MAX_PROOF_OPS_PER_REQUEST}]"
            )
        if self.proof_trace_candidates_per_request < self.proof_ops_per_request:
            raise ValueError(
                "proof_trace_candidates_per_request must cover proof_ops_per_request"
            )
        if self.organic_decode_sample_bps != self.canary_decode_sample_bps:
            raise ValueError(
                "organic_decode_sample_bps and canary_decode_sample_bps must "
                "be equal so the decode audit rate carries no signal about "
                "which requests are canaries"
            )
        _require_bool(
            self.deferred_proof_enabled,
            field_name="policy.deferred_proof_enabled",
        )

    @property
    def decode_sample_bps(self) -> int:
        """Return the single decode-audit rate used by every request path.

        The two underlying fields are kept for wire compatibility but
        ``validate`` requires them to be equal.  Request builders must read
        this instead of either field: a canary and an organic request have to
        put the identical value on the wire, or the coordinator can tell them
        apart at phase one and serve canaries honestly while cheating
        everything else.
        """

        if self.organic_decode_sample_bps != self.canary_decode_sample_bps:
            raise ValueError(
                "mesh verification policy decode rates disagree"
            )
        return int(self.organic_decode_sample_bps)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        payload = {
            "version": int(self.version),
            "profile": self.profile,
            "trace_manifest_format": self.trace_manifest_format,
            "base_proof_sample_bps": int(self.base_proof_sample_bps),
            "organic_decode_sample_bps": int(self.organic_decode_sample_bps),
            "canary_decode_sample_bps": int(self.canary_decode_sample_bps),
            "proof_ops_per_request": int(self.proof_ops_per_request),
            "proof_trace_candidates_per_request": int(
                self.proof_trace_candidates_per_request
            ),
            "deferred_proof_enabled": bool(self.deferred_proof_enabled),
            "challenge_scheme": self.challenge_scheme,
            "postcommit_hard_audit_bps": int(self.postcommit_hard_audit_bps),
        }
        assert_endpoint_free_payload(payload, path="policy")
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MeshVerificationPolicy":
        # Pre-tiering snapshots have no hard-audit rate; absent means the
        # old behavior (every postcommit audit is hard).
        data = dict(data)
        data.setdefault("postcommit_hard_audit_bps", 10_000)
        _require_exact_keys(data, cls._FIELDS, context="policy")
        policy = cls(
            version=_require_int(data["version"], field_name="policy.version"),
            profile=_require_string(data["profile"], field_name="policy.profile"),
            trace_manifest_format=_require_string(
                data["trace_manifest_format"],
                field_name="policy.trace_manifest_format",
            ),
            base_proof_sample_bps=_require_int(
                data["base_proof_sample_bps"],
                field_name="policy.base_proof_sample_bps",
            ),
            organic_decode_sample_bps=_require_int(
                data["organic_decode_sample_bps"],
                field_name="policy.organic_decode_sample_bps",
            ),
            canary_decode_sample_bps=_require_int(
                data["canary_decode_sample_bps"],
                field_name="policy.canary_decode_sample_bps",
            ),
            proof_ops_per_request=_require_int(
                data["proof_ops_per_request"],
                field_name="policy.proof_ops_per_request",
            ),
            proof_trace_candidates_per_request=_require_int(
                data["proof_trace_candidates_per_request"],
                field_name="policy.proof_trace_candidates_per_request",
            ),
            deferred_proof_enabled=_require_bool(
                data["deferred_proof_enabled"],
                field_name="policy.deferred_proof_enabled",
            ),
            challenge_scheme=_require_string(
                data["challenge_scheme"],
                field_name="policy.challenge_scheme",
            ),
            postcommit_hard_audit_bps=_require_int(
                data["postcommit_hard_audit_bps"],
                field_name="policy.postcommit_hard_audit_bps",
            ),
        )
        policy.validate()
        return policy


@dataclass(frozen=True)
class MeshCoordinatorIdentity:
    """The exact chain slot and hotkey authority behind a mesh snapshot."""

    chain_id: int
    netuid: int
    coordinator_uid: int
    coordinator_hotkey: str
    coordinator_evm_address: str
    model_index: int

    _FIELDS = {
        "chain_id",
        "netuid",
        "coordinator_uid",
        "coordinator_hotkey",
        "coordinator_evm_address",
        "model_index",
    }

    def validate(self) -> None:
        _require_int(self.chain_id, field_name="coordinator.chain_id")
        if not 1 <= self.chain_id < 2**64:
            raise ValueError("chain_id must be in [1, 2^64)")
        _require_int(self.netuid, field_name="coordinator.netuid")
        if not 0 <= self.netuid <= 65_535:
            raise ValueError("netuid must fit uint16")
        _require_int(self.coordinator_uid, field_name="coordinator.coordinator_uid")
        if not 0 <= self.coordinator_uid < 2**32:
            raise ValueError("coordinator_uid must fit uint32")
        _require_string(
            self.coordinator_hotkey,
            field_name="coordinator.coordinator_hotkey",
        )
        if not _SS58_RE.fullmatch(self.coordinator_hotkey):
            raise ValueError("coordinator_hotkey must be a bounded SS58 string")
        _require_string(
            self.coordinator_evm_address,
            field_name="coordinator.coordinator_evm_address",
        )
        if not _EVM_ADDRESS_RE.fullmatch(self.coordinator_evm_address):
            raise ValueError(
                "coordinator_evm_address must be a lowercase 20-byte EVM address"
            )
        _require_int(self.model_index, field_name="coordinator.model_index")
        if not 0 <= self.model_index < 2**32:
            raise ValueError("model_index must fit uint32")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        payload = {
            "chain_id": int(self.chain_id),
            "netuid": int(self.netuid),
            "coordinator_uid": int(self.coordinator_uid),
            "coordinator_hotkey": self.coordinator_hotkey,
            "coordinator_evm_address": self.coordinator_evm_address.lower(),
            "model_index": int(self.model_index),
        }
        assert_endpoint_free_payload(payload, path="coordinator")
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MeshCoordinatorIdentity":
        _require_exact_keys(data, cls._FIELDS, context="coordinator")
        identity = cls(
            chain_id=_require_int(data["chain_id"], field_name="coordinator.chain_id"),
            netuid=_require_int(data["netuid"], field_name="coordinator.netuid"),
            coordinator_uid=_require_int(
                data["coordinator_uid"], field_name="coordinator.coordinator_uid"
            ),
            coordinator_hotkey=_require_string(
                data["coordinator_hotkey"],
                field_name="coordinator.coordinator_hotkey",
            ),
            coordinator_evm_address=_require_string(
                data["coordinator_evm_address"],
                field_name="coordinator.coordinator_evm_address",
            ),
            model_index=_require_int(
                data["model_index"], field_name="coordinator.model_index"
            ),
        )
        identity.validate()
        return identity


@dataclass(frozen=True)
class MeshVerificationSnapshot:
    """Immutable signed view of the facts needed to verify a private mesh."""

    mesh_id: str
    generation: int
    epoch: int
    issued_at_unix: int
    expires_at_unix: int
    coordinator: MeshCoordinatorIdentity
    model: MeshModelAnchors
    policy: MeshVerificationPolicy
    expected_stage_count: int
    stages: tuple[MeshVerificationStage, ...]
    signature: str = ""
    signature_scheme: str = SNAPSHOT_SIGNATURE_SCHEME
    version: int = SNAPSHOT_VERSION

    _FIELDS = {
        "version",
        "mesh_id",
        "generation",
        "epoch",
        "issued_at_unix",
        "expires_at_unix",
        "coordinator",
        "model",
        "policy",
        "expected_stage_count",
        "stages",
        "signature_scheme",
        "signature",
    }

    def _ordered_stages(self) -> tuple[MeshVerificationStage, ...]:
        return tuple(
            sorted(
                self.stages,
                key=lambda stage: (stage.layer_start, stage.layer_end, stage.stage_id),
            )
        )

    def validate(self, *, require_signature: bool = False) -> None:
        _require_int(self.version, field_name="snapshot.version")
        if self.version != SNAPSHOT_VERSION:
            raise ValueError("unsupported mesh verification snapshot version")
        _require_string(self.mesh_id, field_name="snapshot.mesh_id")
        if not _SAFE_MESH_ID_RE.fullmatch(self.mesh_id):
            raise ValueError("mesh_id must be a bounded opaque identifier")
        _require_int(self.generation, field_name="snapshot.generation")
        if not 1 <= self.generation < 2**63:
            raise ValueError("generation must be in [1, 2^63)")
        _require_int(self.epoch, field_name="snapshot.epoch")
        if not 0 <= self.epoch < 2**63:
            raise ValueError("epoch must be in [0, 2^63)")
        _require_int(self.issued_at_unix, field_name="snapshot.issued_at_unix")
        _require_int(self.expires_at_unix, field_name="snapshot.expires_at_unix")
        if not 1 <= self.issued_at_unix < 2**63:
            raise ValueError("issued_at_unix must be in [1, 2^63)")
        if not 1 <= self.expires_at_unix < 2**63:
            raise ValueError("expires_at_unix must be in [1, 2^63)")
        if self.expires_at_unix <= self.issued_at_unix:
            raise ValueError("expires_at_unix must be after issued_at_unix")
        if not isinstance(self.coordinator, MeshCoordinatorIdentity):
            raise ValueError("coordinator must be a MeshCoordinatorIdentity")
        if not isinstance(self.model, MeshModelAnchors):
            raise ValueError("model must be MeshModelAnchors")
        if not isinstance(self.policy, MeshVerificationPolicy):
            raise ValueError("policy must be MeshVerificationPolicy")
        self.coordinator.validate()
        self.model.validate()
        self.policy.validate()
        _require_int(
            self.expected_stage_count,
            field_name="snapshot.expected_stage_count",
        )
        if not 1 <= self.expected_stage_count <= MAX_STAGES:
            raise ValueError(f"expected_stage_count must be in [1, {MAX_STAGES}]")
        if not isinstance(self.stages, tuple):
            raise ValueError("stages must be an immutable tuple")
        if len(self.stages) != self.expected_stage_count:
            raise ValueError("expected_stage_count does not match stages")
        if any(not isinstance(stage, MeshVerificationStage) for stage in self.stages):
            raise ValueError("stages must contain MeshVerificationStage values")

        ordered = self._ordered_stages()
        seen: set[str] = set()
        seen_proof_keys: set[tuple[str, bytes]] = set()
        seen_proof_commitments: set[str] = set()
        try:
            coordinator_public_key = bytes(
                getattr(
                    _keypair_from_ss58(self.coordinator.coordinator_hotkey),
                    "public_key",
                    b"",
                )
            )
        except Exception as exc:
            raise ValueError("coordinator hotkey must be a valid SS58 key") from exc
        if len(coordinator_public_key) != 32:
            raise ValueError("coordinator hotkey must encode a 32-byte public key")
        cursor = 0
        for stage in ordered:
            stage.validate(total_layers=self.model.total_layers)
            if stage.stage_id in seen:
                raise ValueError("stage_id values must be unique")
            seen.add(stage.stage_id)
            proof_public_key = bytes(
                getattr(_keypair_from_ss58(stage.proof_key), "public_key", b"")
            )
            if len(proof_public_key) != 32:
                raise ValueError("stage proof key must encode a 32-byte public key")
            proof_key_identity = (stage.proof_key_scheme, proof_public_key)
            if proof_key_identity in seen_proof_keys:
                raise ValueError("every stage must use a dedicated proof key")
            if proof_public_key == coordinator_public_key:
                raise ValueError(
                    "stage proof keys must be distinct from the coordinator hotkey"
                )
            if stage.proof_commitment in seen_proof_commitments:
                raise ValueError("every stage must use a dedicated proof commitment")
            seen_proof_keys.add(proof_key_identity)
            seen_proof_commitments.add(stage.proof_commitment)
            if stage.layer_start != cursor:
                raise ValueError("stages must cover every model layer exactly once")
            cursor = stage.layer_end
        if cursor != self.model.total_layers:
            raise ValueError("stages must cover every model layer exactly once")

        _require_string(self.signature_scheme, field_name="snapshot.signature_scheme")
        if self.signature_scheme != SNAPSHOT_SIGNATURE_SCHEME:
            raise ValueError("unsupported snapshot signature scheme")
        _require_string(self.signature, field_name="snapshot.signature")
        if require_signature and not self.signature:
            raise ValueError("snapshot signature is required")
        if self.signature and not _HEX_SIGNATURE_RE.fullmatch(self.signature):
            raise ValueError(
                "signature must be a lowercase 64-byte hex Sr25519 signature"
            )

    def validate_freshness(
        self,
        *,
        now_unix: int,
        expected_epoch: int | None = None,
        max_future_skew_s: int = 30,
    ) -> None:
        """Validate time and optional epoch bounds without consulting the chain."""

        self.validate()
        _require_int(now_unix, field_name="now_unix")
        _require_int(max_future_skew_s, field_name="max_future_skew_s")
        if max_future_skew_s < 0:
            raise ValueError("max_future_skew_s must be >= 0")
        if now_unix + max_future_skew_s < self.issued_at_unix:
            raise ValueError("snapshot was issued too far in the future")
        if now_unix >= self.expires_at_unix:
            raise ValueError("snapshot has expired")
        if expected_epoch is not None:
            _require_int(expected_epoch, field_name="expected_epoch")
            if self.epoch != expected_epoch:
                raise ValueError("snapshot epoch does not match expected epoch")

    def validate_expected_bindings(
        self,
        *,
        expected_mesh_id: str | None = None,
        expected_generation: int | None = None,
        expected_coordinator: MeshCoordinatorIdentity | None = None,
        expected_model: MeshModelAnchors | None = None,
        expected_policy: MeshVerificationPolicy | None = None,
    ) -> None:
        """Bind a valid snapshot to validator/chain-approved exact values."""

        self.validate()
        if expected_mesh_id is not None:
            _require_string(expected_mesh_id, field_name="expected_mesh_id")
            if self.mesh_id != expected_mesh_id:
                raise ValueError("snapshot mesh_id does not match expected mesh_id")
        if expected_generation is not None:
            _require_int(expected_generation, field_name="expected_generation")
            if self.generation != expected_generation:
                raise ValueError(
                    "snapshot generation does not match expected generation"
                )
        if expected_coordinator is not None:
            if not isinstance(expected_coordinator, MeshCoordinatorIdentity):
                raise ValueError("expected_coordinator must be MeshCoordinatorIdentity")
            expected_coordinator.validate()
            if self.coordinator != expected_coordinator:
                raise ValueError(
                    "snapshot coordinator does not match expected coordinator"
                )
        if expected_model is not None:
            if not isinstance(expected_model, MeshModelAnchors):
                raise ValueError("expected_model must be MeshModelAnchors")
            expected_model.validate()
            if self.model != expected_model:
                raise ValueError("snapshot model does not match expected model")
        if expected_policy is not None:
            if not isinstance(expected_policy, MeshVerificationPolicy):
                raise ValueError("expected_policy must be MeshVerificationPolicy")
            expected_policy.validate()
            if self.policy != expected_policy:
                raise ValueError("snapshot policy does not match expected policy")

    def to_dict(self, *, include_signature: bool = True) -> dict[str, Any]:
        self.validate(require_signature=False)
        payload: dict[str, Any] = {
            "version": int(self.version),
            "mesh_id": self.mesh_id,
            "generation": int(self.generation),
            "epoch": int(self.epoch),
            "issued_at_unix": int(self.issued_at_unix),
            "expires_at_unix": int(self.expires_at_unix),
            "coordinator": self.coordinator.to_dict(),
            "model": self.model.to_dict(),
            "policy": self.policy.to_dict(),
            "expected_stage_count": int(self.expected_stage_count),
            "stages": [stage.to_dict() for stage in self._ordered_stages()],
            "signature_scheme": self.signature_scheme,
        }
        if include_signature:
            payload["signature"] = self.signature
        assert_endpoint_free_payload(payload)
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MeshVerificationSnapshot":
        # Run the privacy guard before reporting ordinary schema errors so an
        # endpoint-bearing extension is always rejected explicitly, including
        # when its field is otherwise unknown to this version of the schema.
        assert_endpoint_free_payload(data)
        _require_exact_keys(data, cls._FIELDS, context="snapshot")
        coordinator = data["coordinator"]
        model = data["model"]
        policy = data["policy"]
        stages = data["stages"]
        if not isinstance(coordinator, Mapping):
            raise ValueError("snapshot.coordinator must be an object")
        if not isinstance(model, Mapping):
            raise ValueError("snapshot.model must be an object")
        if not isinstance(policy, Mapping):
            raise ValueError("snapshot.policy must be an object")
        if not isinstance(stages, Sequence) or isinstance(stages, (str, bytes)):
            raise ValueError("snapshot.stages must be an array")
        if len(stages) > MAX_STAGES:
            raise ValueError(f"snapshot.stages exceeds the {MAX_STAGES} stage limit")
        if any(not isinstance(stage, Mapping) for stage in stages):
            raise ValueError("every snapshot stage must be an object")
        snapshot = cls(
            version=_require_int(data["version"], field_name="snapshot.version"),
            mesh_id=_require_string(data["mesh_id"], field_name="snapshot.mesh_id"),
            generation=_require_int(
                data["generation"], field_name="snapshot.generation"
            ),
            epoch=_require_int(data["epoch"], field_name="snapshot.epoch"),
            issued_at_unix=_require_int(
                data["issued_at_unix"], field_name="snapshot.issued_at_unix"
            ),
            expires_at_unix=_require_int(
                data["expires_at_unix"], field_name="snapshot.expires_at_unix"
            ),
            coordinator=MeshCoordinatorIdentity.from_dict(coordinator),
            model=MeshModelAnchors.from_dict(model),
            policy=MeshVerificationPolicy.from_dict(policy),
            expected_stage_count=_require_int(
                data["expected_stage_count"],
                field_name="snapshot.expected_stage_count",
            ),
            stages=tuple(MeshVerificationStage.from_dict(stage) for stage in stages),
            signature_scheme=_require_string(
                data["signature_scheme"], field_name="snapshot.signature_scheme"
            ),
            signature=_require_string(
                data["signature"], field_name="snapshot.signature"
            ),
        )
        snapshot.validate()
        return snapshot

    def body_hash(self) -> bytes:
        return hashlib.sha256(
            SNAPSHOT_BODY_DOMAIN
            + canonical_json_bytes(self.to_dict(include_signature=False))
        ).digest()

    def body_hash_hex(self) -> str:
        return self.body_hash().hex()

    def snapshot_hash(self) -> bytes:
        return hashlib.sha256(
            SNAPSHOT_ENVELOPE_DOMAIN
            + canonical_json_bytes(self.to_dict(include_signature=True))
        ).digest()

    def snapshot_hash_hex(self) -> str:
        return self.snapshot_hash().hex()

    def stage_coverage_hash(self) -> bytes:
        self.validate()
        coverage = {
            "total_layers": int(self.model.total_layers),
            "expected_stage_count": int(self.expected_stage_count),
            "stages": [stage.to_dict() for stage in self._ordered_stages()],
        }
        return hashlib.sha256(
            STAGE_COVERAGE_DOMAIN + canonical_json_bytes(coverage)
        ).digest()

    def stage_coverage_hash_hex(self) -> str:
        return self.stage_coverage_hash().hex()

    @property
    def expected_stage_ids(self) -> tuple[str, ...]:
        self.validate()
        return tuple(stage.stage_id for stage in self._ordered_stages())


def build_mesh_verification_snapshot(
    mesh_spec: MeshSpec,
    *,
    coordinator: MeshCoordinatorIdentity,
    policy: MeshVerificationPolicy,
    generation: int,
    epoch: int,
    issued_at_unix: int,
    expires_at_unix: int,
    stage_bindings: Sequence[MeshVerificationStageBinding],
) -> MeshVerificationSnapshot:
    """Build an unsigned endpoint-free snapshot from an operational mesh.

    ``mesh_spec`` remains the private routing document.  Only its model
    anchors, opaque mesh identifier, and compute-stage layer ranges are copied.
    Callers must provide validator/chain-approved coordinator and policy values
    plus one public binding for every non-empty stage.
    """

    if not isinstance(mesh_spec, MeshSpec):
        raise ValueError("mesh_spec must be a MeshSpec")
    mesh_spec.validate()
    if not isinstance(coordinator, MeshCoordinatorIdentity):
        raise ValueError("coordinator must be a MeshCoordinatorIdentity")
    coordinator.validate()
    if not isinstance(policy, MeshVerificationPolicy):
        raise ValueError("policy must be a MeshVerificationPolicy")
    policy.validate()

    _require_int(generation, field_name="generation")
    _require_int(epoch, field_name="epoch")
    _require_int(issued_at_unix, field_name="issued_at_unix")
    _require_int(expires_at_unix, field_name="expires_at_unix")

    if coordinator.coordinator_uid != mesh_spec.coordinator_uid:
        raise ValueError("coordinator uid does not match mesh_spec")
    if coordinator.coordinator_hotkey != mesh_spec.coordinator_hotkey:
        raise ValueError("coordinator hotkey does not match mesh_spec")
    if epoch != mesh_spec.epoch:
        raise ValueError("snapshot epoch does not match mesh_spec epoch")
    if policy.trace_manifest_format != mesh_spec.proof_trace_manifest_format:
        raise ValueError(
            "policy trace_manifest_format does not match "
            "mesh_spec.proof_trace_manifest_format"
        )

    if not mesh_spec.model_package_hash:
        raise ValueError("mesh_spec.model_package_hash is required")
    if not mesh_spec.model_tensor_manifest_root:
        raise ValueError("mesh_spec.model_tensor_manifest_root is required")
    if not mesh_spec.tokenizer_hash:
        raise ValueError("mesh_spec.tokenizer_hash is required")

    if not isinstance(stage_bindings, Sequence) or isinstance(
        stage_bindings, (str, bytes, bytearray)
    ):
        raise ValueError("stage_bindings must be a sequence")
    if len(stage_bindings) > MAX_STAGES:
        raise ValueError(f"stage_bindings exceeds the {MAX_STAGES} stage limit")

    compute_members = {
        member.stage_index: member
        for member in mesh_spec.members
        if member.layers.end > member.layers.start
    }
    all_members = {member.stage_index: member for member in mesh_spec.members}
    bindings_by_index: dict[int, MeshVerificationStageBinding] = {}
    for binding in stage_bindings:
        if not isinstance(binding, MeshVerificationStageBinding):
            raise ValueError(
                "stage_bindings must contain MeshVerificationStageBinding values"
            )
        binding.validate()
        if binding.stage_index in bindings_by_index:
            raise ValueError(
                f"duplicate stage binding for stage_index {binding.stage_index}"
            )
        bindings_by_index[binding.stage_index] = binding

    bound_indexes = set(bindings_by_index)
    compute_indexes = set(compute_members)
    orchestration_indexes = {
        stage_index
        for stage_index, member in all_members.items()
        if member.layers.end == member.layers.start
    }
    bound_orchestration = sorted(bound_indexes & orchestration_indexes)
    if bound_orchestration:
        raise ValueError(
            "stage bindings must not include orchestration-only stage indexes: "
            + ", ".join(str(index) for index in bound_orchestration)
        )
    missing = sorted(compute_indexes - bound_indexes)
    if missing:
        raise ValueError(
            "missing bindings for compute stage indexes: "
            + ", ".join(str(index) for index in missing)
        )
    extra = sorted(bound_indexes - compute_indexes)
    if extra:
        raise ValueError(
            "bindings contain unknown or non-compute stage indexes: "
            + ", ".join(str(index) for index in extra)
        )

    model = MeshModelAnchors(
        model_id=mesh_spec.model_id,
        model_package_hash=mesh_spec.model_package_hash,
        model_tensor_manifest_root=mesh_spec.model_tensor_manifest_root,
        tokenizer_hash=mesh_spec.tokenizer_hash,
        total_layers=mesh_spec.total_layers,
        max_context_len=mesh_spec.max_context_len,
        quantization_scheme=mesh_spec.quantization_scheme,
        activation_dtype=mesh_spec.activation_dtype,
    )
    model.validate()

    stages: list[MeshVerificationStage] = []
    for member in sorted(
        compute_members.values(),
        key=lambda item: (item.layers.start, item.layers.end, item.stage_index),
    ):
        binding = bindings_by_index[member.stage_index]
        stages.append(
            MeshVerificationStage(
                stage_id=derive_opaque_stage_id(
                    mesh_id=mesh_spec.mesh_id,
                    stage_index=member.stage_index,
                    layer_start=member.layers.start,
                    layer_end=member.layers.end,
                    stage_identity_commitment=(binding.stage_identity_commitment),
                ),
                layer_start=member.layers.start,
                layer_end=member.layers.end,
                proof_key_scheme=binding.proof_key_scheme,
                proof_key=binding.proof_key,
                proof_commitment=binding.proof_commitment,
            )
        )

    snapshot = MeshVerificationSnapshot(
        mesh_id=mesh_spec.mesh_id,
        generation=generation,
        epoch=epoch,
        issued_at_unix=issued_at_unix,
        expires_at_unix=expires_at_unix,
        coordinator=coordinator,
        model=model,
        policy=policy,
        expected_stage_count=len(stages),
        stages=tuple(stages),
    )
    snapshot.validate()
    assert_endpoint_free_payload(snapshot.to_dict())
    return snapshot


def snapshot_signature_message(snapshot: MeshVerificationSnapshot) -> bytes:
    """Return the domain-separated bytes signed by the coordinator hotkey."""

    snapshot.validate()
    return SNAPSHOT_SIGNATURE_DOMAIN + snapshot.body_hash_hex().encode("ascii")


def sign_mesh_verification_snapshot(
    snapshot: MeshVerificationSnapshot, keypair: Any
) -> MeshVerificationSnapshot:
    """Return a copy signed by the snapshot's declared coordinator hotkey."""

    snapshot.validate()
    signer = getattr(keypair, "ss58_address", "")
    if not isinstance(signer, str):
        raise ValueError("signing key ss58_address must be a string")
    if signer != snapshot.coordinator.coordinator_hotkey:
        raise ValueError("signing key does not match coordinator_hotkey")
    signature = keypair.sign(snapshot_signature_message(snapshot))
    if isinstance(signature, (bytes, bytearray)):
        signature_hex = bytes(signature).hex()
    else:
        signature_hex = str(signature)
        if signature_hex.startswith("0x"):
            signature_hex = signature_hex[2:]
    signed = replace(snapshot, signature=signature_hex.lower())
    signed.validate(require_signature=True)
    return signed


def rotate_mesh_verification_snapshot(
    existing: MeshVerificationSnapshot,
    *,
    epoch: int,
    keypair: Any,
    now_unix: int | None = None,
) -> MeshVerificationSnapshot:
    """Re-sign a snapshot for a new scoring epoch, content unchanged.

    Validators pin one signed snapshot per epoch (the anti-replay guard),
    so a long-lived mesh refreshes the binding at every boundary: same
    mesh, model anchors, stages, and policy; only epoch, generation, and
    the freshness window move. Rotating an already EXPIRED snapshot is
    valid by design (that is the recovery case), so the input's freshness
    is not enforced. Returns ``existing`` unchanged when it already binds
    ``epoch`` so redelivered rotation commands never burn a generation.
    """

    _require_int(epoch, field_name="epoch")
    if epoch < 0 or epoch >= 2**63:
        raise ValueError("epoch must be an integer in [0, 2^63)")
    if int(existing.epoch) == int(epoch):
        return existing
    issued_at = int(time.time()) if now_unix is None else int(now_unix)
    ttl_seconds = max(
        60, int(existing.expires_at_unix) - int(existing.issued_at_unix)
    )
    rotated = replace(
        existing,
        epoch=int(epoch),
        generation=int(existing.generation) + 1,
        issued_at_unix=issued_at,
        expires_at_unix=issued_at + ttl_seconds,
        signature="",
    )
    return sign_mesh_verification_snapshot(rotated, keypair)


def sign_mesh_verification_snapshot_with_wallet(
    snapshot: MeshVerificationSnapshot,
    *,
    wallet_name: str,
    hotkey_name: str,
) -> MeshVerificationSnapshot:
    """Load an existing Bittensor hotkey and sign without touching a coldkey."""

    return sign_mesh_verification_snapshot(
        snapshot,
        load_hotkey_keypair(wallet_name, hotkey_name),
    )


def verify_mesh_verification_snapshot_signature(
    snapshot: MeshVerificationSnapshot,
    *,
    expected_hotkey: str | None = None,
    expected_epoch: int | None = None,
    expected_mesh_id: str | None = None,
    expected_generation: int | None = None,
    expected_coordinator: MeshCoordinatorIdentity | None = None,
    expected_model: MeshModelAnchors | None = None,
    expected_policy: MeshVerificationPolicy | None = None,
    now_unix: int | None = None,
    max_future_skew_s: int = 30,
) -> bool:
    """Verify structure, exact approved bindings, freshness, and signature.

    The optional expected objects are compared in full.  Validators should
    pass the chain-resolved coordinator/model values and their accepted proof
    policy, rather than treating a self-signed declaration as authorization.
    """

    try:
        snapshot.validate(require_signature=True)
        snapshot.validate_expected_bindings(
            expected_mesh_id=expected_mesh_id,
            expected_generation=expected_generation,
            expected_coordinator=expected_coordinator,
            expected_model=expected_model,
            expected_policy=expected_policy,
        )
        if expected_hotkey is not None:
            _require_string(expected_hotkey, field_name="expected_hotkey")
            if snapshot.coordinator.coordinator_hotkey != expected_hotkey:
                return False
        if expected_epoch is not None:
            _require_int(expected_epoch, field_name="expected_epoch")
            if snapshot.epoch != expected_epoch:
                return False
        if now_unix is not None:
            snapshot.validate_freshness(
                now_unix=now_unix,
                expected_epoch=expected_epoch,
                max_future_skew_s=max_future_skew_s,
            )
        keypair = _keypair_from_ss58(snapshot.coordinator.coordinator_hotkey)
        return bool(
            keypair.verify(
                snapshot_signature_message(snapshot),
                bytes.fromhex(snapshot.signature),
            )
        )
    except Exception:
        return False


__all__ = [
    "MAX_STAGES",
    "MeshCoordinatorIdentity",
    "MeshModelAnchors",
    "MeshVerificationPolicy",
    "MeshVerificationSnapshot",
    "MeshVerificationStage",
    "MeshVerificationStageBinding",
    "SNAPSHOT_SIGNATURE_SCHEME",
    "SNAPSHOT_VERSION",
    "assert_endpoint_free_payload",
    "build_mesh_verification_snapshot",
    "derive_mesh_verification_stage_bindings",
    "derive_opaque_stage_id",
    "new_opaque_stage_id",
    "rotate_mesh_verification_snapshot",
    "sign_mesh_verification_snapshot",
    "sign_mesh_verification_snapshot_with_wallet",
    "snapshot_signature_message",
    "verify_mesh_verification_snapshot_signature",
]
