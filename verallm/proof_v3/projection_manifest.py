"""Signed per-projection weight-root manifest for proof-v3 audits.

The manifest is the AUTHENTICATED set of per-projection weight roots the
validator checks miner reveals against.  It is:

* PUBLIC -- anyone can fetch it, verify the signature, and run an audit;
  the roots are not secret.
* TRUSTED only if signed by the subnet-owner authority (EVM ECDSA, the
  same key set that owns the contracts).  The signature is the sole trust
  anchor, so the manifest can be hosted anywhere (HTTPS, object storage, HF) with
  NO contract change -- a tampered copy fails signature verification.

This closes the one gap in the recompute audit: the weight root the
validator trusts comes from THIS signed manifest, never from the (untrusted)
miner.  Reuses the ECDSA scheme of proof_v2.manifest (encode_defunct +
recover_message, 65-byte canonical signatures).
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import dataclass

from verallm.proof_v3.errors import ProofV3Error, ProofV3VerificationError
from verallm.proof_v3.rmsnorm_runtime_semantics import (
    DEFAULT_RMSNORM_EPSILON_BITS_V3,
    RMSNORM_WEIGHT_GAIN_V3,
    validate_rmsnorm_runtime_semantics_v3,
)

_MANIFEST_DOMAIN = b"VERATHOS/PROOF_V3/PROJECTION_MANIFEST/V1"
LM_HEAD_CATALOG_BINDING_V3 = 2
LM_HEAD_CATALOG_ABI_V3 = "pallas.lm_head.catalog.v1"

__all__ = [
    "LM_HEAD_CATALOG_ABI_V3",
    "LM_HEAD_CATALOG_BINDING_V3",
    "ProjectionManifestEntryV3",
    "ProjectionManifestV3",
    "build_projection_manifest_v3",
    "sign_projection_manifest_v3",
    "verify_projection_manifest_v3",
]


@dataclass(frozen=True, slots=True)
class ProjectionManifestEntryV3:
    """One projection's authenticated weight root."""

    name: str            # "L{i}.{proj}"
    root: bytes          # FlatWeightMerkle root of the int8 weight
    orientation: str     # "out_in" (contiguous rows) | "in_out" (strided cols)
    in_dim: int
    out_dim: int
    # SIGNED absmax/127 dequant scale of the int8 weight as IEEE-754 bits
    # (little-endian u64 of the float64).  0 = legacy entry without a
    # signed scale; corridor checks REQUIRE a nonzero signed scale.
    scale_bits: int = 0
    # Exact squared L2 norm of each canonical int8 output row.  Complete
    # projection proofs authenticate the whole X.W output without sending
    # Merkle weight rows.  The runtime-output corridor still needs ||W[o]||²,
    # so qualified QKV entries carry these authority-signed statistics.
    # Empty keeps the legacy sampled-weight-row path available.
    row_sq: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.row_sq, tuple):
            raise ProofV3Error("projection row_sq must be a tuple")
        if self.row_sq:
            if len(self.row_sq) != self.out_dim:
                raise ProofV3Error(
                    "projection row_sq must cover every output row"
                )
            maximum = int(self.in_dim) * 128 * 128
            if any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                or value > maximum
                for value in self.row_sq
            ):
                raise ProofV3Error(
                    "projection row_sq contains an invalid int8 row norm"
                )


@dataclass(frozen=True, slots=True)
class ProjectionManifestV3:
    """Full per-projection manifest for one model (unsigned payload)."""

    model_id: str
    chunk_size: int
    entries: tuple[ProjectionManifestEntryV3, ...]
    # SIGNED per-model aggregate-corridor acceptance cap as IEEE-754 bits
    # (little-endian u64 of the float64), calibrated by the subnet owner on
    # honest serving traces at model registration.  0 = absent; the
    # verifier then falls back to a conservative model-agnostic default.
    corridor_chi2_bits: int = 0
    # SIGNED policy: 1 = the full-vocab lm_head binding is MANDATORY (the top
    # anchor MUST bind every committed logit to the registered lm_head; the
    # verifier fails closed if it cannot). 0 = optional (legacy reveal path).
    lm_head_binding: int = 0
    # Mode 2: the signed manifest authenticates a compact Pallas commitment
    # catalog for every LM-head output column. The validator uses it for the
    # nonce-derived full-vocabulary folds without loading model weights.
    lm_head_catalog_abi_id: str = ""
    lm_head_catalog_operation_root: bytes = b""
    lm_head_catalog_sha256: bytes = b""
    lm_head_catalog_size: int = 0
    # SIGNED quantization-stability policy for the greedy top anchor.  The
    # runtime executes the model's native LM head while the economic relation
    # authenticates an exact int8(hidden) x int8(lm_head) surrogate.  Honest
    # near-ties may change order under that reduction, so a qualified
    # model/quantization may admit the observed token among the first K
    # deterministic surrogate candidates.  K>1 is valid only when
    # lm_head_binding=1 makes EVERY surrogate logit model-bound.
    lm_head_argmax_top_k: int = 1
    # SIGNED per-cell corridor sigma safety factor as IEEE-754 bits (u64 of the
    # float64). The width of the honest quantization corridor is calibrated by
    # the owner TOGETHER with corridor_chi2_bits, so it must be signed too.
    # 0 = absent -> conservative model-agnostic default (_CORRIDOR_SIGMA).
    corridor_sigma_bits: int = 0
    # sha256 of the canonical owner-side qualification report from which both
    # corridor thresholds were derived. Production lean releases require it.
    corridor_qualification_digest: bytes = b""
    # SIGNED attention architecture + calibration for the selected-head
    # attention recompute. head counts/dim and the softmax quant/exp tables are
    # model calibration the verifier must NOT accept from the prover; the full
    # tables are large, so their sha256 DIGESTS are signed here and the runtime
    # tables are checked against them. 0/empty = attention recompute not enabled.
    attn_num_heads: int = 0
    attn_head_dim: int = 0
    attn_quant_offset: int = 0
    attn_quant_table_digest: bytes = b""
    attn_exp_table_digest: bytes = b""
    # SIGNED policy: 1 = the sampled-chunk attention audit is MANDATORY (the
    # request is rejected fail-closed unless its attention audit passed).
    attn_audit_required: int = 0
    # SIGNED attention scheme version. 1 = scored/v1 (integer-exact score
    # oracle + fixed exp table + signed fixed calibration, the ONLY scheme a
    # qualified hard profile may select). 0/absent = legacy product-domain,
    # proven UNSOUND for runtime-output binding (fb13b99) -- the adapter
    # fail-closes on it whenever the attention audit is required.
    attn_scheme: int = 0
    # sha256 of the canonical scored-calibration blob (per-layer per-head
    # fixed-point scales incl. per-dim k scales + bridge bounds); the bulk
    # calibration ships beside the manifest and is checked against this
    # SIGNED digest (same pattern as the table digests above).  Used when
    # ONE calibration covers the whole model.
    attn_calibration_digest: bytes = b""
    # sha256 of the canonical scored-calibration SET blob (context bands +
    # sampling policy, scored_calibration_set_digest_v3).  Present when the
    # model uses per-context-band calibrations: the validator selects the
    # band for the request's key count, so exactly one of
    # attn_calibration_digest / attn_calibration_set_digest is set.
    attn_calibration_set_digest: bytes = b""
    # Digest of the signed small artifact that defines QKV layout, Q/K
    # normalization, RoPE and logical paged-cache semantics used to interpret
    # raw execution-anchor rows. Mandatory for streaming attention audits.
    attn_runtime_semantics_digest: bytes = b""
    # Digest of the signed Qwen GDN recurrence/state-layout artifact.
    # Mandatory when the qualified execution profile contains GDN layers.
    gdn_runtime_semantics_digest: bytes = b""
    # Digest of the signed sparse-MoE routing, dispatch, expert, shared-expert,
    # aggregation and residual semantics. Mandatory for sparse-MoE profiles.
    moe_runtime_semantics_digest: bytes = b""
    # SIGNED decoder/final RMSNorm ABI. Most model families use the stored
    # weight directly; Gemma/Qwen3.5-style layers apply ``1 + weight``.
    # Epsilon is encoded as the exact float64 bit pattern used at runtime.
    rms_norm_semantics_id: str = RMSNORM_WEIGHT_GAIN_V3
    rms_norm_epsilon_bits: int = DEFAULT_RMSNORM_EPSILON_BITS_V3

    def __post_init__(self) -> None:
        for name, bits in (
            ("corridor_sigma_bits", self.corridor_sigma_bits),
            ("corridor_chi2_bits", self.corridor_chi2_bits),
        ):
            if (
                isinstance(bits, bool)
                or not isinstance(bits, int)
                or not 0 <= bits < 1 << 64
            ):
                raise ProofV3Error(f"{name} is malformed")
            if bits:
                value = struct.unpack("<d", struct.pack("<Q", bits))[0]
                if not math.isfinite(value) or value <= 0.0:
                    raise ProofV3Error(f"{name} is malformed")
        if (
            self.corridor_qualification_digest
            and (
                not isinstance(self.corridor_qualification_digest, bytes)
                or len(self.corridor_qualification_digest) != 32
            )
        ):
            raise ProofV3Error(
                "projection corridor qualification digest is malformed"
            )
        if self.lm_head_binding not in (0, 1, LM_HEAD_CATALOG_BINDING_V3):
            raise ProofV3Error("lm_head_binding mode is unsupported")
        if (
            type(self.lm_head_argmax_top_k) is not int
            or not 1 <= self.lm_head_argmax_top_k <= 32
        ):
            raise ProofV3Error(
                "lm_head_argmax_top_k must be an integer in [1, 32]"
            )
        if self.lm_head_argmax_top_k > 1 and not self.lm_head_binding:
            raise ProofV3Error(
                "lm_head_argmax_top_k > 1 requires lm_head_binding"
            )
        catalog_fields = (
            self.lm_head_catalog_abi_id,
            self.lm_head_catalog_operation_root,
            self.lm_head_catalog_sha256,
            self.lm_head_catalog_size,
        )
        if self.lm_head_binding == LM_HEAD_CATALOG_BINDING_V3:
            if (
                self.lm_head_catalog_abi_id != LM_HEAD_CATALOG_ABI_V3
                or not isinstance(self.lm_head_catalog_operation_root, bytes)
                or len(self.lm_head_catalog_operation_root) != 32
                or not isinstance(self.lm_head_catalog_sha256, bytes)
                or len(self.lm_head_catalog_sha256) != 32
                or isinstance(self.lm_head_catalog_size, bool)
                or not isinstance(self.lm_head_catalog_size, int)
                or not 0 < self.lm_head_catalog_size <= 33 << 20
            ):
                raise ProofV3Error(
                    "catalog-bound LM-head manifest fields are malformed"
                )
        elif catalog_fields != ("", b"", b"", 0):
            raise ProofV3Error(
                "LM-head catalog fields require catalog binding mode"
            )
        validate_rmsnorm_runtime_semantics_v3(
            self.rms_norm_semantics_id,
            self.rms_norm_epsilon_bits,
        )
        for name, digest in (
            ("attention runtime semantics", self.attn_runtime_semantics_digest),
            ("GDN runtime semantics", self.gdn_runtime_semantics_digest),
            ("MoE runtime semantics", self.moe_runtime_semantics_digest),
        ):
            if digest and (
                not isinstance(digest, bytes)
                or len(digest) != 32
                or digest == bytes(32)
            ):
                raise ProofV3Error(f"{name} digest is malformed")

    def canonical_bytes(self) -> bytes:
        """Deterministic serialization (the signed + hashed payload)."""

        obj = {
            "v": 1,
            "model_id": self.model_id,
            "chunk_size": int(self.chunk_size),
            **(
                {"corridor_chi2_bits": int(self.corridor_chi2_bits)}
                if self.corridor_chi2_bits
                else {}
            ),
            **(
                {"lm_head_binding": int(self.lm_head_binding)}
                if self.lm_head_binding
                else {}
            ),
            **(
                {
                    "lm_head_catalog_abi_id":
                        self.lm_head_catalog_abi_id,
                    "lm_head_catalog_operation_root":
                        self.lm_head_catalog_operation_root.hex(),
                    "lm_head_catalog_sha256":
                        self.lm_head_catalog_sha256.hex(),
                    "lm_head_catalog_size":
                        int(self.lm_head_catalog_size),
                }
                if self.lm_head_binding == LM_HEAD_CATALOG_BINDING_V3
                else {}
            ),
            **(
                {"lm_head_argmax_top_k": int(self.lm_head_argmax_top_k)}
                if self.lm_head_argmax_top_k != 1
                else {}
            ),
            **(
                {"corridor_sigma_bits": int(self.corridor_sigma_bits)}
                if self.corridor_sigma_bits
                else {}
            ),
            **(
                {
                    "corridor_qualification_digest":
                        self.corridor_qualification_digest.hex()
                }
                if self.corridor_qualification_digest
                else {}
            ),
            **(
                {
                    "attn_num_heads": int(self.attn_num_heads),
                    "attn_head_dim": int(self.attn_head_dim),
                    "attn_quant_offset": int(self.attn_quant_offset),
                    "attn_quant_table_digest": self.attn_quant_table_digest.hex(),
                    "attn_exp_table_digest": self.attn_exp_table_digest.hex(),
                }
                if self.attn_num_heads
                else {}
            ),
            **(
                {"attn_audit_required": int(self.attn_audit_required)}
                if self.attn_audit_required
                else {}
            ),
            **(
                {"attn_scheme": int(self.attn_scheme)}
                if self.attn_scheme
                else {}
            ),
            **(
                {"attn_calibration_digest": self.attn_calibration_digest.hex()}
                if self.attn_calibration_digest
                else {}
            ),
            **(
                {
                    "attn_calibration_set_digest":
                        self.attn_calibration_set_digest.hex()
                }
                if self.attn_calibration_set_digest
                else {}
            ),
            **(
                {
                    "attn_runtime_semantics_digest":
                        self.attn_runtime_semantics_digest.hex()
                }
                if self.attn_runtime_semantics_digest
                else {}
            ),
            **(
                {
                    "gdn_runtime_semantics_digest":
                        self.gdn_runtime_semantics_digest.hex()
                }
                if self.gdn_runtime_semantics_digest
                else {}
            ),
            **(
                {
                    "moe_runtime_semantics_digest":
                        self.moe_runtime_semantics_digest.hex()
                }
                if self.moe_runtime_semantics_digest
                else {}
            ),
            **(
                {
                    "rms_norm_semantics_id":
                        self.rms_norm_semantics_id
                }
                if self.rms_norm_semantics_id != RMSNORM_WEIGHT_GAIN_V3
                else {}
            ),
            **(
                {
                    "rms_norm_epsilon_bits":
                        int(self.rms_norm_epsilon_bits)
                }
                if (
                    self.rms_norm_epsilon_bits
                    != DEFAULT_RMSNORM_EPSILON_BITS_V3
                )
                else {}
            ),
            "entries": [
                {
                    "name": e.name,
                    "root": e.root.hex(),
                    "orientation": e.orientation,
                    "in_dim": int(e.in_dim),
                    "out_dim": int(e.out_dim),
                    **(
                        {"scale_bits": int(e.scale_bits)}
                        if e.scale_bits
                        else {}
                    ),
                    **(
                        {"row_sq": [int(value) for value in e.row_sq]}
                        if e.row_sq
                        else {}
                    ),
                }
                for e in sorted(self.entries, key=lambda e: e.name)
            ],
        }
        body = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
        return _MANIFEST_DOMAIN + body

    def digest(self) -> bytes:
        return hashlib.sha256(self.canonical_bytes()).digest()

    def root_for(self, name: str) -> bytes | None:
        for e in self.entries:
            if e.name == name:
                return e.root
        return None

    def entry_for(self, name: str) -> ProjectionManifestEntryV3 | None:
        for e in self.entries:
            if e.name == name:
                return e
        return None

    def to_json(self) -> str:
        return self.canonical_bytes()[len(_MANIFEST_DOMAIN):].decode()

    @classmethod
    def from_json(cls, text: str) -> "ProjectionManifestV3":
        obj = json.loads(text)
        entries = tuple(
            ProjectionManifestEntryV3(
                name=e["name"],
                root=bytes.fromhex(e["root"]),
                orientation=e["orientation"],
                in_dim=int(e["in_dim"]),
                out_dim=int(e["out_dim"]),
                scale_bits=int(e.get("scale_bits", 0)),
                row_sq=tuple(int(value) for value in e.get("row_sq", ())),
            )
            for e in obj["entries"]
        )
        return cls(
            model_id=obj["model_id"],
            chunk_size=int(obj["chunk_size"]),
            entries=entries,
            corridor_chi2_bits=int(obj.get("corridor_chi2_bits", 0)),
            lm_head_binding=int(obj.get("lm_head_binding", 0)),
            lm_head_catalog_abi_id=obj.get(
                "lm_head_catalog_abi_id",
                "",
            ),
            lm_head_catalog_operation_root=bytes.fromhex(
                obj.get("lm_head_catalog_operation_root", "")
            ),
            lm_head_catalog_sha256=bytes.fromhex(
                obj.get("lm_head_catalog_sha256", "")
            ),
            lm_head_catalog_size=int(
                obj.get("lm_head_catalog_size", 0)
            ),
            lm_head_argmax_top_k=int(obj.get("lm_head_argmax_top_k", 1)),
            corridor_sigma_bits=int(obj.get("corridor_sigma_bits", 0)),
            corridor_qualification_digest=bytes.fromhex(
                obj.get("corridor_qualification_digest", "")
            ),
            attn_num_heads=int(obj.get("attn_num_heads", 0)),
            attn_head_dim=int(obj.get("attn_head_dim", 0)),
            attn_quant_offset=int(obj.get("attn_quant_offset", 0)),
            attn_quant_table_digest=bytes.fromhex(
                obj.get("attn_quant_table_digest", "")
            ),
            attn_exp_table_digest=bytes.fromhex(
                obj.get("attn_exp_table_digest", "")
            ),
            attn_audit_required=int(obj.get("attn_audit_required", 0)),
            attn_scheme=int(obj.get("attn_scheme", 0)),
            attn_calibration_digest=bytes.fromhex(
                obj.get("attn_calibration_digest", "")
            ),
            attn_calibration_set_digest=bytes.fromhex(
                obj.get("attn_calibration_set_digest", "")
            ),
            attn_runtime_semantics_digest=bytes.fromhex(
                obj.get("attn_runtime_semantics_digest", "")
            ),
            gdn_runtime_semantics_digest=bytes.fromhex(
                obj.get("gdn_runtime_semantics_digest", "")
            ),
            moe_runtime_semantics_digest=bytes.fromhex(
                obj.get("moe_runtime_semantics_digest", "")
            ),
            rms_norm_semantics_id=obj.get(
                "rms_norm_semantics_id",
                RMSNORM_WEIGHT_GAIN_V3,
            ),
            rms_norm_epsilon_bits=int(
                obj.get(
                    "rms_norm_epsilon_bits",
                    DEFAULT_RMSNORM_EPSILON_BITS_V3,
                )
            ),
        )


def build_projection_manifest_v3(
    *,
    model_id: str,
    weights: dict,
    chunk_size: int = 128,
    orientation: str = "out_in",
    fast: bool = False,
) -> ProjectionManifestV3:
    """Owner-side: compute per-projection roots from real weights.

    ``weights[name]`` is the projection weight ``[out_dim][in_dim]``.  Uses
    the SAME canonical root computation as the miner-side audit material so
    honest roots match byte-for-byte.  ``orientation='out_in'`` is correct
    for o_proj/down_proj (output dim == hidden); gate_up/qkv are transposed
    in the registry and get their own strided-reveal handling later.
    """

    from verallm.miner.proof_v3_projection_audit import (
        projection_weight_root_and_row_sq_v3,
        projection_weight_root_v3,
    )
    from verallm.proof_v3.economic_wire import scale_to_bits_v3

    if not weights:
        raise ProofV3Error("manifest needs at least one projection weight")
    from verallm.proof_v3.economic_challenge import (
        FULL_ATTENTION_AUDITED_PROJECTIONS_V3,
        GDN_AUDITED_PROJECTIONS_V3,
    )

    row_norm_projections = {
        manifest_suffix
        for _x_suffix, _s_suffix, manifest_suffix in (
            FULL_ATTENTION_AUDITED_PROJECTIONS_V3
            + GDN_AUDITED_PROJECTIONS_V3
        )
    }
    entries = []
    for name in sorted(weights):
        include_row_sq = (
            ".moe." not in name.lower()
            and name.rsplit(".", 1)[-1].lower() in row_norm_projections
        )
        if fast:
            # tensor-fast root path (byte-identical to the reference,
            # locked by test) for full-model inventories
            if include_row_sq:
                root, in_dim, out_dim, absmax, row_sq = (
                    projection_weight_root_and_row_sq_v3(
                        weights[name], chunk_size
                    )
                )
            else:
                from verallm.miner.proof_v3_projection_audit import (
                    projection_weight_root_only_v3,
                )

                root, in_dim, out_dim, absmax = (
                    projection_weight_root_only_v3(
                        weights[name], chunk_size
                    )
                )
                row_sq = ()
        else:
            root, rows, _tree, in_dim, out_dim = (
                projection_weight_root_v3(weights[name], chunk_size))
            row_sq = (
                tuple(sum(value * value for value in row) for row in rows)
                if include_row_sq
                else ()
            )
            weight = weights[name]
            try:
                absmax = float(weight.abs().max())
            except AttributeError:
                absmax = max(
                    abs(float(value)) for row in weight for value in row
                )
        scale = max(absmax, 1e-8) / 127.0
        entries.append(ProjectionManifestEntryV3(
            name=name, root=root, orientation=orientation,
            in_dim=in_dim, out_dim=out_dim,
            scale_bits=scale_to_bits_v3(scale),
            row_sq=row_sq))
    return ProjectionManifestV3(
        model_id=model_id, chunk_size=chunk_size, entries=tuple(entries))


def sign_projection_manifest_v3(
    manifest: ProjectionManifestV3, *, private_key: str
) -> bytes:
    """Owner-side: 65-byte EVM ECDSA signature over the manifest digest."""

    from eth_account import Account
    from eth_account.messages import encode_defunct

    signable = encode_defunct(primitive=manifest.digest())
    signed = Account.sign_message(signable, private_key=private_key)
    sig = bytes(signed.signature)
    if len(sig) != 65:
        raise ProofV3Error("manifest signature must be 65 bytes")
    return sig


def verify_projection_manifest_v3(
    manifest: ProjectionManifestV3,
    *,
    signatures,
    expected_authorities,
    authority_threshold: int = 1,
) -> tuple[str, ...]:
    """Validator-side: verify authority signatures over the manifest.

    Recovers each signer from the manifest digest and requires at least
    ``authority_threshold`` DISTINCT signers from ``expected_authorities``
    (checksummed or lowercase 0x-addresses).  Raises on failure; returns
    the recovered authority addresses on success.
    """

    from eth_account import Account
    from eth_account.messages import encode_defunct

    expected = {a.lower() for a in expected_authorities}
    if not expected:
        raise ProofV3VerificationError("no expected manifest authorities")
    if authority_threshold < 1:
        raise ProofV3VerificationError("authority threshold must be >= 1")

    signable = encode_defunct(primitive=manifest.digest())
    recovered: set[str] = set()
    for signature in signatures:
        sig = bytes.fromhex(signature) if isinstance(signature, str) else bytes(signature)
        if len(sig) != 65:
            raise ProofV3VerificationError("manifest signature must be 65 bytes")
        try:
            signer = Account.recover_message(signable, signature=sig)
        except Exception as exc:  # noqa: BLE001
            raise ProofV3VerificationError(
                "manifest signature recovery failed") from exc
        signer_l = signer.lower()
        if signer_l in expected:
            recovered.add(signer_l)
    if len(recovered) < authority_threshold:
        raise ProofV3VerificationError(
            "manifest is not signed by the required subnet-owner authority "
            f"({len(recovered)}/{authority_threshold} authorized signers)")
    return tuple(sorted(recovered))
