"""Validator-owned signed-artifact verification for economic_recompute_v3.

The ONLY trusted weight source on the validator side is the authority-signed
per-projection manifest (:mod:`projection_manifest`, EVM ECDSA against the
on-chain owner authority).  A wire payload never supplies trusted weight
rows, embedding callbacks or LM-head callbacks: it supplies OPENINGS, and
this module authenticates every one of them against the signed manifest
roots the validator already holds.  Unknown projection name, wrong
orientation, wrong dimension, missing chunk coverage, extra chunks, or any
chunk/path/row inconsistency fails closed.
"""

from __future__ import annotations

from verallm.proof_v3.economic_wire import EconomicWeightRowRevealV3
from verallm.proof_v3.errors import ProofV3Error, ProofV3VerificationError
from verallm.proof_v3.projection_manifest import (
    ProjectionManifestV3,
    verify_projection_manifest_v3,
)

__all__ = [
    "EconomicVerifiedArtifactsV3",
    "open_manifest_weight_range_v3",
    "open_manifest_weight_row_v3",
    "required_chunk_indices_v3",
]


def required_chunk_indices_v3(
    *, row_index: int, in_dim: int, chunk_size: int
) -> tuple[int, ...]:
    """The exact flat-chunk set containing one contiguous ``out_in`` row."""

    if chunk_size < 1 or in_dim < 1 or row_index < 0:
        raise ProofV3Error("weight row chunk geometry is malformed")
    start = row_index * in_dim
    return tuple(range(start // chunk_size, (start + in_dim - 1) // chunk_size + 1))


def flat_range_sibling_digests_v3(
    merkle_tree, first_leaf: int, last_leaf: int
) -> tuple[bytes, ...]:
    """Deduplicated range-multiproof siblings for contiguous leaves.

    Climb order, at most two edge digests per level: the left sibling
    when the range's low edge is a right child, the right sibling when
    the high edge is a left child (odd-end levels self-duplicate in the
    flat tree, so their right edge needs no digest)."""

    siblings: list[bytes] = []
    low, high = int(first_leaf), int(last_leaf)
    level_size = merkle_tree.num_leaves
    level = 0
    while level_size > 1:
        offset = merkle_tree.level_offsets[level]
        if low % 2 == 1:
            siblings.append(bytes(merkle_tree._get_hash(offset + low - 1)))
        if high % 2 == 0 and high != level_size - 1:
            siblings.append(bytes(merkle_tree._get_hash(offset + high + 1)))
        low //= 2
        high //= 2
        level += 1
        level_size = (level_size + 1) // 2
    return tuple(siblings)


def open_manifest_weight_row_v3(
    *,
    tree,
    row_index: int,
    in_dim: int,
    chunk_size: int,
    weight_tensor=None,
) -> EconomicWeightRowRevealV3:
    """Miner-side: build one wire weight-row reveal from a FlatWeightMerkle."""

    required = required_chunk_indices_v3(
        row_index=row_index, in_dim=in_dim, chunk_size=chunk_size
    )
    blob = bytearray()
    for chunk_index in required:
        _path, chunk_data = tree.get_proof(
            chunk_index, W_tensor=weight_tensor
        )
        blob.extend(chunk_data)
    return EconomicWeightRowRevealV3(
        row_index=row_index,
        chunk_blob=bytes(blob),
        range_siblings=flat_range_sibling_digests_v3(
            tree._tree, required[0], required[-1]
        ),
    )


def open_manifest_weight_range_v3(
    *,
    tree,
    first_row: int,
    row_count: int,
    in_dim: int,
    out_dim: int,
    chunk_size: int,
    weight_tensor=None,
):
    """Miner-side contiguous multi-row opening with one range multiproof."""

    from verallm.proof_v3.economic_moe_wire import (
        EconomicMoeWeightRangeRevealV3,
    )

    if (
        isinstance(first_row, bool)
        or not isinstance(first_row, int)
        or isinstance(row_count, bool)
        or not isinstance(row_count, int)
        or first_row < 0
        or row_count <= 0
        or first_row + row_count > out_dim
        or in_dim <= 0
        or out_dim <= 0
        or chunk_size <= 0
    ):
        raise ProofV3Error("weight range geometry is malformed")
    first_byte = first_row * in_dim
    last_byte = (first_row + row_count) * in_dim - 1
    required = tuple(
        range(first_byte // chunk_size, last_byte // chunk_size + 1)
    )
    blob = bytearray()
    for chunk_index in required:
        _path, chunk_data = tree.get_proof(
            chunk_index,
            W_tensor=weight_tensor,
        )
        blob.extend(chunk_data)
    return EconomicMoeWeightRangeRevealV3(
        first_row=first_row,
        row_count=row_count,
        chunk_blob=bytes(blob),
        range_siblings=flat_range_sibling_digests_v3(
            tree._tree,
            required[0],
            required[-1],
        ),
    )


def _require_projection_catalog_quantization_v3(
    *,
    static_manifest,
    projection_manifest: ProjectionManifestV3,
) -> None:
    """Require the Pallas catalog to use the runtime's exact scalar int8 ABI."""

    from verallm.proof_v3.economic_wire import bits_to_scale_v3
    from verallm.proof_v3.lean_projection_fold import (
        lean_projection_operation_key_v3,
    )
    from verallm.proof_v3.moe_projection_inventory import (
        parse_moe_projection_name_v3,
    )

    descriptor_by_key = {
        (
            descriptor.layer,
            descriptor.operation_id,
            -1 if descriptor.expert_id is None else descriptor.expert_id,
        ): descriptor
        for descriptor in static_manifest.operations
    }
    for entry in projection_manifest.entries:
        if parse_moe_projection_name_v3(entry.name) is not None:
            # Sparse-MoE weights use their individually signed Merkle roots
            # and post-commit range openings.  They deliberately do not
            # occupy the dense full-output Pallas catalog.
            continue
        if not entry.name.startswith("l") or "." not in entry.name:
            continue
        layer_text, projection = entry.name.split(".", 1)
        if (
            not layer_text[1:].isdigit()
            or projection.endswith(("_bias", "_norm"))
            or projection in {"input_norm", "post_norm"}
        ):
            continue
        try:
            layer_index = int(layer_text[1:])
            key = lean_projection_operation_key_v3(
                layer_index=layer_index,
                projection=projection,
            )
            descriptor = descriptor_by_key[
                (layer_index, key.operation_id, key.expert_idx)
            ]
        except (ProofV3Error, KeyError) as exc:
            raise ProofV3VerificationError(
                f"optimized hard projection manifest does not cover "
                f"{entry.name!r}"
            ) from exc
        expected_scale_q32 = int(
            round(bits_to_scale_v3(entry.scale_bits) * (1 << 32))
        )
        if (
            descriptor.weight_block_scales_q32
            or descriptor.weight_scale_q32 != expected_scale_q32
        ):
            raise ProofV3VerificationError(
                f"optimized hard projection quantization disagrees for "
                f"{entry.name!r}"
            )


class EconomicVerifiedArtifactsV3:
    """Validator-owned view over one authority-signed projection manifest."""

    def __init__(self, manifest: ProjectionManifestV3, *, _verified: bool) -> None:
        if not _verified:
            raise ProofV3Error(
                "EconomicVerifiedArtifactsV3 must be built through from_signed "
                "or from_preverified"
            )
        if not isinstance(manifest, ProjectionManifestV3):
            raise ProofV3Error("manifest has an unexpected type")
        self._manifest = manifest
        # Validator-owned on-chain tokenizer anchor. Production execution
        # profiles bind this exact digest; it is never accepted from proof
        # bytes or inferred from the manifest's model-id string.
        self.tokenizer_binding_digest = None
        # Optional validator-owned authenticated int8 lm_head tensor. When set
        # (via authenticate_lm_head_v3), the top anchor binds EVERY committed
        # logit to int8(lm_head) @ int8(hidden), closing the suppression gap.
        # None -> reveal path unchanged (sampled-cell + winner binding only).
        self.lm_head_int8 = None
        # Optional authority-authenticated, weightless Pallas LM-head catalog.
        # Catalog binding mode requires this exact object and never falls back
        # to validator-resident model weights.
        self.lm_head_catalog_binding = None
        # Authority-authenticated proof-v2 Pallas catalog used by lean
        # complete-output projection folds. It is loaded by the validator
        # from the same ModelSpec-bound catalog bridge, never from proof bytes.
        self.lean_projection_catalog = None
        # Canonical attention audit, verified INLINE by the adapter:
        # attention_section = the post-nonce reveal (EconomicAttentionSectionV3),
        # attention_roots = the pre-nonce committed capture roots
        # (EconomicAttentionLayerRootsV3). The adapter authenticates the reveal
        # against the roots itself (no external verdict). Set by the validator
        # from the proof before verify_economic_recompute_v3.
        self.attention_section = None
        self.attention_roots = None
        # Pre-nonce attention-root commitment (#5) recovered by the validator
        # from the authenticated capture_chain_digest / execution_root; the
        # adapter requires the section roots to hash to it.
        self.attention_root_commitment = None
        # Digest-verified scored calibration blob (ScoredCalibrationV3,
        # loaded by the validator via load_scored_calibration_v3). When the
        # manifest pins attn_calibration_digest, the adapter fails closed
        # unless this is present and its digest matches.
        self.attn_calibration = None
        # Multi-band variant (ScoredCalibrationSetV3, loaded by the
        # validator via load_scored_calibration_set_v3). When the manifest
        # pins attn_calibration_set_digest, the adapter fails closed unless
        # this is present, its canonical digest matches the SIGNED set
        # digest, and a band covers the section's authenticated key count.
        self.attn_calibration_set = None
        # Digest-authenticated runtime adapter for interpreting raw QKV
        # execution anchors. It is small model qualification data, not model
        # weights, and is mandatory for streaming attention verification.
        self.attention_runtime_semantics = None
        # Digest-authenticated Qwen GDN recurrence/state-layout artifact.
        self.gdn_runtime_semantics = None
        # Digest-authenticated sparse-MoE logical execution artifact.
        self.moe_runtime_semantics = None
        # Miner-supplied opening of the audited layer's o_proj INPUT oracle
        # (l{layer}.attn_o_x) at the attention plan's sampled token
        # positions -- the runtime side of the output bridge.
        self.attention_bridge_opening = None
        # SCORED_SCHEME_RATIONAL_V2 attention transport rides the
        # PROOF's canonical attention request section
        # (EconomicAttentionRequestSectionV3), never this validator-
        # owned container -- the adapter reconstructs the pre-nonce
        # fold from the section against the authenticated capture
        # chain itself. Only the digest-verified calibration SET
        # (validator-loaded, above) lives here.

    @classmethod
    def from_signed(
        cls,
        manifest: ProjectionManifestV3,
        *,
        signatures,
        expected_authorities,
        authority_threshold: int = 1,
    ) -> "EconomicVerifiedArtifactsV3":
        """Verify the authority signatures, then wrap the manifest."""

        verify_projection_manifest_v3(
            manifest,
            signatures=signatures,
            expected_authorities=expected_authorities,
            authority_threshold=authority_threshold,
        )
        return cls(manifest, _verified=True)

    @classmethod
    def from_preverified(
        cls, manifest: ProjectionManifestV3
    ) -> "EconomicVerifiedArtifactsV3":
        """Wrap a manifest whose signature the caller has already verified."""

        return cls(manifest, _verified=True)

    @property
    def manifest(self) -> ProjectionManifestV3:
        return self._manifest

    def authenticate_attention_runtime_semantics_v3(self, semantics) -> None:
        from verallm.proof_v3.attention_runtime_semantics import (
            AttentionRuntimeSemanticsV3,
        )

        signed = self._manifest.attn_runtime_semantics_digest
        if not signed:
            raise ProofV3VerificationError(
                "signed manifest pins no attention runtime semantics"
            )
        if (
            not isinstance(semantics, AttentionRuntimeSemanticsV3)
            or semantics.digest() != signed
        ):
            raise ProofV3VerificationError(
                "attention runtime semantics do not match the signed manifest"
            )
        self.attention_runtime_semantics = semantics

    def authenticate_gdn_runtime_semantics_v3(self, semantics) -> None:
        from verallm.proof_v3.gdn_runtime_semantics import (
            GdnRuntimeSemanticsV3,
        )

        signed = self._manifest.gdn_runtime_semantics_digest
        if not signed:
            raise ProofV3VerificationError(
                "signed manifest pins no GDN runtime semantics"
            )
        if (
            not isinstance(semantics, GdnRuntimeSemanticsV3)
            or semantics.digest() != signed
        ):
            raise ProofV3VerificationError(
                "GDN runtime semantics do not match the signed manifest"
            )
        self.gdn_runtime_semantics = semantics

    def authenticate_moe_runtime_semantics_v3(self, semantics) -> None:
        from verallm.proof_v3.moe_runtime_semantics import (
            MoeRuntimeSemanticsV3,
        )

        signed = self._manifest.moe_runtime_semantics_digest
        if not signed:
            raise ProofV3VerificationError(
                "signed manifest pins no MoE runtime semantics"
            )
        if (
            not isinstance(semantics, MoeRuntimeSemanticsV3)
            or semantics.digest() != signed
        ):
            raise ProofV3VerificationError(
                "MoE runtime semantics do not match the signed manifest"
            )
        self.moe_runtime_semantics = semantics

    def authenticate_tokenizer_binding_v3(self, tokenizer_digest: bytes) -> None:
        """Attach the validator-verified on-chain tokenizer digest."""

        if (
            not isinstance(tokenizer_digest, bytes)
            or len(tokenizer_digest) != 32
            or tokenizer_digest == bytes(32)
        ):
            raise ProofV3VerificationError(
                "tokenizer binding digest is malformed"
            )
        self.tokenizer_binding_digest = tokenizer_digest

    def authenticate_lm_head_catalog_v3(self, catalog) -> None:
        """Attach the signed compact LM-head commitment catalog."""

        import hashlib

        from verallm.proof_v3.economic_lm_head_catalog_fold import (
            EconomicLmHeadCatalogArtifactV3,
        )
        from verallm.proof_v3.projection_manifest import (
            LM_HEAD_CATALOG_ABI_V3,
            LM_HEAD_CATALOG_BINDING_V3,
        )

        manifest = self._manifest
        if manifest.lm_head_binding != LM_HEAD_CATALOG_BINDING_V3:
            raise ProofV3VerificationError(
                "signed manifest does not select catalog LM-head binding"
            )
        if not isinstance(catalog, EconomicLmHeadCatalogArtifactV3):
            raise ProofV3VerificationError(
                "LM-head catalog has an unexpected type"
            )
        encoded = catalog.canonical_bytes()
        entry = self.entry("lm_head")
        if (
            manifest.lm_head_catalog_abi_id != LM_HEAD_CATALOG_ABI_V3
            or len(encoded) != manifest.lm_head_catalog_size
            or hashlib.sha256(encoded).digest()
            != manifest.lm_head_catalog_sha256
            or catalog.operation_root
            != manifest.lm_head_catalog_operation_root
            or catalog.hidden_dim != entry.in_dim
            or catalog.vocab != entry.out_dim
        ):
            raise ProofV3VerificationError(
                "LM-head catalog does not match the signed manifest"
            )
        self.lm_head_catalog_binding = catalog.binding

    def authenticate_lean_projection_catalog_v3(
        self,
        verified_v2_catalog_binding,
    ) -> None:
        """Attach the ModelSpec-authenticated complete projection catalog."""

        from verallm.proof_v3.catalog import VerifiedV2CatalogBindingV3
        from verallm.proof_v3.lean_projection_fold import (
            LeanProjectionCatalogV3,
            lean_projection_operation_key_v3,
        )
        from verallm.proof_v3.moe_projection_inventory import (
            parse_moe_projection_name_v3,
        )

        if not isinstance(
            verified_v2_catalog_binding,
            VerifiedV2CatalogBindingV3,
        ):
            raise ProofV3VerificationError(
                "lean projection catalog requires a verified proof-v2 binding"
            )
        verified_v2_catalog_binding.require_verified_v2_provenance()
        static = verified_v2_catalog_binding.static_manifest
        if static.model_id != self._manifest.model_id:
            raise ProofV3VerificationError(
                "lean projection catalog model does not match the signed manifest"
            )
        _require_projection_catalog_quantization_v3(
            static_manifest=static,
            projection_manifest=self._manifest,
        )
        catalog = LeanProjectionCatalogV3.from_verified_v2_catalog_binding(
            verified_v2_catalog_binding=verified_v2_catalog_binding,
        )
        for entry in self._manifest.entries:
            if parse_moe_projection_name_v3(entry.name) is not None:
                continue
            if not entry.name.startswith("l") or "." not in entry.name:
                continue
            layer_text, projection = entry.name.split(".", 1)
            if not layer_text[1:].isdigit() or projection.endswith(
                ("_bias", "_norm")
            ):
                continue
            if projection in {"input_norm", "post_norm"}:
                continue
            try:
                operation = catalog.operation(
                    lean_projection_operation_key_v3(
                    layer_index=int(layer_text[1:]),
                    projection=projection,
                )
                )
            except ProofV3Error as exc:
                raise ProofV3VerificationError(
                    f"lean projection catalog does not cover {entry.name!r}"
                ) from exc
            if (
                operation.input_dim != entry.in_dim
                or operation.output_dim != entry.out_dim
            ):
                raise ProofV3VerificationError(
                    f"lean projection catalog geometry disagrees for "
                    f"{entry.name!r}"
                )
            if len(entry.row_sq) != entry.out_dim:
                raise ProofV3VerificationError(
                    f"optimized hard projection entry {entry.name!r} has "
                    "no complete signed row-norm inventory"
                )
        self.lean_projection_catalog = catalog

    def entry(self, name: str):
        entry = self._manifest.entry_for(name)
        if entry is None:
            raise ProofV3VerificationError(
                f"projection {name!r} is not in the signed manifest"
            )
        return entry

    def has_entry(self, name: str) -> bool:
        return self._manifest.entry_for(name) is not None

    def dims(self, name: str) -> tuple[int, int]:
        entry = self.entry(name)
        return entry.in_dim, entry.out_dim

    def scale_for(self, name: str) -> float:
        """The SIGNED int8 dequant scale of one projection (corridors)."""

        from verallm.proof_v3.economic_wire import bits_to_scale_v3

        entry = self.entry(name)
        if not entry.scale_bits:
            raise ProofV3VerificationError(
                f"projection {name!r} has no signed quantization scale"
            )
        return bits_to_scale_v3(entry.scale_bits)

    def weight_row_sq(self, name: str, row_index: int) -> int:
        """Return one authority-signed canonical-int8 weight-row norm."""

        entry = self.entry(name)
        if (
            isinstance(row_index, bool)
            or not isinstance(row_index, int)
            or row_index < 0
            or row_index >= entry.out_dim
            or len(entry.row_sq) != entry.out_dim
        ):
            raise ProofV3VerificationError(
                f"projection {name!r} has no signed norm for row {row_index!r}"
            )
        return entry.row_sq[row_index]

    def rms_norm_parameters(self) -> tuple[float, float]:
        """Return the authority-signed decoder/final RMSNorm ABI."""

        from verallm.proof_v3.rmsnorm_runtime_semantics import (
            decode_rmsnorm_runtime_semantics_v3,
        )

        return decode_rmsnorm_runtime_semantics_v3(
            self._manifest.rms_norm_semantics_id,
            self._manifest.rms_norm_epsilon_bits,
        )

    def verify_weight_row(
        self, *, name: str, reveal: EconomicWeightRowRevealV3
    ) -> tuple[int, ...]:
        """Authenticate one revealed int8 row against the signed manifest root.

        Returns the signed int8 row values on success.  Everything about the
        expectation (root, orientation, dims, chunk geometry, exact chunk
        set) is validator-owned; the reveal only carries openings.
        """

        from zkllm.crypto.merkle import hash_flat_chunk, hash_leaf, hash_node

        if not isinstance(reveal, EconomicWeightRowRevealV3):
            raise ProofV3VerificationError(
                "weight row reveal has an unexpected type"
            )
        entry = self.entry(name)
        if entry.orientation != "out_in":
            raise ProofV3VerificationError(
                f"projection {name!r} orientation {entry.orientation!r} is not "
                "supported by the economic audit"
            )
        if reveal.row_index >= entry.out_dim:
            raise ProofV3VerificationError(
                f"projection {name!r} row index exceeds the manifest out_dim"
            )
        chunk_size = self._manifest.chunk_size
        required = required_chunk_indices_v3(
            row_index=reveal.row_index,
            in_dim=entry.in_dim,
            chunk_size=chunk_size,
        )
        # geometry is validator-owned: total leaves and every per-chunk
        # byte length derive from the SIGNED dims (int8: 1 byte/element;
        # the flat tree's final chunk may be short)
        total_bytes = entry.in_dim * entry.out_dim
        total_chunks = -(-total_bytes // chunk_size)
        lengths = []
        expected_blob_len = 0
        for chunk_index in required:
            start = chunk_index * chunk_size
            length = min(chunk_size, total_bytes - start)
            if length <= 0:
                raise ProofV3VerificationError(
                    f"projection {name!r} required chunk exceeds the tree"
                )
            lengths.append(length)
            expected_blob_len += length
        if len(reveal.chunk_blob) != expected_blob_len:
            raise ProofV3VerificationError(
                f"projection {name!r} chunk blob does not match the "
                "required range"
            )
        nodes: list[bytes] = []
        cursor = 0

        for chunk_index, length in zip(required, lengths):
            chunk_bytes = reveal.chunk_blob[cursor:cursor + length]
            cursor += length
            nodes.append(hash_leaf(hash_flat_chunk(chunk_bytes)))
        low, high = required[0], required[-1]
        level_size = total_chunks
        sibling_iter = iter(reveal.range_siblings)
        try:
            while level_size > 1:
                if low % 2 == 1:
                    nodes.insert(0, next(sibling_iter))
                    low -= 1
                if high % 2 == 0:
                    if high == level_size - 1:
                        # odd-sized level: the flat tree pairs its last
                        # node with itself
                        nodes.append(nodes[-1])
                    else:
                        nodes.append(next(sibling_iter))
                    high += 1
                nodes = [
                    hash_node(nodes[i], nodes[i + 1])
                    for i in range(0, len(nodes), 2)
                ]
                low //= 2
                high //= 2
                level_size = (level_size + 1) // 2
        except StopIteration:
            raise ProofV3VerificationError(
                f"projection {name!r} range proof is missing siblings"
            )
        if (
            next(sibling_iter, None) is not None
            or len(nodes) != 1
            or nodes[0] != entry.root
        ):
            raise ProofV3VerificationError(
                f"projection {name!r} row range did not reconstruct the "
                "signed manifest root"
            )
        import numpy as np

        start = reveal.row_index * entry.in_dim
        blob_start = required[0] * chunk_size
        row_offset = start - blob_start
        if (
            row_offset < 0
            or row_offset + entry.in_dim > len(reveal.chunk_blob)
        ):
            raise ProofV3VerificationError(
                f"projection {name!r} authenticated chunks omit the row"
            )
        row = np.frombuffer(
            reveal.chunk_blob,
            dtype=np.int8,
            count=entry.in_dim,
            offset=row_offset,
        )
        if int(row.size) != entry.in_dim:
            raise ProofV3VerificationError(
                f"projection {name!r} authenticated chunks omit the row"
            )
        return tuple(row.tolist())

    def verify_weight_range(self, *, name: str, reveal) -> bytes:
        """Authenticate contiguous complete rows against one manifest root."""

        from verallm.proof_v3.economic_moe_wire import (
            EconomicMoeWeightRangeRevealV3,
        )
        from zkllm.crypto.merkle import hash_flat_chunk, hash_leaf, hash_node

        if not isinstance(reveal, EconomicMoeWeightRangeRevealV3):
            raise ProofV3VerificationError(
                "weight range reveal has an unexpected type"
            )
        entry = self.entry(name)
        if entry.orientation != "out_in":
            raise ProofV3VerificationError(
                f"projection {name!r} orientation {entry.orientation!r} is not "
                "supported by the economic audit"
            )
        if reveal.first_row + reveal.row_count > entry.out_dim:
            raise ProofV3VerificationError(
                f"projection {name!r} row range exceeds the manifest out_dim"
            )
        chunk_size = self._manifest.chunk_size
        first_byte = reveal.first_row * entry.in_dim
        byte_count = reveal.row_count * entry.in_dim
        last_byte = first_byte + byte_count - 1
        required = tuple(
            range(first_byte // chunk_size, last_byte // chunk_size + 1)
        )
        total_bytes = entry.in_dim * entry.out_dim
        total_chunks = -(-total_bytes // chunk_size)
        lengths = []
        expected_blob_len = 0
        for chunk_index in required:
            start = chunk_index * chunk_size
            length = min(chunk_size, total_bytes - start)
            if length <= 0:
                raise ProofV3VerificationError(
                    f"projection {name!r} required chunk exceeds the tree"
                )
            lengths.append(length)
            expected_blob_len += length
        if len(reveal.chunk_blob) != expected_blob_len:
            raise ProofV3VerificationError(
                f"projection {name!r} chunk blob does not match the required range"
            )
        nodes = []
        cursor = 0
        for length in lengths:
            chunk_bytes = reveal.chunk_blob[cursor : cursor + length]
            cursor += length
            nodes.append(hash_leaf(hash_flat_chunk(chunk_bytes)))
        low, high = required[0], required[-1]
        level_size = total_chunks
        sibling_iter = iter(reveal.range_siblings)
        try:
            while level_size > 1:
                if low % 2 == 1:
                    nodes.insert(0, next(sibling_iter))
                    low -= 1
                if high % 2 == 0:
                    if high == level_size - 1:
                        nodes.append(nodes[-1])
                    else:
                        nodes.append(next(sibling_iter))
                    high += 1
                nodes = [
                    hash_node(nodes[index], nodes[index + 1])
                    for index in range(0, len(nodes), 2)
                ]
                low //= 2
                high //= 2
                level_size = (level_size + 1) // 2
        except StopIteration as exc:
            raise ProofV3VerificationError(
                f"projection {name!r} range proof is missing siblings"
            ) from exc
        if (
            next(sibling_iter, None) is not None
            or len(nodes) != 1
            or nodes[0] != entry.root
        ):
            raise ProofV3VerificationError(
                f"projection {name!r} row range did not reconstruct the "
                "signed manifest root"
            )
        blob_start = required[0] * chunk_size
        offset = first_byte - blob_start
        if offset < 0 or offset + byte_count > len(reveal.chunk_blob):
            raise ProofV3VerificationError(
                f"projection {name!r} authenticated chunks omit the row range"
            )
        return reveal.chunk_blob[offset : offset + byte_count]
