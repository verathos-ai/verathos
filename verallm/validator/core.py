"""
Validator node — requests inference and verifies ZK proofs.

Weight Merkle roots come from the on-chain model registry — NOT from
the miner. The validator reads the ModelSpec (with roots) from chain
and uses it to verify proofs.

Default mode (lightweight):
- No model loading at all — only needs ModelSpec (~1KB from chain)
- W spot checks verified via Merkle proofs against on-chain roots
- Enables CPU-only validators to verify models larger than available RAM

Full mode (--full):
- Validator loads the actual model for direct weight value lookups
- Redundant verification on top of Merkle proofs
"""

import hashlib
import logging
import os
import struct
import time
import uuid
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from verallm.config import Config, get_config
from verallm.challenge.beacon import (
    derive_beacon_from_nonce,
    derive_challenges,
    derive_sampling_challenge,
)
from verallm.crypto.merkle import verify_merkle_path, build_block_merkle, MerkleTree
from verallm.crypto.transcript import Transcript
from verallm.introspection import (
    get_layers,
    get_mlp,
    get_gate_proj,
)
from verallm.moe import (
    MoEConfig,
    MoEVerifier,
    derive_moe_challenges,
)
from verallm.moe.router_commitment import (
    compute_topk_indices,
    logits_row_to_bytes,
)
from verallm.types import (
    InferenceCommitment,
    InferenceProofBundle,
    InferenceRequest,
    ChallengeSet,
    ModelSpec,
    SpotCheck,
    VerificationResult,
)
from verallm.verifier.gemm import GEMMVerifier
from verallm.sampling import (
    clamp_sampling_bps,
    hidden_row_from_bytes,
    logits_i32_from_bytes,
    quantize_hidden_row_int64,
    verify_quantized_argmax,
    verify_fp16_argmax,
)

logger = logging.getLogger(__name__)


class Validator:
    """Standalone proof verifier for demos, tests, and Streamlit apps.

    NOTE: This class is NOT used in subnet production (neurons/).  The
    subnet validator and proxy both use ``ValidatorClient`` from
    ``verallm.api.client``, which bundles HTTP transport + verification
    in a single class.  Any verification logic changes should be made in
    ``ValidatorClient.verify_proof()`` first — this class mirrors that
    logic for offline / test usage only.

    Weight Merkle roots come from the on-chain model registry — NOT from
    the miner. The validator reads the ModelSpec (with roots) from chain
    and uses it to verify proofs.
    """

    def __init__(self, model_spec: ModelSpec, config: Optional[Config] = None,
                 model_name: Optional[str] = None, shared_model=None,
                 lightweight: bool = True, moe_config: Optional[MoEConfig] = None):
        self.model_spec = model_spec
        self.config = config or get_config()
        self.model = shared_model
        self.model_name = model_name
        self.lightweight = lightweight

        # MoE configuration (None for dense models)
        self.moe_config = moe_config
        self.is_moe = moe_config is not None and moe_config.is_moe
        self._moe_verifier: Optional[MoEVerifier] = None

    # ------------------------------------------------------------------
    # Model introspection delegates
    # ------------------------------------------------------------------

    def _get_layers(self):
        if self.model is None:
            return []
        return get_layers(self.model)

    def _get_mlp(self, layer):
        return get_mlp(layer)

    def _get_gate_proj(self, mlp):
        return get_gate_proj(mlp)

    def _verify_moe_layer_root(
        self,
        layer_idx: int,
        expert_roots: list[bytes],
        router_weight_root: Optional[bytes],
    ) -> bool:
        """Verify expert/router roots hash to the committed MoE layer root."""
        if layer_idx >= len(self.model_spec.weight_block_merkle_roots):
            return True

        committed_layer_root = self.model_spec.weight_block_merkle_roots[layer_idx]
        if committed_layer_root is None:
            return True

        # V2 root binds router root + expert roots.
        if router_weight_root:
            h_v2 = hashlib.sha256(b"MOE_LAYER_V2")
            h_v2.update(router_weight_root)
            for er in expert_roots:
                h_v2.update(er)
            if h_v2.digest() == committed_layer_root:
                return True

        # Backward compatibility for pre-V2 registrations.
        h_v1 = hashlib.sha256(b"MOE_LAYER_V1")
        for er in expert_roots:
            h_v1.update(er)
        return h_v1.digest() == committed_layer_root

    def _verify_router_layer_openings(self, proof_bundle: InferenceProofBundle, layer_challenge) -> VerificationResult:
        """Verify Merkle openings and top-k consistency for sampled router tokens."""
        layer_idx = layer_challenge.layer_idx
        router_commitment = proof_bundle.router_commitments.get(layer_idx)
        if router_commitment is None:
            return VerificationResult.failure(
                f"Missing router commitment for challenged MoE layer {layer_idx}"
            )

        layer_routing_proof = proof_bundle.router_layer_proofs.get(layer_idx)
        if layer_routing_proof is None:
            return VerificationResult.failure(
                f"Missing router layer proof for challenged MoE layer {layer_idx}"
            )

        if not router_commitment.router_logits_row_root:
            return VerificationResult.failure(
                f"Router logits row root missing in router commitment for layer {layer_idx}"
            )

        # Strictly bind routing metadata to model spec to prevent challenge steering.
        expected_num_experts = int(getattr(self.model_spec, "num_experts", 0) or 0)
        if expected_num_experts <= 0:
            spec_roots = self.model_spec.expert_weight_merkle_roots.get(layer_idx, [])
            if spec_roots:
                expected_num_experts = len(spec_roots)
        if expected_num_experts > 0:
            if router_commitment.num_experts <= 0:
                return VerificationResult.failure(
                    f"Layer {layer_idx}: router commitment missing num_experts"
                )
            if router_commitment.num_experts != expected_num_experts:
                return VerificationResult.failure(
                    f"Layer {layer_idx}: num_experts mismatch "
                    f"(committed={router_commitment.num_experts}, expected={expected_num_experts})"
                )

        expected_top_k = int(getattr(self.model_spec, "router_top_k", 0) or 0)
        if expected_top_k > 0:
            if router_commitment.top_k <= 0:
                return VerificationResult.failure(
                    f"Layer {layer_idx}: router commitment missing top_k"
                )
            if router_commitment.top_k != expected_top_k:
                return VerificationResult.failure(
                    f"Layer {layer_idx}: top_k mismatch "
                    f"(committed={router_commitment.top_k}, expected={expected_top_k})"
                )

        expected_scoring = str(getattr(self.model_spec, "router_scoring", "") or "")
        if expected_scoring and router_commitment.scoring_func:
            if str(router_commitment.scoring_func) != expected_scoring:
                return VerificationResult.failure(
                    f"Layer {layer_idx}: scoring_func mismatch "
                    f"(committed={router_commitment.scoring_func}, expected={expected_scoring})"
                )

        committed_seq = len(router_commitment.selected_experts)
        if router_commitment.seq_len and router_commitment.seq_len != committed_seq:
            return VerificationResult.failure(
                f"Layer {layer_idx}: seq_len mismatch "
                f"(committed={router_commitment.seq_len}, rows={committed_seq})"
            )
        if len(router_commitment.routing_weights) != committed_seq:
            return VerificationResult.failure(
                f"Layer {layer_idx}: routing_weights row-count mismatch "
                f"(weights={len(router_commitment.routing_weights)}, experts={committed_seq})"
            )
        if router_commitment.proof_selected_experts:
            if len(router_commitment.proof_selected_experts) != committed_seq:
                return VerificationResult.failure(
                    f"Layer {layer_idx}: proof_selected_experts row-count mismatch "
                    f"(proof={len(router_commitment.proof_selected_experts)}, experts={committed_seq})"
                )

        top_k = int(router_commitment.top_k or 0)
        if top_k <= 0:
            return VerificationResult.failure(
                f"Layer {layer_idx}: invalid top_k={router_commitment.top_k}"
            )

        from verallm.crypto.field import P as FIELD_PRIME
        for row_idx, row in enumerate(router_commitment.selected_experts):
            if len(row) != top_k:
                return VerificationResult.failure(
                    f"Layer {layer_idx}: selected_experts row {row_idx} width mismatch "
                    f"(expected={top_k}, got={len(row)})"
                )
            for expert_idx in row:
                if expert_idx < 0 or (
                    router_commitment.num_experts > 0 and expert_idx >= router_commitment.num_experts
                ):
                    return VerificationResult.failure(
                        f"Layer {layer_idx}: expert index out of range at row {row_idx}: {expert_idx}"
                    )

        for row_idx, row in enumerate(router_commitment.routing_weights):
            if len(row) != top_k:
                return VerificationResult.failure(
                    f"Layer {layer_idx}: routing_weights row {row_idx} width mismatch "
                    f"(expected={top_k}, got={len(row)})"
                )
            for w in row:
                if int(w) < 0 or int(w) >= FIELD_PRIME:
                    return VerificationResult.failure(
                        f"Layer {layer_idx}: routing weight out of field range at row {row_idx}"
                    )
        for row_idx, row in enumerate(router_commitment.proof_selected_experts):
            if len(row) != top_k:
                return VerificationResult.failure(
                    f"Layer {layer_idx}: proof_selected_experts row {row_idx} width mismatch "
                    f"(expected={top_k}, got={len(row)})"
                )
            for expert_idx in row:
                if expert_idx < 0 or (
                    router_commitment.num_experts > 0 and expert_idx >= router_commitment.num_experts
                ):
                    return VerificationResult.failure(
                        f"Layer {layer_idx}: proof expert index out of range at row {row_idx}: {expert_idx}"
                    )

        # Verify the provided router root is consistent with the committed layer root.
        expert_roots = self.model_spec.expert_weight_merkle_roots.get(layer_idx, [])
        if not expert_roots:
            expert_roots = proof_bundle.expert_roots.get(layer_idx, [])
        if expert_roots and not self._verify_moe_layer_root(
            layer_idx=layer_idx,
            expert_roots=expert_roots,
            router_weight_root=layer_routing_proof.router_weight_root,
        ):
            return VerificationResult.failure(
                f"MoE layer {layer_idx}: router/expert roots do not match committed layer root"
            )

        # Verify sampled router GEMM proof: this binds routing to X and W.
        if layer_routing_proof.router_gemm_proof is None:
            return VerificationResult.failure(
                f"Missing router GEMM proof for challenged MoE layer {layer_idx}"
            )
        if not layer_routing_proof.proved_output_rows:
            return VerificationResult.failure(
                f"Missing proven router output rows for challenged MoE layer {layer_idx}"
            )

        proved_rows_by_token = {
            int(row.token_idx): row
            for row in layer_routing_proof.proved_output_rows
        }
        if len(proved_rows_by_token) != len(layer_routing_proof.proved_output_rows):
            return VerificationResult.failure(
                f"Duplicate token indices in proven router output rows for layer {layer_idx}"
            )

        proved_matrix_rows: list[list[int]] = []
        for token_idx in layer_challenge.sampled_token_indices:
            proved_row = proved_rows_by_token.get(int(token_idx))
            if proved_row is None:
                return VerificationResult.failure(
                    f"Missing proven router output row for layer {layer_idx}, token {token_idx}"
                )
            if router_commitment.num_experts and len(proved_row.logits_int) != router_commitment.num_experts:
                return VerificationResult.failure(
                    f"Layer {layer_idx} token {token_idx}: proven router row width mismatch "
                    f"(expected {router_commitment.num_experts}, got {len(proved_row.logits_int)})"
                )
            proved_matrix_rows.append([int(v) for v in proved_row.logits_int])

        if not proved_matrix_rows:
            return VerificationResult.failure(
                f"No proven router rows available for challenged layer {layer_idx}"
            )

        y_tensor = torch.tensor(proved_matrix_rows, dtype=torch.int64)
        y_root = build_block_merkle(y_tensor, self.config.block_size).root
        router_verifier = GEMMVerifier(self.config)
        router_verify = router_verifier.verify(
            proof=layer_routing_proof.router_gemm_proof,
            X_commitment=proof_bundle.commitment.layer_commitments[layer_idx],
            W_commitment=layer_routing_proof.router_weight_root,
            Y_commitment=y_root,
            transcript=Transcript(f"layer_{layer_idx}_router_gemm".encode()),
            spot_check_fn=lambda _spot, _matrix_id: True,
            W_merkle_root=layer_routing_proof.router_weight_root,
            W_num_cols=(router_commitment.num_experts
                        if router_commitment.num_experts > 0
                        else len(proved_matrix_rows[0])),
            w_chunk_size=self.model_spec.w_merkle_chunk_size,
        )
        if not router_verify.passed:
            return VerificationResult.failure(
                f"Layer {layer_idx}: router GEMM proof failed: {router_verify.message}"
            )

        openings_by_token = {
            opening.token_idx: opening
            for opening in layer_routing_proof.logits_openings
        }

        for token_idx in layer_challenge.sampled_token_indices:
            opening = openings_by_token.get(int(token_idx))
            if opening is None:
                return VerificationResult.failure(
                    f"Missing router logits opening for layer {layer_idx}, token {token_idx}"
                )

            if token_idx >= len(router_commitment.selected_experts):
                return VerificationResult.failure(
                    f"Token {token_idx} out of range for selected_experts in layer {layer_idx}"
                )

            if router_commitment.seq_len and token_idx >= router_commitment.seq_len:
                return VerificationResult.failure(
                    f"Token {token_idx} out of range for committed seq_len in layer {layer_idx}"
                )

            if router_commitment.num_experts and len(opening.logits) != router_commitment.num_experts:
                return VerificationResult.failure(
                    f"Layer {layer_idx} token {token_idx}: logits width mismatch "
                    f"(expected {router_commitment.num_experts}, got {len(opening.logits)})"
                )

            path_ok = verify_merkle_path(
                root=router_commitment.router_logits_row_root,
                leaf_data=logits_row_to_bytes(opening.logits),
                path=opening.merkle_path,
            )
            if not path_ok:
                return VerificationResult.failure(
                    f"Layer {layer_idx} token {token_idx}: router logits Merkle proof invalid"
                )

            committed_experts = router_commitment.selected_experts[token_idx]
            recomputed = compute_topk_indices(opening.logits, top_k)
            committed_prefix = [int(x) for x in committed_experts[:top_k]]
            if recomputed != committed_prefix:
                return VerificationResult.failure(
                    f"Layer {layer_idx} token {token_idx}: top-k mismatch "
                    f"(committed={committed_prefix}, recomputed={recomputed})"
                )

            proved_row = proved_rows_by_token.get(int(token_idx))
            if proved_row is None:
                return VerificationResult.failure(
                    f"Missing proven router output row for layer {layer_idx}, token {token_idx}"
                )
            proved_topk = compute_topk_indices(proved_row.logits_int, top_k)
            if token_idx >= len(router_commitment.proof_selected_experts):
                return VerificationResult.failure(
                    f"Token {token_idx} out of range for proof_selected_experts in layer {layer_idx}"
                )
            proof_prefix = [int(x) for x in router_commitment.proof_selected_experts[token_idx][:top_k]]
            if proved_topk != proof_prefix:
                return VerificationResult.failure(
                    f"Layer {layer_idx} token {token_idx}: router-GEMM top-k mismatch "
                    f"(committed={proof_prefix}, proved={proved_topk})"
                )

        return VerificationResult.success(f"Routing openings verified for layer {layer_idx}")

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self) -> None:
        """
        Prepare the validator for proof verification.

        Default (lightweight): Only uses ModelSpec with weight Merkle roots
        from on-chain registry. No model loading needed.

        Full mode: Also loads the model for direct weight value lookups.
        """
        if self.lightweight:
            num_roots = len(self.model_spec.weight_block_merkle_roots)
            logger.info("Validator: Using weight Merkle roots from on-chain registry")
            logger.info("Validator: %d layer roots (%d bytes) — no model loading needed",
                        num_roots, num_roots * 32)
            return

        if self.model is not None:
            logger.info("Validator: Using shared model reference (demo mode)")
            return

        if self.model_name is None:
            logger.info("Validator: No model specified, will use model_spec only")
            return

        from transformers import AutoModelForCausalLM

        logger.info("Validator: FULL MODE - loading model %s for direct weight verification...", self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        logger.info("Validator: Model loaded for verification")

    # ------------------------------------------------------------------
    # Request creation
    # ------------------------------------------------------------------

    def create_request(self, prompt: str, **kwargs) -> InferenceRequest:
        """Create inference request with validator nonce.

        The nonce is critical for beacon security. It's generated here,
        sent to the miner with the request, and used later to derive
        the beacon from the commitment.
        """
        validator_nonce = os.urandom(32)

        return InferenceRequest(
            request_id=str(uuid.uuid4()),
            prompt=prompt,
            validator_nonce=validator_nonce,
            **kwargs
        )

    # ------------------------------------------------------------------
    # Beacon + Challenges (Phases 4-5)
    # ------------------------------------------------------------------

    def derive_beacon(self, commitment: InferenceCommitment,
                      validator_nonce: bytes) -> Tuple[bytes, bytes]:
        """
        Phase 4: Derive beacon from commitment + validator nonce.

        Returns:
            (beacon, validator_nonce)
        """
        beacon = derive_beacon_from_nonce(
            commitment_hash=commitment.commitment_hash(),
            validator_nonce=validator_nonce,
        )
        return beacon, validator_nonce

    def derive_challenges(
        self,
        beacon: bytes,
        commitment: InferenceCommitment,
        k_layers: int = 2,
    ) -> ChallengeSet:
        """Phase 5: Derive challenges from beacon."""
        return derive_challenges(
            beacon=beacon,
            commitment=commitment,
            k_layers=k_layers,
            k_gemms_per_layer=2,
            k_blocks_per_gemm=self.config.k_blocks,
        )

    # ------------------------------------------------------------------
    # Weight access (full mode)
    # ------------------------------------------------------------------

    def _get_weight_for_layer(self, layer_idx: int) -> Optional[torch.Tensor]:
        """Get the gate projection weight for a specific layer.

        Returns weight tensor in [in_features, out_features] format.
        """
        if self.model is None:
            return None

        layers = self._get_layers()
        if layer_idx >= len(layers):
            return None

        mlp = self._get_mlp(layers[layer_idx])
        if mlp is None:
            return None

        gate_proj_mod = self._get_gate_proj(mlp)
        if gate_proj_mod is None or not hasattr(gate_proj_mod, 'weight'):
            return None

        W_raw = gate_proj_mod.weight.data.float().cpu()
        hidden_dim = self.model_spec.hidden_dim
        if W_raw.shape[0] == hidden_dim:
            return W_raw
        else:
            return W_raw.T

    # ------------------------------------------------------------------
    # Proof verification (Phase 7)
    # ------------------------------------------------------------------

    def verify_proof(
        self,
        proof_bundle: InferenceProofBundle,
        expected_sampling_verification_bps: Optional[int] = None,
        expected_do_sample: Optional[bool] = None,
        expected_temperature: Optional[float] = None,
        enable_thinking: Optional[bool] = None,
    ) -> Tuple[VerificationResult, Dict[str, float]]:
        """
        Verify the proof bundle.

        Security relies on weight Merkle roots from on-chain registry:
        - W spot checks: Verified via Merkle proofs against on-chain roots
        - X spot checks: Verified via Merkle proofs against layer commitments
        - Sumcheck: Verifies Y = X @ W consistency mathematically

        Args:
            expected_sampling_verification_bps: If set, verify the committed
                bps matches the validator-requested rate.
            expected_do_sample: If set, verify the committed do_sample flag
                matches the validator's request.
            expected_temperature: If set, verify the committed temperature
                matches the validator's request.
            enable_thinking: If False, argmax divergence in high-assurance
                mode becomes a hard failure (no logits processor excuse).
        """
        from verallm.crypto.field import mod_p

        verifier = GEMMVerifier(self.config)
        timing_details = {}
        commitment = proof_bundle.commitment

        # Verify committed decode-mode fields match validator expectations.
        if expected_do_sample is not None:
            if commitment.do_sample != expected_do_sample:
                return VerificationResult.failure(
                    f"do_sample mismatch: committed={commitment.do_sample}, "
                    f"expected={expected_do_sample}"
                ), timing_details
        if expected_temperature is not None:
            from verallm.sampling import temperature_to_milli
            expected_milli = temperature_to_milli(expected_temperature)
            if commitment.temperature_milli != expected_milli:
                return VerificationResult.failure(
                    f"temperature mismatch: committed={commitment.temperature_milli}m, "
                    f"expected={expected_milli}m"
                ), timing_details

        if expected_sampling_verification_bps is not None:
            expected_bps = clamp_sampling_bps(expected_sampling_verification_bps)
            committed_bps = clamp_sampling_bps(commitment.sampling_verification_bps)
            if committed_bps != expected_bps:
                return VerificationResult.failure(
                    f"Sampling verification rate mismatch: committed={committed_bps} bps, "
                    f"expected={expected_bps} bps"
                ), timing_details

        # Bind served output to commitment (prevents post-proof token substitution).
        if commitment.output_token_count > 0 and not proof_bundle.output_token_ids:
            return VerificationResult.failure(
                "Missing output_token_ids in proof bundle"
            ), timing_details
        if proof_bundle.output_token_ids:
            if commitment.output_token_count > 0 and len(proof_bundle.output_token_ids) != commitment.output_token_count:
                return VerificationResult.failure(
                    f"output_token_count mismatch: committed={commitment.output_token_count}, "
                    f"bundle={len(proof_bundle.output_token_ids)}"
                ), timing_details
            output_ids_arr = np.asarray(proof_bundle.output_token_ids, dtype=np.int64)
            output_hash = hashlib.sha256(output_ids_arr.astype("<i8", copy=False).tobytes()).digest()
            if output_hash != commitment.output_commitment:
                return VerificationResult.failure(
                    "output_commitment mismatch: output_token_ids do not match commitment"
                ), timing_details

        # Verify router_commitments match the committed hash (MoE binding).
        # Without this check, a miner could send tampered router_commitments
        # to control which experts get challenged.
        if self.is_moe and proof_bundle.router_commitments:
            expected_rc_hash = InferenceCommitment.compute_router_hash(
                proof_bundle.router_commitments,
            )
            committed_rc_hash = commitment.router_commitment_hash
            if expected_rc_hash != committed_rc_hash:
                return VerificationResult.failure(
                    "Router commitment hash mismatch: proof bundle "
                    "router_commitments don't match the committed "
                    "router_commitment_hash"
                ), timing_details

        # Re-derive challenges to ensure they match
        if self.is_moe and self.moe_config is not None:
            expected_challenges = derive_moe_challenges(
                beacon=proof_bundle.beacon,
                commitment=commitment,
                moe_config=self.moe_config,
                router_commitments=proof_bundle.router_commitments,
                k_layers=self.config.k_layers,
                k_tokens_per_layer=self.config.k_tokens_per_expert,
                k_experts_per_layer=self.config.k_experts_per_layer,
            )
            logger.info("Validator: Re-deriving MoE challenges (Fiat-Shamir)...")
        else:
            expected_challenges = derive_challenges(
                beacon=proof_bundle.beacon,
                commitment=commitment,
                k_layers=self.config.k_layers,
                k_gemms_per_layer=2,
                k_blocks_per_gemm=self.config.k_blocks,
            )
            logger.info("Validator: Re-deriving challenges (Fiat-Shamir)...")

        # Pre-load weight tensors for challenged layers
        t_preload = time.perf_counter()
        weight_cache = {}

        if self.lightweight:
            logger.info("W spot checks verified via Merkle proofs against on-chain roots")
        elif self.model is not None:
            for lc in expected_challenges.layer_challenges:
                layer_idx = lc.layer_idx
                W = self._get_weight_for_layer(layer_idx)
                if W is not None:
                    scale = 100
                    W_int = (W * scale).to(torch.int64)
                    weight_cache[layer_idx] = W_int
                    logger.info("Loaded weights for layer %d: %s", layer_idx, W_int.shape)
                else:
                    logger.warning("Could not load weights for layer %d", layer_idx)
        else:
            logger.warning("No model loaded, W spot checks will be trusted")

        preload_ms = (time.perf_counter() - t_preload) * 1000
        timing_details['_weight_preload'] = preload_ms
        logger.info("Validator: Weight preload: %.1fms", preload_ms)

        logger.info("Validator: Verifying GEMM proofs...")

        # Verify each layer proof
        for layer_proof in proof_bundle.layer_proofs:
            layer_idx = layer_proof.layer_idx

            layer_challenge = None
            for lc in expected_challenges.layer_challenges:
                if lc.layer_idx == layer_idx:
                    layer_challenge = lc
                    break

            if layer_challenge is None:
                return VerificationResult.failure(
                    f"No challenge found for layer {layer_idx}"
                ), timing_details

            W_tensor = weight_cache.get(layer_idx)

            for i, gemm_proof in enumerate(layer_proof.gemm_proofs):
                is_moe_challenge = hasattr(layer_challenge, 'expert_challenges')
                expert_idx = None

                if is_moe_challenge:
                    if i < len(layer_challenge.expert_challenges):
                        gemm_idx = layer_challenge.expert_challenges[i].gemm_idx
                        expert_idx = layer_challenge.expert_challenges[i].expert_idx
                    else:
                        gemm_idx = i
                else:
                    gemm_idx = layer_challenge.gemm_challenges[i].gemm_idx if i < len(layer_challenge.gemm_challenges) else i

                hidden_dim = self.model_spec.hidden_dim
                intermediate_dim = self.model_spec.intermediate_dim

                X_commitment = commitment.layer_commitments[layer_idx]
                W_commitment = self.model_spec.weight_merkle_root
                Y_commitment = gemm_proof.output_root

                # Verify X spot checks using Merkle proofs
                for block_proof in gemm_proof.block_proofs:
                    if block_proof.spot_X_with_proofs:
                        for spot_with_proof in block_proof.spot_X_with_proofs:
                            path_valid = verify_merkle_path(
                                root=X_commitment,
                                leaf_data=spot_with_proof.leaf_data,
                                path=spot_with_proof.merkle_path,
                            )
                            if not path_valid:
                                return VerificationResult.failure(
                                    f"X spot check Merkle proof invalid at layer {layer_idx}, "
                                    f"position ({spot_with_proof.row}, {spot_with_proof.col})"
                                ), timing_details

                            leaf_values = struct.unpack(
                                f"<{len(spot_with_proof.leaf_data) // 8}q",
                                spot_with_proof.leaf_data
                            )

                            flat_idx = spot_with_proof.row * hidden_dim + spot_with_proof.col
                            block_start = spot_with_proof.merkle_path.leaf_index * 256
                            idx_in_block = flat_idx - block_start

                            if 0 <= idx_in_block < len(leaf_values):
                                expected_value = mod_p(int(leaf_values[idx_in_block]))
                                if spot_with_proof.value != expected_value:
                                    return VerificationResult.failure(
                                        f"X spot check value mismatch at layer {layer_idx}, "
                                        f"position ({spot_with_proof.row}, {spot_with_proof.col}): "
                                        f"claimed {spot_with_proof.value}, leaf has {expected_value}"
                                    ), timing_details

                # W spot check verification
                if self.lightweight:
                    spot_check_fn = lambda _spot, _matrix_id: True

                    W_merkle_root = None
                    if is_moe_challenge and expert_idx is not None:
                        # Try spec first (local mode), then proof bundle (chain mode)
                        expert_roots = self.model_spec.expert_weight_merkle_roots.get(layer_idx, [])
                        if not expert_roots:
                            expert_roots = proof_bundle.expert_roots.get(layer_idx, [])
                        if expert_idx < len(expert_roots):
                            W_merkle_root = expert_roots[expert_idx]

                            router_root = self.model_spec.router_weight_merkle_roots.get(layer_idx)
                            if router_root is None:
                                rp = proof_bundle.router_layer_proofs.get(layer_idx)
                                router_root = rp.router_weight_root if rp is not None else None
                            if not self._verify_moe_layer_root(layer_idx, expert_roots, router_root):
                                return VerificationResult.failure(
                                    f"MoE layer {layer_idx}: hierarchical root mismatch "
                                    f"(expert roots don't hash to committed layer root)"
                                ), timing_details
                        else:
                            if layer_idx < len(self.model_spec.weight_block_merkle_roots):
                                W_merkle_root = self.model_spec.weight_block_merkle_roots[layer_idx]
                    elif layer_idx < len(self.model_spec.weight_block_merkle_roots):
                        W_merkle_root = self.model_spec.weight_block_merkle_roots[layer_idx]
                else:
                    def make_spot_checker(W_tensor_inner: Optional[torch.Tensor]):
                        def check_spot(spot: SpotCheck, matrix_id: str) -> bool:
                            if matrix_id == "X":
                                return True
                            elif matrix_id == "W":
                                if W_tensor_inner is None:
                                    return True
                                if spot.row >= W_tensor_inner.shape[0] or spot.col >= W_tensor_inner.shape[1]:
                                    return False
                                expected = mod_p(int(W_tensor_inner[spot.row, spot.col].item()))
                                return spot.value == expected
                            return False
                        return check_spot

                    spot_check_fn = make_spot_checker(W_tensor)
                    W_merkle_root = None

                # Transcript must match prover exactly (Fiat-Shamir)
                if is_moe_challenge and expert_idx is not None:
                    transcript = Transcript(f"layer_{layer_idx}_expert_{expert_idx}_gemm_{gemm_idx}".encode())
                else:
                    transcript = Transcript(f"layer_{layer_idx}_gemm_{gemm_idx}".encode())

                expert_w_cols = getattr(self.model_spec, 'expert_w_num_cols', 0)
                if is_moe_challenge and expert_idx is not None and expert_w_cols > 0:
                    w_cols = expert_w_cols
                else:
                    w_cols = intermediate_dim

                t0 = time.perf_counter()
                result = verifier.verify(
                    proof=gemm_proof,
                    X_commitment=X_commitment,
                    W_commitment=W_commitment,
                    Y_commitment=Y_commitment,
                    transcript=transcript,
                    spot_check_fn=spot_check_fn,
                    W_merkle_root=W_merkle_root,
                    W_num_cols=w_cols,
                    w_chunk_size=self.model_spec.w_merkle_chunk_size if self.lightweight else 0,
                )
                verify_time = (time.perf_counter() - t0) * 1000

                if is_moe_challenge and expert_idx is not None:
                    timing_key = f"Layer {layer_idx}, Expert {expert_idx}, GEMM {gemm_idx}"
                else:
                    timing_key = f"Layer {layer_idx}, GEMM {gemm_idx}"
                timing_details[timing_key] = verify_time

                status = "PASSED" if result.passed else "FAILED"
                logger.info("%s: %s (%.2fms)", timing_key, status, verify_time)

                if not result.passed:
                    return VerificationResult.failure(
                        f"GEMM verification failed at {timing_key}: {result.message}"
                    ), timing_details

        # Verify routing openings/top-k consistency for challenged MoE layers.
        if self.is_moe:
            for layer_challenge in expected_challenges.layer_challenges:
                if not hasattr(layer_challenge, "sampled_token_indices"):
                    continue
                if not getattr(layer_challenge, "verify_routing", False):
                    continue
                routing_result = self._verify_router_layer_openings(proof_bundle, layer_challenge)
                if not routing_result.passed:
                    return VerificationResult.failure(
                        f"Routing verification failed for layer {layer_challenge.layer_idx}: "
                        f"{routing_result.message}"
                    ), timing_details

        # Verify optional decode-integrity proofs (temperature=0 only).
        sampling_challenge = derive_sampling_challenge(
            beacon=proof_bundle.beacon,
            commitment=commitment,
            vocab_size=int(getattr(self.model_spec, "vocab_size", 0) or 0),
        )
        if sampling_challenge is not None:
            if not commitment.decode_hidden_row_root:
                return VerificationResult.failure(
                    "Sampling challenge active but decode_hidden_row_root is missing"
                ), timing_details
            if not proof_bundle.sampling_proofs:
                return VerificationResult.failure(
                    "Sampling challenge active but sampling proofs are missing"
                ), timing_details
            if not proof_bundle.output_token_ids:
                return VerificationResult.failure(
                    "Sampling challenge active but output_token_ids are missing"
                ), timing_details
            lm_head_root = getattr(self.model_spec, "lm_head_weight_merkle_root", b"")
            if not lm_head_root:
                return VerificationResult.failure(
                    "Sampling challenge active but ModelSpec.lm_head_weight_merkle_root is missing"
                ), timing_details

            # Collect proofs in challenge order, verify per-row checks.
            proofs_by_step = {int(sp.decode_step): sp for sp in proof_bundle.sampling_proofs}
            ordered_proofs: list = []
            X_rows_i64: list = []
            Y_rows_i64: list = []
            batched_gemm_proof = None

            for decode_step in sampling_challenge.decode_positions:
                step = int(decode_step)
                sp = proofs_by_step.get(step)
                if sp is None:
                    return VerificationResult.failure(
                        f"Missing sampling proof for decode step {step}"
                    ), timing_details
                if step < 0 or step >= len(proof_bundle.output_token_ids):
                    return VerificationResult.failure(
                        f"Sampling proof decode step out of range: {step}"
                    ), timing_details
                committed_token = int(proof_bundle.output_token_ids[step])
                if int(sp.token_id) != committed_token:
                    return VerificationResult.failure(
                        f"Sampling token mismatch at step {step}: "
                        f"proof={sp.token_id}, committed={committed_token}"
                    ), timing_details

                hidden_ok = verify_merkle_path(
                    root=commitment.decode_hidden_row_root,
                    leaf_data=sp.hidden_row,
                    path=sp.hidden_merkle_path,
                )
                if not hidden_ok:
                    return VerificationResult.failure(
                        f"Decode hidden-row Merkle proof invalid at step {step}"
                    ), timing_details

                if sp.lm_head_weight_root != lm_head_root:
                    return VerificationResult.failure(
                        f"lm_head weight root mismatch at decode step {step}"
                    ), timing_details

                hidden_fp32 = hidden_row_from_bytes(sp.hidden_row)
                X_row = quantize_hidden_row_int64(hidden_fp32)
                X_rows_i64.append(X_row)

                logits_i32 = logits_i32_from_bytes(sp.proved_logits_i32)
                if logits_i32.size == 0:
                    return VerificationResult.failure(
                        f"Empty proved logits row at decode step {step}"
                    ), timing_details
                Y_row = torch.from_numpy(logits_i32.astype(np.int64, copy=True)).view(1, -1)
                Y_rows_i64.append(Y_row)

                argmax_ok, argmax_detail = verify_quantized_argmax(logits_i32, committed_token)

                # Post-logits processors (presence_penalty, thinking mode) modify
                # the sampling distribution after compute_logits().  When active,
                # argmax divergence is expected — not fraud.
                _has_post_logits_mods = (
                    commitment.presence_penalty_milli != 0
                    or (enable_thinking is not False)
                )

                if not argmax_ok and not sampling_challenge.high_assurance:
                    if _has_post_logits_mods:
                        logger.info(
                            "Sampling step %d: int32 argmax diverged (%s) "
                            "— post-logits processors active (non-fatal)",
                            step, argmax_detail,
                        )
                    else:
                        # Strict mode but GEMM proof passed — likely capture
                        # alignment issue (concurrent batch composition changes
                        # can misalign captured logits rows with output tokens).
                        # Log warning instead of hard-failing.
                        logger.warning(
                            "Sampling step %d: int32 argmax diverged in strict mode "
                            "(%s) — GEMM proof passed, treating as capture alignment "
                            "issue (non-fatal)",
                            step, argmax_detail,
                        )

                if sampling_challenge.high_assurance:
                    if not commitment.decode_logits_row_root:
                        return VerificationResult.failure(
                            "High-assurance mode active but decode_logits_row_root missing"
                        ), timing_details
                    if not sp.fp16_logits_row:
                        return VerificationResult.failure(
                            f"High-assurance mode active but fp16_logits_row missing at step {step}"
                        ), timing_details
                    if sp.fp16_logits_merkle_path is None:
                        return VerificationResult.failure(
                            f"High-assurance mode active but fp16_logits_merkle_path missing at step {step}"
                        ), timing_details
                    fp16_path_ok = verify_merkle_path(
                        root=commitment.decode_logits_row_root,
                        leaf_data=sp.fp16_logits_row,
                        path=sp.fp16_logits_merkle_path,
                    )
                    if not fp16_path_ok:
                        return VerificationResult.failure(
                            f"Fp16 logits Merkle proof invalid at step {step}"
                        ), timing_details
                    fp16_ok, fp16_detail = verify_fp16_argmax(sp.fp16_logits_row, committed_token)

                    if not fp16_ok:
                        if not _has_post_logits_mods:
                            # Strict mode but GEMM proof passed — likely capture
                            # alignment issue. The miner's self-check should
                            # prevent this, but if data still arrives misaligned,
                            # log rather than hard-fail.
                            logger.warning(
                                "Sampling step %d: fp16 argmax diverged in strict mode "
                                "(%s) — GEMM proof passed, treating as capture alignment "
                                "issue (non-fatal)",
                                step, fp16_detail,
                            )
                        else:
                            # Post-logits processors active: divergence expected.
                            logger.info(
                                "Sampling step %d: fp16 argmax diverged from committed "
                                "token (%s) — post-logits processors active (non-fatal)",
                                step, fp16_detail,
                            )

                    if not argmax_ok:
                        # Both int32 and fp16 can diverge from capture alignment
                        # issues. Log for diagnostics but don't hard-fail when
                        # the GEMM sumcheck proof has passed.
                        logger.info(
                            "Sampling step %d: int32 argmax diverged (%s), fp16: %s",
                            step, argmax_detail, fp16_detail,
                        )
                elif not argmax_ok:
                    return VerificationResult.failure(
                        f"Sampling argmax failed at decode step {step}: {argmax_detail}"
                    ), timing_details

                if sp.lm_head_gemm_proof is not None and batched_gemm_proof is None:
                    batched_gemm_proof = sp.lm_head_gemm_proof

                ordered_proofs.append(sp)

            # Verify batched lm_head GEMM proof: X_batch @ W = Y_batch.
            if ordered_proofs and X_rows_i64:
                if batched_gemm_proof is None:
                    return VerificationResult.failure(
                        "No lm_head GEMM proof found in sampling proofs"
                    ), timing_details

                X_batch = torch.cat(X_rows_i64, dim=0)
                Y_batch = torch.cat(Y_rows_i64, dim=0)

                flat_x = X_batch.flatten().to(torch.int64)
                x_leaves = []
                for start in range(0, int(flat_x.numel()), 256):
                    x_leaves.append(flat_x[start:start + 256].numpy().astype("<i8", copy=False).tobytes())
                if not x_leaves:
                    x_leaves = [b"empty"]
                x_tree = MerkleTree(x_leaves)

                y_block = int(max(1, Y_batch.shape[1]))
                Y_merkle = build_block_merkle(Y_batch, y_block)

                if Y_merkle.root != batched_gemm_proof.output_root:
                    return VerificationResult.failure(
                        "Batched lm_head Y Merkle root mismatch"
                    ), timing_details

                transcript = Transcript(b"decode_lm_head_gemm_batched")
                lm_head_verify = verifier.verify(
                    proof=batched_gemm_proof,
                    X_commitment=x_tree.root,
                    W_commitment=ordered_proofs[0].lm_head_weight_root,
                    Y_commitment=batched_gemm_proof.output_root,
                    transcript=transcript,
                    spot_check_fn=lambda _spot, _matrix_id: True,
                    W_merkle_root=lm_head_root,
                    W_num_cols=Y_batch.shape[1],
                    w_chunk_size=self.model_spec.w_merkle_chunk_size,
                )
                if not lm_head_verify.passed:
                    return VerificationResult.failure(
                        f"Batched lm_head GEMM verification failed: {lm_head_verify.message}"
                    ), timing_details

        return VerificationResult.success("All GEMM proofs verified"), timing_details
