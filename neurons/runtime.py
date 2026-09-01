"""Runtime-family detection and scoring profiles for miner endpoints.

The on-chain ``MinerRegistry.MinerModel`` struct has no runtime field, so the
free-form ``quant`` string carries the family: GGUF-mesh endpoints register
with a ``gguf_mesh`` prefix (for example ``gguf_mesh_q4_k_m``). vLLM solo
endpoints keep plain quant strings (``fp8``, ``int4``, ``bf16``, ...), so the
flag defaults to false everywhere and existing miners are unaffected.

Mesh scoring profiles are derived from the owner-curated mesh catalogue
(verallm/registry/models.py MESH_MODELS): every on-chain mesh model id the
owner ships is an approved profile, and nothing else is. There is
deliberately no prefix, case-folded, or fuzzy fallback: unapproved GGUF
aliases fail closed instead of borrowing another model's score.

Context policy (matches the vLLM path): light canaries and organic traffic
run and score at the ADVERTISED (registered) context, which the mesh flow
guarantees is the launch's measured KV auto-fit. Only HARD-tier proof
challenges are context-capped, and that cap lives with the proof
calibration, not here.

See docs/architecture/mesh_contract_integration.md (Registration Conventions).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Mapping

logger = logging.getLogger(__name__)

MESH_QUANT_PREFIX = "gguf_mesh"


@dataclass(frozen=True)
class MeshScoringFacts:
    """The three quality facts epoch scoring consumes for a model."""

    active_params_b: float
    moe_dense_equivalent: float
    generation_quality: float
    # The logical emission-bucket key. Without it every quant rung of a
    # mesh-only model formed its OWN bucket, while
    # vLLM variants of one base model correctly shared one. Which rung an
    # operator serves is their choice; the quant quality factor already
    # prices that choice inside the shared bucket.
    base_model: str = ""


@dataclass(frozen=True)
class MeshModelScoringProfile:
    """Validator scoring metadata for one exact on-chain mesh model ID.

    Mesh model IDs are deliberately separate from the logical model registry
    IDs used by single-GPU runtimes.  ``logical_model_id`` links to the vLLM
    registry entry when one exists; mesh-only models carry the catalogue's
    own quality facts instead (``quality_params_b`` as the dense-equivalent
    estimate plus ``generation_quality``).
    """

    model_id: str
    logical_model_id: str
    runtime_family: str
    gguf_scheme: str
    scoring_quant: str
    native_context_len: int
    quality_params_b: float = 0.0
    generation_quality: float = 1.0
    # Emission-bucket group for mesh-only models (models WITH a
    # logical_model_id group through that registry entry's base_model
    # instead). The catalogue's tokenizer repo IS the base model.
    base_model: str = ""

    @property
    def registry_quant(self) -> str:
        """Return the exact MinerRegistry quant marker for this profile."""
        return f"{self.runtime_family}_{self.gguf_scheme.lower()}"

    def accepts_registry_quant(self, quant: str | None) -> bool:
        """Whether ``quant`` identifies this profile's GGUF scheme."""
        return (quant or "").strip().lower() == self.registry_quant

    def scored_context_len(self, advertised_context_len: int) -> int:
        """The context scoring uses: the advertised (registered) value.

        The mesh registration flow measures this with the KV auto-fit and
        validators canary at it, so the advert IS the exercised capability.
        """
        try:
            return max(0, int(advertised_context_len or 0))
        except (TypeError, ValueError):
            return 0

    def fallback_scoring_facts(self) -> MeshScoringFacts | None:
        """Quality facts for mesh-only models with no registry entry."""
        if self.quality_params_b <= 0:
            return None
        return MeshScoringFacts(
            active_params_b=self.quality_params_b,
            moe_dense_equivalent=self.quality_params_b,
            generation_quality=self.generation_quality,
            base_model=self.base_model,
        )


_PROFILE_CACHE: dict[str, MeshModelScoringProfile] | None = None


def _build_profiles() -> dict[str, MeshModelScoringProfile]:
    # Imported lazily: verallm.mesh.registration imports this module, so a
    # module-level import of verallm.registry.models would risk a cycle.
    from verallm.registry.models import MESH_MODELS

    profiles: dict[str, MeshModelScoringProfile] = {}
    for entry in MESH_MODELS:
        for variant in entry.quants:
            profiles[variant.mesh_model_id] = MeshModelScoringProfile(
                model_id=variant.mesh_model_id,
                logical_model_id=entry.registry_model_id,
                runtime_family=MESH_QUANT_PREFIX,
                gguf_scheme=variant.gguf_scheme,
                # Quant quality resolves through the catalogue's per-scheme
                # GGUF table in compute_model_base_utility (a q4_k_m and an
                # iq2_m of the same model must not score the same quality).
                scoring_quant=f"gguf_{variant.gguf_scheme.lower()}",
                native_context_len=int(entry.native_context_len),
                quality_params_b=float(entry.quality_params_b),
                generation_quality=float(entry.generation_quality),
                base_model=str(
                    getattr(entry, "tokenizer_hf_repo", "") or ""
                ),
            )
    return profiles


def _profiles() -> Mapping[str, MeshModelScoringProfile]:
    global _PROFILE_CACHE
    if _PROFILE_CACHE is None:
        _PROFILE_CACHE = _build_profiles()
    return _PROFILE_CACHE


#: Rate limit for miss-triggered registry reloads (seconds). A registry
#: sync landing on disk AFTER this process imported the module otherwise
#: leaves every new model "not approved" until a manual restart — the
#: process-predates-registry trap.
_REGISTRY_RELOAD_MIN_INTERVAL_S = 600.0
_registry_reload_at = 0.0


def _reload_profiles_on_miss(model_id: str) -> Mapping[str, MeshModelScoringProfile]:
    """One rate-limited on-disk registry reload when a model id misses.

    verallm.registry.models is a pure-data module (dataclass catalogs, no
    side effects), so importlib.reload picks up a synced file in place.
    Only a genuinely new on-disk approval changes the outcome; a miss for
    a truly unknown id stays a miss and costs one reload per interval.
    """
    global _PROFILE_CACHE, _registry_reload_at
    import time

    now = time.monotonic()
    if now - _registry_reload_at < _REGISTRY_RELOAD_MIN_INTERVAL_S:
        return _profiles()
    _registry_reload_at = now
    try:
        import importlib

        import verallm.registry.models as _registry_models

        importlib.reload(_registry_models)
        rebuilt = _build_profiles()
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("mesh registry reload failed: %s", exc)
        return _profiles()
    _PROFILE_CACHE = rebuilt
    if model_id in rebuilt:
        logger.info(
            "mesh registry reloaded from disk: %s is now approved "
            "(process predated the registry sync)",
            model_id,
        )
    return rebuilt


def is_mesh_quant(quant: str | None) -> bool:
    """True when a MinerRegistry quant string marks a GGUF-mesh endpoint."""
    if not quant:
        return False
    return quant.strip().lower().startswith(MESH_QUANT_PREFIX)


def normalize_mesh_quant(quant: str | None) -> str:
    """Return a quant string guaranteed to carry the mesh prefix.

    ``None``/empty -> ``gguf_mesh``; ``q4_k_m`` -> ``gguf_mesh_q4_k_m``;
    already-prefixed values pass through unchanged.
    """
    q = (quant or "").strip().lower()
    if not q:
        return MESH_QUANT_PREFIX
    if q.startswith(MESH_QUANT_PREFIX):
        return q
    return f"{MESH_QUANT_PREFIX}_{q}"


def get_mesh_model_scoring_profile(
    model_id: str | None,
    quant: str | None = None,
) -> MeshModelScoringProfile | None:
    """Resolve an exact mesh model ID, optionally validating its quant marker.

    There is intentionally no prefix, case-folded, or fuzzy model-ID fallback:
    unapproved GGUF aliases fail closed instead of borrowing a logical model's
    score.
    """
    profile = _profiles().get(model_id or "")
    if profile is None and model_id:
        # The registry file may be newer than this process (owner sync
        # without restart); one rate-limited reload closes that gap.
        profile = _reload_profiles_on_miss(model_id).get(model_id)
    if profile is None:
        return None
    if quant is not None and not profile.accepts_registry_quant(quant):
        return None
    return profile
