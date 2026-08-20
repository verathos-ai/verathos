"""Secret-seeded proof-v3 canary planning and prompt construction."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Tuple

from neurons.runtime import is_mesh_quant
from verallm.proof_v3.canary_policy import (
    CANARY_HARD_DECODE_ANCHORS_V3,
    CANARY_OWNER_CONTEXT_SIZING_ABI_V3,
    DEFAULT_CANARY_CANDIDATE_HARD_BPS_V3,
    DEFAULT_CANARY_CANDIDATE_TARGET_PER_EPOCH_V3,
    DEFAULT_CANARY_OWNER_FULL_MAX_DRAW_BPS_V3,
    DEFAULT_CANARY_OWNER_FULL_MIN_PROMPT_BPS_V3,
    LEGACY_CANARY_OWNER_CONTEXT_SIZING_ABI_V3,
    DEFAULT_CANARY_REPEAT_PREFIX_MIN_TOKENS_V3,
    DEFAULT_CANARY_REPEAT_PREFIX_TARGET_BPS_V3,
    MIN_CANARY_HARD_DECODE_ANCHOR_BPS_V3,
    MIN_CANARY_HARD_DECODE_TAIL_BPS_V3,
    canary_prompt_token_tolerance_v3,
)


_TASK_FRAMES = (
    "Review the material below and produce a concise decision memo with assumptions, risks, and concrete next steps.",
    "Read the source material and extract the key facts into a structured report. Resolve apparent inconsistencies explicitly.",
    "Rewrite the supplied material for a technically literate audience while preserving every numerical detail.",
    "Analyze the following collection and answer with a comparison table followed by a short recommendation.",
    "Treat the following as working notes. Organize them into a coherent implementation brief and identify missing information.",
    "Summarize the document, then list claims that can be checked directly from the text and claims that require outside evidence.",
    "Transform the source into a clear tutorial with an example, a counterexample, and a final checklist.",
    "Extraire les informations essentielles du document, puis rédiger une synthèse structurée en français.",
    "Analiza el material siguiente y prepara un informe breve con hallazgos, límites y acciones recomendadas.",
    "Inspect the records below, group related entries, and explain any trend or anomaly without inventing data.",
    "Use the supplied reference as the sole source. Draft a question-and-answer guide that covers both common and edge cases.",
    "Convert the source material into release notes for operators, separating behavior changes, compatibility, and validation steps.",
    "Draft a response to the records below that distinguishes observations, interpretations, and unresolved questions.",
    "Turn the source into an implementation checklist, preserving identifiers and measurable acceptance criteria.",
    "Compare the entries, identify the strongest supported conclusion, and state what the evidence does not establish.",
    "Prepare an executive summary followed by a technical appendix that cites the supplied record identifiers.",
    "Audit the following material for internal consistency and propose the smallest set of corrections.",
    "Create a migration guide from the source, including prerequisites, validation steps, and rollback conditions.",
    "Answer the implied user request using only the supplied evidence, then list two useful follow-up questions.",
    "Convert the records into a chronological incident report without assigning causes that are not documented.",
    "Produce a compact data dictionary and explain how the fields relate across the supplied entries.",
    "Write a neutral briefing that makes the numerical trade-offs understandable to a non-specialist.",
    "Generate a test plan from the material, with normal cases, boundary cases, and expected outcomes.",
    "Refactor the supplied notes into a clean API specification with inputs, outputs, errors, and examples.",
    "Identify duplicate or conflicting entries, reconcile them where possible, and preserve uncertainty where not.",
    "Prepare a handover note that separates completed work, current state, blockers, and the next operator actions.",
    "Summarize the evidence as a set of claims, attaching the relevant identifier and measurement to each claim.",
    "Convert the material into a troubleshooting guide ordered from the least disruptive check to the most invasive.",
    "Rédigez une note de synthèse en français, en conservant exactement les identifiants et les valeurs numériques.",
    "Elabora una guía operativa en español basada exclusivamente en los datos proporcionados.",
    "Formuliere aus den Unterlagen eine kurze technische Bewertung mit Annahmen, Grenzen und nächsten Schritten.",
    "Create a concise FAQ whose answers are grounded entirely in the supplied source material.",
    "Write a reviewer comment that explains the main issue, its evidence, and one proportionate recommendation.",
    "Transform the records into a release-readiness assessment with explicit pass, fail, and unknown criteria.",
    "Extract a dependency graph in prose, then explain which dependency creates the largest scheduling risk.",
    "Prepare a comparison of the observed baseline and revised state, including every recorded measurement.",
    "Turn the source into a reproducible runbook and call out any step whose required input is missing.",
    "Analyze the dataset as if preparing a quality-control report; report anomalies without correcting the source.",
    "Write a short proposal that uses the records as constraints and includes one conservative alternative.",
    "Convert the supplied entries into normalized pseudocode and explain any ambiguous branch conditions.",
    "Produce a change-impact note covering users, operators, compatibility, observability, and validation.",
    "Create an annotated outline for a longer report, citing the source references beneath each section.",
    "Explain the material through one worked example and one counterexample while retaining the source values.",
    "Classify each statement as fact, inference, decision, or open question and provide a concise synthesis.",
)

_DOMAINS = (
    "warehouse logistics",
    "renewable-energy planning",
    "software release operations",
    "clinical trial administration",
    "museum collection management",
    "regional rail scheduling",
    "agricultural water use",
    "customer support quality",
    "network capacity planning",
    "public procurement",
    "scientific data curation",
    "multilingual publishing",
    "manufacturing maintenance",
    "university course design",
    "financial reconciliation",
    "environmental monitoring",
    "distributed database operations",
    "medical device inventory",
    "airport ground services",
    "telecommunications maintenance",
    "library digitization",
    "semiconductor fabrication",
    "insurance claims review",
    "food safety inspection",
    "satellite mission planning",
    "construction project controls",
    "open-source package maintenance",
    "emergency response logistics",
    "battery storage operations",
    "pharmaceutical supply chains",
    "urban mobility analysis",
    "data-center energy management",
    "archaeological field records",
    "maritime cargo handling",
    "laboratory instrument calibration",
    "accessibility compliance",
    "privacy incident response",
    "geospatial survey processing",
    "robotics fleet maintenance",
    "educational assessment design",
    "weather observation quality",
    "digital asset preservation",
    "municipal budget planning",
    "industrial process control",
    "cybersecurity patch coordination",
    "retail demand forecasting",
    "translation quality assurance",
    "wildlife conservation monitoring",
)

_SOURCE_INTROS = (
    "The following working material concerns {domain}.",
    "Below is a mixed-format extract from a {domain} workflow.",
    "A colleague supplied these records from {domain}.",
    "Use the following {domain} notes as the complete reference set.",
    "The source below combines observations from {domain}.",
    "This is an unedited export from a {domain} review.",
    "The material that follows was assembled during {domain} planning.",
    "Treat the entries below as a bounded {domain} case file.",
    "The following records describe a routine {domain} task.",
    "Here is a partial but authoritative {domain} working set.",
    "The source is a collection of {domain} status records.",
    "The supplied evidence comes from a {domain} handover.",
)

_GROUNDING_RULES = (
    "Keep identifiers, dates, and quantities exact.",
    "Do not add facts that are absent from the source.",
    "Separate direct evidence from interpretation.",
    "Preserve contradictory entries instead of silently resolving them.",
    "Cite record references wherever a conclusion depends on one.",
    "State uncertainty explicitly when the records do not settle a point.",
    "Retain units and distinguish measured values from targets.",
    "Use the source as the sole factual authority.",
    "Do not normalize values unless the requested output requires it.",
    "Keep the chronology intact when dates affect the conclusion.",
    "Call out missing inputs rather than guessing them.",
    "Make each recommendation traceable to at least one supplied entry.",
)

_SOURCE_HEADERS = (
    "Working notes:",
    "Source records:",
    "Reference extract:",
    "Collected entries:",
    "Input material:",
    "Case records:",
    "Raw export:",
    "Evidence set:",
)

_LOW_CLOSINGS = (
    "Answer directly and keep the result grounded in the supplied material.",
    "Return the requested result without adding unrelated background.",
    "Keep the response concise, but do not omit a relevant recorded value.",
    "Use a clear structure and finish with the most useful next action.",
    "Do not reproduce the entire source unless a quotation is necessary.",
    "Prefer precise language over broad generalizations.",
    "Make the final response usable without access to any outside source.",
    "Conclude with the main limitation of the available evidence.",
)

_FULL_CLOSINGS = (
    "Return only the requested deliverable and keep every conclusion traceable to the source.",
    "Use the full record set; do not infer a trend from an isolated entry.",
    "Preserve important edge cases even if they complicate the summary.",
    "Finish with a compact checklist that can be applied to the complete material.",
    "Treat repeated entries as evidence only after checking their identifiers and dates.",
    "Keep the response self-contained and explicitly identify unresolved conflicts.",
    "Base the recommendation on the whole supplied record, not only the most recent section.",
    "Ensure that numerical comparisons retain their original units and reference IDs.",
)

_LONG_OUTPUT_DIRECTIVES = (
    (
        "Develop the deliverable exhaustively and continue until it contains "
        "at least {minimum_output_tokens} output tokens. Do not end after an "
        "initial summary; expand every section with grounded examples, "
        "counterexamples, and checks from the supplied records."
    ),
    (
        "This is a long-form drafting request. Produce at least "
        "{minimum_output_tokens} output tokens before concluding. Continue "
        "with numbered evidence notes, worked comparisons, and implementation "
        "details rather than stopping at a short answer."
    ),
    (
        "Use the available response budget for a complete technical treatment "
        "of at least {minimum_output_tokens} output tokens. If the main answer "
        "finishes early, add traceable case analyses and verification "
        "checklists based only on the source."
    ),
    (
        "Write an extended self-contained report of no fewer than "
        "{minimum_output_tokens} output tokens. Keep developing distinct "
        "source-grounded sections until that minimum is reached; do not use "
        "filler or introduce outside facts."
    ),
    (
        "Return a detailed long-form deliverable and keep writing until at "
        "least {minimum_output_tokens} output tokens have been produced. "
        "Cover each source category, numerical relationship, ambiguity, and "
        "validation step separately."
    ),
    (
        "The requested output is intentionally extensive: produce at least "
        "{minimum_output_tokens} output tokens. After the primary analysis, "
        "continue with a source-by-source appendix and a comprehensive "
        "evidence-to-conclusion matrix."
    ),
    (
        "Prepare a full-length treatment of at least "
        "{minimum_output_tokens} output tokens. Do not conclude early; expand "
        "the methodology, edge cases, alternatives, and reproducible checks "
        "while remaining grounded in the supplied material."
    ),
    (
        "Continue the response through at least {minimum_output_tokens} output "
        "tokens. Use substantive numbered sections, detailed examples, and "
        "record-level cross-checks so the requested length remains useful."
    ),
)

_ACTORS = (
    "North team",
    "Orchid group",
    "River unit",
    "Delta office",
    "Atlas project",
    "Cedar lab",
    "Juniper service",
    "Harbor desk",
    "Summit program",
    "Lumen division",
    "Quartz cohort",
    "Meadow site",
)

_ACTIONS = (
    "validated the revised schedule",
    "flagged a dependency mismatch",
    "completed the second review",
    "requested a narrower acceptance rule",
    "recorded stable throughput",
    "found an inconsistent identifier",
    "approved the staged migration",
    "deferred the optional extension",
    "reconciled the source records",
    "measured a lower error rate",
    "documented an operational constraint",
    "confirmed the fallback procedure",
)

_QUALIFIERS = (
    "after comparing the two independent records",
    "under the stated resource limit",
    "without changing the public interface",
    "for the next scheduled window",
    "with the earlier baseline held constant",
    "after excluding incomplete observations",
    "using the same evaluation criteria",
    "while preserving backwards compatibility",
    "subject to one final consistency check",
    "with no unresolved data-loss warning",
)


@dataclass
class CanaryTest:
    """One independently scheduled canary obligation."""

    miner_address: str
    miner_endpoint: str
    model_id: str
    model_index: int
    max_context_len: int
    target_block: int
    test_index: int
    test_type: str
    prompt: str
    max_new_tokens: int
    temperature: float
    verify_proof: bool
    verify_tee: bool = False
    # GGUF-mesh runtime: execution is routed through the mesh verification
    # lane (receipts + light/hard audits) instead of the vLLM proof path.
    # Scheduling is identical to every other endpoint; only transport and
    # verification differ.  The hard demand for a mesh canary is derived
    # from ``verify_proof`` at dispatch time.
    verify_mesh: bool = False
    mesh_audit_tier: str = ""
    enable_thinking: bool = False
    presence_penalty: Optional[float] = 0.0
    top_k: Optional[int] = 0
    top_p: Optional[float] = 1.0
    proof_protocol_versions: Tuple[int, ...] = ()
    quant: str = ""
    obligation_id: str = ""
    target_prompt_tokens: int = 0
    measured_prompt_tokens: int = 0
    minimum_observed_output_tokens: int = 0
    prompt_seed: bytes = field(default=b"", repr=False)
    repeat_prefix_seed: bytes = field(default=b"", repr=False)
    repeat_prefix_target_tokens: int = 0
    attempt_count: int = 0
    max_attempts: int = 0
    is_full_context_debt: bool = False
    full_pair_id: str = ""
    full_pair_slot: int = -1
    full_pair_hold_seconds: int = 0
    rejection_intervals: List[Tuple[float, float]] = field(
        default_factory=list,
        repr=False,
    )


def _derive(
    domain: bytes,
    *parts: bytes,
) -> bytes:
    return hashlib.sha256(domain + b"\x00".join(parts)).digest()


def _seed_int(seed: bytes, label: bytes, modulus: int) -> int:
    if modulus <= 0:
        return 0
    return int.from_bytes(_derive(label, seed)[:8], "big") % modulus


def _log_uniform_decode_bands_v3(
    max_decode_tokens: int,
) -> tuple[tuple[int, int], ...]:
    """Partition the complete signed decode range into logarithmic bands."""

    cap = int(max_decode_tokens)
    if cap <= 0:
        raise ValueError("hard-decode cap must be positive")
    bands: list[tuple[int, int]] = []
    lower = 1
    while lower <= cap:
        upper = min(cap, 2 * lower - 1)
        bands.append((lower, upper))
        lower = upper + 1
    # A power-of-two cap would otherwise leave a high-cost singleton band.
    # Merge that endpoint into the preceding band while retaining full support.
    if len(bands) > 1 and bands[-1][0] == bands[-1][1]:
        previous = bands[-2]
        bands[-2] = (previous[0], bands[-1][1])
        bands.pop()
    return tuple(bands)


def _signed_full_decode_tokens_v3(
    seed: bytes,
    *,
    category: str,
    max_decode_tokens: int,
    common_max_decode_tokens: int,
) -> int:
    """Select one decode length from the signed paired distribution."""

    cap = int(max_decode_tokens)
    if cap <= 0:
        raise ValueError("hard-decode cap must be positive")
    if category == "common":
        lower = min(64, cap)
        upper = min(
            cap,
            max(lower, int(common_max_decode_tokens)),
        )
        return lower + _seed_int(
            seed,
            b"full_decode_common",
            upper - lower + 1,
        )
    if category == "anchor":
        anchors = tuple(
            dict.fromkeys(
                min(cap, int(value))
                for value in CANARY_HARD_DECODE_ANCHORS_V3
            )
        )
        return anchors[_seed_int(seed, b"full_decode_anchor", len(anchors))]
    if category == "tail":
        bands = _log_uniform_decode_bands_v3(cap)
        lower, upper = bands[
            _seed_int(seed, b"full_decode_tail_band", len(bands))
        ]
        return lower + _seed_int(
            seed,
            b"full_decode_tail_offset",
            upper - lower + 1,
        )
    raise ValueError("hard-decode category is unsupported")


def _paired_full_decode_categories_v3(
    seed: bytes,
    *,
    anchor_bps: int,
    tail_bps: int,
) -> tuple[str, str]:
    """Return equal-marginal categories with at most one expensive slot."""

    anchors = int(anchor_bps)
    tail = int(tail_bps)
    extended = anchors + tail
    if (
        anchors < MIN_CANARY_HARD_DECODE_ANCHOR_BPS_V3
        or tail < MIN_CANARY_HARD_DECODE_TAIL_BPS_V3
        or 2 * extended > 10_000
    ):
        raise ValueError("paired hard-decode rates are invalid")
    categories = ["common", "common"]
    if _seed_int(seed, b"full_decode_extended_draw", 10_000) < 2 * extended:
        slot = _seed_int(seed, b"full_decode_extended_slot", 2)
        kind_draw = _seed_int(seed, b"full_decode_extended_kind", extended)
        categories[slot] = "anchor" if kind_draw < anchors else "tail"
    return categories[0], categories[1]


def _independent_full_decode_category_v3(
    seed: bytes,
    *,
    anchor_bps: int,
    tail_bps: int,
) -> str:
    """Select one lower-heavy decode category for an independent candidate."""

    anchors = int(anchor_bps)
    tail = int(tail_bps)
    if (
        anchors < MIN_CANARY_HARD_DECODE_ANCHOR_BPS_V3
        or tail < MIN_CANARY_HARD_DECODE_TAIL_BPS_V3
        or anchors + tail >= 10_000
    ):
        raise ValueError("independent hard-decode rates are invalid")
    draw = _seed_int(seed, b"full_decode_category", 10_000)
    if draw < anchors:
        return "anchor"
    if draw < anchors + tail:
        return "tail"
    return "common"


def _log_uniform_prompt_bands_v3(
    minimum: int,
    maximum: int,
) -> tuple[tuple[int, int], ...]:
    """Partition [minimum, maximum] into doubling bands from the minimum.

    Mesh-lane full-context sizing draws a band uniformly, so equal band
    probability with doubling widths makes SMALL prompts the common case
    while every size up to the advertised maximum stays reachable - the
    decode-side twin of ``_log_uniform_decode_bands_v3``.
    """

    low = int(minimum)
    high = int(maximum)
    if low <= 0 or high < low:
        raise ValueError("prompt band range is invalid")
    bands: list[tuple[int, int]] = []
    lower = low
    while lower <= high:
        upper = min(high, 2 * lower - 1)
        bands.append((lower, upper))
        lower = upper + 1
    # Merge a trailing singleton into the previous band (same rationale
    # as the decode bands: no high-cost singleton band).
    if len(bands) > 1 and bands[-1][0] == bands[-1][1]:
        previous = bands[-2]
        bands[-2] = (previous[0], bands[-1][1])
        bands.pop()
    return tuple(bands)


def _mesh_full_prompt_target_v3(
    seed: bytes,
    *,
    minimum: int,
    maximum: int,
    max_draw_bps: int,
) -> int:
    """Min-heavy full-context prompt size for MESH miners.

    llama.cpp prefill is minutes-long at advertised-maximum sizes and
    monopolizes the mesh, so unlike vLLM the mesh lane
    must not exercise the maximum every epoch. The signed Bernoulli tail
    keeps the exact maximum always POSSIBLE - capacity cannot be faked
    down - while the log-uniform bands make small draws the common case.
    Seed-deterministic: every validator draws the same size.
    """

    low = int(minimum)
    high = int(maximum)
    if low <= 0 or high < low:
        raise ValueError("mesh full-prompt range is invalid")
    if _seed_int(seed, b"mesh_full_prompt_max_draw", 10_000) < int(
        max_draw_bps
    ):
        return high
    bands = _log_uniform_prompt_bands_v3(low, high)
    # GEOMETRIC band weighting, not uniform: uniform gave the widest
    # (largest) band the same probability as the smallest, so a 264k
    # advert drew a 200k+ multi-minute prefill roughly every third full
    # canary. Halving each band's weight relative to the previous keeps
    # every size reachable (and the exact maximum through the signed
    # tail above) while the MEDIAN draw lands in the smallest bands.
    # Integer thresholds keep the draw seed-deterministic across
    # validators exactly like the uniform pick it replaces.
    weights = [2 ** (len(bands) - 1 - i) for i in range(len(bands))]
    total = sum(weights)
    pick = _seed_int(seed, b"mesh_full_prompt_band", total)
    cumulative = 0
    band_index = len(bands) - 1
    for i, weight in enumerate(weights):
        cumulative += weight
        if pick < cumulative:
            band_index = i
            break
    lower, upper = bands[band_index]
    return lower + _seed_int(
        seed,
        b"mesh_full_prompt_offset",
        upper - lower + 1,
    )


def _owner_full_prompt_target_v3(
    seed: bytes,
    *,
    minimum: int,
    maximum: int,
    max_draw_bps: int,
    sizing_abi_id: str = CANARY_OWNER_CONTEXT_SIZING_ABI_V3,
) -> int:
    """Draw an owner-only min-heavy context with an exact-ceiling tail.

    The explicit sizing ABI preserves the v6 uniform-band schedule while a
    validator is still consuming that signed policy.  Policy v7 weights each
    successive doubling band by one half, matching the lower-heavy mesh
    schedule without making any prompt-length band impossible.
    """

    low = int(minimum)
    high = int(maximum)
    tail_bps = int(max_draw_bps)
    if low <= 0 or high < low:
        raise ValueError("owner full-prompt range is invalid")
    if not 0 < tail_bps <= 10_000:
        raise ValueError("owner full-prompt maximum rate is invalid")
    sizing_abi = str(sizing_abi_id)
    if sizing_abi == LEGACY_CANARY_OWNER_CONTEXT_SIZING_ABI_V3:
        max_domain = b"owner_full_prompt_max_draw"
        band_domain = b"owner_full_prompt_band"
        offset_domain = b"owner_full_prompt_offset"
    elif sizing_abi == CANARY_OWNER_CONTEXT_SIZING_ABI_V3:
        max_domain = b"owner_geometric_prompt_max_draw"
        band_domain = b"owner_geometric_prompt_band"
        offset_domain = b"owner_geometric_prompt_offset"
    else:
        raise ValueError("owner full-prompt sizing ABI is unsupported")
    if _seed_int(seed, max_domain, 10_000) < tail_bps:
        return high
    bands = _log_uniform_prompt_bands_v3(low, high)
    if sizing_abi == LEGACY_CANARY_OWNER_CONTEXT_SIZING_ABI_V3:
        band_index = _seed_int(seed, band_domain, len(bands))
    else:
        weights = [1 << (len(bands) - 1 - i) for i in range(len(bands))]
        pick = _seed_int(seed, band_domain, sum(weights))
        cumulative = 0
        band_index = len(bands) - 1
        for i, weight in enumerate(weights):
            cumulative += weight
            if pick < cumulative:
                band_index = i
                break
    lower, upper = bands[band_index]
    return lower + _seed_int(
        seed,
        offset_domain,
        upper - lower + 1,
    )

def _miner_runs_mesh(miner) -> bool:
    """THE mesh-lane predicate: scheduling and verification routing must
    agree on it, so both call this one helper (a divergence would size a
    canary for one lane and verify it in the other)."""

    return bool(getattr(miner, "mesh_enabled", False)) or is_mesh_quant(
        str(getattr(miner, "quant", "") or "")
    )


def _record(seed: bytes, index: int, style: int) -> str:
    row = _derive(b"VERATHOS/CANARY/RECORD/V3", seed, index.to_bytes(8, "big"))
    actor = _ACTORS[int.from_bytes(row[:2], "big") % len(_ACTORS)]
    action = _ACTIONS[int.from_bytes(row[2:4], "big") % len(_ACTIONS)]
    qualifier = _QUALIFIERS[int.from_bytes(row[4:6], "big") % len(_QUALIFIERS)]
    day = 1 + row[6] % 28
    month = 1 + row[7] % 12
    value = 100 + int.from_bytes(row[8:12], "big") % 90_000
    ratio = 10 + int.from_bytes(row[12:14], "big") % 981
    ref = row[14:20].hex().upper()
    if style == 0:
        return (
            f"On {2024 + row[20] % 3:04d}-{month:02d}-{day:02d}, {actor} "
            f"{action} {qualifier}. Reference {ref} records {value} units "
            f"and a normalized rate of {ratio / 10:.1f}. The note remains "
            "provisional until the next ordinary review cycle."
        )
    if style == 1:
        return (
            f'{{"ref":"{ref}","date":"{2024 + row[20] % 3:04d}-{month:02d}-{day:02d}",'
            f'"owner":"{actor}","value":{value},"rate":{ratio / 10:.1f},'
            f'"status":"{action}","condition":"{qualifier}"}}'
        )
    if style == 2:
        return (
            f"{ref},{2024 + row[20] % 3:04d}-{month:02d}-{day:02d},"
            f"{actor},{value},{ratio / 10:.1f},{action},{qualifier}"
        )
    if style == 3:
        return (
            f"- **{ref}** — {actor} {action}; value `{value}`, rate "
            f"`{ratio / 10:.1f}`; condition: {qualifier}."
        )
    if style == 4:
        return (
            f"{2024 + row[20] % 3:04d}-{month:02d}-{day:02d} "
            f"ref={ref} owner={actor!r} value={value} "
            f"rate={ratio / 10:.1f} event={action!r} note={qualifier!r}"
        )
    if style == 5:
        return (
            f"{ref}:\n  date: "
            f"{2024 + row[20] % 3:04d}-{month:02d}-{day:02d}\n"
            f"  owner: {actor}\n  value: {value}\n"
            f"  rate: {ratio / 10:.1f}\n  status: {action}\n"
            f"  condition: {qualifier}"
        )
    if style == 6:
        return (
            f"[{ref}] Question: What was recorded for {actor} on "
            f"{2024 + row[20] % 3:04d}-{month:02d}-{day:02d}?\n"
            f"Answer: The entry says it {action} {qualifier}; measured "
            f"value {value}, normalized rate {ratio / 10:.1f}."
        )
    return (
        f"record_{ref.lower()} = {{'date': "
        f"'{2024 + row[20] % 3:04d}-{month:02d}-{day:02d}', "
        f"'owner': {actor!r}, 'value': {value}, "
        f"'rate': {ratio / 10:.1f}, 'status': {action!r}, "
        f"'condition': {qualifier!r}}}"
    )


def render_secret_seeded_prompt(
    seed: bytes,
    *,
    character_budget: int,
    full_context: bool,
    minimum_output_tokens: int = 0,
) -> str:
    """Render varied natural/structured material from validator-secret entropy."""

    if not isinstance(seed, bytes) or len(seed) != 32:
        raise ValueError("canary prompt seed must be exactly 32 bytes")
    if character_budget < 256:
        character_budget = 256
    if (
        isinstance(minimum_output_tokens, bool)
        or not isinstance(minimum_output_tokens, int)
        or minimum_output_tokens < 0
    ):
        raise ValueError("minimum_output_tokens must be a nonnegative integer")
    frame = _TASK_FRAMES[_seed_int(seed, b"frame", len(_TASK_FRAMES))]
    domain = _DOMAINS[_seed_int(seed, b"domain", len(_DOMAINS))]
    style = _seed_int(seed, b"style", 8)
    source_intro = _SOURCE_INTROS[
        _seed_int(seed, b"source_intro", len(_SOURCE_INTROS))
    ].format(domain=domain)
    grounding_rule = _GROUNDING_RULES[
        _seed_int(seed, b"grounding_rule", len(_GROUNDING_RULES))
    ]
    source_header = _SOURCE_HEADERS[
        _seed_int(seed, b"source_header", len(_SOURCE_HEADERS))
    ]
    intro = f"{frame}\n\n{source_intro} {grounding_rule}\n\n"
    if style == 1:
        intro += f"{source_header} JSON Lines\n"
    elif style == 2:
        intro += "ref,date,owner,value,rate,status,condition\n"
    elif style == 5:
        intro += f"{source_header} YAML\n"
    elif style == 7:
        intro += f"{source_header} Python-like records\n"
    else:
        intro += source_header + "\n"
    parts = [intro]
    length = len(intro)
    index = 0
    separator = "\n" if style in (1, 2, 3, 4, 7) else "\n\n"
    while length < character_budget:
        item = _record(seed, index, style)
        parts.append(item)
        parts.append(separator)
        length += len(item) + len(separator)
        index += 1
    endings = _FULL_CLOSINGS if full_context else _LOW_CLOSINGS
    closing = "\n" + endings[_seed_int(seed, b"closing", len(endings))]
    if minimum_output_tokens:
        directive = _LONG_OUTPUT_DIRECTIVES[
            _seed_int(seed, b"long_output_directive", len(_LONG_OUTPUT_DIRECTIVES))
        ].format(minimum_output_tokens=minimum_output_tokens)
        closing += "\n\n" + directive
    parts.append(closing)
    return "".join(parts)


def _refine_prompt_under_token_target(
    prompt: str,
    count: int,
    *,
    seed: bytes,
    target: int,
    token_counter: Callable[[str], int],
) -> tuple[str, int]:
    """Fill a final whole-record gap with one bounded natural text prefix."""

    source = " ".join(
        _record(seed, (1 << 20) + index, 6)
        for index in range(8)
    )
    words = tuple(source.split())
    if not words:
        return prompt, count
    prefix = "\n\nAdditional source detail: "

    def candidate(word_count: int) -> tuple[str, int]:
        value = prompt + prefix + " ".join(words[:word_count]) + "."
        return value, int(token_counter(value))

    best = (prompt, count)
    low = 1
    high = len(words)
    boundary = 0
    while low <= high:
        middle = (low + high) // 2
        value, measured = candidate(middle)
        if measured <= 0:
            raise ValueError("canary tokenizer produced no prompt tokens")
        if measured <= target:
            boundary = middle
            if measured > best[1]:
                best = (value, measured)
            low = middle + 1
        else:
            high = middle - 1

    # Tokenization can move slightly non-monotonically at whitespace and
    # punctuation boundaries. Check a bounded neighborhood rather than
    # assuming the binary-search boundary is exact.
    for word_count in range(
        max(1, boundary - 8),
        min(len(words), boundary + 8) + 1,
    ):
        value, measured = candidate(word_count)
        if measured <= 0:
            raise ValueError("canary tokenizer produced no prompt tokens")
        if measured <= target and measured > best[1]:
            best = (value, measured)
    return best


def _materialize_repeat_prefix(
    *,
    seed: bytes,
    target: int,
    token_counter: Callable[[str], int],
    full_context: bool,
) -> tuple[str, int]:
    """Build one deterministic shared prefix below its signed token target."""

    if len(seed) != 32 or target <= 0:
        raise ValueError("canary repeat-prefix parameters are invalid")
    tolerance = canary_prompt_token_tolerance_v3(target)
    goal = target - max(1, tolerance // 2)
    chars = max(256, target * 4)
    best: tuple[str, int] | None = None
    for _ in range(10):
        value = render_secret_seeded_prompt(
            seed,
            character_budget=chars,
            full_context=full_context,
        )
        measured = int(token_counter(value))
        if measured <= 0:
            raise ValueError("canary tokenizer produced no prefix tokens")
        if measured <= target and (best is None or measured > best[1]):
            best = (value, measured)
        if measured <= target and target - measured <= tolerance:
            return value, measured
        chars = max(256, int(chars * (goal / measured)))
    if best is None:
        raise ValueError(
            "validator tokenizer could not construct the signed repeat prefix"
        )
    return best


def materialize_canary_prompt(
    test: CanaryTest,
    *,
    token_counter: Callable[[str], int],
) -> tuple[str, int]:
    """Build and measure a prompt near its signed tokenizer-derived target.

    The returned count is the exact validator-tokenizer result.  The target is
    an upper bound, so prompt plus the signed decode reserve never exceeds the
    advertised context.
    """

    target = int(test.target_prompt_tokens)
    if target <= 0:
        count = int(token_counter(test.prompt))
        if count <= 0:
            raise ValueError("canary tokenizer produced no prompt tokens")
        test.measured_prompt_tokens = count
        return test.prompt, count
    seed = bytes(test.prompt_seed)
    if len(seed) != 32:
        raise ValueError("canary prompt seed is unavailable")
    repeat_prefix = ""
    repeat_prefix_tokens = 0
    if test.repeat_prefix_target_tokens:
        repeat_prefix, repeat_prefix_tokens = _materialize_repeat_prefix(
            seed=bytes(test.repeat_prefix_seed),
            target=int(test.repeat_prefix_target_tokens),
            token_counter=token_counter,
            full_context=test.test_type == "full_context",
        )
        if repeat_prefix_tokens >= target:
            raise ValueError(
                "canary repeat prefix leaves no request-specific tail"
            )
    separator = "\n\n--- Continued request-specific material ---\n\n"

    def render_candidate(character_budget: int) -> str:
        unique = render_secret_seeded_prompt(
            seed,
            character_budget=character_budget,
            full_context=test.test_type == "full_context",
            minimum_output_tokens=test.minimum_observed_output_tokens,
        )
        if not repeat_prefix:
            return unique
        return repeat_prefix + separator + unique

    # Rendered records are intentionally kept whole so the canary remains
    # natural-looking.  Some tokenizers move by more than 32 tokens at a
    # record boundary, especially in the low-context bands.
    tolerance = canary_prompt_token_tolerance_v3(target)
    # Aim for the middle of the admitted under-target window. Rendering grows
    # in complete, natural-looking records, so targeting the upper boundary
    # directly can converge a few tokens above it even when the tokenizer
    # estimate is otherwise accurate.
    goal = target - max(1, tolerance // 2)
    chars = max(256, int((target - repeat_prefix_tokens) * 4.0))
    prompt = ""
    count = 0
    best_under: tuple[str, int] | None = None
    under_chars: int | None = None
    over_chars: int | None = None
    # Token/character ratios vary across qualified tokenizers and prompt
    # styles. A bounded proportional correction reaches the target without a
    # token-by-token Python loop at long context.
    for _ in range(8):
        prompt = render_candidate(chars)
        count = int(token_counter(prompt))
        if count <= 0:
            raise ValueError("canary tokenizer produced no prompt tokens")
        if count <= target and (
            best_under is None or count > best_under[1]
        ):
            best_under = (prompt, count)
            under_chars = chars
        elif count > target:
            over_chars = chars
        if count <= target and target - count <= tolerance:
            break
        chars = max(256, int(chars * (goal / count)))

    # A small tokenizer overshoot can survive every proportional step when
    # rendering advances in whole records. Establish an under-target bracket
    # before the bounded binary search instead of failing a healthy schedule.
    if best_under is None and count > target:
        high = over_chars if over_chars is not None else chars
        low = high
        for _ in range(12):
            if low <= 256:
                break
            low = max(256, low // 2)
            candidate = render_candidate(low)
            candidate_count = int(token_counter(candidate))
            if candidate_count <= 0:
                raise ValueError("canary tokenizer produced no prompt tokens")
            if candidate_count <= target:
                best_under = (candidate, candidate_count)
                under_chars = low
                over_chars = high
                break
            high = low
            over_chars = high

    # A tokenizer can be locally discontinuous at a record boundary. If the
    # fast proportional pass straddled the target, use a bounded character
    # search and retain only measured under-target candidates.
    if (
        (count > target or target - count > tolerance)
        and under_chars is not None
        and over_chars is not None
        and under_chars < over_chars
    ):
        low = under_chars
        high = over_chars
        for _ in range(12):
            if high - low <= 1:
                break
            chars = (low + high) // 2
            candidate = render_candidate(chars)
            candidate_count = int(token_counter(candidate))
            if candidate_count <= 0:
                raise ValueError("canary tokenizer produced no prompt tokens")
            if candidate_count <= target:
                low = chars
                if (
                    best_under is None
                    or candidate_count > best_under[1]
                ):
                    best_under = (candidate, candidate_count)
            else:
                high = chars
        if best_under is not None:
            prompt, count = best_under
    elif best_under is not None and count > target:
        prompt, count = best_under
    if count <= target and target - count > tolerance:
        prompt, count = _refine_prompt_under_token_target(
            prompt,
            count,
            seed=seed,
            target=target,
            token_counter=token_counter,
        )
    if count > target or target - count > tolerance:
        raise ValueError(
            "validator tokenizer could not construct the signed canary "
            f"context target (target={target}, measured={count})"
        )
    test.prompt = prompt
    test.measured_prompt_tokens = count
    return prompt, count


@dataclass
class CanaryScheduler:
    """Plan the exact signed light and hard obligations for each endpoint."""

    epoch_number: int
    epoch_start_block: int
    epoch_blocks: int
    validator_hotkey: str = ""
    validator_seed: bytes = b""
    small_count: int = 2
    advertised_context_light_count: int = 0
    full_context_count: int = 1
    proof_sample_rate: float = 1.0
    probation_entries: Set[Tuple[str, int]] = field(default_factory=set)
    low_context_min_tokens: int = 512
    low_context_max_tokens: int = 2_048
    low_context_max_decode_tokens: int = 192
    full_context_decode_reserve_tokens: int = 256
    full_context_max_decode_tokens: int = 96
    full_context_max_attempts: int = 4
    hard_candidate_target_per_epoch: int = (
        DEFAULT_CANARY_CANDIDATE_TARGET_PER_EPOCH_V3
    )
    hard_candidate_bps: int = DEFAULT_CANARY_CANDIDATE_HARD_BPS_V3
    advertised_context_target_bps: int = 9_000
    # Mesh lane only (vLLM full-context sizing is untouched): min-heavy
    # full-context draws between a floor and the advertised context, plus
    # the signed Bernoulli exact-maximum tail. See
    # _mesh_full_prompt_target_v3 for why the mesh cannot exercise the
    # maximum every epoch.
    # 5% of the advertised context (was 10%): with geometric band
    # weighting the floor anchors the smallest, most-likely band, so a
    # lower floor directly lowers the median full-canary cost; the
    # low-context clamp in _mesh_full_prompt_floor still keeps every
    # full draw above the small-canary band.
    mesh_full_min_prompt_bps: int = 500
    mesh_full_max_draw_bps: int = 500
    owner_full_min_prompt_bps: int = (
        DEFAULT_CANARY_OWNER_FULL_MIN_PROMPT_BPS_V3
    )
    owner_full_max_draw_bps: int = (
        DEFAULT_CANARY_OWNER_FULL_MAX_DRAW_BPS_V3
    )
    owner_context_sizing_abi_id: str = CANARY_OWNER_CONTEXT_SIZING_ABI_V3
    hard_decode_anchor_bps: int = MIN_CANARY_HARD_DECODE_ANCHOR_BPS_V3
    hard_decode_tail_bps: int = MIN_CANARY_HARD_DECODE_TAIL_BPS_V3
    late_decode_min_output_bps: int = 9_000
    repeat_prefix_target_bps: int = (
        DEFAULT_CANARY_REPEAT_PREFIX_TARGET_BPS_V3
    )
    repeat_prefix_min_tokens: int = (
        DEFAULT_CANARY_REPEAT_PREFIX_MIN_TOKENS_V3
    )
    hard_audit_enabled: bool = False
    hard_audit_due_entries: Set[Tuple[str, int]] = field(
        default_factory=set,
    )
    hard_context_limits_by_model: Dict[str, int] = field(default_factory=dict)
    hard_decode_limits_by_model: Dict[str, int] = field(default_factory=dict)
    low_completion_reserve_blocks: int = 1
    full_completion_reserve_blocks: int = 1
    tests: List[CanaryTest] = field(default_factory=list)
    _epoch_salt: bytes = field(default_factory=lambda: os.urandom(32), repr=False)

    def _test_seed(self, miner, test_index: int) -> bytes:
        return _derive(
            b"VERATHOS/CANARY/SECRET/V3",
            self.validator_seed,
            self._epoch_salt,
            self.epoch_number.to_bytes(8, "big"),
            self.validator_hotkey.encode(),
            str(miner.address).lower().encode(),
            str(miner.model_id).encode(),
            int(miner.model_index).to_bytes(4, "big"),
            str(miner.endpoint).encode(),
            test_index.to_bytes(4, "big"),
        )

    def _obligation_id(self, seed: bytes) -> str:
        return _derive(b"VERATHOS/CANARY/OBLIGATION/V3", seed)[:16].hex()

    def _repeat_prefix_seed(self, miner, *, group: bytes) -> bytes:
        return _derive(
            b"VERATHOS/CANARY/REPEAT_PREFIX/V3",
            self.validator_seed,
            self._epoch_salt,
            self.epoch_number.to_bytes(8, "big"),
            self.validator_hotkey.encode(),
            str(miner.address).lower().encode(),
            str(miner.model_id).encode(),
            int(miner.model_index).to_bytes(4, "big"),
            str(miner.endpoint).encode(),
            group,
        )

    def _repeat_prefix_target(self, minimum_prompt_tokens: int) -> int:
        minimum = int(minimum_prompt_tokens)
        target = max(
            int(self.repeat_prefix_min_tokens),
            minimum * int(self.repeat_prefix_target_bps) // 10_000,
        )
        # Preserve a substantial request-specific suffix even under a custom
        # signed policy and tokenizer tolerance.
        return min(target, max(1, minimum // 2))

    def _mesh_full_prompt_floor(self, max_target: int) -> int:
        """Smallest full-context size the mesh lane may draw.

        Scales with the claim (mesh_full_min_prompt_bps of the maximum)
        but always clears the low-context band, so a full-context draw
        can never be confused with - or be cheaper than - a low canary.
        """

        return min(
            int(max_target),
            max(
                int(self.low_context_max_tokens) + 1,
                int(max_target)
                * int(self.mesh_full_min_prompt_bps)
                // 10_000,
            ),
        )

    def _safe_context_target(
        self,
        context_limit: int,
        *,
        decode_reserve_tokens: int | None = None,
    ) -> int:
        limit = max(1, int(context_limit))
        fraction_target = max(
            1,
            limit * int(self.advertised_context_target_bps) // 10_000,
        )
        reserve_target = max(
            1,
            limit
            - max(
                int(self.full_context_decode_reserve_tokens),
                int(decode_reserve_tokens or 0),
            ),
        )
        return min(fraction_target, reserve_target)

    def _owner_full_prompt_floor(self, max_target: int) -> int:
        """Return the signed floor, kept above the low-context lane."""

        return min(
            int(max_target),
            max(
                int(self.low_context_max_tokens) + 1,
                int(max_target)
                * int(self.owner_full_min_prompt_bps)
                // 10_000,
            ),
        )

    def plan_epoch(self, miners: list) -> List[CanaryTest]:
        self.tests = []
        for miner in miners:
            test_index = 0
            low_repeat_seed = self._repeat_prefix_seed(
                miner,
                group=b"low",
            )
            low_repeat_target = self._repeat_prefix_target(
                self.low_context_min_tokens
            )
            is_tee = bool(getattr(miner, "tee_enabled", False))
            versions = tuple(
                getattr(miner, "proof_protocol_versions", ()) or ()
            )
            quant = str(getattr(miner, "quant", "") or "")
            if self.hard_audit_enabled and self.small_count:
                raise ValueError(
                    "hard auditor low/full candidates must use one marked "
                    "schedule rather than a recognizable fixed low inventory"
                )
            low_span = (
                self.low_context_max_tokens - self.low_context_min_tokens + 1
            )
            for _ in range(self.small_count):
                seed = self._test_seed(miner, test_index)
                target = self.low_context_min_tokens + _seed_int(
                    seed,
                    b"low_tokens",
                    low_span,
                )
                max_new = max(
                    32,
                    min(
                        self.low_context_max_decode_tokens,
                        64 + _seed_int(seed, b"low_decode", 129),
                    ),
                )
                prompt = render_secret_seeded_prompt(
                    seed,
                    character_budget=min(2_000, max(512, target * 2)),
                    full_context=False,
                )
                self.tests.append(
                    CanaryTest(
                        miner_address=miner.address,
                        miner_endpoint=miner.endpoint,
                        model_id=miner.model_id,
                        model_index=miner.model_index,
                        max_context_len=miner.max_context_len,
                        target_block=self._target_block(
                            miner,
                            test_index,
                            completion_reserve_blocks=(
                                self.low_completion_reserve_blocks
                            ),
                        ),
                        test_index=test_index,
                        test_type="small",
                        prompt=prompt,
                        max_new_tokens=max_new,
                        temperature=0.0,
                        verify_proof=False,
                        verify_tee=is_tee,
                        enable_thinking=False,
                        presence_penalty=0.0,
                        top_k=0,
                        top_p=1.0,
                        proof_protocol_versions=versions,
                        quant=quant,
                        obligation_id=self._obligation_id(seed),
                        target_prompt_tokens=target,
                        prompt_seed=seed,
                        repeat_prefix_seed=low_repeat_seed,
                        repeat_prefix_target_tokens=low_repeat_target,
                    )
                )
                test_index += 1
            if self.hard_audit_enabled:
                if (
                    self.advertised_context_light_count != 1
                    or self.full_context_count != 0
                ):
                    raise ValueError(
                        "hard auditor requires one advertised-context light "
                        "plus its marked qualified-context schedule"
                    )
                if not 0 < int(self.late_decode_min_output_bps) <= 10_000:
                    raise ValueError(
                        "hard auditor late-decode output rate is invalid"
                    )
                candidate_target = int(
                    self.hard_candidate_target_per_epoch
                )
                candidate_hard_bps = int(self.hard_candidate_bps)
                if not 0 < candidate_target <= int(self.epoch_blocks):
                    raise ValueError(
                        "hard auditor candidate target is invalid"
                    )
                if not 0 < candidate_hard_bps < 10_000:
                    raise ValueError(
                        "hard auditor candidate hard rate must leave light "
                        "decoys"
                    )
                model_id = str(miner.model_id)
                miner_key = (
                    str(miner.address).lower(),
                    int(miner.model_index),
                )
                hard_audit_due = (
                    miner_key in self.hard_audit_due_entries
                    or miner_key in self.probation_entries
                )
                qualified_context = int(
                    self.hard_context_limits_by_model.get(model_id, 0) or 0
                )
                signed_decode_cap = int(
                    self.hard_decode_limits_by_model.get(model_id, 0) or 0
                )
                if qualified_context <= 1 or signed_decode_cap <= 0:
                    # A compatibility-window endpoint may legitimately expose
                    # only v1 for a model that has no qualified v3 release.
                    # Do not let that one endpoint abort the owner's complete
                    # epoch schedule.  Its requests remain bounded by the
                    # signed global canary policy and protocol negotiation
                    # still selects v1 (or fails closed when v1 is no longer
                    # owner-allowed).  Any endpoint that negotiates v3 is
                    # checked against the model-specific signed policy before
                    # the request is sent.
                    qualified_context = max(
                        2,
                        int(getattr(miner, "max_context_len", 0) or 0),
                    )
                    signed_decode_cap = max(
                        1,
                        min(
                            int(self.low_context_max_decode_tokens),
                            qualified_context - 1,
                        ),
                    )
                context_limit = min(
                    int(miner.max_context_len),
                    qualified_context,
                )
                decode_cap = min(
                    signed_decode_cap,
                    context_limit - 1,
                )
                if decode_cap <= 0:
                    raise ValueError(
                        "hard auditor model limits leave no prompt capacity"
                    )
                full_repeat_seed = self._repeat_prefix_seed(
                    miner,
                    group=b"marked_full",
                )

                # The advertised-capacity light request uses the same signed
                # min-heavy sizing rule as marked candidates. Its ceiling is
                # the endpoint's safe advertised target, while cryptographic
                # coverage remains bounded by the signed hard-audit reach.
                seed = self._test_seed(miner, test_index)
                advertised_decode = int(self.full_context_max_decode_tokens)
                advertised_max = self._safe_context_target(
                    int(miner.max_context_len),
                    decode_reserve_tokens=advertised_decode,
                )
                marked_full_max = self._safe_context_target(
                    context_limit,
                    decode_reserve_tokens=decode_cap,
                )
                miner_is_mesh = _miner_runs_mesh(miner)
                sampling_enabled = int(self.owner_full_max_draw_bps) < 10_000
                advertised_target = advertised_max
                full_repeat_basis = marked_full_max
                if miner_is_mesh:
                    # Mesh lane: min-heavy draw between the mesh floor and
                    # the advertised context with GEOMETRIC band weighting
                    # (llama.cpp prefill is minutes-long at advertised
                    # maxima). Exact maximum stays
                    # reachable through the signed Bernoulli tail.
                    advertised_target = _mesh_full_prompt_target_v3(
                        seed,
                        minimum=self._mesh_full_prompt_floor(advertised_max),
                        maximum=advertised_max,
                        max_draw_bps=self.mesh_full_max_draw_bps,
                    )
                    # The shared repeat prefix must fit the SMALLEST size
                    # any full-context draw can produce.
                    full_repeat_basis = min(
                        self._mesh_full_prompt_floor(marked_full_max),
                        self._mesh_full_prompt_floor(advertised_max),
                    )
                elif sampling_enabled:
                    advertised_target = _owner_full_prompt_target_v3(
                        seed,
                        minimum=self._owner_full_prompt_floor(advertised_max),
                        maximum=advertised_max,
                        max_draw_bps=self.owner_full_max_draw_bps,
                        sizing_abi_id=self.owner_context_sizing_abi_id,
                    )
                    full_repeat_basis = min(
                        self._owner_full_prompt_floor(marked_full_max),
                        self._owner_full_prompt_floor(advertised_max),
                    )
                full_repeat_target = self._repeat_prefix_target(
                    full_repeat_basis
                )
                self.tests.append(
                    CanaryTest(
                        miner_address=miner.address,
                        miner_endpoint=miner.endpoint,
                        model_id=miner.model_id,
                        model_index=miner.model_index,
                        max_context_len=miner.max_context_len,
                        target_block=self._target_block(
                            miner,
                            test_index,
                            completion_reserve_blocks=(
                                self.full_completion_reserve_blocks
                            ),
                        ),
                        test_index=test_index,
                        test_type="full_context",
                        prompt=render_secret_seeded_prompt(
                            seed,
                            character_budget=2_000,
                            full_context=True,
                        ),
                        max_new_tokens=advertised_decode,
                        temperature=0.0,
                        verify_proof=False,
                        verify_tee=is_tee,
                        enable_thinking=False,
                        presence_penalty=0.0,
                        top_k=0,
                        top_p=1.0,
                        proof_protocol_versions=versions,
                        quant=quant,
                        obligation_id=self._obligation_id(seed),
                        target_prompt_tokens=advertised_target,
                        prompt_seed=seed,
                        repeat_prefix_seed=full_repeat_seed,
                        repeat_prefix_target_tokens=full_repeat_target,
                        max_attempts=self.full_context_max_attempts,
                    )
                )
                test_index += 1

                # A secret Bernoulli draw at every block produces a binomial
                # number of candidates with the signed expectation. Each
                # candidate receives an independent hidden hard/light mark.
                # The miner therefore never learns a public-epoch countdown.
                schedule_basis = (
                    self.validator_seed,
                    self._epoch_salt,
                    self.epoch_number.to_bytes(8, "big"),
                    self.validator_hotkey.encode(),
                    str(miner.address).lower().encode(),
                    str(miner.model_id).encode(),
                    int(miner.model_index).to_bytes(4, "big"),
                    str(miner.endpoint).encode(),
                )
                candidate_events: list[tuple[int, bytes]] = []
                for block_offset in range(int(self.epoch_blocks)):
                    event_seed = _derive(
                        b"VERATHOS/CANARY/MARKED_EVENT/V3",
                        *schedule_basis,
                        block_offset.to_bytes(8, "big"),
                    )
                    if (
                        _seed_int(
                            event_seed,
                            b"candidate_event",
                            int(self.epoch_blocks),
                        )
                        >= candidate_target
                    ):
                        continue
                    candidate_events.append((block_offset, event_seed))

                # The independent process is never capped after a pass, so a
                # miner cannot infer a safe tail. A bounded rolling hard-audit
                # drought nevertheless needs a floor: when the validator's
                # authenticated pass ledger says this endpoint is due, ensure
                # that one secret candidate exists and force exactly one of
                # the otherwise ordinary candidates to hard after precommit.
                if hard_audit_due and not candidate_events:
                    forced_seed = _derive(
                        b"VERATHOS/CANARY/MARKED_DROUGHT_EVENT/V3",
                        *schedule_basis,
                    )
                    forced_offset = _seed_int(
                        forced_seed,
                        b"forced_candidate_block",
                        int(self.epoch_blocks),
                    )
                    candidate_events.append((forced_offset, forced_seed))
                forced_hard_offset = None
                if hard_audit_due:
                    selector = _derive(
                        b"VERATHOS/CANARY/MARKED_DROUGHT_SELECT/V3",
                        *schedule_basis,
                    )
                    forced_hard_offset = candidate_events[
                        _seed_int(
                            selector,
                            b"forced_candidate_index",
                            len(candidate_events),
                        )
                    ][0]

                for block_offset, event_seed in sorted(candidate_events):
                    seed = _derive(
                        b"VERATHOS/CANARY/MARKED_PROMPT/V3",
                        event_seed,
                    )
                    category = _independent_full_decode_category_v3(
                        seed,
                        anchor_bps=self.hard_decode_anchor_bps,
                        tail_bps=self.hard_decode_tail_bps,
                    )
                    max_new = _signed_full_decode_tokens_v3(
                        seed,
                        category=category,
                        max_decode_tokens=decode_cap,
                        common_max_decode_tokens=(
                            self.low_context_max_decode_tokens
                        ),
                    )
                    target = self._safe_context_target(
                        context_limit,
                        decode_reserve_tokens=max_new,
                    )
                    if miner_is_mesh:
                        # Every marked candidate draws independently (its
                        # own seed); the advertised light above and each
                        # candidate can therefore land on different sizes.
                        target = _mesh_full_prompt_target_v3(
                            seed,
                            minimum=self._mesh_full_prompt_floor(target),
                            maximum=target,
                            max_draw_bps=self.mesh_full_max_draw_bps,
                        )
                    elif sampling_enabled:
                        target = _owner_full_prompt_target_v3(
                            seed,
                            minimum=self._owner_full_prompt_floor(target),
                            maximum=target,
                            max_draw_bps=self.owner_full_max_draw_bps,
                            sizing_abi_id=self.owner_context_sizing_abi_id,
                        )
                    late_decode = (
                        max_new > self.low_context_max_decode_tokens
                    )
                    minimum_output = 0
                    if late_decode:
                        minimum_output = max(
                            1,
                            (
                                max_new * int(self.late_decode_min_output_bps)
                                + 9_999
                            )
                            // 10_000,
                        )
                    self.tests.append(
                        CanaryTest(
                            miner_address=miner.address,
                            miner_endpoint=miner.endpoint,
                            model_id=miner.model_id,
                            model_index=miner.model_index,
                            max_context_len=miner.max_context_len,
                            target_block=(
                                self.epoch_start_block + block_offset
                            ),
                            test_index=test_index,
                            test_type="full_context",
                            prompt=render_secret_seeded_prompt(
                                seed,
                                character_budget=2_000,
                                full_context=True,
                                minimum_output_tokens=minimum_output,
                            ),
                            max_new_tokens=max_new,
                            temperature=0.0,
                            verify_proof=(
                                block_offset == forced_hard_offset
                                or (
                                    _seed_int(
                                        event_seed,
                                        b"candidate_hard_mark",
                                        10_000,
                                    )
                                    < candidate_hard_bps
                                )
                            ),
                            verify_tee=is_tee,
                            enable_thinking=False,
                            presence_penalty=0.0,
                            top_k=0,
                            top_p=1.0,
                            proof_protocol_versions=versions,
                            quant=quant,
                            obligation_id=self._obligation_id(seed),
                            target_prompt_tokens=target,
                            prompt_seed=seed,
                            repeat_prefix_seed=full_repeat_seed,
                            repeat_prefix_target_tokens=full_repeat_target,
                            minimum_observed_output_tokens=minimum_output,
                            max_attempts=self.full_context_max_attempts,
                        )
                    )
                    test_index += 1
            else:
                # Compatibility/light-only planners retain their fixed bounded
                # full-context requests. The active hard auditor uses the
                # validator-secret marked construction above.
                light_full_count = (
                    self.advertised_context_light_count
                    + self.full_context_count
                )
                for _ in range(light_full_count):
                    seed = self._test_seed(miner, test_index)
                    max_new = int(self.full_context_max_decode_tokens)
                    target = self._safe_context_target(
                        miner.max_context_len,
                        decode_reserve_tokens=max_new,
                    )
                    prompt = render_secret_seeded_prompt(
                        seed,
                        character_budget=2_000,
                        full_context=True,
                    )
                    self.tests.append(
                        CanaryTest(
                            miner_address=miner.address,
                            miner_endpoint=miner.endpoint,
                            model_id=miner.model_id,
                            model_index=miner.model_index,
                            max_context_len=miner.max_context_len,
                            target_block=self._target_block(
                                miner,
                                test_index,
                                completion_reserve_blocks=(
                                    self.full_completion_reserve_blocks
                                ),
                            ),
                            test_index=test_index,
                            test_type="full_context",
                            prompt=prompt,
                            max_new_tokens=max_new,
                            temperature=0.0,
                            verify_proof=False,
                            verify_tee=is_tee,
                            enable_thinking=False,
                            presence_penalty=0.0,
                            top_k=0,
                            top_p=1.0,
                            proof_protocol_versions=versions,
                            quant=quant,
                            obligation_id=self._obligation_id(seed),
                            target_prompt_tokens=target,
                            prompt_seed=seed,
                            repeat_prefix_seed=low_repeat_seed,
                            repeat_prefix_target_tokens=low_repeat_target,
                            max_attempts=self.full_context_max_attempts,
                        )
                    )
                    test_index += 1
        # Mark tests whose endpoint runs the GGUF-mesh runtime.  This does
        # not alter scheduling in any way; it only routes the execution of
        # these obligations through the mesh verification lane.
        mesh_keys = {
            (str(miner.address).lower(), int(miner.model_index))
            for miner in miners
            # Same predicate the SIZING used above: a divergence would
            # size a canary for one lane and verify it in the other.
            if _miner_runs_mesh(miner)
        }
        if mesh_keys:
            for test in self.tests:
                if (
                    str(test.miner_address).lower(),
                    int(test.model_index),
                ) in mesh_keys:
                    test.verify_mesh = True
        self.tests.sort(key=lambda item: (item.target_block, item.miner_address))
        return self.tests

    def get_pending_tests(self, current_block: int) -> List[CanaryTest]:
        pending = [item for item in self.tests if item.target_block <= current_block]
        self.tests = [item for item in self.tests if item.target_block > current_block]
        return pending

    def reschedule(
        self,
        test: CanaryTest,
        *,
        current_block: int,
    ) -> bool:
        """Place a fresh precommit-busy attempt later in the same epoch.

        The opaque obligation remains the same, but the request seed and
        prompt change. Repeating the miner-visible prompt would identify the
        retry as validator traffic and weaken canary indistinguishability.
        """

        reserve_blocks = (
            self.full_completion_reserve_blocks
            if test.test_type == "full_context"
            else self.low_completion_reserve_blocks
        )
        last_block = self._latest_dispatch_block(
            completion_reserve_blocks=reserve_blocks,
        )
        first = int(current_block) + 1
        if first > last_block:
            return False
        seed = _derive(
            b"VERATHOS/CANARY/RESCHEDULE/V3",
            self._epoch_salt,
            (
                test.full_pair_id
                if test.full_pair_id
                else test.obligation_id
            ).encode(),
            test.attempt_count.to_bytes(4, "big"),
        )
        test.prompt_seed = _derive(
            b"VERATHOS/CANARY/RETRY_PROMPT/V3",
            test.prompt_seed,
            seed,
        )
        test.prompt = render_secret_seeded_prompt(
            test.prompt_seed,
            character_budget=2_000,
            full_context=test.test_type == "full_context",
            minimum_output_tokens=test.minimum_observed_output_tokens,
        )
        test.measured_prompt_tokens = 0
        test.target_block = first + (
            int.from_bytes(seed[:8], "big") % (last_block - first + 1)
        )
        self.tests.append(test)
        self.tests.sort(key=lambda item: (item.target_block, item.miner_address))
        return True

    def _latest_dispatch_block(
        self,
        *,
        completion_reserve_blocks: int,
    ) -> int:
        reserve = int(completion_reserve_blocks)
        epoch_blocks = int(self.epoch_blocks)
        if reserve < 1:
            raise ValueError("canary completion reserve must be positive")
        if reserve >= epoch_blocks:
            raise ValueError(
                "canary completion reserve does not fit inside the epoch"
            )
        return self.epoch_start_block + epoch_blocks - reserve

    def _target_block(
        self,
        miner,
        test_index: int,
        *,
        completion_reserve_blocks: int,
    ) -> int:
        seed = _derive(
            b"VERATHOS/CANARY/SCHEDULE/V3",
            self.epoch_number.to_bytes(8, "big"),
            self.validator_hotkey.encode(),
            str(miner.address).lower().encode(),
            str(miner.model_id).encode(),
            int(miner.model_index).to_bytes(4, "big"),
            str(miner.endpoint).encode(),
            test_index.to_bytes(4, "big"),
            self._epoch_salt,
        )
        latest = self._latest_dispatch_block(
            completion_reserve_blocks=completion_reserve_blocks,
        )
        usable = latest - self.epoch_start_block + 1
        return (
            self.epoch_start_block
            + int.from_bytes(seed[:8], "big") % usable
        )


def generate_small_canary_prompt(
    epoch_number: int,
    miner_address: str,
    test_index: int,
    validator_seed: bytes,
) -> tuple[str, int, float]:
    """Compatibility helper for a deterministic secret-seeded low prompt."""

    seed = _derive(
        b"VERATHOS/CANARY/SMALL/V3",
        validator_seed,
        epoch_number.to_bytes(8, "big"),
        miner_address.encode(),
        test_index.to_bytes(4, "big"),
    )
    return (
        render_secret_seeded_prompt(
            seed,
            character_budget=1_200,
            full_context=False,
        ),
        96 + _seed_int(seed, b"decode", 97),
        0.0,
    )


def generate_full_context_canary_prompt(
    epoch_number: int,
    miner_address: str,
    test_index: int,
    max_context_len: int,
    validator_seed: bytes,
) -> tuple[str, int, float]:
    """Compatibility helper; exact long materialization occurs at dispatch."""

    seed = _derive(
        b"VERATHOS/CANARY/FULL/V3",
        validator_seed,
        epoch_number.to_bytes(8, "big"),
        miner_address.encode(),
        test_index.to_bytes(4, "big"),
        int(max_context_len).to_bytes(8, "big"),
    )
    return (
        render_secret_seeded_prompt(
            seed,
            character_budget=2_000,
            full_context=True,
        ),
        96,
        0.0,
    )


__all__ = [
    "CanaryScheduler",
    "CanaryTest",
    "generate_full_context_canary_prompt",
    "generate_small_canary_prompt",
    "materialize_canary_prompt",
    "render_secret_seeded_prompt",
]
