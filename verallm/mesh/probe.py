"""Pre-registration probe gate for GGUF mesh deployments.

Measures a running mesh over the pool's operator chat lane (the driver runs
each probe against its own coordinator on loopback with the mesh's pinned
verification snapshot; the manager applies its validator-mode proof
assertion to every result) and checks the public endpoint out of band.

What a passing gate proves: the coordinator -> llama-server -> RPC stages ->
proof adapter -> receipt aggregation path produces responses whose GGML GEMM
proof verifies across every declared compute stage, bound to the mesh's
signed snapshot, at the measured latency and throughput, and the public
endpoint is reachable in the validator-auth posture.

What it does NOT prove, stated so nobody over-reads a pass:
- Probes are not authenticated-validator requests, so the postcommit
  two-phase path and the artifact-privacy scrub are not exercised.
- Probes run over loopback: probe TTFT excludes the WAN hop and TLS, so it
  is a LOWER BOUND on validator-observed TTFT.
- Reachability shows the endpoint answers in validator-auth posture, not
  that any particular validator's key is allowlisted.

Threshold provenance is annotated per check: protocol-derived checks carry
no tunables; judgement calls are flags with defaults explained in
rationale strings.
"""
from __future__ import annotations

import json
import random
import socket
import ssl
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from statistics import median
from typing import Any, Callable, Mapping, Sequence

# The validator's full-context canary prompt fills 0.8 * the ADVERTISED
# context (neurons/canary.py) with 200 output tokens and a 900 s inference
# budget (neurons/config.py). The probe replays that shape: a mesh must
# actually serve the context it registers. Only HARD-tier canaries are
# context-capped (calibrated proof corridors); light canaries and organic
# traffic run and score at the advertised value.
FULL_CONTEXT_FILL_RATIO = 0.8
FULL_CONTEXT_OUTPUT_TOKENS = 200
VALIDATOR_FULL_CONTEXT_BUDGET_S = 900.0
# Default gate budget: 0.6 of the validator ceiling, because the probe skips
# TLS and WAN while epoch canaries contend for the single serving slot.
DEFAULT_FULL_CONTEXT_BUDGET_S = 540.0
# Survival floor: small canaries generate up to 300 tokens against a 300 s
# timeout, so 1.0 tok/s is where canaries start timing out and the miner
# stops acking the 5-receipt scoring minimum. Default gate = 3x margin.
DEFAULT_MIN_TOK_S = 3.0
DEFAULT_PROBE_SAMPLES = 3
DEFAULT_TTFT_ADVISORY_S = 10.0


@dataclass(frozen=True)
class ProbeSample:
    ok: bool
    error: str = ""
    ttft_s: float = -1.0
    total_s: float = -1.0
    pickup_s: float = -1.0
    completion_tokens: int = 0
    prompt_tokens: int = 0
    engine_tps: float = 0.0
    prompt_tps: float = 0.0
    verified: bool = False
    receipt_verified: bool = False
    receipts: int = 0
    proof_stages: int = 0
    expected_stage_count: int = 0
    proof_mode: str = ""
    proof_receipt_root: str = ""
    mesh_response_commitment_hash: str = ""
    verification_snapshot_hash: str = ""

    @property
    def wall_tok_s(self) -> float:
        """completion_tokens / total wall seconds: the validator-equivalent
        throughput metric (validator TPS divides by phase-one wall clock,
        not decode time)."""
        if self.total_s <= 0 or self.completion_tokens <= 0:
            return 0.0
        return self.completion_tokens / self.total_s


def probe_sample_from_chat_result(result: Mapping[str, Any]) -> ProbeSample:
    usage = result.get("usage") or {}
    return ProbeSample(
        ok=str(result.get("status", "")) == "ok" and not result.get("error"),
        error=str(result.get("error", "") or ""),
        ttft_s=float(result.get("ttft_s", -1.0) or -1.0),
        total_s=float(result.get("total_s", -1.0) or -1.0),
        pickup_s=float(result.get("pickup_s", -1.0) or -1.0),
        completion_tokens=int(usage.get("completion_tokens", 0) or 0),
        prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
        engine_tps=float(result.get("engine_tps", 0.0) or 0.0),
        prompt_tps=float(result.get("prompt_tps", 0.0) or 0.0),
        verified=bool(result.get("verified")),
        receipt_verified=bool(result.get("receipt_verified")),
        receipts=int(result.get("receipts", 0) or 0),
        proof_stages=int(result.get("proof_stages", 0) or 0),
        expected_stage_count=int(result.get("expected_stage_count", 0) or 0),
        proof_mode=str(result.get("proof_mode", "") or ""),
        proof_receipt_root=str(result.get("proof_receipt_root", "") or ""),
        mesh_response_commitment_hash=str(
            result.get("mesh_response_commitment_hash", "") or ""
        ),
        verification_snapshot_hash=str(
            result.get("verification_snapshot_hash", "") or ""
        ),
    )


@dataclass(frozen=True)
class GateCheck:
    name: str
    kind: str  # "hard" | "advisory"
    passed: bool
    observed: str
    threshold: str
    rationale: str
    remediation: str = ""


@dataclass
class GateReport:
    samples: list[ProbeSample] = field(default_factory=list)
    hard_samples: list[ProbeSample] = field(default_factory=list)
    full_context: ProbeSample | None = None
    reachability: list[GateCheck] = field(default_factory=list)
    checks: list[GateCheck] = field(default_factory=list)

    @property
    def all_checks(self) -> list[GateCheck]:
        return [*self.checks, *self.reachability]

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.all_checks if check.kind == "hard")

    def render(self) -> str:
        lines = []
        for check in self.all_checks:
            marker = "PASS" if check.passed else (
                "FAIL" if check.kind == "hard" else "note"
            )
            lines.append(
                f"[{marker}] {check.name}: {check.observed}"
                + (f" (threshold {check.threshold})" if check.threshold else "")
            )
            if not check.passed:
                lines.append(f"       why: {check.rationale}")
                if check.remediation:
                    lines.append(f"       fix: {check.remediation}")
        return "\n".join(lines)


@dataclass(frozen=True)
class ProbeGateConfig:
    samples: int = DEFAULT_PROBE_SAMPLES
    # Validators draw both proof tiers: organic traffic and most canaries
    # verify the LIGHT relation, while audit draws force the HARD GEMM
    # sumcheck. A mesh is ready to register only when both verify, so the
    # gate always exercises at least one explicit hard-tier probe (an
    # upgrade-only request that works on every lane, chain-bound or not).
    hard_samples: int = 1
    min_tok_s: float = DEFAULT_MIN_TOK_S
    full_context: bool = True
    full_context_budget_s: float = DEFAULT_FULL_CONTEXT_BUDGET_S
    # The context length about to be registered on chain. With the measured
    # auto-fit flow this is the KV budget the launch measured, not a typed
    # guess.
    max_context_len: int = 32_768
    # What the launch's KV auto-fit measured on this hardware (0 = unknown,
    # e.g. dev pools or callers that predate measurement). When known, the
    # gate hard-fails a registration that advertises more context than the
    # mesh can actually serve.
    measured_ctx_budget: int = 0
    small_max_tokens: int = 200


# Varied filler vocabulary: short, common, single-token words on the
# approved GGUF tokenizers, so one word still approximates one token.
_PROBE_FILLER_WORDS = (
    "data field stone river light march track sound plain frame glass "
    "north point cloud grain steel house metal water south chart brick "
    "mount plate range shore table bound crane depth flint gorge"
).split()


def build_full_context_probe_prompt(max_context_len: int) -> str:
    """A canary-sized prefill prompt, unique from the first token.

    The validator fills 0.8 * the advertised context in TOKENS; without the
    model's tokenizer we approximate one token per short word, which for the
    filler below is within a few percent on the approved GGUF models. Slight
    over-fill only makes the gate stricter.

    The filler must be VARIED, not one repeated word: validator canaries
    are seeded varied records (neurons/canary.py), and tens of thousands
    of copies of a single token drive hybrid-SSM models into numeric
    collapse upstream llama.cpp does not survive. A probe measuring a
    shape validators never send gates the wrong thing.

    The nonce prefix is load-bearing: llama-server reuses the cached KV
    prefix of the previous request in a slot, so two probes sharing their
    opening tokens only evaluate the uncached suffix. A unique first token defeats all
    prefix reuse, so every probe measures a cold prefill exactly like a
    validator canary. The filler itself is seeded per probe for the same
    reason: identical bodies across probes would LCP-match past the nonce
    on requantized caches.
    """
    fill_tokens = int(max_context_len * FULL_CONTEXT_FILL_RATIO)
    nonce = uuid.uuid4().hex[:12]
    rng = random.Random(nonce)
    words = [
        _PROBE_FILLER_WORDS[rng.randrange(len(_PROBE_FILLER_WORDS))]
        for _ in range(max(0, fill_tokens))
    ]
    prompt = " ".join(words)
    return (
        f"probe {nonce}\n"
        + prompt
        + "\n\nSummarize the content above in one short sentence."
    )


def run_probe_gate(
    *,
    call: Callable[[str, dict], dict],
    mesh_key: str,
    expected_snapshot_hash: str,
    expected_stage_count: int,
    max_rtt_ms: float,
    config: ProbeGateConfig,
    log: Callable[[str], None] = lambda _line: None,
) -> GateReport:
    """Run the in-band probe series against a serving mesh.

    ``call(route, body)`` is a management-authed pool client. Reachability
    checks are separate (check_public_endpoint) because they run against
    the public URL, not the pool manager.

    Mid-gate snapshot ROTATION is legitimate: the manager's epoch follower
    re-signs every serving chain-bound mesh at each chain epoch boundary
    (~72 min mainnet, ~36 min testnet), and a full gate can span one. The
    gate therefore tracks the SET of hashes that were current during its
    run (refreshed from the manager whenever a probe reports a different
    binding), retries the one probe that straddled the re-sign, and
    accepts samples bound to any current-at-the-time hash. Before this,
    a first gate that crosses a boundary can otherwise fail snapshot binding
    and require a full rerun.
    """
    report = GateReport()
    valid_hashes: list[str] = (
        [expected_snapshot_hash] if expected_snapshot_hash else []
    )

    def _current_mesh_hash() -> str:
        try:
            status = call("/v1/pool/status", {})
            mesh = dict((status.get("meshes") or {}).get(mesh_key) or {})
            return str(mesh.get("verification_snapshot_hash", "") or "")
        except (RuntimeError, OSError):
            return ""

    def _sample_rotated(sample: ProbeSample) -> bool:
        if not valid_hashes:
            return False
        if sample.ok:
            return bool(
                sample.verification_snapshot_hash
                and sample.verification_snapshot_hash not in valid_hashes
            )
        return "snapshot" in str(sample.error or "").lower()

    def _probe_once(body: dict) -> ProbeSample:
        try:
            result = call(
                "/v1/pool/chat",
                {
                    "mesh_key": mesh_key,
                    "probe": True,
                    "stream": True,
                    "thinking": False,
                    **body,
                },
            )
        except (RuntimeError, OSError) as exc:
            return ProbeSample(ok=False, error=str(exc))
        return probe_sample_from_chat_result(result)

    def _probe(body: dict) -> ProbeSample:
        sample = _probe_once(body)
        if not _sample_rotated(sample):
            return sample
        # A different binding is only acceptable when the MANAGER confirms
        # the mesh rotated to it; anything else stays a hard failure.
        current = _current_mesh_hash()
        if not current or current in valid_hashes:
            return sample
        valid_hashes.append(current)
        log(
            "snapshot rotated mid-gate (epoch boundary); accepting "
            f"{current[:12]} and retrying the straddled probe"
        )
        retried = _probe_once(body)
        return retried

    for index in range(max(1, config.samples)):
        log(f"probe {index + 1}/{config.samples} (small)")
        sample = _probe(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Briefly describe what a Merkle tree is used for."
                        ),
                    }
                ],
                "max_tokens": config.small_max_tokens,
                "timeout": VALIDATOR_FULL_CONTEXT_BUDGET_S,
            }
        )
        report.samples.append(sample)

    for index in range(max(0, config.hard_samples)):
        log(f"probe {index + 1}/{config.hard_samples} (hard tier)")
        sample = _probe(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Name two properties a cryptographic hash "
                            "function must have."
                        ),
                    }
                ],
                "max_tokens": config.small_max_tokens,
                # Upgrade-only tier request: forces the HARD GEMM sumcheck
                # relation on any lane, exactly what a validator audit draw
                # asserts. It can never downgrade an organic light serve.
                "proof_tier": "hard",
                "timeout": VALIDATOR_FULL_CONTEXT_BUDGET_S,
            }
        )
        report.hard_samples.append(sample)

    if config.full_context:
        log("probe full-context (canary-shaped)")
        report.full_context = _probe(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": build_full_context_probe_prompt(
                            config.max_context_len
                        ),
                    }
                ],
                "max_tokens": FULL_CONTEXT_OUTPUT_TOKENS,
                "timeout": min(config.full_context_budget_s + 30.0, 960.0),
            }
        )

    report.checks = _evaluate_checks(
        report,
        expected_snapshot_hash=expected_snapshot_hash,
        expected_stage_count=expected_stage_count,
        max_rtt_ms=max_rtt_ms,
        config=config,
        accepted_snapshot_hashes=tuple(valid_hashes),
    )
    return report


def _evaluate_checks(
    report: GateReport,
    *,
    expected_snapshot_hash: str,
    expected_stage_count: int,
    max_rtt_ms: float,
    config: ProbeGateConfig,
    accepted_snapshot_hashes: tuple[str, ...] = (),
) -> list[GateCheck]:
    checks: list[GateCheck] = []
    samples = report.samples
    ok_samples = [sample for sample in samples if sample.ok]

    failures = [sample.error for sample in samples if not sample.ok]
    checks.append(
        GateCheck(
            name="verified-serve",
            kind="hard",
            passed=bool(samples) and not failures
            and all(
                sample.verified and sample.receipt_verified
                for sample in samples
            ),
            observed=(
                f"{len(ok_samples)}/{len(samples)} probes verified"
                + (f"; first failure: {failures[0][:200]}" if failures else "")
            ),
            threshold="every probe verified",
            rationale=(
                "any proof failure zeroes the epoch score outright; there is "
                "no partial credit to trade against"
            ),
            remediation="read the mesh error in `verathos mesh fleet` and the driver logs",
        )
    )

    checks.append(
        GateCheck(
            name="stage-coverage",
            kind="hard",
            # A decode-audited request carries one ADDITIONAL receipt (the
            # decode-audit receipt) on top of the per-stage GEMM receipts,
            # so receipts may legitimately exceed the stage count; requiring
            # equality failed honest audited probes (observed at
            # receipts=3 over 2 stages). Stages must match exactly; receipts
            # must cover them.
            passed=bool(ok_samples)
            and all(
                sample.proof_stages
                == (sample.expected_stage_count or expected_stage_count)
                and (sample.receipts or 0) >= sample.proof_stages
                for sample in ok_samples
            ),
            observed=", ".join(
                f"{sample.proof_stages}/"
                f"{sample.expected_stage_count or expected_stage_count}"
                f" ({sample.receipts} receipts)"
                for sample in ok_samples
            )
            or "no successful probes",
            threshold="proof stages == declared stages, receipts >= stages",
            rationale="a missing stage receipt fails validator verification",
        )
    )

    hard_ok = [sample for sample in report.hard_samples if sample.ok]
    hard_failures = [
        sample.error for sample in report.hard_samples if not sample.ok
    ]
    if report.hard_samples:
        checks.append(
            GateCheck(
                name="hard-proof",
                kind="hard",
                passed=not hard_failures
                and all(
                    sample.verified and sample.receipt_verified
                    for sample in report.hard_samples
                ),
                observed=(
                    f"{len(hard_ok)}/{len(report.hard_samples)} hard-tier "
                    "probes verified"
                    + (
                        "; "
                        + ", ".join(
                            f"{sample.total_s:.1f}s wall"
                            for sample in hard_ok
                            if sample.total_s > 0
                        )
                        if hard_ok
                        else ""
                    )
                    + (
                        f"; first failure: {hard_failures[0][:200]}"
                        if hard_failures
                        else ""
                    )
                ),
                threshold="every hard-tier probe verified",
                rationale=(
                    "validator audit draws force the hard GEMM sumcheck; a "
                    "mesh that only passes the light tier fails its first "
                    "audit and lands in probation"
                ),
                remediation=(
                    "read the driver logs; hard failures usually mean the "
                    "slot-view template or the proof adapter is broken for "
                    "this model"
                ),
            )
        )

    snapshot_values = {
        sample.verification_snapshot_hash for sample in ok_samples
    }
    if expected_snapshot_hash:
        # A mid-gate epoch rotation legitimately re-signs the snapshot;
        # every sample must bind a hash that was current at its time
        # (initial + manager-confirmed rotations), never anything else.
        accepted = set(accepted_snapshot_hashes) or {expected_snapshot_hash}
        threshold = (
            f"all == {expected_snapshot_hash[:12]}"
            if len(accepted) == 1
            else "all in {"
            + ", ".join(sorted(value[:12] for value in accepted))
            + "} (epoch rotation mid-gate)"
        )
        checks.append(
            GateCheck(
                name="snapshot-binding",
                kind="hard",
                passed=bool(ok_samples)
                and bool(snapshot_values)
                and snapshot_values <= accepted,
                observed=", ".join(
                    sorted(value[:12] for value in snapshot_values)
                )
                or "none",
                threshold=threshold,
                rationale=(
                    "validators bind every canary to the mesh's signed "
                    "verification snapshot"
                ),
            )
        )
    else:
        # Dev-mode pools have no signed verification snapshot; hard-failing
        # here made the standalone self-test gate unusable outside validator
        # mode. Deploy always runs validator pools, where the hash is set
        # and the hard check above applies.
        checks.append(
            GateCheck(
                name="snapshot-binding",
                kind="advisory",
                passed=True,
                observed="dev pool (no signed snapshot)",
                threshold="hard-checked only for validator-mode pools",
                rationale=(
                    "validators bind every canary to the mesh's signed "
                    "verification snapshot; dev meshes have none to bind"
                ),
            )
        )

    if config.measured_ctx_budget > 0:
        # Honesty gate: the registered maximum may not exceed what the KV
        # auto-fit measured on this hardware. Validators fill their
        # full-context canary to 0.8 * the REGISTERED value and score
        # context at that same value, so over-advertising fails an honest
        # canary and under-advertising throws score away; the measured
        # value is the only correct registration.
        checks.append(
            GateCheck(
                name="registered-context",
                kind="hard",
                passed=config.max_context_len <= config.measured_ctx_budget,
                observed=(
                    f"registering {config.max_context_len}, measured "
                    f"{config.measured_ctx_budget}"
                ),
                threshold="registered <= measured KV auto-fit",
                rationale=(
                    "validators canary at the registered maximum; a mesh "
                    "advertising more context than its KV budget holds fails "
                    "an honest audit"
                ),
                remediation=(
                    "omit --max-context-len so the measured value is "
                    "registered automatically"
                ),
            )
        )
    else:
        checks.append(
            GateCheck(
                name="registered-context",
                kind="advisory",
                passed=True,
                observed=(
                    f"registering {config.max_context_len}; no measured KV "
                    "budget available to check it against"
                ),
                threshold="informational",
                rationale=(
                    "launch a subnet pool so the KV auto-fit measures the "
                    "real budget and this becomes a hard check"
                ),
            )
        )

    if config.full_context:
        full = report.full_context
        full_ok = (
            full is not None
            and full.ok
            and full.verified
            and full.total_s <= config.full_context_budget_s
        )
        checks.append(
            GateCheck(
                name="full-context",
                kind="hard",
                passed=bool(full_ok),
                observed=(
                    "no probe ran"
                    if full is None
                    else (
                        full.error[:200]
                        if not full.ok
                        else f"verified in {full.total_s:.0f}s"
                    )
                ),
                threshold=f"verified within {config.full_context_budget_s:.0f}s",
                rationale=(
                    "the validator runs one canary at 0.8 * the registered "
                    "context in prompt tokens with a "
                    f"{VALIDATOR_FULL_CONTEXT_BUDGET_S:.0f}s budget; repeated "
                    "full-context failures trigger probation and score 0. The "
                    "0.6x gate margin covers TLS, WAN, and canary contention "
                    "the loopback probe does not see (tune with "
                    "--full-context-budget-s)"
                ),
                remediation=(
                    "lower --max-context-len, improve placement (co-located "
                    "workers), or use a faster GPU for the driver"
                ),
            )
        )

    tok_rates = [sample.wall_tok_s for sample in ok_samples if sample.wall_tok_s > 0]
    median_tok_s = median(tok_rates) if tok_rates else 0.0
    checks.append(
        GateCheck(
            name="throughput-floor",
            kind="hard",
            passed=median_tok_s >= config.min_tok_s,
            observed=f"median {median_tok_s:.1f} tok/s (wall clock)",
            threshold=f">= {config.min_tok_s:.1f} tok/s",
            rationale=(
                "small canaries generate up to 300 tokens against a 300s "
                "timeout: ~1 tok/s is where canaries start timing out and the "
                "5-receipt scoring minimum stops being met. The default 3x "
                "margin is a judgement call (tune with --min-tok-s)"
            ),
            remediation="prefer a co-located placement; see the RTT warning below",
        )
    )

    if max_rtt_ms > 0:
        # Every decode token crosses the worst link twice; the pool's own
        # placement warning uses the same ceiling formula.
        ceiling = 1000.0 / (2.0 * max_rtt_ms)
        engine_rates = [
            sample.engine_tps for sample in ok_samples if sample.engine_tps > 0
        ]
        observed_engine = median(engine_rates) if engine_rates else 0.0
        checks.append(
            GateCheck(
                name="link-ceiling",
                kind="advisory",
                passed=observed_engine <= 0 or observed_engine >= ceiling * 0.5,
                observed=(
                    f"engine {observed_engine:.1f} tok/s over a "
                    f"{max_rtt_ms:.0f} ms worst link (~{ceiling:.0f} tok/s ceiling)"
                ),
                threshold="informational",
                rationale=(
                    "a slow link caps decode throughput regardless of GPU "
                    "speed; co-located workers run at full speed"
                ),
            )
        )

    ttfts = [sample.ttft_s for sample in ok_samples if sample.ttft_s >= 0]
    median_ttft = median(ttfts) if ttfts else -1.0
    checks.append(
        GateCheck(
            name="ttft",
            kind="advisory",
            passed=median_ttft < 0 or median_ttft <= DEFAULT_TTFT_ADVISORY_S,
            observed=(
                "no streaming TTFT observed"
                if median_ttft < 0
                else f"median {median_ttft:.2f}s (loopback lower bound)"
            ),
            threshold="informational; scoring is peer-relative",
            rationale=(
                "the validator's TTFT factor is a peer-relative sqrt ratio "
                "clamped at 1.3x with no absolute term, so the gate reports "
                "TTFT and never fails on it; remember validators also pay "
                "the WAN and TLS cost this loopback number excludes"
            ),
        )
    )
    return checks


def check_public_endpoint(
    endpoint: str, *, timeout: float = 10.0, min_tls_days: float = 1.0
) -> list[GateCheck]:
    """Out-of-band checks against the URL that goes on-chain.

    probe_worker is deliberately NOT used: /capability requires the internal
    HMAC on a production coordinator, so it 401s from outside; /health is
    the only public route, and the validator routes' 403 is itself the
    signal that the mesh terminates the endpoint in validator-auth posture.
    """
    checks: list[GateCheck] = []
    endpoint = endpoint.rstrip("/")
    # Mainnet endpoints must use TLS, but the stock setup intentionally uses
    # a self-signed certificate when an operator has no domain.  Validator
    # requests authenticate the application payload and use the same
    # unverified TLS transport; the deployment gate must therefore test that
    # real transport instead of imposing a system-CA requirement the runtime
    # does not have.  The explicit certificate check below still requires a
    # successful TLS handshake and sufficient remaining certificate lifetime.
    url_context = (
        ssl._create_unverified_context()
        if endpoint.startswith("https://")
        else None
    )

    def _get(path: str):
        request = urllib.request.Request(endpoint + path, method="GET")
        with urllib.request.urlopen(
            request, timeout=timeout, context=url_context
        ) as response:
            return response.status, json.loads(response.read() or b"{}")

    def _request(method: str, path: str, payload: dict | None):
        request = urllib.request.Request(
            endpoint + path,
            data=(
                json.dumps(payload).encode() if payload is not None else None
            ),
            method=method,
            headers=(
                {"Content-Type": "application/json"}
                if payload is not None
                else {}
            ),
        )
        try:
            with urllib.request.urlopen(
                request, timeout=timeout, context=url_context
            ) as response:
                return response.status, json.loads(response.read() or b"{}")
        except urllib.error.HTTPError as exc:
            try:
                body = json.loads(exc.read() or b"{}")
            except ValueError:
                body = {}
            return exc.code, body

    try:
        status, health = _get("/health")
        health_ok = status == 200 and bool(health.get("service") or health.get("status"))
        observed = f"HTTP {status}"
    except (urllib.error.URLError, OSError, ValueError) as exc:
        health_ok = False
        observed = str(exc)[:200]
    checks.append(
        GateCheck(
            name="public-health",
            kind="hard",
            passed=health_ok,
            observed=observed,
            threshold="HTTP 200 from /health",
            rationale="validators must reach the registered endpoint",
            remediation=(
                "check DNS, firewall, and any reverse proxy in front of the "
                "coordinator's mesh port"
            ),
        )
    )

    # Probe each route with the METHOD validators actually use: POST for
    # chat, GET for the snapshot. A POSTed snapshot request is not a
    # validator route at all and lands in the internal-HMAC lane, whose
    # 401 body says "internal auth header" and read as a broken posture.
    for method, path, payload in (
        ("POST", "/v1/chat/completions", {}),
        ("GET", "/v1/mesh/verification-snapshot", None),
    ):
        try:
            status, body = _request(method, path, payload)
            error_text = str(body.get("error", ""))
            # Two honest refusal shapes exist: 403 "validator
            # authentication is required" when the worker has no
            # validator allowlist configured, and 401 "missing or
            # duplicate validator auth header" when the allowlist is
            # active and signature verification runs first
            # (http_auth.verify_validator_http_request). Both prove the
            # coordinator terminates the port and refuses
            # unauthenticated validator routes; demanding exactly 403
            # rejects correctly configured workers once the in-worker
            # allowlist refresher is active.
            posture_ok = (
                status == 403 and "validator" in error_text
            ) or (status == 401 and "validator auth" in error_text)
            if status == 200:
                observed = "HTTP 200: validator auth is DISABLED"
            elif status in (401, 403) and not posture_ok:
                observed = f"HTTP {status} with unexpected body {body!r}"
            else:
                observed = f"HTTP {status}"
        except (urllib.error.URLError, OSError, ValueError) as exc:
            posture_ok = False
            observed = str(exc)[:200]
        checks.append(
            GateCheck(
                name=f"validator-auth-posture {path}",
                kind="hard",
                passed=posture_ok,
                observed=observed,
                threshold=(
                    "HTTP 403 (no allowlist) or HTTP 401 (allowlist "
                    "active) refusing unauthenticated validator routes"
                ),
                rationale=(
                    "a validator-auth refusal proves the endpoint is "
                    "reachable, terminated by the mesh coordinator, and "
                    "refusing unauthenticated validator routes; a 200 "
                    "means validator auth is off, a 404/HTML means the "
                    "proxy routes to the wrong upstream"
                ),
            )
        )

    if endpoint.startswith("https://"):
        host_port = endpoint[len("https://") :].split("/", 1)[0]
        host, _sep, port_text = host_port.partition(":")
        port = int(port_text or 443)
        try:
            context = ssl._create_unverified_context()
            with socket.create_connection((host, port), timeout=timeout) as sock:
                with context.wrap_socket(sock, server_hostname=host) as tls:
                    der_cert = tls.getpeercert(binary_form=True)
            if not der_cert:
                raise ssl.SSLError("TLS peer returned no certificate")
            pem_cert = ssl.DER_cert_to_PEM_cert(der_cert)
            # CERT_NONE deliberately leaves getpeercert()'s decoded mapping
            # empty.  Decode the certificate bytes with CPython's standard
            # certificate helper so expiry remains a hard gate without
            # introducing a new runtime dependency.
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".pem", encoding="ascii"
            ) as cert_file:
                cert_file.write(pem_cert)
                cert_file.flush()
                cert = ssl._ssl._test_decode_cert(cert_file.name)
            not_after = cert.get("notAfter", "")
            expires = (
                ssl.cert_time_to_seconds(not_after) if not_after else 0.0
            )
            days_left = (expires - time.time()) / 86_400 if expires else -1.0
            checks.append(
                GateCheck(
                    name="tls-certificate",
                    kind="hard",
                    passed=days_left >= min_tls_days,
                    observed=f"expires in {days_left:.1f} days ({not_after})",
                    threshold=f">= {min_tls_days:.0f} day(s) beyond the 24h lease",
                    rationale=(
                        "a certificate expiring inside the lease breaks every "
                        "validator connection mid-lease"
                    ),
                )
            )
        except (OSError, ssl.SSLError, ValueError) as exc:
            checks.append(
                GateCheck(
                    name="tls-certificate",
                    kind="hard",
                    passed=False,
                    observed=str(exc)[:200],
                    threshold="successful TLS handshake with a current certificate",
                    rationale=(
                        "validators require encrypted transport; request and proof "
                        "authentication do not depend on a public certificate authority"
                    ),
                )
            )
    else:
        checks.append(
            GateCheck(
                name="tls-certificate",
                kind="advisory",
                passed=False,
                observed="plain http endpoint",
                threshold="https recommended",
                rationale=(
                    "validators tolerate http today, but everything on this "
                    "path is signed either way; prefer TLS in production"
                ),
            )
        )
    return checks
