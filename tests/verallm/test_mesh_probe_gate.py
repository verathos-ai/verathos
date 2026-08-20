"""Probe gate checks: canary-derived thresholds and verdicts."""
from __future__ import annotations

import pytest

from verallm.mesh.probe import (
    DEFAULT_FULL_CONTEXT_BUDGET_S,
    GateReport,
    ProbeGateConfig,
    ProbeSample,
    build_full_context_probe_prompt,
    probe_sample_from_chat_result,
    run_probe_gate,
)

SNAPSHOT = "ab" * 32


def _result(**overrides):
    values = {
        "status": "ok",
        "error": "",
        "content": "answer",
        "verified": True,
        "receipt_verified": True,
        "receipts": 2,
        "proof_stages": 2,
        "expected_stage_count": 2,
        "proof_mode": "verathos_ggml_gemm_v1",
        "proof_receipt_root": "cd" * 32,
        "mesh_response_commitment_hash": "ef" * 32,
        "verification_snapshot_hash": SNAPSHOT,
        "usage": {"completion_tokens": 120, "prompt_tokens": 30},
        "engine_tps": 40.0,
        "prompt_tps": 900.0,
        "ttft_s": 0.6,
        "total_s": 5.0,
        "pickup_s": 0.05,
    }
    values.update(overrides)
    return values


def _gate(results, *, config=None, max_rtt_ms=1.0, status_hash=SNAPSHOT):
    """Drive run_probe_gate against canned chat results.

    The default config disables the hard-tier sample so the historical
    queues (2 small + 1 full-context) line up; hard-tier behaviour has its
    own tests below. ``status_hash`` is what the manager reports as the
    mesh's CURRENT snapshot when the gate re-checks after a divergent
    probe binding (the mid-gate rotation path).
    """
    queue = list(results)

    def call(route, body):
        if route == "/v1/pool/status":
            return {
                "meshes": {
                    "m-1": {"verification_snapshot_hash": status_hash}
                }
            }
        assert route == "/v1/pool/chat"
        # Probe requests must ride the probe lane with streaming TTFT.
        assert body["probe"] is True
        assert body["stream"] is True
        assert body["thinking"] is False
        return queue.pop(0)

    return run_probe_gate(
        call=call,
        mesh_key="m-1",
        expected_snapshot_hash=SNAPSHOT,
        expected_stage_count=2,
        max_rtt_ms=max_rtt_ms,
        config=config or ProbeGateConfig(samples=2, hard_samples=0),
    )


def _check(report: GateReport, name: str):
    return next(check for check in report.checks if check.name == name)


def test_all_green_gate_passes():
    report = _gate([_result(), _result(), _result(total_s=200.0)])
    assert report.passed
    assert _check(report, "verified-serve").passed
    assert _check(report, "stage-coverage").passed
    assert _check(report, "snapshot-binding").passed
    assert _check(report, "full-context").passed
    assert _check(report, "throughput-floor").passed
    # No em-dashes in operator-facing output.
    assert "\u2014" not in report.render()


def test_incomplete_stage_coverage_fails():
    report = _gate([_result(proof_stages=1), _result(), _result()])
    assert not report.passed
    assert not _check(report, "stage-coverage").passed


def test_snapshot_mismatch_fails():
    report = _gate(
        [_result(verification_snapshot_hash="00" * 32), _result(), _result()]
    )
    assert not _check(report, "snapshot-binding").passed
    assert not report.passed


def test_mid_gate_rotation_is_accepted_and_straddler_retried():
    """The manager's epoch follower re-signs serving meshes at every chain
    epoch boundary; a gate spanning one must accept the rotated hash it
    CONFIRMS with the manager and retry the straddled probe (before this,
    every boundary-crossing first gate failed and burned a full rerun)."""

    rotated = "bb" * 32
    report = _gate(
        [
            _result(),  # probe 1 under the initial snapshot
            _result(verification_snapshot_hash=rotated),  # rotation seen
            _result(verification_snapshot_hash=rotated),  # retried probe
            _result(verification_snapshot_hash=rotated, total_s=200.0),
        ],
        status_hash=rotated,
    )
    assert _check(report, "snapshot-binding").passed
    assert "epoch rotation" in _check(report, "snapshot-binding").threshold
    assert report.passed


def test_unconfirmed_foreign_snapshot_still_fails():
    """A probe bound to a hash the manager does NOT report as current is a
    real fault, never excused as rotation."""

    report = _gate(
        [
            _result(verification_snapshot_hash="00" * 32),
            _result(),
            _result(total_s=200.0),
        ],
        status_hash=SNAPSHOT,
    )
    assert not _check(report, "snapshot-binding").passed
    assert not report.passed


def test_full_context_over_budget_fails_with_canary_rationale():
    report = _gate(
        [
            _result(),
            _result(),
            _result(total_s=DEFAULT_FULL_CONTEXT_BUDGET_S + 60.0),
        ]
    )
    check = _check(report, "full-context")
    assert not check.passed
    assert "900" in check.rationale
    assert not report.passed


def test_throughput_floor_uses_wall_clock_not_engine_tps():
    # 120 tokens over 100 s = 1.2 wall tok/s even though the engine says 40:
    # the validator divides by phase-one wall clock, so the gate must too.
    slow = _result(total_s=100.0, engine_tps=40.0)
    report = _gate([slow, slow, _result(total_s=200.0)])
    check = _check(report, "throughput-floor")
    assert not check.passed
    assert "1.2 tok/s" in check.observed
    sample = probe_sample_from_chat_result(slow)
    assert sample.wall_tok_s == pytest.approx(1.2)


def test_hard_tier_probe_asserts_the_hard_relation():
    """The gate sends an explicit upgrade-only hard-tier probe and hard-fails
    when it does not verify: a mesh that only passes light fails its first
    validator audit."""
    seen_bodies = []
    queue = [
        _result(),  # small
        _result(total_s=42.0),  # hard tier
        _result(total_s=200.0),  # full context
    ]

    def call(route, body):
        seen_bodies.append(body)
        return queue.pop(0)

    report = run_probe_gate(
        call=call,
        mesh_key="m-1",
        expected_snapshot_hash=SNAPSHOT,
        expected_stage_count=2,
        max_rtt_ms=1.0,
        config=ProbeGateConfig(samples=1, hard_samples=1),
    )
    assert report.passed
    check = _check(report, "hard-proof")
    assert check.kind == "hard"
    assert check.passed
    assert "42.0s" in check.observed
    hard_bodies = [b for b in seen_bodies if b.get("proof_tier") == "hard"]
    assert len(hard_bodies) == 1
    # Small and full-context probes never carry a tier field: organic light
    # serving must stay indistinguishable.
    assert all(
        "proof_tier" not in b for b in seen_bodies if b not in hard_bodies
    )


def test_hard_tier_failure_blocks_the_gate():
    queue = [
        _result(),
        _result(verified=False, receipt_verified=False),
        _result(total_s=200.0),
    ]

    def call(route, body):
        return queue.pop(0)

    report = run_probe_gate(
        call=call,
        mesh_key="m-1",
        expected_snapshot_hash=SNAPSHOT,
        expected_stage_count=2,
        max_rtt_ms=1.0,
        config=ProbeGateConfig(samples=1, hard_samples=1),
    )
    assert not report.passed
    assert not _check(report, "hard-proof").passed


def test_registered_context_above_measurement_fails_hard():
    report = _gate(
        [_result(), _result(), _result(total_s=100.0)],
        config=ProbeGateConfig(
            samples=2,
            hard_samples=0,
            max_context_len=131_072,
            measured_ctx_budget=120_000,
        ),
    )
    check = _check(report, "registered-context")
    assert check.kind == "hard"
    assert not check.passed
    assert "measured" in check.observed
    assert not report.passed


def test_registered_context_at_measurement_passes():
    report = _gate(
        [_result(), _result(), _result(total_s=200.0)],
        config=ProbeGateConfig(
            samples=2,
            hard_samples=0,
            max_context_len=389_120,
            measured_ctx_budget=389_120,
        ),
    )
    check = _check(report, "registered-context")
    assert check.kind == "hard"
    assert check.passed
    assert report.passed


def test_registered_context_without_measurement_is_advisory():
    report = _gate(
        [_result(), _result(), _result(total_s=200.0)],
        config=ProbeGateConfig(
            samples=2, hard_samples=0, max_context_len=131_072
        ),
    )
    check = _check(report, "registered-context")
    assert check.kind == "advisory"
    assert check.passed
    assert report.passed


def test_ttft_is_advisory_never_hard():
    report = _gate(
        [_result(ttft_s=45.0), _result(ttft_s=45.0), _result(total_s=100.0)]
    )
    check = _check(report, "ttft")
    assert check.kind == "advisory"
    assert not check.passed  # flagged, but...
    # ...advisory checks never block the gate.
    hard_failures = [
        c for c in report.checks if c.kind == "hard" and not c.passed
    ]
    assert not hard_failures
    assert report.passed


def test_ttft_sentinel_does_not_fail_anything():
    report = _gate(
        [_result(ttft_s=-1.0), _result(ttft_s=-1.0), _result(total_s=100.0)]
    )
    assert report.passed
    assert "no streaming TTFT" in _check(report, "ttft").observed


def test_probe_transport_error_fails_verified_serve():
    def call(route, body):
        raise RuntimeError("HTTP 500: mesh did not respond in time")

    report = run_probe_gate(
        call=call,
        mesh_key="m-1",
        expected_snapshot_hash=SNAPSHOT,
        expected_stage_count=2,
        max_rtt_ms=1.0,
        config=ProbeGateConfig(samples=1, hard_samples=0, full_context=False),
    )
    check = _check(report, "verified-serve")
    assert not check.passed
    assert "did not respond" in check.observed


def test_full_context_prompt_fills_the_advertised_context():
    # 0.8 * the advertised value, UNCAPPED: a mesh registered at its
    # measured 131k must actually serve ~104k prompt tokens. The first
    # line is the uniqueness nonce; the fill sits after it.
    prompt = build_full_context_probe_prompt(131_072)
    nonce_line, fill = prompt.split("\n", 1)
    assert nonce_line.startswith("probe ")
    words = fill.split("\n\n")[0].split(" ")
    assert len(words) == int(131_072 * 0.8)
    small = build_full_context_probe_prompt(8_192)
    assert len(
        small.split("\n", 1)[1].split("\n\n")[0].split(" ")
    ) == int(8_192 * 0.8)


def test_full_context_prompt_is_unique_per_call():
    """llama-server reuses the cached KV prefix of the previous request in
    a slot; probes sharing their opening tokens only measure the uncached
    suffix (and the gate's certified probe would under-measure a cold
    validator canary). Every prompt must differ from its first token."""
    a = build_full_context_probe_prompt(8_192)
    b = build_full_context_probe_prompt(8_192)
    assert a != b
    assert a.split("\n", 1)[0] != b.split("\n", 1)[0]


def test_probe_body_contains_no_novel_fields():
    """The probe must not add fingerprintable fields to the inference lane:
    only the manager control fields (mesh_key/probe/stream/thinking/timeout,
    plus the upgrade-only proof_tier on the explicit hard sample) and the
    ordinary chat fields may appear."""
    seen_bodies = []

    def call(route, body):
        seen_bodies.append(body)
        return _result()

    run_probe_gate(
        call=call,
        mesh_key="m-1",
        expected_snapshot_hash=SNAPSHOT,
        expected_stage_count=2,
        max_rtt_ms=1.0,
        config=ProbeGateConfig(samples=1, hard_samples=1, full_context=True),
    )
    allowed = {
        "mesh_key",
        "probe",
        "stream",
        "thinking",
        "messages",
        "max_tokens",
        "timeout",
        "proof_tier",
    }
    for body in seen_bodies:
        assert set(body) <= allowed


def test_audited_probe_extra_receipt_passes_stage_coverage():
    # A decode-audited request carries one additional receipt on top of the
    # per-stage GEMM receipts; equality with the stage count rejected honest
    # audited probes (observed at receipts=3 over 2 stages).
    report = _gate(
        [_result(receipts=3), _result(), _result(total_s=200.0)],
    )
    check = _check(report, "stage-coverage")
    assert check.passed
    assert "3 receipts" in check.observed


def test_missing_stage_receipt_still_fails_stage_coverage():
    report = _gate(
        [_result(receipts=1, proof_stages=1), _result(), _result(total_s=200.0)],
    )
    assert not _check(report, "stage-coverage").passed


def test_dev_pool_without_snapshot_is_advisory_not_hard():
    queue = [
        _result(verification_snapshot_hash=""),
        _result(verification_snapshot_hash=""),
        _result(verification_snapshot_hash="", total_s=200.0),
    ]

    def call(route, body):
        return queue.pop(0)

    report = run_probe_gate(
        call=call,
        mesh_key="m-1",
        expected_snapshot_hash="",
        expected_stage_count=2,
        max_rtt_ms=1.0,
        config=ProbeGateConfig(samples=2, hard_samples=0),
    )
    check = _check(report, "snapshot-binding")
    assert check.kind == "advisory"
    assert check.passed
    assert report.passed
