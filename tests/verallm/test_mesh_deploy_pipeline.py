"""Deploy pipeline ordering: measure first, gate, and no chain write before
the gate passes; the registered context is the launch's measurement."""
from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

import verallm.mesh.deploy as deploy_module
import verallm.mesh.registration as registration_module
from verallm.chain.miner_lifecycle import LifecycleRefusal, LifecyclePlan
from verallm.mesh.deploy import DeployConfig, run_deploy
from verallm.mesh.probe import GateCheck, GateReport, ProbeGateConfig, ProbeSample

MODEL_ID = "qwen2.5-7b-q4-k-m"
PRIVATE_KEY = "0x" + "11" * 32
SNAPSHOT = "ab" * 32
MEASURED = 120_000
# Without an explicit override, the derived registration is rounded DOWN to
# a multiple of 1024 (the fake pool's 1.0s timing probe makes the time cap
# effectively unbounded, so the KV fit wins).
MEASURED_ROUNDED = (MEASURED // 1024) * 1024


def _signer_address() -> str:
    from eth_account import Account

    return Account.from_key(PRIVATE_KEY).address


def _anchors():
    from verallm.mesh.registration import MeshChainAnchors

    return MeshChainAnchors(
        model_id=MODEL_ID,
        gguf_scheme="q4_k_m",
        registry_quant="gguf_mesh_q4_k_m",
        chain_quantization_scheme="gguf_q4_k_m",
        model_package_hash="01" * 32,
        model_tensor_manifest_root="02" * 32,
        tokenizer_hash="03" * 32,
        total_layers=28,
        max_context_len=0,
    )


class _FakePool:
    """Records every route call; models the measure -> register -> relaunch
    lifecycle: the first launch serves UNREGISTERED (no snapshot, no index)
    and reports a measured context; after register-model, a launch serves
    chain-bound with a signed snapshot."""

    def __init__(self):
        self.calls: list[tuple[str, dict]] = []
        self.meshes: dict[str, dict] = {}
        self.launch_count = 0
        self.registered_model: dict | None = None
        self.stopped: list[str] = []
        self.serving_mode = "subnet"
        self.renewal_suspended = False

    def __call__(self, route: str, body: dict) -> dict:
        self.calls.append((route, dict(body)))
        if route == "/v1/pool/status":
            models = {
                MODEL_ID: {
                    "hf_repo": "org/model",
                    "hf_files": ["model.gguf"],
                    "model_bytes": 4_000_000_000,
                }
            }
            if self.launch_count:
                models[MODEL_ID]["measured_ctx_budget"] = MEASURED
            if self.registered_model is not None:
                models[MODEL_ID].update(self.registered_model)
            return {
                "status": "ok",
                "serving_mode": self.serving_mode,
                "coordinator_address": _signer_address(),
                "validator_binding": {
                    "chain_id": 945,
                    "netuid": 405,
                    "coordinator_uid": 2,
                    "epoch": 1,
                },
                "workers": {
                    "pod-gpu0": {
                        "status": "idle",
                        # deliberately the PRE-RENAME key: pins the
                        # compat shim for workers running older code
                        "capability": {
                            "validator_driver_ready": True,
                            "rpc_device": "CUDA0",
                        },
                    },
                    "pod-gpu1": {
                        "status": "idle",
                        "capability": {
                            "subnet_driver_ready": False,
                            "rpc_device": "CUDA1",
                        },
                    },
                },
                "meshes": {k: dict(m) for k, m in self.meshes.items()},
                "models": models,
            }
        if route == "/v1/pool/recommend":
            return {
                "status": "ok",
                "suggestions": [
                    {
                        "workers": ["pod-gpu0", "pod-gpu1"],
                        "driver": "pod-gpu0",
                        "max_rtt_ms": 1.0,
                        "link_class": "local",
                        "driver_vram_gb": 24.0,
                    },
                    {
                        "workers": ["pod-gpu0", "far"],
                        "driver": "pod-gpu0",
                        "max_rtt_ms": 95.0,
                        "link_class": "far",
                        "driver_vram_gb": 24.0,
                    },
                ],
                "reasons": {"pod-gpu0": "available (can drive)"},
            }
        if route == "/v1/pool/register-model":
            self.registered_model = dict(body)
            return {"status": "ok"}
        if route == "/v1/pool/launch":
            self.launch_count += 1
            mesh_key = f"m-test{self.launch_count}"
            chain_bound = bool(
                self.registered_model is not None
                and self.registered_model.get("model_index") is not None
                and not body.get("_deploy_measurement_unbound")
            )
            self.meshes = {
                mesh_key: {
                    "model_id": MODEL_ID,
                    "driver": "pod-gpu0",
                    "members": ["pod-gpu0", "pod-gpu1"],
                    "status": "serving",
                    "routing_ready": True,
                    "measured_ctx_budget": MEASURED,
                    **(
                        {
                            "verification_snapshot_hash": SNAPSHOT,
                            "model_index": int(
                                self.registered_model["model_index"]
                            ),
                            "max_context_len": int(
                                self.registered_model.get(
                                    "max_context_len", 0
                                )
                            ),
                            # The real pool surfaces the chain anchors on
                            # every chain-bound mesh; deploy compares them
                            # to decide whether a re-deploy may keep the
                            # serving mesh or must relaunch it.
                            "model_package_hash": str(
                                self.registered_model.get(
                                    "model_package_hash", ""
                                )
                            ),
                            "model_tensor_manifest_root": str(
                                self.registered_model.get(
                                    "model_tensor_manifest_root", ""
                                )
                            ),
                            "tokenizer_hash": str(
                                self.registered_model.get("tokenizer_hash", "")
                            ),
                            "quantization_scheme": str(
                                self.registered_model.get(
                                    "quantization_scheme", ""
                                )
                            ),
                        }
                        if chain_bound
                        else {}
                    ),
                }
            }
            return {"status": "launching", "mesh_key": mesh_key}
        if route == "/v1/pool/stop":
            self.stopped.append(body.get("mesh_key"))
            self.meshes = {}
            return {"status": "ok"}
        if route == "/v1/pool/chat":
            mesh = next(iter(self.meshes.values()), {})
            return {
                "status": "ok",
                "content": "ready",
                "verified": True,
                "receipt_verified": True,
                "receipts": 2,
                "proof_stages": 2,
                "expected_stage_count": 2,
                "verification_snapshot_hash": mesh.get(
                    "verification_snapshot_hash", ""
                ),
                "usage": {"completion_tokens": 4, "prompt_tokens": 8},
                "ttft_s": 0.4,
                "total_s": 1.0,
            }
        if route == "/v1/pool/registration-state":
            if "suspend_renewal" in body:
                self.renewal_suspended = bool(body["suspend_renewal"])
                return {
                    "status": "ok",
                    "registration": {
                        "model_id": MODEL_ID,
                        "renewal_suspended": self.renewal_suspended,
                    },
                    "registrations": {
                        MODEL_ID: {
                            "model_id": MODEL_ID,
                            "renewal_suspended": self.renewal_suspended,
                        }
                    },
                    "renewal_in_progress": False,
                }
            if body.get("registration") is not None:
                self.renewal_suspended = bool(
                    body["registration"].get("renewal_suspended")
                )
            return {"status": "ok", "registration": body.get("registration")}
        raise AssertionError(f"unexpected route {route}")


def _passing_gate(**overrides):
    report = GateReport(
        samples=[ProbeSample(ok=True, verified=True, receipt_verified=True)],
        hard_samples=[
            ProbeSample(ok=True, verified=True, receipt_verified=True)
        ],
        checks=[
            GateCheck(
                name="verified-serve",
                kind="hard",
                passed=True,
                observed="ok",
                threshold="",
                rationale="",
            )
        ],
    )
    for key, value in overrides.items():
        setattr(report, key, value)
    return report


def _failing_gate():
    return _passing_gate(
        checks=[
            GateCheck(
                name="throughput-floor",
                kind="hard",
                passed=False,
                observed="median 1.2 tok/s",
                threshold=">= 3.0",
                rationale="canaries would time out",
            )
        ]
    )


@pytest.fixture()
def wired(monkeypatch):
    """Stub chain and gate seams; record chain writes."""
    chain_calls: list[str] = []
    monkeypatch.setattr(
        registration_module,
        "resolve_mesh_chain_anchors",
        lambda cfg, model_id, **kwargs: _anchors(),
    )
    monkeypatch.setattr(
        registration_module,
        "ensure_evm_registered",
        lambda *args, **kwargs: chain_calls.append("registerEvm") or True,
    )
    monkeypatch.setattr(
        registration_module,
        "plan_mesh_registration",
        lambda cfg, target, signer, **kwargs: LifecyclePlan(
            action="append", predicted_index=0, reason="fresh"
        ),
    )

    def fake_register(
        cfg, target, *, private_key, expected_index=None, **kwargs
    ):
        chain_calls.append("registerModel")
        return registration_module.RegistrationOutcome(
            action="append",
            index=0,
            tx_hash="0x" + "ab" * 32,
            expires_at=int(time.time()) + 86_400,
        )

    monkeypatch.setattr(
        registration_module, "register_mesh_endpoint", fake_register
    )
    monkeypatch.setattr(
        deploy_module, "run_probe_gate", lambda **kwargs: _passing_gate()
    )
    monkeypatch.setattr(
        deploy_module, "check_public_endpoint", lambda endpoint, **kw: []
    )
    return chain_calls


def _config(**overrides) -> DeployConfig:
    values = dict(
        model_id=MODEL_ID,
        endpoint="https://mesh.example:9443",
        chain_config=SimpleNamespace(chain_id=945, netuid=405),
        private_key=PRIVATE_KEY,
        hotkey_seed=b"\x05" * 32,
        uid=2,
        netuid=405,
        probe=ProbeGateConfig(samples=1, hard_samples=0, full_context=False),
        assume_yes=True,
        launch_wait_s=10.0,
        poll_interval_s=0.0,
    )
    values.update(overrides)
    return DeployConfig(**values)


def test_happy_path_measures_gates_then_registers(wired):
    pool = _FakePool()
    report = run_deploy(_config(), call=pool, out=lambda _line: None, sleep=lambda _s: None)
    assert not report.failed
    assert report.registration is not None
    # Chain writes happen only after the gate.
    assert wired == ["registerEvm", "registerModel"]
    # The registered context comes from the launch's MEASUREMENT (the fast
    # fake timing probe leaves the KV fit as the binding cap), never typed;
    # the derived registration rounds down to a multiple of 1024.
    assert report.measured_ctx_budget == MEASURED
    assert report.registered_context_len == MEASURED_ROUNDED
    assert pool.registered_model is not None
    assert pool.registered_model["max_context_len"] == MEASURED_ROUNDED
    assert pool.registered_model["measured_ctx_budget"] == MEASURED
    assert pool.registered_model["model_index"] == 0
    # Measurement launch, then chain-bound relaunch replacing it.
    assert pool.launch_count == 2
    assert pool.stopped == ["m-test1"]
    routes = [route for route, _body in pool.calls]
    assert routes.index("/v1/pool/launch") < routes.index(
        "/v1/pool/register-model"
    )
    assert "/v1/pool/registration-state" in routes


def test_deploy_threads_stored_registration_to_plan_and_write(
    wired, monkeypatch
):
    previous = {"model_id": MODEL_ID, "index": 7}
    seen: list[tuple[str, object]] = []

    def plan(cfg, target, signer, *, previous_registration=None):
        seen.append(("plan", previous_registration))
        return LifecyclePlan(
            action="update-endpoint", predicted_index=0, reason="replacement"
        )

    def register(
        cfg,
        target,
        *,
        private_key,
        expected_index=None,
        previous_registration=None,
    ):
        seen.append(("register", previous_registration))
        return registration_module.RegistrationOutcome(
            action="update-endpoint",
            index=0,
            tx_hash="0x" + "ab" * 32,
            expires_at=int(time.time()) + 86_400,
        )

    monkeypatch.setattr(registration_module, "plan_mesh_registration", plan)
    monkeypatch.setattr(registration_module, "register_mesh_endpoint", register)
    pool = _FakePool()
    report = run_deploy(
        _config(previous_registration=previous),
        call=pool,
        out=lambda _line: None,
        sleep=lambda _s: None,
    )

    assert not report.failed
    assert seen == [("plan", previous), ("register", previous)]


def test_mainnet_deploy_refuses_non_cuda_before_launch_or_chain_write(wired):
    pool = _FakePool()
    original = pool.__call__

    def call(route, body):
        result = original(route, body)
        if route == "/v1/pool/status":
            result["validator_binding"]["chain_id"] = 964
            result["validator_binding"]["netuid"] = 96
            result["workers"]["pod-gpu1"]["capability"]["rpc_device"] = "MTL0"
        return result

    report = run_deploy(
        _config(
            chain_config=SimpleNamespace(chain_id=964, netuid=96),
            netuid=96,
        ),
        call=call,
        out=lambda _line: None,
        sleep=lambda _s: None,
    )
    assert report.failed
    assert report.stages[-1]["name"] == "placement"
    assert "mixed CUDA/non-CUDA" in report.stages[-1]["detail"]
    assert pool.launch_count == 0
    assert wired == []


def test_mainnet_deploy_accepts_all_cuda_workers(wired):
    pool = _FakePool()
    original = pool.__call__

    def call(route, body):
        result = original(route, body)
        if route == "/v1/pool/status":
            result["validator_binding"]["chain_id"] = 964
            result["validator_binding"]["netuid"] = 96
        return result

    report = run_deploy(
        _config(
            chain_config=SimpleNamespace(chain_id=964, netuid=96),
            netuid=96,
        ),
        call=call,
        out=lambda _line: None,
        sleep=lambda _s: None,
    )
    assert not report.failed
    assert "registerModel" in wired


def test_testnet_deploy_keeps_metal_available(wired):
    pool = _FakePool()
    original = pool.__call__

    def call(route, body):
        result = original(route, body)
        if route == "/v1/pool/status":
            result["workers"]["pod-gpu0"]["capability"]["rpc_device"] = "MTL0"
            result["workers"]["pod-gpu1"]["capability"]["rpc_device"] = "MTL1"
        return result

    report = run_deploy(
        _config(), call=call, out=lambda _line: None, sleep=lambda _s: None
    )
    assert not report.failed
    assert "registerModel" in wired


def test_failed_gate_sends_no_tx_and_keeps_serving(wired, monkeypatch):
    monkeypatch.setattr(
        deploy_module, "run_probe_gate", lambda **kwargs: _failing_gate()
    )
    pool = _FakePool()
    lines: list[str] = []
    report = run_deploy(
        _config(), call=pool, out=lines.append, sleep=lambda _s: None
    )
    assert report.failed
    assert report.registration is None
    assert "registerModel" not in wired
    assert "registerEvm" not in wired
    # The measurement mesh is NOT stopped: it keeps serving unregistered.
    assert pool.stopped == []
    text = "\n".join(lines)
    assert "UNREGISTERED" in text
    # The report points at the best alternative placement.
    assert "far" in text


def test_force_overrides_gate_but_not_final_posture(wired, monkeypatch):
    monkeypatch.setattr(
        deploy_module, "run_probe_gate", lambda **kwargs: _failing_gate()
    )
    posture_failure = [
        GateCheck(
            name="validator-auth-posture /v1/chat/completions",
            kind="hard",
            passed=False,
            observed="HTTP 200: validator auth is DISABLED",
            threshold="403",
            rationale="",
        )
    ]
    monkeypatch.setattr(
        deploy_module,
        "check_public_endpoint",
        lambda endpoint, **kw: posture_failure,
    )
    pool = _FakePool()
    report = run_deploy(
        _config(force=True), call=pool, out=lambda _l: None, sleep=lambda _s: None
    )
    assert report.failed
    assert "registerModel" not in wired

    # With clean reachability, --force does register.
    monkeypatch.setattr(
        deploy_module, "check_public_endpoint", lambda endpoint, **kw: []
    )
    pool = _FakePool()
    report = run_deploy(
        _config(force=True), call=pool, out=lambda _l: None, sleep=lambda _s: None
    )
    assert not report.failed
    assert "registerModel" in wired


def test_dry_run_stops_before_launch_and_tx(wired):
    pool = _FakePool()
    report = run_deploy(
        _config(dry_run=True), call=pool, out=lambda _l: None, sleep=lambda _s: None
    )
    assert not report.failed
    assert report.registration is None
    assert wired == []
    routes = [route for route, _body in pool.calls]
    assert "/v1/pool/launch" not in routes
    assert "/v1/pool/register-model" not in routes


def test_context_override_beyond_jitter_of_measurement_refused(wired):
    pool = _FakePool()
    report = run_deploy(
        _config(max_context_len=200_000),  # measured 120k, +10% = 132k
        call=pool,
        out=lambda _l: None,
        sleep=lambda _s: None,
    )
    assert report.failed
    assert "measured" in report.stages[-1]["detail"]
    assert wired == []


def test_context_override_within_jitter_is_accepted(wired):
    pool = _FakePool()
    report = run_deploy(
        _config(max_context_len=126_000),  # within 10% of the 120k measurement
        call=pool,
        out=lambda _l: None,
        sleep=lambda _s: None,
    )
    assert not report.failed
    # The explicit value registers exactly: no derivation, no rounding.
    assert report.registered_context_len == 126_000
    assert pool.registered_model["max_context_len"] == 126_000
    # And no timing probe ran: the only chat is the final verification.
    chat_calls = [body for route, body in pool.calls if route == "/v1/pool/chat"]
    assert len(chat_calls) == 1


def _timed_chat_pool(pool, total_s):
    """Wrap _FakePool so every /v1/pool/chat reports the given wall time
    (the timing probe reads total_s; the final verification ignores it)."""
    original = pool.__call__

    def call(route, body):
        result = original(route, body)
        if route == "/v1/pool/chat":
            result["total_s"] = total_s
        return result

    return call


def test_slow_timing_probe_derives_cap_below_kv_fit(wired):
    """200s at the 32768-token timing probe: rate = 200/32768 s per context
    token, cap = floor(0.75 * 540 / rate) = 66355, rounded down to 65536,
    which beats the 120k KV fit."""
    pool = _FakePool()
    lines: list[str] = []
    report = run_deploy(
        _config(),
        call=_timed_chat_pool(pool, 200.0),
        out=lines.append,
        sleep=lambda _s: None,
    )
    assert not report.failed
    assert report.registered_context_len == 65_536
    assert pool.registered_model["max_context_len"] == 65_536
    # The measurement is still reported honestly alongside the capped value.
    assert pool.registered_model["measured_ctx_budget"] == MEASURED
    text = "\n".join(lines)
    assert "time cap 66355" in text
    assert "registering 65536" in text
    assert "measured KV fit 120000" in text


def test_fast_timing_probe_lets_the_kv_fit_win(wired):
    """1.0s at 32768 tokens puts the time cap in the millions; the KV fit
    is the binding limit, rounded down to a multiple of 1024."""
    pool = _FakePool()
    report = run_deploy(
        _config(), call=pool, out=lambda _l: None, sleep=lambda _s: None
    )
    assert not report.failed
    assert report.registered_context_len == MEASURED_ROUNDED
    assert MEASURED_ROUNDED % 1024 == 0
    assert MEASURED_ROUNDED <= MEASURED


def test_derived_context_never_below_the_floor(wired):
    """A mesh so slow the raw cap lands near 2.6k tokens still registers
    the 8192 floor; whether it can actually serve that in budget is the
    gate's certified full-context probe to decide."""
    pool = _FakePool()
    report = run_deploy(
        _config(),
        call=_timed_chat_pool(pool, 5000.0),
        out=lambda _l: None,
        sleep=lambda _s: None,
    )
    assert not report.failed
    assert report.registered_context_len == 8192
    assert pool.registered_model["max_context_len"] == 8192


def test_validator_budget_override_scales_the_cap(wired):
    """validator_budget_s mirrors the validator's
    canary_full_context_inference_timeout: with the real 900s ceiling the
    same 200s probe derives floor(0.75 * 900 * 32768 / 200) = 110592."""
    pool = _FakePool()
    report = run_deploy(
        _config(validator_budget_s=900.0),
        call=_timed_chat_pool(pool, 200.0),
        out=lambda _l: None,
        sleep=lambda _s: None,
    )
    assert not report.failed
    assert report.registered_context_len == 110_592
    assert pool.registered_model["max_context_len"] == 110_592


def test_timing_probe_failure_fails_the_context_stage(wired):
    pool = _FakePool()
    original = pool.__call__

    def call(route, body):
        if route == "/v1/pool/chat":
            return {"status": "error", "error": "slots busy"}
        return original(route, body)

    report = run_deploy(
        _config(), call=call, out=lambda _l: None, sleep=lambda _s: None
    )
    assert report.failed
    assert report.stages[-1]["name"] == "context"
    assert "timing probe" in report.stages[-1]["detail"]
    assert wired == []


def test_derive_time_capped_context_rounding_and_floor():
    from verallm.mesh.deploy import derive_time_capped_context

    # Cap wins over the KV fit and rounds down to a 1024 multiple.
    cap, registered = derive_time_capped_context(
        measured=120_000,
        samples=[(32_768, 200.0)],
        budget_s=540.0,
        safety_margin=0.75,
    )
    assert (cap, registered) == (66_355, 65_536)
    # A measurement below the floor registers as-is: the registration must
    # never exceed the measured KV fit, floor or not.
    _cap, registered = derive_time_capped_context(
        measured=6_000,
        samples=[(6_000, 600.0)],
        budget_s=540.0,
        safety_margin=0.75,
    )
    assert registered == 6_000


def test_derive_time_capped_context_two_point_superlinear_fit():
    """The
    131k inside budget, but prefill attention is superlinear and the mesh
    overran the deadline. Two points fit t(n) = a*n + b*n^2; the cap is
    where the predicted wall time hits the allowance, far below the
    single-point extrapolation."""
    from verallm.mesh.deploy import derive_time_capped_context

    # Ground truth t(n) = 1.5e-3*n + 3.5e-8*n^2:
    # t(32768) = 86.7s, t(65536) = 248.6s, t(131072) = 798s (way over).
    def t(n):
        return 1.5e-3 * n + 3.5e-8 * n * n

    cap, registered = derive_time_capped_context(
        measured=131_072,
        samples=[(32_768, t(32_768)), (65_536, t(65_536))],
        budget_s=540.0,
        safety_margin=0.75,
    )
    # Predicted wall time AT the cap equals the 405s allowance, and the
    # cap sits far below both the KV fit and the linear extrapolation.
    assert abs(t(cap) - 405.0) < 1.0
    assert cap < 100_000
    assert registered == (cap // 1024) * 1024
    # The registered value's predicted wall time is inside the allowance.
    assert t(registered) <= 405.0


def test_derive_time_capped_context_two_point_fallbacks():
    from verallm.mesh.deploy import derive_time_capped_context

    # Sublinear/noisy second point (same wall time at a larger context):
    # negative curvature falls back to the slowest observed linear rate,
    # matching the single-point result.
    cap_two, _ = derive_time_capped_context(
        measured=120_000,
        samples=[(32_768, 200.0), (49_561, 200.0)],
        budget_s=540.0,
        safety_margin=0.75,
    )
    cap_one, _ = derive_time_capped_context(
        measured=120_000,
        samples=[(32_768, 200.0)],
        budget_s=540.0,
        safety_margin=0.75,
    )
    assert cap_two == cap_one == 66_355
    # Strongly superlinear pair (linear term would go negative): pure
    # quadratic through the larger point, cap = n2 * sqrt(allowance/t2).
    cap, _ = derive_time_capped_context(
        measured=200_000,
        samples=[(32_768, 20.0), (65_536, 200.0)],
        budget_s=540.0,
        safety_margin=0.75,
    )
    assert cap == int(65_536 * (405.0 / 200.0) ** 0.5)


def test_index_mismatch_stops_the_mesh(wired, monkeypatch):
    def mismatch(cfg, target, *, private_key, expected_index=None):
        raise LifecycleRefusal(
            "registered at index 1 but the running mesh was launched against "
            "index 0; its signed snapshots are wrong."
        )

    monkeypatch.setattr(
        registration_module, "register_mesh_endpoint", mismatch
    )
    pool = _FakePool()
    report = run_deploy(_config(), call=pool, out=lambda _l: None, sleep=lambda _s: None)
    assert report.failed
    # The chain-bound relaunch (m-test2) is stopped, after the measurement
    # mesh (m-test1) was replaced.
    assert pool.stopped == ["m-test1", "m-test2"]


def test_endpoint_update_runtime_failure_stops_the_replacement_mesh(
    wired, monkeypatch
):
    previous = {"model_id": MODEL_ID, "index": 0}

    def refuse(
        cfg,
        target,
        *,
        private_key,
        expected_index=None,
        previous_registration=None,
    ):
        assert previous_registration is previous
        raise RuntimeError("deactivate after updateEndpoint timed out")

    monkeypatch.setattr(registration_module, "register_mesh_endpoint", refuse)
    pool = _FakePool()
    report = run_deploy(
        _config(previous_registration=previous),
        call=pool,
        out=lambda _line: None,
        sleep=lambda _s: None,
    )

    assert report.failed
    assert pool.stopped == ["m-test1", "m-test2"]


def test_registration_state_failure_requires_idempotent_redeploy(
    wired,
):
    pool = _FakePool()

    def call(route, body):
        if route == "/v1/pool/registration-state":
            raise RuntimeError("manager write unavailable")
        return pool(route, body)

    report = run_deploy(
        _config(), call=call, out=lambda _line: None, sleep=lambda _s: None
    )

    assert report.failed
    assert report.stages[-1]["name"] == "renewal-state"
    assert "re-run the same `verathos mesh deploy`" in report.stages[-1][
        "detail"
    ]
    # The verified chain-bound mesh remains serving; the command fails so the
    # operator must rerun and persist renewal state rather than editing state.
    assert pool.stopped == ["m-test1"]


def test_dev_pool_refused(wired):
    pool = _FakePool()
    pool.serving_mode = "dev"
    report = run_deploy(_config(), call=pool, out=lambda _l: None, sleep=lambda _s: None)
    assert report.failed
    assert "subnet" in report.stages[-1]["detail"]
    assert wired == []


def test_legacy_validator_mode_string_is_accepted(wired):
    pool = _FakePool()
    pool.serving_mode = "validator"
    report = run_deploy(_config(), call=pool, out=lambda _l: None, sleep=lambda _s: None)
    assert not report.failed
    assert "registerModel" in wired


def test_hotkey_not_matching_uid_fails_before_any_write(wired, monkeypatch):
    monkeypatch.setattr(
        registration_module,
        "resolve_uid_for_hotkey",
        lambda network, netuid, ss58: 7,
    )
    pool = _FakePool()
    report = run_deploy(
        _config(
            coordinator_hotkey_ss58="5Hotkey",
            subtensor_network="test",
            uid=2,
        ),
        call=pool,
        out=lambda _l: None,
        sleep=lambda _s: None,
    )
    assert report.failed
    assert report.stages[-1]["name"] == "hotkey-binding"
    assert wired == []


def test_unregistered_hotkey_fails_before_any_write(wired, monkeypatch):
    def refuse(network, netuid, ss58):
        raise LifecycleRefusal(
            f"hotkey {ss58} is not registered on netuid {netuid} of {network!r}"
        )

    monkeypatch.setattr(
        registration_module, "resolve_uid_for_hotkey", refuse
    )
    pool = _FakePool()
    report = run_deploy(
        _config(coordinator_hotkey_ss58="5Hotkey", subtensor_network="test"),
        call=pool,
        out=lambda _l: None,
        sleep=lambda _s: None,
    )
    assert report.failed
    assert "not registered on netuid" in report.stages[-1]["detail"]
    assert wired == []


def test_mesh_error_during_launch_aborts(wired):
    pool = _FakePool()
    original = pool.__call__

    def call(route, body):
        result = original(route, body)
        if route == "/v1/pool/status" and pool.launch_count:
            for mesh in result["meshes"].values():
                mesh["status"] = "error"
                mesh["error"] = "backend self-test failed"
                mesh["routing_ready"] = False
        return result

    report = run_deploy(_config(), call=call, out=lambda _l: None, sleep=lambda _s: None)
    assert report.failed
    assert "backend self-test failed" in report.stages[-1]["detail"]
    assert "registerModel" not in wired


def _slow_fetch_pool(pool, now, serve_at, pct):
    """Wrap _FakePool: the launched mesh stays in a fetch phase until the
    fake clock reaches ``serve_at``; the driver reports ``pct()``."""
    original = pool.__call__

    def call(route, body):
        result = original(route, body)
        if (
            route == "/v1/pool/status"
            and pool.launch_count
            and now["t"] < serve_at
        ):
            for mesh in result["meshes"].values():
                mesh["status"] = "fetching"
                mesh["routing_ready"] = False
            result["workers"]["pod-gpu0"]["status"] = f"fetching {pct()}%"
        return result

    return call


def test_launch_wait_extends_while_fetch_reports_progress(wired):
    """launch_wait_s bounds time WITHOUT progress, not the whole launch: a
    200GB first-time fetch on a slow pipe takes hours and must not fail
    mid-download while the percentage is still moving (observed on a
    100Mbit box: 45 min timeout at 14% of a 239GB model, three deploy
    re-runs to crawl through one download)."""
    now = {"t": 0.0}

    def sleep(_s):
        now["t"] += 3.0

    pool = _FakePool()
    # Serving at t=60 with launch_wait_s=10: an absolute deadline would
    # have failed at t=10, long before the fetch finishes.
    call = _slow_fetch_pool(pool, now, serve_at=60.0, pct=lambda: int(now["t"]))
    report = run_deploy(
        _config(),
        call=call,
        out=lambda _l: None,
        sleep=sleep,
        clock=lambda: now["t"],
    )
    assert not report.failed
    assert "registerModel" in wired


def test_launch_wait_fails_on_stalled_fetch(wired):
    """No state change for launch_wait_s is still a hard failure: a hung
    download must not wait forever."""
    now = {"t": 0.0}

    def sleep(_s):
        now["t"] += 3.0

    lines: list[str] = []
    pool = _FakePool()
    call = _slow_fetch_pool(pool, now, serve_at=9_999.0, pct=lambda: 14)
    report = run_deploy(
        _config(),
        call=call,
        out=lines.append,
        sleep=sleep,
        clock=lambda: now["t"],
    )
    assert report.failed
    assert any("no launch progress" in line for line in lines)
    assert "registerModel" not in wired


def test_unreachable_endpoint_aborts_before_the_gate(monkeypatch, wired):
    """The measurement mesh binds the same public port the registration
    will claim, so a dead endpoint is knowable BEFORE the probe gate.
    Aborting there saves the whole gate + chain-bound relaunch (~30 min
    live) and, like the final posture check, is never forceable."""
    from verallm.mesh.probe import GateCheck

    dead = [
        GateCheck(
            name="public-health",
            kind="hard",
            passed=False,
            observed="connection refused",
            threshold="HTTP 200 from /health",
            rationale="",
        )
    ]
    monkeypatch.setattr(
        deploy_module, "check_public_endpoint", lambda endpoint, **kw: dead
    )
    gate_ran: list[bool] = []
    monkeypatch.setattr(
        deploy_module,
        "run_probe_gate",
        lambda *a, **kw: gate_ran.append(True),
    )
    pool = _FakePool()
    report = run_deploy(
        _config(force=True), call=pool, out=lambda _l: None, sleep=lambda _s: None
    )
    assert report.failed
    assert report.stages[-1]["name"] == "endpoint-reachability"
    assert gate_ran == []
    assert "registerModel" not in wired


def _bound_mesh(**overrides) -> dict:
    anchors = _anchors()
    mesh = {
        "model_index": 0,
        "verification_snapshot_hash": SNAPSHOT,
        "model_package_hash": anchors.model_package_hash,
        "model_tensor_manifest_root": anchors.model_tensor_manifest_root,
        "tokenizer_hash": anchors.tokenizer_hash,
        "quantization_scheme": anchors.chain_quantization_scheme,
    }
    mesh.update(overrides)
    return mesh


def test_already_bound_requires_current_chain_anchors():
    anchors = _anchors()
    assert deploy_module.mesh_already_bound(_bound_mesh(), 0, anchors)
    # Wrong index or missing snapshot: never bound.
    assert not deploy_module.mesh_already_bound(_bound_mesh(model_index=1), 0, anchors)
    assert not deploy_module.mesh_already_bound(
        _bound_mesh(verification_snapshot_hash=""), 0, anchors
    )
    # Any stale anchor (for example a restamped tokenizer hash on the chain
    # ModelSpec) forces a relaunch so the snapshot re-signs current anchors.
    assert not deploy_module.mesh_already_bound(
        _bound_mesh(tokenizer_hash="ff" * 32), 0, anchors
    )
    assert not deploy_module.mesh_already_bound(
        _bound_mesh(model_package_hash="ff" * 32), 0, anchors
    )
    assert not deploy_module.mesh_already_bound(
        _bound_mesh(quantization_scheme="gguf_q2_k"), 0, anchors
    )


def test_deploy_adopts_a_mid_launch_mesh(wired):
    """A deploy killed between launching and its chain write leaves a
    mesh mid-launch holding the workers; the rerun must adopt it (wait
    for serving, measure against it) instead of launching a second mesh
    into 'worker busy' ."""
    pool = _FakePool()
    pool.launch_count = 1  # a prior launch already measured the context
    pool.meshes = {
        "m-orphan": {
            "model_id": MODEL_ID,
            "driver": "pod-gpu0",
            "members": ["pod-gpu0", "pod-gpu1"],
            "status": "driving",
            "routing_ready": False,
        }
    }
    status_polls = {"n": 0}
    original = pool.__call__

    def call(route, body):
        result = original(route, body)
        if route == "/v1/pool/status" and "m-orphan" in pool.meshes:
            status_polls["n"] += 1
            if status_polls["n"] >= 3:
                pool.meshes["m-orphan"]["status"] = "serving"
                pool.meshes["m-orphan"]["routing_ready"] = True
        return result

    report = run_deploy(
        _config(), call=call, out=lambda _l: None, sleep=lambda _s: None
    )
    assert not report.failed
    # The orphan was adopted for measurement (no fresh measurement
    # launch); the only launch is the chain-bound relaunch after the
    # gate, which first stops the adopted mesh.
    launches = [b for r, b in pool.calls if r == "/v1/pool/launch"]
    assert len(launches) == 1
    assert launches[0].get("pending_binding_ok") is True
    assert "m-orphan" in pool.stopped
    assert wired == ["registerEvm", "registerModel"]


def test_deploy_clears_a_dead_mesh_before_launching(wired):
    """A mesh in error/stopping state still pins its workers; the deploy
    stops it and waits for idle workers before the measurement launch,
    instead of failing with 'worker busy'."""
    pool = _FakePool()
    pool.meshes = {
        "m-dead": {
            "model_id": MODEL_ID,
            "driver": "pod-gpu0",
            "members": ["pod-gpu0", "pod-gpu1"],
            "status": "error",
            "routing_ready": False,
        }
    }
    report = run_deploy(
        _config(), call=pool, out=lambda _l: None, sleep=lambda _s: None
    )
    assert not report.failed
    assert pool.stopped[0] == "m-dead"
    assert report.mesh_key != "m-dead"
    assert wired == ["registerEvm", "registerModel"]


def test_stock_endpoint_replacement_recalibrates_registered_context(
    wired, monkeypatch
):
    """Index reuse never pins a replacement host to the old context."""

    pool = _FakePool()
    pool.launch_count = 1
    pool.meshes = {
        "m-existing": {
            "model_id": MODEL_ID,
            "driver": "pod-gpu0",
            "members": ["pod-gpu0", "pod-gpu1"],
            "status": "serving",
            "routing_ready": True,
            "verification_snapshot_hash": SNAPSHOT,
            "model_index": 0,
        }
    }
    pool.registered_model = {
        "model_id": MODEL_ID,
        "model_index": 0,
        "max_context_len": 16_384,
        "model_package_hash": "01" * 32,
        "model_tensor_manifest_root": "02" * 32,
        "tokenizer_hash": "03" * 32,
        "quantization_scheme": "gguf_q4_k_m",
    }
    gated_contexts: list[int] = []

    def gate(**kwargs):
        config = kwargs["config"]
        gated_contexts.append(config.max_context_len)
        return _passing_gate()

    monkeypatch.setattr(deploy_module, "run_probe_gate", gate)

    planned_contexts: list[int] = []

    def plan(_cfg, target, _signer, **_kwargs):
        planned_contexts.append(target.max_context_len)
        return LifecyclePlan(
            action="update-endpoint-refresh",
            predicted_index=0,
            reason="same-index replacement with refreshed context",
        )

    def register(_cfg, target, **_kwargs):
        assert target.max_context_len == MEASURED_ROUNDED
        return registration_module.RegistrationOutcome(
            action="update-endpoint",
            index=0,
            tx_hash="0x" + "ab" * 32,
            expires_at=int(time.time()) + 86_400,
        )

    monkeypatch.setattr(registration_module, "plan_mesh_registration", plan)
    monkeypatch.setattr(registration_module, "register_mesh_endpoint", register)

    previous = {
        "model_id": MODEL_ID,
        "max_context_len": 16_384,
        "endpoint": "https://old.example:9443",
        "index": 0,
    }
    report = run_deploy(
        _config(previous_registration=previous),
        call=pool,
        out=lambda _line: None,
        sleep=lambda _s: None,
    )

    assert not report.failed
    assert pool.stopped[0] == "m-existing"
    measurement_launch = next(
        body
        for route, body in pool.calls
        if route == "/v1/pool/launch"
        and body.get("_deploy_measurement_unbound")
    )
    assert measurement_launch["_deploy_measurement_unbound"] is True
    assert gated_contexts == [MEASURED_ROUNDED]
    assert report.measured_ctx_budget == MEASURED
    assert report.registered_context_len == MEASURED_ROUNDED
    assert planned_contexts == [MEASURED_ROUNDED]
    assert pool.registered_model["max_context_len"] == MEASURED_ROUNDED
    final_mesh = next(iter(pool.meshes.values()))
    assert final_mesh["model_index"] == 0
    assert final_mesh["max_context_len"] == MEASURED_ROUNDED


def test_same_endpoint_replacement_deactivates_before_stopping_runtime(
    wired, monkeypatch
):
    pool = _FakePool()
    pool.launch_count = 1
    pool.meshes = {
        "m-existing": {
            "model_id": MODEL_ID,
            "driver": "pod-gpu0",
            "members": ["pod-gpu0", "pod-gpu1"],
            "status": "serving",
            "routing_ready": True,
            "verification_snapshot_hash": SNAPSHOT,
            "model_index": 0,
        }
    }
    pool.registered_model = {
        "model_id": MODEL_ID,
        "model_index": 0,
        "max_context_len": 16_384,
        "model_package_hash": "01" * 32,
        "model_tensor_manifest_root": "02" * 32,
        "tokenizer_hash": "03" * 32,
        "quantization_scheme": "gguf_q4_k_m",
    }
    ordering: list[str] = []

    def deactivate(*_args, **_kwargs):
        assert pool.stopped == []
        ordering.append("deactivate-chain-slot")
        return registration_module.RegistrationOutcome(
            action="deactivate-for-recalibration",
            index=0,
            tx_hash="0x" + "cd" * 32,
            expires_at=int(time.time()) + 86_400,
        )

    original = pool.__call__

    def call(route, body):
        if route == "/v1/pool/stop":
            ordering.append("stop-runtime")
        return original(route, body)

    monkeypatch.setattr(
        registration_module,
        "deactivate_mesh_endpoint_for_recalibration",
        deactivate,
    )
    previous = {
        "model_id": MODEL_ID,
        "max_context_len": 16_384,
        "endpoint": "https://mesh.example:9443",
        "index": 0,
    }

    report = run_deploy(
        _config(previous_registration=previous),
        call=call,
        out=lambda _line: None,
        sleep=lambda _seconds: None,
    )

    assert not report.failed
    assert ordering[:2] == ["deactivate-chain-slot", "stop-runtime"]
    assert any(
        stage["name"] == "measurement-slot" and stage["status"] == "ok"
        for stage in report.stages
    )


def test_same_endpoint_replacement_without_mesh_deactivates_before_launch(
    wired, monkeypatch
):
    pool = _FakePool()
    pool.registered_model = {
        "model_id": MODEL_ID,
        "model_index": 0,
        "max_context_len": 16_384,
        "model_package_hash": "01" * 32,
        "model_tensor_manifest_root": "02" * 32,
        "tokenizer_hash": "03" * 32,
        "quantization_scheme": "gguf_q4_k_m",
    }
    ordering: list[str] = []

    def deactivate(*_args, **_kwargs):
        assert pool.launch_count == 0
        ordering.append("deactivate-chain-slot")
        return registration_module.RegistrationOutcome(
            action="deactivate-for-recalibration",
            index=0,
            tx_hash="0x" + "cd" * 32,
            expires_at=int(time.time()) + 86_400,
        )

    original = pool.__call__

    def call(route, body):
        if route == "/v1/pool/launch":
            ordering.append("launch-runtime")
        return original(route, body)

    monkeypatch.setattr(
        registration_module,
        "deactivate_mesh_endpoint_for_recalibration",
        deactivate,
    )
    previous = {
        "model_id": MODEL_ID,
        "max_context_len": 16_384,
        "endpoint": "https://mesh.example:9443",
        "index": 0,
    }

    report = run_deploy(
        _config(previous_registration=previous),
        call=call,
        out=lambda _line: None,
        sleep=lambda _seconds: None,
    )

    assert not report.failed
    assert ordering[:2] == ["deactivate-chain-slot", "launch-runtime"]
    assert pool.stopped == ["m-test1"]


def test_replacement_refuses_manager_without_renewal_suspension(
    wired,
):
    class LegacyPool(_FakePool):
        def __call__(self, route, body):
            if (
                route == "/v1/pool/registration-state"
                and "suspend_renewal" in body
            ):
                # A rolled-back manager can echo a flag persisted by newer
                # code while not implementing the in-flight coordination.
                return {
                    "status": "ok",
                    "registrations": {
                        MODEL_ID: {
                            "model_id": MODEL_ID,
                            "renewal_suspended": True,
                        }
                    },
                }
            return super().__call__(route, body)

    pool = LegacyPool()
    pool.meshes = {
        "m-existing": {
            "model_id": MODEL_ID,
            "driver": "pod-gpu0",
            "members": ["pod-gpu0", "pod-gpu1"],
            "status": "serving",
            "routing_ready": True,
            "verification_snapshot_hash": SNAPSHOT,
            "model_index": 0,
        }
    }
    previous = {
        "model_id": MODEL_ID,
        "max_context_len": 16_384,
        "endpoint": "https://old.example:9443",
        "index": 0,
    }

    report = run_deploy(
        _config(previous_registration=previous),
        call=pool,
        out=lambda _line: None,
        sleep=lambda _seconds: None,
    )

    assert report.failed
    assert (
        "did not prove lease-renewal quiescence"
        in report.stages[-1]["detail"]
    )
    assert pool.stopped == []
    assert pool.launch_count == 0


def test_replacement_refuses_manager_that_inherits_chain_binding(
    wired, monkeypatch
):
    class LegacyPool(_FakePool):
        def __call__(self, route, body):
            if route == "/v1/pool/launch":
                body = dict(body)
                body.pop("_deploy_measurement_unbound", None)
            return super().__call__(route, body)

    pool = LegacyPool()
    pool.launch_count = 1
    pool.registered_model = {
        "model_id": MODEL_ID,
        "model_index": 0,
        "max_context_len": 16_384,
        "model_package_hash": "01" * 32,
        "model_tensor_manifest_root": "02" * 32,
        "tokenizer_hash": "03" * 32,
        "quantization_scheme": "gguf_q4_k_m",
    }
    pool.meshes = {
        "m-existing": {
            "model_id": MODEL_ID,
            "driver": "pod-gpu0",
            "members": ["pod-gpu0", "pod-gpu1"],
            "status": "serving",
            "routing_ready": True,
            "verification_snapshot_hash": SNAPSHOT,
            "model_index": 0,
        }
    }
    monkeypatch.setattr(
        deploy_module, "run_probe_gate", lambda **_kwargs: _passing_gate()
    )

    report = run_deploy(
        _config(
            previous_registration={
                "model_id": MODEL_ID,
                "endpoint": "https://old.example:9443",
                "max_context_len": 16_384,
                "index": 0,
            }
        ),
        call=pool,
        out=lambda _line: None,
        confirm=lambda _message: True,
        sleep=lambda _seconds: None,
    )

    assert report.failed
    assert any(
        stage["name"] == "measurement"
        and "did not honor" in stage["detail"]
        for stage in report.stages
    )
    assert len(pool.stopped) == 2


def test_replacement_refuses_stale_model_level_measurement(wired, monkeypatch):
    class MissingFreshMeasurementPool(_FakePool):
        def __call__(self, route, body):
            response = super().__call__(route, body)
            if route == "/v1/pool/status":
                for mesh in response.get("meshes", {}).values():
                    mesh.pop("measured_ctx_budget", None)
            return response

    pool = MissingFreshMeasurementPool()
    pool.launch_count = 1
    pool.registered_model = {
        "model_id": MODEL_ID,
        "model_index": 0,
        "max_context_len": 16_384,
        "measured_ctx_budget": 98_304,
        "model_package_hash": "01" * 32,
        "model_tensor_manifest_root": "02" * 32,
        "tokenizer_hash": "03" * 32,
        "quantization_scheme": "gguf_q4_k_m",
    }
    pool.meshes = {
        "m-existing": {
            "model_id": MODEL_ID,
            "driver": "pod-gpu0",
            "members": ["pod-gpu0", "pod-gpu1"],
            "status": "serving",
            "routing_ready": True,
            "verification_snapshot_hash": SNAPSHOT,
            "model_index": 0,
        }
    }
    monkeypatch.setattr(
        deploy_module, "run_probe_gate", lambda **_kwargs: _passing_gate()
    )

    report = run_deploy(
        _config(
            previous_registration={
                "model_id": MODEL_ID,
                "endpoint": "https://old.example:9443",
                "max_context_len": 16_384,
                "index": 0,
            }
        ),
        call=pool,
        out=lambda _line: None,
        confirm=lambda _message: True,
        sleep=lambda _seconds: None,
    )

    assert report.failed
    assert any(
        stage["name"] == "measurement"
        and "fresh per-mesh" in stage["detail"]
        for stage in report.stages
    )


def test_replacement_decline_keeps_registered_mesh_serving(wired):
    pool = _FakePool()
    pool.launch_count = 1
    pool.meshes = {
        "m-existing": {
            "model_id": MODEL_ID,
            "driver": "pod-gpu0",
            "members": ["pod-gpu0", "pod-gpu1"],
            "status": "serving",
            "routing_ready": True,
            "verification_snapshot_hash": SNAPSHOT,
            "model_index": 0,
        }
    }
    previous = {
        "model_id": MODEL_ID,
        "max_context_len": 16_384,
        "endpoint": "https://old.example:9443",
        "index": 0,
    }

    report = run_deploy(
        _config(previous_registration=previous, assume_yes=False),
        call=pool,
        out=lambda _line: None,
        confirm=lambda _message: False,
        sleep=lambda _s: None,
    )

    assert report.failed
    assert pool.stopped == []
    assert pool.meshes["m-existing"]["status"] == "serving"
    assert not any(route == "/v1/pool/launch" for route, _ in pool.calls)


def test_serving_context_cap_pins_launch_and_clamps_registration(
    wired, monkeypatch
):
    """The KV auto-fit measures memory, not correctness: a runtime bug
    past a position threshold serves degenerate output . The catalogue serving cap pins the launch,
    clamps the derived registration, and refuses explicit overrides
    beyond it."""
    import verallm.registry.models as registry_models

    monkeypatch.setattr(
        registry_models,
        "mesh_model_serving_context_cap",
        lambda model_id: 32_768,
    )
    pool = _FakePool()
    report = run_deploy(
        _config(), call=pool, out=lambda _l: None, sleep=lambda _s: None
    )
    assert not report.failed
    # The measurement launch carried the pin.
    launches = [b for r, b in pool.calls if r == "/v1/pool/launch"]
    assert launches[0].get("max_context_len") == 32_768
    # The fake pool reports MEASURED=120000 regardless (a reused mesh
    # measured unpinned); the registration is clamped to the cap.
    assert report.registered_context_len == 32_768
    assert pool.registered_model["max_context_len"] == 32_768

    # An explicit override beyond the cap is refused outright.
    report = run_deploy(
        _config(max_context_len=120_000),
        call=_FakePool(),
        out=lambda _l: None,
        sleep=lambda _s: None,
    )
    assert report.failed
    assert report.stages[-1]["name"] == "context"
    assert "serving cap" in report.stages[-1]["detail"]


def test_verification_probe_failure_bisects_the_servable_ceiling(wired):
    """A verification-class probe failure is a model+runtime context cliff
    . The deploy must register the
    largest window that VERIFIES instead of erroring out, and transient
    errors must never shrink the window (covered by the test above)."""
    pool = _FakePool()
    original = pool.__call__
    probed = []

    def call(route, body):
        if route == "/v1/pool/chat" and body.get("probe"):
            content = str((body.get("messages") or [{}])[0].get("content", ""))
            words = content.split()
            approx_ctx = int(len(words) / 0.8)
            probed.append(approx_ctx)
            if approx_ctx > 40_000:
                return {
                    "status": "error",
                    "error": (
                        "decode audit verification failed: decode audit "
                        "token is not the committed f32 argmax"
                    ),
                }
        return original(route, body)

    report = run_deploy(
        _config(), call=call, out=lambda _l: None, sleep=lambda _s: None
    )
    # The registration must sit at or below the cliff, never above it.
    assert report.registered_context_len > 0
    assert report.registered_context_len <= 40_960
    # And the bisect actually probed between the pass and the failure.
    assert any(32_768 < ctx <= 65_536 for ctx in probed)


def test_context_refresh_forces_chain_bound_relaunch():
    """An in-place registration refresh that changes only the registered
    context must NOT be treated as already-bound: the serving mesh's
    snapshot binds the old value and every validator pin fails 'snapshot
    model does not match expected model' until relaunch ."""

    from types import SimpleNamespace

    from verallm.mesh.deploy import mesh_already_bound

    anchors = SimpleNamespace(
        model_package_hash="p" * 8,
        model_tensor_manifest_root="r" * 8,
        tokenizer_hash="t" * 8,
        chain_quantization_scheme="gguf_mesh_iq2_m",
    )
    mesh = {
        "model_index": 62,
        "verification_snapshot_hash": "s" * 12,
        "model_package_hash": "p" * 8,
        "model_tensor_manifest_root": "r" * 8,
        "tokenizer_hash": "t" * 8,
        "quantization_scheme": "gguf_mesh_iq2_m",
        "max_context_len": 98304,
    }
    assert mesh_already_bound(mesh, 62, anchors, registered_context=98304)
    assert not mesh_already_bound(mesh, 62, anchors, registered_context=59392)
