"""Runbook check-5 scenarios driven through the real mesh canary path.

Every unit exercised here already passes in isolation; what had zero
coverage was the wiring.  ``_execute_mesh_canary`` is the only place the
output-sanity gate, the binding-violation classifier and the EMA
penalties meet, so these tests stub the coordinator at ``run_mesh_canary``
and keep the database, scorer and probation tracker real.

Scenarios, matching ``mesh_hardening_e2e_runbook.md`` section 5:

- degenerate output from a coordinator whose proof verifies is struck
- a plausible output with a verifying proof passes untouched
- a postcommit abort is a binding violation and zeroes the EMA
- an ordinary proof failure halves the EMA instead
- probation synced from the database survives slot re-registration
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

import neurons.mesh_verify as mesh_verify
from neurons.canary import CanaryTest
from neurons.mesh_verify import MeshCanaryResult
from neurons.scoring import (
    CompositeScorer,
    MinerScoreState,
    ModelEntryScore,
    ProbationTracker,
)
from neurons.validator import ValidatorNeuron
from neurons.validator_db import ValidatorStateDB

ADDR = "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
ENDPOINT = "https://coordinator.example"
MODEL_ID = "test/mesh-model"
MODEL_INDEX = 7
EPOCH = 12
UID = 1


def _canary() -> CanaryTest:
    return CanaryTest(
        miner_address=ADDR,
        miner_endpoint=ENDPOINT,
        model_id=MODEL_ID,
        model_index=MODEL_INDEX,
        max_context_len=32768,
        target_block=1,
        test_index=0,
        test_type="small",
        prompt="write a short sentence about the sea",
        max_new_tokens=128,
        temperature=0.0,
        verify_proof=True,
        verify_mesh=True,
    )


def _result(**overrides) -> MeshCanaryResult:
    base = dict(
        ok=True,
        reason="",
        full_text="The sea is calm tonight, the tide is full.",
        input_tokens=9,
        output_tokens=12,
        ttft_ms=80.0,
        inference_ms=900.0,
        phase_one_start_ts=1_700_000_000.0,
        phase_one_end_ts=1_700_000_001.0,
        total_ms=1500.0,
        receipt={"receipt_hash": "ab" * 32},
        artifact={},
        openai_request={},
    )
    base.update(overrides)
    return MeshCanaryResult(**base)


@pytest.fixture
def neuron(tmp_path, monkeypatch):
    db = ValidatorStateDB(db_path=os.path.join(str(tmp_path), "validator.db"))
    db.upsert_entry(ADDR, MODEL_INDEX, MODEL_ID, ENDPOINT, "", 4096, epoch=1)
    db.save_score(ADDR, MODEL_INDEX, 8.0, 20, 15)

    n = ValidatorNeuron.__new__(ValidatorNeuron)
    n._db = db
    n._running = True
    n._current_epoch = EPOCH
    n._expected_receipts = {}
    n._inflight_canaries = {}
    n._closing_inflight_canaries = {}
    n._last_known_block = 100
    n._validator_hotkey_ss58 = "validator-hotkey"
    n._validator_private_key = b"\x01" * 32
    n.config = SimpleNamespace(
        canary_inference_timeout=30,
        canary_full_context_inference_timeout=60,
        subtensor_network="test",
        chain_id=0,
        netuid=405,
    )
    n.scorer = CompositeScorer()
    n.scorer.states[UID] = MinerScoreState(
        uid=UID,
        address=ADDR,
        entries={
            MODEL_INDEX: ModelEntryScore(
                model_id=MODEL_ID, model_index=MODEL_INDEX, ema_score=0.8,
            )
        },
    )
    n._probation_tracker = ProbationTracker(
        state_path=os.path.join(str(tmp_path), "probation.json"),
    )
    n._mesh_snapshot_cache = {
        n._mesh_snapshot_cache_key(ADDR, MODEL_INDEX, EPOCH): object()
    }
    n._epoch_miners = [
        SimpleNamespace(
            address=ADDR,
            model_index=MODEL_INDEX,
            endpoint=ENDPOINT,
            model_id=MODEL_ID,
        )
    ]
    n._mesh_snapshot_trust_anchors = lambda miner, epoch_number: SimpleNamespace(
        coordinator="coordinator-hotkey",
    )
    n._maintenance_grace_active = lambda *args, **kwargs: False
    n._get_miner_ss58 = lambda address, key_type="hotkey": ""
    n._write_shared_state = lambda: None

    pushed = []

    def _push(**kwargs):
        pushed.append(kwargs)
        return True

    n._push_receipt_to_miner = _push
    n._pushed_receipts = pushed

    yield n
    db.close()


def _run(neuron, monkeypatch, result: MeshCanaryResult) -> None:
    monkeypatch.setattr(
        mesh_verify, "run_mesh_canary", lambda **kwargs: result,
    )
    neuron._execute_mesh_canary(_canary(), EPOCH)


def _db_ema(neuron) -> float:
    return neuron._db.load_all_scores()[(ADDR, MODEL_INDEX)]["ema_score"]


def test_degenerate_output_with_verifying_proof_is_struck(neuron, monkeypatch):
    _run(
        neuron,
        monkeypatch,
        _result(ok=True, output_tokens=120, full_text=" \n\t" * 40),
    )

    assert neuron._db.is_on_probation(ADDR, MODEL_INDEX) is True
    assert neuron._probation_tracker.is_on_probation((ADDR, MODEL_INDEX))
    assert neuron.scorer.states[UID].entries[MODEL_INDEX].ema_score == 0.4
    assert _db_ema(neuron) == 4.0
    assert len(neuron._pushed_receipts) == 1
    assert neuron._pushed_receipts[0]["proof_verified"] is False


def test_plausible_output_with_verifying_proof_passes(neuron, monkeypatch):
    _run(neuron, monkeypatch, _result())

    assert neuron._db.is_on_probation(ADDR, MODEL_INDEX) is False
    assert not neuron._probation_tracker.is_on_probation((ADDR, MODEL_INDEX))
    assert neuron.scorer.states[UID].entries[MODEL_INDEX].ema_score == 0.8
    assert _db_ema(neuron) == 8.0
    assert len(neuron._pushed_receipts) == 1
    assert neuron._pushed_receipts[0]["proof_verified"] is True


def test_postcommit_abort_is_a_binding_violation_and_zeroes_the_ema(
    neuron, monkeypatch,
):
    _run(
        neuron,
        monkeypatch,
        _result(
            ok=False,
            transport_error=True,
            transport_phase="postcommit",
            transport_status_code=None,
            reason="postcommit request failed: connection reset",
        ),
    )

    assert neuron._db.is_on_probation(ADDR, MODEL_INDEX) is True
    assert neuron.scorer.states[UID].entries[MODEL_INDEX].ema_score == 0.0
    assert _db_ema(neuron) == 0.0
    # The abort still produces a receipt row; it must be a failed one.
    assert len(neuron._pushed_receipts) == 1
    assert neuron._pushed_receipts[0]["proof_verified"] is False


def test_ordinary_proof_failure_halves_the_ema_instead(neuron, monkeypatch):
    _run(
        neuron,
        monkeypatch,
        _result(ok=False, reason="stage boundary root mismatch at stage 1"),
    )

    assert neuron._db.is_on_probation(ADDR, MODEL_INDEX) is True
    assert neuron.scorer.states[UID].entries[MODEL_INDEX].ema_score == 0.4
    assert _db_ema(neuron) == 4.0


def test_probation_synced_from_db_survives_slot_re_registration(neuron):
    db = neuron._db
    db.enter_probation(ADDR, MODEL_INDEX, epoch=3)
    # Re-register the same operator on a brand-new slot with a renamed
    # endpoint: the slot inherits the live probation in the database, and
    # the in-memory tracker (which has never seen the new slot) must
    # follow the database rather than treat the slot as clean.
    new_index = MODEL_INDEX + 1
    db.upsert_entry(
        ADDR, new_index, MODEL_ID, "https://renamed.example", "", 4096, epoch=4,
    )
    assert db.is_on_probation(ADDR, new_index) is True

    assert not neuron._probation_tracker.is_on_probation((ADDR, new_index))
    neuron._sync_probation_tracker_from_db()
    assert neuron._probation_tracker.is_on_probation((ADDR, new_index))
