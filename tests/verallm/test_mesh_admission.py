"""Uniform token-ledger admission (verallm/mesh/admission.py).

The unified llama KV is one shared pool; the ledger refuses what cannot
fit RIGHT NOW with an instant 503 (no queue, owner decision) before any
nonce is claimed, identically for every caller: canaries are
byte-indistinguishable from organic traffic and any lane-aware treatment
would be a cheating oracle.
"""

from __future__ import annotations

import threading

from verallm.mesh.admission import (
    Admission,
    KVAdmissionLedger,
    ReservationGuard,
)


def test_token_and_slot_double_gating():
    ledger = KVAdmissionLedger(capacity_tokens=100, slots=2)
    assert ledger.try_admit(60) is Admission.ADMITTED
    # Token-blocked despite a free slot.
    assert ledger.try_admit(60) is Admission.BUSY
    assert ledger.try_admit(40) is Admission.ADMITTED
    # Slot-blocked despite free tokens.
    assert ledger.try_admit(1) is Admission.BUSY
    ledger.release(60)
    assert ledger.try_admit(10) is Admission.ADMITTED


def test_oversized_is_not_busy():
    """A demand above the per-request contract can NEVER be served under
    the contract: OVERSIZED, and the route 400s it uniformly (the
    measured-fit pool exceeds the contract, so llama itself would happily
    serve it - unaccounted)."""
    ledger = KVAdmissionLedger(
        capacity_tokens=1000, slots=4, per_request_cap=100
    )
    assert ledger.try_admit(101) is Admission.OVERSIZED
    assert ledger.try_admit(100) is Admission.ADMITTED
    # OVERSIZED consumed nothing.
    snap = ledger.snapshot()
    assert snap["in_flight_tokens"] == 100
    assert snap["in_flight_slots"] == 1


def test_release_pairs_and_never_goes_negative():
    ledger = KVAdmissionLedger(capacity_tokens=50, slots=1)
    assert ledger.try_admit(50) is Admission.ADMITTED
    ledger.release(50)
    ledger.release(50)  # double release must not corrupt the pool
    snap = ledger.snapshot()
    assert snap["in_flight_tokens"] == 0
    assert snap["in_flight_slots"] == 0
    assert ledger.try_admit(50) is Admission.ADMITTED


def test_default_per_request_cap_is_capacity():
    ledger = KVAdmissionLedger(capacity_tokens=64, slots=8)
    assert ledger.per_request_cap == 64
    assert ledger.try_admit(65) is Admission.OVERSIZED


def test_thread_hammer_invariants():
    """Concurrent admit/release never exceeds capacity or slot count and
    always drains back to zero."""
    ledger = KVAdmissionLedger(capacity_tokens=1_000, slots=8)
    violations: list[str] = []

    def worker() -> None:
        for _ in range(300):
            if ledger.try_admit(100) is Admission.ADMITTED:
                snap = ledger.snapshot()
                if snap["in_flight_tokens"] > snap["capacity_tokens"]:
                    violations.append("tokens over capacity")
                if snap["in_flight_slots"] > snap["slots"]:
                    violations.append("slots over count")
                ledger.release(100)

    threads = [threading.Thread(target=worker) for _ in range(16)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not violations
    snap = ledger.snapshot()
    assert snap["in_flight_tokens"] == 0
    assert snap["in_flight_slots"] == 0


def test_release_is_reservation_matched_never_steals_a_peer():
    """A double release must not return a DIFFERENT thread's in-flight
    reservation to the pool ."""

    ledger = KVAdmissionLedger(capacity_tokens=200, slots=4)
    assert ledger.try_admit(50) is Admission.ADMITTED

    admitted = threading.Event()
    done = threading.Event()

    def peer() -> None:
        assert ledger.try_admit(50) is Admission.ADMITTED
        admitted.set()
        done.wait(10.0)
        ledger.release(50)

    thread = threading.Thread(target=peer)
    thread.start()
    try:
        assert admitted.wait(5.0)
        ledger.release(50)  # pairs the main thread's admission
        ledger.release(50)  # double release: must NOT touch the peer's
        snap = ledger.snapshot()
        assert snap["in_flight_slots"] == 1
        assert snap["in_flight_tokens"] == 50
    finally:
        done.set()
        thread.join(timeout=5)
    snap = ledger.snapshot()
    assert snap["in_flight_slots"] == 0
    assert snap["in_flight_tokens"] == 0


def test_reservation_guard_releases_exactly_once():
    ledger = KVAdmissionLedger(capacity_tokens=100, slots=2)
    assert ledger.try_admit(60) is Admission.ADMITTED
    guard = ReservationGuard(ledger, 60)
    # Every exit path calls release defensively; only the first lands.
    assert guard.release() is True
    assert guard.release() is False
    assert guard.release() is False
    snap = ledger.snapshot()
    assert snap["in_flight_slots"] == 0
    assert snap["in_flight_tokens"] == 0
    # And the pool is intact for the next request.
    assert ledger.try_admit(100) is Admission.ADMITTED


def test_reap_dead_owners_releases_and_reports_the_leak():
    """A reservation whose owning thread died without releasing (the
    wedge class that permanently shrank the pool until relaunch) is
    returned to the pool by the sweep, with diagnostics for the log."""

    ledger = KVAdmissionLedger(capacity_tokens=100, slots=1)

    def leaky() -> None:
        assert ledger.try_admit(70) is Admission.ADMITTED
        # Thread exits WITHOUT releasing.

    thread = threading.Thread(target=leaky, name="leaky-handler")
    thread.start()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert ledger.try_admit(10) is Admission.BUSY  # slot leaked

    reaped = ledger.reap_dead_owners()
    assert len(reaped) == 1
    assert reaped[0]["tokens"] == 70
    assert reaped[0]["owner"] == "leaky-handler"
    assert reaped[0]["held_seconds"] >= 0
    # Sweep is idempotent and the pool is whole again.
    assert ledger.reap_dead_owners() == []
    assert ledger.try_admit(10) is Admission.ADMITTED


def test_reap_dead_owners_never_touches_live_reservations():
    ledger = KVAdmissionLedger(capacity_tokens=100, slots=2)
    assert ledger.try_admit(40) is Admission.ADMITTED  # this thread lives
    assert ledger.reap_dead_owners() == []
    snap = ledger.snapshot()
    assert snap["in_flight_slots"] == 1
    assert snap["in_flight_tokens"] == 40
    ledger.release(40)


def test_release_after_sweep_race_is_a_noop():
    """The wedged thread may eventually unwind (socket timeout) after the
    sweep already reclaimed its dead-owner reservation; its late release
    must not underflow another request's accounting."""

    ledger = KVAdmissionLedger(capacity_tokens=100, slots=2)

    class _DeadThread(threading.Thread):
        pass

    dead = _DeadThread(name="already-dead")
    dead.start()
    dead.join(timeout=5)
    assert ledger.try_admit(30, owner=dead) is Admission.ADMITTED
    assert len(ledger.reap_dead_owners()) == 1
    # Late release from the (revived-by-timeout) owner: nothing matches.
    ledger.release(30, owner=dead)
    assert ledger.try_admit(20) is Admission.ADMITTED
    snap = ledger.snapshot()
    assert snap["in_flight_slots"] == 1
    assert snap["in_flight_tokens"] == 20
    ledger.release(20)


def test_snapshot_reports_owner_diagnostics():
    ledger = KVAdmissionLedger(capacity_tokens=100, slots=2)
    assert ledger.try_admit(25) is Admission.ADMITTED
    snap = ledger.snapshot()
    assert len(snap["reservations"]) == 1
    entry = snap["reservations"][0]
    assert entry["tokens"] == 25
    assert entry["owner"] == threading.current_thread().name
    assert entry["owner_alive"] is True
    assert entry["held_seconds"] >= 0
    ledger.release(25)
    assert ledger.snapshot()["reservations"] == []


def test_live_overlap_scenario_two_full_context_canaries():
    """With a registered contract of 98304 and fitted budget of 389120,
    two 88k canaries coexist; a third is refused
    instantly instead of llama erroring all streams mid-flight."""
    ledger = KVAdmissionLedger(
        capacity_tokens=196_608, slots=8, per_request_cap=98_304
    )
    first = 88_425 + 96
    second = 88_430 + 96
    assert ledger.try_admit(first) is Admission.ADMITTED
    assert ledger.try_admit(second) is Admission.ADMITTED
    # A third giant does not fit the remaining pool: clean busy.
    assert ledger.try_admit(98_304) is Admission.BUSY
    # Small organic traffic still flows beside the two giants.
    assert ledger.try_admit(4_000) is Admission.ADMITTED
    # On the MEASURED-fit pool (389k) even a third giant coexists.
    big = KVAdmissionLedger(
        capacity_tokens=389_120, slots=8, per_request_cap=98_304
    )
    for demand in (first, second, 98_304):
        assert big.try_admit(demand) is Admission.ADMITTED
