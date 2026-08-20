"""Uniform token-ledger admission for the mesh coordinator.

llama.cpp's unified KV (``--kv-unified``) is ONE shared token pool that
``--parallel`` slots draw from; llama admits requests into slots without
reserving context and then errors the stream when the pool runs dry
("Context size has been exceeded" -
88k-token validator canaries died mid-flight as unretryable 500s and cost
the epoch). The coordinator knows every request's demand before
forwarding (prompt tokens via the backend tokenizer + ``max_tokens``), so
it can refuse what cannot fit BEFORE llama breaks it.

Design rules (owner-fixed):

* **No queue.** A request that does not fit right now gets an instant,
  clean 503 - the validator's busy machinery reschedules canaries with
  fresh prompts, and the router fails organic traffic over to another
  miner instead of parking users in a line.
* **Uniform.** The ledger never inspects WHO is asking. Canaries are
  byte-indistinguishable from organic streaming traffic by design, and
  any lane-aware treatment (priority, reserved capacity) would be a
  cheating oracle. Every request is admitted or refused by the same
  arithmetic.
* **No reserved capacity.** Honesty about busyness is enforced at epoch
  close instead: a miner's full-context 503 windows are only forgiven
  when signed validator-observed receipts prove real overlapping work.
* **A reservation cannot outlive its thread.** Every admission records
  its owning thread; releases are matched per reservation (idempotent),
  and ``reap_dead_owners`` reconciles anything whose owner died without
  releasing so abandoned requests cannot permanently consume slots.
"""

from __future__ import annotations

import enum
import threading
import time


class Admission(enum.Enum):
    ADMITTED = "admitted"
    BUSY = "busy"
    OVERSIZED = "oversized"


class _Reservation:
    """One admitted in-flight request, bound to the thread that owns it."""

    __slots__ = ("tokens", "owner", "admitted_monotonic")

    def __init__(
        self,
        tokens: int,
        owner: threading.Thread | None,
        admitted_monotonic: float,
    ) -> None:
        self.tokens = tokens
        self.owner = owner
        self.admitted_monotonic = admitted_monotonic


class ReservationGuard:
    """Releases ONE admitted reservation exactly once, from any exit path.

    The handler that admitted a request unwinds through several layers
    (route finally, connection finish) and each calls ``release()``
    defensively; only the first call reaches the ledger. The guard plus
    the ledger's own per-reservation idempotency make a double release
    structurally harmless, so every exit path can release without
    coordination.
    """

    __slots__ = ("tokens", "_ledger", "_owner", "_released", "_lock")

    def __init__(
        self,
        ledger: "KVAdmissionLedger",
        tokens: int,
        *,
        owner: threading.Thread | None = None,
    ) -> None:
        self.tokens = max(0, int(tokens))
        self._ledger = ledger
        self._owner = owner if owner is not None else threading.current_thread()
        self._released = False
        self._lock = threading.Lock()

    def release(self) -> bool:
        """Release the reservation; True only for the call that did it."""

        with self._lock:
            if self._released:
                return False
            self._released = True
        self._ledger.release(self.tokens, owner=self._owner)
        return True


class KVAdmissionLedger:
    """Non-blocking token + slot accounting for one llama backend.

    ``capacity_tokens`` MUST be the backend's FITTED context (llama's
    ``/props`` ``n_ctx``), never the requested budget: the KV auto-fit
    ladder may have descended below the request, and over-admitting
    reintroduces exactly the mid-flight KV errors this ledger removes.

    ``per_request_cap`` is the registered per-request contract. A demand
    above it is not "busy" - it can NEVER be served under the contract -
    and is classified OVERSIZED so the route rejects it with a uniform
    400. The coordinator owns this enforcement: the unified KV is sized
    to the MEASURED fit, which legitimately exceeds the contract, so
    llama itself no longer rejects over-contract prompts - and
    forwarding one unreserved would consume pool tokens the ledger never
    accounted.
    """

    def __init__(
        self,
        *,
        capacity_tokens: int,
        slots: int,
        per_request_cap: int = 0,
    ) -> None:
        self.capacity_tokens = max(0, int(capacity_tokens))
        self.slots = max(1, int(slots))
        cap = int(per_request_cap or 0)
        self.per_request_cap = cap if cap > 0 else self.capacity_tokens
        self._lock = threading.Lock()
        self._in_flight_tokens = 0
        self._in_flight_slots = 0
        self._reservations: dict[int, _Reservation] = {}
        self._next_reservation_id = 0

    def try_admit(
        self,
        demand_tokens: int,
        *,
        owner: threading.Thread | None = None,
    ) -> Admission:
        """Admit, refuse-as-busy, or classify as permanently oversized.

        NON-BLOCKING by design: the caller turns BUSY into an immediate
        503 before any nonce is claimed or SSE byte is sent.

        Every admission records a reservation owned by ``owner`` (the
        calling thread by default) so the dead-owner sweep can reconcile
        a reservation whose thread died without releasing, which would
        otherwise shrink the pool until relaunch.
        """

        demand = max(0, int(demand_tokens))
        if demand > self.per_request_cap:
            return Admission.OVERSIZED
        if owner is None:
            owner = threading.current_thread()
        with self._lock:
            if self._in_flight_slots >= self.slots:
                return Admission.BUSY
            if self._in_flight_tokens + demand > self.capacity_tokens:
                return Admission.BUSY
            self._in_flight_tokens += demand
            self._in_flight_slots += 1
            self._next_reservation_id += 1
            self._reservations[self._next_reservation_id] = _Reservation(
                demand,
                owner,
                time.monotonic(),
            )
            return Admission.ADMITTED

    def configure(
        self,
        *,
        capacity_tokens: int,
        per_request_cap: int = 0,
    ) -> None:
        """Upgrade capacity in place once the backend's real fit is known.

        The ledger exists from the first request (slots-only while llama
        is still loading), and REPLACING the instance on resolution would
        orphan in-flight reservations - releases would land on a fresh
        ledger and the accounting would silently reset (caught by the
        route tests). Upgrading keeps the in-flight counts.
        """

        with self._lock:
            self.capacity_tokens = max(0, int(capacity_tokens))
            cap = int(per_request_cap or 0)
            self.per_request_cap = cap if cap > 0 else self.capacity_tokens

    def release(
        self,
        demand_tokens: int,
        *,
        owner: threading.Thread | None = None,
    ) -> None:
        """Return an admitted reservation; must pair every ADMITTED.

        Idempotent per reservation: the release is matched STRICTLY to a
        recorded in-flight reservation with the same owner and token
        count, and a call with no match - a double release, or a release
        racing the dead-owner sweep - is a no-op. Blind counter
        decrements (the previous behavior) would let a double release
        return ANOTHER thread's in-flight reservation to the pool and
        over-admit into llama's unified KV. A release from the wrong
        thread therefore no-ops too; that reservation is reconciled by
        the dead-owner sweep once its admitting thread exits.
        """

        demand = max(0, int(demand_tokens))
        if owner is None:
            owner = threading.current_thread()
        with self._lock:
            for reservation_id, entry in self._reservations.items():
                if entry.tokens == demand and entry.owner is owner:
                    self._drop_reservation_locked(reservation_id)
                    return

    def reap_dead_owners(self) -> list[dict]:
        """Release reservations whose owning thread is no longer alive.

        Defense in depth: with per-request release guards on every exit
        path this should never find anything, and each hit is a release
        bug that would previously have shrunk the pool until relaunch.
        The caller logs every returned entry loudly.
        """

        now = time.monotonic()
        reaped: list[dict] = []
        with self._lock:
            dead = [
                reservation_id
                for reservation_id, entry in self._reservations.items()
                if isinstance(entry.owner, threading.Thread)
                and not entry.owner.is_alive()
            ]
            for reservation_id in dead:
                entry = self._drop_reservation_locked(reservation_id)
                reaped.append(
                    {
                        "tokens": entry.tokens,
                        "owner": getattr(entry.owner, "name", str(entry.owner)),
                        "held_seconds": round(
                            now - entry.admitted_monotonic, 3
                        ),
                    }
                )
        return reaped

    def _drop_reservation_locked(self, reservation_id: int) -> _Reservation:
        entry = self._reservations.pop(reservation_id)
        self._in_flight_tokens = max(0, self._in_flight_tokens - entry.tokens)
        self._in_flight_slots = max(0, self._in_flight_slots - 1)
        return entry

    def snapshot(self) -> dict:
        """Diagnostic view (tests, status endpoints)."""

        with self._lock:
            now = time.monotonic()
            return {
                "capacity_tokens": self.capacity_tokens,
                "in_flight_tokens": self._in_flight_tokens,
                "slots": self.slots,
                "in_flight_slots": self._in_flight_slots,
                "per_request_cap": self.per_request_cap,
                # Owner diagnostics: who holds each in-flight reservation
                # and for how long, so a wedged serve can be diagnosed
                # from the status route instead of a faulthandler dump.
                "reservations": [
                    {
                        "tokens": entry.tokens,
                        "owner": getattr(entry.owner, "name", str(entry.owner)),
                        "owner_alive": (
                            entry.owner.is_alive()
                            if isinstance(entry.owner, threading.Thread)
                            else None
                        ),
                        "held_seconds": round(
                            now - entry.admitted_monotonic, 3
                        ),
                    }
                    for entry in self._reservations.values()
                ],
            }
