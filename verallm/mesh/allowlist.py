"""Validator allowlist refresh for standalone mesh workers.

The coordinator's validator auth reads ``validators.json`` and REFUSES the
file once its metagraph refresh is older than
``DEFAULT_VALIDATOR_ALLOWLIST_MAX_AGE_SECONDS`` (verallm/mesh/http_auth.py),
so the allowlist cannot be hand-written: something must keep it fresh. On a
vLLM miner that is the miner process itself
(neurons/miner.py _refresh_validator_allowlist). A standalone mesh worker
has no vLLM miner, so ``mesh pool worker`` runs this refresher when it is
given a network and netuid; the file shape is identical to the miner's so
the same auth middleware reads both.
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)

DEFAULT_ALLOWLIST_REFRESH_INTERVAL_S = 5 * 60.0
# Until the first successful write the worker cannot drive at all (the pool
# reports "member only: ... validator allowlist is unavailable"), so a
# startup failure must not cost a full refresh interval.
ALLOWLIST_FIRST_SUCCESS_RETRY_S = 15.0
# Hard wall-clock bound on one refresh attempt; see run_allowlist_refresher.
ALLOWLIST_REFRESH_ATTEMPT_TIMEOUT_S = 120.0


def preimport_chain_modules() -> None:
    """Import the chain stack from the CALLING thread.

    ``bittensor`` pulls in ``scalecodec``, whose module-level import graph is
    not thread-safe against a concurrent import of the same packages: when
    the refresher thread starts while the worker's own startup is still
    importing them, CPython raises
    ``deadlock detected by _ModuleLock('scalecodec.utils.ss58')`` and the
    refresher never writes an allowlist, so placement refuses the worker.
    Importing here, before the thread exists, removes the race entirely;
    a failure is left to the caller's normal error path.
    """

    import bittensor  # noqa: F401


def normalize_subtensor_network(value: str) -> str:
    """Chain endpoints for bittensor clients, scheme-normalized.

    Newer SDK versions only dial ws(s):// endpoints and choke on http(s)://
    with a DNS error (observed: the allowlist refresher hung its first
    refresh forever against ``http://<node>:9944``); older versions accept
    both. Normalize so operators can paste either form.
    """

    network = str(value or "").strip()
    if network.startswith("http://"):
        return "ws://" + network[len("http://"):]
    if network.startswith("https://"):
        return "wss://" + network[len("https://"):]
    return network


def refresh_validator_allowlist(
    *,
    subtensor_network: str,
    netuid: int,
    out_path: str | Path,
    chain_config_path: str = "",
    allow_extra: Iterable[str] = (),
) -> int:
    """Write the current validator hotkeys for a subnet to ``out_path``.

    Mirrors the vLLM miner's refresh: metagraph validators with a permit
    and at least the on-chain ``minValidatorStake`` (skipped when no chain
    config is available), plus any manually allowed hotkeys. Returns the
    number of validators written. Raises on metagraph failure so the
    caller decides whether staleness is fatal.
    """
    import bittensor as bt

    subtensor = bt.Subtensor(
        network=normalize_subtensor_network(subtensor_network)
    )
    metagraph = subtensor.metagraph(int(netuid))

    min_validator_stake = 0.0
    if chain_config_path:
        try:
            from verallm.chain.config import ChainConfig
            from verallm.chain.validator_registry import ValidatorRegistryClient

            chain_config = ChainConfig.from_json(chain_config_path)
            registry = ValidatorRegistryClient(chain_config)
            min_validator_stake = registry.get_min_validator_stake() / 1e9
        except Exception as exc:
            logger.debug("minValidatorStake unavailable: %s", exc)

    validators: list[dict[str, object]] = []
    count = int(metagraph.n.item())
    for uid in range(count):
        has_permit = hasattr(metagraph, "validator_permit") and bool(
            metagraph.validator_permit[uid]
        )
        stake = float(
            metagraph.S[uid]
            if hasattr(metagraph, "S")
            else metagraph.stake[uid]
        )
        if has_permit and stake >= min_validator_stake:
            validators.append(
                {
                    "uid": uid,
                    "hotkey_ss58": metagraph.hotkeys[uid],
                    "stake": stake,
                }
            )

    existing = {str(v["hotkey_ss58"]) for v in validators}
    for ss58 in allow_extra:
        ss58 = str(ss58).strip()
        if ss58 and ss58 not in existing:
            validators.append({"uid": -1, "hotkey_ss58": ss58, "stake": 0})
            existing.add(ss58)

    out = Path(out_path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated_at": int(time.time()),
        "netuid": int(netuid),
        "validators": validators,
    }
    tmp_path = str(out) + ".tmp"
    with open(tmp_path, "w") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(tmp_path, out)
    return len(validators)


def run_allowlist_refresher(
    *,
    subtensor_network: str,
    netuid: int,
    out_path: str | Path,
    chain_config_path: str = "",
    allow_extra: Iterable[str] = (),
    interval_s: float = DEFAULT_ALLOWLIST_REFRESH_INTERVAL_S,
    stop_event=None,
) -> None:
    """Daemon loop for the pool worker process. Errors are logged, never
    fatal: a transient RPC failure must not kill the worker, and the next
    tick retries well inside the allowlist freshness window."""
    import threading

    waiter = stop_event if stop_event is not None else threading.Event()
    wrote_once = False
    while not waiter.is_set():
        # HARD per-attempt deadline: the substrate websocket has no read
        # timeout, so a starved RPC node hung one refresh call forever -
        # no error, no retry, and 74 minutes later the coordinator's
        # fail-closed freshness wall 503'd every validator route and
        # cost a probation reset. A hung attempt is abandoned (its
        # daemon thread dies with the TCP session); abandonment is
        # bounded by the retry interval, so at most a handful of
        # sockets can ever be in flight.
        attempt: dict[str, object] = {}

        def _attempt() -> None:
            try:
                attempt["written"] = refresh_validator_allowlist(
                    subtensor_network=subtensor_network,
                    netuid=netuid,
                    out_path=out_path,
                    chain_config_path=chain_config_path,
                    allow_extra=allow_extra,
                )
            except Exception as exc:  # noqa: BLE001 - logged below
                attempt["error"] = exc

        worker = threading.Thread(
            target=_attempt,
            daemon=True,
            name="allowlist-refresh-attempt",
        )
        worker.start()
        worker.join(ALLOWLIST_REFRESH_ATTEMPT_TIMEOUT_S)
        if worker.is_alive():
            logger.error(
                "validator allowlist refresh hung past %.0fs (chain RPC "
                "starved?); abandoning the attempt and retrying",
                ALLOWLIST_REFRESH_ATTEMPT_TIMEOUT_S,
            )
        elif "error" in attempt:
            logger.error(
                "validator allowlist refresh failed (will retry): %s",
                attempt["error"],
            )
        else:
            wrote_once = True
            logger.info(
                "validator allowlist refreshed: %d validators on netuid %d",
                int(attempt.get("written", 0) or 0),
                netuid,
            )
        # Before the first success the worker is undriveable, so retry fast;
        # afterwards the file is valid and the normal interval keeps it
        # inside the coordinator's freshness window.
        waiter.wait(
            interval_s if wrote_once else min(interval_s, ALLOWLIST_FIRST_SUCCESS_RETRY_S)
        )
