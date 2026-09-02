"""Mesh worker validator-allowlist refresher: a standalone mesh miner has
no vLLM miner process to keep validators.json fresh, and the coordinator
refuses a stale file, so the worker must be able to refresh it itself."""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from verallm.mesh.allowlist import refresh_validator_allowlist


class _Tensorish:
    def __init__(self, value):
        self._value = value

    def item(self):
        return self._value

    def __getitem__(self, index):
        return self._value[index]


def _fake_bittensor(monkeypatch, *, hotkeys, permits, stakes):
    metagraph = types.SimpleNamespace(
        n=_Tensorish(len(hotkeys)),
        hotkeys=list(hotkeys),
        validator_permit=list(permits),
        S=list(stakes),
    )

    class _Subtensor:
        def __init__(self, network=""):
            self.network = network

        def metagraph(self, netuid):
            assert netuid == 405
            return metagraph

    module = types.SimpleNamespace(Subtensor=_Subtensor)
    monkeypatch.setitem(sys.modules, "bittensor", module)
    return metagraph


def test_refresh_writes_permitted_validators_and_extras(
    tmp_path: Path, monkeypatch
):
    _fake_bittensor(
        monkeypatch,
        hotkeys=["5Owner", "5NoPermit", "5Vali2"],
        permits=[True, False, True],
        stakes=[1000.0, 5000.0, 250.0],
    )
    out = tmp_path / "validators.json"
    written = refresh_validator_allowlist(
        subtensor_network="test",
        netuid=405,
        out_path=out,
        allow_extra=("5Manual", "5Owner"),
    )
    payload = json.loads(out.read_text())
    hotkeys = [v["hotkey_ss58"] for v in payload["validators"]]
    # Permitted validators plus the manual extra; no duplicate for 5Owner,
    # and the unpermitted hotkey never enters the allowlist.
    assert hotkeys == ["5Owner", "5Vali2", "5Manual"]
    assert written == 3
    assert payload["netuid"] == 405
    assert payload["updated_at"] > 0
    # The file shape matches what the coordinator's validator auth reads
    # (same as the vLLM miner's refresh).
    assert all(
        set(v) == {"uid", "hotkey_ss58", "stake"}
        for v in payload["validators"]
    )


def test_refresh_raises_on_metagraph_failure(tmp_path: Path, monkeypatch):
    class _Subtensor:
        def __init__(self, network=""):
            raise RuntimeError("chain unreachable")

    monkeypatch.setitem(
        sys.modules, "bittensor", types.SimpleNamespace(Subtensor=_Subtensor)
    )
    out = tmp_path / "validators.json"
    with pytest.raises(RuntimeError, match="chain unreachable"):
        refresh_validator_allowlist(
            subtensor_network="test", netuid=405, out_path=out
        )
    # Nothing half-written on failure.
    assert not out.exists()


def test_refresh_closes_owned_subtensor_after_success(tmp_path: Path, monkeypatch):
    import verallm.mesh.allowlist as allowlist

    _fake_bittensor(
        monkeypatch,
        hotkeys=["5Owner"],
        permits=[True],
        stakes=[1000.0],
    )
    closed: list[object] = []
    monkeypatch.setattr(
        allowlist, "close_owned_subtensor", lambda subtensor: closed.append(subtensor)
    )

    refresh_validator_allowlist(
        subtensor_network="test",
        netuid=405,
        out_path=tmp_path / "validators.json",
    )

    assert len(closed) == 1


def test_refresh_closes_owned_subtensor_after_metagraph_failure(
    tmp_path: Path, monkeypatch
):
    import verallm.mesh.allowlist as allowlist

    class _Subtensor:
        def __init__(self, network=""):
            self.network = network

        def metagraph(self, _netuid):
            raise RuntimeError("metagraph unavailable")

    monkeypatch.setitem(
        sys.modules, "bittensor", types.SimpleNamespace(Subtensor=_Subtensor)
    )
    closed: list[object] = []
    monkeypatch.setattr(
        allowlist, "close_owned_subtensor", lambda subtensor: closed.append(subtensor)
    )

    with pytest.raises(RuntimeError, match="metagraph unavailable"):
        refresh_validator_allowlist(
            subtensor_network="test",
            netuid=405,
            out_path=tmp_path / "validators.json",
        )

    assert len(closed) == 1


def test_refresher_retries_fast_until_the_first_success(monkeypatch, tmp_path):
    """Before the first successful write the worker cannot drive at all
    (placement reports "member only: ... allowlist is unavailable"), so a
    startup failure must not cost a whole refresh interval."""
    import threading

    import verallm.mesh.allowlist as allowlist

    waits: list[float] = []
    calls: list[int] = []

    class _Waiter:
        def __init__(self) -> None:
            self._stop = False

        def is_set(self) -> bool:
            return self._stop

        def wait(self, seconds: float) -> None:
            waits.append(seconds)
            if len(waits) >= 3:
                self._stop = True

    def flaky(**_kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("deadlock detected by _ModuleLock('scalecodec')")
        return 7

    monkeypatch.setattr(allowlist, "refresh_validator_allowlist", flaky)
    allowlist.run_allowlist_refresher(
        subtensor_network="test",
        netuid=405,
        out_path=tmp_path / "validators.json",
        interval_s=300.0,
        stop_event=_Waiter(),
    )
    # First wait is the fast retry, later waits use the normal interval.
    assert waits[0] == allowlist.ALLOWLIST_FIRST_SUCCESS_RETRY_S
    assert waits[1] == 300.0


def test_preimport_chain_modules_loads_bittensor():
    """The refresher thread must never be the first importer of the chain
    stack; this is what the worker calls on the main thread to guarantee it."""
    import sys

    import verallm.mesh.allowlist as allowlist

    allowlist.preimport_chain_modules()
    assert "bittensor" in sys.modules
