"""Shared mesh onboarding machinery: pool creation, idempotency, join flow.

This is the single implementation behind both setup front-ends
(``verathos setup mesh-coordinator`` and ``verathos mesh setup``), so its
guarantees are tested here once, not per front-end.
"""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import verallm.mesh.onboarding as onboarding
from verallm.mesh.onboarding import (
    ExistingSetup,
    ManagerProbe,
    MeshCoordinatorOptions,
    MeshCoordinatorResult,
    POOL_MANAGER_PM2_NAME,
    endpoint_is_loopback,
    materialize_worker_token,
    resolve_manager_endpoint,
    resolve_network,
    setup_mesh_coordinator,
    validator_preflight,
)


def _empty_scan(*args, **kwargs) -> ExistingSetup:
    return ExistingSetup(
        pools=[], manager=ManagerProbe(reachable=False), pm2_apps=[]
    )


@pytest.fixture()
def wired(monkeypatch, tmp_path):
    """Stub every side effect; capture PM2 argv and health polls."""
    calls: dict[str, object] = {"pm2": [], "health": []}
    monkeypatch.setattr(onboarding, "find_repo_root", lambda: tmp_path)
    monkeypatch.setattr(onboarding, "_run_setup_script", lambda repo: None)
    monkeypatch.setattr(onboarding.shutil, "which", lambda name: "/usr/bin/pm2")
    monkeypatch.setattr(onboarding, "scan_existing_setup", _empty_scan)

    def fake_run(argv, **kwargs):
        calls["pm2"].append(list(argv))

        class _Done:
            returncode = 0

        return _Done()

    monkeypatch.setattr(onboarding.subprocess, "run", fake_run)
    monkeypatch.setattr(
        onboarding,
        "_poll_manager_health",
        lambda endpoint, pool_id: calls["health"].append((endpoint, pool_id)),
    )
    return calls


def test_dev_mode_creates_pool_and_prints_join_one_liner(wired, tmp_path):
    opts = MeshCoordinatorOptions(
        serving_mode="dev",
        manager_host="198.51.100.7",
        pools_root=tmp_path / "pools",
        skip_install=True,
    )
    result = setup_mesh_coordinator(opts)
    assert result.pool_dir.is_dir()
    assert result.pool_id.startswith("pool-")
    assert result.manager_endpoint == "http://198.51.100.7:9500"
    assert result.dashboard_url.endswith("/operator")
    assert "--mesh-worker --token vtpool_" in result.join_command
    assert result.admin_token_file.exists()
    # The join command carries the worker token, never the admin token.
    admin_token = result.admin_token_file.read_text().strip()
    assert admin_token not in result.join_command
    # Health was polled locally against the freshly minted pool id.
    assert wired["health"] == [("http://127.0.0.1:9500", result.pool_id)]


def test_pm2_manager_unit_argv(wired, tmp_path):
    opts = MeshCoordinatorOptions(
        serving_mode="dev",
        manager_host="198.51.100.7",
        pools_root=tmp_path / "pools",
        skip_install=True,
    )
    setup_mesh_coordinator(opts)
    start = next(
        argv for argv in wired["pm2"] if argv[:2] == ["pm2", "start"]
    )
    assert POOL_MANAGER_PM2_NAME in start
    assert "--interpreter" in start and "none" in start
    # Unlike worker units the manager auto-restarts: no crash breaker to reset.
    assert "--no-autorestart" not in start
    serve_tail = start[start.index("--") + 1 :]
    assert serve_tail[:5] == ["-m", "neurons.cli", "mesh", "pool", "serve"]
    assert "--host" in serve_tail and "0.0.0.0" in serve_tail


def test_validator_mode_preflight_runs_before_state_is_written(wired, tmp_path):
    pools_root = tmp_path / "pools"
    opts = MeshCoordinatorOptions(
        serving_mode="validator",
        manager_host="198.51.100.7",
        pools_root=pools_root,
        skip_install=True,
    )
    with pytest.raises(SystemExit, match="--owner-account"):
        setup_mesh_coordinator(opts)
    assert not pools_root.exists()
    assert wired["pm2"] == []


def test_validator_mode_requires_https():
    opts = MeshCoordinatorOptions(
        serving_mode="validator",
        manager_host="198.51.100.7",
        owner_account="5F" + "a" * 46,
        coordinator_address="0x" + "1" * 40,
        validator_shared_state="/tmp/state.json",
        chain_id=945,
        netuid=405,
        coordinator_uid=2,
        epoch=1,
    )
    endpoint = resolve_manager_endpoint(opts)
    assert endpoint.startswith("http://")
    with pytest.raises(SystemExit, match="HTTPS"):
        validator_preflight(opts, endpoint)
    # An nginx-fronted HTTPS override passes without local TLS files.
    opts.manager_endpoint = "https://pool.example:9543"
    validator_preflight(opts, resolve_manager_endpoint(opts))


def test_tls_files_switch_endpoint_scheme(tmp_path):
    opts = MeshCoordinatorOptions(
        manager_host="198.51.100.7",
        tls_cert=str(tmp_path / "cert.pem"),
        tls_key=str(tmp_path / "key.pem"),
    )
    assert resolve_manager_endpoint(opts) == "https://198.51.100.7:9500"


# --- idempotency: the second run must never mint silently ------------------


def test_rerun_refuses_when_machine_already_has_mesh_state(monkeypatch, tmp_path):
    """The exact incident: a re-run minted pool #2 and crash-looped on 9500."""

    monkeypatch.setattr(
        onboarding,
        "scan_existing_setup",
        lambda *a, **k: ExistingSetup(
            pools=[tmp_path / "pools" / "pool-existing"],
            manager=ManagerProbe(reachable=True, pool_id="pool-existing"),
            pm2_apps=[POOL_MANAGER_PM2_NAME],
        ),
    )
    created: list = []
    monkeypatch.setattr(onboarding, "find_repo_root", lambda: tmp_path)
    monkeypatch.setattr(
        onboarding, "_run_setup_script", lambda repo: created.append("install")
    )
    opts = MeshCoordinatorOptions(
        serving_mode="dev",
        manager_host="198.51.100.7",
        pools_root=tmp_path / "pools",
    )
    with pytest.raises(SystemExit, match="already has mesh state"):
        setup_mesh_coordinator(opts)
    # Refusal happens before any install or state write.
    assert created == []
    assert not (tmp_path / "pools").exists()


def test_explicit_new_pool_overrides_the_refusal(wired, tmp_path, monkeypatch):
    monkeypatch.setattr(
        onboarding,
        "scan_existing_setup",
        lambda *a, **k: ExistingSetup(
            pools=[tmp_path / "pool-old"],
            manager=ManagerProbe(reachable=False),
            pm2_apps=[],
        ),
    )
    monkeypatch.setattr(onboarding, "port_in_use", lambda *a, **k: False)
    opts = MeshCoordinatorOptions(
        serving_mode="dev",
        manager_host="198.51.100.7",
        pools_root=tmp_path / "pools",
        skip_install=True,
        allow_new_pool=True,
    )
    result = setup_mesh_coordinator(opts)
    assert result.pool_id.startswith("pool-")


def test_new_pool_refuses_a_taken_port_pm2_does_not_own(monkeypatch, tmp_path):
    """A foreign process on the manager port must refuse, not crash-loop."""

    monkeypatch.setattr(onboarding, "port_in_use", lambda *a, **k: True)
    monkeypatch.setattr(
        onboarding,
        "probe_manager",
        lambda *a, **k: ManagerProbe(reachable=True, pool_id="pool-foreign"),
    )
    monkeypatch.setattr(onboarding, "pm2_mesh_apps", lambda: [])
    opts = MeshCoordinatorOptions(
        serving_mode="dev",
        manager_host="198.51.100.7",
        pools_root=tmp_path / "pools",
        skip_install=True,
        allow_new_pool=True,
    )
    with pytest.raises(SystemExit, match="cannot be replaced safely"):
        setup_mesh_coordinator(opts)


def test_scan_finds_pools_in_both_roots(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    legacy = tmp_path / ".verathos" / "mesh-pool" / "pool-legacy11111"
    legacy.mkdir(parents=True)
    (legacy / "pool-state.json").write_text("{}")
    chosen_root = tmp_path / "pools"
    current = chosen_root / "pool-current2222"
    current.mkdir(parents=True)
    (current / "pool-state.json").write_text("{}")
    # A dir without pool-state.json is noise, not a pool.
    (chosen_root / "pool-broken").mkdir()

    scan = onboarding.scan_existing_setup(chosen_root, manager_port=1)
    found = {p.name for p in scan.pools}
    assert found == {"pool-legacy11111", "pool-current2222"}


def test_result_from_existing_pool_round_trips(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    from verallm.mesh.pool import create_pool_state

    pool_dir, token = create_pool_state(
        tmp_path / "pools",
        manager_endpoint="http://198.51.100.7:9500",
        serving_mode="dev",
    )
    result = onboarding.result_from_existing_pool(pool_dir)
    assert result.reused
    assert result.pool_id == pool_dir.name
    assert result.manager_endpoint == "http://198.51.100.7:9500"
    assert token.encode() in result.join_command


# --- network naming --------------------------------------------------------


def test_resolve_network_reads_the_shipped_chain_configs():
    testnet = resolve_network("testnet")
    assert testnet == {
        "network": "testnet",
        "chain_config": testnet["chain_config"],
        "chain_id": 945,
        "netuid": 405,
    }
    assert testnet["chain_config"].endswith("chain_config_testnet.json")
    mainnet = resolve_network("mainnet")
    assert (mainnet["chain_id"], mainnet["netuid"]) == (964, 96)


def test_resolve_network_local_and_unknown():
    assert resolve_network("local") is None
    assert resolve_network("") is None
    with pytest.raises(SystemExit, match="unknown network"):
        resolve_network("moonnet")


# --- multi-machine join flow ----------------------------------------------


def test_materialize_worker_token_is_owner_only(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    target = materialize_worker_token("vtpool_secret\n")
    assert target.read_text() == "vtpool_secret\n"
    assert (target.stat().st_mode & 0o777) == 0o600
    assert (target.parent.stat().st_mode & 0o777) == 0o700


def test_describe_worker_token_decodes_and_probes(monkeypatch):
    from verallm.mesh.pool import MeshPoolToken

    token = MeshPoolToken(
        pool_id="pool-abc",
        manager_endpoint="http://127.0.0.1:9500",
        pool_secret="s3cret",
    )
    monkeypatch.setattr(
        onboarding,
        "probe_manager",
        lambda endpoint, timeout=5.0: ManagerProbe(
            reachable=True, pool_id="pool-abc"
        ),
    )
    info = onboarding.describe_worker_token(token.encode())
    assert info["pool_id"] == "pool-abc"
    assert info["manager_endpoint"] == "http://127.0.0.1:9500"
    assert info["loopback"] is True
    assert info["reachable"] is True


def test_endpoint_is_loopback():
    assert endpoint_is_loopback("http://127.0.0.1:9500")
    assert endpoint_is_loopback("http://localhost:9500")
    assert not endpoint_is_loopback("https://pool.example:9543")
    assert not endpoint_is_loopback("http://198.51.100.7:9500")


# --- the join-token command ------------------------------------------------


def test_pool_join_token_prints_the_one_liner_again(tmp_path, monkeypatch, capsys):
    """The token used to be write-once: shown at creation and never again."""

    import argparse

    import verallm.mesh.cli as cli_module
    from verallm.mesh.pool import create_pool_state

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    pool_dir, token = create_pool_state(
        tmp_path / "pools",
        manager_endpoint="http://198.51.100.7:9500",
        serving_mode="dev",
    )
    cli_module.cmd_pool_join_token(argparse.Namespace(pool=str(pool_dir)))
    output = capsys.readouterr().out
    assert token.pool_id in output
    assert "--mesh-worker --token " + token.encode() in output
    # A reachable endpoint gets no loopback warning.
    assert "loopback" not in output


def test_pool_join_token_warns_on_loopback_endpoint(tmp_path, monkeypatch, capsys):
    import argparse

    import verallm.mesh.cli as cli_module
    from verallm.mesh.pool import create_pool_state

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    pool_dir, _token = create_pool_state(
        tmp_path / "pools",
        manager_endpoint="http://127.0.0.1:9500",
        serving_mode="dev",
    )
    cli_module.cmd_pool_join_token(argparse.Namespace(pool=str(pool_dir)))
    output = capsys.readouterr().out
    assert "loopback" in output
    assert "THIS machine" in output


def test_pool_join_token_discovers_a_single_known_pool(
    tmp_path, monkeypatch, capsys
):
    import argparse
    import tempfile

    import verallm.mesh.cli as cli_module
    from verallm.mesh.pool import create_pool_state

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path / "faketmp"))
    _pool_dir, token = create_pool_state(
        tmp_path / "pools",
        manager_endpoint="http://198.51.100.7:9500",
        serving_mode="dev",
    )
    cli_module.cmd_pool_join_token(argparse.Namespace(pool=""))
    assert token.pool_id in capsys.readouterr().out


# --- local GPU enrollment --------------------------------------------------


def _result(tmp_path) -> MeshCoordinatorResult:
    return MeshCoordinatorResult(
        pool_dir=tmp_path,
        pool_id="pool-x",
        manager_endpoint="http://198.51.100.7:9500",
        worker_token="vtpool_abc",
        admin_token_file=tmp_path / "pool-admin-token.txt",
        pm2_name="verathos-pool-manager",
        dashboard_url="http://198.51.100.7:9500/operator",
        join_command="curl ...",
    )


def test_coordinator_joins_local_gpus_when_asked(monkeypatch, tmp_path):
    from verallm.mesh import units as units_module

    joined: list[tuple[str, list[str]]] = []
    monkeypatch.setattr(
        onboarding,
        "setup_mesh_worker",
        lambda token, passthrough: joined.append((token, list(passthrough))),
    )
    monkeypatch.setattr(
        units_module,
        "detect_gpus",
        lambda: [units_module.GpuInfo(index=0, name="RTX 4090", vram_gb=24)],
    )
    opts = MeshCoordinatorOptions(join_local_gpus=True, non_interactive=True)
    assert onboarding.maybe_join_local_gpus(_result(tmp_path), opts)
    # The local join uses the WORKER token, never the admin secret.
    assert joined == [("vtpool_abc", [])]


def test_coordinator_only_without_explicit_flag_in_yes_mode(monkeypatch, tmp_path):
    from verallm.mesh import units as units_module

    joined: list = []
    monkeypatch.setattr(
        onboarding, "setup_mesh_worker", lambda *a: joined.append(a)
    )
    monkeypatch.setattr(
        units_module,
        "detect_gpus",
        lambda: [units_module.GpuInfo(index=0, name="RTX 4090", vram_gb=24)],
    )
    # --yes without --join-local-gpus: never join implicitly.
    opts = MeshCoordinatorOptions(join_local_gpus=None, non_interactive=True)
    assert not onboarding.maybe_join_local_gpus(_result(tmp_path), opts)
    assert joined == []


def test_interactive_prompt_defaults_to_joining(monkeypatch, tmp_path):
    from verallm.mesh import units as units_module

    joined: list = []
    prompts: list[str] = []
    monkeypatch.setattr(
        onboarding, "setup_mesh_worker", lambda *a: joined.append(a)
    )
    monkeypatch.setattr(
        units_module,
        "detect_gpus",
        lambda: [
            units_module.GpuInfo(index=0, name="RTX 4090", vram_gb=24),
            units_module.GpuInfo(index=1, name="RTX 4090", vram_gb=24),
        ],
    )
    monkeypatch.setattr(
        onboarding,
        "_confirm",
        lambda message, default=True: prompts.append(message) or default,
    )
    monkeypatch.setattr("builtins.input", lambda prompt="": "1")
    opts = MeshCoordinatorOptions(join_local_gpus=None, non_interactive=False)
    assert onboarding.maybe_join_local_gpus(_result(tmp_path), opts)
    assert len(joined) == 1
    assert "2 GPU(s)" in prompts[0]
    # Choice 1 = one worker with all GPUs: no --gpus split flag is passed.
    assert "--gpus" not in joined[0][1]


def test_no_gpu_machine_never_prompts(monkeypatch, tmp_path):
    from verallm.mesh import units as units_module

    monkeypatch.setattr(units_module, "detect_gpus", lambda: [])
    monkeypatch.setattr(
        onboarding,
        "_confirm",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("prompted")),
    )
    opts = MeshCoordinatorOptions(join_local_gpus=None, non_interactive=False)
    assert not onboarding.maybe_join_local_gpus(_result(tmp_path), opts)


def test_stale_manager_pool_id_mismatch_fails(monkeypatch):
    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return json.dumps({"status": "ok", "pool_id": "pool-stale"}).encode()

    monkeypatch.setattr(
        onboarding.urllib.request, "urlopen", lambda *a, **k: _Resp()
    )
    monkeypatch.setattr(onboarding, "_HEALTH_POLL_ATTEMPTS", 2)
    monkeypatch.setattr(onboarding, "_HEALTH_POLL_INTERVAL_S", 0.0)
    with pytest.raises(SystemExit, match="stale manager"):
        onboarding._poll_manager_health("http://127.0.0.1:9500", "pool-fresh")


def test_loopback_equivalent_keeps_scheme_and_port():
    assert (
        onboarding.loopback_equivalent("https://198.51.100.7:9543")
        == "https://127.0.0.1:9543"
    )
    assert (
        onboarding.loopback_equivalent("http://pool.example:9500/")
        == "http://127.0.0.1:9500"
    )
    # No explicit port falls back to the default manager port.
    assert (
        onboarding.loopback_equivalent("http://pool.example")
        == "http://127.0.0.1:9500"
    )


def test_local_member_worker_ids_matches_loopback_and_public_ip(monkeypatch):
    monkeypatch.setattr(onboarding, "detect_public_ip", lambda: "198.51.100.7")
    status = {
        "workers": {
            "w-loop": {
                "stale": False,
                "endpoints": {"rpc": "127.0.0.1:50052"},
            },
            "w-pub": {
                "stale": False,
                "endpoints": {"rpc": "198.51.100.7:50053"},
            },
            "w-remote": {
                "stale": False,
                "endpoints": {"rpc": "203.0.113.9:50052"},
            },
            "w-stale-local": {
                "stale": True,
                "endpoints": {"rpc": "127.0.0.1:50054"},
            },
        }
    }
    assert sorted(onboarding.local_member_worker_ids(status)) == [
        "w-loop",
        "w-pub",
    ]


def test_join_local_gpus_skips_when_gpus_already_serve(monkeypatch, tmp_path):
    """The double-join regression: a reuse re-run once stacked a second pair
    of worker units onto two GPUs that were already pool members."""
    from verallm.mesh import units as units_module

    monkeypatch.setattr(
        units_module,
        "detect_gpus",
        lambda: [
            units_module.GpuInfo(index=0, name="RTX 5090", vram_gb=31),
            units_module.GpuInfo(index=1, name="RTX 5090", vram_gb=31),
        ],
    )
    monkeypatch.setattr(
        onboarding,
        "pool_management_status",
        lambda pool_dir, timeout=5.0: {
            "workers": {
                "w-a": {"stale": False, "endpoints": {"rpc": "127.0.0.1:50052"}},
                "w-b": {"stale": False, "endpoints": {"rpc": "127.0.0.1:50053"}},
            }
        },
    )
    monkeypatch.setattr(
        onboarding,
        "setup_mesh_worker",
        lambda *a: (_ for _ in ()).throw(AssertionError("joined again")),
    )
    opts = MeshCoordinatorOptions(join_local_gpus=True, non_interactive=True)
    assert not onboarding.maybe_join_local_gpus(_result(tmp_path), opts)


def test_join_local_gpus_skips_on_partial_membership(monkeypatch, tmp_path):
    from verallm.mesh import units as units_module

    monkeypatch.setattr(
        units_module,
        "detect_gpus",
        lambda: [
            units_module.GpuInfo(index=0, name="RTX 5090", vram_gb=31),
            units_module.GpuInfo(index=1, name="RTX 5090", vram_gb=31),
        ],
    )
    monkeypatch.setattr(
        onboarding,
        "pool_management_status",
        lambda pool_dir, timeout=5.0: {
            "workers": {
                "w-a": {"stale": False, "endpoints": {"rpc": "127.0.0.1:50052"}},
            }
        },
    )
    joined: list = []
    monkeypatch.setattr(
        onboarding, "setup_mesh_worker", lambda *a: joined.append(a)
    )
    opts = MeshCoordinatorOptions(join_local_gpus=True, non_interactive=True)
    assert not onboarding.maybe_join_local_gpus(_result(tmp_path), opts)
    assert joined == []


def test_join_local_gpus_proceeds_when_manager_is_unreachable(
    monkeypatch, tmp_path
):
    """No live view means no membership evidence: the join itself is still
    the idempotent path (per-worker ids), so proceed as before."""
    from verallm.mesh import units as units_module

    monkeypatch.setattr(
        units_module,
        "detect_gpus",
        lambda: [units_module.GpuInfo(index=0, name="RTX 5090", vram_gb=31)],
    )
    monkeypatch.setattr(
        onboarding, "pool_management_status", lambda *a, **k: None
    )
    joined: list = []
    monkeypatch.setattr(
        onboarding,
        "setup_mesh_worker",
        lambda token, passthrough: joined.append(token),
    )
    opts = MeshCoordinatorOptions(join_local_gpus=True, non_interactive=True)
    assert onboarding.maybe_join_local_gpus(_result(tmp_path), opts)
    assert joined == ["vtpool_abc"]


def test_pool_management_status_returns_none_without_admin_token(tmp_path):
    assert onboarding.pool_management_status(tmp_path) is None


def test_enrolled_but_unjoined_units_get_a_restart_not_a_reinstall(
    monkeypatch, tmp_path
):
    """Units already enrolled for this pool that are not currently pool
    members (crashed / unreachable manager) must be restarted, never
    re-planned through join_pool.sh."""

    from verallm.mesh import units as units_module

    monkeypatch.setattr(
        units_module,
        "detect_gpus",
        lambda: [units_module.GpuInfo(index=0, name="RTX 5090", vram_gb=31)],
    )
    # Live view answers, but no local workers are members.
    monkeypatch.setattr(
        onboarding,
        "pool_management_status",
        lambda pool_dir, timeout=5.0: {"workers": {}},
    )
    planned = units_module.plan_worker_units(
        [units_module.GpuInfo(index=0, name="RTX 5090", vram_gb=31)],
        worker_id_base="host",
        home=tmp_path,
    )
    registry = {
        "version": 1,
        "pool_id": "pool-x",
        "manager_endpoint": "http://198.51.100.7:9500",
        "token_file": "",
        "units": [unit.__dict__ for unit in planned],
    }
    monkeypatch.setattr(
        units_module, "load_unit_registry", lambda path=None: registry
    )
    monkeypatch.setattr(
        onboarding, "pm2_mesh_apps", lambda: [planned[0].pm2_name]
    )
    restarted: list[list[str]] = []
    monkeypatch.setattr(
        onboarding.subprocess,
        "run",
        lambda argv, **k: restarted.append(list(argv)),
    )
    monkeypatch.setattr(
        onboarding,
        "setup_mesh_worker",
        lambda *a: (_ for _ in ()).throw(AssertionError("reinstalled")),
    )
    opts = MeshCoordinatorOptions(join_local_gpus=True, non_interactive=True)
    assert onboarding.maybe_join_local_gpus(_result(tmp_path), opts)
    pm2_calls = [argv for argv in restarted if argv and argv[0] == "pm2"]
    assert pm2_calls == [["pm2", "restart", planned[0].pm2_name]]


def test_subnet_manager_unit_carries_the_lease_renewal_identity(wired, tmp_path):
    """A subnet pool's manager owns the on-chain lease. Without the signing
    wallet and chain config it comes up healthy, serves fine, and then the
    registration expires with its 24h lease and the miner drops off chain
    with nothing in the pool logs explaining it. The operator already gives
    this identity for the driver, so the manager unit must carry it."""
    from bittensor_wallet import Keypair

    opts = MeshCoordinatorOptions(
        serving_mode="validator",
        manager_endpoint="http://127.0.0.1:9500",
        owner_account=Keypair.create_from_uri("//MeshPoolOwner").ss58_address,
        coordinator_address="0x" + "1" * 40,
        validator_shared_state="/tmp/state.json",
        chain_id=945,
        netuid=405,
        coordinator_uid=1,
        epoch=21_497,
        pools_root=tmp_path / "pools",
        skip_install=True,
        worker_wallet_name="test_miner96",
        worker_wallet_hotkey="default",
        chain_config="/root/verathos/chain_config_testnet.json",
    )
    setup_mesh_coordinator(opts)
    start = next(argv for argv in wired["pm2"] if argv[:2] == ["pm2", "start"])
    serve_tail = start[start.index("--") + 1 :]
    assert serve_tail[serve_tail.index("--wallet-name") + 1] == "test_miner96"
    assert serve_tail[serve_tail.index("--wallet-hotkey") + 1] == "default"
    assert (
        serve_tail[serve_tail.index("--chain-config") + 1]
        == "/root/verathos/chain_config_testnet.json"
    )


def test_dev_manager_unit_stays_walletless(wired, tmp_path):
    """A dev pool has no chain lease, so it must not be handed a signing
    wallet just because one was supplied for its local GPUs."""
    opts = MeshCoordinatorOptions(
        serving_mode="dev",
        manager_host="198.51.100.7",
        pools_root=tmp_path / "pools",
        skip_install=True,
        worker_wallet_name="test_miner96",
        worker_wallet_hotkey="default",
    )
    setup_mesh_coordinator(opts)
    start = next(argv for argv in wired["pm2"] if argv[:2] == ["pm2", "start"])
    assert "--wallet-name" not in start


def test_manager_crash_loop_error_detects_restart_movement(monkeypatch):
    """A unit that keeps restarting must be reported with its error tail;
    the health poll alone can land inside a crash loop's up-window."""

    import json as _json
    import types

    from verallm.mesh import onboarding

    counters = iter([100, 103])

    def fake_run(argv, capture_output=True, text=True):
        if argv[:2] == ["pm2", "jlist"]:
            return types.SimpleNamespace(
                stdout=_json.dumps(
                    [
                        {
                            "name": "verathos-pool-manager",
                            "pm2_env": {"restart_time": next(counters)},
                        }
                    ]
                ),
                returncode=0,
            )
        assert argv[:2] == ["pm2", "logs"]
        return types.SimpleNamespace(
            stdout="TLS key not found: /root/.verathos/tls/key.pem\n",
            returncode=0,
        )

    monkeypatch.setattr(onboarding.subprocess, "run", fake_run)
    monkeypatch.setattr(onboarding.time, "sleep", lambda _s: None)

    message = onboarding.manager_crash_loop_error(
        "verathos-pool-manager", settle_seconds=0.0
    )
    assert "crash-looping" in message
    assert "restarts 100 -> 103" in message
    assert "TLS key not found" in message


def test_manager_crash_loop_error_quiet_on_stable_unit(monkeypatch):
    import json as _json
    import types

    from verallm.mesh import onboarding

    def fake_run(argv, capture_output=True, text=True):
        return types.SimpleNamespace(
            stdout=_json.dumps(
                [
                    {
                        "name": "verathos-pool-manager",
                        "pm2_env": {"restart_time": 7},
                    }
                ]
            ),
            returncode=0,
        )

    monkeypatch.setattr(onboarding.subprocess, "run", fake_run)
    monkeypatch.setattr(onboarding.time, "sleep", lambda _s: None)

    assert (
        onboarding.manager_crash_loop_error(
            "verathos-pool-manager", settle_seconds=0.0
        )
        == ""
    )


def test_api_tls_pubkey_pin_uses_argv_and_returns_curl_pin(monkeypatch, tmp_path):
    pool_dir = tmp_path / "pool with ' quote"
    pool_dir.mkdir()
    certfile = pool_dir / "api-tls-cert.pem"
    certfile.write_text("certificate", encoding="utf-8")
    calls = []

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        if argv[1] == "x509":
            return SimpleNamespace(returncode=0, stdout=b"public-key-pem")
        assert argv == ["openssl", "pkey", "-pubin", "-outform", "der"]
        assert kwargs["input"] == b"public-key-pem"
        return SimpleNamespace(returncode=0, stdout=b"public-key-der")

    monkeypatch.setattr(onboarding.subprocess, "run", fake_run)

    expected = base64.b64encode(
        hashlib.sha256(b"public-key-der").digest()
    ).decode("ascii")
    assert onboarding.api_tls_pubkey_pin(pool_dir) == expected
    assert calls[0][0] == [
        "openssl", "x509", "-pubkey", "-noout", "-in", str(certfile)
    ]
    assert all(isinstance(call[0], list) for call in calls)


def test_api_tls_pubkey_pin_returns_empty_on_openssl_failure(
    monkeypatch, tmp_path
):
    (tmp_path / "api-tls-cert.pem").write_text("certificate", encoding="utf-8")
    monkeypatch.setattr(
        onboarding.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout=b""),
    )
    assert onboarding.api_tls_pubkey_pin(tmp_path) == ""
