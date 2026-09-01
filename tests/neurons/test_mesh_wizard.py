"""The flag-driven mesh wizard is a shim over verallm.mesh.onboarding.

The machinery itself (pool creation, idempotency, health polling, GPU
enrollment) is tested in tests/verallm/test_mesh_onboarding.py; these tests
pin the shim contract: stable import surface, CLI dispatch, and the flag
plumbing into MeshCoordinatorOptions.
"""
from __future__ import annotations

import pytest

import neurons.mesh_wizard as mesh_wizard


def test_shim_reexports_the_shared_machinery():
    """External callers import these names from neurons.mesh_wizard."""

    import verallm.mesh.onboarding as onboarding

    assert mesh_wizard.MeshCoordinatorOptions is onboarding.MeshCoordinatorOptions
    assert mesh_wizard.setup_mesh_coordinator is onboarding.setup_mesh_coordinator
    assert mesh_wizard.setup_mesh_worker is onboarding.setup_mesh_worker
    assert mesh_wizard.maybe_join_local_gpus is onboarding.maybe_join_local_gpus
    assert mesh_wizard.POOL_MANAGER_PM2_NAME == "verathos-pool-manager"


def test_setup_dispatch_routes_mesh_roles(monkeypatch):
    import neurons.cli as neurons_cli

    seen: list[tuple[str, list[str]]] = []
    monkeypatch.setattr(
        mesh_wizard,
        "run_mesh_wizard",
        lambda role, argv: seen.append((role, list(argv))),
    )
    neurons_cli.cmd_setup(["mesh-coordinator", "--yes", "--skip-install"])
    assert seen == [("mesh-coordinator", ["--yes", "--skip-install"])]


def test_mesh_worker_requires_a_token(monkeypatch):
    with pytest.raises(SystemExit, match="vtpool_"):
        mesh_wizard.run_mesh_wizard("mesh-worker", [])


def test_mesh_worker_token_file_passthrough(monkeypatch):
    calls: list[tuple[str, list[str]]] = []
    monkeypatch.setattr(
        mesh_wizard,
        "setup_mesh_worker",
        lambda token, passthrough: calls.append((token, list(passthrough))),
    )
    mesh_wizard.run_mesh_wizard(
        "mesh-worker", ["--token-file", "/tmp/t.txt", "--gpus", "0"]
    )
    assert calls == [("", ["--token-file", "/tmp/t.txt", "--gpus", "0"])]


def _capture_opts(monkeypatch):
    captured: dict = {}

    def fake_setup(opts):
        captured["opts"] = opts
        raise SystemExit(0)  # stop before any side effect

    monkeypatch.setattr(mesh_wizard, "setup_mesh_coordinator", fake_setup)
    return captured


def test_network_testnet_fills_chain_binding_from_config(monkeypatch):
    captured = _capture_opts(monkeypatch)
    with pytest.raises(SystemExit):
        mesh_wizard.run_mesh_wizard(
            "mesh-coordinator",
            ["--yes", "--network", "testnet", "--host", "198.51.100.7"],
        )
    opts = captured["opts"]
    assert opts.serving_mode == "subnet"
    assert (opts.chain_id, opts.netuid) == (945, 405)


def test_network_local_maps_to_dev(monkeypatch):
    captured = _capture_opts(monkeypatch)
    with pytest.raises(SystemExit):
        mesh_wizard.run_mesh_wizard(
            "mesh-coordinator",
            ["--yes", "--network", "local", "--host", "198.51.100.7"],
        )
    assert captured["opts"].serving_mode == "dev"


def test_new_pool_flag_reaches_the_options(monkeypatch):
    captured = _capture_opts(monkeypatch)
    with pytest.raises(SystemExit):
        mesh_wizard.run_mesh_wizard(
            "mesh-coordinator", ["--yes", "--new-pool", "--host", "198.51.100.7"]
        )
    assert captured["opts"].allow_new_pool is True
    captured2 = _capture_opts(monkeypatch)
    with pytest.raises(SystemExit):
        mesh_wizard.run_mesh_wizard(
            "mesh-coordinator", ["--yes", "--host", "198.51.100.7"]
        )
    assert captured2["opts"].allow_new_pool is None
