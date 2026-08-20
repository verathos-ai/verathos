"""`verathos mesh manage`: the day-2 operator board over shared flows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

import verallm.mesh.flows as flows
import verallm.mesh.manage_board as board
from verallm.mesh.flows import WizardIO


def _io(answers: list[str], lines: list[str]) -> WizardIO:
    answer_iter = iter(answers)
    return WizardIO(ask=lambda _prompt: next(answer_iter), out=lines.append)


def _pool(tmp_path: Path) -> Path:
    (tmp_path / "pool-state.json").write_text('{"serving_mode": "dev"}')
    return tmp_path


def _view(meshes=None, workers=None):
    return {
        "live": True,
        "models": [],
        "models_detail": {},
        "meshes": dict(meshes or {}),
        "workers": dict(workers or {}),
    }


def test_board_renders_meshes_and_quits(tmp_path, monkeypatch):
    pool_dir = _pool(tmp_path)
    monkeypatch.setattr(
        flows,
        "_pool_view",
        lambda p: _view(
            meshes={
                "m-1": {
                    "model_id": "qwen2.5-7b-q4-k-m",
                    "status": "serving",
                    "routing_ready": True,
                    "driver": "w-a",
                    "members": ["w-a"],
                }
            },
            workers={
                "w-a": {
                    "status": "serving",
                    "capability": {"gpu_name": "RTX 5090", "vram_gb": 31},
                    "catalog": [],
                }
            },
        ),
    )
    from verallm.mesh import model_catalog

    monkeypatch.setattr(
        model_catalog, "assemble_candidates", lambda view, network="", **kw: []
    )
    lines: list[str] = []
    io = _io(["11"], lines)  # quit
    args = argparse.Namespace(pool=str(pool_dir))
    assert board.run_mesh_manage(args, io=io) == 0
    text = "\n".join(lines)
    assert "m-1" in text
    assert "qwen2.5-7b-q4-k-m" in text
    assert "meshes (1)" in text


def test_launch_action_delegates_to_the_shared_flow(tmp_path, monkeypatch):
    pool_dir = _pool(tmp_path)
    monkeypatch.setattr(flows, "_pool_view", lambda p: _view())
    from verallm.mesh import model_catalog

    monkeypatch.setattr(
        model_catalog, "assemble_candidates", lambda view, network="", **kw: []
    )
    launched: list = []
    monkeypatch.setattr(
        flows,
        "launch_flow",
        lambda io, p, **kw: launched.append((str(p), kw)) or "verified",
    )
    lines: list[str] = []
    io = _io(["1", "11"], lines)  # launch, then quit
    board.run_mesh_manage(argparse.Namespace(pool=str(pool_dir)), io=io)
    assert launched and launched[0][0] == str(pool_dir)
    assert launched[0][1].get("show_panel") is False


def test_stop_action_posts_pool_stop_and_waits(tmp_path, monkeypatch):
    pool_dir = _pool(tmp_path)
    state = {"stopped": False}

    def fake_view(p):
        if state["stopped"]:
            return _view()
        return _view(
            meshes={
                "m-x": {
                    "model_id": "q7b",
                    "status": "serving",
                    "routing_ready": True,
                    "driver": "w-a",
                    "members": ["w-a"],
                }
            }
        )

    monkeypatch.setattr(flows, "_pool_view", fake_view)
    from verallm.mesh import model_catalog

    monkeypatch.setattr(
        model_catalog, "assemble_candidates", lambda view, network="", **kw: []
    )
    ran: list[list[str]] = []

    def fake_run_cli(argv):
        ran.append(argv)
        if argv[:2] == ["pool", "stop"]:
            state["stopped"] = True
            return 0, json.dumps({"status": "stopping", "mesh_key": "m-x"})
        return 0, ""

    monkeypatch.setattr(flows, "_run_cli", fake_run_cli)
    lines: list[str] = []
    # stop -> single mesh auto-picked -> confirm (Enter) -> quit
    io = _io(["3", "", "11"], lines)
    board.run_mesh_manage(argparse.Namespace(pool=str(pool_dir)), io=io)
    stop = next(a for a in ran if a[:2] == ["pool", "stop"])
    assert "--mesh-key" in stop and "m-x" in stop
    text = "\n".join(lines)
    assert "GPUs are free" in text


def test_chat_action_execs_with_the_picked_mesh_key(tmp_path, monkeypatch):
    pool_dir = _pool(tmp_path)
    monkeypatch.setattr(
        flows,
        "_pool_view",
        lambda p: _view(
            meshes={
                "m-a": {
                    "model_id": "q7b",
                    "status": "serving",
                    "routing_ready": True,
                    "driver": "w-a",
                    "members": ["w-a"],
                }
            }
        ),
    )
    from verallm.mesh import model_catalog

    monkeypatch.setattr(
        model_catalog, "assemble_candidates", lambda view, network="", **kw: []
    )
    # In-process on purpose: a child interpreter re-pays every import and
    # looked like a hard hang under disk-saturating model loads.
    from verallm.mesh import cli as mesh_cli

    chats: list[argparse.Namespace] = []
    monkeypatch.setattr(
        mesh_cli, "cmd_mesh_chat", lambda ns: chats.append(ns)
    )
    lines: list[str] = []
    io = _io(["5", "11"], lines)  # chat (single mesh auto-picked), quit
    board.run_mesh_manage(argparse.Namespace(pool=str(pool_dir)), io=io)
    assert chats, "chat never launched"
    ns = chats[0]
    assert ns.mesh_key == "m-a"
    assert ns.pool == str(pool_dir)
    # The parser's defaults ride along; 4096 is the usable reply budget.
    assert ns.max_tokens == 4096


def test_manage_requires_a_pool(tmp_path, monkeypatch):
    from verallm.mesh import pool as pool_module

    monkeypatch.setattr(pool_module, "known_pool_dirs", lambda: [])
    with pytest.raises(SystemExit, match="no pool"):
        board.run_mesh_manage(
            argparse.Namespace(pool=""), io=_io([], [])
        )


def test_manage_parser_is_wired():
    from verallm.mesh import cli as mesh_cli

    parser = mesh_cli.build_parser()
    args = parser.parse_args(["manage", "--pool", "/x"])
    assert args.func is mesh_cli.cmd_mesh_manage
    assert args.pool == "/x"


def test_read_only_actions_do_not_redraw_the_board(tmp_path, monkeypatch):
    """After an action the menu returns behind the COMPACT overview
    (machines + meshes, no latency map) and must NOT re-assemble the
    catalog - that stays on the entry render and the explicit redraw."""

    pool_dir = _pool(tmp_path)
    monkeypatch.setattr(flows, "_pool_view", lambda p: _view())
    from verallm.mesh import model_catalog

    calls = {"assemble": 0}

    def fake_assemble(view, network="", **kw):
        calls["assemble"] += 1
        return []

    monkeypatch.setattr(model_catalog, "assemble_candidates", fake_assemble)
    lines: list[str] = []
    # machines -> back, then quit: only the ENTRY board assembles.
    io = _io(["7", "4", "11"], lines)
    board.run_mesh_manage(argparse.Namespace(pool=str(pool_dir)), io=io)
    assert calls["assemble"] == 1
    text = "\n".join(lines)
    # The compact overview re-rendered the machines panel (entry + after
    # the action) without the latency links line.
    assert text.count("machines (") >= 2
    assert "links" not in text.split("machines (")[-1].split("what do")[0]


def test_dev_pool_action_8_upgrades_in_place(tmp_path, monkeypatch):
    """Action 8 on a local-only pool flips it to subnet mode IN PLACE:
    stops meshes, rewrites the SAME state file with the chain binding
    (same pool id, same secrets), restarts the manager with the wallet,
    rejoins the local workers, and lands in the deploy lane. No retire,
    no new pool dir."""

    import verallm.mesh.onboarding as onboarding

    pool_dir = _pool(tmp_path)
    (pool_dir / "pool-state.json").write_text(
        json.dumps(
            {
                "version": 1,
                "pool_id": "pool-dev123",
                "serving_mode": "dev",
                "manager_endpoint": "http://127.0.0.1:9500",
                "pool_secret": "worker-secret",
                "management_secret": "admin-secret",
                "workers": {"w-a": {"status": "idle"}},
                "meshes": {},
                "snapshot_generation": 0,
            }
        )
    )
    state = {"stopped": False}

    def fake_view(p):
        if state["stopped"]:
            return _view()
        return _view(
            meshes={
                "m-old": {
                    "model_id": "q7b",
                    "status": "serving",
                    "routing_ready": True,
                    "driver": "w-a",
                    "members": ["w-a"],
                }
            }
        )

    monkeypatch.setattr(flows, "_pool_view", fake_view)
    from verallm.mesh import model_catalog

    monkeypatch.setattr(
        model_catalog, "assemble_candidates", lambda view, network="", **kw: []
    )
    ran: list[list[str]] = []

    def fake_run_cli(argv):
        ran.append(argv)
        if argv[:2] == ["pool", "stop"]:
            state["stopped"] = True
        return 0, ""

    monkeypatch.setattr(flows, "_run_cli", fake_run_cli)
    opts = onboarding.MeshCoordinatorOptions(
        serving_mode="subnet",
        owner_account="5HEDSywHjCaxgdGZ5CFN36LywPbBtaG8juZdKrjcrWWuJnJh",
        coordinator_address="0x0eeba349CB4473c011805603AfD9Aee40a923A59",
        validator_shared_state=str(tmp_path / "shared_state.json"),
        chain_id=945,
        netuid=405,
        coordinator_uid=1,
        epoch=123,
        worker_wallet_name="test_miner96",
        worker_wallet_hotkey="default",
        validator_allowlist_path=str(tmp_path / "validators.json"),
        manager_port=9500,
    )
    monkeypatch.setattr(
        board, "_ask_subnet_binding", lambda io, p: opts
    )
    started: list = []
    monkeypatch.setattr(
        onboarding,
        "start_manager_for_pool",
        lambda p, o: started.append((str(p), o.serving_mode)),
    )
    joined: list = []
    monkeypatch.setattr(
        onboarding,
        "setup_mesh_worker",
        lambda token, passthrough: joined.append((token, list(passthrough))),
    )
    monkeypatch.setattr(board, "_local_gpu_group_spec", lambda: "0,1")
    deployed: list = []
    monkeypatch.setattr(
        flows, "_offer_deploy", lambda io, p: deployed.append(str(p))
    )
    pm2: list = []
    monkeypatch.setattr(
        board.subprocess, "run", lambda argv, **k: pm2.append(list(argv))
    )
    lines: list[str] = []
    # action 8; confirm upgrade (Enter = yes); then quit
    io = _io(["8", "", "11"], lines)
    board.run_mesh_manage(argparse.Namespace(pool=str(pool_dir)), io=io)

    # The SAME state file was upgraded in place: id + secrets preserved.
    upgraded = json.loads((pool_dir / "pool-state.json").read_text())
    assert upgraded["serving_mode"] == "subnet"
    assert upgraded["pool_id"] == "pool-dev123"
    assert upgraded["pool_secret"] == "worker-secret"
    assert upgraded["management_secret"] == "admin-secret"
    assert upgraded["validator_binding"]["netuid"] == 405
    assert upgraded["workers"] == {"w-a": {"status": "idle"}}
    assert any(a[:2] == ["pool", "stop"] for a in ran)
    assert ["pm2", "delete", "verathos-pool-manager"] in pm2
    assert started == [(str(pool_dir), "subnet")]
    # Local rejoin carries the signing wallet and the EXISTING topology.
    assert joined and joined[0][0].startswith("vtpool_")
    passthrough = joined[0][1]
    assert "--wallet-name" in passthrough
    assert passthrough[passthrough.index("--gpus") + 1] == "0,1"
    assert deployed == [str(pool_dir)]
    # No retired copy: the pool was never recreated.
    assert not list(tmp_path.parent.glob(f"{tmp_path.name}.retired-*"))


def test_upgrade_refuses_while_a_mesh_serves(tmp_path):
    """The writer itself refuses a non-stopped mesh: chain-bound
    snapshots must sign the new binding, so serving through an upgrade
    would keep stale identities alive."""

    from verallm.mesh.pool import upgrade_pool_state_to_subnet

    (tmp_path / "pool-state.json").write_text(
        json.dumps(
            {
                "pool_id": "pool-x",
                "serving_mode": "dev",
                "manager_endpoint": "http://127.0.0.1:9500",
                "pool_secret": "s",
                "management_secret": "m",
                "workers": {},
                "meshes": {"m-1": {"status": "serving"}},
            }
        )
    )
    with pytest.raises(ValueError, match="stop the running meshes"):
        upgrade_pool_state_to_subnet(
            tmp_path,
            owner_account="5HEDSywHjCaxgdGZ5CFN36LywPbBtaG8juZdKrjcrWWuJnJh",
            coordinator_address="0x0eeba349CB4473c011805603AfD9Aee40a923A59",
            validator_shared_state_path="/tmp/shared.json",
            chain_id=945,
            netuid=405,
            coordinator_uid=1,
            epoch=1,
        )


def test_upgrade_refuses_remote_http_and_already_subnet(tmp_path):
    from verallm.mesh.pool import upgrade_pool_state_to_subnet

    base = {
        "pool_id": "pool-x",
        "serving_mode": "dev",
        "manager_endpoint": "http://203.0.113.5:9500",
        "pool_secret": "s",
        "management_secret": "m",
        "workers": {},
        "meshes": {},
    }
    kwargs = dict(
        owner_account="5HEDSywHjCaxgdGZ5CFN36LywPbBtaG8juZdKrjcrWWuJnJh",
        coordinator_address="0x0eeba349CB4473c011805603AfD9Aee40a923A59",
        validator_shared_state_path="/tmp/shared.json",
        chain_id=945,
        netuid=405,
        coordinator_uid=1,
        epoch=1,
    )
    (tmp_path / "pool-state.json").write_text(json.dumps(base))
    with pytest.raises(ValueError, match="HTTPS"):
        upgrade_pool_state_to_subnet(tmp_path, **kwargs)
    # An https endpoint override fixes it and re-mints the token files
    # with the new endpoint (same secret).
    _dir, token, endpoint_changed = upgrade_pool_state_to_subnet(
        tmp_path, manager_endpoint="https://203.0.113.5:9543", **kwargs
    )
    assert endpoint_changed is True
    assert token.manager_endpoint == "https://203.0.113.5:9543"
    assert token.pool_secret == "s"
    (tmp_path / "pool-state.json").write_text(
        json.dumps({**base, "serving_mode": "subnet"})
    )
    with pytest.raises(ValueError, match="already a subnet pool"):
        upgrade_pool_state_to_subnet(tmp_path, **kwargs)


@pytest.mark.parametrize("stored_mode", ["subnet", "validator"])
def test_go_live_on_a_subnet_pool_offers_deploy_not_recreate(
    tmp_path, monkeypatch, stored_mode
):
    """Action 8 on a pool that already IS subnet mode goes straight to the
    deploy lane. State files persist "subnet" since the mode rename (and
    "validator" before it); comparing the raw literal shipped a board that
    offered to retire and recreate an already-subnet pool."""

    (tmp_path / "pool-state.json").write_text(
        json.dumps({"serving_mode": stored_mode})
    )
    monkeypatch.setattr(flows, "_pool_view", lambda p: _view())
    from verallm.mesh import model_catalog

    monkeypatch.setattr(
        model_catalog, "assemble_candidates", lambda view, network="", **kw: []
    )
    deployed: list[str] = []
    monkeypatch.setattr(
        flows, "_offer_deploy", lambda io, p: deployed.append(str(p))
    )

    def _never_migrate(io, pool_dir):
        raise AssertionError(
            "an already-subnet pool must never be offered the recreate flow"
        )

    monkeypatch.setattr(board, "_migrate_to_subnet", _never_migrate)
    lines: list[str] = []
    io = _io(["8", "11"], lines)
    board.run_mesh_manage(argparse.Namespace(pool=str(tmp_path)), io=io)
    assert deployed == [str(tmp_path)]


@pytest.mark.parametrize(
    ("stored_mode", "label"),
    [
        ("subnet", "on the subnet"),
        ("validator", "on the subnet"),
        ("dev", "local only"),
    ],
)
def test_board_title_shows_the_serving_mode(
    tmp_path, monkeypatch, stored_mode, label
):
    (tmp_path / "pool-state.json").write_text(
        json.dumps({"serving_mode": stored_mode})
    )
    monkeypatch.setattr(flows, "_pool_view", lambda p: _view())
    from verallm.mesh import model_catalog

    monkeypatch.setattr(
        model_catalog, "assemble_candidates", lambda view, network="", **kw: []
    )
    lines: list[str] = []
    io = _io(["11"], lines)
    board.run_mesh_manage(argparse.Namespace(pool=str(tmp_path)), io=io)
    assert label in "\n".join(lines)


def test_mesh_picker_backs_out_to_the_menu(tmp_path, monkeypatch):
    """b at any sub-picker returns to the board menu instead of acting."""

    pool_dir = _pool(tmp_path)
    serving = {
        "model_id": "q7b",
        "status": "serving",
        "routing_ready": True,
        "driver": "w-a",
        "members": ["w-a"],
    }
    monkeypatch.setattr(
        flows,
        "_pool_view",
        lambda p: _view(meshes={"m-a": dict(serving), "m-b": dict(serving)}),
    )
    from verallm.mesh import model_catalog

    monkeypatch.setattr(
        model_catalog, "assemble_candidates", lambda view, network="", **kw: []
    )
    execs: list[list[str]] = []
    monkeypatch.setattr(
        board.subprocess, "run", lambda argv, **k: execs.append(list(argv))
    )
    lines: list[str] = []
    # chat -> two meshes to pick from -> b (back) -> quit
    io = _io(["5", "b", "11"], lines)
    board.run_mesh_manage(argparse.Namespace(pool=str(pool_dir)), io=io)
    assert execs == [], "back must not launch the chat"


def test_watch_action_reattaches_to_a_forming_mesh(tmp_path, monkeypatch):
    """Action 2 attaches wait_for_mesh to a forming mesh and probes on
    success, so closing the launch flow never orphans the progress."""

    pool_dir = _pool(tmp_path)
    monkeypatch.setattr(
        flows,
        "_pool_view",
        lambda p: _view(
            meshes={
                "m-f": {
                    "model_id": "big-model",
                    "status": "driving",
                    "driver": "w-a",
                    "members": ["w-a", "w-b"],
                }
            }
        ),
    )
    from verallm.mesh import model_catalog

    monkeypatch.setattr(
        model_catalog, "assemble_candidates", lambda view, network="", **kw: []
    )
    waited: list[tuple[str, str]] = []
    probed: list[str] = []
    monkeypatch.setattr(
        flows,
        "wait_for_mesh",
        lambda io, p, mesh_key, model_id, **kw: waited.append(
            (mesh_key, model_id)
        )
        or True,
    )
    monkeypatch.setattr(
        flows,
        "run_verified_probe",
        lambda io, p, mesh_key: probed.append(mesh_key),
    )
    lines: list[str] = []
    io = _io(["2", "11"], lines)  # watch (single forming mesh), quit
    board.run_mesh_manage(argparse.Namespace(pool=str(pool_dir)), io=io)
    assert waited == [("m-f", "big-model")]
    assert probed == ["m-f"]


def test_pool_view_carries_subnet_identity_from_the_state_file(
    tmp_path, monkeypatch
):
    """With the manager down, the view still names mode/network/hotkey and
    the stored registration from pool-state.json, normalizing the legacy
    "validator" mode literal."""
    from verallm.mesh import onboarding

    monkeypatch.setattr(
        onboarding, "pool_management_status", lambda p, timeout=5.0: None
    )
    (tmp_path / "pool-state.json").write_text(
        json.dumps(
            {
                "serving_mode": "validator",
                "subtensor_network": "test",
                "wallet_name": "test_miner96",
                "wallet_hotkey": "default",
                "validator_binding": {"netuid": 405, "chain_id": 945},
                "mesh_registration": {"mesh_key": "m-1", "index": 40},
                "workers": {},
                "meshes": {},
            }
        )
    )
    view = flows._pool_view(tmp_path)
    assert view["live"] is False
    assert view["serving_mode"] == "subnet"
    assert view["subtensor_network"] == "test"
    assert view["wallet_name"] == "test_miner96"
    assert view["validator_binding"]["netuid"] == 405
    assert view["mesh_registration"]["index"] == 40


def test_registration_menu_shows_running_deploy_instead_of_register(
    monkeypatch,
):
    """A model with a live deploy marker shows 'deploy in progress' and
    picking it does NOTHING - the register offer was one keypress from two
    deploys racing the same chain slot ."""

    from verallm.mesh import manage_board

    view = {
        "live": True,
        "serving_mode": "subnet",
        "meshes": {
            "m-glm": {
                "model_id": "glm-5.2-iq2-m",
                "status": "serving",
            }
        },
        "deploys": {"glm-5.2-iq2-m": {"stage": "gate", "updated_at_unix": 1}},
        "workers": {},
        "models_detail": {},
        "mesh_registrations": {},
        "mesh_registration": {},
    }
    monkeypatch.setattr(flows, "pool_registrations", lambda v: {})
    offered = []
    monkeypatch.setattr(
        flows,
        "_offer_deploy",
        lambda io, pool_dir, **kw: offered.append(kw),
    )
    lines: list[str] = []
    io = _io(["1"], lines)  # pick the deploy-in-progress row
    manage_board._subnet_registration(io, Path("/tmp/p"), view)
    text = "\n".join(lines)
    assert "deploy in progress" in text
    assert "stage: gate" in text
    assert offered == [], "picking the running deploy must not deploy again"


def test_registration_menu_lists_unregistered_first_with_sections(
    monkeypatch,
):
    """Unregistered models lead the list under a 'ready to register'
    header; registered ones sit below an 'already registered' header, so
    the Enter default targets a fresh registration and a re-deploy takes
    a deliberate pick. Numbering counts selectable rows only, in display
    order (headers are unnumbered)."""

    from verallm.mesh import manage_board

    view = {
        "live": True,
        "serving_mode": "subnet",
        "meshes": {
            # Sorted mesh iteration would put the registered glm first;
            # the menu must still list the unregistered 9b on top.
            "m-aaa-glm": {"model_id": "glm-5.2-iq2-m", "status": "serving"},
            "m-zzz-9b": {"model_id": "qwen3.5-9b-q4-k-xl", "status": "serving"},
        },
        "deploys": {},
        "workers": {},
        "models_detail": {},
        "mesh_registrations": {},
        "mesh_registration": {},
    }
    monkeypatch.setattr(
        flows,
        "pool_registrations",
        lambda v: {
            "glm-5.2-iq2-m": {"index": 46, "mesh_key": "m-aaa-glm"}
        },
    )
    offered = []
    monkeypatch.setattr(
        flows,
        "_offer_deploy",
        lambda io, pool_dir, **kw: offered.append(kw),
    )

    # Default Enter picks row 1 = the unregistered model.
    lines: list[str] = []
    manage_board._subnet_registration(_io([""], lines), Path("/tmp/p"), view)
    assert offered == [
        {"mesh_key": "m-zzz-9b", "model_id": "qwen3.5-9b-q4-k-xl"}
    ]
    text = "\n".join(lines)
    assert "ready to register" in text
    assert "already registered" in text
    assert "RE-deploy" in text
    # The unregistered row is numbered 1, the registered one 2.
    assert text.index("1) ") < text.index("qwen3.5-9b-q4-k-xl")
    unreg_pos = text.index("ready to register")
    reg_pos = text.index("already registered")
    assert unreg_pos < reg_pos

    # Picking 2 maps to the registered model despite the headers.
    offered.clear()
    manage_board._subnet_registration(_io(["2"], []), Path("/tmp/p"), view)
    assert offered == [
        {"mesh_key": "m-aaa-glm", "model_id": "glm-5.2-iq2-m"}
    ]
