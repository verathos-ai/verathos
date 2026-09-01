"""The guided setup wizard: the interactive front-end over onboarding.py.

The wizard owns no machinery: pool creation, PM2, health polling and the
idempotency guard live in verallm/mesh/onboarding.py (tested in
test_mesh_onboarding.py). These tests pin the front-end contract: answers
map onto MeshCoordinatorOptions, existing state is offered for reuse
instead of duplicated, and the worker flow validates a pasted token
before acting.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import verallm.mesh.onboarding as onboarding
import verallm.mesh.flows as flows
import verallm.mesh.setup_wizard as wizard
from verallm.mesh.onboarding import (
    ExistingSetup,
    ManagerProbe,
    MeshCoordinatorResult,
)
from verallm.mesh.setup_wizard import WizardIO, run_mesh_setup


def _fake_result(tmp_path: Path) -> MeshCoordinatorResult:
    return MeshCoordinatorResult(
        pool_dir=tmp_path / "pool-fresh",
        pool_id="pool-fresh",
        manager_endpoint="http://198.51.100.7:9500",
        worker_token="vtpool_abc",
        admin_token_file=tmp_path / "pool-admin-token.txt",
        pm2_name="verathos-pool-manager",
        dashboard_url="http://198.51.100.7:9500/operator",
        join_command="curl ... --mesh-worker --token vtpool_abc",
    )


def _io(answers: list[str], lines: list[str]) -> WizardIO:
    answer_iter = iter(answers)
    return WizardIO(ask=lambda _prompt: next(answer_iter), out=lines.append)


def test_worker_flow_validates_a_pasted_token(monkeypatch):
    joined: list[tuple[str, list[str]]] = []
    monkeypatch.setattr(onboarding, "detect_public_ip", lambda: "")
    monkeypatch.setattr(
        onboarding,
        "describe_worker_token",
        lambda raw: {
            "pool_id": "pool-abc",
            "manager_endpoint": "http://198.51.100.7:9500",
            "loopback": False,
            "reachable": True,
            "reachable_pool_id": "pool-abc",
            "error": "",
        },
    )
    monkeypatch.setattr(
        onboarding,
        "setup_mesh_worker",
        lambda token, passthrough: joined.append((token, list(passthrough))),
    )
    lines: list[str] = []
    # role; token; advertise-host (skip); wallet (skip)
    io = _io(["2", "vtpool_sometoken", "", ""], lines)
    assert run_mesh_setup(object(), io=io) == 0
    assert joined == [("vtpool_sometoken", [])]
    assert "pool-abc" in "\n".join(lines)


def test_worker_flow_forwards_advertise_and_wallet(monkeypatch):
    joined: list[tuple[str, list[str]]] = []
    monkeypatch.setattr(onboarding, "detect_public_ip", lambda: "203.0.113.9")
    monkeypatch.setattr(
        onboarding,
        "describe_worker_token",
        lambda raw: {
            "pool_id": "pool-abc",
            "manager_endpoint": "http://198.51.100.7:9500",
            "loopback": False,
            "reachable": True,
            "reachable_pool_id": "pool-abc",
            "error": "",
        },
    )
    monkeypatch.setattr(
        onboarding,
        "setup_mesh_worker",
        lambda token, passthrough: joined.append((token, list(passthrough))),
    )
    lines: list[str] = []
    # role; token; advertise default (Enter); wallet; hotkey default; allowlist
    io = _io(["2", "vtpool_tok", "", "miner", "", "/tmp/allow.json"], lines)
    assert run_mesh_setup(object(), io=io) == 0
    token, passthrough = joined[0]
    assert token == "vtpool_tok"
    assert passthrough == [
        "--advertise-host",
        "203.0.113.9",
        "--wallet-name",
        "miner",
        "--wallet-hotkey",
        "default",
        "--validator-allowlist-path",
        "/tmp/allow.json",
    ]


def test_worker_flow_reads_a_token_file_with_diagnostics(
    monkeypatch, tmp_path
):
    """The file-path input once skipped every diagnostic the pasted-token
    path ran; both must decode, warn, and probe identically."""

    token_file = tmp_path / "token.txt"
    token_file.write_text("vtpool_fromfile\n")
    described: list[str] = []
    joined: list[tuple[str, list[str]]] = []
    monkeypatch.setattr(onboarding, "detect_public_ip", lambda: "")
    monkeypatch.setattr(
        onboarding,
        "describe_worker_token",
        lambda raw: described.append(raw)
        or {
            "pool_id": "pool-abc",
            "manager_endpoint": "http://198.51.100.7:9500",
            "loopback": False,
            "reachable": True,
            "reachable_pool_id": "pool-abc",
            "error": "",
        },
    )
    monkeypatch.setattr(
        onboarding,
        "setup_mesh_worker",
        lambda token, passthrough: joined.append((token, list(passthrough))),
    )
    lines: list[str] = []
    io = _io(["2", str(token_file), "", ""], lines)
    assert run_mesh_setup(object(), io=io) == 0
    assert described == ["vtpool_fromfile"]
    assert joined == [("vtpool_fromfile", [])]


def test_worker_flow_refuses_an_unreachable_token(monkeypatch):
    joined: list = []
    monkeypatch.setattr(
        onboarding,
        "describe_worker_token",
        lambda raw: {
            "pool_id": "pool-abc",
            "manager_endpoint": "http://10.0.0.9:9500",
            "loopback": False,
            "reachable": False,
            "reachable_pool_id": "",
            "error": "connection refused",
        },
    )
    monkeypatch.setattr(
        onboarding, "setup_mesh_worker", lambda *a: joined.append(a)
    )
    lines: list[str] = []
    # decline the "join anyway" confirm
    io = _io(["2", "vtpool_sometoken", "n"], lines)
    assert run_mesh_setup(object(), io=io) == 2
    assert joined == []
    assert "does not answer" in "\n".join(lines)


def test_worker_flow_warns_on_a_loopback_token(monkeypatch):
    joined: list = []
    monkeypatch.setattr(
        onboarding,
        "describe_worker_token",
        lambda raw: {
            "pool_id": "pool-abc",
            "manager_endpoint": "http://127.0.0.1:9500",
            "loopback": True,
            "reachable": True,
            "reachable_pool_id": "pool-abc",
            "error": "",
        },
    )
    monkeypatch.setattr(onboarding, "detect_public_ip", lambda: "")
    monkeypatch.setattr(
        onboarding,
        "setup_mesh_worker",
        lambda token, passthrough: joined.append(token),
    )
    lines: list[str] = []
    io = _io(["2", "vtpool_sometoken", "", ""], lines)
    assert run_mesh_setup(object(), io=io) == 0
    assert joined == ["vtpool_sometoken"]
    assert "127.0.0.1" in "\n".join(lines)


def test_known_pool_registry_round_trip(tmp_path, monkeypatch):
    import verallm.mesh.pool as pool_module

    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    import tempfile

    monkeypatch.setattr(
        tempfile, "gettempdir", lambda: str(tmp_path / "faketmp")
    )
    a = tmp_path / "somewhere" / "pool-aaaa"
    b = tmp_path / "elsewhere" / "pool-bbbb"
    for d in (a, b):
        d.mkdir(parents=True)
        pool_module.record_known_pool(d)
    pool_module.record_known_pool(a)  # idempotent
    assert pool_module.known_pool_dirs() == [a.resolve(), b.resolve()]
    # Vanished dirs drop out instead of breaking discovery.
    import shutil

    shutil.rmtree(b)
    assert pool_module.known_pool_dirs() == [a.resolve()]


def test_create_pool_state_records_itself(tmp_path, monkeypatch):
    import verallm.mesh.pool as pool_module

    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    import tempfile

    monkeypatch.setattr(
        tempfile, "gettempdir", lambda: str(tmp_path / "faketmp")
    )
    out, _token = pool_module.create_pool_state(
        str(tmp_path / "root"),
        manager_endpoint="http://127.0.0.1:9599",
        serving_mode="dev",
    )
    assert out.resolve() in pool_module.known_pool_dirs()


def test_zero_flag_discovery_finds_registered_pool(tmp_path, monkeypatch):
    import argparse

    import verallm.mesh.cli as cli_module
    import verallm.mesh.pool as pool_module

    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    import tempfile

    monkeypatch.setattr(
        tempfile, "gettempdir", lambda: str(tmp_path / "faketmp")
    )
    assert pool_module is not None
    out, _token = pool_module.create_pool_state(
        str(tmp_path / "anywhere"),
        manager_endpoint="http://127.0.0.1:9598",
        serving_mode="dev",
    )
    args = argparse.Namespace(pool="", pool_token="", pool_token_file="")
    token = cli_module._resolve_pool_context(args)
    assert token.manager_endpoint == "http://127.0.0.1:9598"


def test_gpu_worker_refuses_numpy_sumcheck_fallback(monkeypatch):
    import verallm.mesh.cli as cli_module
    from zkllm.crypto import sumcheck_fast

    monkeypatch.setattr(
        cli_module, "_detect_gpu_capability", lambda: ("RTX 5090", 31, ["RTX 5090"], [31])
    )
    monkeypatch.setattr(sumcheck_fast, "_HAS_CUDA", False, raising=False)
    monkeypatch.setattr(sumcheck_fast, "_HAS_NATIVE", False, raising=False)
    monkeypatch.delenv("VERATHOS_ALLOW_SLOW_SUMCHECK", raising=False)
    with pytest.raises(SystemExit, match="NumPy fallback"):
        cli_module._require_native_prover_on_gpu_boxes()
    # Explicit dev escape hatch still works.
    monkeypatch.setenv("VERATHOS_ALLOW_SLOW_SUMCHECK", "1")
    cli_module._require_native_prover_on_gpu_boxes()
    # CPU-only machines (no GPU detected) keep the warning-only behavior.
    monkeypatch.delenv("VERATHOS_ALLOW_SLOW_SUMCHECK", raising=False)
    monkeypatch.setattr(
        cli_module, "_detect_gpu_capability", lambda: ("", 0, [], [])
    )
    cli_module._require_native_prover_on_gpu_boxes()


def test_footer_reports_resolved_tier_and_proof_seconds():
    from verallm.mesh.cli import _proof_seconds, _resolved_tier_label
    from verallm.mesh.ggml_proof import (
        VERATHOS_GGML_GEMM_PROOF_MODE,
        VERATHOS_GGML_LIGHT_PROOF_MODE,
    )

    # An "auto" session reports what actually ran, not the word auto.
    assert (
        _resolved_tier_label({"proof_mode": VERATHOS_GGML_LIGHT_PROOF_MODE}, "auto")
        == "light"
    )
    assert (
        _resolved_tier_label({"proof_mode": VERATHOS_GGML_GEMM_PROOF_MODE}, "auto")
        == "hard"
    )
    assert _resolved_tier_label({}, "hard") == "hard"

    # 20 tokens at 100 tok/s = 0.2s decode; 5.0 total minus ttft and decode.
    final = {
        "total_s": 5.0,
        "engine_tps": 100.0,
        "ttft_s": 0.3,
        "usage": {"completion_tokens": 20},
    }
    assert _proof_seconds(final) == pytest.approx(4.5, abs=0.01)

    # Missing inputs yield None rather than an invented number, and a
    # light reply whose remainder is noise reports nothing.
    assert _proof_seconds({"total_s": 5.0}) is None
    assert _proof_seconds({**final, "total_s": 0.51}) is None


def test_chat_footer_helpers_exist_and_are_wired():
    # These were deleted once while their call sites remained, making the
    # chat footer a NameError. Pin both the helpers and the call sites.
    import inspect

    import verallm.mesh.cli as cli_module

    source = inspect.getsource(cli_module.cmd_mesh_chat)
    assert "_resolved_tier_label(" in source
    assert "_proof_seconds(" in source
    assert callable(cli_module._resolved_tier_label)
    assert callable(cli_module._proof_seconds)


def test_serve_never_enables_anchor_capture_by_default():
    # Anchoring every weight tensor on every request is the full-inventory
    # design v3 measured at +51.5% and rejected; the CPU backend also caps
    # at 64 stages, which would truncate the inventory nondeterministically.
    # Anchoring is armed per request through the capture token, never here.
    import inspect

    import verallm.mesh.cli as cli_module

    assert "VERATHOS_GGML_ANCHOR_TENSORS" not in inspect.getsource(
        cli_module.cmd_serve
    )


def test_pool_view_prefers_the_live_manager(monkeypatch, tmp_path):
    """Reading pool-state.json behind a live manager is how the wizard once
    said 'a mesh is already serving' about a long-stopped mesh."""

    (tmp_path / "pool-state.json").write_text(
        '{"meshes": {"m-ghost": {"status": "serving"}}, '
        '"model_registry": {"stale-model": {}}}'
    )
    monkeypatch.setattr(
        onboarding,
        "pool_management_status",
        lambda pool_dir, timeout=5.0: {
            "meshes": {
                "m-live": {
                    "status": "serving",
                    "routing_ready": True,
                    "model_id": "m1",
                }
            },
            "models": {"m1": {}},
            "workers": {},
        },
    )
    view = flows._pool_view(tmp_path)
    assert view["live"] is True
    assert list(view["meshes"]) == ["m-live"]
    assert flows._serving_mesh(tmp_path) == "m-live"
    assert flows._pool_models(tmp_path) == ["m1"]


def test_serving_mesh_requires_routing_ready_on_live_view(monkeypatch, tmp_path):
    monkeypatch.setattr(
        onboarding,
        "pool_management_status",
        lambda pool_dir, timeout=5.0: {
            "meshes": {
                "m-dead": {
                    "status": "serving",
                    "routing_ready": False,
                    "driver_stale": True,
                }
            },
            "models": {},
            "workers": {},
        },
    )
    assert flows._serving_mesh(tmp_path) == ""


def test_pool_view_falls_back_to_disk_when_manager_is_down(
    monkeypatch, tmp_path
):
    (tmp_path / "pool-state.json").write_text(
        '{"meshes": {"m-x": {"status": "serving"}}, '
        '"model_registry": {"m9": {}}}'
    )
    monkeypatch.setattr(
        onboarding, "pool_management_status", lambda *a, **k: None
    )
    view = flows._pool_view(tmp_path)
    assert view["live"] is False
    # With no live manager NOTHING is claimable as serving (chat, launch,
    # and probe all need the manager); only the model list falls back.
    assert flows._serving_mesh(tmp_path) == ""
    assert flows._pool_models(tmp_path) == ["m9"]


def test_offer_deploy_skips_local_only_pools(monkeypatch, tmp_path):
    (tmp_path / "pool-state.json").write_text('{"serving_mode": "dev"}')
    lines: list[str] = []
    io = _io([], lines)
    flows._offer_deploy(io, tmp_path)
    assert "local-only" in "\n".join(lines)


def test_offer_deploy_runs_the_deploy_pipeline(monkeypatch, tmp_path):
    (tmp_path / "pool-state.json").write_text(
        '{"serving_mode": "validator", '
        '"validator_binding": {"chain_id": 945}}'
    )
    monkeypatch.setattr(flows, "_serving_mesh", lambda pool_dir: "m-1")
    monkeypatch.setattr(
        flows,
        "_pool_view",
        lambda pool_dir: {
            "live": True,
            "models": ["qwen2.5-7b"],
            "meshes": {
                "m-1": {
                    "status": "serving",
                    "model_id": "qwen2.5-7b",
                    "driver": "w-drv",
                }
            },
            "workers": {
                "w-drv": {
                    "endpoints": {"mesh": "http://198.51.100.7:9443"}
                }
            },
        },
    )
    monkeypatch.setattr(
        flows,
        "_network_binding_for_pool",
        lambda pool_dir: {
            "network": "testnet",
            "chain_config": "/repo/chain_config_testnet.json",
            "chain_id": 945,
            "netuid": 405,
        },
    )
    calls: list[list[str]] = []

    class _Done:
        returncode = 0

    monkeypatch.setattr(
        flows.subprocess, "run", lambda argv, **k: calls.append(argv) or _Done()
    )
    lines: list[str] = []
    # confirm deploy; accept default endpoint; wallet name; default hotkey
    io = _io(["y", "", "miner-wallet", ""], lines)
    flows._offer_deploy(io, tmp_path)
    assert len(calls) == 1
    argv = calls[0]
    joined = " ".join(argv)
    assert "deploy" in argv
    assert "qwen2.5-7b" in argv
    assert "--chain-config" in argv
    assert "/repo/chain_config_testnet.json" in argv
    assert "--endpoint" in argv
    # The ADVERTISED scheme is kept: fabricating https:// against the
    # driver's plain-HTTP port failed three hard probe checks.
    assert "http://198.51.100.7:9443" in joined
    assert "--wallet" in argv and "miner-wallet" in argv
    assert str(tmp_path / "pool-admin-token.txt") in argv


def test_offer_deploy_declined_prints_the_manual_lane(monkeypatch, tmp_path):
    (tmp_path / "pool-state.json").write_text(
        '{"serving_mode": "validator", '
        '"validator_binding": {"chain_id": 945}}'
    )
    monkeypatch.setattr(flows, "_serving_mesh", lambda pool_dir: "m-1")
    monkeypatch.setattr(
        flows,
        "_pool_view",
        lambda pool_dir: {
            "live": True,
            "models": [],
            "meshes": {"m-1": {"status": "serving", "model_id": "q"}},
            "workers": {},
        },
    )
    ran: list = []
    monkeypatch.setattr(
        flows.subprocess, "run", lambda *a, **k: ran.append(a)
    )
    lines: list[str] = []
    io = _io(["n"], lines)
    flows._offer_deploy(io, tmp_path)
    assert ran == []
    assert "verathos mesh deploy" in "\n".join(lines)


def test_subnet_single_box_flow_builds_valid_options(tmp_path, monkeypatch):
    """The 'on the subnet' path once collected eight answers and then died
    on the HTTPS preflight; single-box subnet pools are loopback-legal and
    must carry the worker signing identity."""

    captured: dict = {}
    monkeypatch.setattr(
        onboarding,
        "scan_existing_setup",
        lambda *a, **k: ExistingSetup(
            pools=[], manager=ManagerProbe(reachable=False), pm2_apps=[]
        ),
    )
    monkeypatch.setattr(onboarding, "detect_public_ip", lambda: "")

    def fake_setup(opts):
        captured["opts"] = opts
        return _fake_result(tmp_path)

    monkeypatch.setattr(onboarding, "setup_mesh_coordinator", fake_setup)
    monkeypatch.setattr(
        onboarding, "print_coordinator_result", lambda result: None
    )
    monkeypatch.setattr(wizard, "_finish_with_probe", lambda io, pool: None)

    lines: list[str] = []
    io = _io(
        [
            "1",            # role: coordinator
            "2",            # on the subnet
            "1",            # testnet
            "5FvtOwner",    # owner coldkey SS58
            "0xCoord",      # coordinator EVM address
            "",             # shared-state path (default)
            "ws://10.0.0.5:9944",  # own subtensor node (recommended)
            "10",           # miner UID
            "123",          # epoch
            "miner",        # worker signing wallet
            "",             # hotkey (default)
            "/tmp/allow.json",  # allowlist
            "1",            # single box (loopback manager)
            "",             # manager port (default)
            str(tmp_path / "pools"),  # state dir
        ],
        lines,
    )
    assert run_mesh_setup(object(), io=io) == 0
    assert captured["opts"].subtensor_endpoint == "ws://10.0.0.5:9944"
    opts = captured["opts"]
    assert opts.serving_mode == "subnet"
    assert opts.manager_host == "127.0.0.1"
    assert opts.netuid == 405
    assert opts.worker_wallet_name == "miner"
    assert opts.worker_wallet_hotkey == "default"
    assert opts.validator_allowlist_path == "/tmp/allow.json"
    # The worker allowlist refresher needs network + netuid (and the chain
    # config for the minValidatorStake filter when present); without them
    # the allowlist goes stale and the pool refuses the box as a driver.
    passthrough = onboarding.worker_passthrough(opts)
    assert passthrough[:6] == [
        "--wallet-name",
        "miner",
        "--wallet-hotkey",
        "default",
        "--validator-allowlist-path",
        "/tmp/allow.json",
    ]
    assert passthrough[6:8] == ["--subtensor-network", "test"]
    assert passthrough[8:10] == ["--netuid", "405"]


def test_subnet_multi_box_requires_https(tmp_path, monkeypatch):
    monkeypatch.setattr(
        onboarding,
        "scan_existing_setup",
        lambda *a, **k: ExistingSetup(
            pools=[], manager=ManagerProbe(reachable=False), pm2_apps=[]
        ),
    )
    monkeypatch.setattr(onboarding, "detect_public_ip", lambda: "")
    lines: list[str] = []
    io = _io(
        [
            "1", "2", "1", "5FvtOwner", "0xCoord", "", "", "10", "123",
            "", "2", "2", "http://plain.example:9500",
        ],
        lines,
    )
    with pytest.raises(SystemExit, match="https"):
        run_mesh_setup(object(), io=io)


def test_spinner_degrades_to_plain_change_lines_when_piped(capsys, monkeypatch):
    """Agents and pipes must get grep-able lines, never animation bytes."""

    from verallm.mesh import render

    monkeypatch.setenv("NO_COLOR", "1")
    with render.spinner("waiting for workers...") as spin:
        spin.update("waiting for workers... (1 joined)")
        spin.update("waiting for workers... (1 joined)")  # duplicate: silent
        spin.update("waiting for workers... (2 joined)")
    out = capsys.readouterr().out
    assert out == (
        "  waiting for workers...\n"
        "  waiting for workers... (1 joined)\n"
        "  waiting for workers... (2 joined)\n"
    )
    assert "\r" not in out and "\033" not in out


def test_discover_local_gguf_models_groups_shards(tmp_path):
    from verallm.mesh import units as units_module

    models = tmp_path / "models"
    models.mkdir()
    (models / "qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf").write_bytes(
        b"a" * 10
    )
    (models / "qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf").write_bytes(
        b"b" * 5
    )
    (models / "tiny-1b-q8_0.gguf").write_bytes(b"c" * 3)
    # Incomplete shard set: never offered.
    (models / "broken-13b-q4_k_m-00001-of-00003.gguf").write_bytes(b"d")
    found = units_module.discover_local_gguf_models([models])
    by_id = {entry["model_id"]: entry for entry in found}
    assert sorted(by_id) == ["qwen2.5-7b-instruct-q4_k_m", "tiny-1b-q8_0"]
    assert by_id["qwen2.5-7b-instruct-q4_k_m"]["bytes"] == 15
    assert len(by_id["qwen2.5-7b-instruct-q4_k_m"]["files"]) == 2
    assert len(by_id["tiny-1b-q8_0"]["files"]) == 1


def test_add_model_writes_catalog_entry_from_local_gguf(
    tmp_path, monkeypatch, capsys
):
    import argparse

    import verallm.mesh.cli as cli_module
    from verallm.mesh import gguf_manifest as manifest_module
    from verallm.mesh import units as units_module

    shard = tmp_path / "tiny-1b-q8_0.gguf"
    shard.write_bytes(b"x" * 64)
    catalog = tmp_path / "catalog.json"
    monkeypatch.setattr(
        units_module, "load_unit_registry", lambda path=None: None
    )
    built: list = []

    def fake_build(paths):
        built.append(paths)
        return {
            "version": 1,
            "model_files": [
                {"index": 0, "path": str(shard), "n_bytes": 64}
            ],
            "tensors": [
                {"name": "blk.0.attn_q.weight"},
                {"name": "blk.21.ffn_down.weight"},
                {"name": "output.weight"},
            ],
        }

    monkeypatch.setattr(
        cli_module, "_local_pool_views", lambda: [], raising=True
    )
    monkeypatch.setattr(
        manifest_module, "build_gguf_tensor_manifest", fake_build
    )
    monkeypatch.setattr(
        manifest_module,
        "save_gguf_tensor_manifest",
        lambda manifest, path: Path(path).write_text("{}") or Path(path),
    )
    args = argparse.Namespace(
        gguf=str(shard),
        model_id="",
        hf_repo="",
        hf_files="",
        catalog=str(catalog),
        no_restart=True,
    )
    cli_module.cmd_mesh_add_model(args)
    entries = json.loads(catalog.read_text())
    assert len(entries) == 1
    entry = entries[0]
    assert entry["model_id"] == "tiny-1b-q8_0"
    assert entry["llama_model"] == str(shard)
    assert entry["layers"] == 22  # max blk index 21 + 1
    assert entry["model_bytes"] == 64
    assert "hf_repo" not in entry
    # Re-run replaces, never duplicates.
    cli_module.cmd_mesh_add_model(args)
    assert len(json.loads(catalog.read_text())) == 1


def test_fit_annotation_matches_manager_sizing():
    from verallm.mesh import model_catalog

    workers = {
        "w-a": {"capability": {"vram_gb": 31}},
        "w-b": {"capability": {"vram_gb": 31}},
    }
    fits, text = model_catalog.fit_annotation(4_700_000_000, workers)
    assert fits and text == "fits on one GPU"
    fits, text = model_catalog.fit_annotation(30_000_000_000, workers)
    assert fits and "split across 2 GPUs" in text
    fits, text = model_catalog.fit_annotation(80_000_000_000, workers)
    assert not fits and "needs ~" in text
    fits, text = model_catalog.fit_annotation(0, workers)
    assert not fits and text == "size unknown"
    fits, text = model_catalog.fit_annotation(1, {})
    assert not fits and text == "no workers joined"


def test_fit_annotation_counts_gpus_inside_a_multi_gpu_worker():
    """One worker owning 4 GPUs must not read as one enormous GPU.

    The worker advertises the SUM in vram_gb, so sizing against that alone
    told the operator a 238 GB model 'fits on one GPU' of 80 GB.
    """
    from verallm.mesh import model_catalog

    four_a100 = {
        "w-quad": {
            "capability": {"vram_gb": 320, "per_gpu_vram_gb": [80, 80, 80, 80]}
        }
    }
    fits, text = model_catalog.fit_annotation(40_000_000_000, four_a100)
    assert fits and text == "fits on one GPU"
    fits, text = model_catalog.fit_annotation(238_577_580_768, four_a100)
    assert fits and text == "fits split across 4 GPUs on one machine"
    fits, text = model_catalog.fit_annotation(467_289_111_904, four_a100)
    assert not fits and "needs ~" in text

    # Staying on one machine wins over spanning two, even when both fit.
    two_machines = {
        "w-quad": {
            "capability": {"vram_gb": 320, "per_gpu_vram_gb": [80, 80, 80, 80]}
        },
        "w-pair": {"capability": {"vram_gb": 160, "per_gpu_vram_gb": [80, 80]}},
    }
    fits, text = model_catalog.fit_annotation(238_577_580_768, two_machines)
    assert fits and text == "fits split across 4 GPUs on one machine"
    # Larger than any single machine but within the pool: spans machines,
    # and says so (350 GB x 1.3 headroom = 455 GB, over one machine's 320).
    fits, text = model_catalog.fit_annotation(350_000_000_000, two_machines)
    assert fits and text == "fits split across 6 GPUs on 2 machines"


def test_assemble_candidates_merges_subnet_disk_and_pool(monkeypatch):
    from verallm.mesh import model_catalog
    from verallm.mesh import units as units_module

    monkeypatch.setattr(
        model_catalog,
        "subnet_model_catalog",
        lambda network, repo_root=None, **kw: [
            {"model_id": "qwen3.6-27b-q4-k-m", "layers": 64, "quant": "q4_k_m"},
            # Chain-listed but retired in the shipped catalogue: must be
            # hidden from new launches even while the chain (or its cache)
            # still lists it .
            {"model_id": "qwen2.5-7b-q4-k-m", "layers": 28, "quant": "q4_k_m"},
            {"model_id": "big-70b", "layers": 80, "quant": "q4_k_m"},
        ],
    )
    monkeypatch.setattr(
        units_module,
        "discover_local_gguf_models",
        lambda roots=None: [
            {
                "model_id": "qwen3.6-27b-q4-k-m",
                "files": [Path("/models/q.gguf")],
                "bytes": 16_817_244_384,
            },
            {
                "model_id": "local-extra-1b",
                "files": [Path("/models/x.gguf")],
                "bytes": 1_000_000_000,
            },
        ],
    )
    view = {
        "workers": {
            "w-a": {"capability": {"vram_gb": 31}, "catalog": []},
            "w-b": {"capability": {"vram_gb": 31}, "catalog": []},
        },
        "models_detail": {},
    }
    rows = model_catalog.assemble_candidates(view, network="testnet")
    by_id = {row["model_id"]: row for row in rows}
    qwen = by_id["qwen3.6-27b-q4-k-m"]
    assert qwen["launchable"] and qwen["fits"]
    assert qwen["where"] == "on this machine, no download"
    assert {"subnet", "disk"} <= qwen["origins"]
    # Retired in the shipped catalogue: never offered for new launches.
    assert "qwen2.5-7b-q4-k-m" not in by_id
    # Chain model with no source anywhere: listed, not launchable.
    big = by_id["big-70b"]
    assert not big["launchable"]
    assert "missing from the model catalogue" in big["where"]
    # Model onboarding is the subnet owner's process: a local GGUF the
    # chain does not list is not an operator-facing option at all.
    assert "local-extra-1b" not in by_id
    # Launchable subnet models sort first.
    assert rows[0]["model_id"] == "qwen3.6-27b-q4-k-m"


def test_assemble_candidates_folds_disk_files_under_catalogue_id(
    monkeypatch,
):
    """A disk shard set whose exact basenames and bytes match a catalogue
    variant folds under the canonical CHAIN-REGISTERED mesh id with its
    quant and score; without the chain registration the file is not an
    operator-facing option at all (onboarding is the owner's process)."""

    from verallm.mesh import model_catalog
    from verallm.mesh import units as units_module
    from verallm.registry.models import MESH_GGUF_MODELS

    entry, variant = MESH_GGUF_MODELS["deepseek-v4-flash-0731-iq1-m"]
    files = [Path("/models") / Path(f).name for f in variant.hf_files]
    monkeypatch.setattr(
        units_module,
        "discover_local_gguf_models",
        lambda roots=None: [
            {
                # Filename-derived id: never matches the registered id.
                "model_id": "DeepSeek-V4-Flash-0731-UD-IQ1_M",
                "files": files,
                "bytes": variant.model_bytes,
            },
        ],
    )
    view = {
        "workers": {
            "w-a": {"capability": {"vram_gb": 80}, "catalog": []},
            "w-b": {"capability": {"vram_gb": 80}, "catalog": []},
        },
        "models_detail": {},
    }

    # Not on the chain: hidden entirely, under either id.
    monkeypatch.setattr(
        model_catalog,
        "subnet_model_catalog",
        lambda network, repo_root=None, **kw: [],
    )
    rows = model_catalog.assemble_candidates(view, network="testnet")
    assert not rows

    # Chain-registered: the disk copy folds under the canonical id.
    monkeypatch.setattr(
        model_catalog,
        "subnet_model_catalog",
        lambda network, repo_root=None, **kw: [
            {
                "model_id": "deepseek-v4-flash-0731-iq1-m",
                "layers": 43,
                "quant": "iq1_m",
            }
        ],
    )
    rows = model_catalog.assemble_candidates(view, network="testnet")
    ids = {row["model_id"] for row in rows}
    assert "DeepSeek-V4-Flash-0731-UD-IQ1_M" not in ids
    row = {r["model_id"]: r for r in rows}["deepseek-v4-flash-0731-iq1-m"]
    assert row["disk_files"] == [str(f) for f in files]
    assert row["quant"] == "iq1_m"
    assert row["base_score"] > 0
    assert row["launchable"]


def test_assemble_candidates_hides_retired_models(monkeypatch):
    """A retired catalogue release is never offered for new launches; it
    only stays visible (unlaunchable) while the pool actually holds or
    serves it."""

    from verallm.mesh import model_catalog
    from verallm.mesh import units as units_module

    monkeypatch.setattr(
        model_catalog,
        "subnet_model_catalog",
        lambda network, repo_root=None, **kw: [],
    )
    monkeypatch.setattr(
        units_module,
        "discover_local_gguf_models",
        lambda roots=None: [
            {
                "model_id": "deepseek-v4-flash-iq3-xxs",
                "files": [Path("/models/d.gguf")],
                "bytes": 102_999_887_616,
            },
        ],
    )
    workers = {
        "w-a": {"capability": {"vram_gb": 80}, "catalog": []},
        "w-b": {"capability": {"vram_gb": 80}, "catalog": []},
    }

    # On disk only: the retired release disappears from the pick list.
    rows = model_catalog.assemble_candidates(
        {"workers": workers, "models_detail": {}}, network=""
    )
    assert "deepseek-v4-flash-iq3-xxs" not in {r["model_id"] for r in rows}

    # Actively served: still shown, but never launchable again.
    view = {
        "workers": {
            **workers,
            "w-c": {
                "capability": {"vram_gb": 80},
                "catalog": [
                    {
                        "model_id": "deepseek-v4-flash-iq3-xxs",
                        "layers": 43,
                        "model_bytes": 102_999_887_616,
                    }
                ],
            },
        },
        "models_detail": {},
        "meshes": {
            "m-old": {
                "model_id": "deepseek-v4-flash-iq3-xxs",
                "status": "serving",
                "driver": "w-c",
            }
        },
    }
    rows = model_catalog.assemble_candidates(view, network="")
    old = {r["model_id"]: r for r in rows}["deepseek-v4-flash-iq3-xxs"]
    assert old["retired"] and not old["launchable"]
    assert old["where"] == "retired · superseded release"


def test_finish_flow_launches_with_the_chosen_placement(
    tmp_path, monkeypatch
):
    """Pick from the catalog table -> full recommendation table -> launch
    carries the chosen workers and driver."""

    from verallm.mesh import model_catalog

    (tmp_path / "pool-state.json").write_text('{"serving_mode": "dev"}')
    serving = {"key": ""}
    monkeypatch.setattr(flows, "_serving_mesh", lambda pool_dir: serving["key"]
    )
    monkeypatch.setattr(
        flows,
        "_pool_view",
        lambda pool_dir: {
            "live": True,
            "models": ["qwen2.5-7b-q4-k-m"],
            "models_detail": {},
            "meshes": {},
            "workers": {
                "w-a": {"capability": {"vram_gb": 31}, "catalog": []},
            },
        },
    )
    monkeypatch.setattr(flows, "_pool_models", lambda pool_dir: ["qwen2.5-7b-q4-k-m"]
    )
    monkeypatch.setattr(
        model_catalog,
        "assemble_candidates",
        lambda view, network="", repo_root=None, **kw: [
            {
                "model_id": "qwen2.5-7b-q4-k-m",
                "origins": {"subnet", "pool"},
                "model_bytes": 4_700_000_000,
                "layers": 28,
                "disk_files": [],
                "hf_repo": "",
                "hf_files": [],
                "in_pool": True,
                "fits": True,
                "fit": "fits on one GPU",
                "where": "in the pool",
                "launchable": True,
            }
        ],
    )
    monkeypatch.setattr(
        onboarding,
        "pool_recommend",
        lambda pool_dir, model_id: {
            "suggestions": [
                {
                    "workers": ["w-a", "w-b"],
                    "driver": "w-a",
                    "link_class": "local",
                    "max_rtt_ms": 0.4,
                },
                {
                    "workers": ["w-b"],
                    "driver": "w-b",
                    "link_class": "local",
                    "max_rtt_ms": 0.0,
                },
            ],
            "reasons": {},
        },
    )
    ran: list[list[str]] = []

    def fake_run_cli(argv):
        ran.append(argv)
        if argv[:2] == ["pool", "launch"]:
            serving["key"] = "m-new"
            return 0, ""
        if argv[:2] == ["pool", "probe"]:
            return 0, '{"verified": true, "proof_stages": 2, "receipts": 2, "engine_tps": 100, "ttft_s": 0.1}'
        return 0, ""

    monkeypatch.setattr(flows, "_run_cli", fake_run_cli)
    minted: list[dict] = []

    def fake_management_request(pool_dir, route, body=None, **kw):
        if route == "/v1/pool/api-keys":
            minted.append(dict(body or {}))
            return {"status": "ok", "api_key": "vrt_pk_testkey123"}
        return None

    monkeypatch.setattr(
        onboarding, "pool_management_request", fake_management_request
    )
    lines: list[str] = []
    # model pick: Enter (=1); placement pick: "1"; API key offer: yes
    io = _io(["", "1", "y"], lines)
    wizard._finish_with_probe(io, tmp_path)
    launch = next(a for a in ran if a[:2] == ["pool", "launch"])
    assert "--workers" in launch
    assert launch[launch.index("--workers") + 1] == "w-a,w-b"
    assert launch[launch.index("--driver") + 1] == "w-a"
    # Spawn-and-watch: the child returns at spawn and wait_for_mesh
    # narrates the formation. Without this the first "mesh launching"
    # tick appeared only after the whole launch finished (frozen-looking
    # screen, live operator feedback).
    assert "--no-wait" in launch
    text = "\n".join(lines)
    assert "what can this pool serve?" in text
    assert "recommended placement" in text
    assert "verified end to end" in text
    # A verified first launch offers the private API and mints on accept.
    assert minted and minted[0]["action"] == "create"
    assert "vrt_pk_testkey123" in text
    assert "apikey expose" in text


def test_offer_deploy_registered_model_offers_rerun_defaulting_no(
    monkeypatch, tmp_path
):
    """An actively registered model must never be offered as 'register now'
    (contradictory, observed); the offer becomes an explicit re-run
    with a NO default."""
    (tmp_path / "pool-state.json").write_text(
        '{"serving_mode": "validator", '
        '"validator_binding": {"chain_id": 945}}'
    )
    monkeypatch.setattr(flows, "_serving_mesh", lambda pool_dir: "m-1")
    monkeypatch.setattr(
        flows,
        "_pool_view",
        lambda pool_dir: {
            "live": True,
            "models": ["glm-5.2-iq2-m"],
            "meshes": {
                "m-1": {"status": "serving", "model_id": "glm-5.2-iq2-m"}
            },
            "workers": {},
            "mesh_registration": {
                "model_id": "glm-5.2-iq2-m",
                "index": 41,
                "endpoint": "http://198.51.100.7:20143",
            },
        },
    )
    ran: list = []
    monkeypatch.setattr(
        flows.subprocess, "run", lambda *a, **k: ran.append(a)
    )
    prompts: list[str] = []
    answers = iter([""])  # Enter accepts the default, which must be NO

    def _ask(prompt: str) -> str:
        prompts.append(prompt)
        return next(answers)

    lines: list[str] = []
    io = WizardIO(ask=_ask, out=lines.append)
    flows._offer_deploy(io, tmp_path)

    assert ran == []  # default declined the re-run
    joined = "\n".join(prompts)
    assert "already registered at index 41" in joined
    assert "re-run deploy" in joined
    assert "register glm-5.2-iq2-m on the subnet now" not in joined


def test_subnet_catalog_stale_serves_and_refreshes_in_background(
    monkeypatch, tmp_path
):
    """The chain list is stale-while-revalidate: an expired disk cache is
    served INSTANTLY and refreshed by a single-flight background thread —
    the forced inline re-read hung boards for minutes on a congested node
    ."""

    import time as _time

    from verallm.mesh import model_catalog

    monkeypatch.setattr(
        Path, "home", classmethod(lambda cls: tmp_path)
    )
    model_catalog._subnet_catalog_cache.clear()
    model_catalog._subnet_catalog_refresh_inflight.clear()

    # Seed an EXPIRED disk cache.
    rows = [{"model_id": "m-old", "layers": 1, "quant": "q4_k_m"}]
    model_catalog._write_subnet_catalog_disk("testnet", rows)
    path = model_catalog._subnet_catalog_cache_path("testnet")
    stale = json.loads(path.read_text())
    stale["fetched_at_unix"] = _time.time() - 3600
    path.write_text(json.dumps(stale))

    spawned = []
    monkeypatch.setattr(
        model_catalog,
        "_spawn_subnet_catalog_refresh",
        lambda key, network, repo_root: spawned.append(key),
    )
    t0 = _time.monotonic()
    got = model_catalog.subnet_model_catalog("testnet")
    assert _time.monotonic() - t0 < 1.0, "stale serve must be instant"
    assert [r["model_id"] for r in got] == ["m-old"]
    assert spawned == ["testnet"], "expired cache must trigger the refresh"

    # No cache at all: bounded wait, then an empty list — never a hang.
    model_catalog._subnet_catalog_cache.clear()
    path.unlink()
    monkeypatch.setattr(
        model_catalog, "_SUBNET_CATALOG_FETCH_TIMEOUT_S", 0.5
    )
    t0 = _time.monotonic()
    got = model_catalog.subnet_model_catalog("testnet")
    assert got == []
    assert _time.monotonic() - t0 < 5.0, "first-run read must be bounded"
