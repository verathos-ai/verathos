"""Tests for the move of legacy validators to decide light validators."""

from __future__ import annotations

import json
import os
import shlex
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import pytest

from neurons import decide_migration as dm

URL = "https://github.com/example-org/verathos.git"
SHA = "a" * 40
OTHER_SHA = "b" * 40
DIGEST = "c" * 64
OWNER = "5Owner"
VALI = "5Vali"


class _Log:
    def __init__(self):
        self.lines: list[tuple[str, str]] = []

    def __getattr__(self, level):
        return lambda message, *a, **k: self.lines.append((level, str(message)))

    def at(self, level: str) -> list[str]:
        return [message for lvl, message in self.lines if lvl == level]


class _Chain:
    def __init__(self, *, owner=OWNER, commitment=f"verathos/catalog/v1 sha256:{DIGEST}", block=2000):
        self.owner = owner
        self.commitment = commitment
        self.block = block
        self.closed = False

    def get_subnet_owner_hotkey(self, netuid):
        return self.owner

    def get_uid_for_hotkey_on_subnet(self, hotkey, netuid):
        return 0

    def get_commitment(self, netuid, uid):
        return self.commitment

    def get_current_block(self):
        return self.block

    def close(self):
        self.closed = True


class _Exec(Exception):
    pass


class _Host:
    """git, the decide installer and pm2 as seen through the migration's runner."""

    def __init__(self, home: Path, *, fetched_sha=SHA, install_code=0, pm2_log="light mode: setting the weights"):
        self.home = home
        self.fetched_sha = fetched_sha
        self.install_code = install_code
        self.pm2_log = pm2_log
        self.commands: list[list[str]] = []
        self.apps: dict[str, dict] = {}
        self.spawned: list[str] = []
        self.execs: list[tuple] = []

    def run(self, cmd, *, timeout, env=None, cwd=None, log_path=None):
        self.commands.append(list(cmd))
        if cmd[0] == "git":
            return self._git(cmd[1:])
        if cmd[0] == "bash" and cmd[1].endswith("scripts/install.sh"):
            return self._install(cmd)
        if cmd[0] == "pm2":
            return self._pm2(cmd[1:])
        if cmd[-1] == "--help":
            return 0, "usage: verathos-validator"
        raise AssertionError(f"unexpected command {cmd}")

    def _git(self, args):
        if args[0] == "clone":
            target = Path(args[-1])
            assert not target.exists() or not any(target.iterdir()), "git clones into a missing or empty directory"
            (target / ".git").mkdir(parents=True)
            (target / ".git" / "config").write_text(f'[remote "origin"]\n\turl = {args[-2]}\n')
            return 0, ""
        checkout = Path(args[1])
        sub = args[2]
        if sub == "fetch":
            return 0, ""
        if sub == "rev-parse" and args[3] == "FETCH_HEAD^{commit}":
            return 0, self.fetched_sha + "\n"
        if sub == "rev-parse" and args[3] == "HEAD":
            return 0, SHA + "\n"
        if sub == "checkout":
            (checkout / "src" / "verathos" / "install").mkdir(parents=True, exist_ok=True)
            (checkout / "src" / "verathos" / "install" / "__init__.py").write_text("")
            (checkout / "pyproject.toml").write_text('[project]\nname = "verathos"\n')
            return 0, ""
        raise AssertionError(f"unexpected git {args}")

    def _install(self, cmd):
        if self.install_code:
            return self.install_code, "error: cannot fetch v1.0.0"
        option = dict(zip(cmd[3:-1], cmd[4:], strict=True))
        decide = Path(option["--dir"])
        binary = decide / ".venv" / "bin" / "verathos-validator"
        binary.parent.mkdir(parents=True, exist_ok=True)
        binary.write_text("#!/bin/sh\n")
        binary.chmod(0o755)
        python = binary.with_name("python")
        python.write_text("")
        python.chmod(0o755)
        config = {"mode": "light", "wallet_name": option["--wallet"], "hotkey_name": option["--hotkey"],
                  "netuid": int(option["--netuid"]), "network": option["--network"]}
        if "--dry-run" in cmd:
            config["dry_run"] = True
        (self.home / ".verathos").mkdir(exist_ok=True)
        (self.home / ".verathos" / "validator.json").write_text(json.dumps(config))
        return 0, "Setting up the validator"

    def _pm2(self, args):
        if args[0] == "jlist":
            return 0, json.dumps(list(self.apps.values()))
        if args[0] == "delete":
            self.apps.pop(args[1], None)
            return 0, ""
        if args[0] == "start":
            name = args[args.index("--name") + 1]
            log = self.home / ".pm2" / "logs" / f"{name}-error.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open("a") as handle:
                handle.write(self.pm2_log + "\n")
            self.apps[name] = {"name": name, "pm2_env": {"status": "online", "pm_err_log_path": str(log),
                                                         "pm_out_log_path": str(log.with_name(f"{name}-out.log"))}}
            return 0, ""
        raise AssertionError(f"unexpected pm2 {args}")

    def spawn(self, script, log_path):
        self.spawned.append(script)

    def execve(self, path, argv, env):
        self.execs.append((path, argv, env.get("HOME")))
        raise _Exec()

    def ran(self, first: str) -> list[list[str]]:
        return [c for c in self.commands if c[0] == first or (len(c) > 1 and c[1].endswith(first))]


@pytest.fixture
def log(monkeypatch):
    stub = _Log()
    monkeypatch.setattr(dm, "bt", SimpleNamespace(logging=stub))
    return stub


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv(dm.OPT_OUT_ENV, raising=False)
    for name in ("https_proxy", "HTTPS_PROXY", "all_proxy", "ALL_PROXY", "PM2_HOME"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(dm.shutil, "which", lambda name: f"/usr/bin/{name}")
    wallet = tmp_path / ".bittensor" / "wallets" / "vali"
    (wallet / "hotkeys").mkdir(parents=True)
    (wallet / "hotkeys" / "default").write_text("{}")
    (wallet / "coldkeypub.txt").write_text("{}")
    return tmp_path


@pytest.fixture
def pinned(monkeypatch):
    monkeypatch.setattr(dm, "DECIDE_REPO_URL", URL)
    monkeypatch.setattr(dm, "DECIDE_REF", "v1.0.0")
    monkeypatch.setattr(dm, "DECIDE_SHA", SHA)
    monkeypatch.setattr(dm, "DECIDE_TAG_SIGNERS", "")
    monkeypatch.setattr(dm, "NOT_BEFORE_BLOCK", {96: 1000, 405: None})


def _target(home: Path, **overrides) -> dm.Target:
    values = dict(wallet="vali", hotkey="default", hotkey_ss58=VALI, network="finney", netuid=96,
                  home=home / ".verathos", decide_dir=home / "verathos-decide")
    values.update(overrides)
    return dm.Target(**values)


def _migration(home: Path, host: _Host | None = None, *, chain=None, pm2_app="validator", **target):
    host = host or _Host(home)
    return dm.DecideMigration(
        _target(home, **target),
        pm2_app=pm2_app,
        chain_factory=lambda: chain or _Chain(),
        runner=host.run,
        spawn=host.spawn,
        execve=host.execve,
        sleep=lambda seconds: None,
        reachable=lambda host_name: True,
    )


def _marker(home: Path) -> dict:
    return json.loads((home / ".verathos" / "decide-migration.json").read_text())


# ── Trigger ──────────────────────────────────────────────────────


def test_catalog_digest_accepts_only_decide_catalog_commitments():
    assert dm.catalog_digest(f"verathos/catalog/v1 sha256:{DIGEST}") == DIGEST
    assert dm.catalog_digest(f"verathos/catalog/v1 sha256:{DIGEST[:-1]}") is None
    assert dm.catalog_digest(f"verathos/catalog/v1 sha256:{DIGEST.upper()}") is None
    assert dm.catalog_digest(f"verathos/validator/v1 {DIGEST}") is None
    assert dm.catalog_digest("") is None
    assert dm.catalog_digest(None) is None


def test_trigger_holds_for_a_non_owner_at_not_before(home, pinned, log):
    chain = _Chain(block=1000)
    migration = _migration(home, chain=chain)
    assert migration.weights_allowed()
    assert migration.check() is True
    assert chain.closed
    assert not migration.weights_allowed()
    assert any("at or past 1000" in line for line in log.at("info"))


@pytest.mark.parametrize(
    "chain, reason",
    [
        (_Chain(block=999), "moves to decide at block 1000"),
        (_Chain(commitment=""), "has not committed a decide catalog"),
        (_Chain(commitment="0x" + "11" * 32), "has not committed a decide catalog"),
        (_Chain(commitment=f"verathos/catalog/v1 sha256:{DIGEST[:40]}"), "has not committed a decide catalog"),
        (_Chain(owner=VALI), "subnet owner's hotkey"),
    ],
)
def test_trigger_does_not_hold(home, pinned, log, chain, reason):
    migration = _migration(home, chain=chain)
    assert migration.check() is False
    assert migration.weights_allowed()
    assert any(reason in line for line in log.at("info"))


@pytest.mark.parametrize(
    "unpin",
    [
        {"DECIDE_SHA": ""},
        {"DECIDE_SHA": "v1.0.0"},
        {"DECIDE_REPO_URL": ""},
        {"NOT_BEFORE_BLOCK": {96: None}},
        {"NOT_BEFORE_BLOCK": {405: 1000}},
        {"DECIDE_REF": "", "DECIDE_TAG_SIGNERS": "maintainer ssh-ed25519 AAAA"},
    ],
)
def test_unpinned_release_never_reads_the_chain(home, pinned, log, monkeypatch, unpin):
    for name, value in unpin.items():
        monkeypatch.setattr(dm, name, value)

    def no_chain():
        raise AssertionError("the chain must not be read")

    migration = _migration(home)
    migration._chain_factory = no_chain
    assert migration.check() is False
    assert migration.weights_allowed()


def test_opt_out_by_environment(home, pinned, log, monkeypatch):
    monkeypatch.setenv(dm.OPT_OUT_ENV, "1")
    migration = _migration(home)
    migration._chain_factory = lambda: (_ for _ in ()).throw(AssertionError("no chain read"))
    assert migration.check() is False
    assert migration.weights_allowed()
    assert any("opted out" in line for line in log.at("info"))


def test_opt_out_by_file_ends_a_migration_before_retirement(home, pinned, log):
    migration = _migration(home)
    assert migration.check() is True
    migration._write_marker(dm.MIGRATING)
    assert not migration.weights_allowed()
    (home / ".verathos" / "no-decide-migration").touch()
    assert migration.weights_allowed()
    assert migration.check() is False
    assert _marker(home)["state"] == dm.IDLE


def test_unreadable_chain_keeps_the_last_decision(home, pinned, log):
    chain = _Chain()
    migration = _migration(home, chain=chain)
    assert migration.check() is True

    def broken():
        raise ConnectionError("websocket closed")

    migration._chain_factory = broken
    assert migration.check() is False
    assert not migration.weights_allowed()
    assert any("chain read failed" in line for line in log.at("warning"))


def test_marker_from_an_earlier_run_keeps_weights_off_until_the_chain_answers(home, pinned, log):
    _migration(home)._write_marker(dm.MIGRATING, attempts=1)
    restarted = _migration(home)
    assert not restarted.weights_allowed()
    restarted._chain_factory = lambda: _Chain(commitment="")
    assert restarted.check() is False
    assert restarted.weights_allowed()
    assert _marker(home)["state"] == dm.IDLE


def test_retired_marker_stops_weights_for_good(home, pinned, log, monkeypatch):
    _migration(home)._write_marker(dm.RETIRED)
    restarted = _migration(home)
    assert not restarted.weights_allowed()
    monkeypatch.setenv(dm.OPT_OUT_ENV, "1")
    (home / ".verathos" / "no-decide-migration").touch()
    assert not restarted.weights_allowed()


def test_marker_of_another_hotkey_is_not_this_validators(home, pinned, log):
    _migration(home, hotkey_ss58="5Other")._write_marker(dm.RETIRED)
    migration = _migration(home)
    assert migration.state() == ""
    assert migration.weights_allowed()


# ── Commands ─────────────────────────────────────────────────────


def test_install_and_pm2_commands(home, pinned):
    migration = _migration(home)
    decide = str(home / "verathos-decide")
    assert migration.install_command() == [
        "bash", f"{decide}/scripts/install.sh", "validator", "--light", "--yes", "--no-service",
        "--dir", decide, "--source", URL, "--ref", SHA,
        "--network", "finney", "--netuid", "96", "--wallet", "vali", "--hotkey", "default",
    ]
    assert migration.pm2_start_command() == [
        "pm2", "start", f"{decide}/.venv/bin/verathos-validator", "--name", "verathos-validator",
        "--interpreter", "none", "--cwd", str(home / ".verathos"),
        "--", "--config", str(home / ".verathos" / "validator.json"),
    ]
    command = _migration(home, network="ws://10.0.0.5:9944", netuid=405, dry_run=True).install_command()
    assert command[command.index("--network") + 1] == "test"
    assert command[command.index("--netuid") + 1] == "405"
    assert command[-1] == "--dry-run"


@pytest.mark.parametrize(
    "network, netuid, expected",
    [
        ("ws://10.0.0.5:9944", 96, "finney"),
        ("ws://subtensor.internal:9944", 405, "test"),
        ("finney", 96, "finney"),
        ("test", 405, "test"),
        ("archive", 96, "archive"),
        ("local", 96, "local"),
        ("ws://127.0.0.1:9944", 96, "ws://127.0.0.1:9944"),
        ("ws://localhost:9944", 405, "ws://localhost:9944"),
        ("wss://subtensor.example.org:443", 96, "wss://subtensor.example.org:443"),
    ],
)
def test_decide_gets_a_network_it_reaches_over_tls_or_on_this_machine(home, pinned, network, netuid, expected):
    command = _migration(home, network=network, netuid=netuid).install_command()
    assert command[command.index("--network") + 1] == expected


def test_manual_commands_finish_the_move_by_hand(home, pinned):
    pm2 = _migration(home).manual_commands()
    assert pm2[0] == (f"git clone --no-checkout {URL} {home}/verathos-decide && "
                      f"git -C {home}/verathos-decide checkout --detach {SHA}")
    assert pm2[1] == shlex.join(_migration(home).install_command())
    assert pm2[2].endswith("&& pm2 delete validator && pm2 save")
    assert "pm2 start" in pm2[2] and "--interpreter none" in pm2[2]
    plain = _migration(home, pm2_app=None).manual_commands(installed=True)
    assert len(plain) == 1 and "verathos-validator --config" in plain[0]


def test_miner_install_commands(home, pinned):
    lines = dm.miner_install_commands("miner", "hk1")
    assert lines[0] == (f"git clone --no-checkout {URL} {home}/verathos-decide && "
                        f"git -C {home}/verathos-decide checkout --detach {SHA}")
    assert lines[1] == (f"bash {home}/verathos-decide/scripts/install.sh miner --dir {home}/verathos-decide "
                        f"--source {URL} --ref {SHA} --wallet miner --hotkey hk1")


def test_miner_notice_is_one_line_with_the_block_and_the_commands(home, pinned, monkeypatch):
    notice = dm.miner_notice("miner", "hk1", 96)
    assert "\n" not in notice
    assert "netuid 96 at block 1000" in notice
    assert " && ".join(dm.miner_install_commands("miner", "hk1")) in notice
    assert "--wallet '<wallet>' --hotkey '<hotkey>'" in dm.miner_notice(None, None, 96)
    assert dm.miner_notice("miner", "hk1", 405) is None
    monkeypatch.setattr(dm, "DECIDE_SHA", "")
    assert dm.miner_notice("miner", "hk1", 96) is None


# ── Migration ────────────────────────────────────────────────────


def test_migration_under_pm2_installs_starts_and_removes_the_legacy_app(home, pinned, log):
    host = _Host(home)
    migration = _migration(home, host)
    assert migration.check()
    assert migration.migrate() is True
    assert _marker(home)["state"] == dm.RETIRED
    assert not migration.weights_allowed()
    installs = host.ran("scripts/install.sh")
    starts = [c for c in host.commands if c[:2] == ["pm2", "start"]]
    assert len(installs) == 1 and len(starts) == 1
    assert host.commands.index(installs[0]) < host.commands.index(starts[0])
    assert "verathos-validator" in host.apps
    assert not any(".bittensor" in part for command in host.commands for part in command)
    assert host.spawned == []
    migration.hand_over()
    assert host.spawned == ["sleep 5; pm2 delete validator; pm2 save"]
    assert host.execs == []


def test_failed_install_keeps_weights_off_and_retries_next_epoch(home, pinned, log):
    host = _Host(home, install_code=1)
    migration = _migration(home, host)
    assert migration.check()
    assert migration.migrate() is False
    marker = _marker(home)
    assert marker["state"] == dm.MIGRATING and marker["attempts"] == 1
    assert "exited with 1: error: cannot fetch v1.0.0" in marker["last_error"]
    assert not migration.weights_allowed()
    assert not [c for c in host.commands if c[:2] == ["pm2", "start"]]
    critical = "\n".join(log.at("critical"))
    assert "decide migration failed" in critical
    assert shlex.join(migration.install_command()) in critical
    assert "pm2 delete validator && pm2 save" in critical
    assert "no-decide-migration" in critical

    host.install_code = 0
    migration.on_epoch_close()
    migration._worker.join(timeout=5)
    assert _marker(home)["attempts"] == 2
    assert _marker(home)["state"] == dm.RETIRED
    assert host.spawned == ["sleep 5; pm2 delete validator; pm2 save"]


def test_unpinned_commit_runs_no_decide_code(home, pinned, log):
    host = _Host(home, fetched_sha=OTHER_SHA)
    migration = _migration(home, host)
    assert migration.migrate() is False
    assert "not the pinned" in _marker(home)["last_error"]
    assert not host.ran("scripts/install.sh")
    assert not [c for c in host.commands if "checkout" in c]
    assert not (home / "verathos-decide.partial").exists()
    assert not (home / "verathos-decide").exists()


def test_preflight_failure_changes_nothing(home, pinned, log, monkeypatch):
    host = _Host(home)
    migration = _migration(home, host)
    migration._reachable = lambda name: name != "pypi.org"
    monkeypatch.setattr(dm.shutil, "disk_usage", lambda path: SimpleNamespace(free=1 << 30))
    assert migration.migrate() is False
    error = _marker(home)["last_error"]
    assert "GiB free" in error and "no HTTPS connection to pypi.org" in error
    assert host.commands == []


def test_failure_is_logged_with_the_commands_when_the_marker_cannot_be_written(home, pinned, log, monkeypatch):
    host = _Host(home)
    migration = _migration(home, host)

    def disk_full(state, **fields):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(migration, "_write_marker", disk_full)
    assert migration.migrate() is False
    critical = "\n".join(log.at("critical"))
    assert "decide migration failed: OSError: [Errno 28] No space left on device" in critical
    assert shlex.join(migration.install_command()) in critical
    assert "no-decide-migration" in critical
    assert any("decide-migration.json not written" in line for line in log.at("error"))
    assert not migration.weights_allowed()
    assert host.commands == []


def test_empty_decide_directory_is_cloned_into_where_it_is(home, pinned, log, monkeypatch):
    # An empty ~/verathos-decide may be a mount point (a docker volume): it can be neither removed nor replaced.
    decide = home / "verathos-decide"
    decide.mkdir()
    real_rmdir, real_rename = Path.rmdir, Path.rename

    def busy(path, *args):
        if path == decide or (args and Path(args[0]) == decide):
            raise OSError(16, "Device or resource busy", str(path))
        return (real_rename if args else real_rmdir)(path, *args)

    monkeypatch.setattr(Path, "rmdir", busy)
    monkeypatch.setattr(Path, "rename", busy)
    assert _migration(home).migrate() is True
    assert f"url = {URL}" in (decide / ".git" / "config").read_text()
    assert (decide / "pyproject.toml").is_file()
    assert not (home / "verathos-decide.partial").exists()


def test_clone_left_unchecked_out_is_cloned_again(home, pinned, log):
    decide = home / "verathos-decide"
    (decide / ".git").mkdir(parents=True)
    (decide / ".git" / "config").write_text(f'[remote "origin"]\n\turl = {URL}\n')
    assert dm.unfinished_clone(decide)
    assert _migration(home).preflight() == []
    assert _migration(home).migrate() is True
    other = home / "other"
    (other / ".git").mkdir(parents=True)
    (other / ".git" / "config").write_text('[remote "origin"]\n\turl = https://example.org/x.git\n')
    assert not dm.unfinished_clone(other)

def test_preflight_names_a_missing_ssh_keygen(home, pinned, log, monkeypatch):
    monkeypatch.setattr(dm.shutil, "which", lambda name: None if name == "ssh-keygen" else f"/usr/bin/{name}")
    assert "missing: ssh-keygen (openssh-client)" in _migration(home).preflight()

def test_preflight_refuses_another_decide_setup(home, pinned, log):
    (home / ".verathos").mkdir()
    (home / ".verathos" / "validator.json").write_text(json.dumps({"wallet_name": "other", "hotkey_name": "x"}))
    (home / "verathos-decide").mkdir()
    (home / "verathos-decide" / "notes.txt").write_text("mine")
    problems = _migration(home).preflight()
    assert any("other/x" in p for p in problems)
    assert any("holds other files" in p for p in problems)


def test_decide_that_does_not_reach_light_mode_is_removed_again(home, pinned, log):
    host = _Host(home, pm2_log="this is the subnet owner's hotkey: its validator is the auditor")
    migration = _migration(home, host)
    assert migration.migrate() is False
    assert "refused to start" in _marker(home)["last_error"]
    assert "verathos-validator" not in host.apps
    assert ["pm2", "delete", "verathos-validator"] in host.commands
    assert host.spawned == []
    assert not migration.weights_allowed()


def test_migration_without_pm2_hands_over_by_exec(home, pinned, log, monkeypatch):
    host = _Host(home)
    migration = _migration(home, host, pm2_app=None)
    chdirs = []
    monkeypatch.setattr(dm.os, "chdir", chdirs.append)
    assert migration.migrate() is True
    assert host.commands[-1][-1] == "--help"
    assert _marker(home)["state"] == dm.RETIRED
    with pytest.raises(_Exec):
        migration.hand_over()
    binary = str(home / "verathos-decide" / ".venv" / "bin" / "verathos-validator")
    assert host.execs == [(binary, [binary, "--config", str(home / ".verathos" / "validator.json")], str(home))]
    assert chdirs == [home / ".verathos"]
    assert not [c for c in host.commands if c[0] == "pm2"]


def test_exec_writes_out_the_queued_log_lines_first(home, pinned, log, monkeypatch):
    host = _Host(home)
    migration = _migration(home, host, pm2_app=None)
    monkeypatch.setattr(dm.os, "chdir", lambda path: None)
    assert migration.migrate() is True
    events = []
    log._listener = SimpleNamespace(stop=lambda: events.append("stop"), start=lambda: events.append("start"))

    def execve(path, argv, env):
        events.append("exec")
        raise PermissionError("not executable")

    migration._execve = execve
    migration.hand_over()
    assert events == ["stop", "exec", "start"]
    assert any("cannot start decide's validator" in line for line in log.at("critical"))

def test_failed_exec_returns_to_migrating(home, pinned, log, monkeypatch):
    host = _Host(home)
    migration = _migration(home, host, pm2_app=None)
    monkeypatch.setattr(dm.os, "chdir", lambda path: None)
    assert migration.migrate() is True

    def broken(path, argv, env):
        raise PermissionError("not executable")

    migration._execve = broken
    migration.hand_over()
    assert _marker(home)["state"] == dm.MIGRATING
    assert not migration.weights_allowed()
    assert any("cannot start decide's validator" in line for line in log.at("critical"))


def test_dry_run_validator_moves_as_a_dry_run(home, pinned, log):
    host = _Host(home)
    migration = _migration(home, host, dry_run=True)
    assert migration.migrate() is True
    assert host.ran("scripts/install.sh")[0][-1] == "--dry-run"


@pytest.mark.parametrize("network, lines", [("ws://10.0.0.5:9944", 1), ("wss://subtensor.example.org:443", 0)])
def test_migration_logs_one_line_when_decide_takes_the_chains_name(home, pinned, log, monkeypatch, network, lines):
    migration = _migration(home, network=network)
    assert migration.migrate() is True
    said = [message for _, message in log.lines if network in message]
    assert len(said) == lines
    assert all("uses finney, the chain of netuid 96" in message for message in said)
    monkeypatch.setattr(dm.bt, "Subtensor", lambda network: network, raising=False)
    assert migration._default_chain() == network


def test_weights_in_flight_finish_before_the_move(home, pinned, log):
    host = _Host(home)
    migration = _migration(home, host)
    polls = iter([False, False, True])
    migration.weights_idle = lambda: next(polls)
    assert migration.migrate() is True

    stuck = _migration(home, _Host(home))
    stuck._write_marker(dm.IDLE)
    clock = iter(range(0, 10_000, 100))
    stuck._monotonic = lambda: next(clock)
    stuck.weights_idle = lambda: False
    assert stuck.migrate() is False
    assert "still in flight" in _marker(home)["last_error"]


# ── Start and epoch close ────────────────────────────────────────


def test_retired_validator_hands_over_at_start(home, pinned, log, monkeypatch):
    migration = _migration(home)
    migration._write_marker(dm.RETIRED)

    class Parked(Exception):
        pass

    def hand_off():
        raise Parked()

    monkeypatch.setattr(migration, "hand_off_retired", hand_off)
    migration._chain_factory = lambda: (_ for _ in ()).throw(AssertionError("no chain read"))
    with pytest.raises(Parked):
        migration.startup()


def test_handover_of_a_retired_pm2_validator_starts_decide_if_needed(home, pinned, log):
    host = _Host(home)
    migration = _migration(home, host)
    binary = home / "verathos-decide" / ".venv" / "bin" / "verathos-validator"
    binary.parent.mkdir(parents=True)
    for path in (binary, binary.with_name("python")):
        path.write_text("")
        path.chmod(0o755)
    reason = migration._hand_off_once()
    assert "verathos-validator" in host.apps
    assert host.spawned == ["sleep 5; pm2 delete validator; pm2 save"]
    assert "did not remove validator" in reason

    host.spawned.clear()
    starts = len([c for c in host.commands if c[:2] == ["pm2", "start"]])
    migration._hand_off_once()
    assert len([c for c in host.commands if c[:2] == ["pm2", "start"]]) == starts
    assert host.spawned == ["sleep 5; pm2 delete validator; pm2 save"]


def _parks_once(waits: list) -> Callable[[float], None]:
    def sleep(seconds):
        waits.append(seconds)
        if len(waits) > 1:
            raise AssertionError("parked a second time")

    return sleep


def test_retired_validator_installs_decide_again_when_it_is_gone(home, pinned, log, monkeypatch):
    # A container recreated without the volume of ~/verathos-decide: the marker on ~/.verathos says retired.
    host = _Host(home)
    migration = _migration(home, host, pm2_app=None)
    migration._write_marker(dm.RETIRED)
    monkeypatch.setattr(dm.os, "chdir", lambda path: None)
    waits = []
    migration._sleep = _parks_once(waits)
    with pytest.raises(_Exec):
        migration.hand_off_retired()
    assert waits == []
    assert len(host.ran("scripts/install.sh")) == 1
    assert host.commands[-1][-1] == "--help"
    binary = str(home / "verathos-decide" / ".venv" / "bin" / "verathos-validator")
    assert host.execs == [(binary, [binary, "--config", str(home / ".verathos" / "validator.json")], str(home))]
    marker = _marker(home)
    assert marker["state"] == dm.RETIRED and marker["reinstalls"] == 1 and marker["last_error"] == ""
    assert any("installing decide v1.0.0 again" in line for line in log.at("warning"))
    assert not migration.weights_allowed()


def test_failed_reinstall_logs_the_commands_and_tries_again_an_epoch_later(home, pinned, log, monkeypatch):
    host = _Host(home, install_code=1)
    migration = _migration(home, host, pm2_app=None)
    migration._write_marker(dm.RETIRED)
    monkeypatch.setattr(dm.os, "chdir", lambda path: None)
    waits = []
    parks = _parks_once(waits)

    def sleep(seconds):
        parks(seconds)
        assert _marker(home)["state"] == dm.RETIRED and "exited with 1" in _marker(home)["last_error"]
        host.install_code = 0  # the next attempt succeeds

    migration._sleep = sleep
    with pytest.raises(_Exec):
        migration.hand_off_retired()
    assert waits == [dm.RETRY_S]
    assert len(host.ran("scripts/install.sh")) == 2
    critical = "\n".join(log.at("critical"))
    assert "could not hand over to decide's validator: decide's installer exited with 1" in critical
    assert shlex.join(migration.install_command()) in critical
    marker = _marker(home)
    assert marker["state"] == dm.RETIRED and marker["reinstalls"] == 2 and marker["last_error"] == ""


def test_retired_pm2_validator_installs_decide_again_before_starting_it(home, pinned, log):
    host = _Host(home)
    migration = _migration(home, host)
    migration._write_marker(dm.RETIRED)
    assert "did not remove validator" in migration._hand_off_once()
    installs = host.ran("scripts/install.sh")
    starts = [c for c in host.commands if c[:2] == ["pm2", "start"]]
    assert len(installs) == 1 and len(starts) == 1
    assert host.commands.index(installs[0]) < host.commands.index(starts[0])
    assert host.spawned == ["sleep 5; pm2 delete validator; pm2 save"]


def test_environment_whose_python_is_gone_is_built_anew(home, pinned, log, monkeypatch):
    # ~/verathos-decide on a volume, the Python uv installed for it in the recreated container's lost filesystem.
    decide = home / "verathos-decide"
    (decide / "src" / "verathos" / "install").mkdir(parents=True)
    (decide / "src" / "verathos" / "install" / "__init__.py").write_text("")
    (decide / "pyproject.toml").write_text('[project]\nname = "verathos"\n')
    (decide / ".git").mkdir()
    binary = decide / ".venv" / "bin" / "verathos-validator"
    binary.parent.mkdir(parents=True)
    binary.write_text("#!/bin/sh\n")
    binary.chmod(0o755)
    binary.with_name("python").symlink_to(home / "lost" / "python3.12")
    (decide / ".venv" / "stale").write_text("")
    host = _Host(home)
    migration = _migration(home, host, pm2_app=None)
    migration._write_marker(dm.RETIRED)
    monkeypatch.setattr(dm.os, "chdir", lambda path: None)
    assert not migration.decide_installed()
    migration._sleep = _parks_once([])
    with pytest.raises(_Exec):
        migration.hand_off_retired()
    assert not (decide / ".venv" / "stale").exists()
    assert migration.decide_installed()
    assert len(host.ran("scripts/install.sh")) == 1


def test_startup_migrates_when_due_and_continues_when_it_fails(home, pinned, log):
    host = _Host(home, install_code=2)
    migration = _migration(home, host)
    migration.startup()
    assert not migration.weights_allowed()
    assert _marker(home)["state"] == dm.MIGRATING


def test_epoch_close_runs_one_migration_at_a_time(home, pinned, log):
    host = _Host(home)
    migration = _migration(home, host)
    gate = threading.Event()
    entered = threading.Event()
    original = host._install

    def slow_install(cmd):
        entered.set()
        assert gate.wait(timeout=5)
        return original(cmd)

    host._install = slow_install
    migration.on_epoch_close()
    assert entered.wait(timeout=5)
    assert migration.busy
    first = migration._worker
    migration.on_epoch_close()
    assert migration._worker is first
    gate.set()
    first.join(timeout=5)
    assert not migration.busy
    assert len(host.ran("scripts/install.sh")) == 1


def test_epoch_close_of_a_retired_validator_retries_its_removal(home, pinned, log):
    host = _Host(home)
    migration = _migration(home, host)
    migration._write_marker(dm.RETIRED)
    migration.on_epoch_close()
    assert host.spawned == ["sleep 5; pm2 delete validator; pm2 save"]
    assert any("pm2 delete validator && pm2 save" in line for line in log.at("critical"))


def test_run_command_kills_the_process_group_at_the_timeout(tmp_path):
    code, output = dm.run_command(["sh", "-c", "echo started; sleep 30"], timeout=0.5,
                                  log_path=tmp_path / "log")
    assert code == 124
    assert "started" in output and "timed out" in output
    assert "$ sh -c" in (tmp_path / "log").read_text()
    code, output = dm.run_command(["sh", "-c", "echo ok"], timeout=10)
    assert (code, output.strip()) == (0, "ok")


def test_run_command_returns_a_long_output_whole_unless_it_logs_it(tmp_path):
    long = ["sh", "-c", "head -c 10000 /dev/zero | tr '\\0' x; echo end"]
    code, output = dm.run_command(long, timeout=10)
    assert code == 0 and output == "x" * 10000 + "end\n"
    code, output = dm.run_command(long, timeout=10, log_path=tmp_path / "log")
    assert code == 0 and len(output) == 4096 and output.endswith("end\n")
    assert "x" * 10000 + "end" in (tmp_path / "log").read_text()


def test_pm2_app_list_longer_than_4_kib_is_read_whole(home, pinned, log):
    # Two apps with their environments make pm2 jlist longer than the 4 KiB kept of a logged command.
    apps = [{"name": "validator", "pm2_env": {"status": "online", "env": {"PATH": "/usr/local/bin:" * 400}}},
            {"name": "verathos-validator", "pm2_env": {"status": "online"}}]
    listing = json.dumps(apps)
    assert len(listing) > 4096
    (home / ".verathos").mkdir()
    pm2 = home / ".local" / "bin" / "pm2"
    pm2.parent.mkdir(parents=True)
    pm2.write_text("#!/bin/sh\necho '[PM2][WARN] Current process list is not synchronized with saved list'\n"
                   f"cat <<'EOF'\n{listing}\nEOF\necho '(node:7) Warning: a notice after the list'\n")
    pm2.chmod(0o755)
    migration = dm.DecideMigration(_target(home), pm2_app="validator")
    assert migration._pm2_online("verathos-validator")
    assert migration._pm2_app_info("validator")["pm2_env"]["env"]["PATH"].startswith("/usr/local/bin:")
    assert migration._pm2_app_info("proxy") is None


def test_child_environment_drops_this_python_environment(home, monkeypatch):
    monkeypatch.setenv("VIRTUAL_ENV", "/opt/legacy/.venv-validator")
    monkeypatch.setenv("PYTHONPATH", "/opt/legacy")
    env = _migration(home)._child_env()
    assert "VIRTUAL_ENV" not in env and "PYTHONPATH" not in env
    assert env["HOME"] == str(home)
    assert env["PATH"].split(os.pathsep)[0] == str(home / ".local" / "bin")


def test_pm2_commands_do_not_inherit_the_legacy_apps_pm2_variables(home, pinned, log, monkeypatch):
    monkeypatch.setenv("pm_id", "3")
    monkeypatch.setenv("name", "validator")
    monkeypatch.setenv("pm_exec_path", "/root/verathos/.venv-validator/bin/python")
    monkeypatch.setenv("PM2_HOME", "/root/.pm2")
    envs = []
    host = _Host(home)

    def run(cmd, *, timeout, env=None, cwd=None, log_path=None):
        envs.append((cmd[0], env))
        return host.run(cmd, timeout=timeout, env=env, cwd=cwd, log_path=log_path)

    migration = _migration(home, host)
    migration._runner = run
    monkeypatch.setattr(migration, "_default_pm2_logs", lambda: [])
    migration.start_under_pm2()
    pm2_envs = [env for first, env in envs if first == "pm2"]
    assert pm2_envs and all(set(env) <= set(dm.PM2_ENV_KEYS) for env in pm2_envs)
    assert all(env["HOME"] == str(home) and env["PM2_HOME"] == "/root/.pm2" for env in pm2_envs)


# ── Validator integration ────────────────────────────────────────


class _Migration:
    def __init__(self, allowed=False, busy=False):
        self.allowed = allowed
        self.busy = busy
        self.calls: list[str] = []

    def weights_allowed(self):
        return self.allowed

    def on_epoch_close(self):
        self.calls.append("migration")


def _validator():
    from neurons.validator import ValidatorNeuron

    neuron = ValidatorNeuron.__new__(ValidatorNeuron)
    neuron._weight_setting_enabled = True
    return neuron


def test_validator_sets_no_weights_after_retirement(home, pinned, monkeypatch):
    from neurons import validator

    real = _migration(home)
    real._write_marker(dm.RETIRED)
    neuron = _validator()
    neuron._decide_migration = real

    def no_subtensor(*args, **kwargs):
        raise AssertionError("no weights may be sent after retirement")

    monkeypatch.setattr(validator.bt, "Subtensor", no_subtensor)
    assert neuron._set_weights({1: 0.4, 2: 0.6}) is False


def test_validator_sets_no_weights_once_the_trigger_holds(home, pinned, monkeypatch):
    from neurons import validator

    migration = _migration(home)
    neuron = _validator()
    neuron._decide_migration = migration
    sent = []

    class Subtensor:
        def __init__(self, network):
            pass

        def set_weights(self, **kwargs):
            sent.append(kwargs)

        def close(self):
            pass

    monkeypatch.setattr(validator.bt, "Subtensor", Subtensor)
    neuron.config = SimpleNamespace(subtensor_network="finney", netuid=96)
    neuron._wallet = object()
    neuron._close_subtensor = lambda obj: None
    assert neuron._set_weights({1: 1.0}) is True
    assert migration.check()
    assert neuron._set_weights({1: 1.0}) is False
    assert len(sent) == 1


def test_validator_start_runs_the_migration_check(monkeypatch):
    from neurons import validator

    started = []
    fake = SimpleNamespace(startup=lambda: started.append(True))
    monkeypatch.setattr(dm.DecideMigration, "for_validator", classmethod(lambda cls, config, dry_run: fake))
    assert validator._start_decide_migration(SimpleNamespace(), dry_run=True) is fake
    assert started == [True]

    def unloadable(cls, config, dry_run):
        raise FileNotFoundError("hotkey file not found")

    monkeypatch.setattr(dm.DecideMigration, "for_validator", classmethod(unloadable))
    assert validator._start_decide_migration(SimpleNamespace()) is None


def test_epoch_close_checks_the_migration_before_its_weights():
    neuron = _validator()
    neuron._running = True
    neuron._epoch_close_lock = threading.Lock()
    neuron._epoch_close_local = threading.local()
    neuron._epoch_close_futures = {}
    neuron._last_closed_epoch = 4
    neuron._pending_epoch_close = 5
    neuron._closing_inflight_canaries = {}
    neuron._inflight_canaries = {}
    neuron._auto_updater = None
    neuron._scoring = None
    neuron.scorer = SimpleNamespace(ema_alpha=0.1, throughput_power=1.0)
    migration = _Migration()
    neuron._decide_migration = migration
    neuron._close_epoch = lambda epoch: migration.calls.append("close")
    neuron._schedule_weight_update = lambda: migration.calls.append("weights")
    neuron._run_queued_epoch_close(SimpleNamespace(epoch_number=5, weight_update_due=True))
    assert migration.calls == ["close", "migration", "weights"]


def test_epoch_without_planned_canaries_closes_and_checks_the_migration():
    # No miners found since the start: no canaries were planned, and the epoch still closes and checks the move.
    neuron = _validator()
    neuron._current_epoch = 5
    neuron._epoch_start_block = 100
    neuron._epoch_miners = []
    neuron._expected_receipts = {}
    neuron._expected_canary_obligations = {}
    neuron._hard_canary_obligation_ids = set()
    neuron._validator_canary_failures = set()
    neuron._canary_penalized_keys = set()
    neuron._busy_skips = {}
    neuron._busy_skip_probations = {}
    neuron._canary_accounting_lock = threading.Lock()
    neuron._shared_hard_prefetch_lock = threading.Lock()
    neuron._shared_hard_prefetch_results = {}
    neuron._scoring = None
    neuron._proof_v3_releases = {}
    neuron._proof_v3_canary_policy = None
    neuron._proof_protocol_rollout_cfg = None
    neuron._proof_v3_failure_policy_cfg = None
    neuron._maintenance_grace_cfg = None
    neuron._capacity_audit_cfg = None
    neuron._weight_update_due = True
    neuron.config = SimpleNamespace(proof_v3_hard_auditor_hotkey_ss58="")
    queued = []
    neuron._epoch_close_futures = {}
    neuron._epoch_close_executor = SimpleNamespace(submit=lambda run, state: queued.append((run, state)))
    assert neuron._queue_epoch_close(5) is True
    [(run, state)] = queued
    assert state._canary_errors == {} and state._canary_error_times == {}

    neuron._running = True
    neuron._epoch_close_lock = threading.Lock()
    neuron._epoch_close_local = threading.local()
    neuron._last_closed_epoch = 4
    neuron._pending_epoch_close = 5
    neuron._closing_inflight_canaries = {}
    neuron._inflight_canaries = {}
    migration = _Migration()
    neuron._decide_migration = migration
    neuron._close_epoch = lambda epoch: migration.calls.append("close")
    neuron._schedule_weight_update = lambda: migration.calls.append("weights")
    run(state)
    assert migration.calls == ["close", "migration", "weights"]

def test_auto_update_waits_for_a_running_migration():
    neuron = _validator()
    neuron._weight_update_lock = threading.Lock()
    neuron._weight_updates_pending = 0
    neuron._epoch_close_futures = {}
    neuron._inflight_canaries = {}
    neuron._closing_inflight_canaries = {}
    neuron._queued_canaries = {}
    neuron._decide_migration = _Migration(busy=True)
    assert neuron._auto_update_busy() is True
    neuron._decide_migration = _Migration(busy=False)
    assert neuron._auto_update_busy() is False
    assert neuron._weight_updates_idle() is True
    neuron._weight_updates_pending = 1
    assert neuron._weight_updates_idle() is False
