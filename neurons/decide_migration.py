"""Move this validator to a decide light validator once the subnet moves to decide.

The subnet moves when its owner hotkey commits a decide catalog (``verathos/catalog/v1 sha256:<digest>``). From
``NOT_BEFORE_BLOCK`` on, a validator that is not the subnet owner's and has not opted out stops setting weights,
installs decide at the pinned commit with the same wallet, hotkey, network and netuid (``install.sh validator
--light --no-service``) and hands over: under pm2 it starts ``verathos-validator`` there, waits for its
``light mode:`` line and removes its own pm2 app; otherwise it replaces this process with decide's validator.
decide reaches the chain over TLS or on this machine only: a network it does not reach that way (a plain ``ws://``
node on another machine) becomes the chain's name for the netuid, ``finney`` or ``test``.

The check runs at start, before the auto-updater, and after each epoch close. ``~/.verathos/decide-migration.json``
records the state: ``migrating`` from the trigger on, ``retired`` once decide runs. A retired validator never sets
weights again and hands over at start; when decide's validator is gone (a container recreated without the volume of
the decide directory), it installs the pinned decide again first. A failed attempt leaves this validator running
without setting weights, logs the commands to finish by hand, and is retried after the next epoch close.

Opt out, before retirement, with ``VERATHOS_NO_DECIDE_MIGRATION=1`` or the file ``~/.verathos/no-decide-migration``.
Wallets are read only by the decide installer; this checkout and its database stay.
"""

from __future__ import annotations

import ipaddress
import json
import os
import platform
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import urlparse

import bittensor as bt
from bittensor.core.settings import NETWORK_MAP

# ── Pinned decide release. Empty values keep the migration off. ──
DECIDE_REPO_URL = "https://github.com/verathos-ai/verathos-decide"
# Release tag fetched with the commit (for example "v1.0.0"); empty: the commit alone.
DECIDE_REF = "v1.0.0"
# The 40-hex commit the installation must check out.
DECIDE_SHA = "00e95e9e112b509f41d0b7c47ee1938bd72b5f91"
# ssh allowed-signers lines for decide's release tags. When set, the signature of the tag DECIDE_REF must verify
# against them before any decide code runs.
DECIDE_TAG_SIGNERS = (
    'verathos-release namespaces="git" ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIA7pWYysUPrBGrcYGk0BPMaWN9Li0S+czHKLlmsn3bal\n'
)
# First block at which validators move, per netuid. A netuid without a block never moves.
NOT_BEFORE_BLOCK: dict[int, Optional[int]] = {96: 9287280, 405: None}
# The chain's name per netuid: decide's network when it does not reach this validator's node (see decide_network).
CHAIN_NAMES = {96: "finney", 405: "test"}

CATALOG_PREFIX = "verathos/catalog/v1 sha256:"
OPT_OUT_ENV = "VERATHOS_NO_DECIDE_MIGRATION"
DECIDE_APP = "verathos-validator"
MIGRATING = "migrating"
RETIRED = "retired"
IDLE = "idle"

MIN_FREE_BYTES = 2 << 30
EGRESS_HOSTS = ("github.com", "pypi.org", "files.pythonhosted.org")
UV_HOST = "astral.sh"
CHECK_TIMEOUT_S = 60.0
GIT_TIMEOUT_S = 600.0
INSTALL_TIMEOUT_S = 1800.0
STARTUP_TIMEOUT_S = 600.0
WEIGHTS_IDLE_TIMEOUT_S = 300.0
REMOVAL_WAIT_S = 120.0
RETRY_S = 360 * 12.0
LIGHT_MODE_LINE = "light mode:"
# The environment pm2 commands and the decide app get: under pm2 this process also carries pm2's own variables.
PM2_ENV_KEYS = (
    "PATH", "HOME", "USER", "LOGNAME", "SHELL", "LANG", "LC_ALL", "LC_CTYPE", "TZ", "TERM", "PM2_HOME",
    "NVM_DIR", "NVM_BIN", "XDG_RUNTIME_DIR", "SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE",
    "http_proxy", "https_proxy", "no_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "ALL_PROXY",
)
# Lines with which decide's validator refuses to run as a light validator.
DECIDE_REFUSALS = ("subnet owner's hotkey", "source_hotkey")


class MigrationError(RuntimeError):
    """A step of the migration failed; the message says which and why."""


@dataclass(frozen=True)
class Decision:
    """Whether to move now: True, False, or None when the chain could not tell."""

    migrate: Optional[bool]
    reason: str


@dataclass(frozen=True)
class Target:
    """This validator's identity and the paths of its decide setup."""

    wallet: str
    hotkey: str
    hotkey_ss58: str
    network: str
    netuid: int
    home: Path  # ~/.verathos, shared with decide
    decide_dir: Path  # ~/verathos-decide
    dry_run: bool = False  # a validator that sets no weights (--no-set-weights) moves as a dry run

    @property
    def config_path(self) -> Path:
        return self.home / "validator.json"

    @property
    def decide_bin(self) -> Path:
        return self.decide_dir / ".venv" / "bin" / "verathos-validator"

    @property
    def decide_python(self) -> Path:
        return self.decide_dir / ".venv" / "bin" / "python"

    @property
    def marker_path(self) -> Path:
        return self.home / "decide-migration.json"

    @property
    def opt_out_path(self) -> Path:
        return self.home / "no-decide-migration"

    @property
    def log_path(self) -> Path:
        return self.home / "decide-migration.log"


def catalog_digest(commitment: object) -> Optional[str]:
    """The catalog digest of a decide catalog commitment, else None."""
    if not isinstance(commitment, str) or not commitment.startswith(CATALOG_PREFIX):
        return None
    value = commitment[len(CATALOG_PREFIX):]
    return value if re.fullmatch(r"[0-9a-f]{64}", value) else None


def source_configured() -> bool:
    """Whether this release pins a decide source."""
    return bool(DECIDE_REPO_URL) and re.fullmatch(r"[0-9a-f]{40}", DECIDE_SHA) is not None and (
        bool(DECIDE_REF) or not DECIDE_TAG_SIGNERS
    )


def not_before(netuid: int) -> Optional[int]:
    value = NOT_BEFORE_BLOCK.get(int(netuid))
    return value if isinstance(value, int) and value > 0 else None


def decide_network(network: str, netuid: int) -> str:
    """The network decide's validator takes: ``network`` when decide reaches the node it names over TLS or on this
    machine (``finney``, ``test``, ``archive``, ``local``, a ``wss://`` URL, a ``ws://`` URL on loopback: the rule of
    decide's ``chain_network``), else the chain's name for ``netuid``."""
    parts = urlparse(NETWORK_MAP.get(network, network))
    if parts.scheme == "wss" and parts.hostname or parts.scheme == "ws" and _loopback(parts.hostname):
        return network
    return CHAIN_NAMES.get(int(netuid), network)


def _loopback(host: Optional[str]) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host or "").is_loopback
    except ValueError:
        return False


def is_decide_checkout(path: Path) -> bool:
    """A checkout of decide, as its installer recognises one."""
    pyproject = path / "pyproject.toml"
    try:
        named = re.search(r'^name = "verathos"$', pyproject.read_text(), re.MULTILINE) is not None
    except OSError:
        return False
    return named and (path / "src" / "verathos" / "install" / "__init__.py").is_file()


def miner_install_commands(wallet: str, hotkey: str, decide_dir: Optional[Path] = None) -> list[str]:
    """The commands that install a decide miner from the pinned source."""
    d = shlex.quote(str(decide_dir or Path.home() / "verathos-decide"))
    return [
        f"git clone --no-checkout {shlex.quote(DECIDE_REPO_URL)} {d} && git -C {d} checkout --detach {DECIDE_SHA}",
        f"bash {d}/scripts/install.sh miner --dir {d} --source {shlex.quote(DECIDE_REPO_URL)} --ref {DECIDE_SHA}"
        f" --wallet {shlex.quote(wallet)} --hotkey {shlex.quote(hotkey)}",
    ]


def miner_notice(wallet: Optional[str], hotkey: Optional[str], netuid: int) -> Optional[str]:
    """The line a miner logs at start once a decide release is pinned for its netuid, else None."""
    first = not_before(netuid)
    if not source_configured() or first is None:
        return None
    commands = " && ".join(miner_install_commands(wallet or "<wallet>", hotkey or "<hotkey>"))
    return (
        f"Verathos moves to decide on netuid {netuid} at block {first}; from then on validators set weights for "
        f"decide miners only. Install a decide miner with this hotkey (its GPUs and models: docs/MINING.md of "
        f"{DECIDE_REPO_URL}): {commands}"
    )


def unfinished_clone(path: Path) -> bool:
    """A clone of the pinned source that an attempt left unchecked out in ``path`` (it stopped midway)."""
    try:
        if [child.name for child in path.iterdir()] != [".git"]:
            return False
        config = (path / ".git" / "config").read_text()
    except OSError:
        return False
    pattern = rf"^\s*url = {re.escape(DECIDE_REPO_URL)}$"
    return bool(DECIDE_REPO_URL) and re.search(pattern, config, re.MULTILINE) is not None


def _discard(path: Path, *, keep: bool) -> None:
    """Remove ``path``, or with ``keep`` only what it holds (a mount point cannot be removed)."""
    if not keep:
        shutil.rmtree(path, ignore_errors=True)
        return
    for child in path.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child, ignore_errors=True)
        else:
            child.unlink(missing_ok=True)


def run_command(
    cmd: list[str],
    *,
    timeout: float,
    env: Optional[dict[str, str]] = None,
    cwd: Optional[Path] = None,
    log_path: Optional[Path] = None,
) -> tuple[int, str]:
    """Run ``cmd`` in its own session without a terminal and return its exit code and output (with ``log_path``:
    the output is appended there and the last 4 KiB returned). At the timeout its process group is killed and the
    code is 124."""
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        out = log_path.open("a+b")
        out.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] $ {shlex.join(cmd)}\n".encode())
        out.flush()
    else:
        out = tempfile.TemporaryFile()
    with out:
        start = out.seek(0, os.SEEK_END)
        proc = subprocess.Popen(
            cmd, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT,
            cwd=cwd, env=env, start_new_session=True,
        )
        try:
            code = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
            code = 124
        end = out.seek(0, os.SEEK_END)
        out.seek(max(start, end - 4096) if log_path is not None else start)
        text = out.read().decode(errors="replace")
    if code == 124:
        text += f"\n(timed out after {timeout:.0f} s)"
    return code, text


def spawn_detached(script: str, log_path: Path) -> None:
    """Run ``script`` under sh in a process that is no child of this one, so that stopping this process and its
    process tree (``pm2 delete``) does not stop it."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    wrapper = f"( {script} ) >>{shlex.quote(str(log_path))} 2>&1 </dev/null &"
    subprocess.Popen(["sh", "-c", wrapper], start_new_session=True).wait(timeout=30)


def _last_line(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else "no output"


class DecideMigration:
    """The migration of one validator hotkey on this machine (see the module docstring)."""

    def __init__(
        self,
        target: Target,
        *,
        pm2_app: Optional[str] = None,
        chain_factory: Optional[Callable[[], object]] = None,
        runner: Callable[..., tuple[int, str]] = run_command,
        spawn: Callable[[str, Path], None] = spawn_detached,
        execve: Callable[[str, list[str], dict[str, str]], None] = os.execve,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        reachable: Optional[Callable[[str], bool]] = None,
    ) -> None:
        self.target = target
        self.pm2_app = pm2_app
        self.weights_idle: Optional[Callable[[], bool]] = None
        self._chain_factory = chain_factory or self._default_chain
        self._runner = runner
        self._spawn = spawn
        self._execve = execve
        self._sleep = sleep
        self._monotonic = monotonic
        self._reachable = reachable or self._default_reachable
        self._lock = threading.Lock()
        self._triggered: Optional[bool] = None
        self._last_reason = ""
        self._migrating = False
        self._worker: Optional[threading.Thread] = None

    @classmethod
    def for_validator(cls, config, *, dry_run: bool = False) -> "DecideMigration":
        """The migration of the validator ``config`` describes, with its hotkey loaded and pm2 detected."""
        from neurons.auto_update import _detect_pm2

        wallet_cls = getattr(bt, "Wallet", None) or bt.wallet
        wallet = wallet_cls(name=config.wallet_name, hotkey=config.hotkey_name)
        target = Target(
            wallet=config.wallet_name,
            hotkey=config.hotkey_name,
            hotkey_ss58=wallet.hotkey.ss58_address,
            network=str(config.subtensor_network),
            netuid=int(config.netuid),
            home=Path.home() / ".verathos",
            decide_dir=Path.home() / "verathos-decide",
            dry_run=dry_run,
        )
        return cls(target, pm2_app=_detect_pm2())

    # ── State ────────────────────────────────────────────────────

    def opted_out(self) -> bool:
        flag = os.environ.get(OPT_OUT_ENV, "").strip().lower()
        return flag in ("1", "true", "yes", "on") or self.target.opt_out_path.exists()

    def read_marker(self) -> dict:
        try:
            data = json.loads(self.target.marker_path.read_text())
        except (OSError, ValueError):
            return {}
        if not isinstance(data, dict) or data.get("hotkey") != self.target.hotkey_ss58:
            return {}
        return data

    def state(self) -> str:
        return str(self.read_marker().get("state") or "")

    def _write_marker(self, state: str, **fields) -> None:
        data = self.read_marker()
        data.update(fields)
        data.update(
            state=state,
            hotkey=self.target.hotkey_ss58,
            wallet=self.target.wallet,
            hotkey_name=self.target.hotkey,
            netuid=self.target.netuid,
            decide_dir=str(self.target.decide_dir),
            decide_sha=DECIDE_SHA,
            legacy_pm2_app=self.pm2_app,
            updated=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        )
        path = self.target.marker_path
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
        os.replace(tmp, path)

    def weights_allowed(self) -> bool:
        """Whether this validator may set weights: never once retired; not while it moves, unless opted out."""
        state = self.state()
        if state == RETIRED:
            return False
        if self.opted_out():
            return True
        with self._lock:
            triggered = self._triggered
        if triggered is None:
            return state != MIGRATING
        return not triggered

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._migrating

    # ── Trigger ──────────────────────────────────────────────────

    def _local_decision(self) -> Optional[Decision]:
        if not source_configured() or not_before(self.target.netuid) is None:
            return Decision(False, f"no decide release is pinned for netuid {self.target.netuid}")
        if self.opted_out():
            return Decision(False, f"opted out ({OPT_OUT_ENV} or {self.target.opt_out_path})")
        return None

    def evaluate(self, chain) -> Decision:
        """Whether to move now, read from ``chain`` (a bittensor Subtensor)."""
        local = self._local_decision()
        if local is not None:
            return local
        netuid = self.target.netuid
        owner = chain.get_subnet_owner_hotkey(netuid)
        if not owner:
            return Decision(None, "the subnet owner hotkey could not be read")
        if owner == self.target.hotkey_ss58:
            return Decision(False, "this is the subnet owner's hotkey; it stays on this release")
        uid = chain.get_uid_for_hotkey_on_subnet(owner, netuid)
        digest = catalog_digest(chain.get_commitment(netuid, uid)) if uid is not None else None
        if digest is None:
            return Decision(False, "the subnet owner has not committed a decide catalog")
        block = int(chain.get_current_block())
        first = not_before(netuid)
        if block < first:
            return Decision(False, f"the subnet owner commits decide catalog {digest[:12]}; this validator moves "
                                   f"to decide at block {first} (now {block})")
        return Decision(True, f"the subnet owner commits decide catalog {digest[:12]} and block {block} is at or "
                              f"past {first}")

    def _evaluate_bounded(self) -> Decision:
        local = self._local_decision()
        if local is not None:
            return local
        result: dict[str, Decision] = {}

        def work() -> None:
            chain = None
            try:
                chain = self._chain_factory()
                result["decision"] = self.evaluate(chain)
            except Exception as exc:
                result["decision"] = Decision(None, f"chain read failed: {exc}")
            finally:
                close = getattr(chain, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception:
                        pass

        worker = threading.Thread(target=work, name="decide-migration-check", daemon=True)
        worker.start()
        worker.join(CHECK_TIMEOUT_S)
        return result.get("decision") or Decision(None, f"chain read timed out after {CHECK_TIMEOUT_S:.0f} s")

    def check(self) -> bool:
        """Evaluate the trigger, record it and return whether to move now."""
        decision = self._evaluate_bounded()
        with self._lock:
            if decision.migrate is not None:
                self._triggered = decision.migrate
            changed = decision.reason != self._last_reason
            self._last_reason = decision.reason
        if changed:
            log = bt.logging.warning if decision.migrate is None else bt.logging.info
            log(f"decide migration: {decision.reason}")
        if decision.migrate is False and self.state() == MIGRATING:
            self._write_marker(IDLE, idle_reason=decision.reason)
        return decision.migrate is True

    # ── Hooks ────────────────────────────────────────────────────

    def startup(self) -> None:
        """At validator start, before any validator work: hand over if retired, else move if the trigger holds.
        Returns when this validator keeps running."""
        if self.state() == RETIRED:
            self.hand_off_retired()
        if self.check() and self.migrate():
            self.hand_off_retired()

    def on_epoch_close(self) -> None:
        """After an epoch close, before its weights: evaluate the trigger and start the migration in the
        background when it holds."""
        if self.state() == RETIRED:
            # Decide runs; pm2 has not removed this process yet.
            if self.pm2_app:
                self._spawn_pm2_removal()
            bt.logging.critical(
                "decide migration: this validator is retired and decide's validator runs; remove this one: "
                + self._removal_hint()
            )
            return
        if self.busy:
            return
        if self.check():
            self._worker = threading.Thread(target=self._migrate_and_hand_over, name="decide-migration",
                                            daemon=True)
            self._worker.start()

    def _migrate_and_hand_over(self) -> None:
        if self.migrate():
            self.hand_over()

    # ── Migration ────────────────────────────────────────────────

    def migrate(self) -> bool:
        """Stop weights, install decide and, under pm2, start it. True once decide runs (the marker says
        ``retired``); on failure, logs why with the commands to finish by hand and returns False."""
        with self._lock:
            if self._migrating:
                return False
            self._migrating = True
            self._triggered = True
        try:
            attempts = int(self.read_marker().get("attempts") or 0) + 1
            self._write_marker(MIGRATING, attempts=attempts)
            bt.logging.warning(
                f"decide migration: weight setting stopped; installing decide {DECIDE_REF or DECIDE_SHA[:12]} "
                f"in {self.target.decide_dir} (attempt {attempts}, log {self.target.log_path})"
            )
            network = decide_network(self.target.network, self.target.netuid)
            if network != self.target.network:
                bt.logging.warning(
                    f"decide migration: decide reaches the chain over TLS or on this machine only, so its validator "
                    f"uses {network}, the chain of netuid {self.target.netuid}, instead of {self.target.network}"
                )
            self._wait_weights_idle()
            problems = self.preflight()
            if problems:
                raise MigrationError("preflight: " + "; ".join(problems))
            self.fetch_decide()
            self._drop_broken_venv()
            self.run_installer()
            self.verify_install()
            if self.opted_out():
                bt.logging.warning(
                    "decide migration: opted out before the handover; decide is installed but not started"
                )
                with self._lock:
                    self._triggered = False
                self._write_marker(IDLE, idle_reason="opted out before the handover")
                return False
            if self.pm2_app:
                self.start_under_pm2()
            else:
                self.smoke_test()
            self._write_marker(RETIRED, last_error="")
            bt.logging.success(
                f"decide migration: decide's light validator is installed with hotkey "
                f"{self.target.wallet}/{self.target.hotkey}; this validator hands over to it"
            )
            return True
        except Exception as exc:
            self._record_failure(str(exc) if isinstance(exc, MigrationError) else f"{type(exc).__name__}: {exc}")
            return False
        finally:
            with self._lock:
                self._migrating = False

    def hand_over(self) -> None:
        """After ``migrate``: remove this pm2 app, or replace this process with decide's validator."""
        if self.pm2_app:
            self._spawn_pm2_removal()
            return
        try:
            self._exec_decide()
        except OSError as exc:
            self._record_failure(f"cannot start decide's validator: {exc}")

    def hand_off_retired(self) -> None:
        """Hand over to decide's validator, installing it again first when it is gone; never returns. Retries once
        per epoch while it cannot."""
        while True:
            reason = self._hand_off_once()
            self._try_write_marker(RETIRED, last_error=reason)
            bt.logging.critical(
                f"decide migration: this validator is retired and could not hand over to decide's validator: "
                f"{reason}; it sets no weights and retries in {RETRY_S / 60:.0f} minutes"
            )
            for line in self.manual_commands(installed=self.decide_installed()):
                bt.logging.critical(f"  {line}")
            self._sleep(RETRY_S)

    def _hand_off_once(self) -> str:
        """One handover, after installing decide again if it is gone; returns why it did not complete (a successful
        one does not return)."""
        try:
            if not self.decide_installed():
                self.reinstall()
            if self.pm2_app:
                if not self._pm2_online(DECIDE_APP):
                    self.start_under_pm2()
                self._spawn_pm2_removal()
                self._sleep(REMOVAL_WAIT_S)
                return f"pm2 did not remove {self.pm2_app} within {REMOVAL_WAIT_S:.0f} s"
            self._exec_decide()
            return "decide's validator did not start"
        except (MigrationError, OSError) as exc:
            return str(exc)

    def decide_installed(self) -> bool:
        """Whether decide's validator can start: its command and the Python of its environment are there. A recreated
        container keeps them only when the decide directory, and the Python uv installed for it, are on volumes."""
        return os.access(self.target.decide_bin, os.X_OK) and os.access(self.target.decide_python, os.X_OK)

    def reinstall(self) -> None:
        """Install the pinned decide again for this retired validator, whose decide is gone: the steps of the move,
        with its preflight, checks and time bounds. The marker stays retired, so no weights are set meanwhile.
        Raises MigrationError."""
        t = self.target
        reinstalls = int(self.read_marker().get("reinstalls") or 0) + 1
        self._try_write_marker(RETIRED, reinstalls=reinstalls)
        bt.logging.warning(
            f"decide migration: this validator moved to decide, but decide's validator is gone from {t.decide_dir}; "
            f"installing decide {DECIDE_REF or DECIDE_SHA[:12]} again (reinstall {reinstalls}, log {t.log_path})"
        )
        try:
            problems = self.preflight()
            if problems:
                raise MigrationError("preflight: " + "; ".join(problems))
            self.fetch_decide()
            self._drop_broken_venv()
            self.run_installer()
            self.verify_install()
            if not self.pm2_app:
                self.smoke_test()
        except MigrationError:
            raise
        except Exception as exc:
            raise MigrationError(f"{type(exc).__name__}: {exc}") from exc
        self._try_write_marker(RETIRED, last_error="")
        bt.logging.success(f"decide migration: decide's light validator is installed again in {t.decide_dir}")

    def _drop_broken_venv(self) -> None:
        """Remove decide's environment when its Python is gone (it lived outside a recreated container's volumes): the
        installer builds it anew, where it refuses to replace an existing one."""
        venv = self.target.decide_dir / ".venv"
        if venv.is_dir() and not os.access(self.target.decide_python, os.X_OK):
            bt.logging.warning(f"decide migration: the Python of {venv} is gone; the environment is built anew")
            shutil.rmtree(venv)

    def _try_write_marker(self, state: str, **fields) -> None:
        """Record ``state``; a marker that cannot be written (a full disk, a read-only home) is logged, not raised."""
        try:
            self._write_marker(state, **fields)
        except OSError as exc:
            bt.logging.error(f"decide migration: {self.target.marker_path} not written: {exc}")

    def _record_failure(self, reason: str) -> None:
        bt.logging.critical(
            f"decide migration failed: {reason}. This validator sets no weights while it waits to move; "
            f"it tries again after the next epoch close (log {self.target.log_path})"
        )
        bt.logging.critical("decide migration: to move by hand:")
        for line in self.manual_commands():
            bt.logging.critical(f"  {line}")
        bt.logging.critical(f"decide migration: to stay on this release instead: touch {self.target.opt_out_path}")
        self._try_write_marker(MIGRATING, last_error=reason)  # after the lines above, which say what to do

    def _wait_weights_idle(self) -> None:
        if self.weights_idle is None:
            return
        deadline = self._monotonic() + WEIGHTS_IDLE_TIMEOUT_S
        while not self.weights_idle():
            if self._monotonic() >= deadline:
                raise MigrationError("a weight submission of this validator is still in flight")
            self._sleep(1.0)

    # ── Steps ────────────────────────────────────────────────────

    def preflight(self) -> list[str]:
        """What stops the installation on this machine, before anything changes."""
        t = self.target
        problems: list[str] = []
        if platform.system() != "Linux":
            problems.append(f"decide runs on Linux, not {platform.system()}")
        free = shutil.disk_usage(Path.home()).free
        if free < MIN_FREE_BYTES:
            problems.append(f"{free / 2**30:.1f} GiB free in {Path.home()}; decide needs {MIN_FREE_BYTES >> 30} GiB")
        has_uv = shutil.which("uv") is not None or (Path.home() / ".local" / "bin" / "uv").exists()
        # ssh-keygen checks the release tag's signature, and decide's installer needs it without services.
        tools = ["git", "bash", "ssh-keygen"] + ([] if has_uv else ["curl"]) + (["pm2"] if self.pm2_app else [])
        missing = [tool for tool in tools if shutil.which(tool) is None]
        if missing:
            problems.append("missing: " + ", ".join("ssh-keygen (openssh-client)" if tool == "ssh-keygen" else tool
                                                    for tool in missing))
        wallet_dir = Path.home() / ".bittensor" / "wallets" / t.wallet
        for needed in (wallet_dir / "hotkeys" / t.hotkey, wallet_dir / "coldkeypub.txt"):
            if not needed.exists():
                problems.append(f"{needed} not found (decide reads the hotkey and coldkeypub.txt there)")
        if t.decide_dir.exists() and any(t.decide_dir.iterdir()) and not (
            is_decide_checkout(t.decide_dir) or unfinished_clone(t.decide_dir)
        ):
            problems.append(f"{t.decide_dir} exists and holds other files")
        problems.extend(self._setup_conflicts())
        if any(os.environ.get(name) for name in ("https_proxy", "HTTPS_PROXY", "all_proxy", "ALL_PROXY")):
            bt.logging.info("decide migration: an HTTPS proxy is configured; direct egress checks skipped")
        else:
            hosts = list(EGRESS_HOSTS) + ([] if has_uv else [UV_HOST])
            repo_host = urlparse(DECIDE_REPO_URL).hostname if "://" in DECIDE_REPO_URL else None
            if repo_host and repo_host not in hosts:
                hosts.insert(0, repo_host)
            unreachable = [host for host in hosts if not self._reachable(host)]
            if unreachable:
                problems.append("no HTTPS connection to " + ", ".join(unreachable))
        return problems

    def _setup_conflicts(self) -> list[str]:
        """A decide validator set up before on this machine that the migration must not change."""
        t = self.target
        problems = []
        try:
            existing = json.loads(t.config_path.read_text())
        except (OSError, ValueError):
            existing = {}
        if isinstance(existing, dict) and existing.get("wallet_name") and (
            (existing.get("wallet_name"), existing.get("hotkey_name")) != (t.wallet, t.hotkey)
        ):
            problems.append(
                f"{t.config_path} sets up decide for {existing.get('wallet_name')}/{existing.get('hotkey_name')}, "
                f"not {t.wallet}/{t.hotkey}"
            )
        try:
            record = json.loads((t.home / "install" / "validator.json").read_text())
        except (OSError, ValueError):
            record = {}
        if isinstance(record, dict) and record.get("manager") in ("systemd", "supervisor") and record.get("services"):
            problems.append(
                f"decide's validator already runs as {record['manager']} services; stop this validator by hand: "
                + self._removal_hint()
            )
        if self.pm2_app == DECIDE_APP:
            problems.append(f"this validator's pm2 app is named {DECIDE_APP}, the name decide's validator takes")
        return problems

    def fetch_decide(self) -> None:
        """Put the pinned decide commit in the decide directory, checked before it is checked out. A new directory
        is cloned beside it and renamed into place; an empty one that exists stays (it may be a mount point, as a
        docker volume is) and the clone is made in it."""
        d = self.target.decide_dir
        ref = DECIDE_REF or DECIDE_SHA
        if d.exists() and any(d.iterdir()) and not unfinished_clone(d):
            self._git(["-C", str(d), "fetch", "--quiet", "--tags", "--force", DECIDE_REPO_URL, ref])
            self._verify_fetched(d)
            self._git(["-C", str(d), "checkout", "--quiet", "--detach", DECIDE_SHA])
        else:
            work = d if d.exists() else d.with_name(d.name + ".partial")
            _discard(work, keep=work == d)
            self._git(["clone", "--quiet", "--no-checkout", DECIDE_REPO_URL, str(work)])
            self._git(["-C", str(work), "fetch", "--quiet", "--tags", "--force", DECIDE_REPO_URL, ref])
            try:
                self._verify_fetched(work)
            except MigrationError:
                _discard(work, keep=work == d)
                raise
            self._git(["-C", str(work), "checkout", "--quiet", "--detach", DECIDE_SHA])
            if work != d:
                work.rename(d)
        head = self._git(["-C", str(d), "rev-parse", "HEAD"]).strip()
        if head != DECIDE_SHA:
            raise MigrationError(f"{d} is at {head[:12]}, not the pinned commit {DECIDE_SHA[:12]}")
        if not is_decide_checkout(d):
            raise MigrationError(f"the pinned commit {DECIDE_SHA[:12]} is not a checkout of decide")

    def _verify_fetched(self, checkout: Path) -> None:
        fetched = self._git(["-C", str(checkout), "rev-parse", "FETCH_HEAD^{commit}"]).strip()
        if fetched != DECIDE_SHA:
            raise MigrationError(
                f"{DECIDE_REF or 'the fetched ref'} of {DECIDE_REPO_URL} is commit {fetched[:12]}, not the pinned "
                f"{DECIDE_SHA[:12]}"
            )
        if DECIDE_TAG_SIGNERS:
            self.verify_tag_signature(checkout)

    def verify_tag_signature(self, checkout: Path) -> None:
        """Verify the ssh signature of the tag DECIDE_REF against DECIDE_TAG_SIGNERS."""
        with tempfile.NamedTemporaryFile("w", suffix=".allowed_signers", delete=False) as signers:
            signers.write(DECIDE_TAG_SIGNERS.strip() + "\n")
        try:
            self._git([
                "-C", str(checkout), "-c", "gpg.format=ssh", "-c", f"gpg.ssh.allowedSignersFile={signers.name}",
                "verify-tag", DECIDE_REF,
            ], what=f"the signature of {DECIDE_REF}")
            tagged = self._git(["-C", str(checkout), "rev-parse", f"{DECIDE_REF}^{{commit}}"]).strip()
        finally:
            os.unlink(signers.name)
        if tagged != DECIDE_SHA:
            raise MigrationError(f"tag {DECIDE_REF} names {tagged[:12]}, not the pinned {DECIDE_SHA[:12]}")

    def install_command(self) -> list[str]:
        t = self.target
        return [
            "bash", str(t.decide_dir / "scripts" / "install.sh"), "validator", "--light", "--yes", "--no-service",
            "--dir", str(t.decide_dir), "--source", DECIDE_REPO_URL, "--ref", DECIDE_SHA,
            "--network", decide_network(t.network, t.netuid), "--netuid", str(t.netuid),
            "--wallet", t.wallet, "--hotkey", t.hotkey,
        ] + (["--dry-run"] if t.dry_run else [])

    def pm2_start_command(self) -> list[str]:
        t = self.target
        return [
            "pm2", "start", str(t.decide_bin), "--name", DECIDE_APP, "--interpreter", "none", "--cwd", str(t.home),
            "--", "--config", str(t.config_path),
        ]

    def run_installer(self) -> None:
        code, output = self._runner(
            self.install_command(), timeout=INSTALL_TIMEOUT_S, env=self._child_env(), cwd=Path.home(),
            log_path=self.target.log_path,
        )
        if code != 0:
            raise MigrationError(f"decide's installer exited with {code}: {_last_line(output)}")

    def verify_install(self) -> None:
        t = self.target
        head = self._git(["-C", str(t.decide_dir), "rev-parse", "HEAD"]).strip()
        if head != DECIDE_SHA:
            raise MigrationError(f"the installer left {t.decide_dir} at {head[:12]}, not {DECIDE_SHA[:12]}")
        if not os.access(t.decide_bin, os.X_OK):
            raise MigrationError(f"{t.decide_bin} is missing after the installation")
        try:
            config = json.loads(t.config_path.read_text())
        except (OSError, ValueError) as exc:
            raise MigrationError(f"{t.config_path} is missing or invalid: {exc}") from None
        expected = {"mode": "light", "wallet_name": t.wallet, "hotkey_name": t.hotkey, "netuid": t.netuid}
        if t.dry_run:
            expected["dry_run"] = True
        wrong = {k: config.get(k) for k, v in expected.items() if config.get(k) != v}
        if wrong:
            raise MigrationError(f"{t.config_path} does not set up this light validator: {wrong}")

    def smoke_test(self) -> None:
        code, output = self._runner([str(self.target.decide_bin), "--help"], timeout=120.0, env=self._child_env(),
                                    cwd=self.target.home, log_path=self.target.log_path)
        if code != 0:
            raise MigrationError(f"decide's validator does not run: {_last_line(output)}")

    def start_under_pm2(self) -> None:
        """Start decide's validator as pm2 app ``verathos-validator`` and wait for its ``light mode:`` line; on
        failure the app is removed again."""
        if self._pm2_app_info(DECIDE_APP) is not None:
            self._pm2(["delete", DECIDE_APP])
        offsets = {path: _size(path) for path in self._default_pm2_logs()}
        code, output = self._runner(self.pm2_start_command(), timeout=120.0, env=self._pm2_env(),
                                    cwd=self.target.home, log_path=self.target.log_path)
        if code != 0:
            raise MigrationError(f"pm2 start {DECIDE_APP} failed: {_last_line(output)}")
        try:
            self._await_light_mode(offsets)
        except MigrationError:
            self._pm2(["delete", DECIDE_APP], check=False)
            raise

    def _await_light_mode(self, offsets: dict[Path, int]) -> None:
        deadline = self._monotonic() + STARTUP_TIMEOUT_S
        while True:
            info = self._pm2_app_info(DECIDE_APP)
            if info is None:
                raise MigrationError(f"{DECIDE_APP} is not in pm2's list after its start")
            env = info.get("pm2_env") or {}
            text = ""
            for key in ("pm_out_log_path", "pm_err_log_path"):
                path = Path(env[key]) if env.get(key) else None
                if path is not None:
                    text += _read_from(path, offsets.get(path, 0))
            if LIGHT_MODE_LINE in text:
                return
            refused = next((line for line in text.splitlines() if any(r in line for r in DECIDE_REFUSALS)), None)
            if refused:
                raise MigrationError(f"decide's validator refused to start: {refused.strip()}")
            if env.get("status") in ("errored", "stopped"):
                raise MigrationError(f"{DECIDE_APP} is {env.get('status')} in pm2: {_last_line(text)}")
            if self._monotonic() >= deadline:
                raise MigrationError(f"no '{LIGHT_MODE_LINE}' line from {DECIDE_APP} within "
                                     f"{STARTUP_TIMEOUT_S:.0f} s: {_last_line(text)}")
            self._sleep(3.0)

    def _spawn_pm2_removal(self) -> None:
        app = shlex.quote(str(self.pm2_app))
        self._spawn(f"sleep 5; pm2 delete {app}; pm2 save", self.target.log_path)

    def _exec_decide(self) -> None:
        t = self.target
        if not os.access(t.decide_bin, os.X_OK):
            raise MigrationError(f"{t.decide_bin} not found")
        bt.logging.info(f"decide migration: replacing this process with {t.decide_bin} --config {t.config_path}")
        os.chdir(t.home)
        # bittensor's logging writes from a queue that only atexit drains, and an exec runs no atexit handlers.
        listener = getattr(bt.logging, "_listener", None)
        if callable(getattr(listener, "stop", None)):
            listener.stop()
        for stream in (sys.stdout, sys.stderr):
            stream.flush()
        try:
            self._execve(str(t.decide_bin), [str(t.decide_bin), "--config", str(t.config_path)], self._child_env())
        finally:  # reached only when the exec failed
            if callable(getattr(listener, "start", None)):
                listener.start()

    # ── Helpers ──────────────────────────────────────────────────

    def manual_commands(self, installed: bool = False) -> list[str]:
        """The commands that finish the migration by hand."""
        t = self.target
        d = shlex.quote(str(t.decide_dir))
        lines = []
        if not installed:
            if not (t.decide_dir / ".git").exists():
                lines.append(f"git clone --no-checkout {shlex.quote(DECIDE_REPO_URL)} {d} && "
                             f"git -C {d} checkout --detach {DECIDE_SHA}")
            lines.append(shlex.join(self.install_command()))
        run = shlex.join(self.pm2_start_command())
        if self.pm2_app:
            lines.append(f"HOME={shlex.quote(str(Path.home()))} {run} && pm2 delete {shlex.quote(self.pm2_app)} "
                         "&& pm2 save")
        else:
            lines.append("stop this validator, then run under the same supervisor: "
                         f"cd {shlex.quote(str(t.home))} && {shlex.quote(str(t.decide_bin))} --config "
                         f"{shlex.quote(str(t.config_path))}")
        return lines

    def _removal_hint(self) -> str:
        if self.pm2_app:
            return f"pm2 delete {self.pm2_app} && pm2 save"
        return "stop its service or process"

    def _child_env(self) -> dict[str, str]:
        """This environment without this Python environment's variables and bin directory, with HOME set."""
        env = {k: v for k, v in os.environ.items()
               if k not in ("VIRTUAL_ENV", "PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP")}
        own_bin = str(Path(sys.prefix) / "bin") if sys.prefix != sys.base_prefix else None
        path = [p for p in env.get("PATH", "/usr/local/bin:/usr/bin:/bin").split(os.pathsep) if p and p != own_bin]
        local_bin = str(Path.home() / ".local" / "bin")
        env["PATH"] = os.pathsep.join([local_bin] + [p for p in path if p != local_bin])
        env["HOME"] = str(Path.home())
        return env

    def _pm2_env(self) -> dict[str, str]:
        child = self._child_env()
        return {key: child[key] for key in PM2_ENV_KEYS if key in child}

    def _git(self, args: list[str], *, what: Optional[str] = None) -> str:
        code, output = self._runner(["git"] + args, timeout=GIT_TIMEOUT_S, env=self._child_env(), cwd=Path.home())
        if code != 0:
            subcommand = args[2] if args[0] == "-C" else args[0]
            raise MigrationError(f"{what or 'git ' + subcommand} failed: {_last_line(output)}")
        return output

    def _pm2(self, args: list[str], *, check: bool = True) -> str:
        code, output = self._runner(["pm2"] + args, timeout=120.0, env=self._pm2_env(), cwd=self.target.home)
        if code != 0 and check:
            raise MigrationError(f"pm2 {args[0]} failed: {_last_line(output)}")
        return output

    def _pm2_app_info(self, name: str) -> Optional[dict]:
        apps = _pm2_apps(self._pm2(["jlist"]))
        if apps is None:
            raise MigrationError("pm2 jlist did not return JSON")
        return next((app for app in apps if app.get("name") == name), None)

    def _pm2_online(self, name: str) -> bool:
        info = self._pm2_app_info(name)
        return info is not None and (info.get("pm2_env") or {}).get("status") == "online"

    def _default_pm2_logs(self) -> list[Path]:
        pm2_home = Path(os.environ.get("PM2_HOME") or Path.home() / ".pm2")
        return [pm2_home / "logs" / f"{DECIDE_APP}-out.log", pm2_home / "logs" / f"{DECIDE_APP}-error.log"]

    def _default_chain(self):
        subtensor_cls = getattr(bt, "Subtensor", None) or bt.subtensor
        return subtensor_cls(network=self.target.network)

    @staticmethod
    def _default_reachable(host: str) -> bool:
        try:
            with socket.create_connection((host, 443), timeout=10):
                return True
        except OSError:
            return False


def _pm2_apps(output: str) -> Optional[list[dict]]:
    """The app list in the output of ``pm2 jlist``, which may come with pm2's notices before or after it."""
    decoder = json.JSONDecoder()
    for start in (match.start() for match in re.finditer(r"\[", output)):
        try:
            value, _ = decoder.raw_decode(output, start)
        except ValueError:
            continue
        if isinstance(value, list) and all(isinstance(app, dict) and "pm2_env" in app for app in value):
            return value
    return None


def _size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _read_from(path: Path, offset: int) -> str:
    try:
        with path.open("rb") as handle:
            size = handle.seek(0, os.SEEK_END)
            handle.seek(offset if offset <= size else 0)
            return handle.read().decode(errors="replace")
    except OSError:
        return ""
