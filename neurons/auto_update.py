"""Auto-update watchtower for Verathos neurons.

Periodically checks the Git remote for new commits and, when the relevant
role version has been bumped, pulls the latest code, reinstalls the
package, and restarts the process.

Version-aware restarts
----------------------
The updater does NOT restart on every commit.  It reads
``neurons/version.py`` from the **remote** (via ``git show``) and
compares the role-specific version:

- Miner processes check ``miner_version``
- Validator and proxy processes check ``validator_version``

Only if the remote version is **higher** does it pull + restart.  This
means a validator-only fix doesn't bounce miners (and vice versa).

CLI integration
---------------
Each neuron adds ``--auto-update`` to its argparser.  When enabled::

    from neurons.auto_update import AutoUpdater
    updater = AutoUpdater(role="validator", check_interval=1800)
    updater.start()
"""

from __future__ import annotations

import logging
import bittensor as bt
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# Repo root = directory containing this file's parent (neurons/ -> repo root)
_REPO_ROOT = Path(__file__).resolve().parent.parent


def _run_git(*args: str, cwd: Optional[Path] = None) -> tuple[int, str]:
    """Run a git command and return (returncode, stdout+stderr)."""
    cmd = ["git"] + list(args)
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd or _REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        )
        output = (result.stdout + result.stderr).strip()
        return result.returncode, output
    except subprocess.TimeoutExpired:
        return 1, "git command timed out"
    except Exception as e:
        return 1, str(e)


def get_local_head() -> Optional[str]:
    """Get the current local HEAD commit hash."""
    rc, out = _run_git("rev-parse", "HEAD")
    return out if rc == 0 else None


def get_current_branch() -> Optional[str]:
    """Get the current branch name."""
    rc, out = _run_git("rev-parse", "--abbrev-ref", "HEAD")
    return out if rc == 0 else None


def fetch_origin() -> bool:
    """Fetch from origin. Returns True on success."""
    rc, out = _run_git("fetch", "origin", "--quiet")
    if rc != 0:
        if "not a git repository" in out:
            bt.logging.debug("Auto-update skipped: not a git repository")
        else:
            bt.logging.warning(f"git fetch failed: {out}")
        return False
    return True


def get_remote_head(branch: Optional[str] = None) -> Optional[str]:
    """Return the remote HEAD hash for the current branch (after fetch)."""
    if branch is None:
        branch = get_current_branch()
        if not branch:
            return None
    rc, remote_hash = _run_git("rev-parse", f"origin/{branch}")
    return remote_hash if rc == 0 else None


def _read_remote_version_file(branch: Optional[str] = None) -> Optional[str]:
    """Read neurons/version.py from the remote branch without pulling.

    Uses ``git show origin/<branch>:neurons/version.py`` so we can
    inspect the remote version before deciding whether to pull.
    """
    if branch is None:
        branch = get_current_branch()
        if not branch:
            return None

    rc, content = _run_git("show", f"origin/{branch}:neurons/version.py")
    if rc != 0:
        bt.logging.warning(f"Cannot read remote version.py: {content}")
        return None
    return content


def _parse_version_from_source(source: str, var_name: str) -> Optional[int]:
    """Extract a version integer from version.py source code.

    Looks for ``<var_name>: int = _encode(MAJOR, MINOR, PATCH)`` or
    the raw assignment pattern and computes the value.

    We parse the MAJOR/MINOR/PATCH constants for each role prefix and
    compute the encoded integer.
    """
    # Determine prefix: miner_version → MINER_, validator_version → VALIDATOR_,
    # spec_version → SPEC_
    prefix_map = {
        "miner_version": "MINER_",
        "validator_version": "VALIDATOR_",
        "spec_version": "SPEC_",
    }
    prefix = prefix_map.get(var_name)
    if not prefix:
        return None

    def _find_int(name: str) -> Optional[int]:
        pattern = rf"^{re.escape(name)}\s*=\s*(\d+)"
        m = re.search(pattern, source, re.MULTILINE)
        return int(m.group(1)) if m else None

    major = _find_int(f"{prefix}MAJOR")
    minor = _find_int(f"{prefix}MINOR")
    patch = _find_int(f"{prefix}PATCH")

    if major is None or minor is None or patch is None:
        return None

    return major * 1_000_000 + minor * 1_000 + patch


def check_remote_version(role: str) -> Optional[tuple[str, int, int]]:
    """Check if the remote has a newer version for the given role.

    Parameters
    ----------
    role:
        "miner", "validator", or "proxy" (proxy uses validator_version).

    Returns
    -------
    (remote_commit, remote_version, local_version) if update needed,
    None if up to date or on error.
    """
    if not fetch_origin():
        return None

    local_head = get_local_head()
    remote_head = get_remote_head()
    if not local_head or not remote_head:
        return None

    # No new commits at all — skip version parsing
    if local_head == remote_head:
        return None

    # New commits exist — check if our role's version changed
    var_name = "miner_version" if role == "miner" else "validator_version"

    # Local version (from the imported module)
    from neurons.version import miner_version, validator_version
    local_version = miner_version if role == "miner" else validator_version

    # Remote version (from git show, without pulling)
    remote_source = _read_remote_version_file()
    if remote_source is None:
        # Can't read remote version.py — remote doesn't have it yet, skip
        bt.logging.debug("Remote has no neurons/version.py — skipping update")
        return None

    remote_version = _parse_version_from_source(remote_source, var_name)
    if remote_version is None:
        # Can't parse version from remote — malformed file, skip
        bt.logging.warning(f"Cannot parse {var_name} from remote version.py — skipping")
        return None

    if remote_version > local_version:
        bt.logging.info(f"Remote {var_name}={remote_version} > local {local_version} (commits: {local_head[:8]}→{remote_head[:8]})")
        return (remote_head, remote_version, local_version)

    # New commits but our role's version didn't change — skip
    bt.logging.debug(f"New commits available ({local_head[:8]}→{remote_head[:8]}) but {var_name} unchanged ({local_version}) — skipping")
    return None


def pull_and_install() -> bool:
    """Pull latest code and reinstall the package.

    Returns True on success, False on failure.
    """
    branch = get_current_branch()
    if not branch:
        bt.logging.error("Cannot determine current branch for pull")
        return False

    # Pull (suppress git hints about diverging branches)
    _run_git("config", "advice.diverging", "false")
    rc, out = _run_git("pull", "origin", branch, "--ff-only")
    if rc != 0:
        bt.logging.debug(f"Fast-forward pull not possible, resetting to origin/{branch}")
        rc2, out2 = _run_git("reset", "--hard", f"origin/{branch}")
        if rc2 != 0:
            bt.logging.error(f"git reset also failed: {out2}")
            return False
        bt.logging.info(f"Reset to origin/{branch} successful")

    new_head = get_local_head()
    _head = new_head[:8] if new_head else "unknown"
    bt.logging.info(f"Pulled to {_head}")

    # Reinstall package (editable mode, same extras)
    bt.logging.info("Reinstalling package...")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-e", ".", "--quiet"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            bt.logging.error(f"pip install failed: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        bt.logging.error("pip install timed out")
        return False

    bt.logging.info("Package reinstalled successfully")
    return True


def _detect_pm2() -> Optional[str]:
    """Detect if running under PM2 and return the process name.

    PM2 sets ``pm_id`` in the environment for managed processes.
    We use ``pm2 jlist`` to find our process name by PID.
    """
    if "pm_id" not in os.environ:
        return None

    try:
        import json
        result = subprocess.run(
            ["pm2", "jlist"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return None

        processes = json.loads(result.stdout)
        my_pid = os.getpid()
        for proc in processes:
            if proc.get("pid") == my_pid:
                return proc.get("name")
            # Also check parent PID (PM2 might have spawned us)
            if proc.get("pid") == os.getppid():
                return proc.get("name")

        # Fallback: use pm_id to find name
        pm_id = os.environ.get("pm_id")
        for proc in processes:
            if str(proc.get("pm_id")) == pm_id:
                return proc.get("name")

    except Exception as e:
        bt.logging.debug(f"PM2 detection failed: {e}")

    return None


def restart_process() -> None:
    """Restart the current process.

    Uses PM2 if detected, otherwise re-execs via os.execv().
    This function does not return.
    """
    pm2_name = _detect_pm2()

    if pm2_name:
        bt.logging.info(f"Restarting via PM2 (process: {pm2_name})...")
        try:
            subprocess.Popen(
                ["pm2", "restart", pm2_name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as e:
            bt.logging.error(f"PM2 restart failed: {e}, falling back to execv")
            _execv_restart()
        # Give PM2 a moment to register the restart, then exit
        time.sleep(2)
        sys.exit(0)
    else:
        _execv_restart()


def _execv_restart() -> None:
    """Re-exec the current process in-place (no PM2)."""
    bt.logging.info(f"Re-executing process: {sys.executable} {sys.argv}")
    os.execv(sys.executable, [sys.executable] + sys.argv)


class AutoUpdater:
    """Background thread that checks for role-specific version bumps.

    Parameters
    ----------
    role:
        "miner", "validator", or "proxy".  Determines which version
        variable is compared (``miner_version`` vs ``validator_version``).
        Proxy uses ``validator_version`` since they share the same codebase.
    check_interval:
        Seconds between update checks (default: 1800 = 30 min).
    restart_delay:
        Seconds to wait after pulling before restarting (default: 5).
        Gives in-flight requests time to complete.
    busy_check:
        Optional callback returning True if the process is busy and should
        NOT be restarted right now (e.g., validator mid-epoch-close).
    """

    def __init__(
        self,
        role: str = "validator",
        check_interval: int = 1800,
        restart_delay: int = 5,
        busy_check: Optional[Callable[[], bool]] = None,
    ):
        self.role = role if role != "proxy" else "validator"  # proxy uses validator_version
        self.check_interval = check_interval
        self.restart_delay = restart_delay
        self.busy_check = busy_check
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._update_pending = False  # Set when update deferred due to busy

    def start(self) -> None:
        """Start the background update checker thread."""
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, name="auto-updater", daemon=True,
        )
        self._thread.start()

        var_name = "miner_version" if self.role == "miner" else "validator_version"
        _branch = get_current_branch() or "unknown"
        bt.logging.info(f"Auto-updater started (role={self.role}, watching={var_name}, interval={self.check_interval}s, branch={_branch})")

    def stop(self) -> None:
        """Signal the updater to stop."""
        self._stop_event.set()

    def _run(self) -> None:
        """Main loop: check → pull → restart."""
        # Initial delay — let the neuron finish startup before first check
        self._wait(60)

        while not self._stop_event.is_set():
            try:
                self._check_and_update()
            except Exception as e:
                bt.logging.error(f"Auto-update check failed: {e}")

            self._wait(self.check_interval)

    def _wait(self, seconds: int) -> None:
        """Interruptible sleep."""
        self._stop_event.wait(timeout=seconds)

    def notify_not_busy(self) -> None:
        """Call this when the process is no longer busy (e.g. epoch close done).

        If an update was deferred, applies it immediately instead of waiting
        for the next check cycle.
        """
        if not self._update_pending:
            return
        bt.logging.info("Deferred update ready — applying now")
        self._update_pending = False
        if not pull_and_install():
            bt.logging.error("Deferred update failed — will retry next cycle")
            return
        bt.logging.info(f"Update applied, restarting in {self.restart_delay}s...")
        time.sleep(self.restart_delay)
        restart_process()

    def _check_and_update(self) -> None:
        """Single update check cycle."""
        result = check_remote_version(self.role)
        if result is None:
            bt.logging.debug(f"No update needed for role={self.role}")
            return

        remote_commit, remote_ver, local_ver = result
        var_name = "miner_version" if self.role == "miner" else "validator_version"
        bt.logging.info(f"Update required: {var_name} {local_ver} → {remote_ver} (remote commit {remote_commit[:8]})")

        # Check if we're busy
        if self.busy_check and self.busy_check():
            bt.logging.info("Process is busy — update scheduled for next idle window")
            self._update_pending = True
            return

        self._update_pending = False
        if not pull_and_install():
            bt.logging.error("Update failed — will retry next cycle")
            return

        bt.logging.info(f"Update applied, restarting in {self.restart_delay}s...")
        self._wait(self.restart_delay)

        if self.busy_check and self.busy_check():
            bt.logging.info("Process became busy during restart delay — deferring")
            self._update_pending = True
            return

        restart_process()
