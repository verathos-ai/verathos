"""A failing worker join must never die silently.

Two silent layers existed: SystemExit(int) exits with no message, and a
`set -e` abort in join_pool.sh on a command with no stderr of its own
produced zero output. Both now speak.
"""

from __future__ import annotations

import inspect
import subprocess

from verallm.mesh import onboarding as onb_mod

JOIN_SCRIPT = "scripts/join_pool.sh"


def test_wizard_raises_systemexit_with_message_not_bare_code():
    src = inspect.getsource(onb_mod.setup_mesh_worker)
    assert "raise SystemExit(completed.returncode)" not in src
    assert "worker join failed" in src


def test_join_script_has_err_trap():
    text = open(JOIN_SCRIPT).read()
    trap_at = text.index("trap 'echo \"join_pool.sh: aborted at line")
    set_at = text.index("set -euo pipefail")
    assert set_at < trap_at


def test_err_trap_fires_on_silent_abort(tmp_path):
    script = tmp_path / "t.sh"
    trap_line = next(
        line
        for line in open(JOIN_SCRIPT).read().split("\n")
        if line.startswith("trap 'echo \"join_pool.sh: aborted")
    )
    script.write_text("set -euo pipefail\n" + trap_line + "\nfalse\n")
    completed = subprocess.run(
        ["bash", str(script)], capture_output=True, text=True
    )
    assert completed.returncode != 0
    assert "aborted at line" in completed.stderr
