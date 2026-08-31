"""A subnet worker without the hot-capacity workspace must die loudly.

Serving with the audit workspace missing means the worker runs
audit-blind, records no evidence, and probates with nothing in its own
logs explaining why. The start path therefore raises unless the operator
explicitly opts out via VERATHOS_ALLOW_NO_CAPACITY_AUDITS=1 (light-tier
or dev setups only).
"""

from __future__ import annotations

import inspect
import subprocess

from verallm.mesh import capacity_audit_worker as caw_mod

JOIN_SCRIPT = "scripts/join_pool.sh"


def test_start_raises_typed_error_on_missing_workspace_by_default():
    src = inspect.getsource(caw_mod)
    block_start = src.index("ok, error = self._preflight_workspace()")
    block = src[block_start : block_start + 2200]
    assert "VERATHOS_ALLOW_NO_CAPACITY_AUDITS" in block
    assert "raise CapacityAuditWorkspaceError" in block
    raise_at = block.index("raise CapacityAuditWorkspaceError")
    optout_at = block.index("VERATHOS_ALLOW_NO_CAPACITY_AUDITS")
    assert optout_at < raise_at
    assert issubclass(caw_mod.CapacityAuditWorkspaceError, RuntimeError)


def test_pool_kills_worker_on_workspace_error_and_retries_transients():
    """The pool's start-catch must not swallow the fatal error.

    A blanket except around audit_worker.start() used to log-and-continue,
    so the advertised fail-closed behavior never happened, and the context
    digest memo pinned the failed start so heartbeats never retried.
    """

    from verallm.mesh import pool as pool_mod

    src = inspect.getsource(pool_mod)
    start_at = src.index("if audit_worker.start():")
    block = src[start_at : start_at + 2400]
    assert "CapacityAuditWorkspaceError" in block
    fatal_at = block.index("os._exit(78)")
    typed_at = block.index("isinstance(exc, CapacityAuditWorkspaceError)")
    assert typed_at < fatal_at
    # Transient failures clear the digest memo so the next sync retries.
    retry_at = block.index('capacity_audit_state["digest"] = ""')
    assert fatal_at < retry_at


def test_join_script_fails_closed_without_wheel(tmp_path):
    text = open(JOIN_SCRIPT).read()
    assert "VERATHOS_ALLOW_NO_CAPACITY_AUDITS" in text
    assert '_die "no hot_capacity_workspace_cuda wheel' in text
    # The tolerant warning survives only inside the explicit opt-out branch.
    warn_at = text.index("capacity audits DISABLED (explicitly allowed")
    gate_at = text.index('[ "${VERATHOS_ALLOW_NO_CAPACITY_AUDITS:-}" = "1" ]')
    assert gate_at < warn_at


def test_join_script_parses():
    subprocess.run(["bash", "-n", JOIN_SCRIPT], check=True)
