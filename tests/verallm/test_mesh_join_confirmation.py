"""The stock join installer observes a durable accepted verdict."""

import os
import subprocess
from pathlib import Path

import pytest

from verallm.mesh.cli import _print_pool_join_accepted


def test_pool_worker_prints_the_installer_join_success_marker(capsys):
    _print_pool_join_accepted({"worker_id": "worker-a"})
    assert capsys.readouterr().out == "pool join accepted: worker 'worker-a'\n"


def test_stock_installer_waits_for_the_same_marker():
    repo = Path(__file__).resolve().parents[2]
    installer = (repo / "scripts/join_pool.sh").read_text(encoding="utf-8")
    assert 'grep -q "pool join accepted"' in installer


def test_stock_installer_clears_stale_pm2_join_verdicts_before_start():
    repo = Path(__file__).resolve().parents[2]
    installer = (repo / "scripts/join_pool.sh").read_text(encoding="utf-8")
    linux_start = installer[installer.index("if command -v pm2") :]

    delete_at = linux_start.index('pm2 delete "$name"')
    clear_out_at = linux_start.index(
        ': > "$HOME/.pm2/logs/${name}-out.log"'
    )
    clear_error_at = linux_start.index(
        ': > "$HOME/.pm2/logs/${name}-error.log"'
    )
    start_at = linux_start.index('pm2 start "$VENV/bin/python"')

    assert delete_at < clear_out_at < start_at
    assert delete_at < clear_error_at < start_at


@pytest.mark.parametrize("existing", [False, True])
def test_stock_installer_initializes_catalog_parent_without_overwriting(
    tmp_path, existing
):
    repo = Path(__file__).resolve().parents[2]
    installer = (repo / "scripts/join_pool.sh").read_text(encoding="utf-8")
    initialization = installer.split("# --- runtime environment", 1)[0].rsplit(
        "PYEOF\n", 1
    )[1]
    catalog = tmp_path / "fresh home" / ".verathos" / "pool-catalog.json"
    previous = '[{"model_id":"existing-model"}]\n'
    if existing:
        catalog.parent.mkdir(parents=True)
        catalog.write_text(previous, encoding="utf-8")
    result = subprocess.run(
        ["bash", "-euc", initialization],
        env={**os.environ, "CATALOG": str(catalog)},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert catalog.read_text(encoding="utf-8") == (previous if existing else "[]\n")
