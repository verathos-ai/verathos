"""The stock join installer observes a durable accepted verdict."""

from pathlib import Path

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
