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
