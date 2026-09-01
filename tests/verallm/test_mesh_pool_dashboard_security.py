"""Stored-XSS regressions for the actively served mesh operator dashboard."""

from __future__ import annotations

import re

from verallm.mesh.pool_dashboard import DASHBOARD_HTML


def test_generated_event_handlers_never_interpolate_runtime_values() -> None:
    handlers = re.findall(
        r"""\bon[a-z]+=(?:"([^"]*)"|'([^']*)')""",
        DASHBOARD_HTML,
    )

    assert handlers
    for double_quoted, single_quoted in handlers:
        handler = double_quoted or single_quoted
        assert "${" not in handler


def test_worker_and_mesh_actions_use_escaped_data_attributes() -> None:
    assert 'data-worker-id="${esc(id)}"' in DASHBOARD_HTML
    assert "removeWorker(this.dataset.workerId)" in DASHBOARD_HTML
    assert "toggleWorker(this.dataset.workerId)" in DASHBOARD_HTML
    assert 'data-mesh-key="${esc(k)}"' in DASHBOARD_HTML
    assert "toggleChat(this.dataset.meshKey)" in DASHBOARD_HTML
    assert "stopMesh(this.dataset.meshKey)" in DASHBOARD_HTML
    assert "openMeshTest(this.dataset.meshKey)" in DASHBOARD_HTML
    assert "sendChat(this.dataset.meshKey)" in DASHBOARD_HTML

    for unsafe in (
        "removeWorker('${esc(id)}')",
        "toggleWorker('${esc(id)}')",
        "toggleChat('${esc(k)}')",
        "stopMesh('${esc(k)}')",
        "openMeshTest('${esc(k)}')",
        "sendChat('${esc(k)}')",
    ):
        assert unsafe not in DASHBOARD_HTML


def test_launch_placement_json_is_data_not_executable_handler_source() -> None:
    assert (
        'const placement=esc(JSON.stringify({workers:s.workers,driver:s.driver,model}));'
        in DASHBOARD_HTML
    )
    assert 'data-placement="${placement}"' in DASHBOARD_HTML
    assert 'onclick="launchPlacement(this)"' in DASHBOARD_HTML
    assert "JSON.parse(el.dataset.placement" in DASHBOARD_HTML
    assert "onclick='doLaunch(${JSON.stringify" not in DASHBOARD_HTML


def test_worker_status_and_capability_output_is_context_safe() -> None:
    start = DASHBOARD_HTML.index("function renderWorkers(workers)")
    end = DASHBOARD_HTML.index("function modelCatalog(", start)
    worker_renderer = DASHBOARD_HTML[start:end]

    assert "const SAFE_STATUS_CLASSES = new Set(" in DASHBOARD_HTML
    assert "function statusClass(value)" in DASHBOARD_HTML
    assert 'class="pill ${cls}"' in worker_renderer
    assert "${esc(label)}</span>" in worker_renderer
    assert 'cap.vram_gb?esc(cap.vram_gb)+" GB"' in worker_renderer
    assert "esc((w.capability||{}).vram_gb||0)" in DASHBOARD_HTML
    assert 'class="pill ${statusClass(phase.st)}"' in DASHBOARD_HTML

    assert 'class="pill ${st}"' not in worker_renderer
    assert 'class="pill ${phase.st}"' not in DASHBOARD_HTML
    assert "${label}</span>" not in worker_renderer
    assert 'cap.vram_gb?cap.vram_gb+" GB"' not in worker_renderer


def test_stream_result_metadata_is_escaped_or_numeric_before_inner_html() -> None:
    start = DASHBOARD_HTML.index("function renderStreamEvent(ev)")
    end = DASHBOARD_HTML.index("try {", start)
    renderer = DASHBOARD_HTML[start:end]

    assert "${esc(ev.error)}" in renderer
    assert "const reportedTokens=Number(" in renderer
    assert "Number.isFinite(reportedTokens)" in renderer
    assert "const engineTps=Number(ev.engine_tps)" in renderer
    assert "Number(ev.receipts)||0" in renderer
    assert "${(ev.usage||{}).completion_tokens}" not in renderer
