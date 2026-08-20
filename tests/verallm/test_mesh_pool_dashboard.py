"""Operator dashboard UI contract."""

from verallm.mesh.pool_dashboard import DASHBOARD_HTML


def test_operator_ui_is_wallet_first() -> None:
    assert "Connect wallet" in DASHBOARD_HTML
    assert "Use pool token" not in DASHBOARD_HTML
    assert "Copy pool token only" not in DASHBOARD_HTML
    assert "verathos.pool.token" not in DASHBOARD_HTML
    assert "out.pool_secret" not in DASHBOARD_HTML


def test_operator_ui_has_miner_workflow_and_verathos_brand() -> None:
    for page in ("overview", "machines", "launch", "meshes"):
        assert f'data-page="{page}"' in DASHBOARD_HTML
        assert f'id="page-{page}"' in DASHBOARD_HTML
    assert "Secure enrollment command" in DASHBOARD_HTML
    assert "Waiting for the command to run" in DASHBOARD_HTML
    assert 'stroke-dasharray="22.3 11.2"' in DASHBOARD_HTML
    assert 'stroke-dasharray="33 11"' in DASHBOARD_HTML
    assert 'stroke-dasharray="45.79 8.67"' in DASHBOARD_HTML


def test_overview_exposes_compact_resources_and_fast_mesh_test() -> None:
    assert 'id="overviewMachines"' in DASHBOARD_HTML
    assert 'id="overviewMeshes"' in DASHBOARD_HTML
    assert 'onclick="openMeshTest(' in DASHBOARD_HTML
    assert 'disabled title="owner access required">Test' not in DASHBOARD_HTML
    assert "max_tokens:256" in DASHBOARD_HTML
    assert 'localStorage.getItem("mesh_thinking")==="1"' in DASHBOARD_HTML


def test_overview_visualizes_current_machine_to_mesh_allocations() -> None:
    assert 'id="topologyView"' in DASHBOARD_HTML
    assert 'id="topologyFlowButton"' in DASHBOARD_HTML
    assert 'id="topologyMatrixButton"' in DASHBOARD_HTML
    assert "new Set(m.members||[])" in DASHBOARD_HTML
    assert "members.forEach(machine=>" in DASHBOARD_HTML
    assert 'class="topology-edge ${cls}"' in DASHBOARD_HTML
    assert 'class="topology-matrix"' in DASHBOARD_HTML
    assert "Each machine runs one active mesh in this version" in DASHBOARD_HTML
    assert 'meshes[k].status||"")!=="stopped"' in DASHBOARD_HTML
    assert "view.dataset.signature===signature" in DASHBOARD_HTML


def test_latency_map_separates_workers_on_the_same_rtt_ring() -> None:
    assert "function separateMapAngles(ids,ang,radial,marker)" in DASHBOARD_HTML
    assert "const minDistance=marker[a]+marker[b]+10" in DASHBOARD_HTML
    assert "separateMapAngles(ids,ang,radial,marker)" in DASHBOARD_HTML
    assert "r=radial[id]" in DASHBOARD_HTML
    assert "function mapNodeLabel(id)" in DASHBOARD_HTML
    assert "function relaxMapLabels(labels,width,height,cx,cy)" in DASHBOARD_HTML
    assert "const overlapX=(a.width+b.width)/2+pad" in DASHBOARD_HTML
    assert "ctx.lineTo(label.x,label.y-4)" in DASHBOARD_HTML


def test_validator_dashboard_only_offers_chain_bound_launch_models() -> None:
    assert 'if(servingMode!=="subnet") Object.values(workers)' in DASHBOARD_HTML
    assert "r.launch_ready!==false" in DASHBOARD_HTML
    assert 's.serving_mode||""' in DASHBOARD_HTML


def test_validator_machine_onboarding_exposes_coordinator_requirements() -> None:
    assert 'id="coordinatorCapable"' in DASHBOARD_HTML
    assert 'id="coordinatorWallet"' in DASHBOARD_HTML
    assert 'id="coordinatorHotkey"' in DASHBOARD_HTML
    assert 'id="validatorAllowlistPath"' in DASHBOARD_HTML
    assert "--wallet-name" in DASHBOARD_HTML
    assert "--wallet-hotkey" in DASHBOARD_HTML
    assert "--validator-allowlist-path" in DASHBOARD_HTML
    assert "no coordinator-ready machine" in DASHBOARD_HTML
    assert 'last.serving_mode!=="subnet"||(cap.subnet_driver_ready??cap.validator_driver_ready)' in (
        DASHBOARD_HTML
    )
    assert '" · next e"+staged' in DASHBOARD_HTML
