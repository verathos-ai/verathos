"""Per-mesh validator score contract for the served operator dashboard."""

from verallm.mesh.pool_dashboard import DASHBOARD_HTML


def test_dashboard_labels_mesh_score_as_validator_ema() -> None:
    assert "function validatorScoreBadge(m)" in DASHBOARD_HTML
    assert "Current coordinator/model slot EMA" in DASHBOARD_HTML
    assert "latest completed sample epoch" in DASHBOARD_HTML
    assert 'class="vsample">sample e' in DASHBOARD_HTML
    assert "<th>Validator score</th>" in DASHBOARD_HTML
    assert "${validatorScoreBadge(m)}" in DASHBOARD_HTML


def test_dashboard_renders_unavailable_mapping_as_not_scored() -> None:
    start = DASHBOARD_HTML.index("function validatorScoreBadge(m)")
    end = DASHBOARD_HTML.index("function renderMeshes", start)
    score_renderer = DASHBOARD_HTML[start:end]

    assert "if(!s.available)" in score_renderer
    assert "not scored" in score_renderer
    assert '"owner only"' in score_renderer
    assert "s.reason" in score_renderer
    assert "Number.isFinite(score)" in score_renderer


def test_dashboard_surfaces_score_status_without_replacing_mesh_health() -> None:
    assert 'class="vscore warning"' not in DASHBOARD_HTML  # assigned dynamically
    assert 'const cls=s.blacklisted?" blocked":(s.probation||s.stale)?" warning":""' in DASHBOARD_HTML
    assert '"slot EMA · carried"' in DASHBOARD_HTML
    assert '"slot EMA · adjusted"' in DASHBOARD_HTML
    assert "latest sample provenance is shown separately" in DASHBOARD_HTML
    assert "validator state is stale" in DASHBOARD_HTML
    assert "validatorScoreBadge(m)" in DASHBOARD_HTML
    assert "meshPhase(m,workers)" in DASHBOARD_HTML


def test_dashboard_exposes_exact_completed_score_provenance_without_clutter() -> None:
    start = DASHBOARD_HTML.index("function validatorScoreBadge(m)")
    end = DASHBOARD_HTML.index("function renderMeshes", start)
    score_renderer = DASHBOARD_HTML[start:end]

    assert "latest.chain_id" in score_renderer
    assert "latest.netuid" in score_renderer
    assert "latest.scored_verification_snapshot_hash" in score_renderer
    assert "latest.scored_snapshot_generation" in score_renderer
    assert 'class="vprov" tabindex="0"' in score_renderer
    assert 'class="vprov-pop"' in score_renderer
    assert "Score provenance:" in score_renderer


def test_dashboard_scrubs_cached_owner_scores_when_session_is_lost() -> None:
    start = DASHBOARD_HTML.index("function redactValidatorScores(meshes)")
    end = DASHBOARD_HTML.index("async function api", start)
    invalidation = DASHBOARD_HTML[start:end]

    assert "validator_score:{" in invalidation
    assert "owner access is required to view validator scores" in invalidation
    assert "last=Object.assign" in invalidation
    assert "renderOverview(workers,meshes)" in invalidation
    assert "renderMeshes(meshes)" in invalidation
    assert "if(CHAT_ABORT) CHAT_ABORT.abort()" in invalidation
    assert "openChat=null" in invalidation
    assert "if(r.status===403 && SESSION&&SESSION.is_owner) invalidateSession()" in DASHBOARD_HTML
    assert "if(resp.status===403&&SESSION&&SESSION.is_owner) invalidateSession()" in DASHBOARD_HTML
    assert "const token=SESSION&&SESSION.token; invalidateSession()" in DASHBOARD_HTML
    assert "const s=canManage()?response:Object.assign" in DASHBOARD_HTML
    assert "meshes:redactValidatorScores(response.meshes||{})" in DASHBOARD_HTML
