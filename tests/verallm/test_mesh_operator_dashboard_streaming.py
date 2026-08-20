"""Streaming contract for the operator dashboard actually served by the pool."""

from verallm.mesh.pool_dashboard import DASHBOARD_HTML


def _chat_script() -> str:
    start = DASHBOARD_HTML.index("let CHAT_ABORT=null;")
    end = DASHBOARD_HTML.index("// Turn a raw mesh status", start)
    return DASHBOARD_HTML[start:end]


def test_served_dashboard_respects_validator_launch_readiness() -> None:
    # The manager normalizes serving_mode to "subnet" in every payload;
    # the pre-rename "validator" literal must never come back (it left a
    # subnet pool's dashboard on the dev branches).
    assert 'if(servingMode!=="subnet") Object.values(workers)' in DASHBOARD_HTML
    assert '"validator"' not in DASHBOARD_HTML
    assert "r.launch_ready!==false" in DASHBOARD_HTML
    assert "(w.catalog||[])" in DASHBOARD_HTML


def test_operator_chat_uses_authenticated_sse_stream() -> None:
    chat = _chat_script()

    assert 'fetch("/v1/pool/chat-stream"' in chat
    assert 'api("/v1/pool/chat",' not in chat
    assert "JSON.stringify(authBody({mesh_key:key" in chat
    assert "resp.body.getReader()" in chat
    assert "new TextDecoder()" in chat
    assert r"buf.match(/\r?\n\r?\n/)" in chat


def test_operator_chat_renders_deltas_but_waits_for_final_to_verify() -> None:
    chat = _chat_script()
    delta = chat.index('if(ev.type==="delta")')
    done = chat.index('else if(ev.type==="done")')
    verified = chat.index("ev.verified===true")

    assert delta < done < verified
    assert "outDiv.textContent=content" in chat[delta:done]
    assert "ev.verified" not in chat[:done]
    assert "stream ended before the proof-bearing final event" in chat


def test_operator_chat_requires_exact_expected_stage_coverage_for_verified_badge() -> None:
    chat = _chat_script()

    assert (
        "const proofStages=Number(ev.proof_stages), "
        "expectedStages=Number(ev.expected_stage_count);"
    ) in chat
    assert "const fullyVerified=ev.verified===true&&ev.receipt_verified===true" in chat
    assert "Number.isInteger(proofStages)&&Number.isInteger(expectedStages)" in chat
    assert "expectedStages>0&&proofStages===expectedStages;" in chat
    assert "Number(ev.proof_stages)>0" not in chat
    assert "fullyVerified?`<span class=\"verbadge\">" in chat
    assert (
        "coordinator verified · ${proofStages} stages · "
        "${Number(ev.receipts)||0} proofs"
    ) in chat
    assert chat.index("const ver=ev.error?") < chat.index(": fullyVerified?")


def test_operator_chat_prevents_overlap_and_cleans_up_abort() -> None:
    chat = _chat_script()

    assert "if(CHAT_SENDING) return" in chat
    assert "const controller=new AbortController()" in chat
    assert "signal:controller.signal" in chat
    assert "if(CHAT_ABORT) CHAT_ABORT.abort()" in chat
    assert "CHAT_ABORT=null; CHAT_SENDING=false; inp.disabled=false" in chat
    assert '"request cancelled"' in chat


def test_operator_chat_rejects_malformed_and_unterminated_sse() -> None:
    chat = _chat_script()

    assert "mesh returned an invalid stream event" in chat
    assert r"buf.match(/\r?\n\r?\n/)" in chat
    assert "if(!sawTerminal&&buf.trim())" in chat
    assert "if(!sawTerminal&&!controller.signal.aborted)" in chat


def test_live_mesh_refresh_preserves_the_open_chat_dom_and_focus() -> None:
    start = DASHBOARD_HTML.index("function renderMeshesPreservingChat(meshes)")
    end = DASHBOARD_HTML.index("function renderStats", start)
    refresh_renderer = DASHBOARD_HTML[start:end]

    assert 'el.querySelectorAll("tr.mesh-chat-row")' in refresh_renderer
    assert "chatRow.remove()" in refresh_renderer
    assert "renderMeshes(meshes)" in refresh_renderer
    assert "replacement.replaceWith(chatRow)" in refresh_renderer
    assert "focused.focus()" in refresh_renderer
    assert "focused.setSelectionRange" in refresh_renderer
    assert "if(!CHAT_SENDING)" in refresh_renderer
    assert (
        "if(openChat) renderMeshesPreservingChat(s.meshes||{}); "
        "else renderMeshes(s.meshes||{});"
    ) in DASHBOARD_HTML
    assert "if(!openChat) renderMeshes(s.meshes||{});" not in DASHBOARD_HTML
