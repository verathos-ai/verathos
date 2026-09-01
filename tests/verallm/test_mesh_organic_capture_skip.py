"""Organic serves skip the structurally witness-free shared capture join.

At --parallel > 1 under the v3 slot-view profile the serve-time capture
channel produces nothing an audit can use (C-side dump budget is zeroed,
tail ring off, light leaves synthesized), yet organic serves used to hold
the shared capture window for their whole stream - which is exactly what
a hard audit's exclusive window must drain, blocking organic chat for the
audit's full duration (20+ min live on glm-iq2). These tests pin the skip
decision's truth table: the skip engages ONLY under the structural-zero
profile with a warm slot-view template, never for validator-lane serves,
never at parallel==1, and the kill switch restores the join.
"""

from __future__ import annotations

import inspect
import time

from verallm.mesh import CapabilityAd
from verallm.mesh.worker import make_worker_server


def _capability() -> CapabilityAd:
    return CapabilityAd(
        uid=1,
        hotkey="5Coord",
        endpoint="http://127.0.0.1:9338",
        supported_backends=["gguf_stage"],
        cached_model_package_hashes=[],
    )


def _make_server(
    tmp_path,
    *,
    n_parallel: int = 4,
    manifest_format: str = "compact-raw-v3",
    candidates: int = 8,
    decode_audit_bps: int = 1000,
):
    return make_worker_server(
        capability=_capability(),
        host="127.0.0.1",
        port=0,
        backend_url="http://127.0.0.1:9",
        require_proof=True,
        proof_trace_enable_file=str(tmp_path / "enable"),
        proof_trace_dir=str(tmp_path / "traces"),
        proof_sample_bps=10000,
        proof_ops_per_request=1,
        proof_trace_candidates_per_request=candidates,
        decode_audit_bps=decode_audit_bps,
        llama_n_parallel=n_parallel,
        llama_n_ubatch=512,
        proof_trace_manifest_format=manifest_format,
    )


def _seed_template(server, tmp_path) -> None:
    key = (str(tmp_path / "traces"), "", "", 0, 0, 0)
    server.verathos_slot_view_template_cache[key] = [{"op": 1}]


_POLICY = {"proof_capture_required": True, "decode_audit_configured": True}


def test_skip_engages_under_structural_zero_profile(tmp_path):
    server = _make_server(tmp_path)
    try:
        _seed_template(server, tmp_path)
        assert server.verathos_organic_capture_skip(
            _POLICY, None, validator_authenticated=False
        )
    finally:
        server.server_close()


def test_server_close_stops_proof_trace_janitor(tmp_path):
    server = _make_server(tmp_path)
    janitor = server.verathos_proof_trace_janitor_thread
    assert janitor is not None
    assert janitor.is_alive()

    server.server_close()

    deadline = time.monotonic() + 1.0
    while janitor.is_alive() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert server.verathos_proof_trace_janitor_stop.is_set()
    assert not janitor.is_alive()


def test_validator_lane_skips_under_structural_zero(tmp_path):
    # A validator-lane serve's shared join stores nothing at parallel>1
    # either. Same structural-zero conditions, same skip.
    server = _make_server(tmp_path)
    try:
        _seed_template(server, tmp_path)
        assert server.verathos_organic_capture_skip(
            _POLICY, None, validator_authenticated=True
        )
    finally:
        server.server_close()


def test_validator_lane_keeps_the_join_outside_structural_zero(tmp_path):
    # parallel==1: the lane guard stays absolute.
    server = _make_server(tmp_path, n_parallel=1)
    try:
        _seed_template(server, tmp_path)
        assert not server.verathos_organic_capture_skip(
            _POLICY, None, validator_authenticated=True
        )
    finally:
        server.server_close()


def test_parallel_one_is_unchanged(tmp_path):
    server = _make_server(tmp_path, n_parallel=1)
    try:
        _seed_template(server, tmp_path)
        assert not server.verathos_organic_capture_skip(
            _POLICY, None, validator_authenticated=False
        )
    finally:
        server.server_close()


def test_legacy_manifest_format_keeps_the_join(tmp_path):
    server = _make_server(tmp_path, manifest_format="compact-raw-v2")
    try:
        _seed_template(server, tmp_path)
        assert not server.verathos_organic_capture_skip(
            _POLICY, None, validator_authenticated=False
        )
    finally:
        server.server_close()


def test_cold_template_cache_keeps_the_join(tmp_path):
    # Until the slot-view template is cached this serve may still be the
    # one that captures it; the predicate's template gate must hold.
    server = _make_server(tmp_path)
    try:
        assert not server.verathos_organic_capture_skip(
            _POLICY, None, validator_authenticated=False
        )
    finally:
        server.server_close()


def test_kill_switch_restores_the_join(tmp_path, monkeypatch):
    monkeypatch.setenv("VERATHOS_MESH_ORGANIC_CAPTURE_SKIP", "0")
    server = _make_server(tmp_path)
    try:
        _seed_template(server, tmp_path)
        assert not server.verathos_organic_capture_skip(
            _POLICY, None, validator_authenticated=False
        )
    finally:
        server.server_close()


def test_capture_off_policy_never_skips(tmp_path):
    # skip=True is only meaningful when capture WOULD have been armed.
    server = _make_server(tmp_path)
    try:
        _seed_template(server, tmp_path)
        assert not server.verathos_organic_capture_skip(
            {"proof_capture_required": False},
            None,
            validator_authenticated=False,
        )
    finally:
        server.server_close()


def test_candidate_capture_still_blocks_outside_structural_zero(tmp_path):
    # Candidate-capture pools without decode audits at parallel==1 keep
    # the historical behavior: candidates are real there, so no skip.
    server = _make_server(tmp_path, n_parallel=1, decode_audit_bps=0)
    try:
        _seed_template(server, tmp_path)
        assert not server.verathos_organic_capture_skip(
            {"proof_capture_required": True, "decode_audit_configured": False},
            None,
            validator_authenticated=False,
        )
    finally:
        server.server_close()


def test_streaming_forward_accepts_the_skip_flag():
    # The streaming forward must expose the same skip plumb as the
    # non-streaming one (route_to_mesh_stream passes it).
    from verallm.mesh import worker as worker_module

    source = inspect.getsource(worker_module)
    assert "def forward_stream_to_backend(" in source
    streaming_def = source.split("def forward_stream_to_backend(", 1)[1]
    signature = streaming_def.split(") -> dict[str, Any]:", 1)[0]
    assert "skip_trace_capture: bool = False" in signature
