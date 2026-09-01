"""llama slot-state save/restore: bounded hard-audit probes.

The save hook persists the committed slot's KV after validator-lane serves
(the postcommit-audit universe); the probe window restores it instead of
re-prefilling an evicted context EAGER. Everything here is miner-local
acceleration and must FAIL OPEN: any miss, mismatch, busy backend, or HTTP
error falls through to today's prompt-cache path, and a wrong restore only
costs its speedup (llama re-evals from the divergence point; the proof
math never trusts restored state).
"""

from __future__ import annotations

import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from verallm.mesh import CapabilityAd
from verallm.mesh.worker import (
    completion_token_ids_hash,
    make_worker_server,
    prompt_token_ids_hash,
)

PROMPT_IDS = [1, 2, 3]
COMPLETION_IDS = [4, 5]
EXPECTED_TOKENS = len(PROMPT_IDS) + len(COMPLETION_IDS)


class _FakeLlama(BaseHTTPRequestHandler):
    """Configurable /slots endpoint double."""

    n_saved = EXPECTED_TOKENS
    n_restored = EXPECTED_TOKENS
    slots: list[dict] = []
    calls: list[str] = []

    def log_message(self, *_args) -> None:  # noqa: N802
        pass

    def do_GET(self):  # noqa: N802
        cls = type(self)
        cls.calls.append(f"GET {self.path}")
        if self.path.rstrip("/") == "/slots":
            body = json.dumps(cls.slots).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self):  # noqa: N802
        cls = type(self)
        cls.calls.append(f"POST {self.path}")
        length = int(self.headers.get("Content-Length", 0) or 0)
        self.rfile.read(length)
        if "action=save" in self.path:
            payload = {
                "id_slot": 0,
                "filename": "x",
                "n_saved": cls.n_saved,
                "n_written": 4096,
                "timings": {"save_ms": 1.0},
            }
        elif "action=restore" in self.path:
            payload = {
                "id_slot": 0,
                "filename": "x",
                "n_restored": cls.n_restored,
                "n_read": 4096,
                "timings": {"restore_ms": 1.0},
            }
        else:
            payload = {}
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _fake_llama():
    handler = type(
        "FakeLlama",
        (_FakeLlama,),
        {
            "n_saved": EXPECTED_TOKENS,
            "n_restored": EXPECTED_TOKENS,
            "slots": [],
            "calls": [],
        },
    )
    backend = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=backend.serve_forever, daemon=True)
    thread.start()
    host, port = backend.server_address
    return backend, thread, handler, f"http://{host}:{port}"


def _make_server(tmp_path, backend_url):
    return make_worker_server(
        capability=CapabilityAd(
            uid=1,
            hotkey="5Coord",
            endpoint="http://127.0.0.1:9338",
            supported_backends=["gguf_stage"],
            cached_model_package_hashes=[],
        ),
        host="127.0.0.1",
        port=0,
        backend_url=backend_url,
        require_proof=True,
        proof_trace_enable_file=str(tmp_path / "enable"),
        proof_trace_dir=str(tmp_path / "traces"),
        proof_sample_bps=10000,
        proof_ops_per_request=1,
        proof_trace_candidates_per_request=8,
        decode_audit_bps=1000,
        llama_n_parallel=4,
        llama_n_ubatch=512,
        proof_trace_manifest_format="compact-raw-v3",
        slot_state_dir=str(tmp_path / "slot-states"),
    )


def _await(predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


def _state_paths(tmp_path, request_id: str):
    import hashlib

    stem = hashlib.sha256(request_id.encode()).hexdigest()[:40]
    root = tmp_path / "slot-states"
    return root / f"slot-state-{stem}.vseq", root / f"slot-state-{stem}.json"


def _receipt_context(request_id: str) -> dict:
    return {
        "request_id": request_id,
        "prompt_token_ids_hash": prompt_token_ids_hash(PROMPT_IDS),
        "completion_token_ids_hash": completion_token_ids_hash(COMPLETION_IDS),
    }


def _shutdown(server, backend, thread) -> None:
    server.server_close()
    backend.shutdown()
    backend.server_close()
    thread.join(timeout=2)


def test_validator_lane_save_writes_sidecar(tmp_path):
    backend, thread, handler, url = _fake_llama()
    server = _make_server(tmp_path, url)
    try:
        server.verathos_save_slot_state(
            request_id="req-a",
            prompt_token_ids=PROMPT_IDS,
            completion_token_ids=COMPLETION_IDS,
            slot_id=0,
            validator_authenticated=True,
        )
        _state, sidecar = _state_paths(tmp_path, "req-a")
        assert _await(sidecar.exists)
        meta = json.loads(sidecar.read_text())
        assert meta["n_saved"] == EXPECTED_TOKENS
        assert meta["prompt_token_ids_hash"] == prompt_token_ids_hash(PROMPT_IDS)
        assert any("action=save" in call for call in handler.calls)
    finally:
        _shutdown(server, backend, thread)


def test_save_accepts_the_final_token_short_cache(tmp_path):
    # llama's slot cache ends ONE token short of prompt+completion (the
    # final sampled token is never fed back as decode input), so n_saved
    # == expected-1 is the systematic healthy case, not a stale save.
    backend, thread, handler, url = _fake_llama()
    handler.n_saved = EXPECTED_TOKENS - 1
    server = _make_server(tmp_path, url)
    try:
        server.verathos_save_slot_state(
            request_id="req-short-cache",
            prompt_token_ids=PROMPT_IDS,
            completion_token_ids=COMPLETION_IDS,
            slot_id=0,
            validator_authenticated=True,
        )
        _state, sidecar = _state_paths(tmp_path, "req-short-cache")
        assert _await(sidecar.exists)
        assert json.loads(sidecar.read_text())["n_saved"] == EXPECTED_TOKENS - 1
    finally:
        _shutdown(server, backend, thread)


def test_organic_save_is_off_by_default(tmp_path):
    backend, thread, handler, url = _fake_llama()
    server = _make_server(tmp_path, url)
    try:
        server.verathos_save_slot_state(
            request_id="req-b",
            prompt_token_ids=PROMPT_IDS,
            completion_token_ids=COMPLETION_IDS,
            slot_id=0,
            validator_authenticated=False,
        )
        time.sleep(0.3)
        assert not handler.calls
    finally:
        _shutdown(server, backend, thread)


def test_organic_save_opt_in_respects_min_tokens(tmp_path, monkeypatch):
    monkeypatch.setenv("VERATHOS_MESH_SLOT_STATE_ORGANIC", "1")
    monkeypatch.setenv("VERATHOS_MESH_SLOT_STATE_MIN_TOKENS", "4")
    backend, thread, handler, url = _fake_llama()
    server = _make_server(tmp_path, url)
    try:
        server.verathos_save_slot_state(
            request_id="req-short",
            prompt_token_ids=[1],
            completion_token_ids=[2],
            slot_id=0,
            validator_authenticated=False,
        )
        time.sleep(0.2)
        assert not handler.calls
        server.verathos_save_slot_state(
            request_id="req-long",
            prompt_token_ids=PROMPT_IDS,
            completion_token_ids=COMPLETION_IDS,
            slot_id=0,
            validator_authenticated=False,
        )
        assert _await(lambda: any("action=save" in c for c in handler.calls))
    finally:
        _shutdown(server, backend, thread)


def test_save_kill_switch(tmp_path, monkeypatch):
    monkeypatch.setenv("VERATHOS_MESH_SLOT_STATE_SAVE", "0")
    backend, thread, handler, url = _fake_llama()
    server = _make_server(tmp_path, url)
    try:
        server.verathos_save_slot_state(
            request_id="req-c",
            prompt_token_ids=PROMPT_IDS,
            completion_token_ids=COMPLETION_IDS,
            slot_id=0,
            validator_authenticated=True,
        )
        time.sleep(0.3)
        assert not handler.calls
    finally:
        _shutdown(server, backend, thread)


def test_stale_save_is_deleted_not_sidecarred(tmp_path):
    # The slot picked up another request before the deferred save ran:
    # llama reports a token count that cannot be this conversation.
    backend, thread, handler, url = _fake_llama()
    handler.n_saved = 999
    server = _make_server(tmp_path, url)
    try:
        state, sidecar = _state_paths(tmp_path, "req-d")
        state.parent.mkdir(parents=True, exist_ok=True)
        state.write_bytes(b"stale")
        server.verathos_save_slot_state(
            request_id="req-d",
            prompt_token_ids=PROMPT_IDS,
            completion_token_ids=COMPLETION_IDS,
            slot_id=0,
            validator_authenticated=True,
        )
        assert _await(lambda: not state.exists())
        time.sleep(0.1)
        assert not sidecar.exists()
    finally:
        _shutdown(server, backend, thread)


def test_sweep_evicts_by_ttl_and_size(tmp_path, monkeypatch):
    monkeypatch.setenv("VERATHOS_MESH_SLOT_STATE_MAX_BYTES", "8")
    backend, thread, _handler, url = _fake_llama()
    server = _make_server(tmp_path, url)
    try:
        root = tmp_path / "slot-states"
        old = root / "slot-state-old.vseq"
        old.write_bytes(b"x" * 4)
        ancient = time.time() - 10 * 3600
        os.utime(old, (ancient, ancient))
        oldest = root / "slot-state-a.vseq"
        oldest.write_bytes(b"x" * 6)
        newest = root / "slot-state-b.vseq"
        newest.write_bytes(b"x" * 6)
        past = time.time() - 60
        os.utime(oldest, (past, past))
        server.verathos_sweep_slot_states()
        assert not old.exists()  # TTL
        assert not oldest.exists()  # size cap, oldest first
        assert newest.exists()
    finally:
        _shutdown(server, backend, thread)


def test_restore_hits_and_pins_the_restored_slot(tmp_path):
    backend, thread, handler, url = _fake_llama()
    handler.slots = [
        {"id": 0, "is_processing": False, "n_prompt_tokens": 1},
        {"id": 1, "is_processing": True},
    ]
    server = _make_server(tmp_path, url)
    try:
        # A real save first, so file + sidecar exist and match.
        server.verathos_save_slot_state(
            request_id="req-e",
            prompt_token_ids=PROMPT_IDS,
            completion_token_ids=COMPLETION_IDS,
            slot_id=0,
            validator_authenticated=True,
        )
        state, sidecar = _state_paths(tmp_path, "req-e")
        state.parent.mkdir(parents=True, exist_ok=True)
        state.write_bytes(b"state")
        assert _await(sidecar.exists)
        timing: dict = {}
        slot = server.verathos_restore_slot_state(
            _receipt_context("req-e"),
            probe_slot=0,
            probe_cache=True,
            timing=timing,
        )
        assert slot == 0
        assert timing["restore"] == "hit"
        assert any("action=restore" in call for call in handler.calls)
    finally:
        _shutdown(server, backend, thread)


def test_restore_hot_slot_skips_the_io(tmp_path):
    backend, thread, handler, url = _fake_llama()
    handler.slots = [
        {
            "id": 0,
            "is_processing": False,
            "n_prompt_tokens": EXPECTED_TOKENS,
        }
    ]
    server = _make_server(tmp_path, url)
    try:
        server.verathos_save_slot_state(
            request_id="req-f",
            prompt_token_ids=PROMPT_IDS,
            completion_token_ids=COMPLETION_IDS,
            slot_id=0,
            validator_authenticated=True,
        )
        state, sidecar = _state_paths(tmp_path, "req-f")
        state.parent.mkdir(parents=True, exist_ok=True)
        state.write_bytes(b"state")
        assert _await(sidecar.exists)
        timing: dict = {}
        slot = server.verathos_restore_slot_state(
            _receipt_context("req-f"),
            probe_slot=0,
            probe_cache=True,
            timing=timing,
        )
        assert slot == 0
        assert timing["restore"] == "hot"
        assert not any("action=restore" in call for call in handler.calls)
    finally:
        _shutdown(server, backend, thread)


def test_restore_falls_to_an_idle_slot(tmp_path):
    backend, thread, handler, url = _fake_llama()
    handler.slots = [
        {"id": 0, "is_processing": True},
        {"id": 2, "is_processing": False, "n_prompt_tokens": 3},
    ]
    server = _make_server(tmp_path, url)
    try:
        server.verathos_save_slot_state(
            request_id="req-g",
            prompt_token_ids=PROMPT_IDS,
            completion_token_ids=COMPLETION_IDS,
            slot_id=0,
            validator_authenticated=True,
        )
        state, sidecar = _state_paths(tmp_path, "req-g")
        state.parent.mkdir(parents=True, exist_ok=True)
        state.write_bytes(b"state")
        assert _await(sidecar.exists)
        timing: dict = {}
        slot = server.verathos_restore_slot_state(
            _receipt_context("req-g"),
            probe_slot=0,
            probe_cache=True,
            timing=timing,
        )
        assert slot == 2
        assert timing["restore"] == "hit"
    finally:
        _shutdown(server, backend, thread)


def test_restore_fail_open_matrix(tmp_path):
    backend, thread, handler, url = _fake_llama()
    handler.slots = [{"id": 0, "is_processing": False, "n_prompt_tokens": 1}]
    server = _make_server(tmp_path, url)
    try:
        # 1) No saved state at all -> miss.
        timing: dict = {}
        assert (
            server.verathos_restore_slot_state(
                _receipt_context("req-none"),
                probe_slot=0,
                probe_cache=True,
                timing=timing,
            )
            == 0
        )
        assert timing["restore"] == "miss"

        # 2) Sidecar hash mismatch -> mismatch, no restore call.
        server.verathos_save_slot_state(
            request_id="req-h",
            prompt_token_ids=PROMPT_IDS,
            completion_token_ids=COMPLETION_IDS,
            slot_id=0,
            validator_authenticated=True,
        )
        state, sidecar = _state_paths(tmp_path, "req-h")
        state.parent.mkdir(parents=True, exist_ok=True)
        state.write_bytes(b"state")
        assert _await(sidecar.exists)
        wrong = _receipt_context("req-h")
        wrong["prompt_token_ids_hash"] = "f" * 64
        timing = {}
        assert (
            server.verathos_restore_slot_state(
                wrong, probe_slot=0, probe_cache=True, timing=timing
            )
            == 0
        )
        assert timing["restore"] == "mismatch"

        # 3) Every slot busy -> busy, no restore call.
        handler.slots = [{"id": 0, "is_processing": True}]
        timing = {}
        assert (
            server.verathos_restore_slot_state(
                _receipt_context("req-h"),
                probe_slot=0,
                probe_cache=True,
                timing=timing,
            )
            == 0
        )
        assert timing["restore"] == "busy"

        # 4) Backend refuses the restore (no KV space) -> error.
        handler.slots = [
            {"id": 0, "is_processing": False, "n_prompt_tokens": 1}
        ]
        handler.n_restored = 0
        timing = {}
        assert (
            server.verathos_restore_slot_state(
                _receipt_context("req-h"),
                probe_slot=0,
                probe_cache=True,
                timing=timing,
            )
            == 0
        )
        assert timing["restore"] == "error"

        # 5) Probe prompt cache disabled -> restore is pointless -> off.
        timing = {}
        assert (
            server.verathos_restore_slot_state(
                _receipt_context("req-h"),
                probe_slot=0,
                probe_cache=False,
                timing=timing,
            )
            == 0
        )
        assert timing["restore"] == "off"
    finally:
        _shutdown(server, backend, thread)
