"""Serve-side admission wedge hardening.

A ~5-minute full-context canary prefill starved interactive clients away;
the serve's handler threads had NO socket timeout, so threads blocked
forever in client-socket reads/sends (faulthandler: sendall inside
_send_json / _write_sse, readinto inside handle_one_request), and each
admitted one held its KVAdmissionLedger reservation permanently. The pool
shrank to an unconditional slots_busy 503 with idle GPUs (~75 threads,
zero established connections) until relaunch. Three layers under test:

1. per-connection socket timeouts bound every individual client-socket
   operation, terminating wedged handler threads without touching SSE
   streams that keep writing successfully,
2. an idempotent per-request ReservationGuard releases the admission on
   EVERY exit path exactly once,
3. the serve's service_actions sweep reconciles reservations whose
   owning thread died without releasing - the backstop that turns a
   future release-path bug from fatal into a loud ERROR log.
"""

from __future__ import annotations

import json
import logging
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.request import Request, urlopen

from verallm.mesh import CapabilityAd, MeshMember, MeshSpec, StageRange
from verallm.mesh.admission import Admission
from verallm.mesh.worker import post_json, serve_worker_in_thread


DIGEST_A = "a" * 64


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _shutdown_server(server, thread) -> None:
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)


def _worker_mesh(worker_endpoint: str) -> MeshSpec:
    return MeshSpec(
        mesh_id="mesh-admission-test",
        mode="private",
        coordinator_uid=1,
        coordinator_hotkey="5Coord",
        model_id="model",
        model_package_hash=DIGEST_A,
        total_layers=4,
        members=[
            MeshMember(
                uid=1,
                hotkey="5Coord",
                endpoint="http://coord.local:9338",
                stage_index=0,
                layers=StageRange(0, 2),
                role="coordinator",
                backend="gguf_stage",
                payout_bps=10000,
            ),
            MeshMember(
                uid=1,
                hotkey="5Coord",
                endpoint=worker_endpoint,
                rpc_endpoint="worker.local:50052",
                rpc_split_weight=1,
                proof_endpoint=worker_endpoint,
                stage_index=1,
                layers=StageRange(2, 4),
                role="worker",
                backend="gguf_stage_worker",
                payout_bps=0,
            ),
        ],
    )


def _worker_capability(worker_endpoint: str) -> CapabilityAd:
    return CapabilityAd(
        uid=1,
        hotkey="5Coord",
        endpoint=worker_endpoint,
        supported_backends=["gguf_stage_worker"],
        cached_model_package_hashes=[DIGEST_A],
    )


def _slow_sse_backend(
    *,
    chunk_count: int,
    chunk_interval_s: float,
):
    """Fake llama-server whose stream spans several handler timeouts.

    Each chunk is written ``chunk_interval_s`` apart so the WHOLE stream
    takes chunk_count * chunk_interval_s - far longer than the handler
    socket timeout under test - while every individual client write
    still completes instantly for a client that reads.
    """

    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, status_code: int, payload: dict) -> None:
            body = json.dumps(payload, sort_keys=True).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
            path = self.path.rstrip("/")
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length else b"{}"
            payload = json.loads(raw.decode("utf-8"))
            if path == "/apply-template":
                self._send_json(
                    200,
                    {
                        "prompt": json.dumps(
                            payload.get("messages", []), sort_keys=True
                        )
                    },
                )
                return
            if path == "/tokenize":
                self._send_json(200, {"tokens": [201, 202, 203]})
                return
            if path != "/v1/chat/completions":
                self._send_json(404, {"error": "not found"})
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            for index in range(chunk_count):
                final = index == chunk_count - 1
                chunk = {
                    "id": "chatcmpl-slow",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": payload.get("model", "mesh-test"),
                    "choices": [
                        {
                            "index": 0,
                            "delta": (
                                {}
                                if final
                                else {
                                    "role": "assistant",
                                    "content": f"tok{index} ",
                                }
                            ),
                            "finish_reason": "stop" if final else None,
                        }
                    ],
                }
                if final:
                    chunk["usage"] = {
                        "prompt_tokens": 1,
                        "completion_tokens": chunk_count - 1,
                        "total_tokens": chunk_count,
                    }
                self.wfile.write(
                    f"data: {json.dumps(chunk, sort_keys=True)}\n\n".encode()
                )
                self.wfile.flush()
                time.sleep(chunk_interval_s)
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

        def log_message(self, fmt: str, *args) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    return server, thread, f"http://{host}:{port}"


def _drain_sse(url: str, payload: dict, *, timeout: float = 30.0) -> list[str]:
    req = Request(
        url,
        data=json.dumps(payload, sort_keys=True).encode("utf-8"),
        headers={
            "Accept": "text/event-stream",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    data_lines: list[str] = []
    with urlopen(req, timeout=timeout) as resp:
        while True:
            raw = resp.readline()
            if not raw:
                break
            line = raw.decode("utf-8").rstrip("\r\n")
            if line.startswith("data:"):
                data_lines.append(line[len("data:"):].strip())
    return data_lines


def _await_ledger_drained(ledger, *, timeout: float = 10.0) -> dict:
    deadline = time.monotonic() + timeout
    snap = ledger.snapshot()
    while time.monotonic() < deadline:
        snap = ledger.snapshot()
        if snap["in_flight_slots"] == 0 and snap["in_flight_tokens"] == 0:
            return snap
        time.sleep(0.05)
    return snap


def test_idle_half_open_connection_times_out_and_frees_the_thread():
    """The wedge class from the incident: a client that connects (or
    vanishes half-open) and never completes a request must be discarded
    within the socket timeout, terminating its handler thread, instead
    of parking in rfile reads forever."""

    server, thread = serve_worker_in_thread(
        capability=_worker_capability("http://worker.local:9338"),
    )
    server.RequestHandlerClass.timeout = 0.5
    host, port = server.server_address
    try:
        baseline = threading.active_count()
        silent = socket.create_connection((host, port), timeout=10.0)
        try:
            started = time.monotonic()
            # Blocks until the server discards the connection.
            leftover = silent.recv(1)
            elapsed = time.monotonic() - started
            assert leftover == b""
            assert elapsed < 5.0
        finally:
            silent.close()
        # The handler thread exits with the discarded connection.
        deadline = time.monotonic() + 5.0
        while (
            threading.active_count() > baseline
            and time.monotonic() < deadline
        ):
            time.sleep(0.05)
        assert threading.active_count() <= baseline
        # And the server keeps serving new connections normally.
        with urlopen(f"http://{host}:{port}/health", timeout=5.0) as resp:
            assert json.loads(resp.read())["status"] == "ok"
    finally:
        _shutdown_server(server, thread)


def test_stalled_request_body_times_out():
    """A client that sends headers but never the promised body wedges the
    handler in the body read; the socket timeout must discard it."""

    server, thread = serve_worker_in_thread(
        capability=_worker_capability("http://worker.local:9338"),
    )
    server.RequestHandlerClass.timeout = 0.5
    host, port = server.server_address
    try:
        stalled = socket.create_connection((host, port), timeout=10.0)
        try:
            stalled.sendall(
                b"POST /v1/chat/completions HTTP/1.1\r\n"
                b"Host: test\r\n"
                b"Content-Type: application/json\r\n"
                b"Content-Length: 4096\r\n"
                b"\r\n"
                b"{\"partial\":"  # never completes
            )
            started = time.monotonic()
            # Server may write nothing or a partial error; it must CLOSE.
            while True:
                data = stalled.recv(4096)
                if data == b"":
                    break
            assert time.monotonic() - started < 5.0
        finally:
            stalled.close()
    finally:
        _shutdown_server(server, thread)


def test_sse_long_write_outlives_the_socket_timeout():
    """An SSE stream that keeps writing successfully must NOT be killed
    by the idle timeout: the bound is per socket operation, and a client
    that reads drains every write immediately. Total stream duration
    here is several times the handler timeout."""

    chunk_interval = 0.25
    chunk_count = 6
    backend, backend_thread, backend_url = _slow_sse_backend(
        chunk_count=chunk_count,
        chunk_interval_s=chunk_interval,
    )
    worker_endpoint = "http://worker.local:9338"
    server, thread = serve_worker_in_thread(
        capability=_worker_capability(worker_endpoint),
        mesh_spec=_worker_mesh(worker_endpoint),
        backend_url=backend_url,
    )
    server.RequestHandlerClass.timeout = 0.5
    host, port = server.server_address
    request = {
        "model": "model",
        "messages": [{"role": "user", "content": "stream slowly"}],
        "stream": True,
    }
    try:
        started = time.monotonic()
        data_lines = _drain_sse(
            f"http://{host}:{port}/v1/chat/completions", request
        )
        elapsed = time.monotonic() - started
        # The stream ran to completion across many timeout periods.
        assert elapsed >= (chunk_count - 1) * chunk_interval
        assert data_lines
        assert data_lines[-1] == "[DONE]"
        relayed = [line for line in data_lines if line.startswith("{")]
        assert len(relayed) >= chunk_count
        # The reservation drained with the request.
        ledger = server.verathos_admission_ledger()
        snap = _await_ledger_drained(ledger)
        assert snap["in_flight_slots"] == 0
        assert snap["in_flight_tokens"] == 0
    finally:
        _shutdown_server(server, thread)
        _shutdown_server(backend, backend_thread)


def test_dead_owner_reservation_is_reaped_and_logged(caplog):
    """Layer-3 backstop: a reservation whose owning thread died without
    releasing is reconciled by the serve's own maintenance hook
    (service_actions inside serve_forever) and logged at ERROR - the
    leak that previously shrank the pool until relaunch."""

    server, thread = serve_worker_in_thread(
        capability=_worker_capability("http://worker.local:9338"),
    )
    try:
        ledger = server.verathos_admission_ledger()

        def leaky() -> None:
            assert ledger.try_admit(1000) is Admission.ADMITTED
            # Dies WITHOUT releasing - the bug class under reconciliation.

        leaker = threading.Thread(target=leaky, name="wedged-handler")
        leaker.start()
        leaker.join(timeout=5)
        assert not leaker.is_alive()
        assert ledger.snapshot()["in_flight_slots"] == 1

        server.admission_reap_interval_s = 0.05
        server._admission_reap_due = 0.0
        with caplog.at_level(logging.ERROR, logger="verallm.mesh.worker"):
            snap = _await_ledger_drained(ledger)
            # The sweep releases the ledger before emitting its diagnostic.
            # On a busy full-suite runner the polling thread can observe the
            # release in that narrow interval, so wait for the asynchronous
            # log side effect instead of making the assertion timing-racy.
            deadline = time.monotonic() + 2.0
            while (
                "released by sweep" not in caplog.text
                and time.monotonic() < deadline
            ):
                time.sleep(0.01)
        assert snap["in_flight_slots"] == 0
        assert snap["in_flight_tokens"] == 0
        assert "released by sweep" in caplog.text
        assert "wedged-handler" in caplog.text
    finally:
        _shutdown_server(server, thread)


def test_capability_route_exposes_admission_only_behind_internal_auth():
    """Ledger diagnostics ride the existing HMAC-walled status route.
    Without the wall (dev serve) the field is absent, and an unsigned
    request against a walled serve never reaches the route: exposing
    real occupancy openly would hand validators an oracle for
    capacity-audit drain windows."""

    from urllib.error import HTTPError

    from verallm.mesh.http_auth import sign_internal_http_request

    secret = "internal-test-secret"
    walled, walled_thread = serve_worker_in_thread(
        capability=_worker_capability("http://worker.local:9338"),
        internal_auth_secret=secret,
    )
    open_server, open_thread = serve_worker_in_thread(
        capability=_worker_capability("http://worker.local:9338"),
    )
    try:
        # The ledger exists from the first admission-relevant touch; a
        # fresh serve legitimately reports no admission section yet.
        walled.verathos_admission_ledger()
        host, port = walled.server_address
        url = f"http://{host}:{port}/capability"
        headers = sign_internal_http_request(
            secret=secret,
            method="GET",
            path="/capability",
            body=b"",
        )
        with urlopen(Request(url, headers=headers), timeout=5.0) as resp:
            payload = json.loads(resp.read())
        assert "admission" in payload
        assert payload["admission"]["in_flight_slots"] == 0
        assert payload["admission"]["reservations"] == []

        try:
            with urlopen(url, timeout=5.0) as resp:
                json.loads(resp.read())
            raise AssertionError("unsigned request must be refused")
        except HTTPError as refused:
            assert refused.code in (401, 403)

        open_host, open_port = open_server.server_address
        with urlopen(
            f"http://{open_host}:{open_port}/capability", timeout=5.0
        ) as resp:
            open_payload = json.loads(resp.read())
        assert "admission" not in open_payload
    finally:
        _shutdown_server(walled, walled_thread)
        _shutdown_server(open_server, open_thread)


def test_normal_request_lifecycle_unchanged():
    """Baseline: an ordinary request is admitted, served, and released
    exactly once; the pool is immediately whole for the next request and
    the sweep finds nothing to reconcile."""

    backend, backend_thread, backend_url = _slow_sse_backend(
        chunk_count=2,
        chunk_interval_s=0.0,
    )
    worker_endpoint = "http://worker.local:9338"
    server, thread = serve_worker_in_thread(
        capability=_worker_capability(worker_endpoint),
        mesh_spec=_worker_mesh(worker_endpoint),
        backend_url=backend_url,
    )
    host, port = server.server_address
    request = {
        "model": "model",
        "messages": [{"role": "user", "content": "stream"}],
        "stream": True,
    }
    try:
        for _ in range(2):
            data_lines = _drain_sse(
                f"http://{host}:{port}/v1/chat/completions", request
            )
            assert data_lines[-1] == "[DONE]"
            ledger = server.verathos_admission_ledger()
            snap = _await_ledger_drained(ledger)
            assert snap["in_flight_slots"] == 0
            assert snap["in_flight_tokens"] == 0
            assert snap["reservations"] == []
        assert ledger.reap_dead_owners() == []
    finally:
        _shutdown_server(server, thread)
        _shutdown_server(backend, backend_thread)
