"""Out-of-band reachability checks against the registered endpoint."""
from __future__ import annotations

import json
import ssl
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import verallm.mesh.probe as probe_module
from verallm.mesh.probe import check_public_endpoint


class _Stub(BaseHTTPRequestHandler):
    health_status = 200
    chat_status = 403
    chat_body = {"error": "validator authentication is required"}

    def log_message(self, *args):
        pass

    def _reply(self, status, payload):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._reply(
                type(self).health_status,
                {"status": "ok", "service": "verathos-mesh", "version": "1"},
            )
        elif self.path == "/v1/mesh/verification-snapshot":
            # Validators GET the snapshot; the posture probe does too.
            self._reply(type(self).chat_status, type(self).chat_body)
        else:
            self._reply(404, {"error": "not found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0) or 0)
        self.rfile.read(length)
        if self.path == "/v1/chat/completions":
            self._reply(type(self).chat_status, type(self).chat_body)
        else:
            self._reply(404, {"error": "not found"})


@pytest.fixture()
def stub():
    class Handler(_Stub):
        pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield Handler, f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    thread.join(timeout=5)


@pytest.fixture()
def self_signed_stub(tmp_path):
    cert = tmp_path / "cert.pem"
    key = tmp_path / "key.pem"
    subprocess.run(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-days",
            "2",
            "-subj",
            "/CN=127.0.0.1",
            "-keyout",
            str(key),
            "-out",
            str(cert),
        ],
        check=True,
        capture_output=True,
    )

    class Handler(_Stub):
        pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(certfile=cert, keyfile=key)
    server.socket = context.wrap_socket(server.socket, server_side=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield Handler, f"https://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    thread.join(timeout=5)


def _by_name(checks, name):
    return next(check for check in checks if check.name.startswith(name))


def test_correct_posture_passes(stub):
    _handler, endpoint = stub
    checks = check_public_endpoint(endpoint, timeout=5.0)
    assert _by_name(checks, "public-health").passed
    assert _by_name(checks, "validator-auth-posture /v1/chat/completions").passed
    assert _by_name(
        checks, "validator-auth-posture /v1/mesh/verification-snapshot"
    ).passed
    # http endpoint: TLS is advisory, not blocking.
    tls = _by_name(checks, "tls-certificate")
    assert tls.kind == "advisory"
    hard = [check for check in checks if check.kind == "hard"]
    assert all(check.passed for check in hard)


def test_allowlist_active_401_posture_passes(stub):
    """A worker WITH a validator allowlist verifies signatures first and
    refuses unauthenticated validator routes with 401 "missing or
    duplicate validator auth header". That is the CORRECT production
    posture, so demanding exactly 403 rejects correctly configured workers."""
    handler, endpoint = stub
    handler.chat_status = 401
    handler.chat_body = {
        "error": "missing or duplicate validator auth header"
    }
    checks = check_public_endpoint(endpoint, timeout=5.0)
    assert _by_name(
        checks, "validator-auth-posture /v1/chat/completions"
    ).passed
    # A bare 401 without the validator-auth marker is still a failure
    # (some other upstream answered).
    handler.chat_body = {"error": "invalid api key"}
    checks = check_public_endpoint(endpoint, timeout=5.0)
    posture = _by_name(
        checks, "validator-auth-posture /v1/chat/completions"
    )
    assert not posture.passed
    assert "unexpected body" in posture.observed


def test_open_chat_route_means_validator_auth_disabled(stub):
    handler, endpoint = stub
    handler.chat_status = 200
    handler.chat_body = {"choices": []}
    checks = check_public_endpoint(endpoint, timeout=5.0)
    posture = _by_name(checks, "validator-auth-posture /v1/chat/completions")
    assert not posture.passed
    assert "DISABLED" in posture.observed


def test_wrong_upstream_404_fails_posture(stub):
    handler, endpoint = stub
    handler.chat_status = 404
    handler.chat_body = {"error": "not found"}
    checks = check_public_endpoint(endpoint, timeout=5.0)
    assert not _by_name(
        checks, "validator-auth-posture /v1/chat/completions"
    ).passed


def test_unreachable_endpoint_fails_health():
    checks = check_public_endpoint("http://127.0.0.1:1", timeout=1.0)
    assert not _by_name(checks, "public-health").passed


def test_stock_self_signed_https_passes_transport_and_expiry(self_signed_stub):
    _handler, endpoint = self_signed_stub
    checks = check_public_endpoint(endpoint, timeout=5.0)

    assert _by_name(checks, "public-health").passed
    assert _by_name(checks, "validator-auth-posture /v1/chat/completions").passed
    tls = _by_name(checks, "tls-certificate")
    assert tls.kind == "hard"
    assert tls.passed
    assert "expires in" in tls.observed


def test_self_signed_https_still_enforces_certificate_lifetime(self_signed_stub):
    _handler, endpoint = self_signed_stub
    checks = check_public_endpoint(endpoint, timeout=5.0, min_tls_days=3.0)

    tls = _by_name(checks, "tls-certificate")
    assert tls.kind == "hard"
    assert not tls.passed


def test_probe_worker_is_never_used(stub, monkeypatch):
    """/capability requires the internal HMAC on production coordinators,
    so the reachability check must never call probe_worker."""
    _handler, endpoint = stub

    def _forbidden(*args, **kwargs):
        raise AssertionError("check_public_endpoint must not call probe_worker")

    import verallm.mesh.worker as worker_module

    monkeypatch.setattr(worker_module, "probe_worker", _forbidden)
    checks = check_public_endpoint(endpoint, timeout=5.0)
    assert checks
    source = open(probe_module.__file__).read()
    assert "probe_worker(" not in source.replace("probe_worker is deliberately", "")
