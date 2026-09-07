from __future__ import annotations

import pytest

from verallm.mesh import local_dial


@pytest.mark.parametrize(
    "endpoint,expected",
    [
        ("http://203.0.113.5:9402/v1/mesh/update", "http://127.0.0.1:9402/v1/mesh/update"),
        ("203.0.113.5:50052", "127.0.0.1:50052"),
        ("http://203.0.113.6:9402/health", "http://203.0.113.6:9402/health"),
        ("https://203.0.113.5:9402/health", "https://203.0.113.5:9402/health"),
        ("http://203.0.113.5.evil:9402/health", "http://203.0.113.5.evil:9402/health"),
        ("http://name:password@203.0.113.5:9402/health", "http://name:password@203.0.113.5:9402/health"),
    ],
)
def test_local_dial_is_exact_host_and_plain_transport_only(endpoint, expected):
    assert local_dial.endpoint_for_host(endpoint, "203.0.113.5") == expected
    assert local_dial.endpoint_for_host(endpoint, "") == endpoint


def test_rpc_command_changes_only_dial_addresses():
    command = ["llama-server", "--rpc", "203.0.113.5:50052,203.0.113.6:50052", "--model", "model.gguf"]
    original = list(command)
    result = local_dial.rpc_command_for_host(command, "203.0.113.5")
    assert result[2] == "127.0.0.1:50052,203.0.113.6:50052"
    assert result[:2] == command[:2] and result[3:] == command[3:]
    assert command == original


@pytest.mark.parametrize("secret", ["", "test-internal-secret"])
def test_post_preserves_payload_and_authentication(monkeypatch, secret):
    import io
    import json
    from verallm.mesh import worker

    monkeypatch.setattr(local_dial, "_advertised_host", "203.0.113.5")
    requests = []
    signed = []

    def open_request(request, **kwargs):
        requests.append(request)
        return io.BytesIO(b'{"ok":true}')

    def sign(**kwargs):
        signed.append(kwargs)
        return {"X-Test-Signature": "unchanged"}

    monkeypatch.setattr(worker, "urlopen", open_request)
    monkeypatch.setattr(worker, "sign_internal_http_request", sign)
    payload = {"endpoint": "http://203.0.113.5:9402", "mesh_spec_hash": "a" * 64}
    assert worker.post_json("http://203.0.113.5:9402/v1/mesh/update", payload,
                            internal_auth_secret=secret) == {"ok": True}
    expected_host = "127.0.0.1" if secret else "203.0.113.5"
    assert requests[0].full_url == f"http://{expected_host}:9402/v1/mesh/update"
    assert json.loads(requests[0].data) == payload
    if secret:
        assert signed[0]["secret"] == secret
        assert signed[0]["path"] == "/v1/mesh/update"
        assert signed[0]["body"] == requests[0].data
    else:
        assert signed == []


def test_authenticated_probe_retains_advertised_identity(monkeypatch):
    from verallm.mesh import worker
    from verallm.mesh.types import CapabilityAd

    public = "http://203.0.113.5:9402"
    capability = CapabilityAd(uid=1, hotkey="test", endpoint=public)
    monkeypatch.setattr(local_dial, "_advertised_host", "203.0.113.5")
    calls = []

    def fetch(url, **kwargs):
        calls.append((url, kwargs.get("internal_auth_secret", "")))
        if url.endswith("/health"):
            return {"status": "ok"}
        return {"capability": capability.to_dict(), "capability_hash": capability.ad_hash_hex()}

    monkeypatch.setattr(worker, "_fetch_json", fetch)
    result = worker.probe_worker(public, internal_auth_secret="test-secret")
    assert calls == [("http://127.0.0.1:9402/health", ""),
                     ("http://127.0.0.1:9402/capability", "test-secret")]
    assert result.endpoint == public
    assert result.capability.endpoint == public
