"""Mesh worker /epoch/receipt push + /epoch/{n}/receipts pull round-trip."""

from __future__ import annotations

import json
import urllib.request

from verallm.mesh.types import CapabilityAd
from verallm.mesh.worker import serve_worker_in_thread

DIGEST_A = "ab" * 32


def _post(url: str, body: dict) -> tuple[int, dict]:
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode())


def _get(url: str) -> tuple[int, dict]:
    with urllib.request.urlopen(url, timeout=10) as r:
        return r.status, json.loads(r.read().decode())


def _receipt(epoch: int = 42) -> dict:
    return {
        "miner_address": "0x" + "aa" * 20,
        "model_id": "mesh-model",
        "model_index": 0,
        "epoch_number": epoch,
        "commitment_hash": DIGEST_A,
        "timestamp": 1_700_000_000,
        "ttft_ms": 100.0,
        "tokens_generated": 32,
        "generation_time_ms": 900.0,
        "tokens_per_sec": 35.5,
        "prompt_tokens": 12,
        "proof_verified": True,
        "proof_requested": True,
        "is_canary": True,
        "validator_hotkey": "cd" * 32,
        "validator_signature": "ef" * 64,
    }


def test_receipt_push_and_pull_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("VERALLM_DATA_DIR", str(tmp_path))
    capability = CapabilityAd(
        uid=11,
        hotkey="5Worker",
        endpoint="http://127.0.0.1:9338",
        supported_backends=["gguf_stage_worker"],
        cached_model_package_hashes=[DIGEST_A],
    )
    server, thread = serve_worker_in_thread(capability=capability)
    host, port = server.server_address
    base = f"http://{host}:{port}"
    try:
        status, out = _post(f"{base}/epoch/receipt", _receipt(epoch=42))
        assert status == 200
        assert out["status"] == "accepted"
        assert out["count"] == 1

        status, out = _post(f"{base}/epoch/receipt", _receipt(epoch=42))
        assert status == 200 and out["count"] == 2

        status, out = _get(f"{base}/epoch/42/receipts")
        assert status == 200
        assert out["receipt_count"] == 2
        assert out["receipts"][0]["model_id"] == "mesh-model"

        status, out = _get(f"{base}/epoch/41/receipts")
        assert status == 200 and out["receipt_count"] == 0

        # Missing signature -> rejected
        bad = _receipt(epoch=43)
        bad.pop("validator_signature")
        status, out = _post(f"{base}/epoch/receipt", bad)
        assert status == 400
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
