import json
import time
from email.message import Message

import pytest
try:
    from bittensor_wallet import Keypair  # modern bittensor
except ImportError:  # pragma: no cover - legacy dependency layout
    from substrateinterface import Keypair

from neurons.request_signing import (
    HDR_HOTKEY,
    HDR_SIGNATURE,
    HDR_TIMESTAMP,
    build_signing_message,
)
from verallm.mesh.http_auth import (
    DEFAULT_VALIDATOR_ALLOWLIST_MAX_AGE_SECONDS,
    HDR_MESH_NONCE,
    HDR_MESH_SIGNATURE,
    HDR_MESH_TIMESTAMP,
    RequestReplayCache,
    ValidatorAllowlist,
    read_validator_allowlist,
    sign_internal_http_request,
    verify_internal_http_request,
    verify_validator_http_request,
)


def _write_allowlist(path, *hotkeys: str, updated_at: int | None = None) -> None:
    path.write_text(
        json.dumps(
            {
                "updated_at": int(time.time()) if updated_at is None else updated_at,
                "netuid": 405,
                "validators": [
                    {"uid": uid, "hotkey_ss58": hotkey, "stake": 1.0}
                    for uid, hotkey in enumerate(hotkeys)
                ],
            }
        ),
        encoding="utf-8",
    )


def _validator_headers(
    keypair: Keypair,
    *,
    method: str,
    path: str,
    body: bytes,
    timestamp: int,
) -> dict[str, str]:
    timestamp_text = str(timestamp)
    signature = keypair.sign(build_signing_message(method, path, body, timestamp_text))
    return {
        HDR_HOTKEY: keypair.ss58_address,
        HDR_SIGNATURE: signature.hex(),
        HDR_TIMESTAMP: timestamp_text,
    }


def test_validator_allowlist_reads_existing_schema_and_fails_closed(tmp_path):
    allowlist_path = tmp_path / "validators.json"
    _write_allowlist(allowlist_path, "5validator-a", "5validator-b")
    assert read_validator_allowlist(allowlist_path) == frozenset(
        {"5validator-a", "5validator-b"}
    )

    allowlist = ValidatorAllowlist(allowlist_path, reload_interval_seconds=0)
    assert allowlist.snapshot() == (
        True,
        frozenset({"5validator-a", "5validator-b"}),
    )

    # A malformed refresh must clear the previously trusted set, not retain it.
    allowlist_path.write_text('{"validators":[{"uid":1}]}', encoding="utf-8")
    assert allowlist.snapshot() == (False, frozenset())

    allowlist_path.unlink()
    assert allowlist.snapshot() == (False, frozenset())


def test_validator_allowlist_can_reject_a_stale_file(tmp_path):
    allowlist_path = tmp_path / "validators.json"
    _write_allowlist(allowlist_path, "5validator", updated_at=100)
    allowlist = ValidatorAllowlist(
        allowlist_path,
        reload_interval_seconds=0,
        clock=lambda: 100 + DEFAULT_VALIDATOR_ALLOWLIST_MAX_AGE_SECONDS + 1,
    )
    assert allowlist.snapshot() == (False, frozenset())


def test_validator_allowlist_expires_cached_trust_at_max_age(tmp_path):
    allowlist_path = tmp_path / "validators.json"
    _write_allowlist(allowlist_path, "5validator", updated_at=100)
    now = [150.0]
    allowlist = ValidatorAllowlist(
        allowlist_path,
        reload_interval_seconds=3600,
        clock=lambda: now[0],
        monotonic_clock=lambda: 1.0,
    )
    assert allowlist.snapshot() == (True, frozenset({"5validator"}))

    now[0] = 100 + DEFAULT_VALIDATOR_ALLOWLIST_MAX_AGE_SECONDS + 1
    assert allowlist.snapshot() == (False, frozenset())


@pytest.mark.parametrize("updated_at", [None, True, float("nan")])
def test_validator_allowlist_rejects_malformed_refresh_time(
    tmp_path,
    updated_at,
):
    allowlist_path = tmp_path / "validators.json"
    allowlist_path.write_text(
        json.dumps(
            {
                "updated_at": updated_at,
                "validators": [{"hotkey_ss58": "5validator"}],
            }
        ),
        encoding="utf-8",
    )
    assert ValidatorAllowlist(allowlist_path).snapshot() == (False, frozenset())


def test_validator_request_checks_raw_request_replay_and_new_signature(tmp_path):
    keypair = Keypair.create_from_seed("11" * 32)
    allowlist_path = tmp_path / "validators.json"
    _write_allowlist(allowlist_path, keypair.ss58_address)
    allowlist = ValidatorAllowlist(allowlist_path)
    replay_cache = RequestReplayCache()

    now = int(time.time())
    method = "POST"
    path = "/v1/chat/completions"
    body = b'{"messages":[{"role":"user","content":"mesh"}]}'
    headers = _validator_headers(
        keypair,
        method=method,
        path=path,
        body=body,
        timestamp=now,
    )

    # Failed tampering does not poison the cache for the authentic request.
    tampered = verify_validator_http_request(
        method=method,
        path=path,
        body=body + b" ",
        headers=headers,
        allowlist=allowlist,
        replay_cache=replay_cache,
    )
    assert not tampered.ok
    assert tampered.status_code == 401

    first = verify_validator_http_request(
        method=method,
        path=path,
        body=body,
        headers=headers,
        allowlist=allowlist,
        replay_cache=replay_cache,
    )
    assert first.ok
    assert first.principal == keypair.ss58_address

    replay = verify_validator_http_request(
        method=method,
        path=path,
        body=body,
        headers=headers,
        allowlist=allowlist,
        replay_cache=replay_cache,
    )
    assert not replay.ok
    assert "already accepted" in replay.reason

    # The same operation is legitimate when the validator signs a fresh
    # timestamp, because its authenticated envelope has changed.
    fresh_headers = _validator_headers(
        keypair,
        method=method,
        path=path,
        body=body,
        timestamp=now + 1,
    )
    fresh = verify_validator_http_request(
        method=method,
        path=path,
        body=body,
        headers=fresh_headers,
        allowlist=allowlist,
        replay_cache=replay_cache,
    )
    assert fresh.ok


def test_validator_request_rejects_unavailable_or_unlisted_allowlist(tmp_path):
    keypair = Keypair.create_from_seed("22" * 32)
    now = int(time.time())
    headers = _validator_headers(
        keypair,
        method="GET",
        path="/epoch/1/receipts",
        body=b"",
        timestamp=now,
    )

    missing = ValidatorAllowlist(tmp_path / "missing.json")
    unavailable = verify_validator_http_request(
        method="GET",
        path="/epoch/1/receipts",
        body=b"",
        headers=headers,
        allowlist=missing,
        replay_cache=RequestReplayCache(),
    )
    assert not unavailable.ok
    assert unavailable.status_code == 503

    allowlist_path = tmp_path / "validators.json"
    _write_allowlist(allowlist_path, "5somebody-else")
    unlisted = verify_validator_http_request(
        method="GET",
        path="/epoch/1/receipts",
        body=b"",
        headers=headers,
        allowlist=ValidatorAllowlist(allowlist_path),
        replay_cache=RequestReplayCache(),
    )
    assert not unlisted.ok
    assert unlisted.status_code == 403


def test_duplicate_validator_header_is_rejected(tmp_path):
    keypair = Keypair.create_from_seed("33" * 32)
    allowlist_path = tmp_path / "validators.json"
    _write_allowlist(allowlist_path, keypair.ss58_address)
    now = int(time.time())
    signed = _validator_headers(
        keypair,
        method="POST",
        path="/inference",
        body=b"{}",
        timestamp=now,
    )
    headers = Message()
    for name, value in signed.items():
        headers.add_header(name, value)
    headers.add_header(HDR_TIMESTAMP, str(now))

    result = verify_validator_http_request(
        method="POST",
        path="/inference",
        body=b"{}",
        headers=headers,
        allowlist=ValidatorAllowlist(allowlist_path),
        replay_cache=RequestReplayCache(),
    )
    assert not result.ok
    assert result.status_code == 401
    assert "duplicate" in result.reason


def test_internal_hmac_binds_exact_method_path_body_and_timestamp():
    secret = b"coordinator-worker-secret"
    timestamp = 2_000_000_000
    method = "POST"
    path = "/v1/mesh/proof"
    body = b'{"request_id":"req-1"}'
    headers = sign_internal_http_request(
        secret=secret,
        method=method,
        path=path,
        body=body,
        timestamp=timestamp,
    )

    valid = verify_internal_http_request(
        secret=secret,
        method=method,
        path=path,
        body=body,
        headers=headers,
        now=timestamp,
    )
    assert valid.ok

    variants = [
        ("GET", path, body, headers),
        (method, path + "/", body, headers),
        (method, path, body + b" ", headers),
        (
            method,
            path,
            body,
            {
                **headers,
                HDR_MESH_TIMESTAMP: str(timestamp + 1),
            },
        ),
    ]
    for changed_method, changed_path, changed_body, changed_headers in variants:
        result = verify_internal_http_request(
            secret=secret,
            method=changed_method,
            path=changed_path,
            body=changed_body,
            headers=changed_headers,
            now=timestamp,
        )
        assert not result.ok
        assert result.status_code == 401


def test_internal_hmac_replay_cache_is_optional_and_allows_fresh_request():
    secret = "shared-secret"
    method = "POST"
    path = "/v1/mesh/proof-selection"
    body = b'{"request_id":"req-2"}'
    now = 2_000_000_000
    headers = sign_internal_http_request(
        secret=secret,
        method=method,
        path=path,
        body=body,
        timestamp=now,
    )

    # A caller may omit replay tracking for an idempotent/private transport.
    assert verify_internal_http_request(
        secret=secret,
        method=method,
        path=path,
        body=body,
        headers=headers,
        now=now,
    ).ok


    assert verify_internal_http_request(
        secret=secret,
        method=method,
        path=path,
        body=body,
        headers=headers,
        now=now,
    ).ok

    replay_cache = RequestReplayCache(clock=lambda: now)
    assert verify_internal_http_request(
        secret=secret,
        method=method,
        path=path,
        body=body,
        headers=headers,
        replay_cache=replay_cache,
        now=now,
    ).ok
    replay = verify_internal_http_request(
        secret=secret,
        method=method,
        path=path,
        body=body,
        headers=headers,
        replay_cache=replay_cache,
        now=now,
    )
    assert not replay.ok
    assert "already accepted" in replay.reason

    fresh_headers = sign_internal_http_request(
        secret=secret,
        method=method,
        path=path,
        body=body,
        timestamp=now + 1,
    )
    assert verify_internal_http_request(
        secret=secret,
        method=method,
        path=path,
        body=body,
        headers=fresh_headers,
        replay_cache=replay_cache,
        now=now + 1,
    ).ok


def test_internal_hmac_nonce_allows_same_body_in_same_second():
    secret = b"shared-secret"
    kwargs = {
        "secret": secret,
        "method": "POST",
        "path": "/v1/mesh/proof/receipt",
        "body": b'{"same":"body"}',
        "timestamp": 2_000_000_000,
    }
    first = sign_internal_http_request(**kwargs)
    second = sign_internal_http_request(**kwargs)

    assert first[HDR_MESH_NONCE] != second[HDR_MESH_NONCE]
    cache = RequestReplayCache(clock=lambda: 2_000_000_000)
    for headers in (first, second):
        assert verify_internal_http_request(
            secret=secret,
            method=kwargs["method"],
            path=kwargs["path"],
            body=kwargs["body"],
            headers=headers,
            replay_cache=cache,
            now=kwargs["timestamp"],
        ).ok


def test_internal_hmac_uses_constant_time_comparison(monkeypatch):
    import verallm.mesh.http_auth as http_auth

    calls = []
    real_compare_digest = http_auth.hmac.compare_digest

    def tracking_compare_digest(left, right):
        calls.append((left, right))
        return real_compare_digest(left, right)

    monkeypatch.setattr(http_auth.hmac, "compare_digest", tracking_compare_digest)
    timestamp = 2_000_000_000
    headers = sign_internal_http_request(
        secret=b"secret",
        method="POST",
        path="/internal",
        body=b"payload",
        timestamp=timestamp,
    )
    result = verify_internal_http_request(
        secret=b"secret",
        method="POST",
        path="/internal",
        body=b"payload",
        headers=headers,
        now=timestamp,
    )
    assert result.ok
    assert len(calls) == 1


def test_replay_cache_never_evicts_a_live_entry_to_admit_another():
    cache = RequestReplayCache(max_entries=1, clock=lambda: 100)
    assert cache.claim(b"first", expires_at=200)
    assert not cache.claim(b"second", expires_at=200)
    assert not cache.claim(b"first", expires_at=200)
    assert cache.claim(b"second", expires_at=300, now=201)
