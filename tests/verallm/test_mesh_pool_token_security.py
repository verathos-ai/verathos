from __future__ import annotations

import argparse
import json
import os
import socket
import stat
import threading
from pathlib import Path

import pytest

import verallm.mesh.cli as mesh_cli
import verallm.mesh.state as mesh_state
from verallm.mesh.pool import (
    POOL_ADMIN_TOKEN_FILE,
    POOL_STATE_FILE,
    POOL_TOKEN_FILE,
    POOL_TOKEN_SCOPE_MANAGEMENT,
    POOL_TOKEN_SCOPE_WORKER,
    MAX_POOL_REQUEST_BODY_BYTES,
    MeshPoolToken,
    PoolManager,
    create_pool_state,
    load_pool_token_file,
    serve_pool_manager,
)
from verallm.mesh.pool_dashboard import DASHBOARD_HTML
from verallm.mesh.state import JOIN_TOKEN_FILE, MESH_STATE_FILE, create_mesh_state, join_mesh
from verallm.mesh.types import MeshSpec


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def _raw_http_status(host: str, port: int, request: bytes) -> int:
    with socket.create_connection((host, port), timeout=5.0) as connection:
        connection.sendall(request)
        raw = connection.recv(4096)
    status_line = raw.split(b"\r\n", 1)[0]
    return int(status_line.split()[1])


def _spec() -> MeshSpec:
    return MeshSpec.new_private_mesh(
        coordinator_uid=4,
        coordinator_hotkey="5Coordinator",
        endpoint="http://coordinator.local:9443",
        model_id="qwen-test",
        model_package_hash="a" * 64,
        total_layers=4,
    )


def test_new_pool_secret_files_are_owner_only_even_with_open_umask(tmp_path):
    previous_umask = os.umask(0)
    try:
        state_dir, token = create_pool_state(
            tmp_path,
            manager_endpoint="http://manager.local:9500",
            serving_mode="dev",
        )
    finally:
        os.umask(previous_umask)

    assert _mode(state_dir / POOL_STATE_FILE) == 0o600
    assert _mode(state_dir / POOL_TOKEN_FILE) == 0o600
    assert _mode(state_dir / POOL_ADMIN_TOKEN_FILE) == 0o600
    assert load_pool_token_file(state_dir / POOL_TOKEN_FILE) == token
    assert token.scope == POOL_TOKEN_SCOPE_WORKER
    assert (
        load_pool_token_file(state_dir / POOL_ADMIN_TOKEN_FILE).scope
        == POOL_TOKEN_SCOPE_MANAGEMENT
    )


def test_pool_token_file_rejects_shared_files_and_symlinks(tmp_path):
    token = MeshPoolToken(
        pool_id="pool-test",
        manager_endpoint="http://manager.local:9500",
        pool_secret="secret",
    )
    source = tmp_path / "pool-token.txt"
    source.write_text(token.encode() + "\n", encoding="utf-8")
    source.chmod(0o600)

    link = tmp_path / "pool-token-link.txt"
    link.symlink_to(source)
    with pytest.raises(ValueError, match="non-symlink"):
        load_pool_token_file(link)

    source.chmod(0o640)
    with pytest.raises(PermissionError, match="group or world"):
        load_pool_token_file(source)


def test_worker_credential_cannot_authorize_management_or_private_status(tmp_path):
    state_dir, worker_token = create_pool_state(
        tmp_path,
        manager_endpoint="http://manager.local:9500",
        serving_mode="dev",
    )
    admin_token = load_pool_token_file(state_dir / POOL_ADMIN_TOKEN_FILE)
    manager = PoolManager(state_dir)

    worker_auth = {"pool_secret": worker_token.pool_secret}
    admin_auth = {"management_secret": admin_token.pool_secret}

    manager._auth(worker_auth)
    manager._auth_manage(admin_auth)
    assert manager.handle_status(admin_auth)["status"] == "ok"
    with pytest.raises(PermissionError, match="admin token"):
        manager._auth_manage(worker_auth)
    with pytest.raises(PermissionError, match="admin token"):
        manager.handle_status(worker_auth)
    with pytest.raises(PermissionError, match="admin token"):
        manager.handle_operator_access(worker_auth)
    with pytest.raises(PermissionError, match="admin token"):
        manager.handle_launch({**worker_auth, "model_id": "anything"})
    with pytest.raises(PermissionError, match="invalid pool token"):
        manager._auth({"pool_secret": admin_token.pool_secret})


def test_pool_http_rejects_unsafe_or_oversized_bodies_before_dispatch(tmp_path):
    state_dir, _token = create_pool_state(
        tmp_path,
        manager_endpoint="http://127.0.0.1:9500",
        serving_mode="dev",
    )
    server = serve_pool_manager(state_dir, host="127.0.0.1", port=0)
    host, port = server.server_address
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        duplicate_length = (
            b"POST /v1/auth/challenge HTTP/1.1\r\n"
            + f"Host: {host}:{port}\r\n".encode()
            + b"Content-Length: 2\r\n"
            + b"Content-Length: 2\r\n"
            + b"Connection: close\r\n\r\n{}"
        )
        assert _raw_http_status(host, port, duplicate_length) == 400

        transfer_encoding = (
            b"POST /v1/auth/challenge HTTP/1.1\r\n"
            + f"Host: {host}:{port}\r\n".encode()
            + b"Transfer-Encoding: chunked\r\n"
            + b"Connection: close\r\n\r\n0\r\n\r\n"
        )
        assert _raw_http_status(host, port, transfer_encoding) == 400

        oversized = (
            b"POST /v1/auth/challenge HTTP/1.1\r\n"
            + f"Host: {host}:{port}\r\n".encode()
            + f"Content-Length: {MAX_POOL_REQUEST_BODY_BYTES + 1}\r\n".encode()
            + b"Connection: close\r\n\r\n"
        )
        assert _raw_http_status(host, port, oversized) == 413

        duplicate_json = b'{"account":"first","account":"second"}'
        duplicate_json_request = (
            b"POST /v1/auth/challenge HTTP/1.1\r\n"
            + f"Host: {host}:{port}\r\n".encode()
            + f"Content-Length: {len(duplicate_json)}\r\n".encode()
            + b"Connection: close\r\n\r\n"
            + duplicate_json
        )
        assert _raw_http_status(host, port, duplicate_json_request) == 400
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


def test_worker_credential_cannot_claim_the_owner_wallet(tmp_path, monkeypatch):
    class AcceptingKeypair:
        def verify(self, message, signature):
            return bool(message) and signature == b"signed"

    monkeypatch.setattr(
        "verallm.mesh.receipt_signing._keypair_from_ss58",
        lambda _account: AcceptingKeypair(),
    )
    state_dir, worker_token = create_pool_state(
        tmp_path,
        manager_endpoint="http://manager.local:9500",
        serving_mode="dev",
    )
    admin_token = load_pool_token_file(state_dir / POOL_ADMIN_TOKEN_FILE)
    manager = PoolManager(state_dir)
    account = "5" + "A" * 47

    challenge = manager.handle_auth_challenge({"account": account})
    with pytest.raises(PermissionError, match="pool admin token"):
        manager.handle_auth_verify(
            {
                "account": account,
                "nonce": challenge["nonce"],
                "signature": b"signed".hex(),
                "pool_secret": worker_token.pool_secret,
            }
        )
    assert not manager.state.get("owner_account")

    challenge = manager.handle_auth_challenge({"account": account})
    session = manager.handle_auth_verify(
        {
            "account": account,
            "nonce": challenge["nonce"],
            "signature": b"signed".hex(),
            "management_secret": admin_token.pool_secret,
        }
    )
    assert session["is_owner"] is True
    assert manager.state["owner_account"] == account


def test_existing_pool_mints_distinct_admin_credential_on_migration(tmp_path):
    state_dir, worker_token = create_pool_state(
        tmp_path,
        manager_endpoint="http://manager.local:9500",
        serving_mode="dev",
    )
    state_path = state_dir / POOL_STATE_FILE
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state.pop("management_secret")
    state_path.write_text(json.dumps(state), encoding="utf-8")
    (state_dir / POOL_ADMIN_TOKEN_FILE).unlink()

    manager = PoolManager(state_dir)
    admin_token = load_pool_token_file(state_dir / POOL_ADMIN_TOKEN_FILE)

    assert admin_token.scope == POOL_TOKEN_SCOPE_MANAGEMENT
    assert admin_token.pool_secret == manager.state["management_secret"]
    assert admin_token.pool_secret != worker_token.pool_secret
    assert _mode(state_path) == 0o600
    assert _mode(state_dir / POOL_ADMIN_TOKEN_FILE) == 0o600


def test_new_mesh_join_and_state_secrets_are_owner_only(
    tmp_path, monkeypatch
):
    previous_umask = os.umask(0)
    try:
        coordinator_dir, _, token = create_mesh_state(
            spec=_spec(),
            root=tmp_path / "coordinator",
        )
    finally:
        os.umask(previous_umask)

    assert _mode(coordinator_dir / MESH_STATE_FILE) == 0o600
    assert _mode(coordinator_dir / JOIN_TOKEN_FILE) == 0o600

    monkeypatch.setattr(
        mesh_state,
        "post_json",
        lambda *args, **kwargs: {"mesh": _spec().to_dict()},
    )
    previous_umask = os.umask(0)
    try:
        worker_dir, _ = join_mesh(
            token=token.encode(),
            endpoint="http://worker.local:9443",
            root=tmp_path / "worker",
        )
    finally:
        os.umask(previous_umask)

    assert _mode(worker_dir / MESH_STATE_FILE) == 0o600


def test_cli_prefers_token_file_and_keeps_raw_flag_compatibility(tmp_path):
    state_dir, token = create_pool_state(
        tmp_path,
        manager_endpoint="http://manager.local:9500",
        serving_mode="dev",
    )
    token_path = state_dir / POOL_TOKEN_FILE
    admin_path = state_dir / POOL_ADMIN_TOKEN_FILE

    loaded = mesh_cli._pool_token_from_args(
        argparse.Namespace(pool_token_file=str(token_path), pool_token="")
    )
    legacy = mesh_cli._pool_token_from_args(
        argparse.Namespace(pool_token_file="", pool_token=token.encode())
    )

    assert loaded == legacy == token
    with pytest.raises(SystemExit, match="expected 'management'"):
        mesh_cli._pool_token_from_args(
            argparse.Namespace(pool_token_file=str(token_path), pool_token=""),
            required_scope=POOL_TOKEN_SCOPE_MANAGEMENT,
        )
    with pytest.raises(SystemExit, match="expected 'worker'"):
        mesh_cli._pool_token_from_args(
            argparse.Namespace(pool_token_file=str(admin_path), pool_token=""),
            required_scope=POOL_TOKEN_SCOPE_WORKER,
        )


def test_pool_create_prints_only_owner_token_file_path(tmp_path, capsys):
    mesh_cli.cmd_pool_create(
        argparse.Namespace(
            root=str(tmp_path),
            manager_endpoint="http://manager.local:9500",
            serving_mode="dev",
            owner_account="",
            coordinator_address="",
            validator_shared_state="",
            chain_id=None,
            netuid=None,
            coordinator_uid=None,
            epoch=None,
            snapshot_ttl_seconds=86_400,
        )
    )
    output = capsys.readouterr().out
    assert "Worker token file:" in output
    assert "Pool admin token file:" in output
    assert POOL_TOKEN_FILE in output
    assert POOL_ADMIN_TOKEN_FILE in output
    assert "vtpool_" not in output


def test_cli_manager_ca_file_sets_ssl_cert_file(tmp_path, monkeypatch):
    state_dir, token = create_pool_state(
        tmp_path,
        manager_endpoint="https://manager.local:9543",
        serving_mode="dev",
    )
    ca_path = tmp_path / "manager-ca.pem"
    ca_path.write_text("test CA")
    seen = {}
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    monkeypatch.setattr(
        mesh_cli.ssl,
        "create_default_context",
        lambda *, cafile: seen.setdefault("cafile", cafile),
    )

    # The CLI intentionally exports SSL_CERT_FILE process-wide; monkeypatch
    # registers nothing for an absent variable, so restore explicitly or the
    # synthetic CA poisons every later httpx client in the test process.
    try:
        loaded = mesh_cli._pool_token_from_args(
            argparse.Namespace(
                pool_token_file=str(state_dir / POOL_TOKEN_FILE),
                pool_token="",
                manager_ca_file=str(ca_path),
            )
        )

        assert loaded == token
        assert seen["cafile"] == str(ca_path)
        assert os.environ["SSL_CERT_FILE"] == str(ca_path.resolve())
    finally:
        os.environ.pop("SSL_CERT_FILE", None)


def test_worker_parser_accepts_owner_token_file_and_legacy_raw_token(tmp_path):
    parser = mesh_cli.build_parser()
    required = [
        "--workdir",
        str(tmp_path / "work"),
        "--advertise-host",
        "worker.local",
    ]
    by_file = parser.parse_args(
        [
            "pool",
            "worker",
            "--pool-token-file",
            str(tmp_path / "token"),
            "--manager-ca-file",
            str(tmp_path / "ca.pem"),
            *required,
        ]
    )
    by_raw = parser.parse_args(
        ["pool", "worker", "--pool-token", "vtpool_legacy", *required]
    )
    assert by_file.pool_token_file == str(tmp_path / "token")
    assert by_file.manager_ca_file == str(tmp_path / "ca.pem")
    assert by_raw.pool_token == "vtpool_legacy"


def test_dashboard_enrollment_uses_a_token_file_without_fetching_raw_token():
    assert "--token-file" in DASHBOARD_HTML
    assert 'api("/v1/operator/access"' not in DASHBOARD_HTML
    assert "ACCESS_TOKEN" not in DASHBOARD_HTML
