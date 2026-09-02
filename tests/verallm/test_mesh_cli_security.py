from __future__ import annotations

import argparse
from types import SimpleNamespace

import pytest

import verallm.chain.wallet as chain_wallet
import verallm.mesh.cli as mesh_cli
import verallm.mesh.receipt_signing as receipt_signing
import verallm.mesh.state as mesh_state
from verallm.mesh.state import (
    MeshJoinToken,
    create_mesh_state,
    fetch_coordinator_mesh_spec,
    join_mesh,
    load_mesh_state,
    state_internal_auth_secret,
)
from verallm.mesh.types import CapabilityAd, MeshSpec


PACKAGE_HASH = "a" * 64


_NATIVE_STACK_SKIP_REASON = (
    "zkllm native proof stack unavailable for this torch/CUDA environment; "
    "verified serving refuses these paths fail-closed without it (the "
    "shipped wheels cover torch 2.10/2.11)"
)


def _native_proof_stack_available() -> bool:
    # Delegates to the SAME probe production serving gates on: a real
    # kernel launch, not symbol presence. A wheel built without this
    # GPU's arch imports fine and then fails every launch; these tests
    # must skip exactly where serving refuses.
    try:
        from verallm.mesh.gguf_manifest import _gpu_merkle_hash_available

        return bool(_gpu_merkle_hash_available())
    except Exception:
        return False


requires_native_proof_stack = pytest.mark.skipif(
    not _native_proof_stack_available(), reason=_NATIVE_STACK_SKIP_REASON
)


def _spec(*, hotkey: str = "5Coordinator") -> MeshSpec:
    return MeshSpec.new_private_mesh(
        coordinator_uid=7,
        coordinator_hotkey=hotkey,
        endpoint="http://coordinator.local:9338",
        model_id="qwen2.5-7b-q4-k-m",
        model_package_hash=PACKAGE_HASH,
        total_layers=4,
    )


def test_mesh_state_uses_join_secret_for_internal_hmac(tmp_path, monkeypatch):
    spec = _spec()
    coordinator_dir, coordinator_state, token = create_mesh_state(
        spec=spec,
        root=tmp_path / "coordinator",
    )
    calls: list[tuple[str, str]] = []

    def fake_post(url, payload, *, timeout, internal_auth_secret):
        calls.append((url, internal_auth_secret))
        return {"mesh": spec.to_dict()}

    monkeypatch.setattr(mesh_state, "post_json", fake_post)
    assert fetch_coordinator_mesh_spec(token.encode()).mesh_id == spec.mesh_id
    worker_dir, joined = join_mesh(
        token=token.encode(),
        endpoint="http://worker.local:9338",
        root=tmp_path / "worker",
    )

    assert joined.mesh_id == spec.mesh_id
    assert calls == [
        ("http://coordinator.local:9338/v1/mesh/spec", token.join_secret),
        ("http://coordinator.local:9338/v1/mesh/join", token.join_secret),
    ]
    assert state_internal_auth_secret(coordinator_state) == token.join_secret
    assert state_internal_auth_secret(load_mesh_state(worker_dir)) == token.join_secret
    assert token.join_secret not in joined.to_dict().values()
    assert coordinator_dir.exists()


def test_finalized_mesh_join_allows_only_exact_capability_replay(
    tmp_path,
    monkeypatch,
):
    state_dir, _, token = create_mesh_state(spec=_spec(), root=tmp_path)
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        mesh_cli,
        "serve_worker",
        lambda **kwargs: captured.update(kwargs),
    )
    mesh_cli.main(["serve", "--mesh", str(state_dir)])
    join_handler = captured["join_handler"]
    assert callable(join_handler)
    capability = CapabilityAd(
        uid=7,
        hotkey="5ExactReplayWorker",
        endpoint="http://worker.local:9338",
        supported_backends=["gguf_stage_worker", "llama_cpp_rpc"],
        rpc_endpoint="worker.local:50052",
        proof_endpoint="http://worker.local:9338",
        gpu_name="test-gpu",
        vram_gb=24,
    )
    request = {
        "join_secret": token.join_secret,
        "capability": capability.to_dict(),
    }
    admitted = join_handler(request)
    persisted = load_mesh_state(state_dir)
    persisted["mesh_finalized"] = True
    mesh_state.save_mesh_state(state_dir, persisted)
    state_before_retry = load_mesh_state(state_dir)

    replayed = join_handler(request)

    assert replayed["status"] == "joined"
    assert replayed["mesh"] == admitted["mesh"]
    assert replayed["mesh_spec_hash"] == admitted["mesh_spec_hash"]
    assert replayed["stage_assignment_hash"] == admitted[
        "stage_assignment_hash"
    ]
    assert replayed["mesh_update_errors"] == []
    assert load_mesh_state(state_dir) == state_before_retry

    altered = capability.to_dict()
    altered["vram_gb"] = 25
    with pytest.raises(PermissionError, match="assignment is finalized"):
        join_handler(
            {"join_secret": token.join_secret, "capability": altered}
        )
    new_member = capability.to_dict()
    new_member["endpoint"] = "http://other-worker.local:9338"
    new_member["rpc_endpoint"] = "other-worker.local:50052"
    new_member["proof_endpoint"] = "http://other-worker.local:9338"
    with pytest.raises(PermissionError, match="assignment is finalized"):
        join_handler(
            {"join_secret": token.join_secret, "capability": new_member}
        )
    assert load_mesh_state(state_dir) == state_before_retry


def test_stage_public_identity_is_pinned_as_member_proof_key(tmp_path):
    stage = receipt_signing.ensure_stage_proof_key_file(tmp_path / "stage.seed")
    spec = _spec()
    assigned = mesh_state.assign_mesh_members(
        spec,
        [
            CapabilityAd(
                uid=7,
                hotkey=spec.coordinator_hotkey,
                endpoint="http://coordinator.local:9338",
                supported_backends=["gguf_stage"],
            ),
            CapabilityAd(
                uid=7,
                hotkey=stage.ss58_address,
                endpoint="http://worker.local:9338",
                supported_backends=["gguf_stage_worker", "llama_cpp_rpc"],
                rpc_endpoint="worker.local:50052",
                proof_endpoint="http://worker.local:9338",
            ),
        ],
        coordinator_computes=False,
    )

    assert assigned.members[1].hotkey == stage.ss58_address
    assert assigned.members[1].proof_key == stage.ss58_address


def test_state_internal_secret_rejects_untrusted_or_incomplete_state():
    with pytest.raises(ValueError, match="role"):
        state_internal_auth_secret({"role": "observer"})
    with pytest.raises(ValueError, match="internal auth secret"):
        state_internal_auth_secret({"role": "coordinator", "join_secret": ""})
    with pytest.raises(ValueError, match="internal auth secret"):
        state_internal_auth_secret({"role": "worker", "join_token": ""})


def test_wallet_serving_identity_derives_receipt_and_evm_from_hotkey(monkeypatch):
    class FakeKeypair:
        ss58_address = "5ServingHotkey"

    keypair = FakeKeypair()
    seed = bytes(range(32))
    seen: dict[str, object] = {}
    monkeypatch.setattr(
        receipt_signing,
        "load_hotkey_keypair",
        lambda wallet, hotkey: keypair,
    )

    def fake_seed(wallet, hotkey, *, keypair):
        seen["seed_keypair"] = keypair
        return seed

    monkeypatch.setattr(receipt_signing, "load_hotkey_seed", fake_seed)
    monkeypatch.setattr(
        receipt_signing,
        "sign_receipt_hash",
        lambda receipt_hash, keypair: f"signed:{receipt_hash}",
    )
    monkeypatch.setattr(chain_wallet, "derive_evm_address", lambda value: "0xDerived")
    monkeypatch.setattr(chain_wallet, "derive_evm_private_key", lambda value: "11" * 32)
    args = argparse.Namespace(
        wallet_name="miner",
        wallet_hotkey="default",
        evm_address="0xderived",
        evm_private_key="0x" + "11" * 32,
        hotkey="",
    )

    signer, evm_address, evm_private_key, challenge_signer = (
        mesh_cli._serving_identity_from_args(args)
    )

    assert challenge_signer is None
    assert args.hotkey == "5ServingHotkey"
    assert seen["seed_keypair"] is keypair
    assert evm_address == "0xDerived"
    assert evm_private_key == "11" * 32
    assert signer is not None and signer("abc") == "signed:abc"


@pytest.mark.parametrize(
    ("address", "private_key", "match"),
    [
        ("0xWrong", "11" * 32, "evm-address"),
        ("0xDerived", "22" * 32, "evm-private-key"),
    ],
)
def test_wallet_identity_rejects_explicit_evm_mismatch(
    monkeypatch, address, private_key, match
):
    class FakeKeypair:
        ss58_address = "5ServingHotkey"

    monkeypatch.setattr(
        receipt_signing,
        "load_hotkey_keypair",
        lambda wallet, hotkey: FakeKeypair(),
    )
    monkeypatch.setattr(
        receipt_signing,
        "load_hotkey_seed",
        lambda wallet, hotkey, *, keypair: b"s" * 32,
    )
    monkeypatch.setattr(chain_wallet, "derive_evm_address", lambda value: "0xDerived")
    monkeypatch.setattr(chain_wallet, "derive_evm_private_key", lambda value: "11" * 32)
    args = argparse.Namespace(
        wallet_name="miner",
        wallet_hotkey="default",
        evm_address=address,
        evm_private_key=private_key,
        hotkey="",
    )
    with pytest.raises(SystemExit, match=match):
        mesh_cli._serving_identity_from_args(args)


def test_mesh_serve_forwards_state_security_to_worker_api(tmp_path, monkeypatch):
    spec = _spec()
    state_dir, state, token = create_mesh_state(spec=spec, root=tmp_path)
    captured: dict[str, object] = {}

    def fake_serve_worker(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(mesh_cli, "serve_worker", fake_serve_worker)
    mesh_cli.main(
        [
            "serve",
            "--mesh",
            str(state_dir),
            "--validator-auth",
            "--require-validator-nonce",
            "--validator-allowlist-path",
            str(tmp_path / "validators.json"),
            "--validator-allowlist-max-age-seconds",
            "321",
            "--evm-address",
            "0x1234",
            "--evm-private-key",
            "11" * 32,
        ]
    )

    assert captured["server_role"] == "coordinator"
    assert captured["validator_auth_enabled"] is True
    assert captured["require_validator_nonce"] is True
    assert captured["validator_allowlist_path"] == str(tmp_path / "validators.json")
    assert captured["validator_allowlist_max_age_seconds"] == 321
    assert captured["internal_auth_secret"] == token.join_secret == state["join_secret"]
    assert captured["evm_address"] == "0x1234"
    assert captured["evm_private_key"] == "11" * 32


@requires_native_proof_stack
def test_pool_worker_forwards_validator_allowlist_max_age(
    tmp_path,
    monkeypatch,
):
    from verallm.mesh.pool import MeshPoolToken

    captured: dict[str, object] = {}
    monkeypatch.setattr(
        "verallm.mesh.pool.pool_worker_loop",
        lambda config, **kwargs: captured.update(config=config),
    )
    token = MeshPoolToken(
        pool_id="pool-test",
        manager_endpoint="https://manager.invalid:19543",
        pool_secret="secret",
    )
    mesh_cli.main(
        [
            "pool",
            "worker",
            "--pool-token",
            token.encode(),
            "--workdir",
            str(tmp_path / "worker"),
            "--advertise-host",
            "worker.invalid",
            "--validator-allowlist-path",
            str(tmp_path / "validators.json"),
            "--validator-allowlist-max-age-seconds",
            "456",
        ]
    )

    config = captured["config"]
    assert config.validator_allowlist_max_age_seconds == 456


def test_mesh_serve_forwards_dedicated_stage_signer_separately(
    tmp_path, monkeypatch
):
    key_file = tmp_path / "stage-proof-key.seed"
    keypair = receipt_signing.ensure_stage_proof_key_file(key_file)
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        mesh_cli,
        "serve_worker",
        lambda **kwargs: captured.update(kwargs),
    )

    mesh_cli.main(
        [
            "serve",
            "--uid",
            "9",
            "--hotkey",
            keypair.ss58_address,
            "--endpoint",
            "http://worker.local:9338",
            "--server-role",
            "worker",
            "--stage-proof-key-file",
            str(key_file),
        ]
    )

    assert captured["receipt_signer"] is None
    assert captured["stage_proof_key"] == keypair.ss58_address
    signer = captured["stage_receipt_signer"]
    assert callable(signer)
    body_hash = "6b" * 32
    signature = signer(body_hash)
    assert receipt_signing.verify_stage_proof_receipt_signature(
        body_hash,
        signature,
        keypair.ss58_address,
        "sr25519",
    )


def test_mesh_serve_rejects_public_stage_seed_file(tmp_path, monkeypatch):
    key_file = tmp_path / "stage-proof-key.seed"
    key_file.write_text("33" * 32 + "\n", encoding="ascii")
    key_file.chmod(0o644)
    monkeypatch.setattr(mesh_cli, "serve_worker", lambda **kwargs: None)

    with pytest.raises(SystemExit, match="group or world"):
        mesh_cli.main(
            [
                "serve",
                "--uid",
                "9",
                "--hotkey",
                "5Worker",
                "--endpoint",
                "http://worker.local:9338",
                "--stage-proof-key-file",
                str(key_file),
            ]
        )


def test_stage_signer_cannot_reuse_coordinator_wallet_identity(tmp_path):
    key_file = tmp_path / "stage-proof-key.seed"
    keypair = receipt_signing.ensure_stage_proof_key_file(key_file)
    args = argparse.Namespace(
        stage_proof_key_file=str(key_file),
        wallet_name="miner",
        hotkey=keypair.ss58_address,
    )

    with pytest.raises(SystemExit, match="must not reuse"):
        mesh_cli._stage_proof_identity_from_args(args)


def test_mesh_serve_rejects_role_that_conflicts_with_state(tmp_path, monkeypatch):
    state_dir, _, _ = create_mesh_state(spec=_spec(), root=tmp_path)
    monkeypatch.setattr(mesh_cli, "serve_worker", lambda **kwargs: None)
    with pytest.raises(SystemExit, match="conflicts with coordinator mesh state"):
        mesh_cli.main(
            [
                "serve",
                "--mesh",
                str(state_dir),
                "--server-role",
                "worker",
            ]
        )


def test_pool_register_model_cli_forwards_exact_context_limit(monkeypatch):
    captured: dict[str, object] = {}

    def fake_client(_args):
        def call(path, body):
            captured["path"] = path
            captured["body"] = body
            return {"status": "ok"}

        return call

    monkeypatch.setattr(mesh_cli, "_pool_client", fake_client)
    monkeypatch.setattr(mesh_cli, "_print_json", lambda _value: None)

    mesh_cli.main(
        [
            "pool",
            "register-model",
            "--pool-token",
            "vtpool_test",
            "--model-id",
            "qwen-test-q4",
            "--hf-repo",
            "operator/model",
            "--hf-files",
            "model.gguf",
            "--layers",
            "40",
            "--model-bytes",
            "1024",
            "--model-index",
            "26",
            "--max-context-len",
            "32768",
        ]
    )

    assert captured["path"] == "/v1/pool/register-model"
    assert captured["body"]["max_context_len"] == 32_768


def test_deploy_cli_forwards_model_specific_stored_registration(monkeypatch):
    from verallm.chain.config import ChainConfig
    import verallm.mesh.deploy as mesh_deploy

    previous = {
        "model_id": "qwen-test-q4",
        "endpoint": "https://old.example:9443",
        "index": 7,
    }
    captured: dict[str, object] = {}

    def call(path, body):
        assert path == "/v1/pool/registration-state"
        assert body == {}
        return {"registrations": {"qwen-test-q4": previous}}

    monkeypatch.setattr(ChainConfig, "resolve_config_path", lambda *a: "test.json")
    monkeypatch.setattr(
        ChainConfig,
        "from_json",
        lambda _path: SimpleNamespace(chain_id=945, netuid=405),
    )
    monkeypatch.setattr(mesh_cli, "_pool_client", lambda _args: call)
    monkeypatch.setattr(
        mesh_cli, "_default_signer_from_pool", lambda _args, _call: None
    )
    monkeypatch.setattr(
        mesh_cli,
        "_deploy_credentials",
        lambda _args: ("0x" + "11" * 32, b"\x05" * 32, "5Hotkey"),
    )

    def run_deploy(config, **kwargs):
        captured["config"] = config
        return SimpleNamespace(failed=False, to_dict=lambda: {})

    monkeypatch.setattr(mesh_deploy, "run_deploy", run_deploy)
    monkeypatch.setattr(
        mesh_cli.os,
        "_exit",
        lambda code: (_ for _ in ()).throw(SystemExit(code)),
    )
    args = argparse.Namespace(
        chain_config="test.json",
        subtensor_network="test",
        subtensor_chain_endpoint="",
        endpoint="https://new.example:9443",
        model_id="qwen-test-q4",
        uid=2,
        max_context_len=None,
        validator_budget_s=None,
        workers="",
        driver="",
        hf_repo="",
        hf_files="",
        model_bytes=0,
        probe_samples=1,
        hard_samples=0,
        min_tok_s=1.0,
        no_full_context_probe=True,
        full_context_budget_s=10.0,
        yes=True,
        force=False,
        dry_run=True,
        json=False,
    )

    with pytest.raises(SystemExit) as exc:
        mesh_cli.cmd_deploy(args)

    assert exc.value.code == 0
    assert captured["config"].previous_registration == previous


def _pinned_token(pin: str):
    from verallm.mesh.pool import MeshPoolToken

    return MeshPoolToken(
        pool_id="pool-pin",
        manager_endpoint="https://198.51.100.7:20102",
        pool_secret="s3cret",
        manager_ca_sha256=pin,
    )


_SELF_SIGNED_PEM = None


def _test_cert_pem(tmp_path) -> str:
    """A real self-signed cert (minted once per run via openssl)."""

    global _SELF_SIGNED_PEM
    if _SELF_SIGNED_PEM is None:
        import subprocess

        key = tmp_path / "k.pem"
        cert = tmp_path / "c.pem"
        subprocess.run(
            [
                "openssl", "req", "-x509", "-newkey", "rsa:2048",
                "-keyout", str(key), "-out", str(cert),
                "-days", "1", "-nodes", "-subj", "/CN=test",
            ],
            capture_output=True,
            check=True,
        )
        _SELF_SIGNED_PEM = cert.read_text()
    return _SELF_SIGNED_PEM


def test_manager_tls_pin_accepts_matching_certificate(tmp_path, monkeypatch):
    import hashlib
    import ssl as _ssl

    pem = _test_cert_pem(tmp_path)
    pin = hashlib.sha256(_ssl.PEM_cert_to_DER_cert(pem)).hexdigest()
    monkeypatch.setattr(
        mesh_cli.ssl, "get_server_certificate", lambda addr: pem
    )
    monkeypatch.setattr(mesh_cli.Path, "home", lambda: tmp_path)
    installed = []
    import urllib.request as _urllib_request

    monkeypatch.setattr(
        _urllib_request, "install_opener", lambda opener: installed.append(opener)
    )

    mesh_cli._install_manager_tls_pin(_pinned_token(pin))

    assert installed, "pin match must install the pinned HTTPS opener"
    pin_file = tmp_path / ".verathos" / "manager-pins" / "pool-pin.pem"
    assert pin_file.read_text() == pem


def test_manager_tls_pin_rejects_mismatched_certificate(tmp_path, monkeypatch):
    pem = _test_cert_pem(tmp_path)
    monkeypatch.setattr(
        mesh_cli.ssl, "get_server_certificate", lambda addr: pem
    )
    with pytest.raises(SystemExit, match="does not match the join token"):
        mesh_cli._install_manager_tls_pin(_pinned_token("00" * 32))


def test_manager_tls_pin_is_a_noop_without_a_pin(monkeypatch):
    def _boom(addr):  # pragma: no cover - must not be called
        raise AssertionError("no network access without a pin")

    monkeypatch.setattr(mesh_cli.ssl, "get_server_certificate", _boom)
    from verallm.mesh.pool import MeshPoolToken

    mesh_cli._install_manager_tls_pin(
        MeshPoolToken(
            pool_id="p",
            manager_endpoint="http://127.0.0.1:9500",
            pool_secret="s",
        )
    )
