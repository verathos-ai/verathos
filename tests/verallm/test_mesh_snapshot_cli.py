"""Operator CLI coverage for signed, endpoint-free mesh snapshots."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from bittensor_wallet import Keypair

import verallm.chain.wallet as chain_wallet
import verallm.mesh.cli as mesh_cli
import verallm.mesh.receipt_signing as receipt_signing
from verallm.mesh.state import (
    create_mesh_state,
    load_mesh_state,
    save_mesh_state,
)
from verallm.mesh.types import (
    MeshMember,
    MeshSpec,
    StageRange,
    canonical_json_bytes,
)
from verallm.mesh.verification_snapshot import (
    MeshVerificationSnapshot,
    verify_mesh_verification_snapshot_signature,
)


EVM_ADDRESS = "0x" + "ab" * 20


def _keypair(uri: str = "//MeshSnapshotCli") -> Keypair:
    return Keypair.create_from_uri(uri)


def _spec(keypair: Keypair) -> MeshSpec:
    coordinator_stage_key = _keypair("//MeshSnapshotCliCoordinatorStage")
    worker_stage_key = _keypair("//MeshSnapshotCliWorkerStage")
    spec = MeshSpec(
        mesh_id="mesh_snapshot_cli",
        mode="private",
        coordinator_uid=7,
        coordinator_hotkey=keypair.ss58_address,
        model_id="qwen2.5-7b-q4-k-m",
        model_package_hash="a1" * 32,
        model_tensor_manifest_root="b2" * 32,
        tokenizer_hash="c3" * 32,
        quantization_scheme="gguf_q4_k_m",
        activation_dtype="f16",
        proof_trace_manifest_format="compact-raw-v3",
        total_layers=4,
        max_context_len=32_768,
        members=[
            MeshMember(
                uid=7,
                hotkey=keypair.ss58_address,
                endpoint="http://coordinator.internal:9338",
                stage_index=0,
                layers=StageRange(0, 2),
                role="coordinator",
                backend="gguf_stage",
                proof_key=coordinator_stage_key.ss58_address,
                proof_endpoint="http://coordinator.internal:9340",
                rpc_endpoint="coordinator.internal:50052",
                payout_bps=10_000,
            ),
            MeshMember(
                uid=7,
                hotkey=keypair.ss58_address,
                endpoint="http://worker-a.internal:9338",
                stage_index=1,
                layers=StageRange(2, 4),
                role="worker",
                backend="gguf_stage_worker",
                proof_key=worker_stage_key.ss58_address,
                proof_endpoint="http://worker-a.internal:9340",
                rpc_endpoint="worker-a.internal:50052",
            ),
        ],
        epoch=123,
    )
    spec.validate()
    return spec


def _patch_wallet(monkeypatch: pytest.MonkeyPatch, keypair: Keypair) -> None:
    monkeypatch.setattr(
        receipt_signing,
        "load_hotkey_keypair",
        lambda wallet_name, hotkey_name: keypair,
    )
    monkeypatch.setattr(
        receipt_signing,
        "load_hotkey_seed",
        lambda wallet_name, hotkey_name, *, keypair: b"s" * 32,
    )
    monkeypatch.setattr(
        chain_wallet,
        "derive_evm_address",
        lambda seed: EVM_ADDRESS,
    )


def _snapshot_args(state_dir: Path, *extra: str) -> list[str]:
    return [
        "snapshot-create",
        str(state_dir),
        "--wallet-name",
        "miner",
        "--wallet-hotkey",
        "default",
        "--chain-id",
        "945",
        "--netuid",
        "405",
        "--coordinator-evm-address",
        EVM_ADDRESS,
        "--model-index",
        "26",
        "--generation",
        "7",
        "--epoch",
        "123",
        "--issued-at-unix",
        "2000000000",
        "--expires-at-unix",
        "2000000600",
        "--policy-profile",
        "gguf_mesh_q4_k_m",
        "--trace-manifest-format",
        "compact-raw-v3",
        "--base-proof-sample-bps",
        "10000",
        "--organic-decode-sample-bps",
        "10000",
        "--canary-decode-sample-bps",
        "10000",
        "--proof-ops-per-request",
        "1",
        *extra,
    ]


def _load_snapshot(path: Path) -> MeshVerificationSnapshot:
    return MeshVerificationSnapshot.from_dict(json.loads(path.read_text()))


def test_snapshot_create_is_signed_canonical_and_stable_under_endpoint_rerouting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    keypair = _keypair()
    state_dir, _, _ = create_mesh_state(spec=_spec(keypair), root=tmp_path)
    _patch_wallet(monkeypatch, keypair)
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"

    mesh_cli.main(_snapshot_args(state_dir, "--output", str(first_path)))
    first_bytes = first_path.read_bytes()
    first = _load_snapshot(first_path)
    assert first_bytes == canonical_json_bytes(json.loads(first_bytes)) + b"\n"
    assert first.stages[0].proof_key == _keypair(
        "//MeshSnapshotCliCoordinatorStage"
    ).ss58_address
    assert first.stages[1].proof_key == _keypair(
        "//MeshSnapshotCliWorkerStage"
    ).ss58_address
    assert first.model.max_context_len == 32_768
    assert verify_mesh_verification_snapshot_signature(
        first,
        expected_hotkey=keypair.ss58_address,
        expected_epoch=123,
        expected_generation=7,
    )

    state = load_mesh_state(state_dir)
    for index, member in enumerate(state["mesh"]["members"]):
        member["endpoint"] = f"http://rerouted-{index}.private:9440"
        member["proof_endpoint"] = f"http://rerouted-{index}.private:9441"
        member["rpc_endpoint"] = f"rerouted-{index}.private:50053"
    save_mesh_state(state_dir, state)
    mesh_cli.main(_snapshot_args(state_dir, "--output", str(second_path)))

    second = _load_snapshot(second_path)
    # Sr25519 signatures may use fresh signing randomness; the signed body and
    # every endpoint-independent stage commitment must remain identical.
    assert second.to_dict(include_signature=False) == first.to_dict(
        include_signature=False
    )
    assert second.body_hash_hex() == first.body_hash_hex()
    assert second.stage_coverage_hash_hex() == first.stage_coverage_hash_hex()
    assert verify_mesh_verification_snapshot_signature(second)
    serialized = second_path.read_text(encoding="utf-8")
    assert "coordinator.internal" not in serialized
    assert "worker-a.internal" not in serialized
    assert "rerouted-" not in serialized
    assert "http://" not in serialized


def test_snapshot_create_defaults_to_state_directory_and_supports_ttl(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    keypair = _keypair()
    state_dir, _, _ = create_mesh_state(spec=_spec(keypair), root=tmp_path)
    _patch_wallet(monkeypatch, keypair)
    args = _snapshot_args(state_dir)
    expires_index = args.index("--expires-at-unix")
    args[expires_index : expires_index + 2] = ["--ttl-seconds", "90"]

    mesh_cli.main(args)

    output = state_dir / mesh_cli.VERIFICATION_SNAPSHOT_FILE
    snapshot = _load_snapshot(output)
    assert snapshot.expires_at_unix == snapshot.issued_at_unix + 90


def test_snapshot_create_rejects_wrong_wallet_and_evm_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    keypair = _keypair()
    state_dir, _, _ = create_mesh_state(spec=_spec(keypair), root=tmp_path)
    _patch_wallet(monkeypatch, _keypair("//WrongSnapshotWallet"))
    with pytest.raises(SystemExit, match="wallet hotkey"):
        mesh_cli.main(_snapshot_args(state_dir))

    _patch_wallet(monkeypatch, keypair)
    args = _snapshot_args(state_dir)
    address_index = args.index("--coordinator-evm-address") + 1
    args[address_index] = "0x" + "cd" * 20
    with pytest.raises(SystemExit, match="does not match"):
        mesh_cli.main(args)


def test_snapshot_create_rejects_unanchored_model_and_policy_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    keypair = _keypair()
    state_dir, _, _ = create_mesh_state(spec=_spec(keypair), root=tmp_path)
    _patch_wallet(monkeypatch, keypair)
    state = load_mesh_state(state_dir)
    state["mesh"]["model_tensor_manifest_root"] = ""
    save_mesh_state(state_dir, state)
    with pytest.raises(SystemExit, match="model_tensor_manifest_root is required"):
        mesh_cli.main(_snapshot_args(state_dir))

    state["mesh"]["model_tensor_manifest_root"] = "b2" * 32
    save_mesh_state(state_dir, state)
    args = _snapshot_args(state_dir)
    trace_index = args.index("--trace-manifest-format") + 1
    args[trace_index] = "compact"
    with pytest.raises(SystemExit, match="trace_manifest_format"):
        mesh_cli.main(args)


def test_serve_auto_detects_snapshot_and_loader_reloads_each_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    keypair = _keypair()
    state_dir, _, _ = create_mesh_state(spec=_spec(keypair), root=tmp_path)
    _patch_wallet(monkeypatch, keypair)
    mesh_cli.main(_snapshot_args(state_dir))
    captured: dict[str, object] = {}
    monkeypatch.setattr(mesh_cli, "serve_worker", lambda **kwargs: captured.update(kwargs))

    mesh_cli.main(
        [
            "serve",
            "--mesh",
            str(state_dir),
            "--evm-address",
            EVM_ADDRESS,
            "--evm-private-key",
            "11" * 32,
            "--validator-auth",
            "--require-validator-nonce",
        ]
    )

    loader = captured["verification_snapshot_loader"]
    assert callable(loader)
    assert loader().generation == 7
    replacement_args = _snapshot_args(state_dir)
    replacement_args[replacement_args.index("--generation") + 1] = "8"
    mesh_cli.main(replacement_args)
    assert loader().generation == 8


def test_serve_rejects_explicit_malformed_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    keypair = _keypair()
    state_dir, _, _ = create_mesh_state(spec=_spec(keypair), root=tmp_path)
    malformed = tmp_path / "malformed.json"
    malformed.write_text('{"endpoint":"http://worker.private:9338"}\n')
    monkeypatch.setattr(mesh_cli, "serve_worker", lambda **kwargs: None)

    with pytest.raises(SystemExit, match="invalid verification snapshot"):
        mesh_cli.main(
            [
                "serve",
                "--mesh",
                str(state_dir),
                "--verification-snapshot",
                str(malformed),
                "--evm-address",
                EVM_ADDRESS,
                "--evm-private-key",
                "11" * 32,
            ]
        )
