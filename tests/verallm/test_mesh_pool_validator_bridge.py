"""Pool-to-validator bridge tests: explicit anchors, finalization, and privacy."""

from __future__ import annotations

import hashlib
import json
import queue
import secrets
import threading
import time
from pathlib import Path

import pytest
from bittensor_wallet import Keypair

import verallm.chain.wallet as chain_wallet
from verallm.mesh import cli as mesh_cli
import verallm.mesh.pool as mesh_pool
import verallm.mesh.receipt_signing as receipt_signing
import verallm.mesh.verification_snapshot as verification_snapshot
from verallm.mesh.llama_cpp import (
    build_llama_server_command,
    llama_tensor_split_layer_ranges,
)
from verallm.mesh.pool import (
    POOL_ADMIN_TOKEN_FILE,
    POOL_STATE_FILE,
    LocalMeshRunner,
    MeshPoolToken,
    PoolManager,
    PoolWorkerConfig,
    create_pool_state,
    load_pool_token_file,
)
from verallm.mesh.state import create_mesh_state, load_mesh_state, save_mesh_state
from verallm.mesh.types import MeshMember, MeshSpec, StageRange
from verallm.mesh.verification_snapshot import (
    MeshVerificationSnapshot,
    assert_endpoint_free_payload,
    verify_mesh_verification_snapshot_signature,
)


EVM_ADDRESS = "0x" + "ab" * 20
OWNER_ACCOUNT = Keypair.create_from_uri("//PoolValidatorOwner").ss58_address
PACKAGE_HASH = "11" * 32
TENSOR_ROOT = "22" * 32
TOKENIZER_HASH = "33" * 32
MAX_CONTEXT_LEN = 32_768
VERIFICATION_SNAPSHOT_HASH = "44" * 32
PROOF_RECEIPT_ROOT = "55" * 32
MESH_RESPONSE_COMMITMENT = "66" * 32


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


def _valid_validator_final_fields(
    **overrides: object,
) -> dict[str, object]:
    fields: dict[str, object] = {
        "verified": True,
        "receipt_verified": True,
        "receipts": 2,
        "proof_stages": 2,
        "proof_mode": mesh_pool.VERATHOS_GGML_GEMM_PROOF_MODE,
        "proof_receipt_root": PROOF_RECEIPT_ROOT,
        "verification_snapshot_hash": VERIFICATION_SNAPSHOT_HASH,
        "mesh_response_commitment_hash": MESH_RESPONSE_COMMITMENT,
    }
    fields.update(overrides)
    return fields


def _write_validator_allowlist(
    path: Path,
    *hotkeys: str,
    updated_at: int | None = None,
) -> None:
    path.write_text(
        json.dumps(
            {
                "updated_at": int(time.time()) if updated_at is None else updated_at,
                "netuid": 405,
                "validators": [
                    {"uid": index, "hotkey_ss58": hotkey, "stake": 1.0}
                    for index, hotkey in enumerate(hotkeys)
                ],
            }
        ),
        encoding="utf-8",
    )


def _validator_pool(tmp_path: Path) -> tuple[Path, PoolManager, dict[str, str]]:
    shared_state_path = tmp_path / "shared.json"
    shared_state_path.write_text("{}", encoding="utf-8")
    state_dir, _ = create_pool_state(
        tmp_path,
        manager_endpoint="http://127.0.0.1:19500",
        serving_mode="validator",
        owner_account=OWNER_ACCOUNT,
        coordinator_address=EVM_ADDRESS,
        validator_shared_state_path=shared_state_path,
        chain_id=945,
        netuid=405,
        coordinator_uid=1,
        epoch=123,
        snapshot_ttl_seconds=600,
    )
    manager = PoolManager(state_dir)
    # The miner identity the manager persists at serve startup; token-only
    # drivers stamp their mesh specs with exactly this SS58.
    manager.state["coordinator_hotkey_ss58"] = (
        "5ENhc47AqS9NB92K7xUkJ5AhtjQmG75aCYNC8g6qqDpDiXLv"
    )
    admin = load_pool_token_file(state_dir / POOL_ADMIN_TOKEN_FILE)
    return state_dir, manager, {"management_secret": admin.pool_secret}


def _register_model(
    manager: PoolManager,
    auth: dict[str, str],
    *,
    layers: int = 4,
) -> None:
    manager.handle_register_model(
        {
            **auth,
            "model_id": "qwen-test-q4",
            "hf_repo": "operator/model",
            "hf_files": ["model.gguf"],
            "layers": layers,
            "model_bytes": 1024,
            "model_index": 26,
            "model_package_hash": PACKAGE_HASH,
            "model_tensor_manifest_root": TENSOR_ROOT,
            "tokenizer_hash": TOKENIZER_HASH,
            "quantization_scheme": "gguf_q4_k_m",
            "max_context_len": MAX_CONTEXT_LEN,
        }
    )
    # These tests model a model whose deploy COMPLETED the chain
    # registration; only confirmed bindings may launch chain-bound.
    manager.state["model_registry"]["qwen-test-q4"]["chain_committed"] = True


def _join_pool_worker(
    manager: PoolManager,
    *,
    worker_id: str,
    with_model: bool,
    vram_gb: int | None = 24,
    keypair: Keypair | None = None,
) -> Keypair:
    keypair = keypair or _worker_stage_key(worker_id)
    catalog = []
    if with_model:
        # A worker credential must not be able to replace the operator-pinned
        # layer/model anchors with its own advert.
        catalog = [
            {
                "model_id": "qwen-test-q4",
                "model_bytes": 999,
                "layers": 999,
                "hf_repo": "untrusted/advert",
                "hf_files": ["different.gguf"],
            }
        ]
    capability = {
        "gpu_name": "test",
        "subnet_driver_ready": worker_id == "driver",
        "stage_proof_key": keypair.ss58_address,
    }
    if vram_gb is not None:
        capability["vram_gb"] = vram_gb
    manager.handle_join(
        _signed_worker_body(
            manager,
            worker_id=worker_id,
            action="join",
            keypair=keypair,
            fields={
                "capability": capability,
                "catalog": catalog,
                "endpoints": {
                    "rpc": f"{worker_id}.private:50052",
                    "proof": f"http://{worker_id}.private:19402",
                    "mesh": f"http://{worker_id}.private:19443",
                },
            },
        )
    )
    return keypair


def _worker_stage_key(worker_id: str) -> Keypair:
    seed = hashlib.sha256(
        f"pool-validator-worker:{worker_id}".encode("utf-8")
    ).hexdigest()
    return Keypair.create_from_uri(f"//{seed}")


def _signed_worker_body(
    manager: PoolManager,
    *,
    worker_id: str,
    action: str,
    fields: dict[str, object] | None = None,
    keypair: Keypair | None = None,
    timestamp: int | None = None,
    nonce: str | None = None,
    session_id: str | None = None,
) -> dict[str, object]:
    keypair = keypair or _worker_stage_key(worker_id)
    body: dict[str, object] = {
        "pool_secret": manager.state["pool_secret"],
        "worker_id": worker_id,
        "worker_session_id": session_id
        or hashlib.sha256(
            f"pool-validator-session:{worker_id}".encode("utf-8")
        ).hexdigest(),
        **dict(fields or {}),
        "worker_auth_action": action,
        "worker_auth_timestamp": (
            int(time.time()) if timestamp is None else timestamp
        ),
        "worker_auth_nonce": nonce or secrets.token_hex(32),
        "worker_proof_key": keypair.ss58_address,
    }
    body["worker_auth_signature"] = (
        receipt_signing.sign_worker_control_body_hash(
            mesh_pool._worker_control_body_hash_hex(body),
            keypair,
            expected_proof_key=keypair.ss58_address,
        )
    )
    return body


def _launch_serving_validator_mesh(
    manager: PoolManager,
    admin: dict[str, str],
) -> tuple[str, dict[str, Keypair]]:
    _register_model(manager, admin)
    keys = {
        "driver": _join_pool_worker(
            manager,
            worker_id="driver",
            with_model=True,
        ),
        "member": _join_pool_worker(
            manager,
            worker_id="member",
            with_model=False,
        ),
    }
    launched = manager.handle_launch(
        {
            **admin,
            "model_id": "qwen-test-q4",
            "workers": ["driver", "member"],
            "driver": "driver",
        }
    )
    mesh_key = launched["mesh_key"]
    drive = manager.handle_heartbeat(
        _signed_worker_body(
            manager,
            worker_id="driver",
            action="heartbeat",
            keypair=keys["driver"],
        )
    )["command"]
    assert drive["action"] == "drive"
    # The WIRE command must carry the coordinator hotkey: a token-only
    # driver stamps the mesh spec (and receipts verify) against exactly
    # this SS58, and it cannot derive it from any local wallet. A drive
    # rebuild must preserve the field instead of substituting a placeholder.
    assert drive["coordinator_hotkey"] == str(
        manager.state.get("coordinator_hotkey_ss58", "") or ""
    )
    assert manager.handle_report(
        _signed_worker_body(
            manager,
            worker_id="driver",
            action="report",
            keypair=keys["driver"],
            fields={
                "command_id": drive["command_id"],
                "command_digest": drive["command_digest"],
                "mesh_key": mesh_key,
                "event": "drive_ready",
                "mesh_id": "mesh-validator-auth",
                "join_token": "vtmesh_validator_auth",
                "coordinator_endpoint": "http://driver.private:19443",
            },
        )
    ) == {"status": "ok", "command_phase": "drive_ready"}
    join = manager.handle_heartbeat(
        _signed_worker_body(
            manager,
            worker_id="member",
            action="heartbeat",
            keypair=keys["member"],
        )
    )["command"]
    assert join["action"] == "join"
    assert manager.handle_report(
        _signed_worker_body(
            manager,
            worker_id="member",
            action="report",
            keypair=keys["member"],
            fields={
                "command_id": join["command_id"],
                "command_digest": join["command_digest"],
                "mesh_key": mesh_key,
                "event": "serving",
            },
        )
    ) == {"status": "ok", "command_completed": join["command_id"]}
    assert manager.handle_report(
        _signed_worker_body(
            manager,
            worker_id="driver",
            action="report",
            keypair=keys["driver"],
            fields={
                "command_id": drive["command_id"],
                "command_digest": drive["command_digest"],
                "mesh_key": mesh_key,
                "event": "serving",
                "verification_snapshot_hash": VERIFICATION_SNAPSHOT_HASH,
            },
        )
    ) == {"status": "ok", "command_completed": drive["command_id"]}
    assert manager.state["meshes"][mesh_key]["status"] == "serving"
    return mesh_key, keys


def _launch_assigned_validator_mesh(
    manager: PoolManager,
    admin: dict[str, str],
    *,
    driver_with_model: bool = True,
) -> tuple[str, dict[str, Keypair]]:
    _register_model(manager, admin)
    keys = {
        "driver": _join_pool_worker(
            manager,
            worker_id="driver",
            with_model=driver_with_model,
        ),
        "member": _join_pool_worker(
            manager,
            worker_id="member",
            with_model=False,
        ),
    }
    launched = manager.handle_launch(
        {
            **admin,
            "model_id": "qwen-test-q4",
            "workers": ["driver", "member"],
            "driver": "driver",
        }
    )
    return launched["mesh_key"], keys


def _start_leased_chat(
    manager: PoolManager,
    admin: dict[str, str],
    mesh_key: str,
    driver_key: Keypair,
    *,
    prompt: str = "prove this response",
) -> tuple[str, queue.Queue, dict[str, object]]:
    manager.handle_chat_poll(
        _signed_worker_body(
            manager,
            worker_id="driver",
            action="chat-poll",
            keypair=driver_key,
            fields={"wait": 0},
        )
    )
    chat_id, chunks = manager.start_chat_stream(
        {
            **admin,
            "mesh_key": mesh_key,
            "prompt": prompt,
        }
    )
    polled = manager.handle_chat_poll(
        _signed_worker_body(
            manager,
            worker_id="driver",
            action="chat-poll",
            keypair=driver_key,
            fields={"wait": 0},
        )
    )
    assert [item["chat_id"] for item in polled["chat"]] == [chat_id]
    assert manager.chat_contexts[chat_id]["state"] == "leased"
    assert (
        polled["chat"][0]["verification_snapshot_hash"]
        == VERIFICATION_SNAPSHOT_HASH
    )
    return chat_id, chunks, polled["chat"][0]


def _start_running_chat(
    manager: PoolManager,
    admin: dict[str, str],
    mesh_key: str,
    driver_key: Keypair,
) -> tuple[str, queue.Queue]:
    chat_id, chunks, lease = _start_leased_chat(
        manager,
        admin,
        mesh_key,
        driver_key,
    )
    delivery_token = lease["delivery_token"]
    wrong_token = "00" * 32
    if delivery_token == wrong_token:
        wrong_token = "ff" * 32
    assert manager.handle_chat_pickup(
        _signed_worker_body(
            manager,
            worker_id="driver",
            action="chat-pickup",
            keypair=driver_key,
            fields={
                "chat_id": chat_id,
                "delivery_token": delivery_token,
            },
        )
    ) == {"status": "ok"}
    assert manager.chat_contexts[chat_id]["state"] == "running"
    assert all(
        item.get("chat_id") != chat_id
        for item in manager.chat_pending.get("driver", [])
    )
    return chat_id, chunks


def test_validator_pool_requires_complete_explicit_chain_binding(tmp_path: Path) -> None:
    shared_state_path = tmp_path / "shared.json"
    shared_state_path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="coordinator_address"):
        create_pool_state(
            tmp_path,
            manager_endpoint="http://127.0.0.1:19500",
            serving_mode="validator",
            owner_account=OWNER_ACCOUNT,
            validator_shared_state_path=shared_state_path,
            chain_id=945,
            netuid=405,
            coordinator_uid=1,
            epoch=123,
        )
    assert {path.name for path in tmp_path.iterdir()} == {"shared.json"}
    with pytest.raises(ValueError, match="owner_account"):
        create_pool_state(
            tmp_path,
            manager_endpoint="http://127.0.0.1:19500",
            serving_mode="validator",
            coordinator_address=EVM_ADDRESS,
            validator_shared_state_path=shared_state_path,
            chain_id=945,
            netuid=405,
            coordinator_uid=1,
            epoch=123,
        )
    assert {path.name for path in tmp_path.iterdir()} == {"shared.json"}
    with pytest.raises(ValueError, match="must use HTTPS"):
        create_pool_state(
            tmp_path,
            manager_endpoint="http://manager.example:19500",
            serving_mode="validator",
            owner_account=OWNER_ACCOUNT,
            coordinator_address=EVM_ADDRESS,
            validator_shared_state_path=shared_state_path,
            chain_id=945,
            netuid=405,
            coordinator_uid=1,
            epoch=123,
        )
    assert {path.name for path in tmp_path.iterdir()} == {"shared.json"}
    with pytest.raises(ValueError, match="serving_mode"):
        create_pool_state(
            tmp_path,
            manager_endpoint="http://127.0.0.1:19500",
            serving_mode="",
        )
    assert {path.name for path in tmp_path.iterdir()} == {"shared.json"}
    # validator_shared_state_path is optional: it only feeds the operator
    # board's score view, which falls back to the chain score cache when a
    # pool has no validator install on the box. The chain binding itself
    # stays mandatory (guards above).
    pool_dir, _token = create_pool_state(
        tmp_path / "no-shared-state",
        manager_endpoint="http://127.0.0.1:19500",
        serving_mode="validator",
        owner_account=OWNER_ACCOUNT,
        coordinator_address=EVM_ADDRESS,
        chain_id=945,
        netuid=405,
        coordinator_uid=1,
        epoch=123,
    )
    state = json.loads(
        (pool_dir / mesh_pool.POOL_STATE_FILE).read_text(encoding="utf-8")
    )
    assert state["validator_binding"]["epoch"] == 123
    assert "validator_shared_state_path" not in state


def test_validator_pool_identity_cannot_be_rebound_at_manager_start(
    tmp_path: Path,
) -> None:
    state_dir, _manager, _admin = _validator_pool(tmp_path)
    assert PoolManager(
        state_dir,
        coordinator_address=EVM_ADDRESS.upper().replace("0X", "0x"),
    ).coordinator_address == EVM_ADDRESS
    with pytest.raises(ValueError, match="override does not match state"):
        PoolManager(
            state_dir,
            coordinator_address="0x" + "cd" * 20,
        )


def test_validator_pool_rejects_malformed_persisted_chain_binding(
    tmp_path: Path,
) -> None:
    state_dir, _manager, _admin = _validator_pool(tmp_path)
    state_path = state_dir / POOL_STATE_FILE
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["validator_binding"]["epoch"] = None
    state_path.write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(ValueError, match="epoch must be an integer"):
        PoolManager(state_dir)


def test_validator_pool_can_stage_next_epoch_without_mutating_live_mesh(
    tmp_path: Path,
) -> None:
    _state_dir, manager, admin = _validator_pool(tmp_path)
    _register_model(manager, admin)
    _join_pool_worker(manager, worker_id="driver", with_model=True)
    launched = manager.handle_launch(
        {
            **admin,
            "model_id": "qwen-test-q4",
            "workers": ["driver"],
            "driver": "driver",
        }
    )
    mesh_key = launched["mesh_key"]
    assert manager.state["meshes"][mesh_key]["validator_binding"]["epoch"] == 123

    staged = manager.handle_set_epoch({**admin, "epoch": 124})

    assert staged == {
        "status": "staged",
        "previous_epoch": 123,
        "epoch": 124,
        "corrected": False,
    }
    assert manager.validator_binding["epoch"] == 124
    assert manager.state["validator_binding"]["epoch"] == 124
    assert manager.state["meshes"][mesh_key]["validator_binding"]["epoch"] == 123
    with pytest.raises(ValueError, match="differ"):
        manager.handle_set_epoch({**admin, "epoch": 124})


def test_validator_pool_can_correct_unused_future_epoch(
    tmp_path: Path,
) -> None:
    _state_dir, manager, admin = _validator_pool(tmp_path)
    manager.handle_set_epoch({**admin, "epoch": 999})

    corrected = manager.handle_set_epoch({**admin, "epoch": 124})

    assert corrected == {
        "status": "staged",
        "previous_epoch": 999,
        "epoch": 124,
        "corrected": True,
    }
    with pytest.raises(ValueError, match="already used"):
        manager.handle_set_epoch({**admin, "epoch": 123})


def test_validator_status_exposes_staged_and_active_epochs(
    tmp_path: Path,
) -> None:
    _state_dir, manager, admin = _validator_pool(tmp_path)
    _register_model(manager, admin)
    _join_pool_worker(manager, worker_id="driver", with_model=True)
    manager.handle_launch(
        {
            **admin,
            "model_id": "qwen-test-q4",
            "workers": ["driver"],
            "driver": "driver",
        }
    )
    manager.handle_set_epoch({**admin, "epoch": 124})

    status = manager.handle_status(admin)
    overview = manager.handle_operator_overview(admin)

    assert status["validator_binding"]["epoch"] == 124
    assert status["validator_epoch_floor"] == 123
    assert overview["validator_binding"]["epoch"] == 124
    assert overview["active_validator_epochs"] == [123]


def test_validator_overview_only_marks_chain_bound_models_launchable(
    tmp_path: Path,
) -> None:
    _state_dir, manager, admin = _validator_pool(tmp_path)
    _join_pool_worker(manager, worker_id="driver", with_model=True)
    before = manager.handle_operator_overview(admin)
    # legacy "validator" input normalizes to the canonical mode name
    assert before["serving_mode"] == "subnet"
    assert before["models"]["qwen-test-q4"]["launch_ready"] is False

    _register_model(manager, admin)
    after = manager.handle_operator_overview(admin)
    assert after["models"]["qwen-test-q4"]["max_context_len"] == MAX_CONTEXT_LEN
    assert after["models"]["qwen-test-q4"]["launch_ready"] is True
    manager.state["model_registry"]["qwen-test-q4"].pop("max_context_len")
    missing_context = manager.handle_operator_overview(admin)
    assert missing_context["models"]["qwen-test-q4"]["launch_ready"] is False


@pytest.mark.parametrize(
    ("bad_context", "error"),
    [
        pytest.param(None, "require max_context_len", id="missing"),
        pytest.param(True, "positive uint32", id="bool"),
        pytest.param(0, "positive uint32", id="zero"),
        pytest.param(2**32, "positive uint32", id="uint32-overflow"),
    ],
)
def test_validator_model_registration_rejects_untrusted_context_limit(
    tmp_path: Path,
    bad_context: object,
    error: str,
) -> None:
    _state_dir, manager, admin = _validator_pool(tmp_path)
    body = {
        **admin,
        "model_id": "qwen-test-q4",
        "hf_repo": "operator/model",
        "hf_files": ["model.gguf"],
        "layers": 4,
        "model_bytes": 1024,
        "model_index": 26,
        "model_package_hash": PACKAGE_HASH,
        "model_tensor_manifest_root": TENSOR_ROOT,
        "tokenizer_hash": TOKENIZER_HASH,
        "quantization_scheme": "gguf_q4_k_m",
    }
    if bad_context is not None:
        body["max_context_len"] = bad_context

    with pytest.raises(ValueError, match=error):
        manager.handle_register_model(body)
    assert "qwen-test-q4" not in manager.state.get("model_registry", {})


def test_validator_launch_command_is_chain_and_model_bound_without_worker_routes(
    tmp_path: Path,
) -> None:
    _state_dir, manager, admin = _validator_pool(tmp_path)
    _register_model(manager, admin)
    _join_pool_worker(manager, worker_id="driver", with_model=True)
    _join_pool_worker(manager, worker_id="member", with_model=False)

    registry = manager.state["model_registry"]["qwen-test-q4"]
    assert registry["layers"] == 4
    assert registry["hf_repo"] == "operator/model"
    assert registry["max_context_len"] == MAX_CONTEXT_LEN
    with pytest.raises(ValueError, match="workers must be unique"):
        manager.handle_launch(
            {
                **admin,
                "model_id": "qwen-test-q4",
                "workers": ["driver", "driver"],
                "driver": "driver",
            }
        )
    launched = manager.handle_launch(
        {
            **admin,
            "model_id": "qwen-test-q4",
            "workers": ["driver", "member"],
            "driver": "driver",
        }
    )
    command = manager.handle_heartbeat(
        _signed_worker_body(
            manager,
            worker_id="driver",
            action="heartbeat",
        )
    )["command"]
    mesh = manager.state["meshes"][launched["mesh_key"]]

    assert launched["driver"] == "driver"
    assert mesh["max_context_len"] == MAX_CONTEXT_LEN
    assert command["serving_mode"] == "subnet"
    assert command["validator_binding"] == {
        "chain_id": 945,
        "netuid": 405,
        "coordinator_uid": 1,
        "epoch": 123,
        "snapshot_ttl_seconds": 600,
    }
    assert command["model_index"] == 26
    assert command["model_package_hash"] == PACKAGE_HASH
    assert command["model_tensor_manifest_root"] == TENSOR_ROOT
    assert command["tokenizer_hash"] == TOKENIZER_HASH
    assert command["quantization_scheme"] == "gguf_q4_k_m"
    assert command["total_layers"] == 4
    assert command["max_context_len"] == MAX_CONTEXT_LEN
    serialized = json.dumps(command, sort_keys=True)
    assert "driver.private" not in serialized
    assert "member.private" not in serialized


def test_validator_worker_control_lifecycle_uses_distinct_signed_stage_keys(
    tmp_path: Path,
) -> None:
    _state_dir, manager, admin = _validator_pool(tmp_path)
    mesh_key, keys = _launch_serving_validator_mesh(manager, admin)

    assert keys["driver"].ss58_address != keys["member"].ss58_address
    assert (
        manager.state["workers"]["driver"]["worker_proof_key"]
        == keys["driver"].ss58_address
    )
    assert (
        manager.state["workers"]["member"]["worker_proof_key"]
        == keys["member"].ss58_address
    )

    chat_id, chunks = _start_running_chat(
        manager,
        admin,
        mesh_key,
        keys["driver"],
    )
    chunk = manager.push_chat_chunk(
        _signed_worker_body(
            manager,
            worker_id="driver",
            action="chat-chunk",
            keypair=keys["driver"],
            fields={
                "chat_id": chat_id,
                "seq": 1,
                "delta": "hello",
            },
        )
    )
    assert chunk == {"status": "ok", "seq": 1}
    assert chunks.get_nowait() == {"type": "delta", "delta": "hello"}

    final = manager.handle_chat_result(
        _signed_worker_body(
            manager,
            worker_id="driver",
            action="chat-result",
            keypair=keys["driver"],
            fields={
                "chat_id": chat_id,
                "seq": 2,
                "content": "hello",
                **_valid_validator_final_fields(),
            },
        )
    )
    assert final == {"status": "ok"}
    done = chunks.get_nowait()
    assert done["type"] == "done"
    assert done["status"] == "ok"
    assert done["verified"] is True
    assert done["receipt_verified"] is True
    assert done["proof_stages"] == 2
    assert done["expected_stage_count"] == 2
    assert mesh_key not in manager.chat_active


def test_validator_stop_keeps_workers_reserved_until_signed_completion(
    tmp_path: Path,
) -> None:
    _state_dir, manager, admin = _validator_pool(tmp_path)
    mesh_key, keys = _launch_serving_validator_mesh(manager, admin)

    assert manager.handle_stop({**admin, "mesh_key": mesh_key}) == {
        "status": "stopping",
        "mesh_key": mesh_key,
    }
    assert manager.state["meshes"][mesh_key]["status"] == "stopping"
    assert {
        manager.state["workers"][worker_id]["status"]
        for worker_id in ("driver", "member")
    } == {"stopping"}

    stop_commands: dict[str, dict[str, object]] = {}
    for worker_id in ("driver", "member"):
        command = manager.handle_heartbeat(
            _signed_worker_body(
                manager,
                worker_id=worker_id,
                action="heartbeat",
                keypair=keys[worker_id],
            )
        )["command"]
        assert command["action"] == "stop"
        stop_commands[worker_id] = command

    first = stop_commands["driver"]
    assert manager.handle_report(
        _signed_worker_body(
            manager,
            worker_id="driver",
            action="report",
            keypair=keys["driver"],
            fields={
                "command_id": first["command_id"],
                "command_digest": first["command_digest"],
                "mesh_key": mesh_key,
                "event": "stopped",
            },
        )
    ) == {"status": "ok", "command_completed": first["command_id"]}
    assert manager.state["workers"]["driver"]["status"] == "idle"
    assert manager.state["workers"]["member"]["status"] == "stopping"
    assert manager.state["meshes"][mesh_key]["status"] == "stopping"

    second = stop_commands["member"]
    assert manager.handle_report(
        _signed_worker_body(
            manager,
            worker_id="member",
            action="report",
            keypair=keys["member"],
            fields={
                "command_id": second["command_id"],
                "command_digest": second["command_digest"],
                "mesh_key": mesh_key,
                "event": "stopped",
            },
        )
    ) == {"status": "ok", "command_completed": second["command_id"]}
    assert mesh_key not in manager.state["meshes"]
    assert manager.state["workers"]["driver"]["mesh"] == ""
    assert manager.state["workers"]["member"]["mesh"] == ""


def test_validator_worker_control_rejects_wrong_identity_action_age_and_replay(
    tmp_path: Path,
) -> None:
    _state_dir, manager, _admin = _validator_pool(tmp_path)
    driver_key = _join_pool_worker(
        manager,
        worker_id="driver",
        with_model=True,
    )
    member_key = _join_pool_worker(
        manager,
        worker_id="member",
        with_model=False,
    )

    with pytest.raises(PermissionError, match="stage identity"):
        manager.handle_heartbeat(
            _signed_worker_body(
                manager,
                worker_id="driver",
                action="heartbeat",
                keypair=member_key,
            )
        )
    with pytest.raises(PermissionError, match="stage identity"):
        manager.handle_report(
            _signed_worker_body(
                manager,
                worker_id="member",
                action="report",
                keypair=driver_key,
                fields={"mesh_key": "missing", "event": "serving"},
            )
        )
    with pytest.raises(PermissionError, match="action mismatch"):
        manager.handle_heartbeat(
            _signed_worker_body(
                manager,
                worker_id="driver",
                action="report",
                keypair=driver_key,
            )
        )
    with pytest.raises(PermissionError, match="timestamp is stale"):
        manager.handle_heartbeat(
            _signed_worker_body(
                manager,
                worker_id="driver",
                action="heartbeat",
                keypair=driver_key,
                timestamp=(
                    int(time.time())
                    - mesh_pool.WORKER_CONTROL_MAX_CLOCK_SKEW_S
                    - 1
                ),
            )
        )

    replay = _signed_worker_body(
        manager,
        worker_id="driver",
        action="heartbeat",
        keypair=driver_key,
        nonce="ab" * 32,
    )
    assert manager.handle_heartbeat(replay)["status"] == "ok"
    with pytest.raises(PermissionError, match="replayed"):
        manager.handle_heartbeat(replay)

    unsigned_join = {
        "pool_secret": manager.state["pool_secret"],
        "worker_id": "unsigned",
        "worker_session_id": "11" * 32,
        "capability": {"stage_proof_key": driver_key.ss58_address},
    }
    with pytest.raises(PermissionError, match="action mismatch"):
        manager.handle_join(unsigned_join)


def test_validator_worker_join_rejects_key_substitution_and_signed_body_tampering(
    tmp_path: Path,
) -> None:
    _state_dir, manager, _admin = _validator_pool(tmp_path)
    driver_key = _worker_stage_key("driver")
    substitute_key = _worker_stage_key("substitute")

    mismatched_capability = _signed_worker_body(
        manager,
        worker_id="driver",
        action="join",
        keypair=driver_key,
        fields={
            "capability": {
                "stage_proof_key": substitute_key.ss58_address,
            },
        },
    )
    with pytest.raises(PermissionError, match="capability stage identity"):
        manager.handle_join(mismatched_capability)
    assert "driver" not in manager.state["workers"]

    _join_pool_worker(
        manager,
        worker_id="driver",
        with_model=True,
        keypair=driver_key,
    )
    substituted_rejoin = _signed_worker_body(
        manager,
        worker_id="driver",
        action="join",
        keypair=substitute_key,
        fields={
            "capability": {
                "stage_proof_key": substitute_key.ss58_address,
            },
        },
    )
    with pytest.raises(PermissionError, match="stage identity"):
        manager.handle_join(substituted_rejoin)
    assert (
        manager.state["workers"]["driver"]["worker_proof_key"]
        == driver_key.ss58_address
    )

    tampered_heartbeat = _signed_worker_body(
        manager,
        worker_id="driver",
        action="heartbeat",
        keypair=driver_key,
    )
    tampered_heartbeat["status"] = "serving"
    with pytest.raises(PermissionError, match="signature is invalid"):
        manager.handle_heartbeat(tampered_heartbeat)


def test_concurrent_first_join_uses_atomic_stage_key_compare_and_set(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _state_dir, manager, _admin = _validator_pool(tmp_path)
    first_key = _worker_stage_key("racing-first")
    second_key = _worker_stage_key("racing-second")
    bodies = [
        _signed_worker_body(
            manager,
            worker_id="racing-worker",
            action="join",
            keypair=keypair,
            fields={
                "capability": {"stage_proof_key": keypair.ss58_address},
            },
        )
        for keypair in (first_key, second_key)
    ]

    original_auth_worker = manager._auth_worker
    both_authenticated = threading.Barrier(2)

    def synchronized_auth_worker(body, **kwargs):
        authenticated = original_auth_worker(body, **kwargs)
        if kwargs.get("action") == "join":
            both_authenticated.wait(timeout=5)
        return authenticated

    monkeypatch.setattr(manager, "_auth_worker", synchronized_auth_worker)
    outcomes: queue.Queue = queue.Queue()

    def attempt_join(body: dict[str, object]) -> None:
        try:
            outcomes.put(("ok", manager.handle_join(body)))
        except Exception as exc:
            outcomes.put(("error", exc))

    threads = [
        threading.Thread(target=attempt_join, args=(body,)) for body in bodies
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=6)
        assert not thread.is_alive()

    results = [outcomes.get_nowait(), outcomes.get_nowait()]
    successes = [value for status, value in results if status == "ok"]
    failures = [value for status, value in results if status == "error"]
    assert len(successes) == 1
    assert len(failures) == 1
    assert isinstance(failures[0], PermissionError)
    assert "stage identity does not match registration" in str(failures[0])
    assert manager.state["workers"]["racing-worker"]["worker_proof_key"] in {
        first_key.ss58_address,
        second_key.ss58_address,
    }


def test_validator_stage_key_has_one_worker_owner(tmp_path: Path) -> None:
    _state_dir, manager, _admin = _validator_pool(tmp_path)
    shared_key = _worker_stage_key("shared-stage-key")
    _join_pool_worker(
        manager,
        worker_id="first-worker",
        with_model=False,
        keypair=shared_key,
    )

    second_owner = _signed_worker_body(
        manager,
        worker_id="second-worker",
        action="join",
        keypair=shared_key,
        fields={
            "capability": {"stage_proof_key": shared_key.ss58_address},
        },
    )
    with pytest.raises(PermissionError, match="already registered"):
        manager.handle_join(second_owner)

    assert "second-worker" not in manager.state["workers"]
    assert (
        manager.state["workers"]["first-worker"]["worker_proof_key"]
        == shared_key.ss58_address
    )


def test_validator_report_rejects_nonmember_and_wrong_mesh_assignment(
    tmp_path: Path,
) -> None:
    _state_dir, manager, admin = _validator_pool(tmp_path)
    mesh_key, keys = _launch_assigned_validator_mesh(manager, admin)
    outsider_key = _join_pool_worker(
        manager,
        worker_id="outsider",
        with_model=False,
    )
    state_before_nonmember = json.dumps(manager.state, sort_keys=True)

    assert manager.handle_report(
        _signed_worker_body(
            manager,
            worker_id="outsider",
            action="report",
            keypair=outsider_key,
            fields={
                "command_id": "cmd-" + "11" * 16,
                "command_digest": "22" * 32,
                "mesh_key": mesh_key,
                "event": "error",
                "message": "must not mutate another mesh",
            },
        )
    ) == {"status": "stale", "command_id": "cmd-" + "11" * 16}
    assert json.dumps(manager.state, sort_keys=True) == state_before_nonmember

    other_mesh_key = "m-known-but-not-assigned"
    manager.state["meshes"][other_mesh_key] = {
        **manager.state["meshes"][mesh_key],
        "mesh_key": other_mesh_key,
        "members": ["member"],
        "driver": "member",
    }
    drive = manager.handle_heartbeat(
        _signed_worker_body(
            manager,
            worker_id="driver",
            action="heartbeat",
            keypair=keys["driver"],
        )
    )["command"]
    state_before_wrong_mesh = json.dumps(manager.state, sort_keys=True)
    with pytest.raises(PermissionError, match="mesh does not match its command"):
        manager.handle_report(
            _signed_worker_body(
                manager,
                worker_id="driver",
                action="report",
                keypair=keys["driver"],
                fields={
                    "command_id": drive["command_id"],
                    "command_digest": drive["command_digest"],
                    "mesh_key": other_mesh_key,
                    "event": "error",
                    "message": "stale topology must not grant report authority",
                },
            )
        )
    assert json.dumps(manager.state, sort_keys=True) == state_before_wrong_mesh

    assert manager.state["meshes"][mesh_key]["status"] == "driving"
    assert manager.state["meshes"][other_mesh_key]["status"] == "driving"


@pytest.mark.parametrize(
    ("event", "driver_with_model", "event_fields", "initial_status"),
    [
        pytest.param(
            "drive_ready",
            True,
            {
                "mesh_id": "forged-member-runtime",
                "join_token": "vtmesh_forged_member",
                "coordinator_endpoint": "http://member.private:19443",
            },
            "driving",
            id="member-drive-ready",
        ),
        pytest.param(
            "fetched",
            False,
            {
                "entry": {
                    "model_id": "qwen-test-q4",
                    "llama_model": "/untrusted/member/model.gguf",
                },
            },
            "fetching",
            id="member-fetched",
        ),
    ],
)
def test_validator_member_cannot_inject_driver_only_lifecycle_events(
    tmp_path: Path,
    event: str,
    driver_with_model: bool,
    event_fields: dict[str, object],
    initial_status: str,
) -> None:
    _state_dir, manager, admin = _validator_pool(tmp_path)
    mesh_key, keys = _launch_assigned_validator_mesh(
        manager,
        admin,
        driver_with_model=driver_with_model,
    )
    assert manager.state["meshes"][mesh_key]["status"] == initial_status
    state_before = json.dumps(manager.state, sort_keys=True)

    assert manager.handle_report(
        _signed_worker_body(
            manager,
            worker_id="member",
            action="report",
            keypair=keys["member"],
            fields={
                "command_id": "cmd-" + "33" * 16,
                "command_digest": "44" * 32,
                "mesh_key": mesh_key,
                "event": event,
                **event_fields,
            },
        )
    ) == {"status": "stale", "command_id": "cmd-" + "33" * 16}

    assert json.dumps(manager.state, sort_keys=True) == state_before
    assert manager.state["meshes"][mesh_key]["status"] == initial_status


def test_validator_driver_rejects_inconsistent_duplicate_drive_ready(
    tmp_path: Path,
) -> None:
    _state_dir, manager, admin = _validator_pool(tmp_path)
    mesh_key, keys = _launch_assigned_validator_mesh(manager, admin)
    drive = manager.handle_heartbeat(
        _signed_worker_body(
            manager,
            worker_id="driver",
            action="heartbeat",
            keypair=keys["driver"],
        )
    )["command"]
    first_runtime = {
        "command_id": drive["command_id"],
        "command_digest": drive["command_digest"],
        "mesh_key": mesh_key,
        "event": "drive_ready",
        "mesh_id": "mesh-runtime-one",
        "join_token": "vtmesh_runtime_one",
        "coordinator_endpoint": "http://driver.private:19443",
    }
    assert manager.handle_report(
        _signed_worker_body(
            manager,
            worker_id="driver",
            action="report",
            keypair=keys["driver"],
            fields=first_runtime,
        )
    ) == {"status": "ok", "command_phase": "drive_ready"}
    member_commands = list(manager.state["workers"]["member"]["commands"])

    assert manager.handle_report(
        _signed_worker_body(
            manager,
            worker_id="driver",
            action="report",
            keypair=keys["driver"],
            fields=first_runtime,
        )
    ) == {
        "status": "ok",
        "command_phase": "drive_ready",
        "duplicate": True,
    }
    assert manager.state["workers"]["member"]["commands"] == member_commands

    with pytest.raises(ValueError, match="conflicting retry"):
        manager.handle_report(
            _signed_worker_body(
                manager,
                worker_id="driver",
                action="report",
                keypair=keys["driver"],
                fields={
                    **first_runtime,
                    "mesh_id": "mesh-runtime-two",
                },
            )
        )

    mesh = manager.state["meshes"][mesh_key]
    assert mesh["mesh_id"] == "mesh-runtime-one"
    assert mesh["join_token"] == "vtmesh_runtime_one"
    assert mesh["coordinator_endpoint"] == "http://driver.private:19443"
    assert manager.state["workers"]["member"]["commands"] == member_commands


def test_validator_drive_ready_retry_after_manager_restart_is_idempotent(
    tmp_path: Path,
) -> None:
    state_dir, manager, admin = _validator_pool(tmp_path)
    mesh_key, keys = _launch_assigned_validator_mesh(manager, admin)
    drive = manager.handle_heartbeat(
        _signed_worker_body(
            manager,
            worker_id="driver",
            action="heartbeat",
            keypair=keys["driver"],
        )
    )["command"]
    drive_ready = {
        "command_id": drive["command_id"],
        "command_digest": drive["command_digest"],
        "mesh_key": mesh_key,
        "event": "drive_ready",
        "mesh_id": "mesh-restart-retry",
        "join_token": "vtmesh_restart_retry",
        "coordinator_endpoint": "http://driver.private:19443",
    }
    assert manager.handle_report(
        _signed_worker_body(
            manager,
            worker_id="driver",
            action="report",
            keypair=keys["driver"],
            fields=drive_ready,
        )
    ) == {"status": "ok", "command_phase": "drive_ready"}
    queued_before_restart = list(
        manager.state["workers"]["member"]["commands"]
    )

    restarted = PoolManager(state_dir)
    assert restarted.handle_report(
        _signed_worker_body(
            restarted,
            worker_id="driver",
            action="report",
            keypair=keys["driver"],
            fields=drive_ready,
        )
    ) == {
        "status": "ok",
        "command_phase": "drive_ready",
        "duplicate": True,
    }
    assert restarted.state["workers"]["member"]["commands"] == (
        queued_before_restart
    )
    assert "verification_snapshot_hash" not in restarted.state["meshes"][
        mesh_key
    ]


def test_validator_multibox_publishes_snapshot_only_after_final_admission(
    tmp_path: Path,
) -> None:
    _state_dir, manager, admin = _validator_pool(tmp_path)
    mesh_key, keys = _launch_assigned_validator_mesh(manager, admin)
    drive = manager.handle_heartbeat(
        _signed_worker_body(
            manager,
            worker_id="driver",
            action="heartbeat",
            keypair=keys["driver"],
        )
    )["command"]
    drive_ready = {
        "command_id": drive["command_id"],
        "command_digest": drive["command_digest"],
        "mesh_key": mesh_key,
        "event": "drive_ready",
        "mesh_id": "mesh-final-admission",
        "join_token": "vtmesh_final_admission",
        "coordinator_endpoint": "http://driver.private:19443",
    }

    with pytest.raises(ValueError, match="must defer"):
        manager.handle_report(
            _signed_worker_body(
                manager,
                worker_id="driver",
                action="report",
                keypair=keys["driver"],
                fields={
                    **drive_ready,
                    "verification_snapshot_hash": VERIFICATION_SNAPSHOT_HASH,
                },
            )
        )

    assert manager.handle_report(
        _signed_worker_body(
            manager,
            worker_id="driver",
            action="report",
            keypair=keys["driver"],
            fields=drive_ready,
        )
    ) == {"status": "ok", "command_phase": "drive_ready"}
    assert "verification_snapshot_hash" not in manager.state["meshes"][mesh_key]

    join = manager.handle_heartbeat(
        _signed_worker_body(
            manager,
            worker_id="member",
            action="heartbeat",
            keypair=keys["member"],
        )
    )["command"]
    assert manager.handle_report(
        _signed_worker_body(
            manager,
            worker_id="member",
            action="report",
            keypair=keys["member"],
            fields={
                "command_id": join["command_id"],
                "command_digest": join["command_digest"],
                "mesh_key": mesh_key,
                "event": "serving",
            },
        )
    ) == {"status": "ok", "command_completed": join["command_id"]}

    driver_serving = {
        "command_id": drive["command_id"],
        "command_digest": drive["command_digest"],
        "mesh_key": mesh_key,
        "event": "serving",
    }
    with pytest.raises(ValueError, match="serving requires"):
        manager.handle_report(
            _signed_worker_body(
                manager,
                worker_id="driver",
                action="report",
                keypair=keys["driver"],
                fields=driver_serving,
            )
        )
    assert manager.state["meshes"][mesh_key]["status"] == "joining"
    assert "verification_snapshot_hash" not in manager.state["meshes"][mesh_key]

    assert manager.handle_report(
        _signed_worker_body(
            manager,
            worker_id="driver",
            action="report",
            keypair=keys["driver"],
            fields={
                **driver_serving,
                "verification_snapshot_hash": VERIFICATION_SNAPSHOT_HASH,
            },
        )
    ) == {"status": "ok", "command_completed": drive["command_id"]}
    mesh = manager.state["meshes"][mesh_key]
    assert mesh["status"] == "serving"
    assert mesh["verification_snapshot_hash"] == VERIFICATION_SNAPSHOT_HASH


def test_validator_chat_poll_cannot_be_stolen_by_a_non_driver(
    tmp_path: Path,
) -> None:
    _state_dir, manager, admin = _validator_pool(tmp_path)
    mesh_key, keys = _launch_serving_validator_mesh(manager, admin)
    manager.handle_chat_poll(
        _signed_worker_body(
            manager,
            worker_id="driver",
            action="chat-poll",
            keypair=keys["driver"],
            fields={"wait": 0},
        )
    )
    chat_id, _chunks = manager.start_chat_stream(
        {**admin, "mesh_key": mesh_key, "prompt": "private operator prompt"}
    )

    stolen = manager.handle_chat_poll(
        _signed_worker_body(
            manager,
            worker_id="member",
            action="chat-poll",
            keypair=keys["member"],
            fields={"wait": 0},
        )
    )
    assert stolen["chat"] == []
    assert [
        item["chat_id"]
        for item in manager.chat_pending.get("driver", [])
    ] == [chat_id]

    assigned = manager.handle_chat_poll(
        _signed_worker_body(
            manager,
            worker_id="driver",
            action="chat-poll",
            keypair=keys["driver"],
            fields={"wait": 0},
        )
    )
    assert [item["chat_id"] for item in assigned["chat"]] == [chat_id]
    assert manager.chat_contexts[chat_id]["driver"] == "driver"
    assert manager.chat_contexts[chat_id]["state"] == "leased"
    assert assigned["chat"][0]["delivery_token"]


def test_lost_poll_response_releases_prompt_with_a_fresh_delivery_token(
    tmp_path: Path,
) -> None:
    _state_dir, manager, admin = _validator_pool(tmp_path)
    mesh_key, keys = _launch_serving_validator_mesh(manager, admin)
    prompt = "lost-poll-response-must-not-lose-this-prompt"
    chat_id, _chunks, first_lease = _start_leased_chat(
        manager,
        admin,
        mesh_key,
        keys["driver"],
        prompt=prompt,
    )
    first_token = str(first_lease["delivery_token"])
    assert first_lease["delivery_attempt"] == 1
    assert manager.chat_contexts[chat_id]["state"] == "leased"

    # Model a successful manager write followed by a lost HTTP response: the
    # driver never acknowledges this lease, so it expires while the in-memory
    # prompt remains queued for delivery.
    manager.chat_contexts[chat_id]["delivery_lease_deadline_mono"] = (
        time.monotonic() - 1.0
    )
    repolled = manager.handle_chat_poll(
        _signed_worker_body(
            manager,
            worker_id="driver",
            action="chat-poll",
            keypair=keys["driver"],
            fields={"wait": 0},
        )
    )
    assert [item["chat_id"] for item in repolled["chat"]] == [chat_id]
    second_lease = repolled["chat"][0]
    assert second_lease["delivery_attempt"] == 2
    assert second_lease["delivery_token"] != first_token
    assert second_lease["messages"][0]["content"] == prompt
    assert [item["chat_id"] for item in manager.chat_pending["driver"]] == [chat_id]


def test_expired_delivery_token_is_stale_but_released_token_can_be_picked_up(
    tmp_path: Path,
) -> None:
    _state_dir, manager, admin = _validator_pool(tmp_path)
    mesh_key, keys = _launch_serving_validator_mesh(manager, admin)
    chat_id, _chunks, first_lease = _start_leased_chat(
        manager,
        admin,
        mesh_key,
        keys["driver"],
    )
    manager.chat_contexts[chat_id]["delivery_lease_deadline_mono"] = (
        time.monotonic() - 1.0
    )
    repolled = manager.handle_chat_poll(
        _signed_worker_body(
            manager,
            worker_id="driver",
            action="chat-poll",
            keypair=keys["driver"],
            fields={"wait": 0},
        )
    )
    second_lease = repolled["chat"][0]

    stale = manager.handle_chat_pickup(
        _signed_worker_body(
            manager,
            worker_id="driver",
            action="chat-pickup",
            keypair=keys["driver"],
            fields={
                "chat_id": chat_id,
                "delivery_token": first_lease["delivery_token"],
            },
        )
    )
    assert stale == {"status": "stale"}
    assert manager.chat_contexts[chat_id]["state"] == "leased"

    accepted = manager.handle_chat_pickup(
        _signed_worker_body(
            manager,
            worker_id="driver",
            action="chat-pickup",
            keypair=keys["driver"],
            fields={
                "chat_id": chat_id,
                "delivery_token": second_lease["delivery_token"],
            },
        )
    )
    assert accepted == {"status": "ok"}
    assert manager.chat_contexts[chat_id]["state"] == "running"
    assert manager.chat_pending.get("driver", []) == []


def test_request_deadline_releases_unacknowledged_chat_immediately(
    tmp_path: Path,
) -> None:
    _state_dir, manager, admin = _validator_pool(tmp_path)
    mesh_key, keys = _launch_serving_validator_mesh(manager, admin)
    chat_id, chunks, lease = _start_leased_chat(
        manager,
        admin,
        mesh_key,
        keys["driver"],
    )
    manager.chat_contexts[chat_id]["request_deadline_mono"] = (
        time.monotonic() - 1.0
    )

    assert manager.handle_chat_pickup(
        _signed_worker_body(
            manager,
            worker_id="driver",
            action="chat-pickup",
            keypair=keys["driver"],
            fields={
                "chat_id": chat_id,
                "delivery_token": lease["delivery_token"],
            },
        )
    ) == {"status": "expired"}
    assert chat_id not in manager.chat_contexts
    assert manager.chat_pending.get("driver", []) == []
    assert mesh_key not in manager.chat_active
    assert chunks.get_nowait() == {
        "type": "error",
        "error": "mesh request expired before the driver acknowledged pickup",
    }


def test_running_chat_deadline_rejects_more_chunks_but_allows_cleanup(
    tmp_path: Path,
) -> None:
    _state_dir, manager, admin = _validator_pool(tmp_path)
    mesh_key, keys = _launch_serving_validator_mesh(manager, admin)
    chat_id, chunks = _start_running_chat(
        manager,
        admin,
        mesh_key,
        keys["driver"],
    )
    manager.chat_contexts[chat_id]["request_deadline_mono"] = (
        time.monotonic() - 1.0
    )

    assert manager.push_chat_chunk(
        _signed_worker_body(
            manager,
            worker_id="driver",
            action="chat-chunk",
            keypair=keys["driver"],
            fields={"chat_id": chat_id, "seq": 1, "delta": "too late"},
        )
    ) == {"status": "expired"}
    assert chunks.get_nowait() == {
        "type": "error",
        "error": "mesh request exceeded its operator deadline",
    }
    assert chat_id in manager.chat_active[mesh_key]

    assert manager.handle_chat_result(
        _signed_worker_body(
            manager,
            worker_id="driver",
            action="chat-result",
            keypair=keys["driver"],
            fields={
                "chat_id": chat_id,
                "seq": 1,
                "error": "operator request deadline expired",
            },
        )
    ) == {"status": "ok"}
    assert mesh_key not in manager.chat_active
    assert chunks.get_nowait()["status"] == "error"


def test_late_verified_final_is_downgraded_to_an_error(tmp_path: Path) -> None:
    _state_dir, manager, admin = _validator_pool(tmp_path)
    mesh_key, keys = _launch_serving_validator_mesh(manager, admin)
    chat_id, chunks = _start_running_chat(
        manager,
        admin,
        mesh_key,
        keys["driver"],
    )
    manager.chat_contexts[chat_id]["request_deadline_mono"] = (
        time.monotonic() - 1.0
    )

    assert manager.handle_chat_result(
        _signed_worker_body(
            manager,
            worker_id="driver",
            action="chat-result",
            keypair=keys["driver"],
            fields={
                "chat_id": chat_id,
                "seq": 1,
                "content": "late success",
                **_valid_validator_final_fields(),
            },
        )
    ) == {"status": "ok"}
    done = chunks.get_nowait()
    assert done["status"] == "error"
    assert done["verified"] is False
    assert "after the operator request deadline" in done["error"]


def test_manager_restart_requires_a_fresh_chat_poll(tmp_path: Path) -> None:
    state_dir, manager, admin = _validator_pool(tmp_path)
    mesh_key, keys = _launch_serving_validator_mesh(manager, admin)
    manager.handle_chat_poll(
        _signed_worker_body(
            manager,
            worker_id="driver",
            action="chat-poll",
            keypair=keys["driver"],
            fields={"wait": 0},
        )
    )
    assert manager.state["workers"]["driver"]["last_chat_poll_unix"] > 0
    manager._save()

    restarted = PoolManager(state_dir)

    assert "last_chat_poll_unix" not in restarted.state["workers"]["driver"]
    with pytest.raises(ValueError, match="not picking up chats"):
        restarted.start_chat_stream(
            {**admin, "mesh_key": mesh_key, "prompt": "must not be stranded"}
        )
    assert restarted.chat_pending == {}


def test_duplicate_chat_pickup_ack_is_idempotent(tmp_path: Path) -> None:
    _state_dir, manager, admin = _validator_pool(tmp_path)
    mesh_key, keys = _launch_serving_validator_mesh(manager, admin)
    chat_id, _chunks, lease = _start_leased_chat(
        manager,
        admin,
        mesh_key,
        keys["driver"],
    )
    pickup_fields = {
        "chat_id": chat_id,
        "delivery_token": lease["delivery_token"],
    }

    assert manager.handle_chat_pickup(
        _signed_worker_body(
            manager,
            worker_id="driver",
            action="chat-pickup",
            keypair=keys["driver"],
            fields=pickup_fields,
        )
    ) == {"status": "ok"}
    assert manager.handle_chat_pickup(
        _signed_worker_body(
            manager,
            worker_id="driver",
            action="chat-pickup",
            keypair=keys["driver"],
            fields=pickup_fields,
        )
    ) == {"status": "ok", "duplicate": True}
    assert manager.chat_contexts[chat_id]["state"] == "running"
    assert manager.chat_pending.get("driver", []) == []


def test_chat_pickup_rejects_wrong_driver_token_and_signed_action(
    tmp_path: Path,
) -> None:
    _state_dir, manager, admin = _validator_pool(tmp_path)
    mesh_key, keys = _launch_serving_validator_mesh(manager, admin)
    chat_id, _chunks, lease = _start_leased_chat(
        manager,
        admin,
        mesh_key,
        keys["driver"],
    )
    delivery_token = lease["delivery_token"]
    wrong_token = "00" * 32
    if delivery_token == wrong_token:
        wrong_token = "ff" * 32

    with pytest.raises(PermissionError, match="assigned driver"):
        manager.handle_chat_pickup(
            _signed_worker_body(
                manager,
                worker_id="member",
                action="chat-pickup",
                keypair=keys["member"],
                fields={
                    "chat_id": chat_id,
                    "delivery_token": delivery_token,
                },
            )
        )
    assert manager.handle_chat_pickup(
        _signed_worker_body(
            manager,
            worker_id="driver",
            action="chat-pickup",
            keypair=keys["driver"],
            fields={
                "chat_id": chat_id,
                "delivery_token": wrong_token,
            },
        )
    ) == {"status": "stale"}
    with pytest.raises(PermissionError, match="action mismatch"):
        manager.handle_chat_pickup(
            _signed_worker_body(
                manager,
                worker_id="driver",
                action="chat-poll",
                keypair=keys["driver"],
                fields={
                    "chat_id": chat_id,
                    "delivery_token": delivery_token,
                },
            )
        )

    assert manager.chat_contexts[chat_id]["state"] == "leased"
    assert manager.handle_chat_pickup(
        _signed_worker_body(
            manager,
            worker_id="driver",
            action="chat-pickup",
            keypair=keys["driver"],
            fields={
                "chat_id": chat_id,
                "delivery_token": delivery_token,
            },
        )
    ) == {"status": "ok"}


def test_chat_chunks_and_results_are_stale_until_pickup_is_acknowledged(
    tmp_path: Path,
) -> None:
    _state_dir, manager, admin = _validator_pool(tmp_path)
    mesh_key, keys = _launch_serving_validator_mesh(manager, admin)
    manager.handle_chat_poll(
        _signed_worker_body(
            manager,
            worker_id="driver",
            action="chat-poll",
            keypair=keys["driver"],
            fields={"wait": 0},
        )
    )
    chat_id, chunks = manager.start_chat_stream(
        {**admin, "mesh_key": mesh_key, "prompt": "do not run before pickup"}
    )

    def push_early_events() -> None:
        assert manager.push_chat_chunk(
            _signed_worker_body(
                manager,
                worker_id="driver",
                action="chat-chunk",
                keypair=keys["driver"],
                fields={"chat_id": chat_id, "seq": 1, "delta": "forged"},
            )
        ) == {"status": "stale"}
        assert manager.handle_chat_result(
            _signed_worker_body(
                manager,
                worker_id="driver",
                action="chat-result",
                keypair=keys["driver"],
                fields={
                    "chat_id": chat_id,
                    "seq": 1,
                    "content": "forged",
                    "verified": True,
                    "receipt_verified": True,
                    "proof_stages": 2,
                },
            )
        ) == {"status": "stale"}

    assert manager.chat_contexts[chat_id]["state"] == "queued"
    push_early_events()
    leased = manager.handle_chat_poll(
        _signed_worker_body(
            manager,
            worker_id="driver",
            action="chat-poll",
            keypair=keys["driver"],
            fields={"wait": 0},
        )
    )
    assert [item["chat_id"] for item in leased["chat"]] == [chat_id]
    assert manager.chat_contexts[chat_id]["state"] == "leased"
    push_early_events()

    assert chunks.empty()
    assert manager.chat_contexts[chat_id]["last_seq"] == 0
    assert [item["chat_id"] for item in manager.chat_pending["driver"]] == [chat_id]


def test_operator_prompt_is_never_persisted_in_pool_state(tmp_path: Path) -> None:
    state_dir, manager, admin = _validator_pool(tmp_path)
    mesh_key, keys = _launch_serving_validator_mesh(manager, admin)
    prompt = "SENTINEL_OPERATOR_PROMPT_MUST_REMAIN_MEMORY_ONLY_9cfe"
    chat_id, _chunks, _lease = _start_leased_chat(
        manager,
        admin,
        mesh_key,
        keys["driver"],
        prompt=prompt,
    )
    assert prompt in json.dumps(manager.chat_pending, sort_keys=True)
    assert prompt not in json.dumps(manager.state, sort_keys=True)

    # Exercise an unrelated persistence write while the prompt is leased.  The
    # pending request must remain manager-ephemeral and absent from disk.
    manager._save()
    persisted = (state_dir / POOL_STATE_FILE).read_text(encoding="utf-8")
    assert prompt not in persisted
    assert "chat_pending" not in persisted
    assert manager.chat_contexts[chat_id]["state"] == "leased"


def test_validator_chat_events_are_bound_to_the_assigned_driver(
    tmp_path: Path,
) -> None:
    _state_dir, manager, admin = _validator_pool(tmp_path)
    mesh_key, keys = _launch_serving_validator_mesh(manager, admin)
    chat_id, chunks = _start_running_chat(
        manager,
        admin,
        mesh_key,
        keys["driver"],
    )

    with pytest.raises(PermissionError, match="assigned driver"):
        manager.push_chat_chunk(
            _signed_worker_body(
                manager,
                worker_id="member",
                action="chat-chunk",
                keypair=keys["member"],
                fields={"chat_id": chat_id, "seq": 1, "delta": "forged"},
            )
        )
    with pytest.raises(PermissionError, match="assigned driver"):
        manager.handle_chat_result(
            _signed_worker_body(
                manager,
                worker_id="member",
                action="chat-result",
                keypair=keys["member"],
                fields={
                    "chat_id": chat_id,
                    "seq": 1,
                    "content": "forged",
                    "verified": True,
                    "receipt_verified": True,
                    "proof_stages": 2,
                },
            )
        )
    assert chunks.empty()
    assert manager.chat_contexts[chat_id]["last_seq"] == 0


def test_validator_chat_sequence_is_idempotent_and_rejects_gaps(
    tmp_path: Path,
) -> None:
    _state_dir, manager, admin = _validator_pool(tmp_path)
    mesh_key, keys = _launch_serving_validator_mesh(manager, admin)
    chat_id, chunks = _start_running_chat(
        manager,
        admin,
        mesh_key,
        keys["driver"],
    )

    gap = manager.push_chat_chunk(
        _signed_worker_body(
            manager,
            worker_id="driver",
            action="chat-chunk",
            keypair=keys["driver"],
            fields={"chat_id": chat_id, "seq": 2, "delta": "late"},
        )
    )
    assert gap == {"status": "gap", "expected_seq": 1}
    first = manager.push_chat_chunk(
        _signed_worker_body(
            manager,
            worker_id="driver",
            action="chat-chunk",
            keypair=keys["driver"],
            fields={"chat_id": chat_id, "seq": 1, "delta": "one"},
        )
    )
    assert first == {"status": "ok", "seq": 1}
    duplicate = manager.push_chat_chunk(
        _signed_worker_body(
            manager,
            worker_id="driver",
            action="chat-chunk",
            keypair=keys["driver"],
            fields={"chat_id": chat_id, "seq": 1, "delta": "one"},
        )
    )
    assert duplicate == {
        "status": "ok",
        "duplicate": True,
        "seq": 1,
    }
    assert chunks.get_nowait() == {"type": "delta", "delta": "one"}
    assert chunks.empty()

    final_gap = manager.handle_chat_result(
        _signed_worker_body(
            manager,
            worker_id="driver",
            action="chat-result",
            keypair=keys["driver"],
            fields={
                "chat_id": chat_id,
                "seq": 3,
                "content": "done",
                **_valid_validator_final_fields(),
            },
        )
    )
    assert final_gap == {"status": "gap", "expected_seq": 2}
    final = manager.handle_chat_result(
        _signed_worker_body(
            manager,
            worker_id="driver",
            action="chat-result",
            keypair=keys["driver"],
            fields={
                "chat_id": chat_id,
                "seq": 2,
                "content": "done",
                **_valid_validator_final_fields(),
            },
        )
    )
    assert final == {"status": "ok"}
    duplicate_final = manager.handle_chat_result(
        _signed_worker_body(
            manager,
            worker_id="driver",
            action="chat-result",
            keypair=keys["driver"],
            fields={
                "chat_id": chat_id,
                "seq": 2,
                "content": "done",
                **_valid_validator_final_fields(),
            },
        )
    )
    assert duplicate_final == {"status": "ok", "duplicate": True}


@pytest.mark.parametrize(
    "proof_fields",
    [
        pytest.param(
            {
                "verified": False,
                "receipt_verified": True,
                "proof_stages": 2,
            },
            id="unverified",
        ),
        pytest.param(
            {
                "verified": True,
                "receipt_verified": False,
                "proof_stages": 2,
            },
            id="receipt-unverified",
        ),
        pytest.param(
            {
                "verified": True,
                "receipt_verified": True,
                "proof_stages": 1,
            },
            id="missing-stage",
        ),
        pytest.param(
            {
                "verified": True,
                "receipt_verified": True,
                "proof_stages": 3,
            },
            id="extra-stage",
        ),
        pytest.param(
            {
                "verified": True,
                "receipt_verified": True,
                "proof_stages": True,
            },
            id="boolean-stage-count",
        ),
        pytest.param(
            {"receipts": 1},
            id="missing-receipt",
        ),
        pytest.param(
            {"receipts": True},
            id="boolean-receipt-count",
        ),
        pytest.param(
            {"proof_mode": "different-proof-mode"},
            id="wrong-proof-mode",
        ),
        pytest.param(
            {"proof_receipt_root": ""},
            id="missing-receipt-root",
        ),
        pytest.param(
            {"verification_snapshot_hash": "77" * 32},
            id="wrong-snapshot",
        ),
        pytest.param(
            {"mesh_response_commitment_hash": ""},
            id="missing-response-commitment",
        ),
    ],
)
def test_validator_final_requires_exact_verified_proof_stage_coverage(
    tmp_path: Path,
    proof_fields: dict[str, object],
) -> None:
    _state_dir, manager, admin = _validator_pool(tmp_path)
    mesh_key, keys = _launch_serving_validator_mesh(manager, admin)
    chat_id, chunks = _start_running_chat(
        manager,
        admin,
        mesh_key,
        keys["driver"],
    )

    assert manager.handle_chat_result(
        _signed_worker_body(
            manager,
            worker_id="driver",
            action="chat-result",
            keypair=keys["driver"],
            fields={
                "chat_id": chat_id,
                "seq": 1,
                "content": "must not be trusted",
                **_valid_validator_final_fields(),
                **proof_fields,
            },
        )
    ) == {"status": "ok"}
    done = chunks.get_nowait()
    assert done["type"] == "done"
    assert done["status"] == "error"
    assert done["verified"] is False
    assert done["receipt_verified"] is False
    assert "all 2 declared compute stages" in done["error"]


@pytest.mark.parametrize(
    "lease_before_disconnect",
    [
        pytest.param(False, id="queued"),
        pytest.param(True, id="leased"),
    ],
)
def test_stream_disconnect_cleans_up_unacknowledged_chat(
    tmp_path: Path,
    lease_before_disconnect: bool,
) -> None:
    _state_dir, manager, admin = _validator_pool(tmp_path)
    mesh_key, keys = _launch_serving_validator_mesh(manager, admin)
    manager.handle_chat_poll(
        _signed_worker_body(
            manager,
            worker_id="driver",
            action="chat-poll",
            keypair=keys["driver"],
            fields={"wait": 0},
        )
    )
    chat_id, _chunks = manager.start_chat_stream(
        {**admin, "mesh_key": mesh_key, "prompt": "cancel before pickup"}
    )
    if lease_before_disconnect:
        leased = manager.handle_chat_poll(
            _signed_worker_body(
                manager,
                worker_id="driver",
                action="chat-poll",
                keypair=keys["driver"],
                fields={"wait": 0},
            )
        )
        assert [item["chat_id"] for item in leased["chat"]] == [chat_id]
        assert manager.chat_contexts[chat_id]["state"] == "leased"
    else:
        assert manager.chat_contexts[chat_id]["state"] == "queued"

    manager.end_chat_stream(chat_id)

    assert chat_id not in manager.chat_contexts
    assert chat_id not in manager.chat_streams
    assert mesh_key not in manager.chat_active
    assert manager.chat_pending.get("driver", []) == []
    polled = manager.handle_chat_poll(
        _signed_worker_body(
            manager,
            worker_id="driver",
            action="chat-poll",
            keypair=keys["driver"],
            fields={"wait": 0},
        )
    )
    assert polled["chat"] == []


def test_stream_disconnect_keeps_running_lock_until_driver_terminal_result(
    tmp_path: Path,
) -> None:
    _state_dir, manager, admin = _validator_pool(tmp_path)
    mesh_key, keys = _launch_serving_validator_mesh(manager, admin)
    chat_id, _chunks = _start_running_chat(
        manager,
        admin,
        mesh_key,
        keys["driver"],
    )

    manager.end_chat_stream(chat_id)

    assert chat_id not in manager.chat_streams
    assert manager.chat_contexts[chat_id]["client_disconnected"] is True
    assert chat_id in manager.chat_active[mesh_key]
    cancelled = manager.push_chat_chunk(
        _signed_worker_body(
            manager,
            worker_id="driver",
            action="chat-chunk",
            keypair=keys["driver"],
            fields={"chat_id": chat_id, "seq": 1, "delta": "late"},
        )
    )
    assert cancelled == {"status": "cancelled"}
    assert chat_id in manager.chat_active[mesh_key]

    assert manager.handle_chat_result(
        _signed_worker_body(
            manager,
            worker_id="driver",
            action="chat-result",
            keypair=keys["driver"],
            fields={
                "chat_id": chat_id,
                "seq": 1,
                "error": "operator disconnected",
            },
        )
    ) == {"status": "ok"}
    assert mesh_key not in manager.chat_active
    assert chat_id not in manager.chat_contexts


def test_pool_chat_final_delivery_retries_with_fresh_signed_requests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage_key = _worker_stage_key("driver")

    class Runner:
        _stage_proof_keypair = stage_key
        _stage_proof_key_ss58 = stage_key.ss58_address

        @staticmethod
        def _local_request_context() -> tuple[str, dict[str, object]]:
            return "", {}

    class Config:
        advertise_host = "worker.local"
        mesh_port = 19443

    result_attempts: list[dict[str, object]] = []

    def fake_post_json(
        endpoint: str,
        body: dict[str, object],
        **_kwargs,
    ) -> dict[str, object]:
        if endpoint.endswith("/v1/chat/completions"):
            return {
                "choices": [{"message": {"content": "retried"}}],
                "usage": {"completion_tokens": 1},
                "verathos_mesh": {
                    "verified": True,
                    "proof_receipt_verified": True,
                    "proof_receipt_count": 2,
                    "proof_mode": mesh_pool.VERATHOS_GGML_GEMM_PROOF_MODE,
                    "proof_receipt_root": PROOF_RECEIPT_ROOT,
                    "verification_snapshot_hash": VERIFICATION_SNAPSHOT_HASH,
                    "mesh_response_commitment_hash": MESH_RESPONSE_COMMITMENT,
                    "proof_receipts": [
                        {"stage_index": 0},
                        {"stage_index": 1},
                    ],
                },
            }
        assert endpoint.endswith("/v1/pool/chat-result")
        result_attempts.append(dict(body))
        if len(result_attempts) < 3:
            raise TimeoutError("lost acknowledgement")
        return {"status": "ok"}

    monkeypatch.setattr(mesh_pool, "post_json", fake_post_json)
    monkeypatch.setattr(mesh_pool.time, "sleep", lambda _seconds: None)

    mesh_pool._run_pool_chat(
        "https://manager.example",
        "pool-secret",
        "driver",
        Config(),
        {
            "chat_id": "c-retry",
            "model": "qwen-test-q4",
            "messages": [{"role": "user", "content": "retry"}],
            "stream": False,
            "_worker_session_id": "77" * 32,
        },
        runner=Runner(),
    )

    assert len(result_attempts) == 3
    assert {body["seq"] for body in result_attempts} == {1}
    assert len({body["worker_auth_nonce"] for body in result_attempts}) == 3
    for body in result_attempts:
        assert body["worker_auth_action"] == "chat-result"
        assert body["worker_proof_key"] == stage_key.ss58_address
        assert body["worker_session_id"] == "77" * 32
        assert receipt_signing.verify_worker_control_body_hash(
            mesh_pool._worker_control_body_hash_hex(body),
            str(body["worker_auth_signature"]),
            stage_key.ss58_address,
        )


def test_pool_chat_refuses_snapshot_rotation_before_coordinator_inference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage_key = _worker_stage_key("driver")

    class Runner:
        _stage_proof_keypair = stage_key
        _stage_proof_key_ss58 = stage_key.ss58_address

        @staticmethod
        def _local_request_context() -> tuple[str, dict[str, object]]:
            return "internal-secret", {
                "verification_snapshot_hash": "77" * 32,
            }

    class Config:
        advertise_host = "worker.local"
        mesh_port = 19443

    calls: list[tuple[str, dict[str, object]]] = []

    def fake_post_json(
        endpoint: str,
        body: dict[str, object],
        **_kwargs,
    ) -> dict[str, object]:
        calls.append((endpoint, dict(body)))
        assert endpoint.endswith("/v1/pool/chat-result")
        return {"status": "ok"}

    monkeypatch.setattr(mesh_pool, "post_json", fake_post_json)

    mesh_pool._run_pool_chat(
        "https://manager.example",
        "pool-secret",
        "driver",
        Config(),
        {
            "chat_id": "c-snapshot-rotated",
            "model": "qwen-test-q4",
            "messages": [{"role": "user", "content": "must not execute"}],
            "stream": False,
            "verification_snapshot_hash": VERIFICATION_SNAPSHOT_HASH,
            "_worker_session_id": "88" * 32,
        },
        runner=Runner(),
    )

    assert len(calls) == 1
    result = calls[0][1]
    assert result.get("verified") is not True
    assert "manager-pinned mesh snapshot" in str(result["error"])


@pytest.mark.parametrize("vram_gb", [None, 0])
def test_validator_launch_rejects_missing_or_nonpositive_vram(
    tmp_path: Path,
    vram_gb: int | None,
) -> None:
    _state_dir, manager, admin = _validator_pool(tmp_path)
    _register_model(manager, admin)
    _join_pool_worker(manager, worker_id="driver", with_model=True)
    _join_pool_worker(
        manager,
        worker_id="member",
        with_model=False,
        vram_gb=vram_gb,
    )

    with pytest.raises(ValueError, match="positive vram_gb"):
        manager.handle_launch(
            {
                **admin,
                "model_id": "qwen-test-q4",
                "workers": ["driver", "member"],
                "driver": "driver",
            }
        )


def test_validator_launch_rejects_split_with_zero_transformer_layer_stage(
    tmp_path: Path,
) -> None:
    _state_dir, manager, admin = _validator_pool(tmp_path)
    _register_model(manager, admin)
    _join_pool_worker(
        manager,
        worker_id="driver",
        with_model=True,
        vram_gb=100,
    )
    _join_pool_worker(
        manager,
        worker_id="member",
        with_model=False,
        vram_gb=1,
    )

    with pytest.raises(ValueError, match="tensor split is not viable"):
        manager.handle_launch(
            {
                **admin,
                "model_id": "qwen-test-q4",
                "workers": ["driver", "member"],
                "driver": "driver",
            }
        )


def test_validator_pool_rejects_driver_without_wallet_and_allowlist_capability(
    tmp_path: Path,
) -> None:
    _state_dir, manager, admin = _validator_pool(tmp_path)
    _register_model(manager, admin)
    _join_pool_worker(manager, worker_id="driver", with_model=True)
    manager.handle_heartbeat(
        _signed_worker_body(
            manager,
            worker_id="driver",
            action="heartbeat",
            fields={"subnet_driver_ready": False},
        )
    )

    with pytest.raises(ValueError, match="validator allowlist"):
        manager.handle_launch(
            {
                **admin,
                "model_id": "qwen-test-q4",
                "workers": ["driver"],
                "driver": "driver",
            }
        )


def test_pool_worker_driver_readiness_requires_fresh_nonempty_allowlist(
    tmp_path: Path,
) -> None:
    allowlist = tmp_path / "validators.json"
    config = PoolWorkerConfig(
        token=MeshPoolToken(
            pool_id="pool-validator",
            manager_endpoint="http://127.0.0.1:19500",
            pool_secret="secret",
        ),
        repo_root=tmp_path,
        workdir=tmp_path / "runner",
        advertise_host="127.0.0.1",
        rpc_port=15052,
        proof_port=19402,
        mesh_port=19443,
        wallet_name="test_miner96",
        wallet_hotkey="default",
        validator_allowlist_path=str(allowlist),
        validator_allowlist_max_age_seconds=900,
    )

    assert config.subnet_driver_ready is False
    _write_validator_allowlist(allowlist)
    assert config.subnet_driver_ready is False
    _write_validator_allowlist(allowlist, "5Validator")
    assert config.subnet_driver_ready is True
    flags = LocalMeshRunner(config)._validator_coordinator_flags(
        {"coordinator_address": EVM_ADDRESS}
    )
    max_age_index = flags.index("--validator-allowlist-max-age-seconds")
    assert flags[max_age_index + 1] == "900"
    _write_validator_allowlist(
        allowlist,
        "5Validator",
        updated_at=int(time.time()) - 901,
    )
    assert config.subnet_driver_ready is False
    allowlist.write_text("{", encoding="utf-8")
    assert config.subnet_driver_ready is False


@pytest.mark.parametrize(
    "bad_context",
    [
        pytest.param(None, id="missing"),
        pytest.param(True, id="bool"),
        pytest.param(0, id="zero"),
        pytest.param(2**32, id="uint32-overflow"),
    ],
)
def test_validator_driver_rejects_untrusted_context_limit_before_runtime_start(
    tmp_path: Path,
    bad_context: object,
) -> None:
    allowlist = tmp_path / "validators.json"
    _write_validator_allowlist(allowlist, "5Validator")
    runner = LocalMeshRunner(
        PoolWorkerConfig(
            token=MeshPoolToken(
                pool_id="pool-validator",
                manager_endpoint="http://127.0.0.1:19500",
                pool_secret="secret",
            ),
            repo_root=tmp_path,
            workdir=tmp_path / "runner",
            advertise_host="127.0.0.1",
            rpc_port=15052,
            proof_port=19402,
            mesh_port=19443,
            wallet_name="test_miner96",
            wallet_hotkey="default",
            validator_allowlist_path=str(allowlist),
        )
    )
    command = {
        "action": "drive",
        "serving_mode": "validator",
        "model_id": "qwen-test-q4",
        "member_count": 1,
        # Chain-bound (an on-chain slot exists): the registered context IS
        # the serve contract, so a missing or malformed limit must refuse
        # before any runtime starts. An UNREGISTERED measurement launch is
        # exempt: it has no registered value yet and the KV auto-fit
        # measures one (see handle_launch / deploy).
        "model_index": 3,
        "validator_binding": {
            "chain_id": 945,
            "netuid": 405,
            "coordinator_uid": 1,
            "epoch": 123,
            "snapshot_ttl_seconds": 600,
        },
    }
    if bad_context is not None:
        command["max_context_len"] = bad_context

    with pytest.raises(ValueError, match="max_context_len.*positive uint32"):
        runner.drive(command)
    assert runner.mesh_dir is None
    assert runner.procs == []


@requires_native_proof_stack
def test_validator_driver_binds_context_limit_into_mesh_and_llama_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coordinator = Keypair.create_from_uri("//PoolValidatorRuntimeCoordinator")
    allowlist = tmp_path / "validators.json"
    _write_validator_allowlist(allowlist, "5Validator")
    config = PoolWorkerConfig(
        token=MeshPoolToken(
            pool_id="pool-validator",
            manager_endpoint="http://127.0.0.1:19500",
            pool_secret="secret",
        ),
        repo_root=tmp_path,
        workdir=tmp_path / "runner",
        advertise_host="127.0.0.1",
        rpc_port=15052,
        proof_port=19402,
        mesh_port=19443,
        wallet_name="test_miner96",
        wallet_hotkey="default",
        validator_allowlist_path=str(allowlist),
        catalog=[
            {
                "model_id": "qwen-test-q4",
                "llama_model": str(tmp_path / "model.gguf"),
                "manifest": str(tmp_path / "manifest.json"),
                "layers": 4,
                "model_bytes": 1024,
            }
        ],
    )
    runner = LocalMeshRunner(config)
    captured: dict[str, list[str]] = {}

    class CoordinatorCommandCaptured(RuntimeError):
        pass

    def capture_spawn(command, log_name, **_kwargs):
        if log_name == "pool-driver-coordinator.log":
            captured["command"] = list(command)
            raise CoordinatorCommandCaptured

    def finish_prewarm(*_args, done=None, result=None, **_kwargs):
        if result is not None:
            result["converged"] = True
        if done is not None:
            done.set()

    monkeypatch.setattr(runner, "_preflight_backend", lambda: None)
    monkeypatch.setattr(runner, "_free_own_ports", lambda **_kwargs: None)
    monkeypatch.setattr(runner, "_spawn", capture_spawn)
    monkeypatch.setattr(mesh_pool, "_prewarm_proof_cache", finish_prewarm)
    monkeypatch.setattr(
        receipt_signing,
        "load_hotkey_keypair",
        lambda _wallet, _hotkey: coordinator,
    )

    import verallm.mesh.gguf_manifest as gguf_manifest
    import verallm.mesh.model_spec as mesh_model_spec

    monkeypatch.setattr(
        gguf_manifest,
        "load_gguf_tensor_manifest",
        lambda _path: {"tensor_manifest_root": TENSOR_ROOT},
    )
    monkeypatch.setattr(mesh_model_spec, "verify_gguf_model_files", lambda _manifest: None)
    monkeypatch.setattr(
        mesh_model_spec,
        "gguf_package_hash",
        lambda _manifest: bytes.fromhex(PACKAGE_HASH),
    )
    command = {
        "action": "drive",
        "serving_mode": "validator",
        "model_id": "qwen-test-q4",
        "member_count": 2,
        "validator_binding": {
            "chain_id": 945,
            "netuid": 405,
            "coordinator_uid": 1,
            "epoch": 123,
            "snapshot_ttl_seconds": 600,
        },
        "coordinator_address": EVM_ADDRESS,
        "model_index": 26,
        "model_package_hash": PACKAGE_HASH,
        "model_tensor_manifest_root": TENSOR_ROOT,
        "tokenizer_hash": TOKENIZER_HASH,
        "quantization_scheme": "gguf_q4_k_m",
        "total_layers": 4,
        "max_context_len": MAX_CONTEXT_LEN,
        "snapshot_generation": 7,
    }

    with pytest.raises(CoordinatorCommandCaptured):
        runner.drive(command)

    assert runner._drive_command["max_context_len"] == MAX_CONTEXT_LEN
    assert runner.mesh_dir is not None
    runtime_spec = MeshSpec.from_dict(load_mesh_state(runner.mesh_dir)["mesh"])
    assert runtime_spec.max_context_len == MAX_CONTEXT_LEN

    coordinator_command = captured["command"]
    serve_args = coordinator_command[coordinator_command.index("serve") :]
    parsed = mesh_cli.build_parser().parse_args(serve_args)
    # Unified KV budget: --ctx-size equals the registry limit and is
    # shared across the parallel slots (kv-unified), so the advertised
    # per-request maximum is truly servable.
    # The requested budget is the CONTRACT floor; measurement raises it
    # at drive time (no VRAM estimate exists for this fake model).
    assert parsed.llama_ctx_size == MAX_CONTEXT_LEN

    backend_argv = build_llama_server_command(
        binary=parsed.llama_server_binary,
        model=parsed.llama_model,
        host=parsed.llama_host,
        port=parsed.llama_port,
        rpc_endpoints=[
            "worker-0.private:50052",
            "worker-1.private:50052",
        ],
        device=parsed.llama_device,
        n_gpu_layers=parsed.llama_n_gpu_layers,
        ctx_size=parsed.llama_ctx_size,
        tensor_split=parsed.llama_tensor_split,
        alias=parsed.llama_alias or "qwen-test-q4",
        extra_args=parsed.llama_extra_arg,
    )
    ctx_index = backend_argv.index("--ctx-size")
    assert backend_argv[ctx_index + 1] == str(MAX_CONTEXT_LEN)


@requires_native_proof_stack
def test_validator_driver_rejects_local_gguf_package_hash_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_validator_allowlist(tmp_path / "validators.json", "5Validator")
    config = PoolWorkerConfig(
        token=MeshPoolToken(
            pool_id="pool-validator",
            manager_endpoint="http://127.0.0.1:19500",
            pool_secret="secret",
        ),
        repo_root=tmp_path,
        workdir=tmp_path / "runner",
        advertise_host="127.0.0.1",
        rpc_port=15052,
        proof_port=19402,
        mesh_port=19443,
        wallet_name="test_miner96",
        wallet_hotkey="default",
        validator_allowlist_path=str(tmp_path / "validators.json"),
        catalog=[
            {
                "model_id": "qwen-test-q4",
                "llama_model": str(tmp_path / "model.gguf"),
                "manifest": str(tmp_path / "manifest.json"),
                "layers": 4,
                "model_bytes": 1024,
            }
        ],
    )
    runner = LocalMeshRunner(config)
    monkeypatch.setattr(runner, "_preflight_backend", lambda: None)
    monkeypatch.setattr(runner, "_free_own_ports", lambda **_kwargs: None)
    monkeypatch.setattr(mesh_pool, "_prewarm_proof_cache", lambda *_args, **_kwargs: None)

    import verallm.mesh.gguf_manifest as gguf_manifest
    import verallm.mesh.model_spec as mesh_model_spec

    monkeypatch.setattr(
        gguf_manifest,
        "load_gguf_tensor_manifest",
        lambda _path: {
            "tensor_manifest_root": TENSOR_ROOT,
            "model_file_sha256": "44" * 32,
        },
    )
    monkeypatch.setattr(
        mesh_model_spec,
        "verify_gguf_model_files",
        lambda _manifest: None,
    )
    command = {
        "action": "drive",
        "serving_mode": "validator",
        "model_id": "qwen-test-q4",
        "member_count": 1,
        "validator_binding": {
            "chain_id": 945,
            "netuid": 405,
            "coordinator_uid": 1,
            "epoch": 123,
            "snapshot_ttl_seconds": 600,
        },
        "coordinator_address": EVM_ADDRESS,
        "model_index": 26,
        "model_package_hash": PACKAGE_HASH,
        "model_tensor_manifest_root": TENSOR_ROOT,
        "tokenizer_hash": TOKENIZER_HASH,
        "quantization_scheme": "gguf_q4_k_m",
        "total_layers": 4,
        "max_context_len": MAX_CONTEXT_LEN,
        "snapshot_generation": 7,
    }

    with pytest.raises(RuntimeError, match="local GGUF package hash"):
        runner.drive(command)


@requires_native_proof_stack
def test_validator_driver_rejects_modified_local_gguf_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"modified-local-gguf")
    _write_validator_allowlist(tmp_path / "validators.json", "5Validator")
    expected_bytes = b"x" * model_path.stat().st_size
    expected_hash = hashlib.sha256(expected_bytes).hexdigest()
    config = PoolWorkerConfig(
        token=MeshPoolToken(
            pool_id="pool-validator",
            manager_endpoint="http://127.0.0.1:19500",
            pool_secret="secret",
        ),
        repo_root=tmp_path,
        workdir=tmp_path / "runner",
        advertise_host="127.0.0.1",
        rpc_port=15052,
        proof_port=19402,
        mesh_port=19443,
        wallet_name="test_miner96",
        wallet_hotkey="default",
        validator_allowlist_path=str(tmp_path / "validators.json"),
        catalog=[
            {
                "model_id": "qwen-test-q4",
                "llama_model": str(model_path),
                "manifest": str(tmp_path / "manifest.json"),
                "layers": 4,
                "model_bytes": model_path.stat().st_size,
            }
        ],
    )
    runner = LocalMeshRunner(config)
    monkeypatch.setattr(runner, "_preflight_backend", lambda: None)
    monkeypatch.setattr(runner, "_free_own_ports", lambda **_kwargs: None)
    monkeypatch.setattr(mesh_pool, "_prewarm_proof_cache", lambda *_args, **_kwargs: None)

    import verallm.mesh.gguf_manifest as gguf_manifest

    monkeypatch.setattr(
        gguf_manifest,
        "load_gguf_tensor_manifest",
        lambda _path: {
            "tensor_manifest_root": TENSOR_ROOT,
            "model_file_sha256": expected_hash,
            "model_files": [
                {
                    "index": 0,
                    "path": str(model_path),
                    "n_bytes": len(expected_bytes),
                    "sha256": expected_hash,
                }
            ],
        },
    )
    command = {
        "action": "drive",
        "serving_mode": "validator",
        "model_id": "qwen-test-q4",
        "member_count": 1,
        "validator_binding": {
            "chain_id": 945,
            "netuid": 405,
            "coordinator_uid": 1,
            "epoch": 123,
            "snapshot_ttl_seconds": 600,
        },
        "coordinator_address": EVM_ADDRESS,
        "model_index": 26,
        "model_package_hash": expected_hash,
        "model_tensor_manifest_root": TENSOR_ROOT,
        "tokenizer_hash": TOKENIZER_HASH,
        "quantization_scheme": "gguf_q4_k_m",
        "total_layers": 4,
        "max_context_len": MAX_CONTEXT_LEN,
        "snapshot_generation": 7,
    }

    with pytest.raises(ValueError, match="GGUF model file hash mismatch"):
        runner.drive(command)


def _finalizer_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    stage_count: int = 2,
    local_stage: bool = False,
    expected_stage_count: int | None = None,
) -> tuple[LocalMeshRunner, Path, Keypair]:
    coordinator = Keypair.create_from_uri("//PoolValidatorCoordinator")
    stage_keys = [
        Keypair.create_from_uri(f"//PoolValidatorStage{index}")
        for index in range(stage_count)
    ]
    members = [
        MeshMember(
            uid=1,
            hotkey=coordinator.ss58_address,
            endpoint="http://coordinator.private:19443",
            stage_index=0,
            layers=StageRange(0, 0),
            role="coordinator",
            backend="gguf_stage",
            payout_bps=10_000,
        )
    ]
    stage_ranges = llama_tensor_split_layer_ranges(4, [1] * stage_count)
    for index, stage_key in enumerate(stage_keys):
        members.append(
            MeshMember(
                uid=1,
                hotkey=f"private-worker-{index}",
                endpoint=f"http://worker-{index}.private:19402",
                stage_index=index + 1,
                layers=stage_ranges[index],
                role="worker",
                backend="gguf_stage_worker",
                proof_key=stage_key.ss58_address,
                proof_endpoint=f"http://worker-{index}.private:19402",
                # Local stage: the coordinator's llama-server computes this
                # stage on local devices; no rpc-server exists.
                rpc_endpoint="" if local_stage else f"worker-{index}.private:50052",
                rpc_split_weight=0 if local_stage else 1,
            )
        )
    spec = MeshSpec(
        mesh_id="mesh_pool_validator_final",
        mode="private",
        coordinator_uid=1,
        coordinator_hotkey=coordinator.ss58_address,
        model_id="qwen-test-q4",
        model_package_ref="/private/models/model.gguf",
        model_package_hash=PACKAGE_HASH,
        model_tensor_manifest_root=TENSOR_ROOT,
        tokenizer_hash=TOKENIZER_HASH,
        quantization_scheme="gguf_q4_k_m",
        max_context_len=MAX_CONTEXT_LEN,
        activation_dtype="f16",
        proof_trace_manifest_format="compact-raw-v3",
        total_layers=4,
        members=members,
        epoch=123,
    )
    spec.validate()
    state_dir, state, _ = create_mesh_state(spec=spec, root=tmp_path / "mesh")
    expected = stage_count if expected_stage_count is None else expected_stage_count
    state["expected_compute_stage_count"] = expected
    state["serving_mode"] = "validator"
    state["verification_snapshot_required"] = True
    save_mesh_state(state_dir, state)
    _write_validator_allowlist(tmp_path / "validators.json", "5Validator")

    config = PoolWorkerConfig(
        token=MeshPoolToken(
            pool_id="pool-validator",
            manager_endpoint="http://127.0.0.1:19500",
            pool_secret="secret",
        ),
        repo_root=tmp_path,
        workdir=tmp_path / "runner",
        advertise_host="127.0.0.1",
        rpc_port=15052,
        proof_port=19402,
        mesh_port=19443,
        wallet_name="test_miner96",
        wallet_hotkey="default",
        validator_allowlist_path=str(tmp_path / "validators.json"),
    )
    runner = LocalMeshRunner(config)
    runner.mesh_dir = state_dir
    runner._drive_command = {
        "serving_mode": "validator",
        "member_count": expected,
        "validator_binding": {
            "chain_id": 945,
            "netuid": 405,
            "coordinator_uid": 1,
            "epoch": 123,
            "snapshot_ttl_seconds": 600,
        },
        "coordinator_address": EVM_ADDRESS,
        "model_index": 26,
        "model_package_hash": PACKAGE_HASH,
        "model_tensor_manifest_root": TENSOR_ROOT,
        "tokenizer_hash": TOKENIZER_HASH,
        "quantization_scheme": "gguf_q4_k_m",
        "total_layers": 4,
        "max_context_len": MAX_CONTEXT_LEN,
        "snapshot_generation": 7,
    }
    monkeypatch.setattr(
        receipt_signing,
        "load_hotkey_keypair",
        lambda _wallet, _hotkey: coordinator,
    )
    monkeypatch.setattr(
        receipt_signing,
        "load_hotkey_seed",
        lambda _wallet, _hotkey, *, keypair: b"s" * 32,
    )
    monkeypatch.setattr(chain_wallet, "derive_evm_address", lambda _seed: EVM_ADDRESS)
    return runner, state_dir, coordinator


def test_pool_finalizer_signs_only_after_exact_assignment_and_never_serializes_routes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, state_dir, coordinator = _finalizer_fixture(tmp_path, monkeypatch)
    snapshot = runner._finalize_verification_snapshot()
    assert isinstance(snapshot, MeshVerificationSnapshot)
    assert snapshot.coordinator.coordinator_uid == 1
    assert snapshot.coordinator.model_index == 26
    assert snapshot.model.max_context_len == MAX_CONTEXT_LEN
    assert snapshot.generation == 7
    assert snapshot.expected_stage_count == 2
    assert snapshot.policy.profile == "gguf_mesh_v1"
    assert verify_mesh_verification_snapshot_signature(
        snapshot,
        expected_hotkey=coordinator.ss58_address,
        expected_epoch=123,
        expected_generation=7,
        now_unix=int(time.time()),
    )
    payload = snapshot.to_dict()
    assert payload["model"]["max_context_len"] == MAX_CONTEXT_LEN
    assert_endpoint_free_payload(payload)
    serialized = json.dumps(payload, sort_keys=True)
    for private_value in (
        "coordinator.private",
        "worker-0.private",
        "worker-1.private",
        "/private/models",
        "private-worker-0",
        "private-worker-1",
    ):
        assert private_value not in serialized
    persisted = load_mesh_state(state_dir)
    assert persisted["mesh_finalized"] is True
    assert persisted["verification_snapshot_hash"] == snapshot.snapshot_hash_hex()
    finalized_again = runner._finalize_verification_snapshot()
    assert finalized_again.to_dict() == snapshot.to_dict()
    internal_secret, first_context = runner._local_request_context()
    _, second_context = runner._local_request_context()
    assert internal_secret
    assert first_context["verification_snapshot_hash"] == snapshot.snapshot_hash_hex()
    assert len(first_context["validator_nonce"]) == 64
    assert len(first_context["validator_request_id"]) == 64
    assert first_context["validator_nonce"] != second_context["validator_nonce"]
    assert (
        first_context["validator_request_id"]
        != second_context["validator_request_id"]
    )


def test_pool_finalizer_refuses_incomplete_member_assignment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, state_dir, _ = _finalizer_fixture(
        tmp_path,
        monkeypatch,
        stage_count=1,
        expected_stage_count=2,
    )
    monkeypatch.setattr(runner, "_ready_timeout_s", lambda: 0.02)
    with pytest.raises(RuntimeError, match="final mesh member assignment"):
        runner._finalize_verification_snapshot()
    assert not (state_dir / "verification_snapshot.json").exists()


def test_pool_finalizer_fence_blocks_publication_after_supersession(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, state_dir, _ = _finalizer_fixture(tmp_path, monkeypatch)
    drive_command_id = "cmd-" + "a1" * 16
    stop_command_id = "cmd-" + "b2" * 16
    runner.fence_command(drive_command_id)
    runner.enter_command(drive_command_id)
    original_sign = verification_snapshot.sign_mesh_verification_snapshot

    def sign_then_supersede(*args, **kwargs):
        signed = original_sign(*args, **kwargs)
        runner.fence_command(stop_command_id)
        return signed

    monkeypatch.setattr(
        verification_snapshot,
        "sign_mesh_verification_snapshot",
        sign_then_supersede,
    )

    with pytest.raises(RuntimeError, match="superseded"):
        runner._finalize_verification_snapshot()

    persisted = load_mesh_state(state_dir)
    assert persisted.get("mesh_finalized") is not True
    assert "verification_snapshot_hash" not in persisted
    assert not (state_dir / "verification_snapshot.json").exists()


def test_pool_finalizer_requires_all_rpc_compute_topology(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, state_dir, _ = _finalizer_fixture(tmp_path, monkeypatch)
    state = load_mesh_state(state_dir)
    state["mesh"]["members"][1]["rpc_endpoint"] = ""
    state["mesh"]["members"][1]["rpc_split_weight"] = 0
    save_mesh_state(state_dir, state)

    with pytest.raises(
        RuntimeError,
        match="committed RPC endpoints and positive split weights",
    ):
        runner._finalize_verification_snapshot()
    assert not (state_dir / "verification_snapshot.json").exists()


def test_pool_finalizer_accepts_local_stage_single_worker_mesh(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A single-worker LOCAL-STAGE validator mesh (no rpc endpoint; the
    coordinator's llama-server computes the stage) finalizes and signs a
    normal snapshot: the committed verification surface is endpoint-free,
    so the transport never reaches validators."""
    runner, state_dir, coordinator = _finalizer_fixture(
        tmp_path, monkeypatch, stage_count=1, local_stage=True
    )
    snapshot = runner._finalize_verification_snapshot()
    assert isinstance(snapshot, MeshVerificationSnapshot)
    assert snapshot.expected_stage_count == 1
    assert len(snapshot.stages) == 1
    stage = snapshot.stages[0]
    assert stage.layer_start == 0 and stage.layer_end == 4
    snapshot.validate(require_signature=True)
    payload = snapshot.to_dict()
    assert (state_dir / "verification_snapshot.json").exists()
    # The endpoint-free guarantee holds for the local-stage shape too.
    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                assert "endpoint" not in key
                assert "rpc" not in key
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)
    walk(payload)


def test_pool_finalizer_local_stage_never_relaxes_multi_stage_rpc(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The local-stage exemption is exactly one no-rpc compute stage. A
    single stage that HAS an rpc endpoint but a zero split weight is a
    broken all-RPC commitment, not a local stage, and must still fail."""
    runner, state_dir, _coordinator = _finalizer_fixture(
        tmp_path, monkeypatch, stage_count=1
    )
    state = load_mesh_state(state_dir)
    state["mesh"]["members"][1]["rpc_split_weight"] = 0
    save_mesh_state(state_dir, state)
    with pytest.raises(
        RuntimeError, match="committed RPC endpoints and positive split weights"
    ):
        runner._finalize_verification_snapshot()
    assert not (state_dir / "verification_snapshot.json").exists()


def test_pool_finalizer_recomputes_committed_rpc_ranges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, state_dir, _ = _finalizer_fixture(tmp_path, monkeypatch)
    state = load_mesh_state(state_dir)
    state["mesh"]["members"][2]["rpc_split_weight"] = 2
    save_mesh_state(state_dir, state)

    with pytest.raises(
        RuntimeError,
        match="RPC placement is invalid.*do not match committed tensor split",
    ):
        runner._finalize_verification_snapshot()
    assert not (state_dir / "verification_snapshot.json").exists()


def test_unconfigured_legacy_pool_cannot_launch_validator_or_dev_traffic(
    tmp_path: Path,
) -> None:
    state_dir, manager, admin = _validator_pool(tmp_path)
    state = json.loads((state_dir / POOL_STATE_FILE).read_text(encoding="utf-8"))
    state.pop("serving_mode")
    (state_dir / POOL_STATE_FILE).write_text(json.dumps(state), encoding="utf-8")
    legacy = PoolManager(state_dir)
    with pytest.raises(ValueError, match="unconfigured"):
        legacy.handle_launch({**admin, "model_id": "anything"})


def test_unregistered_single_worker_drive_ready_needs_no_snapshot(
    tmp_path: Path,
) -> None:
    """An UNREGISTERED subnet launch (no chain slot) signs no snapshot by
    design, so its single-worker drive_ready must be ACCEPTED without a
    verification snapshot hash - requiring one dead-loops the drive (the
    report is rejected, the worker re-drives forever; observed on the
    glm measurement launch). A snapshot hash showing up on an unregistered
    mesh is refused instead: there are no chain anchors behind it."""
    _state_dir, manager, admin = _validator_pool(tmp_path)
    registry = manager.state.setdefault("model_registry", {})
    registry["glm-measure"] = {
        "hf_repo": "operator/model",
        "hf_files": ["model.gguf"],
        "layers": 4,
    }
    keypair = _join_pool_worker(manager, worker_id="driver", with_model=False)
    worker_catalog = manager.state["workers"]["driver"].setdefault("catalog", [])
    worker_catalog.append(
        {
            "model_id": "glm-measure",
            "model_bytes": 1024,
            "layers": 4,
        }
    )
    launched = manager.handle_launch(
        {
            **admin,
            "model_id": "glm-measure",
            "workers": ["driver"],
            "driver": "driver",
        }
    )
    mesh_key = launched["mesh_key"]
    assert manager.state["meshes"][mesh_key].get("model_index") is None
    drive = manager.handle_heartbeat(
        _signed_worker_body(
            manager, worker_id="driver", action="heartbeat", keypair=keypair
        )
    )["command"]
    assert drive["action"] == "drive"

    def _drive_ready_fields(**extra: object) -> dict[str, object]:
        return {
            "command_id": drive["command_id"],
            "command_digest": drive["command_digest"],
            "mesh_key": mesh_key,
            "event": "drive_ready",
            "mesh_id": "mesh-unregistered-measure",
            "join_token": "vtmesh_unregistered_measure",
            "coordinator_endpoint": "http://driver.private:19443",
            **extra,
        }

    # A snapshot hash on an unregistered mesh is refused.
    with pytest.raises(ValueError, match="must not carry"):
        manager.handle_report(
            _signed_worker_body(
                manager,
                worker_id="driver",
                action="report",
                keypair=keypair,
                fields=_drive_ready_fields(
                    verification_snapshot_hash=VERIFICATION_SNAPSHOT_HASH
                ),
            )
        )
    # Without one, the report is accepted (terminal for a single worker)
    # and the mesh serves.
    assert manager.handle_report(
        _signed_worker_body(
            manager,
            worker_id="driver",
            action="report",
            keypair=keypair,
            fields=_drive_ready_fields(),
        )
    ) == {"status": "ok", "command_completed": drive["command_id"]}
    mesh = manager.state["meshes"][mesh_key]
    assert mesh["status"] == "serving"
    assert mesh.get("verification_snapshot_hash", "") == ""


def test_unregistered_mesh_chat_verifies_without_snapshot_binding(
    tmp_path: Path,
) -> None:
    """The operator measurement lane on an UNREGISTERED subnet mesh carries
    full proof coverage but no snapshot (nothing on chain to bind). The
    chat finalizer must accept a snapshot-less verified final there - it
    used to fail every measurement probe with the snapshot-bound coverage
    error - while still refusing a final that smuggles a snapshot hash in."""
    _state_dir, manager, admin = _validator_pool(tmp_path)
    registry = manager.state.setdefault("model_registry", {})
    registry["glm-measure"] = {
        "hf_repo": "operator/model",
        "hf_files": ["model.gguf"],
        "layers": 4,
    }
    keypair = _join_pool_worker(manager, worker_id="driver", with_model=False)
    manager.state["workers"]["driver"].setdefault("catalog", []).append(
        {"model_id": "glm-measure", "model_bytes": 1024, "layers": 4}
    )
    launched = manager.handle_launch(
        {
            **admin,
            "model_id": "glm-measure",
            "workers": ["driver"],
            "driver": "driver",
        }
    )
    mesh_key = launched["mesh_key"]
    drive = manager.handle_heartbeat(
        _signed_worker_body(
            manager, worker_id="driver", action="heartbeat", keypair=keypair
        )
    )["command"]
    manager.handle_report(
        _signed_worker_body(
            manager,
            worker_id="driver",
            action="report",
            keypair=keypair,
            fields={
                "command_id": drive["command_id"],
                "command_digest": drive["command_digest"],
                "mesh_key": mesh_key,
                "event": "drive_ready",
                "mesh_id": "mesh-unregistered-chat",
                "join_token": "vtmesh_unregistered_chat",
                "coordinator_endpoint": "http://driver.private:19443",
            },
        )
    )
    assert manager.state["meshes"][mesh_key]["status"] == "serving"

    def _final_fields(**overrides: object) -> dict[str, object]:
        fields: dict[str, object] = {
            "verified": True,
            "receipt_verified": True,
            "receipts": 1,
            "proof_stages": 1,
            "proof_mode": mesh_pool.VERATHOS_GGML_GEMM_PROOF_MODE,
            "proof_receipt_root": PROOF_RECEIPT_ROOT,
            "mesh_response_commitment_hash": MESH_RESPONSE_COMMITMENT,
        }
        fields.update(overrides)
        return fields

    def _start_unregistered_chat() -> tuple[str, queue.Queue]:
        manager.handle_chat_poll(
            _signed_worker_body(
                manager,
                worker_id="driver",
                action="chat-poll",
                keypair=keypair,
                fields={"wait": 0},
            )
        )
        chat_id, chunks = manager.start_chat_stream(
            {**admin, "mesh_key": mesh_key, "prompt": "prove this response"}
        )
        polled = manager.handle_chat_poll(
            _signed_worker_body(
                manager,
                worker_id="driver",
                action="chat-poll",
                keypair=keypair,
                fields={"wait": 0},
            )
        )
        assert [item["chat_id"] for item in polled["chat"]] == [chat_id]
        # No snapshot rides the unregistered lease.
        assert polled["chat"][0]["verification_snapshot_hash"] == ""
        assert manager.handle_chat_pickup(
            _signed_worker_body(
                manager,
                worker_id="driver",
                action="chat-pickup",
                keypair=keypair,
                fields={
                    "chat_id": chat_id,
                    "delivery_token": polled["chat"][0]["delivery_token"],
                },
            )
        ) == {"status": "ok"}
        return chat_id, chunks

    # A smuggled snapshot hash on the unregistered lane is refused.
    chat_id, chunks = _start_unregistered_chat()
    manager.handle_chat_result(
        _signed_worker_body(
            manager,
            worker_id="driver",
            action="chat-result",
            keypair=keypair,
            fields={
                "chat_id": chat_id,
                "seq": 1,
                "content": "hi",
                **_final_fields(
                    verification_snapshot_hash=VERIFICATION_SNAPSHOT_HASH
                ),
            },
        )
    )
    done = chunks.get_nowait()
    assert done["type"] == "done"
    assert done["verified"] is False
    assert "manifest-bound" in str(done.get("error", ""))

    # A snapshot-less verified final passes.
    chat_id, chunks = _start_unregistered_chat()
    manager.handle_chat_result(
        _signed_worker_body(
            manager,
            worker_id="driver",
            action="chat-result",
            keypair=keypair,
            fields={
                "chat_id": chat_id,
                "seq": 1,
                "content": "hi",
                **_final_fields(),
            },
        )
    )
    done = chunks.get_nowait()
    assert done["type"] == "done"
    assert done["status"] == "ok"
    assert done["verified"] is True
    assert done["receipt_verified"] is True
    assert done["expected_stage_count"] == 1

    # The organic light tier's proof mode passes on a plain unregistered
    # chat (the validator itself accepts light mode for light receipts)...
    chat_id, chunks = _start_unregistered_chat()
    manager.handle_chat_result(
        _signed_worker_body(
            manager,
            worker_id="driver",
            action="chat-result",
            keypair=keypair,
            fields={
                "chat_id": chat_id,
                "seq": 1,
                "content": "hi",
                **_final_fields(
                    proof_mode=mesh_pool.VERATHOS_GGML_LIGHT_PROOF_MODE
                ),
            },
        )
    )
    done = chunks.get_nowait()
    assert done["type"] == "done"
    assert done["verified"] is True

    # ...but an EXPLICIT hard-tier chat must never come back light.
    manager.handle_chat_poll(
        _signed_worker_body(
            manager,
            worker_id="driver",
            action="chat-poll",
            keypair=keypair,
            fields={"wait": 0},
        )
    )
    chat_id, chunks = manager.start_chat_stream(
        {
            **admin,
            "mesh_key": mesh_key,
            "prompt": "prove this response",
            "proof_tier": "hard",
        }
    )
    polled = manager.handle_chat_poll(
        _signed_worker_body(
            manager,
            worker_id="driver",
            action="chat-poll",
            keypair=keypair,
            fields={"wait": 0},
        )
    )
    manager.handle_chat_pickup(
        _signed_worker_body(
            manager,
            worker_id="driver",
            action="chat-pickup",
            keypair=keypair,
            fields={
                "chat_id": chat_id,
                "delivery_token": polled["chat"][0]["delivery_token"],
            },
        )
    )
    manager.handle_chat_result(
        _signed_worker_body(
            manager,
            worker_id="driver",
            action="chat-result",
            keypair=keypair,
            fields={
                "chat_id": chat_id,
                "seq": 1,
                "content": "hi",
                **_final_fields(
                    proof_mode=mesh_pool.VERATHOS_GGML_LIGHT_PROOF_MODE
                ),
            },
        )
    )
    done = chunks.get_nowait()
    assert done["type"] == "done"
    assert done["verified"] is False
    assert "proof_mode" in str(done.get("error", ""))
