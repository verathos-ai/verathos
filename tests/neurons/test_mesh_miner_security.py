from __future__ import annotations

from types import SimpleNamespace

from neurons import miner


def _value(args: list[str], flag: str) -> str:
    return args[args.index(flag) + 1]


def test_wallet_mesh_args_forward_identity_auth_and_isolated_allowlist(
    monkeypatch, tmp_path
):
    allowlist = tmp_path / "testnet-405-validators.json"
    monkeypatch.setenv("VERATHOS_VALIDATORS_PATH", str(allowlist))
    args = SimpleNamespace(wallet="test_miner96", hotkey="default")

    configured = miner._configure_mesh_security_args(
        args=args,
        server_args=[
            "--server-role",
            "worker",
            "--validator-allowlist-path",
            "/tmp/wrong.json",
            "--evm-private-key",
            "stale-forwarded-key",
        ],
        evm_address="0xCoordinator",
        evm_private_key="secret-private-key",
    )

    assert _value(configured, "--server-role") == "coordinator"
    assert _value(configured, "--wallet-name") == "test_miner96"
    assert _value(configured, "--wallet-hotkey") == "default"
    assert _value(configured, "--evm-address") == "0xCoordinator"
    assert "--evm-private-key" not in configured
    assert "secret-private-key" not in configured
    assert "stale-forwarded-key" not in configured
    neuron = miner.MinerNeuron.__new__(miner.MinerNeuron)
    neuron.mesh_dir = "/persistent/mesh-state"
    process_argv = neuron._server_cmd(configured)
    assert "--evm-private-key" not in process_argv
    assert "secret-private-key" not in process_argv
    assert "stale-forwarded-key" not in process_argv
    assert _value(configured, "--validator-allowlist-path") == str(allowlist)
    assert configured.count("--validator-auth") == 1
    assert configured.count("--require-validator-nonce") == 1
    assert "secret-private-key" not in miner.MinerNeuron._redact_cmd(configured)


def test_wallet_allowlist_refresh_is_not_disabled_for_mesh_mode():
    args = SimpleNamespace(wallet="test_miner96", mesh_dir="/tmp/mesh")
    assert miner._validator_allowlist_refresh_enabled(args) is True
    assert miner._validator_allowlist_refresh_enabled(
        SimpleNamespace(wallet=None, mesh_dir="/tmp/mesh")
    ) is False


def test_mesh_mode_never_clears_vllm_compile_caches():
    assert miner._should_clear_stale_compile_caches(
        SimpleNamespace(mesh_dir="/persistent/mesh-state")
    ) is False
    assert miner._should_clear_stale_compile_caches(
        SimpleNamespace(mesh_dir=None)
    ) is True


def test_private_key_mesh_mode_preserves_opt_in_validator_auth(monkeypatch, tmp_path):
    allowlist = tmp_path / "validators.json"
    monkeypatch.setenv("VERATHOS_VALIDATORS_PATH", str(allowlist))
    configured = miner._configure_mesh_security_args(
        args=SimpleNamespace(wallet=None, hotkey="default"),
        server_args=["--validator-auth"],
        evm_address="0xAnvil",
        evm_private_key="anvil-key",
    )
    assert "--validator-auth" in configured
    assert "--require-validator-nonce" in configured
    assert _value(configured, "--validator-allowlist-path") == str(allowlist)
    assert "--wallet-name" not in configured
    assert _value(configured, "--evm-address") == "0xAnvil"
    assert _value(configured, "--evm-private-key") == "anvil-key"
