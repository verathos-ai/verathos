#!/usr/bin/env python3
"""MinerNeuron — Bittensor miner wrapper for Verathos.

Lifecycle:
1. Resolve model config (auto or explicit, with cascading fallback).
2. Auto-associate EVM key if needed (one-time, via Substrate extrinsic).
3. Start the existing verallm.api.server as a subprocess.
4. Wait for /health to return OK.
5. Register model on MinerRegistry (skip if already registered with same params).
6. Enter heartbeat loop: renewModel() every 12 hours.

Usage:
    # Testnet — auto-select best model for GPU:
    python -m neurons.miner \
        --wallet miner --hotkey default --netuid 42 \
        --chain-config chain_config_testnet.json \
        --auto --endpoint https://miner.example.com:8000

    # Anvil — same thing but with --private-key instead of --wallet:
    python -m neurons.miner \
        --private-key 0x59c6...690d --netuid 42 \
        --chain-config chain_config_anvil.json \
        --auto --endpoint http://localhost:8000
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
from typing import Callable, Optional

import bittensor as bt
import httpx

from neurons.config import NeuronConfig
from neurons.model_resolve import (
    add_model_args,
    capacity_gate_vram_gb,
    resolve_model_config,
    validate_capacity_recommended_model,
)
from neurons.subnet_runtime_config import (
    RuntimeSubnetConfigClient,
    apply_runtime_config_to_neuron_config,
)
from neurons.version import spec_version, version_str, miner_version, miner_version_str
from verallm.chain.config import ChainConfig
from verallm.chain.miner_registry import MinerRegistryClient
from verallm.chain.provider import Web3Provider
from verallm.chain.wallet import derive_evm_private_key, derive_evm_address

logger = logging.getLogger(__name__)


PROOF_V2_MANIFEST_ENV = "VERATHOS_PROOF_V2_MANIFEST"
PROOF_V2_WEIGHT_CATALOG_ENV = "VERATHOS_PROOF_V2_WEIGHT_CATALOG"
PROOF_V2_ARTIFACT_BASE_URLS_ENV = "VERATHOS_PROOF_V2_ARTIFACT_BASE_URLS"
PROOF_V2_ARTIFACT_CACHE_DIR_ENV = "VERATHOS_PROOF_V2_ARTIFACT_CACHE_DIR"
PROOF_V3_MANIFEST_ENV = "VERATHOS_PROOF_V3_MANIFEST"
PROOF_V3_EXECUTION_PROFILE_ENV = "VERATHOS_PROOF_V3_EXECUTION_PROFILE"
PROOF_V3_CALIBRATION_SET_ENV = "VERATHOS_PROOF_V3_CALIBRATION_SET"
PROOF_V3_ATTENTION_SEMANTICS_ENV = "VERATHOS_PROOF_V3_ATTENTION_SEMANTICS"
PROOF_V3_GDN_SEMANTICS_ENV = "VERATHOS_PROOF_V3_GDN_SEMANTICS"
PROOF_V3_LM_HEAD_CATALOG_ENV = "VERATHOS_PROOF_V3_LM_HEAD_CATALOG"
PROOF_V3_PROJECTION_MANIFEST_ENV = "VERATHOS_PROOF_V3_PROJECTION_MANIFEST"
PROOF_V3_PROJECTION_CATALOG_ENV = "VERATHOS_PROOF_V3_PROJECTION_CATALOG"
PROOF_V3_RUNTIME_ENCODING_ENV = "VERATHOS_PROOF_V3_RUNTIME_ENCODING"
PROOF_V3_WEIGHT_CACHE_DIR_ENV = "VERATHOS_PROOF_V3_WEIGHT_CACHE_DIR"
PROOF_V3_RELEASE_ENV = "VERATHOS_PROOF_V3_RELEASE"

_PROOF_SAFE_FP8_BACKEND_ENV = (
    "VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER",
    "VLLM_USE_DEEP_GEMM",
)

PROOF_V3_ARTIFACT_REFRESH_INTERVAL_SECONDS = 60.0
PROOF_V3_ARTIFACT_REFRESH_RETRY_SECONDS = 5.0

_MANAGED_NGINX_CONFIG_PATHS = (
    Path("/etc/nginx/sites-available/verathos-miner"),
    Path("/etc/nginx/sites-enabled/verathos-miner"),
    Path("/etc/nginx/nginx.conf"),
)
_MANAGED_NGINX_READ_TIMEOUT_PATTERN = re.compile(
    r"(?m)^(?P<indent>[ \t]*)proxy_read_timeout[ \t]+(?P<seconds>[0-9]+)s;[ \t]*$"
)
_NGINX_READ_TIMEOUT_VALUE_PATTERN = re.compile(
    r"\bproxy_read_timeout[ \t]+(?P<seconds>[0-9]+)s?[ \t]*;"
)
_MANAGED_NGINX_TIMEOUT_GRACE_SECONDS = 60
_MANAGED_NGINX_FALLBACK_READ_TIMEOUT_SECONDS = 960


def _managed_nginx_read_timeout_seconds(config: NeuronConfig) -> int:
    """Return the restart-time proxy ceiling for the active subnet policy."""

    full_context_seconds = math.ceil(
        float(config.canary_full_context_inference_timeout)
    )
    hard_proof_seconds = int(
        getattr(config, "proof_v3_timing_max_hard_proof_timeout_s", 540)
    )
    transport_margin_seconds = int(
        getattr(
            config,
            "proof_v3_timing_transport_margin_s",
            _MANAGED_NGINX_TIMEOUT_GRACE_SECONDS,
        )
    )
    return (
        max(full_context_seconds, hard_proof_seconds)
        + transport_margin_seconds
    )


def _reconcile_managed_nginx_read_timeout(
    *,
    read_timeout_seconds: int = _MANAGED_NGINX_FALLBACK_READ_TIMEOUT_SECONDS,
    config_paths: tuple[Path, ...] = _MANAGED_NGINX_CONFIG_PATHS,
    run_command: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    effective_uid: int | None = None,
) -> bool:
    """Align the stock HTTPS proxy with the active application deadline.

    Full-context inference and hard proofs have their own validator-enforced
    deadlines. Nginx is only their transport ceiling, not another policy
    layer. On each miner restart, the caller derives this value from the
    effective subnet timeout plus the fixed transport grace.
    Only byte-recognizable Verathos-managed server blocks are migrated; custom
    reverse-proxy configurations are never rewritten.
    """

    if (
        isinstance(read_timeout_seconds, bool)
        or not isinstance(read_timeout_seconds, int)
        or not 1 <= read_timeout_seconds <= 86_400
    ):
        raise ValueError("read_timeout_seconds must be an integer in [1, 86400]")

    managed_marker = "ssl_certificate /etc/nginx/ssl/miner.crt;"
    backend_marker = "proxy_pass http://127.0.0.1:"
    target_timeout = f"proxy_read_timeout {read_timeout_seconds}s;"
    selected_uid = os.geteuid() if effective_uid is None else effective_uid
    use_sudo = selected_uid != 0
    changed: list[tuple[Path, str, int]] = []
    seen: set[Path] = set()

    def _install_text(path: Path, content: str, mode: int) -> bool:
        if not use_sudo:
            temporary = path.with_name(f".{path.name}.verathos-timeout.tmp")
            try:
                temporary.write_text(content)
                temporary.chmod(mode)
                os.replace(temporary, path)
                return True
            except OSError:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass
                return False

        temporary_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                prefix="verathos-nginx-timeout-",
                delete=False,
            ) as temporary:
                temporary.write(content)
                temporary_name = temporary.name
            installed = run_command(
                [
                    "sudo",
                    "-n",
                    "install",
                    "-m",
                    f"{mode:o}",
                    "--",
                    temporary_name,
                    str(path),
                ],
                capture_output=True,
                text=True,
                timeout=15,
            )
            return installed.returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            return False
        finally:
            if temporary_name is not None:
                try:
                    Path(temporary_name).unlink(missing_ok=True)
                except OSError:
                    pass

    for candidate in config_paths:
        try:
            path = candidate.resolve(strict=True)
        except (FileNotFoundError, OSError):
            continue
        if path in seen:
            continue
        seen.add(path)
        try:
            original = path.read_text()
        except (OSError, UnicodeError):
            continue
        matches = tuple(_MANAGED_NGINX_READ_TIMEOUT_PATTERN.finditer(original))
        if not matches:
            if backend_marker in original:
                detected = tuple(
                    _NGINX_READ_TIMEOUT_VALUE_PATTERN.finditer(original)
                )
                if detected:
                    configured_min = min(
                        int(match.group("seconds")) for match in detected
                    )
                    if configured_min < read_timeout_seconds:
                        bt.logging.warning(
                            f"Reverse proxy config {path} was not changed; its "
                            f"shortest upstream read timeout is {configured_min}s, "
                            f"below the required {read_timeout_seconds}s. Long "
                            "requests may return HTTP 504"
                        )
                else:
                    bt.logging.warning(
                        f"Reverse proxy config {path} has no recognized explicit "
                        "proxy_read_timeout; long inference or proof-v3 responses "
                        f"may return HTTP 504. Set it to at least {read_timeout_seconds}s"
                    )
            continue
        if managed_marker not in original or backend_marker not in original:
            configured_min = min(int(match.group("seconds")) for match in matches)
            if configured_min < read_timeout_seconds:
                bt.logging.warning(
                    f"Custom nginx config {path} was not changed; its shortest "
                    f"upstream read timeout is {configured_min}s, below the required "
                    f"{read_timeout_seconds}s. Long requests may return HTTP 504"
                )
            continue
        managed_count = original.count(managed_marker)
        backend_count = original.count(backend_marker)
        if (
            managed_count < 1
            or managed_count != backend_count
            or managed_count != len(matches)
        ):
            bt.logging.warning(
                f"Managed nginx config {path} has an ambiguous read timeout; "
                f"set it to at least {read_timeout_seconds}s manually"
            )
            continue
        if all(int(match.group("seconds")) == read_timeout_seconds for match in matches):
            continue
        updated = _MANAGED_NGINX_READ_TIMEOUT_PATTERN.sub(
            lambda match: f"{match.group('indent')}{target_timeout}",
            original,
        )
        try:
            mode = path.stat().st_mode & 0o7777
        except OSError as exc:
            bt.logging.warning(
                f"Could not inspect managed nginx config {path}: {exc}; "
                "set it to at least 360s manually"
            )
            continue
        if not _install_text(path, updated, mode):
            bt.logging.warning(
                f"Could not update managed nginx timeout in {path}; "
                "passwordless sudo is required for a non-root official install"
            )
            continue
        changed.append((path, original, mode))

    if not changed:
        return True

    def _run_nginx(*args: str) -> subprocess.CompletedProcess:
        prefix = ["sudo", "-n"] if use_sudo else []
        return run_command(
            [*prefix, "nginx", *args],
            capture_output=True,
            text=True,
            timeout=15,
        )

    try:
        checked = _run_nginx("-t")
    except (OSError, subprocess.TimeoutExpired) as exc:
        checked = subprocess.CompletedProcess(
            ["nginx", "-t"], 1, "", str(exc)
        )
    if checked.returncode != 0:
        rollback_results = [
            _install_text(path, original, mode)
            for path, original, mode in changed
        ]
        rollback_ok = all(rollback_results)
        bt.logging.warning(
            "Managed nginx timeout migration failed validation and was rolled "
            f"back{'' if rollback_ok else ' incompletely'}: "
            f"{(checked.stderr or checked.stdout).strip()}"
        )
        return False

    try:
        reloaded = _run_nginx("-s", "reload")
    except (OSError, subprocess.TimeoutExpired) as exc:
        reloaded = subprocess.CompletedProcess(
            ["nginx", "-s", "reload"], 1, "", str(exc)
        )
    if reloaded.returncode != 0:
        bt.logging.warning(
            "Managed nginx timeout was updated but nginx reload failed: "
            f"{(reloaded.stderr or reloaded.stdout).strip()}"
        )
        return False

    bt.logging.info(
        "Updated the managed nginx upstream read timeout to "
        f"{read_timeout_seconds}s for full-context inference and proof-v3 "
        "hard responses"
    )
    return True


def _configured_miner_proof_protocol_versions(
    owner_allowed: tuple[int, ...],
    *,
    proof_v3_configured: bool,
) -> tuple[int, ...]:
    """Intersect owner policy with the proof stack this miner will serve.

    Validators support both rollout generations.  A miner with an
    authenticated v3 release serves v3 only; a legacy-configured miner serves
    v1 only.  This keeps ``[1, 3]`` an owner compatibility allowlist without
    forcing updated compressed checkpoints to construct an unused legacy-v1
    weight tree.
    """

    locally_served = {3} if proof_v3_configured else {1}
    return tuple(
        version
        for version in sorted(set(int(v) for v in owner_allowed))
        if version in locally_served
    )


class ProofV3ArtifactRefreshWatcher:
    """Detect and adopt authenticated remote proof-v3 release changes.

    A cheap chain-context-checked index probe runs on the steady-state path.
    A candidate change is fully downloaded and authenticated before a restart
    is scheduled. Live inference, hard proofs, and capacity audits are never
    interrupted; the authenticated update remains pending until the endpoint
    reaches an idle adoption window.
    """

    def __init__(
        self,
        *,
        current_release_sha256: bytes,
        probe_release: Callable[[], object],
        resolve_release: Callable[[], object],
        busy_state: Callable[[], tuple[int, int, bool, bool]],
        restart: Callable[[], None],
        interval_seconds: float = PROOF_V3_ARTIFACT_REFRESH_INTERVAL_SECONDS,
        retry_seconds: float = PROOF_V3_ARTIFACT_REFRESH_RETRY_SECONDS,
    ) -> None:
        if (
            type(current_release_sha256) is not bytes
            or len(current_release_sha256) != 32
        ):
            raise ValueError("current proof-v3 release digest is invalid")
        if not all(
            callable(value)
            for value in (probe_release, resolve_release, busy_state, restart)
        ):
            raise ValueError("proof-v3 artifact refresh callbacks are invalid")
        if min(interval_seconds, retry_seconds) <= 0:
            raise ValueError("proof-v3 artifact refresh timing is invalid")
        self.current_release_sha256 = current_release_sha256
        self.probe_release = probe_release
        self.resolve_release = resolve_release
        self.busy_state = busy_state
        self.restart = restart
        self.interval_seconds = float(interval_seconds)
        self.retry_seconds = float(retry_seconds)
        self._pending_release_sha256: bytes | None = None
        self._pending_since: float | None = None
        self._restart_triggered = False
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def check_once(self, *, now: float | None = None) -> str:
        """Run one deterministic refresh/adoption decision."""

        if self._stop_event.is_set():
            return "stopped"
        if self._restart_triggered:
            return "restart_triggered"
        selected_now = time.monotonic() if now is None else float(now)
        if self._pending_release_sha256 is None:
            probe = self.probe_release()
            candidate = getattr(probe, "release_sha256", None)
            if type(candidate) is not bytes or len(candidate) != 32:
                raise RuntimeError("proof-v3 release probe digest is invalid")
            if candidate == self.current_release_sha256:
                return "unchanged"
            resolved = self.resolve_release()
            resolved_digest = getattr(resolved, "release_sha256", None)
            if type(resolved_digest) is not bytes or len(resolved_digest) != 32:
                raise RuntimeError(
                    "authenticated proof-v3 release digest is invalid"
                )
            if resolved_digest != candidate:
                raise RuntimeError(
                    "authenticated proof-v3 release does not match the probed index"
                )
            self._pending_release_sha256 = resolved_digest
            self._pending_since = selected_now
            bt.logging.warning(
                "Authenticated proof-v3 artifact update detected: "
                f"{self.current_release_sha256.hex()[:16]}... -> "
                f"{resolved_digest.hex()[:16]}...; waiting for an idle "
                "adoption window"
            )

        active_requests, proof_pending, hard_exclusive, capacity_active = (
            self.busy_state()
        )
        if proof_pending > 0 or hard_exclusive or capacity_active:
            return "deferred_proof_or_audit"
        if active_requests > 0:
            return "deferred_active_requests"
        if self._stop_event.is_set():
            return "stopped"

        self._restart_triggered = True
        bt.logging.info(
            "Proof-v3 artifact adoption window is idle; restarting onto "
            "the authenticated release"
        )
        try:
            self.restart()
        except BaseException:
            # A failed restart attempt must remain retryable.  The production
            # restart helper normally does not return, but this also covers a
            # PM2/exec failure without pinning the watcher in a false-success
            # state.
            self._restart_triggered = False
            raise
        return "restart_triggered"

    def _run(self) -> None:
        if self._stop_event.wait(self.interval_seconds):
            return
        while not self._stop_event.is_set():
            delay = self.interval_seconds
            try:
                result = self.check_once()
                if result.startswith("deferred_"):
                    delay = self.retry_seconds
            except Exception as exc:
                bt.logging.warning(
                    f"Proof-v3 artifact refresh rejected; keeping current release: {exc}"
                )
            if self._stop_event.wait(delay):
                return

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="proof-v3-artifact-refresh",
        )
        self._thread.start()
        bt.logging.info(
            "Proof-v3 artifact refresh started "
            f"(interval={self.interval_seconds:.0f}s)"
        )

    def stop(self) -> None:
        self._stop_event.set()


def _proof_v3_hard_auditor_record(
    runtime,
    validator_hotkeys: set[str],
) -> dict[str, object]:
    """Build the sole hard-auditor record forwarded to the server process."""

    disabled: dict[str, object] = {
        "enabled": False,
        "validator_hotkey_ss58": "",
        "config_version": 0,
        "effective_epoch": None,
    }
    if runtime is None:
        return disabled
    policy = runtime.proof_v3_hard_auditor
    hotkey = str(policy.validator_hotkey_ss58 or "")
    if not policy.enabled or hotkey not in validator_hotkeys:
        return {
            **disabled,
            "config_version": int(runtime.version),
            "effective_epoch": runtime.effective_epoch,
        }
    return {
        "enabled": True,
        "validator_hotkey_ss58": hotkey,
        "config_version": int(runtime.version),
        "effective_epoch": runtime.effective_epoch,
    }


def _capacity_audit_worker_poll_interval(config) -> float:
    raw_value = getattr(config, "capacity_audit_worker_poll_s", 2.0)
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        value = 2.0
    return max(0.1, value)


def _normalize_server_args(server_args: list[str]) -> list[str]:
    """Keep wrapper-only argparse leftovers out of the inner miner server."""
    out: list[str] = []
    for arg in server_args:
        if arg == "--":
            continue
        if arg.startswith("--logging."):
            continue
        out.append(arg)
    return out


def _set_server_arg(server_args: list[str], flag: str, value: str) -> list[str]:
    out: list[str] = []
    skip = False
    for arg in server_args:
        if skip:
            skip = False
            continue
        if arg == flag:
            skip = True
            continue
        out.append(arg)
    out.extend([flag, value])
    return out


def _add_server_flag(server_args: list[str], flag: str) -> list[str]:
    return list(server_args) if flag in server_args else [*server_args, flag]


def _remove_server_arg(server_args: list[str], flag: str) -> list[str]:
    """Remove every ``flag value`` pair from forwarded server arguments."""

    out: list[str] = []
    skip = False
    for arg in server_args:
        if skip:
            skip = False
            continue
        if arg == flag:
            skip = True
            continue
        out.append(arg)
    return out


def _configure_mesh_security_args(
    *,
    args,
    server_args: list[str],
    evm_address: str,
    evm_private_key: str,
) -> list[str]:
    """Forward miner identity and fail-closed validator auth to mesh serve.

    Wallet-backed serving derives the EVM key inside the child from the same
    hotkey it uses for receipt signing.  Never duplicate that secret into the
    subprocess argument vector.  Direct-key development mode has no wallet to
    derive from and therefore retains the explicit key pair.
    """

    from verallm.api.validator_auth import DEFAULT_VALIDATORS_PATH

    configured = _set_server_arg(server_args, "--server-role", "coordinator")
    wallet_backed = bool(getattr(args, "wallet", None))
    if wallet_backed:
        configured = _set_server_arg(
            configured,
            "--wallet-name",
            str(args.wallet),
        )
        configured = _set_server_arg(
            configured,
            "--wallet-hotkey",
            str(args.hotkey),
        )
        configured = _remove_server_arg(configured, "--evm-private-key")
    if evm_address:
        configured = _set_server_arg(configured, "--evm-address", evm_address)
    if evm_private_key and not wallet_backed:
        configured = _set_server_arg(
            configured,
            "--evm-private-key",
            evm_private_key,
        )

    # Wallet mode has a metagraph and therefore always protects public mesh
    # inference routes with the same allowlist as the vLLM server.  Explicit
    # --validator-auth remains available for controlled non-wallet setups.
    validator_auth = wallet_backed or (
        "--validator-auth" in configured
    )
    if validator_auth:
        configured = _add_server_flag(configured, "--validator-auth")
        configured = _add_server_flag(configured, "--require-validator-nonce")
        configured = _set_server_arg(
            configured,
            "--validator-allowlist-path",
            os.environ.get("VERATHOS_VALIDATORS_PATH", DEFAULT_VALIDATORS_PATH),
        )
    return configured


def _validator_allowlist_refresh_enabled(args) -> bool:
    """Wallet-backed vLLM and mesh miners both maintain the allowlist."""

    return bool(getattr(args, "wallet", None))


def _forward_proof_v2_artifacts(
    server_args: list[str],
    *,
    manifest: str | None,
    weight_catalog: str | None,
    artifact_base_urls: list[str] | None = None,
    artifact_cache_dir: str | None = None,
) -> list[str]:
    """Forward configured proof-v2 artifacts to the inner miner server.

    Arguments supplied directly to the inner server after ``--`` take
    precedence over wrapper environment defaults.
    """
    out = list(server_args)
    for flag, value in (
        ("--proof-v2-manifest", manifest),
        ("--proof-v2-weight-catalog", weight_catalog),
        ("--proof-v2-artifact-cache-dir", artifact_cache_dir),
    ):
        if value and flag not in out:
            out.extend([flag, value])
    if (
        artifact_base_urls
        and "--proof-v2-artifact-base-url" not in out
    ):
        for value in artifact_base_urls:
            if value:
                out.extend(["--proof-v2-artifact-base-url", value])
    return out


def _forward_proof_v3_artifacts(
    server_args: list[str],
    *,
    manifest: str | None,
    execution_profile: str | None,
    calibration_set: str | None,
    attention_semantics: str | None,
    gdn_semantics: str | None,
    lm_head_catalog: str | None,
    projection_manifest: str | None,
    projection_catalog: str | None,
    runtime_encoding: str | None,
    weight_cache_dir: str | None,
) -> list[str]:
    """Forward one explicit authenticated proof-v3 release."""

    out = list(server_args)
    for flag, value in (
        ("--proof-v3-manifest", manifest),
        ("--proof-v3-execution-profile", execution_profile),
        ("--proof-v3-calibration-set", calibration_set),
        ("--proof-v3-attention-semantics", attention_semantics),
        ("--proof-v3-gdn-semantics", gdn_semantics),
        ("--proof-v3-lm-head-catalog", lm_head_catalog),
        ("--proof-v3-projection-manifest", projection_manifest),
        ("--proof-v3-projection-catalog", projection_catalog),
        ("--proof-v3-runtime-encoding", runtime_encoding),
        ("--proof-v3-weight-cache-dir", weight_cache_dir),
    ):
        if value and flag not in out:
            out.extend([flag, value])
    return out


def _apply_proof_v3_release_descriptor(args, descriptor_path: str) -> None:
    """Expand one authenticated release descriptor into server arguments."""

    from verallm.proof_v3.economic_release_catalog import (
        load_proof_v3_release_descriptor,
        proof_v3_release_artifact_paths,
    )

    source, value = load_proof_v3_release_descriptor(descriptor_path)
    _source, paths = proof_v3_release_artifact_paths(source)
    attributes = {
        "proof_v3_manifest": "manifest",
        "proof_v3_execution_profile": "execution_profile",
        "proof_v3_calibration_set": "calibration_set",
        "proof_v3_attention_semantics": "attention_runtime_semantics",
        "proof_v3_gdn_semantics": "gdn_runtime_semantics",
        "proof_v3_lm_head_catalog": "lm_head_catalog",
        "proof_v3_projection_manifest": "projection_manifest",
        "proof_v3_projection_catalog": "projection_catalog",
    }
    explicitly_configured = tuple(
        name
        for name in attributes
        if str(getattr(args, name, None) or "").strip()
    )
    if explicitly_configured:
        raise RuntimeError(
            "proof-v3 release descriptor conflicts with explicit artifact "
            "paths: " + ", ".join(explicitly_configured)
        )
    for attribute, role in attributes.items():
        setattr(
            args,
            attribute,
            str(paths[role]) if role in paths else None,
        )
    existing_encoding = str(
        getattr(args, "proof_v3_runtime_encoding", None) or ""
    ).strip()
    descriptor_encoding = str(value["runtime_encoding_id"])
    if existing_encoding and existing_encoding != descriptor_encoding:
        raise RuntimeError(
            "proof-v3 release descriptor conflicts with the configured "
            "runtime encoding"
        )
    args.proof_v3_runtime_encoding = descriptor_encoding


def _capacity_audit_state_path(evm_address: str | None, port: int) -> str:
    address = "".join(
        c for c in str(evm_address or "unknown").lower()
        if c.isalnum() or c in ("x",)
    )[:48] or "unknown"
    return f"/tmp/verathos_capacity_audit_{address}_{int(port)}.json"


def _check_external_port(endpoint: str, local_bind_port: int | None = None) -> None:
    """Verify the miner's endpoint port is reachable from the internet.

    Starts a temporary TCP listener on the local server port, asks external
    services to probe the public endpoint port, then shuts it down — all before
    vLLM loads. The local bind port can differ from the public endpoint port on
    providers such as RunPod that forward a public port to container port 8000.
    If a service confirms the public port is closed, abort with a clear error.
    If all services are unreachable, log a warning and continue (skip).
    """
    from urllib.parse import urlparse
    import socket as _socket
    import threading as _threading
    from http.server import HTTPServer, BaseHTTPRequestHandler

    parsed = urlparse(endpoint)
    host = parsed.hostname
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    bind_port = int(local_bind_port or port)

    if not host:
        bt.logging.warning("Cannot parse endpoint host — skipping external port check")
        return

    bt.logging.info(f"Checking if port {port} on {host} is reachable from the internet...")
    if bind_port != port:
        bt.logging.info(f"External port check: binding temporary local listener on port {bind_port}")

    # Start a temporary HTTP server so external services have something to connect to.
    # Runs in a background thread, shut down after the check.
    class _SilentHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")
        def log_message(self, *args):
            pass  # silence

    tmp_server = None
    tmp_thread = None
    try:
        tmp_server = HTTPServer(("0.0.0.0", bind_port), _SilentHandler)
        tmp_thread = _threading.Thread(target=tmp_server.serve_forever, daemon=True)
        tmp_thread.start()
        time.sleep(0.5)  # let it bind
    except OSError as e:
        # Port already in use (maybe server already running) — skip temp server,
        # the real server will handle it
        bt.logging.debug(f"Could not start temp listener on port {port}: {e} — port may already be in use")
        tmp_server = None

    # Try multiple external port-check services for reliability.
    # Each returns (responded: bool, port_open: bool).
    checks: list[tuple[str, bool, bool]] = []

    # Service 1: yougetsignal.com (returns HTML with "is open" or "is closed")
    try:
        resp = httpx.post(
            "https://ports.yougetsignal.com/check-port.php",
            data={"remoteAddress": host, "portNumber": str(port)},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=10.0,
        )
        if resp.status_code == 200:
            body = resp.text.lower()
            is_open = "is open" in body
            checks.append(("yougetsignal", True, is_open))
        else:
            checks.append(("yougetsignal", False, False))
    except Exception:
        checks.append(("yougetsignal", False, False))

    # Service 2: portchecker.io
    try:
        resp = httpx.get(
            f"https://portchecker.io/api/v1/query?host={host}&ports={port}",
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=10.0,
        )
        if resp.status_code == 200:
            body = resp.json()
            # Response: {"host": ..., "ports": [{"port": N, "status": "open"|"closed"}]}
            ports_list = body.get("ports", [])
            is_open = any(p.get("status") == "open" for p in ports_list)
            checks.append(("portchecker.io", True, is_open))
        else:
            checks.append(("portchecker.io", False, False))
    except Exception:
        checks.append(("portchecker.io", False, False))

    # Shut down temp server before returning
    def _cleanup():
        if tmp_server is not None:
            tmp_server.shutdown()

    responded = [(name, is_open) for name, ok, is_open in checks if ok]
    if not responded:
        _cleanup()
        bt.logging.warning(
            "External port check: no check service reachable. "
            "Skipping — port may or may not be open."
        )
        return

    if any(is_open for _, is_open in responded):
        _cleanup()
        bt.logging.success(f"External port check passed: {host}:{port} is reachable")
        return

    # All responding services say port is closed
    _cleanup()
    services_str = ", ".join(name for name, _ in responded)
    bt.logging.error(
        f"\n{'=' * 60}\n"
        f"  EXTERNAL PORT CHECK FAILED\n"
        f"  Port {port} on {host} is NOT reachable from the internet.\n"
        f"  (Checked via: {services_str})\n\n"
        f"  Your miner server is running but nobody can connect to it.\n"
        f"  Common causes:\n"
        f"    - Firewall blocking the port (check: sudo ufw allow {port}/tcp)\n"
        f"    - Cloud provider security group missing inbound rule\n"
        f"    - NAT/router not forwarding port {port} to this machine\n\n"
        f"  Registration aborted — fix the port and restart.\n"
        f"{'=' * 60}"
    )
    sys.exit(1)


def _extract_hotkey_seed(wallet_name: str, hotkey_name: str, wallet) -> bytes:
    """Extract the 32-byte Ed25519 seed from a Bittensor hotkey.

    Works across bittensor v9 (wallet.hotkey.private_key) and v10+ (keyfile JSON).
    """
    # v9: direct attribute access
    if hasattr(wallet.hotkey, "private_key"):
        return bytes(wallet.hotkey.private_key[:32])

    # v10+: read from keyfile JSON
    import json
    from pathlib import Path
    hk_path = Path.home() / f".bittensor/wallets/{wallet_name}/hotkeys/{hotkey_name}"
    hk_data = json.loads(hk_path.read_text())
    return bytes.fromhex(hk_data["secretSeed"].replace("0x", ""))


class MinerNeuron:
    """Wraps the existing VeraLLM miner server with Bittensor chain registration."""

    def __init__(self, config: NeuronConfig):
        self.config = config
        self._provider: Optional[Web3Provider] = None
        self._miner_client: Optional[MinerRegistryClient] = None
        self._server_process: Optional[subprocess.Popen] = None
        self._capacity_audit_worker = None
        self._proof_v3_artifact_watcher: Optional[
            ProofV3ArtifactRefreshWatcher
        ] = None
        self._subnet_runtime_config_client = (
            RuntimeSubnetConfigClient.from_config(config, log=bt.logging)
        )
        self._proof_v3_configured = False
        self._served_proof_protocol_versions: tuple[int, ...] = ()
        self._managed_nginx_timeout_seconds: int | None = None
        self._running = True

        self.evm_pk = ""
        self.evm_addr = ""
        self.hotkey_ss58 = ""
        self.uid: Optional[int] = None  # resolved from metagraph during setup()

    def setup(self, private_key: Optional[str] = None):
        """Initialize chain connection and derive EVM credentials.

        Args:
            private_key: Raw EVM private key (hex). If provided, skips
                bittensor wallet derivation (useful for Anvil testing).
                If None, derives from the bittensor wallet.
        """
        if private_key:
            # Direct private key mode (Anvil / testing)
            from eth_account import Account
            pk = private_key if private_key.startswith("0x") else f"0x{private_key}"
            self.evm_pk = pk
            self.evm_addr = Account.from_key(pk).address
            bt.logging.info(f"EVM address (from --private-key): {self.evm_addr}")
        else:
            # Wallet mode (real testnet / mainnet).  bittensor is already
            # imported at module level (line 39); the previous local
            # `import bittensor as bt` shadowed it, and the corresponding
            # try/except was unreachable AND broken — `bt.logging.error`
            # in the except branch ran before `bt` was bound, raising
            # NameError instead of the intended graceful exit.
            WalletCls = getattr(bt, "Wallet", None) or bt.wallet
            wallet = WalletCls(name=self.config.wallet_name, hotkey=self.config.hotkey_name)
            hotkey_seed = _extract_hotkey_seed(
                self.config.wallet_name, self.config.hotkey_name, wallet,
            )
            self.hotkey_seed = hotkey_seed
            self.hotkey_ss58 = wallet.hotkey.ss58_address

            self.evm_pk = derive_evm_private_key(hotkey_seed)
            self.evm_addr = derive_evm_address(hotkey_seed)
            bt.logging.info(f"EVM address (from wallet): {self.evm_addr}")

        self._provider = Web3Provider(self.config)
        self._miner_client = MinerRegistryClient(self.config, provider=self._provider)

        if not private_key:
            # Resolve UID from Substrate metagraph — needed for registerEvm().
            self.uid = self._resolve_uid_with_retry()
            bt.logging.info(f"Resolved UID: {self.uid}")
        else:
            bt.logging.info("Anvil mode — skipping EVM association check")

    def _resolve_uid_with_retry(self, max_attempts: int = 10) -> Optional[int]:
        """Resolve our UID from the Substrate metagraph with retry.

        Retries with exponential backoff on transient errors (429, timeouts).
        Returns None only in Anvil mode (no wallet).
        """
        import random

        WalletCls = getattr(bt, "Wallet", None) or bt.wallet
        wallet = WalletCls(
            name=self.config.wallet_name, hotkey=self.config.hotkey_name,
        )
        hk_ss58 = wallet.hotkey.ss58_address

        for attempt in range(1, max_attempts + 1):
            try:
                sub = bt.Subtensor(network=self.config.subtensor_network)
                mg = sub.metagraph(self.config.netuid)
                for uid_val in range(len(mg.hotkeys)):
                    if mg.hotkeys[uid_val] == hk_ss58:
                        bt.logging.info(f"UID {uid_val} resolved from metagraph (SS58={hk_ss58})")
                        return uid_val
                # Hotkey not found in metagraph — not registered on subnet
                bt.logging.error(f"Hotkey {hk_ss58} not found in metagraph for netuid {self.config.netuid}. Is this miner registered on the subnet?")
                return None
            except Exception as e:
                if attempt == max_attempts:
                    bt.logging.error(f"Failed to resolve UID after {max_attempts} attempts: {e}. Cannot proceed without UID.")
                    raise RuntimeError(
                        f"Cannot resolve UID from metagraph after {max_attempts} "
                        f"attempts: {e}"
                    ) from e
                delay = min(2 ** attempt * 3, 60) + random.uniform(0, 5)
                bt.logging.warning(f"Metagraph query failed (attempt {attempt}/{max_attempts}): {e} — retrying in {delay:.0f}s")
                time.sleep(delay)

    def _ensure_evm_registered(self):
        """Ensure the miner's EVM address is bound to its current UID.

        Uses self.uid (resolved at startup from Substrate metagraph).
        UID slots can be recycled and a hotkey can later register under a new
        UID, so an existing EVM registration alone is not sufficient. Both
        contract mapping directions must match the current pair.

        Retries with backoff — never silently gives up.
        """
        import random

        if self.uid is None:
            # Anvil mode or UID not resolvable — skip
            return

        for attempt in range(1, 11):
            try:
                associated_uid = self._miner_client.get_associated_uid(
                    self.evm_addr,
                    refresh=True,
                )
                registered_uid = self._miner_client.get_registered_uid_for_evm(
                    self.evm_addr,
                    refresh=True,
                )
                registered_evm = self._miner_client.get_registered_evm_for_uid(
                    self.uid,
                    refresh=True,
                )
                evm_matches = (
                    registered_evm is not None
                    and registered_evm.lower() == self.evm_addr.lower()
                )
                if (
                    associated_uid == self.uid
                    and registered_uid == self.uid
                    and evm_matches
                ):
                    bt.logging.info(
                        f"EVM already registered on MinerRegistry (UID={self.uid})"
                    )
                    return

                if registered_uid is not None or registered_evm is not None:
                    bt.logging.warning(
                        "Stale MinerRegistry EVM binding detected: "
                        f"resolved UID={associated_uid}, registered UID={registered_uid}, "
                        f"UID {self.uid} maps to {registered_evm or 'no EVM'}; "
                        "repairing current binding"
                    )
                else:
                    bt.logging.info(
                        f"Registering EVM -> UID {self.uid} on MinerRegistry"
                    )

                self._miner_client.register_evm(
                    self.uid,
                    hotkey_seed=self.hotkey_seed,
                    netuid=self.config.netuid,
                    private_key=self.evm_pk,
                )

                verified_uid = self._miner_client.get_registered_uid_for_evm(
                    self.evm_addr,
                    refresh=True,
                )
                verified_associated_uid = self._miner_client.get_associated_uid(
                    self.evm_addr,
                    refresh=True,
                )
                verified_evm = self._miner_client.get_registered_evm_for_uid(
                    self.uid,
                    refresh=True,
                )
                if (
                    verified_associated_uid != self.uid
                    or verified_uid != self.uid
                    or not (
                        verified_evm
                        and verified_evm.lower() == self.evm_addr.lower()
                    )
                ):
                    raise RuntimeError(
                        "registerEvm receipt succeeded but the current binding "
                        f"was not visible: resolved UID={verified_associated_uid}, "
                        f"registered UID={verified_uid}, UID {self.uid} maps to "
                        f"{verified_evm or 'no EVM'}"
                    )
                bt.logging.info(f"registerEvm({self.uid}) succeeded")
                return
            except Exception as e:
                if attempt == 10:
                    raise RuntimeError(
                        "Cannot reconcile EVM registration for "
                        f"UID {self.uid} after 10 attempts: {e}"
                    ) from e
                delay = min(2 ** attempt * 3, 60) + random.uniform(0, 5)
                bt.logging.warning(
                    "EVM registration reconciliation failed "
                    f"(attempt {attempt}/10): {e} — retrying in {delay:.0f}s"
                )
                time.sleep(delay)

    # File the server writes when it auto-tunes max_num_seqs for Mamba/GDN
    # models (Qwen3.5, Qwen3.6).  See verallm/miner/vllm_backend.py — when
    # vLLM raises "max_num_seqs exceeds available Mamba cache blocks" on the
    # first attempt, the worker writes the discovered block count here and
    # exits with code 42 so we can relaunch with --max-num-seqs N (in-process
    # retry is impossible because the failed init pins GPU memory we can't
    # release without a fresh process).
    _MAMBA_HINT_PATH = "/tmp/verathos_mamba_max_num_seqs"
    _MAMBA_HINT_EXIT = 42
    _AWQ_GEMM_HINT_PATH = "/tmp/verathos_awq_gemm_fallback"
    _AWQ_GEMM_HINT_EXIT = 43

    def _server_cmd(self, server_args: list[str]) -> list[str]:
        if getattr(self, "mesh_dir", None):
            # GGUF-mesh runtime: supervise the mesh coordinator instead of
            # the vLLM server. server_args are `mesh serve` arguments.
            return [sys.executable, "-m", "neurons.cli", "mesh", "serve",
                    "--mesh", self.mesh_dir] + server_args
        return [sys.executable, "-m", "verallm.api.server"] + server_args

    @staticmethod
    def _redact_cmd(cmd: list[str]) -> str:
        _safe = []
        _skip = False
        for a in cmd:
            if _skip:
                _safe.append("***")
                _skip = False
            elif a in ("--evm-private-key", "--private-key"):
                _safe.append(a)
                _skip = True
            else:
                _safe.append(a)
        return " ".join(_safe)

    @staticmethod
    def _server_preexec():
        try:
            import ctypes
            import signal as _signal

            libc = ctypes.CDLL("libc.so.6")
            libc.prctl(1, _signal.SIGTERM)
        except Exception:
            pass

    def start_server(self, server_args: list[str]):
        """Start the VeraLLM miner server as a subprocess.

        Auto-handles the Mamba/GDN max_num_seqs auto-tune: if the first
        launch exits with code 42 and leaves a hint file, we re-spawn the
        subprocess with a hardware-bounded --max-num-seqs N derived from the
        available Mamba cache-block count vLLM reported. See
        _MAMBA_HINT_PATH above.
        This makes Qwen3.5 / Qwen3.6 / any future Mamba-hybrid model start
        out-of-the-box on any GPU size without operator intervention.
        """
        # Clear any stale hint from a previous run (different GPU, different
        # gpu_memory_utilization, etc. — the hint must match the current
        # process or we'd apply a stale max_num_seqs).
        try:
            import os as _os
            if _os.path.exists(self._MAMBA_HINT_PATH):
                _os.remove(self._MAMBA_HINT_PATH)
            if _os.path.exists(self._AWQ_GEMM_HINT_PATH):
                _os.remove(self._AWQ_GEMM_HINT_PATH)
        except Exception:
            pass

        cmd = self._server_cmd(server_args)
        bt.logging.info(f"Starting miner server: {self._redact_cmd(cmd)}")
        self._server_process = subprocess.Popen(
            cmd,
            start_new_session=True,
            preexec_fn=self._server_preexec if os.name == "posix" else None,
        )

    def _read_mamba_hint(self) -> Optional[int]:
        try:
            with open(self._MAMBA_HINT_PATH) as _hf:
                _v = _hf.read().strip()
            return int(_v) if _v else None
        except Exception:
            return None

    @staticmethod
    def _apply_max_num_seqs(server_args: list[str], n: int) -> list[str]:
        """Insert --max-num-seqs <n> into server_args, replacing any prior."""
        out: list[str] = []
        skip = False
        for a in server_args:
            if skip:
                skip = False
                continue
            if a == "--max-num-seqs":
                skip = True
                continue
            out.append(a)
        out.extend(["--max-num-seqs", str(n)])
        return out

    def _read_awq_gemm_hint(self) -> bool:
        try:
            with open(self._AWQ_GEMM_HINT_PATH) as _hf:
                return _hf.read().strip() == "1"
        except Exception:
            return False

    @staticmethod
    def _apply_awq_gemm_fallback(server_args: list[str]) -> list[str]:
        out = list(server_args)
        if "--awq-gemm-fallback" not in out:
            out.append("--awq-gemm-fallback")
        # AWQ-Marlin and plain AWQ GEMM can expose different loaded-weight
        # layouts to root computation. Recompute the ModelSpec on fallback
        # instead of reusing a cache produced by the failed backend.
        if "--no-cache" not in out:
            out.append("--no-cache")
        return out

    def wait_for_health(
        self,
        endpoint: str,
        server_args: Optional[list[str]] = None,
        max_mamba_retries: int = 1,
        max_awq_gemm_retries: int = 1,
    ):
        """Poll /health until the server is ready. No timeout — model load + Merkle tree
        can take 30+ minutes for large models.

        If the subprocess exits before /health responds, check whether it
        emitted a Mamba auto-tune hint (exit 42 + hint file): on the FIRST
        such occurrence we relaunch the subprocess with --max-num-seqs N
        and resume polling.  Subsequent unexpected exits raise RuntimeError
        — PM2 (or whatever supervisor wraps the miner) will then handle
        the outer-level restart.
        """
        url = f"{endpoint}/health"
        start = time.monotonic()
        last_log = start
        _mamba_retries_done = 0
        _awq_gemm_retries_done = 0
        while True:
            # Detect subprocess death.  If the server exited cleanly via the
            # Mamba auto-tune path (exit 42 + hint file), relaunch with
            # --max-num-seqs N once.  Any other exit -> propagate so the
            # supervisor can react.
            if self._server_process is not None and self._server_process.poll() is not None:
                rc = self._server_process.returncode
                if (
                    rc == self._MAMBA_HINT_EXIT
                    and server_args is not None
                    and _mamba_retries_done < max_mamba_retries
                ):
                    _hint = self._read_mamba_hint()
                    if _hint is not None and _hint > 0:
                        bt.logging.warning(
                            "Server subprocess exited with Mamba auto-tune hint "
                            "(max_num_seqs=%d).  Relaunching with --max-num-seqs=%d.",
                            _hint, _hint,
                        )
                        try:
                            import os as _os
                            _os.remove(self._MAMBA_HINT_PATH)
                        except Exception:
                            pass
                        new_args = self._apply_max_num_seqs(list(server_args), _hint)
                        # Mutate caller's list in place so any subsequent
                        # restart (e.g. auto-update) keeps the tuned flag.
                        server_args[:] = new_args
                        self.start_server(server_args)
                        _mamba_retries_done += 1
                        start = time.monotonic()
                        last_log = start
                        continue
                if (
                    rc == self._AWQ_GEMM_HINT_EXIT
                    and server_args is not None
                    and _awq_gemm_retries_done < max_awq_gemm_retries
                    and self._read_awq_gemm_hint()
                ):
                    bt.logging.warning(
                        "Server subprocess exited with AWQ GEMM fallback hint. "
                        "Relaunching with --awq-gemm-fallback."
                    )
                    try:
                        import os as _os
                        _os.remove(self._AWQ_GEMM_HINT_PATH)
                    except Exception:
                        pass
                    new_args = self._apply_awq_gemm_fallback(list(server_args))
                    server_args[:] = new_args
                    self.start_server(server_args)
                    _awq_gemm_retries_done += 1
                    start = time.monotonic()
                    last_log = start
                    continue
                raise RuntimeError(
                    f"Miner server subprocess exited (returncode={rc}) before "
                    f"/health became ready"
                )

            try:
                resp = httpx.get(url, timeout=5.0)
                if resp.status_code == 200:
                    elapsed = time.monotonic() - start
                    bt.logging.info(f"Miner server healthy after {elapsed:.0f}s")
                    return
            except Exception:
                pass
            now = time.monotonic()
            if now - last_log >= 60.0:
                bt.logging.info(f"Waiting for miner server... ({now - start:.0f}s elapsed)")
                last_log = now
            time.sleep(2.0)

    def query_actual_max_context(self, endpoint: str) -> Optional[int]:
        """Query the running server for the actual max single-request context.

        After vLLM loads the model, the real KV cache pool size is known.
        The actual max context a single request can use is::

            min(kv_pool_tokens, max_model_len)

        Returns None if the server doesn't expose batch-mode health fields
        (e.g. running in single-request mode).
        """
        try:
            resp = httpx.get(f"{endpoint}/health", timeout=5.0)
            if resp.status_code != 200:
                return None
            health = resp.json()
            kv_pool = health.get("kv_pool_tokens")
            max_context = health.get("max_context")  # = max_model_len
            if kv_pool is None or max_context is None:
                return None
            return min(kv_pool, max_context)
        except Exception as e:
            bt.logging.warning(f"Could not query server for actual context: {e}")
            return None

    def _proof_v3_refresh_busy_state(
        self,
        endpoint: str,
    ) -> tuple[int, int, bool, bool]:
        """Return bounded local work state for authenticated release adoption."""

        capacity_active = False
        worker = self._capacity_audit_worker
        if worker is not None:
            try:
                capacity_active = bool(worker._has_active_local_audit())
            except Exception:
                # An unreadable audit state must delay, never authorize, a
                # restart that could destroy an active capacity witness.
                capacity_active = True
        proc = self._server_process
        if proc is None or proc.poll() is not None:
            return 0, 0, False, capacity_active
        try:
            response = httpx.get(f"{endpoint}/health", timeout=5.0)
            response.raise_for_status()
            health = response.json()
            active_requests = max(0, int(health.get("active_requests", 0)))
            proof_pending = max(0, int(health.get("proof_pending", 0)))
            hard_exclusive = bool(health.get("hard_proof_exclusive", False))
            return (
                active_requests,
                proof_pending,
                hard_exclusive,
                capacity_active,
            )
        except Exception:
            # A live but unreadable server is treated as busy. Validators
            # independently enforce the authenticated release, so update
            # adoption never needs to destroy an honest in-flight request.
            return 1, 0, False, capacity_active

    def _auto_update_busy(self, endpoint: str) -> bool:
        """Never restart through live inference, proof, or capacity work."""

        active, proof_pending, hard_exclusive, capacity_active = (
            self._proof_v3_refresh_busy_state(endpoint)
        )
        return bool(
            active > 0
            or proof_pending > 0
            or hard_exclusive
            or capacity_active
        )

    def start_proof_v3_artifact_refresh(
        self,
        *,
        model_id: str,
        base_urls: tuple[str, ...],
        chain_config,
        model_registry_client,
        cache_directory: str | None,
        current_release_sha256: bytes,
        local_health_url: str,
    ) -> None:
        """Watch one remote release and restart onto authenticated changes."""

        if self._proof_v3_artifact_watcher is not None:
            return
        from neurons.auto_update import restart_process
        from verallm.proof_v3.artifact_store import (
            probe_remote_proof_v3_release,
            resolve_remote_proof_v3_release,
        )

        def _probe():
            return probe_remote_proof_v3_release(
                model_id,
                base_urls,
                chain_config=chain_config,
                cache_directory=cache_directory,
            )

        def _resolve():
            return resolve_remote_proof_v3_release(
                model_id,
                base_urls,
                chain_config=chain_config,
                model_registry_client=model_registry_client,
                cache_directory=cache_directory,
            )

        watcher = ProofV3ArtifactRefreshWatcher(
            current_release_sha256=current_release_sha256,
            probe_release=_probe,
            resolve_release=_resolve,
            busy_state=lambda: self._proof_v3_refresh_busy_state(
                local_health_url
            ),
            restart=restart_process,
        )
        self._proof_v3_artifact_watcher = watcher
        watcher.start()

    @staticmethod
    def _ctx_close_enough(old: int, new: int, tolerance: float = 0.10) -> bool:
        """Return True if two context lengths are within tolerance (default 10%).

        KV pool size varies between restarts due to VRAM fragmentation,
        CUDA graph memory overhead, etc.  A 10% tolerance absorbs the
        typical variation (e.g. 225k → 215k = 4.7%) without creating
        unnecessary new on-chain entries that cause index drift.
        """
        if old == 0:
            return False
        return abs(old - new) / old <= tolerance

    def check_existing_registration(
        self,
        model_id: str,
        endpoint: str,
        quant: str,
        max_context_len: int,
    ) -> Optional[int]:
        """Check existing registrations and clean up stale entries.

        Returns the matching model index if already registered with the same
        params (caller should skip re-registration), or None if a new
        registration is needed.

        Only deactivates entries that match on model_id + endpoint + quant but
        have a context length that changed beyond tolerance (restart noise with
        a significantly different KV pool).  Entries with a different quant or
        endpoint are left alone — a miner may legitimately serve the same model
        in multiple quants (int4 + fp16) or on different endpoints.

        Context length comparison uses a 2% tolerance — small KV pool
        variations between restarts are noise (VRAM fragmentation, CUDA graph
        memory, etc.) and don't warrant a new on-chain entry.
        """
        try:
            models = self._miner_client.get_miner_models(self.evm_addr)
        except Exception as e:
            # If we can't query existing entries, raise instead of silently
            # returning None — otherwise the caller registers a duplicate.
            raise RuntimeError(
                f"Cannot check existing registrations (chain RPC error): {e}"
            ) from e

        matching_index: Optional[int] = None

        for i, m in enumerate(models):
            if not m.active:
                continue
            # Different model but same endpoint — stale entry from a model change.
            # Deactivate it so the validator doesn't route to an endpoint serving
            # a different model than what's registered.
            if m.model_id != model_id and m.endpoint == endpoint:
                bt.logging.info(f"Deactivating stale entry at index {i}: model changed from {m.model_id} to {model_id} at same endpoint {endpoint}")
                self._deactivate_with_retry(i)
                continue
            # Different model, different endpoint — leave it alone
            if m.model_id != model_id:
                continue
            # Same model + quant — check if this is "our" slot
            if m.quant != quant:
                continue  # different quant for same model — leave it alone

            if m.endpoint != endpoint:
                # Different endpoint for same model+quant. If --update-endpoint
                # is set, only update if the host matches (port/scheme change on
                # the same machine). Never touch entries from other IPs — those
                # belong to other GPU instances.
                if getattr(self.config, "update_endpoint", False) and matching_index is None:
                    from urllib.parse import urlparse
                    old_host = urlparse(m.endpoint).hostname or ""
                    new_host = urlparse(endpoint).hostname or ""
                    if old_host == new_host:
                        bt.logging.info(f"Endpoint changed at index {i}: {m.endpoint} → {endpoint} — updating on-chain")
                        try:
                            self._miner_client.update_endpoint(i, endpoint, private_key=self.evm_pk)
                            bt.logging.info(f"Endpoint updated at index {i}")
                            matching_index = i
                        except Exception as e:
                            bt.logging.warning(f"updateEndpoint failed: {e} — will register new entry")
                continue

            # Same model, same endpoint, same quant — this is "our" slot
            if self._ctx_close_enough(m.max_context_len, max_context_len):
                if matching_index is not None:
                    # Duplicate active entry — deactivate the older one
                    bt.logging.info(f"Deactivating duplicate entry at index {matching_index} (keeping newer index {i})")
                    self._deactivate_with_retry(matching_index)
                # Context within tolerance — reuse this entry
                # Check if the lease expired (active flag is True in
                # struct but isModelActive() returns False when
                # expiresAt < now).  Renew immediately if so.
                if m.expires_at <= int(time.time()):
                    bt.logging.info(f"Lease expired at index {i} (expiresAt={m.expires_at}, {(time.time() - m.expires_at) / 3600:.1f}h ago) — deactivating and re-registering")
                    # Contract rejects renewModel() on fully expired
                    # leases ("Already expired, re-register").
                    # Deactivate the stale entry so register_on_chain()
                    # creates a fresh one.  Wait for tx confirmation
                    # before returning — the contract checks the raw
                    # active flag, not isModelActive().
                    self._deactivate_with_retry(i)
                    # Wait for deactivation to be mined
                    for _ in range(10):
                        time.sleep(3)
                        try:
                            updated = self._miner_client.get_miner_models(
                                self.evm_addr
                            )
                            if i < len(updated) and not updated[i].active:
                                bt.logging.info(f"Deactivation confirmed at index {i}")
                                break
                        except Exception:
                            pass
                    return None  # force new registration
                else:
                    bt.logging.info(f"Already registered with same config at index {i} (model={model_id}, quant={quant}, ctx={m.max_context_len} vs {max_context_len} — within tolerance) — skipping")
                matching_index = i
            else:
                # Same slot but context changed significantly — stale entry
                bt.logging.info(f"Deactivating stale entry at index {i} (same model/endpoint/quant, ctx changed: {m.max_context_len} → {max_context_len})")
                self._deactivate_with_retry(i)

        return matching_index

    def _renew_with_retry(self, index: int, max_attempts: int = 3) -> None:
        """Renew an expired model lease with retry on transient errors."""
        for attempt in range(1, max_attempts + 1):
            try:
                tx = self._miner_client.renew_model(index, private_key=self.evm_pk)
                bt.logging.info(f"Renewed expired lease at index {index}: {tx}")
                return
            except Exception as e:
                if attempt == max_attempts:
                    bt.logging.warning(f"Failed to renew entry {index} after {max_attempts} attempts: {e}")
                else:
                    time.sleep(2 ** attempt * 5)

    def _deactivate_with_retry(self, index: int, max_attempts: int = 3) -> None:
        """Deactivate a model entry with retry on transient errors."""
        for attempt in range(1, max_attempts + 1):
            try:
                self._miner_client.deactivate_model(index, private_key=self.evm_pk)
                return
            except Exception as e:
                if attempt == max_attempts:
                    bt.logging.warning(f"Failed to deactivate entry {index} after {max_attempts} attempts: {e}")
                else:
                    time.sleep(2 ** attempt * 5)

    def _parse_model_index_from_receipt(self, tx_hash: str) -> Optional[int]:
        """Parse ModelRegistered/ModelReactivated event from tx receipt.

        Returns the on-chain array index of the registered/reactivated entry.
        The contract emits exactly one of:
          - ModelReactivated(miner, index, modelId) — reused existing slot
          - ModelRegistered(miner, modelId, endpoint) — pushed new entry

        ModelReactivated contains the index directly.  ModelRegistered means
        a new push, so index = array_length - 1.
        """
        try:
            w3 = self._miner_client._provider.w3
            contract = self._miner_client._contract
            receipt = w3.eth.get_transaction_receipt(tx_hash)

            for log in receipt.get("logs", []):
                # Try ModelReactivated first (has explicit index)
                try:
                    event = contract.events.ModelReactivated().process_log(log)
                    index = event["args"]["index"]
                    bt.logging.info(f"Contract reactivated existing slot at index {index}")
                    return index
                except Exception:
                    pass
                # Try ModelRegistered (new push → count - 1)
                try:
                    contract.events.ModelRegistered().process_log(log)
                    count = self._miner_client.get_miner_model_count(self.evm_addr)
                    bt.logging.info(f"Contract created new entry at index {count - 1}")
                    return count - 1
                except Exception:
                    pass
        except Exception as e:
            bt.logging.warning(f"Failed to parse model index from receipt: {e}")
        return None

    def _find_registered_model_index(
        self,
        model_id: str,
        endpoint: str,
        quant: str,
        max_context_len: int,
    ) -> Optional[int]:
        """Find this exact active model slot from chain state.

        Public RPCs can lag immediately after a successful registration tx:
        the tx submission path may return before ``eth_getTransactionReceipt``
        can serve the receipt, and a raw ``count - 1`` read can be stale.  The
        capacity-audit worker needs the exact on-chain model index, so fall
        back to querying the miner's model array and matching the slot fields
        validators also use for discovery.
        """
        try:
            models = self._miner_client.get_miner_models(self.evm_addr)
        except Exception as e:
            bt.logging.warning(f"Could not resolve registered model index from chain state: {e}")
            return None

        match: Optional[int] = None
        for i, m in enumerate(models):
            if not getattr(m, "active", False):
                continue
            if m.model_id != model_id:
                continue
            if m.endpoint != endpoint:
                continue
            if m.quant != quant:
                continue
            if not self._ctx_close_enough(m.max_context_len, max_context_len):
                continue
            match = i
        return match

    def register_on_chain(
        self,
        model_id: str,
        endpoint: str,
        quant: str,
        max_context_len: int,
        max_retries: int = 5,
    ) -> int:
        """Register this miner's model on MinerRegistry.

        Returns the on-chain model index for use in heartbeat renewals.

        The entire check→register flow is wrapped in a single retry loop so
        that a transient RPC failure during the pre-check never causes a
        blind registration (which would create duplicates).
        """
        from web3 import Web3

        for attempt in range(1, max_retries + 1):
            try:
                # Step 1: Reconcile EVM -> UID before accepting an existing
                # model entry. A hotkey may have moved to a new UID while its
                # model lease remained active under the same EVM address.
                self._ensure_evm_registered()

                # Step 2: Check for existing model registration (must succeed)
                existing_index = self.check_existing_registration(
                    model_id, endpoint, quant, max_context_len,
                )
                if existing_index is not None:
                    return existing_index

                # Step 3: Register
                spec_ref = Web3.solidity_keccak(["string"], [model_id])
                bt.logging.info(f"Registering on-chain: model={model_id} endpoint={endpoint} quant={quant} ctx={max_context_len}")
                tx = self._miner_client.register_model(
                    model_id=model_id,
                    endpoint=endpoint,
                    model_spec_ref=spec_ref,
                    quant=quant,
                    max_context_len=max_context_len,
                    private_key=self.evm_pk,
                )
                bt.logging.info(f"Registered: {tx}")

                # Step 4: Get the index from the tx receipt events.
                # The contract emits ModelReactivated(miner, index, modelId)
                # when reactivating an existing slot, or ModelRegistered(
                # miner, modelId, endpoint) when pushing a new entry.
                # Parsing the event is always correct — no stale-state or
                # side-effect issues that plagued check_existing_registration.
                event_index = self._parse_model_index_from_receipt(tx)
                if event_index is not None:
                    bt.logging.info(f"Model index from tx receipt: {event_index}")
                    return event_index
                bt.logging.warning("Could not parse model index from tx receipt — resolving slot from chain state")
                for _ in range(10):
                    resolved_index = self._find_registered_model_index(
                        model_id, endpoint, quant, max_context_len,
                    )
                    if resolved_index is not None:
                        bt.logging.info(f"Model index from chain state: {resolved_index}")
                        return resolved_index
                    time.sleep(2)
                raise RuntimeError(
                    "Could not resolve registered model index after registration tx; "
                    "refusing to start capacity-audit worker with an inferred index"
                )
            except Exception as e:
                err_str = str(e)
                # "Duplicate active entry" means the model was already registered
                # (e.g. previous TX succeeded but receipt confirmation was 429'd).
                # Don't retry — loop back to check_existing_registration which
                # will find it and return the index.
                if "Duplicate active entry" in err_str:
                    bt.logging.info("Model already registered on-chain (duplicate detected), verifying...")
                    continue
                if attempt == max_retries:
                    raise
                delay = min(2 ** attempt * 5, 60)
                bt.logging.warning(f"Registration attempt {attempt}/{max_retries} failed ({e}), retrying in {delay}s")
                time.sleep(delay)

    # ------------------------------------------------------------------
    # Validator allowlist refresh (metagraph → JSON file for middleware)
    # ------------------------------------------------------------------

    def _refresh_validator_allowlist(self) -> None:
        """Query the metagraph for validators and write their SS58 hotkeys to disk.

        The JSON file is read by ValidatorAuthMiddleware in the server subprocess.
        Validators must have a permit AND meet the on-chain minValidatorStake
        threshold (read from ValidatorRegistry every refresh cycle).
        """
        from verallm.api.validator_auth import resolve_validators_path
        try:
            sub = self._get_subtensor()
            metagraph = sub.metagraph(netuid=self.config.netuid)

            # Read minValidatorStake from ValidatorRegistry (0 if unavailable)
            min_validator_stake = 0
            if getattr(self.config, 'validator_registry_address', None):
                try:
                    from verallm.chain.validator_registry import ValidatorRegistryClient
                    vr = ValidatorRegistryClient(self.config)
                    min_validator_stake_raw = vr.get_min_validator_stake()
                    # Convert RAO to TAO for comparison with metagraph.S
                    min_validator_stake = min_validator_stake_raw / 1e9
                    bt.logging.debug(f"minValidatorStake from contract: {min_validator_stake:.2f} alpha")
                except Exception as e:
                    bt.logging.debug(f"Could not fetch minValidatorStake: {e}")

            validators = []
            n = metagraph.n.item()
            for uid in range(n):
                has_permit = (
                    hasattr(metagraph, 'validator_permit')
                    and bool(metagraph.validator_permit[uid])
                )
                stake = float(metagraph.S[uid]) if hasattr(metagraph, 'S') else float(metagraph.stake[uid])

                if has_permit and stake >= min_validator_stake:
                    ss58 = metagraph.hotkeys[uid]
                    validators.append({
                        "uid": uid,
                        "hotkey_ss58": ss58,
                        "stake": stake,
                    })

            validator_hotkeys = {
                str(row["hotkey_ss58"])
                for row in validators
                if row.get("hotkey_ss58")
            }
            hard_auditor = _proof_v3_hard_auditor_record(
                None,
                validator_hotkeys,
            )
            try:
                block_value = getattr(metagraph, "block", 0)
                if hasattr(block_value, "item"):
                    block_value = block_value.item()
                block_number = int(block_value or 0)
                epoch_blocks = max(1, int(self.config.epoch_blocks))
                current_epoch = (
                    block_number // epoch_blocks if block_number > 0 else None
                )
                runtime = self._subnet_runtime_config_client.get(
                    current_epoch=current_epoch,
                    force=False,
                )
                hard_auditor = _proof_v3_hard_auditor_record(
                    runtime,
                    validator_hotkeys,
                )
                if runtime is not None:
                    apply_runtime_config_to_neuron_config(runtime, self.config)
                    self._reconcile_runtime_nginx_timeout()
                    policy = runtime.proof_v3_hard_auditor
                    if policy.enabled and not hard_auditor["enabled"]:
                        bt.logging.warning(
                            "Configured proof-v3 hard auditor is not in the "
                            "current validator allowlist; hard reveals remain disabled"
                        )
            except Exception as exc:
                bt.logging.warning(
                    "Could not refresh proof-v3 hard-auditor policy; "
                    f"hard reveals remain disabled: {exc}"
                )

            # Inject manually allowed validators (--allow-validators)
            allow_extra = getattr(self.config, "allow_validators", None) or []
            existing_ss58 = {v["hotkey_ss58"] for v in validators}
            for ss58 in allow_extra:
                if ss58 not in existing_ss58:
                    validators.append({"uid": -1, "hotkey_ss58": ss58, "stake": 0})
                    bt.logging.info(f"Manually allowed validator: {ss58}")

            out_path_obj = resolve_validators_path()
            out_path_obj.parent.mkdir(parents=True, exist_ok=True)
            out_path = str(out_path_obj)
            data = {
                "updated_at": int(time.time()),
                "netuid": self.config.netuid,
                "validators": validators,
                "proof_v3_hard_auditor": hard_auditor,
                "allowed_proof_protocol_versions": list(
                    _configured_miner_proof_protocol_versions(
                        self.config.proof_protocol_allowed_versions,
                        proof_v3_configured=self._proof_v3_configured,
                    )
                ),
            }
            self._served_proof_protocol_versions = tuple(
                data["allowed_proof_protocol_versions"]
            )
            # Atomic write: write to temp file then rename
            tmp_path = out_path + ".tmp"
            with open(tmp_path, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp_path, out_path)

            bt.logging.info(f"Validator allowlist updated: {len(validators)} validators written to {out_path}")

        except Exception as e:
            bt.logging.warning(f"Failed to refresh validator allowlist: {e}")
            raise

    def _reconcile_runtime_nginx_timeout(self) -> bool:
        """Apply a changed subnet timing ceiling without restarting vLLM."""

        target = _managed_nginx_read_timeout_seconds(self.config)
        if self._managed_nginx_timeout_seconds == target:
            return True
        if not _reconcile_managed_nginx_read_timeout(
            read_timeout_seconds=target,
        ):
            bt.logging.warning(
                "Managed nginx timeout reconciliation did not complete; "
                f"verify the upstream read timeout is at least {target}s"
            )
            return False
        self._managed_nginx_timeout_seconds = target
        return True

    _cached_metagraph_line: str = ""

    _subtensor_cache = None

    def _get_subtensor(self):
        if self._subtensor_cache is None:
            self._subtensor_cache = bt.Subtensor(network=self.config.subtensor_network)
        return self._subtensor_cache

    def _refresh_metagraph_stats(self) -> None:
        """Fetch metagraph from RPC and cache the stats line. Called every ~5 min."""
        try:
            sub = self._get_subtensor()
            block = sub.get_current_block()
            mg = sub.metagraph(netuid=self.config.netuid)
            uid = self.uid
            n = mg.n.item()
            if uid is not None and uid < n:
                _get = lambda attr: float(getattr(mg, attr)[uid]) if hasattr(mg, attr) else 0.0
                self._cached_metagraph_parts = (
                    f"UID {uid}",
                    f"incentive={_get('incentive'):.4f}",
                    f"emission={_get('emission'):.2f}α/tempo",
                    f"trust={_get('trust'):.2f}",
                    f"stake={_get('stake'):.2f}α",
                )
                self._cached_metagraph_line = f"Metagraph | block={block} | {' | '.join(self._cached_metagraph_parts)}"
        except Exception as e:
            bt.logging.debug(f"Metagraph refresh failed: {e}")

    def _validator_refresh_loop(self, interval: float = 300.0) -> None:
        """Background thread: refresh validator allowlist every `interval` seconds."""
        backoff = interval
        while self._running:
            success = False
            for attempt in range(1, 4):  # 3 retries
                try:
                    self._refresh_validator_allowlist()
                    backoff = interval  # Reset on success
                    success = True
                    break
                except Exception:
                    if attempt < 3:
                        wait = 10 * attempt  # 10s, 20s
                        bt.logging.debug(f"Allowlist refresh retry {attempt}/3 in {wait}s")
                        time.sleep(wait)
            if not success:
                backoff = min(backoff + 60, 600)  # grow by 1 min, max 10 min
                bt.logging.warning(f"Validator allowlist refresh failed after 3 retries, next attempt in {backoff:.0f}s")

            # Refresh metagraph stats from RPC (independent of allowlist success)
            self._refresh_metagraph_stats()
            if self._cached_metagraph_line:
                bt.logging.info(self._cached_metagraph_line)

            # Sleep in small increments; re-log cached stats every ~60s with fresh block
            deadline = time.monotonic() + backoff
            _last_stats_log = time.monotonic()
            while self._running and time.monotonic() < deadline:
                time.sleep(5.0)
                if hasattr(self, '_cached_metagraph_parts') and time.monotonic() - _last_stats_log >= 60:
                    # Refresh block number cheaply (single RPC, no metagraph sync)
                    try:
                        _block = self._get_subtensor().get_current_block()
                        self._cached_metagraph_line = f"Metagraph | block={_block} | {' | '.join(self._cached_metagraph_parts)}"
                    except Exception:
                        pass  # use stale line
                    if self._cached_metagraph_line:
                        bt.logging.info(self._cached_metagraph_line)
                    _last_stats_log = time.monotonic()

    def start_validator_refresh(self, interval: float = 300.0) -> None:
        """Start the background validator allowlist refresh thread."""
        t = threading.Thread(
            target=self._validator_refresh_loop,
            args=(interval,),
            daemon=True,
            name="validator-refresh",
        )
        t.start()
        bt.logging.info(f"Validator allowlist refresh started (interval={interval}s)")

    def _get_lease_remaining_sec(self, model_index: int) -> Optional[float]:
        """Read remaining lease time from chain (one read RPC, no gas)."""
        try:
            models = self._miner_client.get_miner_models(self.evm_addr)
            if model_index < len(models):
                remaining = models[model_index].expires_at - int(time.time())
                return max(remaining, 0.0)
        except Exception as e:
            bt.logging.warning(f"Could not read lease expiry: {e}")
        return None

    def heartbeat_loop(self, model_index: int = 0):
        """Periodically renew the model lease.

        On startup, checks remaining lease time via a read-only RPC call
        (no gas).  Only renews immediately if the lease won't survive
        until the next scheduled heartbeat.  On failure, retries with
        exponential backoff (30s → 60s → 120s … capped at 10 min) so
        transient RPC rate limits don't cause the 24h lease to expire.
        """
        interval = self.config.heartbeat_interval_sec

        # Check remaining lease — only renew now if it won't last until
        # the next heartbeat, otherwise just wait.
        remaining = self._get_lease_remaining_sec(model_index)
        if remaining is not None and remaining > interval + 3600:
            # Lease has plenty of headroom — sleep until renewal is due.
            # Schedule first renewal at remaining - interval (so the lease
            # gets extended well before it expires).
            sleep_time = remaining - interval
            bt.logging.info(f"Starting heartbeat loop (interval={interval}s, lease has {remaining / 3600:.1f}h remaining — first renewal in {sleep_time / 3600:.1f}h)")
        else:
            # Lease is short or unknown — renew immediately.
            sleep_time = 0
            _lease_info = f"has {remaining / 3600:.1f}h remaining" if remaining is not None else "unknown"
            bt.logging.info(f"Starting heartbeat loop (interval={interval}s, lease {_lease_info} — renewing immediately)")

        last_success = time.monotonic()
        consecutive_failures = 0

        while self._running:
            if sleep_time > 0:
                time.sleep(sleep_time)
            if not self._running:
                break
            try:
                tx = self._miner_client.renew_model(model_index, private_key=self.evm_pk)
                bt.logging.info(f"Renewed model at index {model_index}: {tx}")
                last_success = time.monotonic()
                consecutive_failures = 0
                sleep_time = interval
            except Exception as e:
                err_str = str(e)
                # "Already expired, re-register" means the entry we're
                # renewing has expired — either because we were renewing
                # the wrong index (register_on_chain bug) or because RPC
                # failures exceeded the 24h lease.  Re-register to get a
                # fresh entry and update model_index.
                if "Already expired" in err_str or "re-register" in err_str.lower():
                    bt.logging.warning(f"Lease expired at index {model_index} — re-registering")
                    try:
                        model_index = self.register_on_chain(
                            model_id=self._model_id,
                            endpoint=self._endpoint,
                            quant=self._quant,
                            max_context_len=self._max_context_len,
                        )
                        bt.logging.info(f"Re-registered at index {model_index} — heartbeat continues")
                        last_success = time.monotonic()
                        consecutive_failures = 0
                        sleep_time = interval
                        continue
                    except Exception as re_err:
                        bt.logging.error(f"Re-registration failed: {re_err}")
                        # Fall through to normal backoff

                consecutive_failures += 1
                hours_since_success = (time.monotonic() - last_success) / 3600
                # Exponential backoff: 30s, 60s, 120s, ... capped at 10 min
                sleep_time = min(30 * (2 ** (consecutive_failures - 1)), 600)
                bt.logging.warning(f"Renew failed (attempt {consecutive_failures}, {hours_since_success:.1f}h since last success, retry in {sleep_time}s): {e}")
                if hours_since_success > 20:
                    bt.logging.critical(f"MODEL LEASE EXPIRES IN <4 HOURS! Renewal has failed for {hours_since_success:.1f}h. Check EVM wallet balance and RPC connectivity.")

    def shutdown(self):
        """Clean shutdown."""
        self._running = False
        artifact_watcher = getattr(self, "_proof_v3_artifact_watcher", None)
        if artifact_watcher is not None:
            artifact_watcher.stop()
        if self._capacity_audit_worker is not None:
            self._capacity_audit_worker.stop()
        proc = self._server_process
        if proc:
            if proc.poll() is None:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                except Exception:
                    proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                bt.logging.warning("Server process did not stop after SIGTERM; killing")
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except Exception:
                    proc.kill()
                proc.wait(timeout=5)
            bt.logging.info("Server process terminated")


def parse_args():
    parser = argparse.ArgumentParser(description="Verathos Miner Neuron")
    parser.add_argument("--netuid", type=int, required=True)
    parser.add_argument("--chain-config", default=None,
                        help="Path to chain config JSON. If omitted, derived from --subtensor-network.")
    parser.add_argument("--subtensor-network", default=None,
                        help="Bittensor network name (test or finney). Selects chain config and default RPC URL.")
    parser.add_argument("--subtensor-chain-endpoint", default=None,
                        help="Explicit Substrate+EVM endpoint. Public RPCs are not recommended for stable mining.")
    parser.add_argument("--endpoint", required=True, help="Public miner endpoint URL")
    parser.add_argument("--skip-external-port-check", action="store_true",
                        help="Skip the public endpoint reachability preflight. "
                             "Use only for controlled local/tunneled tests.")
    parser.add_argument("--no-autorestart", action="store_true",
                        help="Don't auto-restart the server subprocess on crash")
    parser.add_argument("--auto-update", action="store_true",
                        help="Enable automatic code updates from git remote. "
                             "Checks every 30 min, pulls and restarts on new commits.")
    parser.add_argument("--auto-update-interval", type=int, default=1800,
                        help="Auto-update check interval in seconds (default: 1800 = 30 min)")
    parser.add_argument("--auto-update-jitter", type=int, default=1800,
                        help="Deterministic stagger window (seconds) before restart on auto-update. "
                             "Derives a per-hotkey offset in [0, jitter) so simultaneous version bumps "
                             "don't take every miner offline at the same time. Default 1800 = half an "
                             "epoch. Set 0 to disable (single-miner / testnet).")
    parser.add_argument("--analytics", action="store_true",
                        help="Enable analytics reporting (request timing, proof stats)")
    parser.add_argument("--capacity-audit", action="store_true",
                        help="Enable miner-side hot-capacity audit worker.")
    parser.add_argument("--capacity-audit-validator-urls", default=None,
                        help="Emergency override for capacity artifact targets. "
                             "By default miners discover validator audit endpoints from chain state.")
    parser.add_argument("--capacity-audit-windows-per-epoch", type=int, default=None,
                        help="Number of deterministic capacity-audit windows per subnet epoch.")
    parser.add_argument("--capacity-audit-max-drain-fraction", type=float, default=None,
                        help="Maximum active endpoint fraction drained in one audit window.")
    parser.add_argument("--capacity-audit-group-stress-fraction", type=float, default=None,
                        help="Share of each audit-window budget reserved for related-slot group stress.")
    parser.add_argument("--capacity-audit-beacon-hash-count", type=int, default=None,
                        help="Number of prior chain-head hashes mixed into the audit selection beacon.")
    parser.add_argument("--capacity-audit-min-registration-age-s", type=float, default=None,
                        help="Minimum endpoint lease age before it can enter capacity-audit cohorts.")
    parser.add_argument("--capacity-audit-lead-blocks", type=int, default=None,
                        help="Blocks between audit selection and audit start.")
    parser.add_argument("--capacity-audit-proof-challenge-delay-blocks", type=int, default=None,
                        help="Blocks between audit start and deferred proof challenge.")
    parser.add_argument("--capacity-audit-drain-seconds", type=float, default=None,
                        help="Nominal endpoint drain period before the timing deadline.")
    parser.add_argument("--capacity-audit-deadline-s", type=float, default=None,
                        help="Timing deadline, in seconds, measured from observed B_start.")
    parser.add_argument("--capacity-audit-transport-grace-s", type=float, default=None,
                        help="Additional final-receipt transport grace after the timing deadline.")
    parser.add_argument("--capacity-audit-payload-deadline-s", type=float, default=None,
                        help="Deferred proof payload timeout after the final timing receipt.")
    parser.add_argument("--capacity-audit-worker-poll-s", type=float, default=None,
                        help="Miner-side capacity-audit chain polling interval in seconds.")

    proof_v2_group = parser.add_argument_group("proof v2")
    proof_v2_group.add_argument(
        "--proof-v2-manifest",
        default=os.environ.get(PROOF_V2_MANIFEST_ENV),
        help=(
            "Path to the chain-authenticated proof-v2 manifest document "
            f"(env: {PROOF_V2_MANIFEST_ENV})."
        ),
    )
    proof_v2_group.add_argument(
        "--proof-v2-weight-catalog",
        default=os.environ.get(PROOF_V2_WEIGHT_CATALOG_ENV),
        help=(
            "Path to the static proof-v2 weight commitment catalog "
            f"(env: {PROOF_V2_WEIGHT_CATALOG_ENV})."
        ),
    )
    proof_v2_group.add_argument(
        "--proof-v2-artifact-base-url",
        action="append",
        default=None,
        help=(
            "HTTPS base URL for a content-addressed proof-v2 artifact store. "
            "Repeat to configure fallback mirrors "
            f"(env: {PROOF_V2_ARTIFACT_BASE_URLS_ENV}, comma-separated)."
        ),
    )
    proof_v2_group.add_argument(
        "--proof-v2-artifact-cache-dir",
        default=os.environ.get(PROOF_V2_ARTIFACT_CACHE_DIR_ENV),
        help=(
            "Local cache directory for downloaded proof-v2 artifacts "
            f"(env: {PROOF_V2_ARTIFACT_CACHE_DIR_ENV})."
        ),
    )

    proof_v3_group = parser.add_argument_group("proof v3")
    proof_v3_group.add_argument(
        "--proof-v3-release",
        default=os.environ.get(PROOF_V3_RELEASE_ENV),
        help=(
            "Complete local proof-v3 release descriptor "
            f"(env: {PROOF_V3_RELEASE_ENV})."
        ),
    )
    proof_v3_group.add_argument(
        "--proof-v3-artifact-base-url",
        action="append",
        default=None,
        help=(
            "HTTPS base URL for content-addressed proof-v3 releases. "
            "Repeat for mirrors; defaults to "
            "VERATHOS_PROOF_V3_ARTIFACT_BASE_URLS."
        ),
    )
    proof_v3_group.add_argument(
        "--proof-v3-artifact-cache-dir",
        default=None,
        help=(
            "Local content-addressed proof-v3 release cache; defaults to "
            "VERATHOS_PROOF_V3_ARTIFACT_CACHE_DIR or VERALLM_DATA_DIR."
        ),
    )
    proof_v3_group.add_argument(
        "--proof-v3-manifest",
        default=os.environ.get(PROOF_V3_MANIFEST_ENV),
        help=f"Authority-signed projection manifest (env: {PROOF_V3_MANIFEST_ENV}).",
    )
    proof_v3_group.add_argument(
        "--proof-v3-execution-profile",
        default=os.environ.get(PROOF_V3_EXECUTION_PROFILE_ENV),
        help=(
            "Authority-signed execution profile "
            f"(env: {PROOF_V3_EXECUTION_PROFILE_ENV})."
        ),
    )
    proof_v3_group.add_argument(
        "--proof-v3-calibration-set",
        default=os.environ.get(PROOF_V3_CALIBRATION_SET_ENV),
        help=f"Manifest-bound calibration set (env: {PROOF_V3_CALIBRATION_SET_ENV}).",
    )
    proof_v3_group.add_argument(
        "--proof-v3-attention-semantics",
        default=os.environ.get(PROOF_V3_ATTENTION_SEMANTICS_ENV),
        help=(
            "Manifest-bound attention runtime semantics "
            f"(env: {PROOF_V3_ATTENTION_SEMANTICS_ENV})."
        ),
    )
    proof_v3_group.add_argument(
        "--proof-v3-gdn-semantics",
        default=os.environ.get(PROOF_V3_GDN_SEMANTICS_ENV),
        help=(
            "Manifest-bound GDN runtime semantics when required "
            f"(env: {PROOF_V3_GDN_SEMANTICS_ENV})."
        ),
    )
    proof_v3_group.add_argument(
        "--proof-v3-lm-head-catalog",
        default=os.environ.get(PROOF_V3_LM_HEAD_CATALOG_ENV),
        help=(
            "Manifest-bound LM-head commitment catalog "
            f"(env: {PROOF_V3_LM_HEAD_CATALOG_ENV})."
        ),
    )
    proof_v3_group.add_argument(
        "--proof-v3-projection-manifest",
        default=os.environ.get(PROOF_V3_PROJECTION_MANIFEST_ENV),
        help=(
            "Authority-signed projection catalog manifest "
            f"(env: {PROOF_V3_PROJECTION_MANIFEST_ENV})."
        ),
    )
    proof_v3_group.add_argument(
        "--proof-v3-projection-catalog",
        default=os.environ.get(PROOF_V3_PROJECTION_CATALOG_ENV),
        help=(
            "Manifest-bound projection catalog "
            f"(env: {PROOF_V3_PROJECTION_CATALOG_ENV})."
        ),
    )
    proof_v3_group.add_argument(
        "--proof-v3-runtime-encoding",
        default=os.environ.get(PROOF_V3_RUNTIME_ENCODING_ENV),
        help=(
            "Qualified activation encoding ID "
            f"(env: {PROOF_V3_RUNTIME_ENCODING_ENV})."
        ),
    )
    proof_v3_group.add_argument(
        "--proof-v3-weight-cache-dir",
        default=os.environ.get(PROOF_V3_WEIGHT_CACHE_DIR_ENV),
        help=(
            "Persistent authenticated static-weight cache "
            f"(env: {PROOF_V3_WEIGHT_CACHE_DIR_ENV})."
        ),
    )

    # TEE (Trusted Execution Environment)
    tee_group = parser.add_argument_group("tee")
    parser.add_argument("--allow-validators", nargs="+", default=None,
                        metavar="SS58",
                        help="Extra validator SS58 hotkeys to allow (bypass permit check)")
    parser.add_argument("--update-endpoint", action="store_true",
                        help="Update existing on-chain endpoint if it changed (e.g. IP/port change). "
                             "Without this flag, a new entry is created for each unique endpoint.")

    tee_group.add_argument("--tee-enabled", action="store_true",
                           help="Enable TEE mode — register attestation on-chain after model registration")
    tee_group.add_argument("--tee-platform", default="mock",
                           choices=["mock", "tdx", "sev-snp", "gpu"],
                           help="TEE attestation platform (default: mock for dev)")

    # Authentication: wallet (testnet) or private-key (Anvil)
    auth = parser.add_argument_group("authentication")
    auth.add_argument("--wallet", default=None,
                      help="Bittensor wallet name (testnet — derives EVM key)")
    auth.add_argument("--hotkey", default="default",
                      help="Bittensor hotkey name (used with --wallet)")
    auth.add_argument("--private-key", default=None,
                      help="EVM private key (Anvil — skips bittensor wallet)")

    # Mesh runtime (GGUF mesh coordinator instead of the vLLM server)
    mesh_group = parser.add_argument_group("mesh")
    mesh_group.add_argument("--mesh-dir", default=None,
                            help="Path to a mesh directory created by 'verathos mesh create'. "
                                 "Runs the GGUF-mesh coordinator instead of the vLLM server; "
                                 "requires explicit --model-id. Arguments after -- are "
                                 "forwarded to 'mesh serve'.")

    # Model selection (auto or explicit, with cascading fallback)
    add_model_args(parser)

    # Bittensor logging flags (--logging.debug, --logging.trace, --logging.info)
    bt.logging.add_args(parser)

    # Everything after -- is passed to the miner server
    args, server_args = parser.parse_known_args()
    return args, _normalize_server_args(server_args)


def _neuron_config_from_args(args) -> NeuronConfig:
    """Construct network-scoped miner config from parsed CLI arguments."""

    overrides = {
        "wallet_name": args.wallet or "default",
        "hotkey_name": args.hotkey,
        "netuid": args.netuid,
    }
    if args.subtensor_network:
        # Bind network-scoped runtime URLs at construction time. Mutating
        # ``subtensor_network`` afterward leaves the default subnet-config and
        # owner-verdict URLs on their previous network.
        overrides["subtensor_network"] = args.subtensor_network
    return NeuronConfig.from_env(**overrides)


def _configure_proof_safe_fp8_backend(quant: str) -> tuple[str, ...]:
    """Install the canonical vLLM FP8 backend defaults before server spawn.

    The setup installer persists these values in ``.env.sh``, but a miner
    started directly (or a newly added PM2 sibling) must not depend on an
    interactive shell having sourced that file.  Explicit incompatible
    overrides fail before model loading instead of selecting an unqualified
    backend later in vLLM startup.
    """

    normalized = str(quant or "").strip().lower().replace("-", "_")
    if not normalized.startswith("fp8"):
        return ()

    defaulted: list[str] = []
    for name in _PROOF_SAFE_FP8_BACKEND_ENV:
        value = os.environ.get(name)
        if value is None or not value.strip():
            os.environ[name] = "0"
            defaulted.append(name)
            continue
        if value.strip() != "0":
            raise RuntimeError(
                f"FP8 proof serving requires {name}=0, got {value!r}. "
                "Remove the override or rerun scripts/setup_miner.sh before "
                "starting the miner; refusing an unqualified vLLM backend "
                "before model load."
            )
    return tuple(defaulted)


def _extract_code_measurement(platform: str, attestation_report: bytes, Web3) -> bytes:
    """Extract and hash the code measurement from an attestation report.

    Returns keccak256(raw_measurement) for on-chain storage as bytes32.
    - TDX: keccak256(mr_td) — 48-byte measurement at report body offset 0x0B8
    - SEV-SNP: keccak256(measurement) — 48-byte field at offset 0x090
    - mock/gpu: keccak256(b"mock") — deterministic placeholder
    """
    if platform == "tdx":
        try:
            # TDX DCAP quote: header(48) + body — mr_td is at body offset 0x0B8 (184), 48 bytes
            body_offset = 48  # after quote header
            mr_td_offset = body_offset + 0x0B8
            mr_td = attestation_report[mr_td_offset : mr_td_offset + 48]
            if len(mr_td) == 48:
                return bytes(Web3.keccak(mr_td))
        except Exception:
            pass
    elif platform == "sev-snp":
        try:
            # SEV-SNP report: measurement at offset 0x090, 48 bytes
            measurement = attestation_report[0x090 : 0x090 + 48]
            if len(measurement) == 48:
                return bytes(Web3.keccak(measurement))
        except Exception:
            pass
    # mock, gpu, or extraction failed — use deterministic placeholder
    return bytes(Web3.keccak(b"mock"))


def _register_tee(neuron, args, model_id: str):
    """Register TEE attestation on-chain after model is live.

    1. Fetch enclave pubkey + attestation from the running server's /tee/info
    2. Compute weight_file_hash (flat SHA256 of safetensors files)
    3. Call registerTEEAttestation() on MinerRegistry
    """
    import httpx

    bt.logging.info(f"TEE registration: fetching attestation from server (platform={args.tee_platform})")

    try:
        # Get TEE info from our own running server
        port = _extract_port(args.endpoint)
        resp = httpx.get(f"http://localhost:{port}/tee/info", timeout=10.0)
        resp.raise_for_status()
        tee_info = resp.json()
    except Exception as e:
        bt.logging.error(f"TEE registration failed — cannot reach /tee/info: {e}")
        return

    enclave_pubkey = bytes.fromhex(tee_info["enclave_public_key"])
    attestation = tee_info.get("attestation", {})
    attestation_report_hex = attestation.get("attestation_report", "")
    if not attestation_report_hex:
        bt.logging.error("TEE registration failed — /tee/info returned no attestation_report")
        return
    attestation_report = bytes.fromhex(attestation_report_hex)

    # keccak256 for on-chain storage
    from web3 import Web3
    attestation_hash = Web3.keccak(attestation_report)

    # Compute weight file hash (cheap single-pass SHA256 over safetensors files)
    try:
        from verallm.tee.weight_hash import compute_weight_file_hash
        weight_file_hash = compute_weight_file_hash(model_id)
    except Exception as e:
        bt.logging.warning(f"TEE: could not compute weight_file_hash ({e}), using zero hash")
        weight_file_hash = b"\x00" * 32

    # Extract code measurement from attestation report
    # TDX: keccak256(mr_td), SEV-SNP: keccak256(launch_digest), mock: keccak256("mock")
    code_measurement = _extract_code_measurement(args.tee_platform, attestation_report, Web3)

    bt.logging.info(
        f"TEE registration: platform={args.tee_platform} "
        f"pubkey={enclave_pubkey.hex()[:16]}... "
        f"weight_hash={weight_file_hash.hex()[:16]}... "
        f"code_measurement={code_measurement.hex()[:16]}..."
    )

    try:
        tx = neuron._miner_client.register_tee_attestation(
            platform=args.tee_platform,
            enclave_pub_key=enclave_pubkey,
            attestation_hash=attestation_hash,
            model_weight_hash=weight_file_hash,
            code_measurement=code_measurement,
            private_key=neuron.evm_pk,
        )
        bt.logging.success(f"TEE attestation registered on-chain: {tx}")
    except Exception as e:
        bt.logging.error(f"TEE registration tx failed: {e}")


def _extract_port(endpoint_url: str) -> int:
    """Extract port from endpoint URL, defaulting to 8000."""
    from urllib.parse import urlparse
    parsed = urlparse(endpoint_url)
    return parsed.port or 8000


def _extract_server_port(server_args: list[str]) -> int:
    """Extract --port from server subprocess args, defaulting to 8000."""
    for i, arg in enumerate(server_args):
        if arg == "--port" and i + 1 < len(server_args):
            try:
                return int(server_args[i + 1])
            except ValueError:
                break
    return 8000  # matches verallm.api.server default


def _compile_cache_dirs(
    *,
    environ=None,
    home=None,
    tmp_root=None,
) -> tuple:
    """Return the exact compile-cache directories owned by this miner."""

    import pathlib

    environ = os.environ if environ is None else environ
    home = pathlib.Path.home() if home is None else pathlib.Path(home)
    tmp_root = pathlib.Path("/tmp") if tmp_root is None else pathlib.Path(tmp_root)
    cache_dirs = [
        home / ".cache" / "vllm" / "torch_compile_cache",
        home / ".triton" / "cache",
    ]
    for name in ("TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR"):
        value = str(environ.get(name, "")).strip()
        if value:
            cache_dirs.append(pathlib.Path(value).expanduser())
    cache_dirs.extend(tmp_root.glob("torchinductor_*"))
    # Keep ordering stable for logs/tests while avoiding repeat deletion when
    # an environment variable points at one of the standard locations.
    return tuple(dict.fromkeys(cache_dirs))


def _compile_cache_runtime_fingerprint(*, repository_root=None) -> str:
    """Fingerprint code and packages that can change generated GPU kernels."""

    import hashlib
    import importlib.metadata

    root = (
        Path(__file__).resolve().parents[1]
        if repository_root is None
        else Path(repository_root)
    )
    runtime_files = [
        root / "neurons" / "miner.py",
        root / "neurons" / "version.py",
        root / "verallm" / "api" / "server.py",
    ]
    runtime_files.extend(
        sorted((root / "verallm" / "miner").glob("*.so"))
    )
    runtime_files.extend(
        sorted((root / "verallm" / "vllm_plugin").glob("*.py"))
    )
    digest = hashlib.sha256()
    digest.update(sys.version.encode("utf-8"))
    for package in ("torch", "triton", "vllm"):
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            version = "missing"
        digest.update(f"\0package:{package}={version}".encode("utf-8"))
    for path in runtime_files:
        if not path.is_file():
            continue
        digest.update(f"\0file:{path.relative_to(root)}\0".encode("utf-8"))
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _clear_stale_compile_caches(
    *,
    marker_path=None,
    lock_path=None,
    runtime_fingerprint=None,
) -> bool:
    """Clear stale compile caches once per installed runtime fingerprint.

    Same-host endpoint processes share the Torch/Triton caches.  A host lock
    prevents sibling miners from deleting a cache another process has just
    warmed, while the content fingerprint preserves mandatory invalidation
    whenever code or the Torch/Triton/vLLM stack changes.
    """
    import fcntl
    import shutil

    state_dir = Path.home() / ".cache" / "verathos"
    marker = (
        state_dir / "compile-cache-runtime.sha256"
        if marker_path is None
        else Path(marker_path)
    )
    lock = (
        state_dir / "compile-cache-runtime.lock"
        if lock_path is None
        else Path(lock_path)
    )
    fingerprint = (
        _compile_cache_runtime_fingerprint()
        if runtime_fingerprint is None
        else str(runtime_fingerprint)
    )
    marker.parent.mkdir(parents=True, exist_ok=True)
    lock.parent.mkdir(parents=True, exist_ok=True)

    cleared = 0
    failed = 0
    with lock.open("a+b") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            if marker.is_file() and marker.read_text().strip() == fingerprint:
                return False
        except OSError:
            pass
        for cache_dir in _compile_cache_dirs():
            if cache_dir.is_dir():
                try:
                    shutil.rmtree(cache_dir)
                    cleared += 1
                except Exception:
                    failed += 1
        if failed:
            bt.logging.warning(
                f"Could not clear {failed} stale torch/triton compile cache(s)"
            )
            return False
        marker.write_text(fingerprint + "\n")
    if cleared:
        bt.logging.info(
            f"Cleared {cleared} stale torch/triton compile caches for a new runtime"
        )
    return True


def _should_clear_stale_compile_caches(args) -> bool:
    """Only the vLLM runtime owns torch/triton compilation caches.

    A GGUF mesh coordinator neither consumes nor owns those directories.  In
    particular, it must not delete shared-host cache paths as a side effect of
    starting an isolated mesh process.
    """

    return not bool(getattr(args, "mesh_dir", None))


def main():
    from neurons.log import setup_neuron_logging, print_banner

    args, server_args = parse_args()
    setup_neuron_logging(args)
    if _should_clear_stale_compile_caches(args):
        _clear_stale_compile_caches()

    # The updater executing the first v1 -> v3 fast-forward is still the old
    # in-memory module and cannot install a newly introduced CUDA wheel.  The
    # restarted miner verifies the installed payload byte-for-byte against the
    # selected bundled wheel and installs it only when missing or stale.
    from neurons.auto_update import ensure_local_proof_v3_cuda_wheel

    if not ensure_local_proof_v3_cuda_wheel():
        bt.logging.error("Proof-v3 CUDA runtime installation failed")
        sys.exit(1)

    if not args.wallet and not args.private_key:
        bt.logging.error("Either --wallet or --private-key is required")
        sys.exit(1)

    # Resolve chain config FIRST so model selection can filter by on-chain models
    resolved_chain_path = ChainConfig.resolve_config_path(
        args.chain_config, args.subtensor_network,
    )
    if resolved_chain_path is None:
        bt.logging.error("Provide --chain-config or --subtensor-network (test/finney)")
        sys.exit(1)
    args.chain_config = resolved_chain_path  # update for downstream use

    config = _neuron_config_from_args(args)
    if getattr(args, "capacity_audit", False):
        config.capacity_audit_enabled = True

    # ── Mesh runtime mode ────────────────────────────────────────
    mesh_mode = bool(getattr(args, "mesh_dir", None))
    if mesh_mode:
        if getattr(args, "auto", False):
            bt.logging.error("--auto is not supported with --mesh-dir; pass --model-id explicitly")
            sys.exit(1)
        if not args.model_id:
            bt.logging.error("--mesh-dir requires an explicit --model-id (the on-chain mesh model id)")
            sys.exit(1)
        if args.tee_enabled:
            bt.logging.error("--tee-enabled is not supported with --mesh-dir")
            sys.exit(1)
        if getattr(config, "capacity_audit_enabled", False):
            bt.logging.error(
                "--capacity-audit is not supported with --mesh-dir "
                "(mesh endpoints are excluded from the vLLM capacity gate)"
            )
            sys.exit(1)

    # Resolve model configuration (auto or explicit)
    if mesh_mode:
        # Mesh runtime: the model is a GGUF package registered on-chain under
        # its own model id; vLLM registry auto-selection does not apply. The
        # quant string carries the runtime family (gguf_mesh_* prefix).
        from types import SimpleNamespace

        from neurons.runtime import normalize_mesh_quant

        if args.max_context_len is None:
            bt.logging.warning("--max-context-len not set for mesh runtime — registering 8192")
        resolved = SimpleNamespace(
            model_id=args.model_id,
            quant=normalize_mesh_quant(args.quant),
            max_context_len=int(args.max_context_len or 8192),
        )
    else:
        resolved = resolve_model_config(
            model_id=args.model_id,
            quant=args.quant,
            max_context_len=args.max_context_len,
            auto=args.auto,
            category=args.category,
            chain_config=resolved_chain_path,
            subtensor_network=args.subtensor_network,
            capacity_audit_required=bool(getattr(config, "capacity_audit_enabled", False)),
        )
    bt.logging.info(f"Model config: {resolved.model_id} quant={resolved.quant} ctx={resolved.max_context_len}")

    try:
        defaulted_fp8_env = _configure_proof_safe_fp8_backend(resolved.quant)
    except RuntimeError as exc:
        bt.logging.error(str(exc))
        sys.exit(1)
    if defaulted_fp8_env:
        bt.logging.info(
            "Applied proof-safe FP8 backend defaults before model load: "
            + ", ".join(f"{name}=0" for name in defaulted_fp8_env)
        )

    explicit_chain_endpoint = getattr(args, "subtensor_chain_endpoint", None)
    if ChainConfig.should_warn_public_rpc(explicit_chain_endpoint):
        bt.logging.warning(
            "Public Bittensor RPC configured. Public endpoints are rate-limited "
            "and may make miner registration, heartbeats, and capacity audits "
            "unstable. For production mining, run a Subtensor node and pass "
            "its endpoint with --subtensor-chain-endpoint."
        )

    rpc_override = ChainConfig.resolve_rpc_url(
        explicit_chain_endpoint,
        args.subtensor_network,
    )
    chain_config = ChainConfig.from_json(
        resolved_chain_path,
        **({"rpc_url": rpc_override} if rpc_override else {}),
    )
    for k in ChainConfig.__dataclass_fields__:
        if getattr(chain_config, k) != ChainConfig.__dataclass_fields__[k].default:
            setattr(config, k, getattr(chain_config, k))

    # Set Substrate network from CLI args
    if explicit_chain_endpoint:
        ep = explicit_chain_endpoint
        ws_ep = ep.replace("http://", "ws://").replace("https://", "wss://")
        config.subtensor_network = ws_ep
    elif args.subtensor_network:
        config.subtensor_network = args.subtensor_network

    # Transfer --allow-validators to config for the refresh loop
    if getattr(args, "allow_validators", None):
        config.allow_validators = args.allow_validators

    # Transfer --update-endpoint to config for registration logic
    if getattr(args, "update_endpoint", False):
        config.update_endpoint = True
    if getattr(args, "capacity_audit", False):
        config.capacity_audit_enabled = True
    if getattr(args, "capacity_audit_validator_urls", None):
        config.capacity_audit_validator_urls = args.capacity_audit_validator_urls
    if getattr(args, "capacity_audit_windows_per_epoch", None) is not None:
        config.capacity_audit_windows_per_epoch = args.capacity_audit_windows_per_epoch
    if getattr(args, "capacity_audit_max_drain_fraction", None) is not None:
        config.capacity_audit_max_drain_fraction = args.capacity_audit_max_drain_fraction
    if getattr(args, "capacity_audit_group_stress_fraction", None) is not None:
        config.capacity_audit_group_stress_fraction = args.capacity_audit_group_stress_fraction
    if getattr(args, "capacity_audit_beacon_hash_count", None) is not None:
        config.capacity_audit_beacon_hash_count = args.capacity_audit_beacon_hash_count
    if getattr(args, "capacity_audit_min_registration_age_s", None) is not None:
        config.capacity_audit_min_registration_age_s = args.capacity_audit_min_registration_age_s
    if getattr(args, "capacity_audit_lead_blocks", None) is not None:
        config.capacity_audit_lead_blocks = args.capacity_audit_lead_blocks
    if getattr(args, "capacity_audit_proof_challenge_delay_blocks", None) is not None:
        config.capacity_audit_proof_challenge_delay_blocks = args.capacity_audit_proof_challenge_delay_blocks
    if getattr(args, "capacity_audit_drain_seconds", None) is not None:
        config.capacity_audit_drain_seconds = args.capacity_audit_drain_seconds
    if getattr(args, "capacity_audit_deadline_s", None) is not None:
        config.capacity_audit_deadline_s = args.capacity_audit_deadline_s
    if getattr(args, "capacity_audit_transport_grace_s", None) is not None:
        config.capacity_audit_transport_grace_s = args.capacity_audit_transport_grace_s
    if getattr(args, "capacity_audit_payload_deadline_s", None) is not None:
        config.capacity_audit_payload_deadline_s = args.capacity_audit_payload_deadline_s
    if getattr(args, "capacity_audit_worker_poll_s", None) is not None:
        config.capacity_audit_worker_poll_s = args.capacity_audit_worker_poll_s

    # ── Early on-chain model check ───────────────────────────────
    # Verify the resolved model is registered on-chain BEFORE loading
    # it into GPU. This avoids wasting ~60s on model load + root
    # computation only to fail at _chain_self_check in the server.
    on_chain_models: list[str] | None = None
    model_client = None
    if chain_config.model_registry_address:
        try:
            from verallm.chain.model_registry import ModelRegistryClient
            model_client = ModelRegistryClient(chain_config)
            on_chain_models = model_client.get_model_list()
            on_chain_lower = {m.lower() for m in (on_chain_models or [])}
            if on_chain_models is not None and resolved.model_id.lower() not in on_chain_lower:
                bt.logging.error(
                    f"Model '{resolved.model_id}' is not registered on-chain. "
                    f"Miners can only serve models registered on the ModelRegistry contract. "
                    f"ModelRegistry: {chain_config.model_registry_address} | "
                    f"Registered models: {sorted(on_chain_models) if on_chain_models else '(none)'}"
                )
                sys.exit(1)
            elif on_chain_models is not None:
                bt.logging.info(f"On-chain model check passed: '{resolved.model_id}' is registered")
        except Exception as e:
            if getattr(config, "capacity_audit_enabled", False):
                bt.logging.error(
                    f"Capacity audit model gate requires ModelRegistry access; "
                    f"startup aborted after RPC error: {e}"
                )
                sys.exit(1)
            bt.logging.warning(
                f"On-chain model check skipped (RPC error: {e}). "
                f"The server will re-check after model load."
            )

    from verallm.proof_v3.artifact_store import (
        configured_proof_v3_artifact_base_urls,
    )

    proof_v3_sources = configured_proof_v3_artifact_base_urls(
        args.proof_v3_artifact_base_url,
        default_values=getattr(
            chain_config,
            "proof_v3_artifact_base_urls",
            (),
        ),
    )
    resolved_v3 = None
    proof_v3_cache_directory = (
        args.proof_v3_artifact_cache_dir
        or getattr(
            chain_config,
            "proof_v3_artifact_cache_dir",
            "",
        )
        or None
    )
    proof_v3_descriptor = str(args.proof_v3_release or "").strip()
    if proof_v3_descriptor and proof_v3_sources:
        bt.logging.info(
            "Using explicit proof-v3 release descriptor instead of remote "
            "artifact discovery"
        )
    if not proof_v3_descriptor and proof_v3_sources:
        if model_client is None:
            bt.logging.error(
                "Remote proof-v3 releases require ModelRegistry access"
            )
            sys.exit(1)
        try:
            from verallm.proof_v3.artifact_store import (
                resolve_remote_proof_v3_release,
            )

            resolved_v3 = resolve_remote_proof_v3_release(
                resolved.model_id,
                proof_v3_sources,
                chain_config=chain_config,
                model_registry_client=model_client,
                cache_directory=proof_v3_cache_directory,
            )
            proof_v3_descriptor = str(resolved_v3.descriptor_path)
            bt.logging.info(
                "Authenticated cached proof-v3 release from "
                f"{resolved_v3.index_source_url}"
            )
        except Exception as exc:
            bt.logging.error(f"Proof-v3 artifact resolution failed: {exc}")
            sys.exit(1)
    if proof_v3_descriptor:
        try:
            _apply_proof_v3_release_descriptor(args, proof_v3_descriptor)
        except Exception as exc:
            bt.logging.error(f"Proof-v3 release configuration failed: {exc}")
            sys.exit(1)

    neuron = MinerNeuron(config)
    neuron._proof_v3_configured = bool(args.proof_v3_manifest)

    def signal_handler(sig, _frame):
        bt.logging.info(f"Received signal {sig}, shutting down")
        neuron.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    neuron.setup(private_key=args.private_key)

    # ── Endpoint scheme posture ──
    # Fail before any model load: an http endpoint on mainnet registers a
    # miner the public proxy will never route.
    from verallm.chain.config import validate_registration_endpoint_scheme
    try:
        validate_registration_endpoint_scheme(args.endpoint, chain_config.chain_id)
    except ValueError as exc:
        bt.logging.error(str(exc))
        sys.exit(1)

    # ── Startup banner ──
    network = args.subtensor_network or ("testnet" if chain_config.chain_id == 945 else "mainnet")
    print_banner(
        "Miner",
        network=network,
        netuid=config.netuid,
        wallet=args.wallet or "(private-key)",
        hotkey=args.hotkey if args.wallet else "",
        evm=neuron.evm_addr or "",
        spec_ver=f"{version_str} ({spec_version})",
        miner_ver=f"{miner_version_str} ({miner_version})",
        model=resolved.model_id,
        quantization=resolved.quant,
        max_context=resolved.max_context_len,
        endpoint=args.endpoint,
        auto_update="enabled" if args.auto_update else "disabled",
    )

    if mesh_mode:
        # Mesh coordinator invocation: bind the public coordinator and pass
        # the serving hotkey/EVM identity plus validator authentication.
        neuron.mesh_dir = args.mesh_dir
        if "--host" not in server_args:
            server_args = ["--host", "0.0.0.0"] + server_args
        if "--port" not in server_args:
            server_args = server_args + ["--port", "9338"]
        server_args = _configure_mesh_security_args(
            args=args,
            server_args=server_args,
            evm_address=neuron.evm_addr,
            evm_private_key=neuron.evm_pk,
        )
    else:
        # Build server args: always ensure --model and --quant are present
        if "--model" not in server_args and "--model-id" not in server_args:
            server_args = [
                "--model", resolved.model_id,
                "--quant", resolved.quant,
            ] + server_args

        # Pass EVM identity to server for anti-hijacking (receipt validation + identity challenge)
        if neuron.evm_addr:
            server_args.extend(["--evm-address", neuron.evm_addr])
        if neuron.evm_pk:
            server_args.extend(["--evm-private-key", neuron.evm_pk])

        # Forward log level to server subprocess
        if getattr(args, "logging.trace", False):
            server_args.extend(["--log-level", "debug"])  # server has no trace, use debug
        elif getattr(args, "logging.debug", False):
            server_args.extend(["--log-level", "debug"])

        # Forward chain config and resolved RPC URL so server can self-check roots
        if args.chain_config and "--chain-config" not in server_args:
            server_args.extend(["--chain-config", args.chain_config])
        if chain_config.rpc_url and "--evm-rpc-url" not in server_args:
            server_args.extend(["--evm-rpc-url", chain_config.rpc_url])

    server_args = _forward_proof_v2_artifacts(
        server_args,
        manifest=args.proof_v2_manifest,
        weight_catalog=args.proof_v2_weight_catalog,
        artifact_base_urls=args.proof_v2_artifact_base_url,
        artifact_cache_dir=args.proof_v2_artifact_cache_dir,
    )
    server_args = _forward_proof_v3_artifacts(
        server_args,
        manifest=args.proof_v3_manifest,
        execution_profile=args.proof_v3_execution_profile,
        calibration_set=args.proof_v3_calibration_set,
        attention_semantics=args.proof_v3_attention_semantics,
        gdn_semantics=args.proof_v3_gdn_semantics,
        lm_head_catalog=args.proof_v3_lm_head_catalog,
        projection_manifest=args.proof_v3_projection_manifest,
        projection_catalog=args.proof_v3_projection_catalog,
        runtime_encoding=args.proof_v3_runtime_encoding,
        weight_cache_dir=args.proof_v3_weight_cache_dir,
    )
    if args.proof_v3_manifest:
        if not neuron.hotkey_ss58:
            bt.logging.error(
                "Proof-v3 requires a Bittensor wallet hotkey identity; "
                "direct EVM private-key mode is unsupported."
            )
            sys.exit(1)
        if "--miner-hotkey-ss58" not in server_args:
            server_args.extend(
                ["--miner-hotkey-ss58", neuron.hotkey_ss58]
            )

    # Forward TEE args to server subprocess
    if args.tee_enabled and "--tee-enabled" not in server_args:
        server_args.append("--tee-enabled")
        server_args.extend(["--tee-platform", args.tee_platform])

    capacity_audit_state_file = ""
    if getattr(config, "capacity_audit_enabled", False):
        capacity_audit_state_file = _capacity_audit_state_path(
            neuron.evm_addr,
            _extract_server_port(server_args),
        )
        server_args = _set_server_arg(
            server_args,
            "--capacity-audit-state-file",
            capacity_audit_state_file,
        )
        try:
            if os.path.exists(capacity_audit_state_file):
                os.remove(capacity_audit_state_file)
        except Exception as exc:
            bt.logging.warning(f"Could not clear stale capacity audit state: {exc}")

    # Write the validator allowlist before either server starts, avoiding an
    # open-access window. Private-key/Anvil mode has no metagraph to refresh.
    if _validator_allowlist_refresh_enabled(args):
        try:
            neuron._refresh_validator_allowlist()
        except Exception as e:
            bt.logging.warning(f"Initial validator allowlist write failed: {e} — server will block until next refresh succeeds")
    neuron._reconcile_runtime_nginx_timeout()
    served_proof_protocol_versions = (
        neuron._served_proof_protocol_versions
        or _configured_miner_proof_protocol_versions(
            config.proof_protocol_allowed_versions,
            proof_v3_configured=bool(args.proof_v3_manifest),
        )
    )
    if not served_proof_protocol_versions:
        bt.logging.error(
            "Miner has no configured proof protocol allowed by the subnet"
        )
        sys.exit(1)
    server_args = _set_server_arg(
        server_args,
        "--allowed-proof-protocol-versions",
        ",".join(
            str(version)
            for version in served_proof_protocol_versions
        ),
    )

    # ── External port reachability check ──
    # Verify the endpoint port is reachable from outside BEFORE loading vLLM
    # (which takes 10+ min). Starts a temporary TCP listener on the port,
    # asks external services to probe it, then shuts it down.
    if args.skip_external_port_check or os.environ.get("VERATHOS_SKIP_EXTERNAL_PORT_CHECK") == "1":
        bt.logging.warning(
            "Skipping external port reachability check. This is intended only "
            "for controlled local/tunneled tests."
        )
    else:
        _check_external_port(args.endpoint, local_bind_port=_extract_server_port(server_args))

    neuron.start_server(server_args)

    # Health-check on localhost. The --endpoint may be behind a reverse proxy
    # that isn't reachable from inside the container. Parse the server's actual
    # port from server_args (mirrors the server's own --port default of 8000).
    local_health_url = f"http://localhost:{_extract_server_port(server_args)}"
    # In mesh mode, pass server_args=None: the vLLM Mamba/AWQ exit-code retry
    # paths do not apply to the mesh coordinator subprocess.
    neuron.wait_for_health(local_health_url, server_args=None if mesh_mode else server_args)

    # Start background refresh loop (periodic updates)
    if _validator_allowlist_refresh_enabled(args):
        neuron.start_validator_refresh(interval=300.0)

    if mesh_mode:
        # No vLLM KV pool to query — register the declared context length.
        reg_context = resolved.max_context_len
    else:
        # Use actual KV pool from the running server instead of the registry estimate.
        # After vLLM loads: real capacity = min(kv_pool_tokens, max_model_len).
        actual_context = neuron.query_actual_max_context(local_health_url)
        if actual_context is not None:
            if actual_context != resolved.max_context_len:
                bt.logging.info(f"On-chain max_context: {actual_context} (actual from vLLM, was {resolved.max_context_len} from registry)")
            else:
                bt.logging.info(f"On-chain max_context: {actual_context} (matches registry)")
            reg_context = actual_context
        else:
            bt.logging.warning(f"Could not query actual context from server — using registry value {resolved.max_context_len}")
            reg_context = resolved.max_context_len

    if getattr(config, "capacity_audit_enabled", False):
        if on_chain_models is None:
            bt.logging.error(
                "Capacity audit model gate requires an on-chain ModelRegistry model list"
            )
            sys.exit(1)
        try:
            from verallm.registry.gpu import detect_gpu_info

            gpu_info = detect_gpu_info()
            ok, reason, expected = validate_capacity_recommended_model(
                model_id=resolved.model_id,
                quant=resolved.quant,
                max_context_len=int(reg_context or 0),
                vram_gb=capacity_gate_vram_gb(gpu_info),
                on_chain_models=on_chain_models,
            )
            if not ok:
                expected_text = ""
                if expected is not None:
                    expected_text = (
                        f" expected model={expected.model_id} "
                        f"quant={expected.quant}"
                    )
                bt.logging.error(f"{reason}.{expected_text}")
                sys.exit(1)
        except SystemExit:
            raise
        except Exception as exc:
            bt.logging.error(f"Capacity audit recommended-model check failed: {exc}")
            sys.exit(1)

    # Store registration params so heartbeat_loop can re-register on lease expiry
    neuron._model_id = resolved.model_id
    neuron._endpoint = args.endpoint
    neuron._quant = resolved.quant
    neuron._max_context_len = reg_context

    model_index = neuron.register_on_chain(
        model_id=resolved.model_id,
        endpoint=args.endpoint,
        quant=resolved.quant,
        max_context_len=reg_context,
    )

    # ── TEE registration / revocation ──
    if args.tee_enabled:
        _register_tee(neuron, args, resolved.model_id)
    else:
        # If TEE is not enabled but on-chain registration exists, revoke it.
        # This keeps on-chain state in sync with the miner's actual mode —
        # stale TEE registrations cause the proxy to route TEE traffic and
        # skip ZK proof verification for this miner.
        try:
            cap = neuron._miner_client.get_tee_capability(neuron.evm_addr)
            if cap.enabled:
                bt.logging.info("TEE not enabled but on-chain registration found — revoking...")
                neuron._miner_client.revoke_tee_attestation(private_key=neuron.evm_pk)
                bt.logging.success("TEE attestation revoked on-chain")
        except Exception as e:
            bt.logging.warning(f"Could not check/revoke TEE registration: {e}")

    if getattr(config, "capacity_audit_enabled", False):
        validator_urls = tuple(
            u.strip()
            for u in str(getattr(config, "capacity_audit_validator_urls", "") or "").split(",")
            if u.strip()
        )
        try:
            from neurons.capacity_audit_miner import CapacityAuditMinerWorker
            from verallm.chain.model_registry import ModelRegistryClient

            model_client = ModelRegistryClient(config)
            neuron._capacity_audit_worker = CapacityAuditMinerWorker(
                config=config,
                miner_client=neuron._miner_client,
                model_client=model_client,
                evm_address=neuron.evm_addr,
                evm_private_key=neuron.evm_pk,
                endpoint=args.endpoint,
                model_id=resolved.model_id,
                model_index=model_index,
                quant=resolved.quant,
                max_context_len=reg_context,
                validator_urls=validator_urls,
                local_health_url=local_health_url,
                audit_state_file=capacity_audit_state_file,
                poll_interval_s=_capacity_audit_worker_poll_interval(config),
            )
            neuron._capacity_audit_worker.start()
        except Exception as e:
            mode = str(getattr(config, "capacity_audit_mode", "observe") or "observe")
            if mode != "observe":
                raise RuntimeError(
                    f"Capacity audit miner worker failed to start in {mode} mode"
                ) from e
            bt.logging.warning(f"Capacity audit miner worker failed to start in observe mode: {e}")

    # ── Auto-updater ──
    if args.auto_update:
        from neurons.auto_update import AutoUpdater, derive_jitter_seed

        def _miner_busy() -> bool:
            return neuron._auto_update_busy(local_health_url)

        updater = AutoUpdater(
            role="miner",
            check_interval=args.auto_update_interval,
            busy_check=_miner_busy,
            jitter_seconds=args.auto_update_jitter,
            jitter_seed=derive_jitter_seed(
                neuron.hotkey_seed,
                _extract_server_port(server_args),
            ),
        )
        updater.start()

    if resolved_v3 is not None:
        neuron.start_proof_v3_artifact_refresh(
            model_id=resolved.model_id,
            base_urls=proof_v3_sources,
            chain_config=chain_config,
            model_registry_client=model_client,
            cache_directory=proof_v3_cache_directory,
            current_release_sha256=resolved_v3.release_sha256,
            local_health_url=local_health_url,
        )

    bt.logging.success(
        f"Miner ready — serving {resolved.model_id} ({resolved.quant}) "
        f"on {args.endpoint}",
    )

    neuron.heartbeat_loop(model_index=model_index)


if __name__ == "__main__":
    main()
