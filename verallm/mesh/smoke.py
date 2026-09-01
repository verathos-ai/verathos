"""Repeatable local smoke runner for verified GGUF mesh inference."""

from __future__ import annotations

import collections
import hashlib
import json
import os
import signal
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from verallm.mesh.gguf_manifest import (
    GGML_TRACE_MAX_ELEMS_FLOOR,
    load_gguf_tensor_manifest,
    suggest_ggml_trace_max_elems_from_gguf_model,
    suggest_ggml_trace_max_elems_from_manifest,
)
from verallm.mesh.state import create_mesh_state, join_mesh
from verallm.mesh.types import MeshSpec
from verallm.mesh.worker import probe_worker


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _post_json(
    url: str,
    payload: dict[str, Any],
    *,
    timeout: float = 300.0,
    internal_auth_secret: str | bytes = "",
) -> dict[str, Any]:
    body = json.dumps(payload, sort_keys=True).encode("utf-8")
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    if internal_auth_secret:
        from urllib.parse import urlparse

        from verallm.mesh.http_auth import sign_internal_http_request

        headers.update(
            sign_internal_http_request(
                secret=internal_auth_secret,
                method="POST",
                path=urlparse(url).path or "/",
                body=body,
            )
        )
    req = Request(
        url,
        data=body,
        headers=headers,
        method="POST",
    )
    try:
        with urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"connect failed: {exc.reason}") from exc
    data = json.loads(raw.decode("utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(f"{url} returned non-object JSON")
    return data


class ManagedProcess:
    """Small portable process-group wrapper used by the smoke runner."""

    def __init__(self, name: str, cmd: list[str], *, cwd: Path, env: dict[str, str]):
        self.name = name
        self.logs: collections.deque[str] = collections.deque(maxlen=200)
        kwargs: dict[str, Any] = {}
        if os.name == "nt":
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        else:
            kwargs["preexec_fn"] = os.setsid
        self.proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            **kwargs,
        )
        self.thread = threading.Thread(target=self._read_logs, daemon=True)
        self.thread.start()

    @property
    def pid(self) -> int:
        return int(self.proc.pid)

    def _read_logs(self) -> None:
        assert self.proc.stdout is not None
        _tee_path = os.environ.get("VERATHOS_MESH_SMOKE_LOG_DIR")
        _tee = None
        if _tee_path:
            try:
                _tee = open(  # noqa: SIM115
                    os.path.join(_tee_path, f"mesh_{self.name}.log"),
                    "a",
                    encoding="utf-8",
                )
            except Exception:
                _tee = None
        for line in self.proc.stdout:
            self.logs.append(line.rstrip("\n"))
            if _tee is not None:
                _tee.write(line)
                _tee.flush()

    def assert_running(self) -> None:
        code = self.proc.poll()
        if code is None:
            return
        tail = "\n".join(self.logs)
        raise RuntimeError(f"{self.name} exited with code {code}\n{tail}")

    def stop(self) -> None:
        if self.proc.poll() is not None:
            return
        if os.name == "nt":
            self.proc.terminate()
            try:
                self.proc.wait(timeout=8)
                return
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=5)
                return
        for sig, timeout_s in ((signal.SIGINT, 8), (signal.SIGTERM, 5), (signal.SIGKILL, 5)):
            try:
                os.killpg(self.proc.pid, sig)
            except ProcessLookupError:
                return
            try:
                self.proc.wait(timeout=timeout_s)
                return
            except subprocess.TimeoutExpired:
                continue


@dataclass(frozen=True)
class SmokeConfig:
    repo_root: Path
    output_root: Path
    model_id: str
    package_hash: str
    layers: int
    llama_server_binary: str
    rpc_worker_binary: str
    model_tensor_manifest_root: str = ""
    llama_model: str = ""
    llama_hf: str = ""
    llama_device: str = "RPC0"
    rpc_device: str = "CUDA0"
    prompt: str = "Explain verified mesh inference in one sentence."
    max_tokens: int = 64
    samples: int = 1
    direct_baseline: bool = False
    require_proof: bool = True
    proof_sample_bps: int = 10_000
    proof_ops_per_request: int = 1
    proof_trace_candidates_per_request: int = 8
    proof_tolerance_abs: float = 8e-2
    proof_tolerance_rel: float = 4e-2
    decode_audit_bps: int = 0
    decode_audit_top_k: int = 8
    proof_gguf_manifest_path: str = ""
    hf_home: str = ""
    timeout: float = 300.0


def _wait_tcp(host: str, port: int, procs: list[ManagedProcess], *, timeout: float) -> None:
    deadline = time.time() + timeout
    last: Exception | None = None
    while time.time() < deadline:
        for proc in procs:
            proc.assert_running()
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return
        except OSError as exc:
            last = exc
            time.sleep(0.2)
    raise RuntimeError(f"TCP endpoint {host}:{port} did not become ready: {last}")


def _wait_probe(
    endpoint: str,
    procs: list[ManagedProcess],
    *,
    timeout: float,
    internal_auth_secret: str | bytes = "",
) -> None:
    deadline = time.time() + timeout
    last: Exception | None = None
    while time.time() < deadline:
        for proc in procs:
            proc.assert_running()
        try:
            probe_worker(
                endpoint,
                timeout=1.0,
                internal_auth_secret=internal_auth_secret,
            )
            return
        except Exception as exc:
            last = exc
            time.sleep(0.25)
    raise RuntimeError(f"{endpoint} did not become probeable: {last}")


def _wait_chat(
    endpoint: str,
    request: dict[str, Any],
    procs: list[ManagedProcess],
    *,
    timeout: float,
    internal_auth_secret: str | bytes = "",
) -> None:
    deadline = time.time() + timeout
    last: Exception | None = None
    while time.time() < deadline:
        for proc in procs:
            proc.assert_running()
        try:
            _post_json(
                endpoint.rstrip("/") + "/v1/chat/completions",
                {**request, "max_tokens": min(8, int(request.get("max_tokens", 8)))},
                timeout=30.0,
                internal_auth_secret=internal_auth_secret,
            )
            return
        except Exception as exc:
            last = exc
            time.sleep(1.0)
    raise RuntimeError(f"{endpoint} did not become inference-ready: {last}")


def _request_for_sample(request: dict[str, Any], *, validator_nonce: bool) -> dict[str, Any]:
    if not validator_nonce:
        return request
    sampled = dict(request)
    sampled["verathos"] = {"validator_nonce": os.urandom(32).hex()}
    return sampled


def _measure(
    endpoint: str,
    request: dict[str, Any],
    procs: list[ManagedProcess],
    *,
    samples: int,
    timeout: float,
    validator_nonce: bool = False,
    internal_auth_secret: str | bytes = "",
) -> dict[str, Any]:
    _wait_chat(
        endpoint,
        _request_for_sample(request, validator_nonce=validator_nonce),
        procs,
        timeout=timeout,
        internal_auth_secret=internal_auth_secret,
    )
    measurements: list[dict[str, Any]] = []
    for _ in range(max(1, samples)):
        for proc in procs:
            proc.assert_running()
        sample_request = _request_for_sample(request, validator_nonce=validator_nonce)
        started = time.perf_counter()
        response = _post_json(
            endpoint.rstrip("/") + "/v1/chat/completions",
            sample_request,
            timeout=timeout,
            internal_auth_secret=internal_auth_secret,
        )
        elapsed = time.perf_counter() - started
        usage = response.get("usage", {})
        mesh = response.get("verathos_mesh", {})
        proof_payload_count = int(
            mesh.get("proof_payload_count")
            or len(mesh.get("proof_payloads", []) or [])
            or len(mesh.get("proof_payload_refs", []) or [])
            or 0
        )
        content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
        measurements.append(
            {
                "elapsed_s": elapsed,
                "completion_tokens": int(usage.get("completion_tokens") or 0),
                "total_tokens": int(usage.get("total_tokens") or 0),
                "content_chars": len(str(content)),
                "verified": mesh.get("verified"),
                "proof_receipt_verified": mesh.get("proof_receipt_verified"),
                "proof_receipt_count": mesh.get("proof_receipt_count"),
                "proof_payload_count": proof_payload_count,
                "proof_verifier_ms": float(mesh.get("receipt", {}).get("proof_verifier_ms") or 0.0),
                "proof_sampled": mesh.get("proof_sampled"),
                "proof_sample_value": mesh.get("proof_sample_value"),
                "proof_sample_bps": mesh.get("proof_sample_bps"),
                "proof_trace_candidates_per_request": mesh.get(
                    "proof_trace_candidates_per_request"
                ),
                "proof_trace_commitment_count": mesh.get("proof_trace_commitment_count"),
                "proof_op_manifest_count": mesh.get("proof_op_manifest_count"),
                "decode_audit_bps": mesh.get("decode_audit_bps"),
                "decode_audit_required": mesh.get("decode_audit_required"),
                "decode_audit_sampled": mesh.get("decode_audit_sampled"),
                "decode_audit_verified": mesh.get("decode_audit_verified"),
                "decode_audit_verifier_ms": float(
                    mesh.get("receipt", {}).get("decode_audit_verifier_ms") or 0.0
                ),
            }
        )
    latencies = [item["elapsed_s"] for item in measurements]
    completion_tokens = [item["completion_tokens"] for item in measurements if item["completion_tokens"]]
    mean_s = statistics.mean(latencies)
    return {
        "samples": measurements,
        "mean_s": mean_s,
        "median_s": statistics.median(latencies),
        "min_s": min(latencies),
        "max_s": max(latencies),
        "mean_completion_tokens": statistics.mean(completion_tokens) if completion_tokens else 0,
        "mean_tok_s": (statistics.mean(completion_tokens) / mean_s) if completion_tokens else 0,
        "proof_sampled_count": sum(1 for item in measurements if item.get("proof_sampled") is True),
        "proof_receipt_count_total": sum(int(item.get("proof_receipt_count") or 0) for item in measurements),
        "proof_payload_count_total": sum(int(item.get("proof_payload_count") or 0) for item in measurements),
        "proof_verifier_ms_total": sum(float(item.get("proof_verifier_ms") or 0.0) for item in measurements),
        "proof_verifier_ms_mean": statistics.mean(
            [float(item.get("proof_verifier_ms") or 0.0) for item in measurements]
        ),
        "proof_verifier_ms_mean_sampled": statistics.mean(
            [
                float(item.get("proof_verifier_ms") or 0.0)
                for item in measurements
                if item.get("proof_sampled") is True
            ]
        )
        if any(item.get("proof_sampled") is True for item in measurements)
        else 0.0,
        "proof_trace_commitment_count_total": sum(
            int(item.get("proof_trace_commitment_count") or 0) for item in measurements
        ),
        "proof_op_manifest_count_total": sum(
            int(item.get("proof_op_manifest_count") or 0) for item in measurements
        ),
        "decode_audit_sampled_count": sum(
            1 for item in measurements if item.get("decode_audit_sampled") is True
        ),
        "decode_audit_verifier_ms_total": sum(
            float(item.get("decode_audit_verifier_ms") or 0.0) for item in measurements
        ),
    }


def _runtime_env(config: SmokeConfig) -> dict[str, str]:
    env = os.environ.copy()
    if config.hf_home:
        env["HF_HOME"] = config.hf_home
    binary_dir = str(Path(config.llama_server_binary).resolve().parent)
    if os.name == "nt":
        env["PATH"] = binary_dir + (os.pathsep + env["PATH"] if env.get("PATH") else "")
    elif sys.platform == "darwin":
        env["DYLD_LIBRARY_PATH"] = binary_dir + (
            os.pathsep + env["DYLD_LIBRARY_PATH"] if env.get("DYLD_LIBRARY_PATH") else ""
        )
    else:
        env["LD_LIBRARY_PATH"] = binary_dir + (
            os.pathsep + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else ""
        )
    if config.decode_audit_bps == 0:
        manifest_format = "compact-raw-v2"
    elif config.decode_audit_bps > 0 and (
        config.proof_sample_bps == 0 or config.proof_sample_bps >= 10_000
    ):
        manifest_format = "decode"
    elif config.proof_sample_bps >= 10_000 and config.decode_audit_bps == 0:
        manifest_format = "none"
    else:
        manifest_format = "compact"
    trace_candidates = max(
        int(config.proof_trace_candidates_per_request),
        int(config.proof_ops_per_request),
    )
    if config.decode_audit_bps == 0 or (
        manifest_format == "decode" and int(config.decode_audit_bps) >= 10_000
    ):
        trace_candidates = 0
    env["VERATHOS_GGML_TRACE_MAX_OPS_PER_CAPTURE"] = str(trace_candidates)
    env["VERATHOS_GGML_TRACE_MAX_OPS_PER_GRAPH"] = str(trace_candidates)
    env.setdefault("VERATHOS_GGML_TRACE_MANIFEST_FORMAT", manifest_format)
    env.setdefault("VERATHOS_GGML_TRACE_MAX_ELEMS", str(_trace_max_elems(config)))
    if config.proof_gguf_manifest_path and int(config.decode_audit_bps) > 0:
        env.setdefault("VERATHOS_GGML_TRACE_SKIP_SRC0_DUMP", "1")
    return env


def _trace_max_elems(config: SmokeConfig) -> int:
    if config.proof_gguf_manifest_path:
        try:
            manifest = load_gguf_tensor_manifest(config.proof_gguf_manifest_path)
            return suggest_ggml_trace_max_elems_from_manifest(manifest)
        except Exception:
            pass
    if config.llama_model:
        return suggest_ggml_trace_max_elems_from_gguf_model(config.llama_model)
    return GGML_TRACE_MAX_ELEMS_FLOOR


def _llama_model_args(config: SmokeConfig) -> list[str]:
    if bool(config.llama_model) == bool(config.llama_hf):
        raise ValueError("exactly one of llama_model or llama_hf is required")
    if config.llama_hf:
        return ["--llama-hf", config.llama_hf]
    return ["--llama-model", config.llama_model]


def _direct_model_args(config: SmokeConfig) -> list[str]:
    if bool(config.llama_model) == bool(config.llama_hf):
        raise ValueError("exactly one of llama_model or llama_hf is required")
    if config.llama_hf:
        return ["-hf", config.llama_hf]
    return ["--model", config.llama_model]


def run_local_smoke(config: SmokeConfig) -> dict[str, Any]:
    """Run local coordinator + worker verified mesh inference and cleanup."""

    output_root = config.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    env = _runtime_env(config)
    request = {
        "model": config.model_id,
        "messages": [{"role": "user", "content": config.prompt}],
        "stream": False,
        "max_tokens": int(config.max_tokens),
        "temperature": 0,
    }
    result: dict[str, Any] = {
        "root": str(output_root),
        "model_id": config.model_id,
        "samples": int(config.samples),
    }

    direct_procs: list[ManagedProcess] = []
    if config.direct_baseline:
        try:
            rpc_port = _free_port()
            llama_port = _free_port()
            direct_rpc = ManagedProcess(
                "direct-rpc",
                [config.rpc_worker_binary, "-H", "127.0.0.1", "-p", str(rpc_port), "--device", config.rpc_device],
                cwd=config.repo_root,
                env=env,
            )
            direct_procs = [direct_rpc]
            _wait_tcp("127.0.0.1", rpc_port, direct_procs, timeout=config.timeout)
            direct_llama = ManagedProcess(
                "direct-llama",
                [
                    config.llama_server_binary,
                    *_direct_model_args(config),
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(llama_port),
                    "--rpc",
                    f"127.0.0.1:{rpc_port}",
                    "--device",
                    config.llama_device,
                    "--n-gpu-layers",
                    "all",
                    "--alias",
                    config.model_id,
                ],
                cwd=config.repo_root,
                env=env,
            )
            direct_procs.append(direct_llama)
            result["direct"] = _measure(
                f"http://127.0.0.1:{llama_port}",
                request,
                direct_procs,
                samples=config.samples,
                timeout=config.timeout,
            )
        finally:
            for proc in reversed(direct_procs):
                proc.stop()

    mesh_procs: list[ManagedProcess] = []
    trace_dir = output_root / "ggml-traces"
    try:
        coord_port = _free_port()
        worker_port = _free_port()
        rpc_port = _free_port()
        llama_port = _free_port()
        coord_endpoint = f"http://127.0.0.1:{coord_port}"
        worker_endpoint = f"http://127.0.0.1:{worker_port}"
        spec = MeshSpec.new_private_mesh(
            coordinator_uid=0,
            coordinator_hotkey="5SmokeCoordinator",
            endpoint=coord_endpoint,
            model_id=config.model_id,
            model_package_ref=config.llama_hf or config.llama_model,
            model_package_hash=config.package_hash,
            model_tensor_manifest_root=config.model_tensor_manifest_root,
            total_layers=config.layers,
        )
        coord_dir, _, token = create_mesh_state(spec=spec, root=output_root / "coord")
        coord_cmd = [
            sys.executable,
            "-m",
            "neurons.cli",
            "mesh",
            "serve",
            "--mesh",
            str(coord_dir),
            "--host",
            "127.0.0.1",
            "--port",
            str(coord_port),
            *_llama_model_args(config),
            "--llama-server-binary",
            config.llama_server_binary,
            "--llama-host",
            "127.0.0.1",
            "--llama-port",
            str(llama_port),
            "--llama-device",
            config.llama_device,
            "--llama-n-gpu-layers",
            "all",
            "--llama-min-rpc-workers",
            "1",
        ]
        if config.require_proof:
            coord_cmd.extend(
                [
                    "--require-proof",
                    "--proof-sample-bps",
                    str(config.proof_sample_bps),
                    "--proof-ops-per-request",
                    str(config.proof_ops_per_request),
                    "--proof-trace-candidates-per-request",
                    str(config.proof_trace_candidates_per_request),
                    "--proof-tolerance-abs",
                    str(config.proof_tolerance_abs),
                    "--proof-tolerance-rel",
                    str(config.proof_tolerance_rel),
                    "--decode-audit-bps",
                    str(config.decode_audit_bps),
                    "--decode-audit-top-k",
                    str(config.decode_audit_top_k),
                ]
            )
            if config.proof_gguf_manifest_path:
                coord_cmd.extend(["--proof-gguf-manifest", config.proof_gguf_manifest_path])
        coord = ManagedProcess("coordinator", coord_cmd, cwd=config.repo_root, env=env)
        mesh_procs = [coord]
        _wait_probe(coord_endpoint, mesh_procs, timeout=config.timeout)
        worker_dir, joined = join_mesh(
            token=token.encode(),
            endpoint=worker_endpoint,
            root=output_root / "worker",
            package_hash=config.package_hash,
            rpc_endpoint=f"127.0.0.1:{rpc_port}",
            proof_endpoint=worker_endpoint,
            timeout=10.0,
        )
        worker_cmd = [
            sys.executable,
            "-m",
            "neurons.cli",
            "mesh",
            "serve",
            "--mesh",
            str(worker_dir),
            "--host",
            "127.0.0.1",
            "--port",
            str(worker_port),
            "--rpc-worker",
            "--rpc-worker-binary",
            config.rpc_worker_binary,
            "--rpc-host",
            "127.0.0.1",
            "--rpc-port",
            str(rpc_port),
            "--rpc-device",
            config.rpc_device,
        ]
        if config.require_proof:
            worker_cmd.extend(
                [
                    "--proof-trace-dir",
                    str(trace_dir),
                    "--require-proof",
                    "--proof-sample-bps",
                    str(config.proof_sample_bps),
                    "--proof-ops-per-request",
                    str(config.proof_ops_per_request),
                    "--proof-trace-candidates-per-request",
                    str(config.proof_trace_candidates_per_request),
                    "--proof-tolerance-abs",
                    str(config.proof_tolerance_abs),
                    "--proof-tolerance-rel",
                    str(config.proof_tolerance_rel),
                    "--decode-audit-bps",
                    str(config.decode_audit_bps),
                    "--decode-audit-top-k",
                    str(config.decode_audit_top_k),
                ]
            )
            if config.proof_gguf_manifest_path:
                worker_cmd.extend(["--proof-gguf-manifest", config.proof_gguf_manifest_path])
        worker = ManagedProcess("worker", worker_cmd, cwd=config.repo_root, env=env)
        mesh_procs.append(worker)
        _wait_probe(worker_endpoint, mesh_procs, timeout=config.timeout)
        verified = _measure(
            coord_endpoint,
            request,
            mesh_procs,
            samples=config.samples,
            timeout=config.timeout,
            validator_nonce=bool(
                config.require_proof
                and 0 < max(config.proof_sample_bps, config.decode_audit_bps) < 10_000
            ),
        )
        sampled_items = [item for item in verified["samples"] if item.get("proof_sampled") is True]
        result.update(
            {
                "mesh_id": joined.mesh_id,
                "coordinator_endpoint": coord_endpoint,
                "worker_endpoint": worker_endpoint,
                "rpc_endpoint": f"127.0.0.1:{rpc_port}",
                "verified": verified,
                "trace_json_count": len(list(trace_dir.glob("*.json"))) if trace_dir.exists() else 0,
                "verified_all_samples": all(
                    item.get("verified") is True and item.get("proof_receipt_verified") is True
                    for item in verified["samples"]
                ),
                "sampled_proofs_verified": all(
                    item.get("verified") is True and item.get("proof_receipt_verified") is True
                    for item in sampled_items
                ),
            }
        )
    finally:
        for proc in reversed(mesh_procs):
            proc.stop()

    if "direct" in result and "verified" in result:
        overhead = float(result["verified"]["mean_s"]) - float(result["direct"]["mean_s"])
        result["mean_overhead_s"] = overhead
        result["mean_overhead_ms"] = overhead * 1000.0
        result["mean_overhead_pct"] = overhead / float(result["direct"]["mean_s"]) * 100.0
    return result


def default_package_hash(model_ref: str) -> str:
    return hashlib.sha256(model_ref.encode("utf-8")).hexdigest()


def default_smoke_root() -> Path:
    return Path(tempfile.mkdtemp(prefix="verathos-mesh-smoke-"))
