"""GPU Confidential Computing mode detection.

Detects whether NVIDIA GPUs are running in CC (Confidential Computing) mode.
CC mode encrypts GPU VRAM and PCIe traffic, extending the TEE boundary to GPU.

Supported GPUs: H100, H200, B200/GB200 (datacenter Hopper/Blackwell only).
Consumer GPUs (RTX series) do NOT support CC mode.
"""

from __future__ import annotations

import subprocess
import shutil
from dataclasses import dataclass


@dataclass
class GpuCCStatus:
    """CC mode status for a single GPU."""

    gpu_index: int
    gpu_name: str
    cc_mode: bool  # True if CC mode is active
    cc_capable: bool  # True if GPU supports CC mode
    error: str = ""  # Non-empty if detection failed

    @property
    def description(self) -> str:
        if self.error:
            return f"GPU {self.gpu_index}: {self.error}"
        mode = "CC ON" if self.cc_mode else "CC OFF"
        capable = " (CC-capable)" if self.cc_capable else " (not CC-capable)"
        return f"GPU {self.gpu_index}: {self.gpu_name} — {mode}{capable}"


# GPU models known to support CC mode
CC_CAPABLE_MODELS = {
    "H100", "H200", "B100", "B200", "GB200",
}


def _is_cc_capable_model(gpu_name: str) -> bool:
    """Check if a GPU name indicates CC capability."""
    name_upper = gpu_name.upper()
    return any(model in name_upper for model in CC_CAPABLE_MODELS)


def detect_gpu_cc_status() -> list[GpuCCStatus]:
    """Detect CC mode status for all NVIDIA GPUs.

    Uses ``nvidia-smi conf-compute -gs`` to query CC mode. Falls back to
    model name heuristic if nvidia-smi doesn't support the query.

    Returns a list of :class:`GpuCCStatus`, one per GPU.
    """
    results: list[GpuCCStatus] = []

    # First, get GPU names
    gpu_names = _get_gpu_names()
    if not gpu_names:
        return results

    # Try nvidia-smi conf-compute for each GPU
    for idx, name in enumerate(gpu_names):
        cc_capable = _is_cc_capable_model(name)
        cc_mode = False
        error = ""

        if cc_capable:
            try:
                cc_mode = _query_cc_mode(idx)
            except CCDetectionError as e:
                error = str(e)

        results.append(GpuCCStatus(
            gpu_index=idx,
            gpu_name=name,
            cc_mode=cc_mode,
            cc_capable=cc_capable,
            error=error,
        ))

    return results


def require_gpu_cc_mode() -> None:
    """Assert that at least one GPU is in CC mode.

    Raises :class:`RuntimeError` if no CC-capable GPU is found or none are
    in CC mode. Called at server startup when ``--tee-enabled``.
    """
    statuses = detect_gpu_cc_status()
    if not statuses:
        raise RuntimeError(
            "No NVIDIA GPUs detected. TEE mode requires CC-capable GPU "
            "(H100, H200, B200)."
        )

    cc_gpus = [s for s in statuses if s.cc_mode]
    if not cc_gpus:
        capable = [s for s in statuses if s.cc_capable]
        if capable:
            names = ", ".join(s.gpu_name for s in capable)
            raise RuntimeError(
                f"CC-capable GPU(s) found ({names}) but CC mode is not active. "
                "Enable CC mode with nvidia-gpu-tools: "
                "--set-cc-mode=on --reset-after-cc-mode-switch"
            )
        else:
            names = ", ".join(s.gpu_name for s in statuses)
            raise RuntimeError(
                f"No CC-capable GPUs found. Detected: {names}. "
                "TEE mode requires H100, H200, or B200."
            )


class CCDetectionError(Exception):
    """Failed to detect GPU CC mode."""


def _get_gpu_names() -> list[str]:
    """Return list of GPU names via nvidia-smi."""
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return []

    try:
        result = subprocess.run(
            [nvidia_smi, "--query-gpu=name", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return []
        return [line.strip() for line in result.stdout.strip().split("\n") if line.strip()]
    except (subprocess.TimeoutExpired, OSError):
        return []


def _query_cc_mode(gpu_index: int) -> bool:
    """Query CC mode for a specific GPU via nvidia-smi.

    Returns True if CC mode is active, False otherwise.
    Raises :class:`CCDetectionError` if the query fails.
    """
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        raise CCDetectionError("nvidia-smi not found")

    try:
        result = subprocess.run(
            [nvidia_smi, "conf-compute", "-gs", "-i", str(gpu_index)],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            # nvidia-smi on non-CC GPUs may not support conf-compute
            if "not supported" in result.stderr.lower() or "invalid" in result.stderr.lower():
                return False
            raise CCDetectionError(
                f"nvidia-smi conf-compute failed (rc={result.returncode}): "
                f"{result.stderr.strip()}"
            )

        output = result.stdout.lower()
        # nvidia-smi conf-compute -gs output includes "CC status: ON" or "CC status: OFF"
        if "on" in output:
            return True
        return False
    except subprocess.TimeoutExpired:
        raise CCDetectionError("nvidia-smi conf-compute timed out")
    except OSError as e:
        raise CCDetectionError(f"Failed to run nvidia-smi: {e}")
