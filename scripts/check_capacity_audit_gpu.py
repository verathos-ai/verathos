#!/usr/bin/env python3
"""Fail-fast support check for miner hot-capacity audit hardware."""

from __future__ import annotations

import argparse
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from neurons.capacity_audit import (
    calibrated_gpu_class_names,
    capacity_audit_gpu_support_status,
    capacity_gpu_workload_spec,
)


def _print_supported() -> None:
    print("Supported calibrated GPUs:")
    for name in calibrated_gpu_class_names():
        print(f"  - {name}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-name", required=True)
    parser.add_argument("--vram-gb", type=int, required=True)
    args = parser.parse_args(argv)

    ok, reason, row = capacity_audit_gpu_support_status(args.gpu_name, args.vram_gb)
    if ok and row is not None:
        spec = capacity_gpu_workload_spec(row)
        print(
            "  Hot-capacity GPU support: "
            f"{row.match_gpu_name}, pass_count={spec['pass_count']}, "
            f"deadline={row.deadline_s:.0f}s"
        )
        return 0

    print("")
    print("  ERROR: This GPU is not currently supported for hot-capacity score-gate mining.")
    print(f"  Detected GPU: {args.gpu_name} ({args.vram_gb} GB)")
    if reason == "uncalibrated_gpu_class" and row is not None:
        print(f"  Matching row exists but is not enabled yet: {row.match_gpu_name}")
    else:
        print("  No calibrated audit row matches the detected GPU name and VRAM.")
    print("")
    _print_supported()
    print("")
    print("  Use a supported calibrated GPU, or calibrate and release a new audit row first.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
