"""Combined hot-capacity audit proof verification."""

from __future__ import annotations

import hashlib
import json
import pathlib
import sys
from typing import Any, Mapping

from neurons.capacity_audit_workspace_proof import verify_workspace_proof_payload


SCRIPT_DIR = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "hot_capacity_workspace"
COMBINED_PROOF_FORMAT = "hot_capacity_combined_proof_v1"
COMBINED_WORKLOAD_VERSION = "hot_capacity_combined"


def _ensure_workspace_path() -> None:
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))


def _sha256_hex(domain: bytes, *parts: bytes) -> str:
    h = hashlib.sha256()
    h.update(domain)
    for part in parts:
        h.update(part)
    return h.hexdigest()


def _combined_transcript_root(
    *,
    capacity_transcript: str,
    capacity_tail_transcript: str,
    fp64_transcript: str,
    capacity_params: dict[str, Any],
    capacity_tail_params: dict[str, Any] | None,
    fp64_params: dict[str, Any],
) -> str:
    payload = json.dumps(
        {
            "version": COMBINED_WORKLOAD_VERSION,
            "capacity_transcript": capacity_transcript,
            "capacity_tail_transcript": capacity_tail_transcript,
            "fp64_transcript": fp64_transcript,
            "capacity_params": capacity_params,
            "capacity_tail_params": capacity_tail_params,
            "fp64_params": fp64_params,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return _sha256_hex(b"VERATHOS_HOT_CAPACITY_COMBINED_TRANSCRIPT_V1", payload)


def _sample_seed_from_b_proof(*, b_proof_seed: str, combined_transcript: str) -> str:
    return _sha256_hex(
        b"VERATHOS_HOT_CAPACITY_COMBINED_SAMPLE_V1",
        bytes.fromhex(str(b_proof_seed).removeprefix("0x")),
        bytes.fromhex(str(combined_transcript)),
    )


def _verify_fp64_identity_proof(proof: Mapping[str, Any]) -> tuple[bool, str]:
    try:
        from hot_capacity_workspace import bench_fp64_identity as fp64_identity  # type: ignore  # noqa: PLC0415
    except ImportError:
        _ensure_workspace_path()
        import bench_fp64_identity as fp64_identity  # type: ignore  # noqa: PLC0415

    return fp64_identity.verify_fp64_identity_proof(proof)


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _passes(params: Mapping[str, Any] | None) -> int:
    if not isinstance(params, Mapping):
        return 0
    try:
        return max(0, int(params.get("passes") or 0))
    except Exception:
        return 0


def verify_combined_proof_payload(
    *,
    proof: Mapping[str, Any],
    final_artifact: Mapping[str, Any],
    expected_combined_transcript_root: str,
    lease_id: str,
    gpu_index: int,
    proof_seed_hex: str,
    proof_challenge_seed_hex: str,
) -> tuple[bool, str]:
    """Verify a combined capacity audit proof payload.

    ``proof_challenge_seed_hex`` is the B_proof-derived seed committed by the
    validator from finalized chain data. The sampled lane openings are derived
    from that seed plus the pre-B_proof combined transcript.
    """

    proof = _as_dict(proof)
    if str(proof.get("format") or "") != COMBINED_PROOF_FORMAT:
        return False, "unsupported_combined_format"
    if str(proof.get("workload_version") or "") != COMBINED_WORKLOAD_VERSION:
        return False, "unsupported_combined_workload"

    combined_root = str(proof.get("combined_transcript_root") or "")
    expected_root = str(expected_combined_transcript_root or "")
    if not combined_root or combined_root != expected_root:
        return False, "combined_transcript_root_mismatch"

    final_commit = _as_dict(final_artifact.get("combined"))
    if final_commit:
        if str(final_commit.get("combined_transcript_root") or "") != combined_root:
            return False, "final_combined_transcript_root_mismatch"
        for key in ("capacity_transcript_root", "capacity_tail_transcript_root", "fp64_transcript_root"):
            if str(final_commit.get(key) or "") != str(proof.get(key) or ""):
                return False, f"final_{key}_mismatch"
        for key in ("capacity_params", "capacity_tail_params", "fp64_params"):
            final_value = final_commit.get(key)
            proof_value = proof.get(key)
            if final_value != proof_value:
                return False, f"final_{key}_mismatch"

    capacity_params = _as_dict(proof.get("capacity_params"))
    capacity_tail_params_raw = proof.get("capacity_tail_params")
    capacity_tail_params = _as_dict(capacity_tail_params_raw)
    fp64_params = _as_dict(proof.get("fp64_params"))
    recomputed_combined = _combined_transcript_root(
        capacity_transcript=str(proof.get("capacity_transcript_root") or ""),
        capacity_tail_transcript=str(proof.get("capacity_tail_transcript_root") or ""),
        fp64_transcript=str(proof.get("fp64_transcript_root") or ""),
        capacity_params=capacity_params,
        capacity_tail_params=capacity_tail_params_raw if isinstance(capacity_tail_params_raw, dict) else None,
        fp64_params=fp64_params,
    )
    if recomputed_combined != combined_root:
        return False, "recomputed_combined_transcript_mismatch"

    b_proof_seed = str(proof.get("b_proof_seed") or "").removeprefix("0x")
    expected_b_proof_seed = str(proof_challenge_seed_hex or "").removeprefix("0x")
    if b_proof_seed != expected_b_proof_seed:
        return False, "b_proof_seed_mismatch"
    expected_sample_seed = _sample_seed_from_b_proof(
        b_proof_seed=b_proof_seed,
        combined_transcript=combined_root,
    )
    sample_seed = str(proof.get("sample_seed") or "").removeprefix("0x")
    if sample_seed != expected_sample_seed:
        return False, "sample_seed_mismatch"

    capacity_proof = _as_dict(proof.get("capacity_proof"))
    cap_passes = _passes(capacity_params)
    if cap_passes <= 0:
        return False, "invalid_capacity_pass_count"
    ok, reason = verify_workspace_proof_payload(
        proof=capacity_proof,
        expected_transcript_root=str(proof.get("capacity_transcript_root") or ""),
        expected_pass0_root=str((capacity_proof.get("root_chain") or [{}])[0].get("pass_root") or ""),
        expected_final_root=str((capacity_proof.get("root_chain") or [{}])[-1].get("pass_root") or ""),
        lease_id=lease_id,
        gpu_index=int(gpu_index),
        proof_seed_hex=str(proof_seed_hex or ""),
        pass_count=cap_passes,
        proof_challenge_seed_hex=sample_seed,
    )
    if not ok:
        return False, f"capacity_{reason}"

    tail_passes = _passes(capacity_tail_params)
    if tail_passes > 0:
        tail_proof = _as_dict(proof.get("capacity_tail_proof"))
        ok, reason = verify_workspace_proof_payload(
            proof=tail_proof,
            expected_transcript_root=str(proof.get("capacity_tail_transcript_root") or ""),
            expected_pass0_root=str((tail_proof.get("root_chain") or [{}])[0].get("pass_root") or ""),
            expected_final_root=str((tail_proof.get("root_chain") or [{}])[-1].get("pass_root") or ""),
            lease_id=lease_id,
            gpu_index=int(gpu_index),
            proof_seed_hex=str(proof_seed_hex or ""),
            pass_count=tail_passes,
            proof_challenge_seed_hex=sample_seed,
        )
        if not ok:
            return False, f"capacity_tail_{reason}"

    fp64_proof = _as_dict(proof.get("fp64_proof"))
    fp64_passes = _passes(fp64_params)
    if fp64_passes > 0:
        ok, reason = _verify_fp64_identity_proof(fp64_proof)
        if not ok:
            return False, f"fp64_{reason}"
    elif fp64_proof:
        return False, "unexpected_fp64_proof"

    return True, "ok"
