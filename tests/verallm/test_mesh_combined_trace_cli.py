import argparse
import json

import pytest

from verallm.mesh import MeshSpec
from verallm.mesh.cli import decide_trace_manifest_format, main as mesh_cli_main
from verallm.mesh.state import create_mesh_state, load_mesh_state, state_mesh_spec


_DIGEST = "a" * 64


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


def _combined_profile(*, decode_audit_bps: int, rpc_worker: bool) -> argparse.Namespace:
    return argparse.Namespace(
        require_proof=True,
        proof_sample_bps=10_000,
        decode_audit_bps=decode_audit_bps,
        proof_trace_candidates_per_request=8,
        rpc_worker=rpc_worker,
        proof_gguf_manifest="",
    )


def _coordinator_state(tmp_path, *, suffix: str):
    spec = MeshSpec.new_private_mesh(
        coordinator_uid=1,
        coordinator_hotkey="5Coord",
        endpoint="http://coord.local:9338",
        model_id="model",
        model_package_hash=_DIGEST,
        total_layers=4,
    )
    return create_mesh_state(spec=spec, root=tmp_path / suffix)[0]


def test_additive_profile_keeps_one_manifest_pin_for_organic_and_canary():
    organic = _combined_profile(decode_audit_bps=1_000, rpc_worker=True)
    canary_override = _combined_profile(decode_audit_bps=10_000, rpc_worker=True)

    # A per-request canary may raise only the decode gate from 10% to 100%.
    # Both rates must therefore be representable by the one mesh-wide format
    # pinned when the organic server starts.
    assert decide_trace_manifest_format(organic) == "compact-raw-v3"
    assert decide_trace_manifest_format(canary_override) == "compact-raw-v3"


@pytest.mark.parametrize("decode_audit_bps", [1_000, 10_000])
@requires_native_proof_stack
def test_slot_view_rpc_runtime_takes_no_serve_dumps(
    tmp_path,
    capsys,
    decode_audit_bps,
):
    """A decode-audited v3 serve is slot-view: witnesses come from
    slot-view leaves, the tail ring, and the exclusive probe window
    (whose selected dumps bypass this budget), so the serve-time dump
    budget is ZERO. Coupling it to the wide candidate window was
    live-measured writing 29 GB of activation dumps during one 31k-token
    prefill and tripling its wall time (407 vs 1216 tok/s)."""
    state_dir = _coordinator_state(tmp_path, suffix=str(decode_audit_bps))

    mesh_cli_main(
        [
            "serve",
            "--mesh",
            str(state_dir),
            "--rpc-worker",
            "--rpc-worker-binary",
            "verathos-rpc-server",
            "--require-proof",
            "--proof-sample-bps",
            "10000",
            "--decode-audit-bps",
            str(decode_audit_bps),
            "--proof-trace-candidates-per-request",
            "11",
            "--rpc-dry-run",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    runtime = payload["proof_runtime_env"]
    pinned = state_mesh_spec(load_mesh_state(state_dir)).proof_trace_manifest_format

    assert runtime["VERATHOS_GGML_TRACE_MANIFEST_FORMAT"] == "compact-raw-v3"
    assert pinned == runtime["VERATHOS_GGML_TRACE_MANIFEST_FORMAT"]
    assert runtime["VERATHOS_GGML_TRACE_MAX_OPS_PER_CAPTURE"] == "0"
    assert runtime["VERATHOS_GGML_TRACE_MAX_OPS_PER_GRAPH"] == "0"
    # Proof capture needs a deterministic op stream: CUDA decode-GEMV
    # fusions splice ops out of the graph nondeterministically, and a
    # slot-view leaf drawn on a fused-away op fails an honest node.
    assert runtime["GGML_CUDA_DISABLE_FUSION"] == "1"
