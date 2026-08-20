"""Interactive mesh CLI startup must stay torch-free.

The package init and the CLI module once pulled torch transitively (via
ggml_proof / gguf_manifest / smoke); every chat, manage, and fleet start
paid seconds of imports on a quiet box and, under a disk-saturating
model fetch, minutes that looked like a hard hang. Heavy modules load
inside the commands that need them.
"""

from __future__ import annotations

import subprocess
import sys


def _torch_free(statement: str) -> None:
    code = (
        "import sys\n"
        f"{statement}\n"
        "raise SystemExit(1 if 'torch' in sys.modules else 0)\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert completed.returncode == 0, (
        f"torch leaked into: {statement}\n{completed.stderr[-800:]}"
    )


def test_mesh_package_import_is_torch_free():
    _torch_free("import verallm.mesh")


def test_mesh_cli_import_is_torch_free():
    _torch_free("import verallm.mesh.cli")


def test_light_package_names_stay_light():
    _torch_free(
        "from verallm.mesh import MeshSpec, normalize_proof_sample_bps"
    )


def test_lazy_names_still_resolve():
    # The lazy façade must still hand out every heavy name on demand.
    from verallm.mesh import GgmlMulMatTrace, quantize_proof_i8

    assert GgmlMulMatTrace is not None
    assert callable(quantize_proof_i8)
