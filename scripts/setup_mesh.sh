#!/bin/bash
# =============================================================================
# Verathos GGUF Mesh Environment Setup
# =============================================================================
#
# Dependency bootstrap shared by both mesh roles:
#   coordinator - runs the pool manager (no GPU required)
#   worker      - serves model stages on its GPUs (join_pool.sh finishes this)
#
# Creates .venv-mesh, the same venv join_pool.sh uses, so one machine can be
# both roles. CUDA torch is installed only when a GPU is present: a CPU-only
# coordinator must not pull the 2.5 GB CUDA wheel, and a GPU box must not be
# downgraded to the CPU wheel by a later coordinator run.
#
# The existing vLLM miner path (setup_miner.sh, .venv-vllm, the "miner" PM2
# unit) is untouched and stays available in parallel.
#
# Usage:
#   bash scripts/setup_mesh.sh                       # Full install
#   bash scripts/setup_mesh.sh --role coordinator    # Same, names the intent
#   bash scripts/setup_mesh.sh --skip-install        # Verify only
#
# =============================================================================

set -e

ROLE="worker"
SKIP_INSTALL=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --role) ROLE="$2"; shift 2 ;;
        --skip-install) SKIP_INSTALL=true; shift ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

case "$ROLE" in
    coordinator|worker) ;;
    *) echo "Unknown role: $ROLE (use coordinator or worker)"; exit 1 ;;
esac

echo ""
echo "============================================================"
echo "  Verathos Mesh Environment Setup ($ROLE)"
echo "============================================================"

# ── Locate repo ──────────────────────────────────────────────────────────────

if [ -f "pyproject.toml" ] && grep -q "verathos" pyproject.toml 2>/dev/null; then
    REPO_DIR="$(pwd)"
else
    REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
echo "  Repo: $REPO_DIR"
cd "$REPO_DIR"

# ── System packages (Linux/apt; macOS ships its toolchain) ──────────────────

SUDO=""
[ "$(id -u)" -ne 0 ] && command -v sudo >/dev/null && SUDO="sudo"
if [ "$SKIP_INSTALL" = false ] && command -v apt-get >/dev/null 2>&1; then
    MISSING=""
    command -v git >/dev/null || MISSING="$MISSING git"
    # Workers verify port ownership with lsof before every drive.
    command -v lsof >/dev/null || MISSING="$MISSING lsof"
    dpkg -s python3-venv >/dev/null 2>&1 || MISSING="$MISSING python3-venv"
    dpkg -s python3-dev  >/dev/null 2>&1 || MISSING="$MISSING python3-dev"
    if [ -n "$MISSING" ]; then
        echo "  Installing system packages:$MISSING"
        $SUDO apt-get update -qq
        # shellcheck disable=SC2086
        $SUDO apt-get install -y -qq $MISSING >/dev/null
    fi
fi

# ── PM2 (process manager for the pool manager and worker units) ─────────────

if [ "$SKIP_INSTALL" = false ] && ! command -v pm2 >/dev/null 2>&1; then
    echo "  Installing PM2..."
    if ! command -v npm >/dev/null 2>&1; then
        if command -v apt-get >/dev/null 2>&1; then
            $SUDO apt-get install -y -qq nodejs npm >/dev/null
        else
            echo "  WARNING: npm not found; install Node.js + PM2 manually."
        fi
    fi
    command -v npm >/dev/null 2>&1 && $SUDO npm install -g pm2 >/dev/null
fi

# ── Python venv (.venv-mesh, shared with join_pool.sh) ──────────────────────

VENV="$REPO_DIR/.venv-mesh"
if [ "$SKIP_INSTALL" = false ]; then
    if [ ! -d "$VENV" ]; then
        python3 -m venv "$VENV"
        "$VENV/bin/pip" install -q --upgrade pip
    fi
    if command -v nvidia-smi >/dev/null 2>&1 \
      && ! "$VENV/bin/python" -c "import torch; assert torch.version.cuda" >/dev/null 2>&1; then
        echo "  Installing CUDA torch (GPU detected)..."
        # Pinned to the majors the shipped zkllm wheels carry native kernels for.
        "$VENV/bin/pip" install -q "torch>=2.10,<2.12" --index-url https://download.pytorch.org/whl/cu128
    fi
    echo "  Installing verathos[neurons]..."
    "$VENV/bin/pip" install -q -e "$REPO_DIR[neurons]" huggingface_hub >/dev/null
fi

# ── zkllm native prover (mandatory on GPU boxes) ────────────────────────────
# Without it every hard proof runs on the NumPy sumcheck fallback, 3-6x
# slower: silent capability loss, not a degraded mode we ship. The prover
# SHIPS as prebuilt wheels in dist/ (one per Python minor, torch/CUDA
# variants embedded, runtime-selected at import) - the public path never
# compiles source; a clean operator box has no toolchain and the public
# repo carries no buildable kernels. Workers refuse to start on a GPU box
# without the native prover.
if command -v nvidia-smi >/dev/null 2>&1; then
    # Public release trees carry no zkllm source: the venv gets the whole
    # package from the shipped wheel (--no-index so only dist/ can satisfy
    # it). Source checkouts skip this (the editable tree shadows the venv
    # package and is handled by the materialization below).
    if ! "$VENV/bin/python" -c "import zkllm" >/dev/null 2>&1; then
        echo "  Installing zkllm prover wheel..."
        "$VENV/bin/pip" install -q --no-cache-dir --no-index \
            --find-links "$REPO_DIR/dist" zkllm \
            || { echo "ERROR: no installable zkllm wheel in dist/ for $("$VENV/bin/python" -V)."; exit 1; }
    fi
    if ! "$VENV/bin/python" - <<'PYEOF' >/dev/null 2>&1
from zkllm.crypto import sumcheck_fast
from zkllm.crypto.pcs_v2 import native_library_path
assert sumcheck_fast._HAS_CUDA or sumcheck_fast._HAS_NATIVE
native_library_path()
PYEOF
    then
        if ! ls "$REPO_DIR/dist"/zkllm-*.whl >/dev/null 2>&1; then
            echo "ERROR: no zkllm wheel in $REPO_DIR/dist - the checkout is incomplete."
            exit 1
        fi
        if [ ! -d "$REPO_DIR/zkllm/cuda" ]; then
            # Public tree, wheel installed, and the native check still
            # failed: the wheel genuinely cannot serve this box.
            echo "ERROR: the shipped zkllm wheel failed its native check on this box"
            echo "       (torch $("$VENV/bin/python" -c 'import torch;print(torch.__version__)' 2>/dev/null))."
            exit 1
        fi
        # Source checkout: the editable tree shadows any wheel-installed
        # package, so materialize the wheel's native libs INTO the tree
        # beside the loader (the same layout the worker bundle ships).
        # The gemm-v2 PCS sidecar rides in the same wheel; serving refuses
        # to start without it, so all native libraries are materialized.
        echo "  Installing zkllm native prover libs from the shipped wheel..."
        ZKLLM_TMP="$(mktemp -d)"
        "$VENV/bin/pip" install --no-cache-dir --no-deps --target "$ZKLLM_TMP" \
            --find-links "$REPO_DIR/dist" zkllm 2>&1 | tail -2
        cp "$ZKLLM_TMP"/zkllm/cuda/zkllm_native*.so* "$REPO_DIR/zkllm/cuda/" 2>/dev/null \
            || { echo "ERROR: shipped zkllm wheel contains no native libs."; rm -rf "$ZKLLM_TMP"; exit 1; }
        cp "$ZKLLM_TMP"/zkllm/crypto/libverathos_pcs_v2.* "$REPO_DIR/zkllm/crypto/" 2>/dev/null \
            || { echo "ERROR: shipped zkllm wheel contains no native PCS library."; rm -rf "$ZKLLM_TMP"; exit 1; }
        rm -rf "$ZKLLM_TMP"
        "$VENV/bin/python" - <<'PYEOF' \
            || { echo "ERROR: zkllm native prover unavailable after wheel install - this Python/torch/CUDA combination has no shipped variant. Report this; do not build from source."; exit 1; }
from zkllm.crypto import sumcheck_fast
from zkllm.crypto.pcs_v2 import native_library_path
assert sumcheck_fast._HAS_CUDA or sumcheck_fast._HAS_NATIVE
native_library_path()
PYEOF
    fi
    echo "  zkllm native prover OK."
fi

# ── Verify ──────────────────────────────────────────────────────────────────

"$VENV/bin/python" -c "import verallm.mesh.pool" \
    || { echo "ERROR: mesh control plane import failed"; exit 1; }
echo "  Mesh environment OK."

# ── verathos command on PATH ────────────────────────────────────────────────
# Mesh-only boxes get the plain `verathos` command globally so operators run
# `verathos mesh fleet` / `verathos mesh pool probe` without activating the
# venv. Boxes that also run the vLLM miner keep that venv's command (its
# .env.sh owns the name there); this box then uses $VENV/bin/verathos.
if [ ! -e /usr/local/bin/verathos ] && [ ! -d "$REPO_DIR/.venv-vllm" ]; then
    $SUDO ln -sfn "$VENV/bin/verathos" /usr/local/bin/verathos 2>/dev/null \
        && echo "  Installed command: verathos -> $VENV/bin/verathos"
fi

# ── Next steps ──────────────────────────────────────────────────────────────

if command -v verathos >/dev/null 2>&1; then
    VERATHOS_CMD="verathos"
else
    VERATHOS_CMD="$VENV/bin/verathos"
fi

echo ""
if [ "$ROLE" = "coordinator" ]; then
    echo "  Next: create the pool and start the manager:"
    echo "    $VERATHOS_CMD setup mesh-coordinator"
else
    echo "  Next: join a pool (get the token from your coordinator):"
    echo "    bash scripts/join_pool.sh --token vtpool_..."
fi
echo ""
