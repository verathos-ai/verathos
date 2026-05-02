#!/bin/bash
# =============================================================================
# Verathos Miner Environment Setup
# =============================================================================
#
# One-command environment setup for Bittensor miners on GPU cloud providers
# (cloud GPU providers, etc.).  Installs all dependencies, checks vLLM CUDA
# compatibility, and verifies the zkllm CUDA extension wheel.
#
# After setup completes, follow the printed next-steps to create a wallet,
# fund your EVM address, and start the miner.
#
# Usage:
#   bash scripts/setup_miner.sh                  # Full install
#   bash scripts/setup_miner.sh --skip-install    # Skip install, just verify
#
# Environment variables:
#   WORKSPACE          - Persistent volume path (default: /workspace)
#   HF_HOME            - HuggingFace cache dir (default: $WORKSPACE/huggingface)
#   VERATHOS_REPO_URL  - Git clone URL (default: current directory if present)
#
# =============================================================================

set -e

# ── Parse arguments ──────────────────────────────────────────────────────────

SKIP_INSTALL=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-install) SKIP_INSTALL=true; shift ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

# ── Paths ────────────────────────────────────────────────────────────────────

# WORKSPACE defaults to /workspace on cloud GPU containers; fall back to
# a user-owned dir on bare-metal servers where /workspace doesn't exist or
# isn't writable.
if [ -z "$WORKSPACE" ]; then
    if [ -d /workspace ] && [ -w /workspace ]; then
        WORKSPACE=/workspace
    else
        WORKSPACE="$HOME/.verathos-workspace"
        mkdir -p "$WORKSPACE"
    fi
fi
export HF_HOME="${HF_HOME:-${WORKSPACE}/huggingface}"
PYTHON="${PYTHON:-python3}"

echo ""
echo "============================================================"
echo "  Verathos Miner Environment Setup"
echo "============================================================"
echo "  Workspace: $WORKSPACE"
echo "  HF cache:  $HF_HOME"
echo "============================================================"

# ── Early GPU check (fail fast before installing anything) ───────────────────
if ! command -v nvidia-smi &>/dev/null; then
    echo "  ERROR: nvidia-smi not found. NVIDIA GPU required."
    exit 1
fi
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
GPU_VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
if [ -z "$GPU_NAME" ]; then
    echo "  ERROR: No GPU detected."
    exit 1
fi
if [ "${GPU_VRAM:-0}" -lt 20000 ] 2>/dev/null; then
    echo "  ERROR: GPU has ${GPU_VRAM}MB VRAM. Minimum 24GB required."
    exit 1
fi
GPU_SM=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
GPU_DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
GPU_DRIVER_MAJOR=$(echo "$GPU_DRIVER" | cut -d. -f1)
echo "  GPU: $GPU_NAME (${GPU_VRAM}MB, sm_${GPU_SM}, driver ${GPU_DRIVER})"
# Early check: RTX 5090 requires NVIDIA driver >= 575.
#
# History: driver 570 had a PTX JIT bug for sm_120 that crashed vLLM
# during CUDA graph capture (verified incident 2026-04-01).  We
# originally set the floor at 580 because that was the next version we
# tested.  On 2026-04-11 a fresh install on driver 575.57.08 went all
# the way through CUDA graph capture + warmup + 5/5 verified canaries
# (UID 91, RTX 5090, Qwen3.5-9B fp16) — confirming 575 is also fine.
# So the practical floor is 575.
#
# Other Blackwell GPUs (RTX PRO 6000, B200, etc.) work on driver 570+.
if [ "${GPU_SM:-0}" -ge 120 ] && [ "${GPU_DRIVER_MAJOR:-0}" -lt 575 ]; then
    if echo "$GPU_NAME" | grep -qi "5090"; then
        echo ""
        echo "  ERROR: RTX 5090 requires NVIDIA driver >= 575."
        echo "  Current driver: ${GPU_DRIVER}"
        echo "  Update: https://developer.nvidia.com/cuda-downloads"
        exit 1
    else
        echo "  Note: Blackwell GPU on driver ${GPU_DRIVER} (< 575) — should work, but update if issues arise."
    fi
fi

# ── Clone or update repo ─────────────────────────────────────────────────────

if [ -f "pyproject.toml" ] && grep -q "verathos" pyproject.toml 2>/dev/null; then
    REPO_DIR="$(pwd)"
    echo "  Using current directory: $REPO_DIR"
elif [ -d "${WORKSPACE}/verathos" ]; then
    REPO_DIR="${WORKSPACE}/verathos"
    echo "  Updating existing repo at $REPO_DIR..."
    cd "$REPO_DIR" && git pull
else
    if [ -z "$VERATHOS_REPO_URL" ]; then
        echo "ERROR: Repo not found and VERATHOS_REPO_URL not set."
        echo "Either clone the repo manually or set VERATHOS_REPO_URL."
        exit 1
    fi
    REPO_DIR="${WORKSPACE}/verathos"
    echo "  Cloning repo to $REPO_DIR..."
    git clone "$VERATHOS_REPO_URL" "$REPO_DIR"
fi

cd "$REPO_DIR"

# ── Platform artifact compatibility ─────────────────────────────────────────
# Verathos ships native artifacts for zkllm and several verallm packages.  Fail
# before the expensive dependency install if this checkout does not contain
# artifacts for the current CPU architecture / Python ABI.

python_tag() {
    $PYTHON -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')"
}

platform_arch_tag() {
    case "$(uname -m)" in
        x86_64|amd64) echo "x86_64" ;;
        aarch64|arm64) echo "aarch64" ;;
        *) uname -m ;;
    esac
}

find_zkllm_wheel() {
    local py_tag="$1"
    local arch_tag="$2"
    find "$REPO_DIR/dist" -maxdepth 1 -type f -name "zkllm-*-${py_tag}-*.whl" 2>/dev/null \
        | grep -E "(linux|manylinux).*(${arch_tag}|arm64)" \
        | sort \
        | head -1
}

check_native_artifacts() {
    local py_tag arch_tag wheel has_error
    py_tag="$(python_tag)"
    arch_tag="$(platform_arch_tag)"
    wheel="$(find_zkllm_wheel "$py_tag" "$arch_tag")"
    has_error=0

    echo "  Platform: $(uname -m) (${arch_tag}), Python ABI: ${py_tag}"

    if [ -z "$wheel" ]; then
        echo ""
        echo "  ERROR: No compatible zkllm wheel found for ${py_tag}/linux_${arch_tag}."
        echo "  Available zkllm wheels:"
        find "$REPO_DIR/dist" -maxdepth 1 -type f -name "zkllm-*.whl" -printf "    %f\n" 2>/dev/null | sort
        echo ""
        echo "  This platform needs a wheel named like:"
        echo "    zkllm-<version>-${py_tag}-${py_tag}-linux_${arch_tag}.whl"
        echo "  or a compatible manylinux_${arch_tag} wheel."
        echo ""
        echo "  DGX Spark is ARM64/aarch64, so x86_64 wheels cannot be used."
        echo "  See docs/dgx_spark_arm64.md for the required ARM64 release artifacts."
        has_error=1
    fi

    if ! $PYTHON - "$REPO_DIR" "$arch_tag" <<'PY'
import pathlib
import sys

repo = pathlib.Path(sys.argv[1])
arch = sys.argv[2]
binary_dirs = [
    repo / "verallm" / "miner",
    repo / "verallm" / "moe",
    repo / "verallm" / "prover",
    repo / "verallm" / "verifier",
]
missing = []
for directory in binary_dirs:
    if not directory.is_dir():
        continue
    so_files = list(directory.glob("*.so"))
    if so_files and not any(arch in path.name or "abi3" in path.name for path in so_files):
        missing.append(str(directory.relative_to(repo)))

if missing:
    print("")
    print("  ERROR: Native verallm extension artifacts are missing for this architecture.")
    print(f"  Required architecture: {arch}")
    print("  Directories with only incompatible extension binaries:")
    for item in missing:
        print(f"    {item}")
    print("")
    print("  Publish matching ARM64/aarch64 extension artifacts or source-build these")
    print("  modules during setup. See docs/dgx_spark_arm64.md.")
    sys.exit(1)
PY
    then
        has_error=1
    fi

    if [ "$has_error" -ne 0 ]; then
        exit 1
    fi
}

check_native_artifacts

# ── LD_LIBRARY_PATH: find pip-installed NVIDIA libs ──────────────────────────
# torch 2.9+ (from vLLM pip) needs libcusparseLt.so.0 which lives in
# site-packages/nvidia/*/lib/ — not on the default search path.

fix_ld_library_path() {
    local nvidia_libs
    nvidia_libs=$($PYTHON -c "
import site, glob, os
dirs = []
try:
    for sp in site.getsitepackages():
        dirs.extend(glob.glob(f'{sp}/nvidia/*/lib'))
        torch_lib = os.path.join(sp, 'torch', 'lib')
        if os.path.isdir(torch_lib):
            dirs.append(torch_lib)
except Exception:
    pass
for d in sorted(glob.glob('/usr/local/cuda*/lib64')):
    dirs.append(d)
print(':'.join(d for d in dirs if os.path.isdir(d)))
" 2>/dev/null || true)

    if [ -n "$nvidia_libs" ]; then
        export LD_LIBRARY_PATH="${nvidia_libs}:${LD_LIBRARY_PATH:-}"
        echo "  LD_LIBRARY_PATH updated (NVIDIA libs detected)"
    fi
}

fix_ld_library_path

# ── vLLM CUDA kernel compatibility check ─────────────────────────────────────

check_vllm_cuda_compat() {
    $PYTHON -c "
import torch, subprocess, site, glob, os, sys

cc = torch.cuda.get_device_capability(0)
sm = cc[0] * 10 + cc[1]
sm_str = f'sm_{sm}'


try:
    sp = site.getsitepackages()[0]
except Exception:
    sp = os.path.join(sys.prefix, 'lib',
         f'python{sys.version_info.major}.{sys.version_info.minor}', 'site-packages')

sos = sorted(glob.glob(f'{sp}/vllm/_*.so'))
if not sos:
    sys.exit(0)

for cmd in ['/usr/local/cuda/bin/cuobjdump', 'cuobjdump']:
    try:
        r = subprocess.run([cmd, '--list-elf', sos[0]],
                           capture_output=True, text=True, timeout=10)
        if r.returncode == 0:
            if sm_str not in r.stdout:
                # PTX forward-compat: NVIDIA GPUs can run cubins from the
                # same major arch via JIT.  e.g. sm_86 (RTX 3090, A6000)
                # runs on sm_80 cubins.  The official vLLM wheel ships
                # only select SMs to save space.  Check if the base SM
                # (same major, minor=0) is present — if so, trust the
                # runtime Marlin check below instead of forcing a rebuild.
                base_sm = (sm // 10) * 10
                base_str = f'sm_{base_sm}'
                if base_sm != sm and base_str in r.stdout:
                    print(f'{sm_str} not in wheel but {base_str} present — using PTX fallback')
                    break
                print(f'{sm_str} not found in vLLM CUDA kernels')
                sys.exit(1)
            break
    except (FileNotFoundError, subprocess.TimeoutExpired):
        continue
else:
    known = {80, 86, 89, 90}
    if sm not in known:
        print(f'{sm_str} not in known wheel archs {known}')
        sys.exit(1)

try:
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        marlin_permute_scales,
    )
    s = torch.ones(1, 128, dtype=torch.float16, device='cuda')
    marlin_permute_scales(s, size_k=128, size_n=128, group_size=128)
    torch.cuda.synchronize()
except Exception as e:
    err = str(e).lower()
    if 'ptx' in err or 'unsupported' in err or 'toolchain' in err:
        print(f'Marlin kernel broken on {sm_str}: {e}')
        sys.exit(1)

# For very new GPUs (sm_120+), pre-built wheels often have PTX compiled
# with an older CUDA toolkit that fails at runtime even though cuobjdump
# shows the arch. Force a source rebuild to be safe.
if sm >= 120:
    import vllm, os
    vllm_dir = os.path.dirname(vllm.__file__)
    # If vLLM was built from source on this machine, there will be a
    # build marker or the .so will have native SASS (not just PTX)
    native_sos = glob.glob(f'{vllm_dir}/**/*sm_{sm}*', recursive=True)
    if not native_sos:
        print(f'{sm_str} is very new — forcing source rebuild for full kernel compatibility')
        sys.exit(1)

sys.exit(0)
" 2>/dev/null
}

rebuild_vllm_from_source() {
    # Fail loudly on any error in this function — a silent half-built vLLM
    # would let the outer script exit 0 with a broken install.
    set -e

    local BUILD_DIR="${WORKSPACE}/vllm-build"
    if ! mkdir -p "$BUILD_DIR" 2>/dev/null; then
        echo "  ERROR: cannot create build dir $BUILD_DIR (WORKSPACE=$WORKSPACE not writable)"
        set +e
        return 1
    fi

    local CUDA_HOME_PATH=""
    for d in /usr/local/cuda /usr/local/cuda-*; do
        if [ -x "$d/bin/nvcc" ]; then
            CUDA_HOME_PATH="$d"
            break
        fi
    done
    if [ -z "$CUDA_HOME_PATH" ]; then
        echo "  ERROR: nvcc not found. Cannot rebuild vLLM from source."
        set +e
        return 1
    fi

    local ARCH_LIST
    # Detect GPU arch without torch (CUDA runtime may be in a bad state after
    # pip upgraded torch mid-process). Use nvidia-smi as fallback.
    ARCH_LIST=$($PYTHON -c "
try:
    import torch
    cc = torch.cuda.get_device_capability(0)
    print(f'{cc[0]}.{cc[1]}')
except Exception:
    import subprocess, re
    out = subprocess.check_output(['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader'], text=True)
    print(out.strip().split('\n')[0])
")

    local NJOBS=$(( $(nproc) < 32 ? $(nproc) : 32 ))

    echo "  CUDA_HOME=$CUDA_HOME_PATH"
    echo "  TORCH_CUDA_ARCH_LIST=$ARCH_LIST"
    echo "  MAX_JOBS=$NJOBS ($(nproc) CPUs available)"

    rm -rf "$BUILD_DIR"
    # Pin to v0.17.1 tag to avoid breaking changes on main branch
    git clone --depth 1 --branch v0.17.1 https://github.com/vllm-project/vllm.git "$BUILD_DIR"
    cd "$BUILD_DIR"

    export PATH="${CUDA_HOME_PATH}/bin:$PATH"
    export CUDA_HOME="$CUDA_HOME_PATH"
    export TORCH_CUDA_ARCH_LIST="$ARCH_LIST"
    export VLLM_TARGET_CUDA_ARCHS="$ARCH_LIST"
    export MAX_JOBS="$NJOBS"

    # Pin setuptools <81 for vLLM v0.17.1 build — newer setuptools rejects
    # the project.license format in vLLM's pyproject.toml.
    $PYTHON -m pip install --no-cache-dir cmake ninja wheel setuptools-scm "setuptools>=77,<81" 2>&1 | tail -3
    echo "  Compiling CUDA kernels for $ARCH_LIST (this will take 10-20 min)..."
    $PYTHON -m pip install --no-cache-dir --no-build-isolation . 2>&1 | \
        while IFS= read -r line; do
            case "$line" in
                *Building*|*Compiling*|*Installing*|*error*|*Error*|*SUCCESS*|*warning:*)
                    echo "  $line" ;;
            esac
        done

    cd "$REPO_DIR"
    rm -rf "$BUILD_DIR"

    # Also rebuild the zkllm CUDA extension for this GPU.
    # The pre-built wheel's .so may not load on new architectures.
    local ZKLLM_CUDA="$($PYTHON -c 'import zkllm; import os; print(os.path.join(os.path.dirname(zkllm.__file__), "cuda"))' 2>/dev/null)"
    if [ -n "$ZKLLM_CUDA" ] && [ -f "$ZKLLM_CUDA/build.py" ]; then
        echo "  Rebuilding zkllm CUDA extension for $ARCH_LIST..."
        (cd "$ZKLLM_CUDA" && $PYTHON build.py 2>&1 | tail -5)
    fi

    echo "  vLLM rebuilt from source for $ARCH_LIST"
    set +e
}

# ── Venv setup ───────────────────────────────────────────────────────────────

VENV_DIR="${REPO_DIR}/.venv-vllm"

# ── System dependencies ────────────────────────────────────────────────────
# ninja-build: required by FlashInfer JIT compilation on Hopper/Blackwell (sm_90+)
if ! command -v ninja &>/dev/null; then
    if sudo -n true 2>/dev/null; then
        if command -v apt-get &>/dev/null; then
            sudo apt-get install -y -qq ninja-build >/dev/null 2>&1 || true
        elif command -v dnf &>/dev/null; then
            sudo dnf install -y ninja-build >/dev/null 2>&1 || true
        fi
    fi
    # If system install failed, pip fallback happens after venv activation below
fi

if [ ! -d "$VENV_DIR" ]; then
    echo ""
    echo "  Creating venv with --system-site-packages..."
    if ! $PYTHON -m venv "$VENV_DIR" --system-site-packages 2>/dev/null; then
        echo "  python3-venv not installed — installing..."
        PY_VER=$($PYTHON -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        if command -v apt-get &>/dev/null; then
            sudo apt-get update -qq && sudo apt-get install -y -qq "python${PY_VER}-venv" 2>&1 | tail -3
        elif command -v dnf &>/dev/null; then
            sudo dnf install -y "python${PY_VER}-venv" 2>&1 | tail -3
        else
            echo "  ERROR: Cannot install python3-venv automatically."
            echo "  Please install it manually: apt install python${PY_VER}-venv"
            exit 1
        fi
        $PYTHON -m venv "$VENV_DIR" --system-site-packages
    fi
else
    echo "  Using existing venv: $VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
PYTHON="$VENV_DIR/bin/python"
fix_ld_library_path

echo "  Python: $($PYTHON --version 2>&1) ($PYTHON)"

# ninja fallback: if system install failed, install via pip into venv
if ! command -v ninja &>/dev/null; then
    $PYTHON -m pip install ninja -q 2>/dev/null || true
fi

# ── Install dependencies ─────────────────────────────────────────────────────

if [ "$SKIP_INSTALL" = false ]; then

    echo ""
    echo "Step 1/5: Installing zkllm wheel..."
    PY_TAG="$(python_tag)"
    ARCH_TAG="$(platform_arch_tag)"
    ZKLLM_WHEEL="$(find_zkllm_wheel "$PY_TAG" "$ARCH_TAG")"
    if [ -n "$ZKLLM_WHEEL" ]; then
        $PYTHON -m pip install --no-cache-dir "$ZKLLM_WHEEL" 2>&1 | tail -5
    else
        echo "  ERROR: No compatible zkllm wheel found for ${PY_TAG}/linux_${ARCH_TAG}."
        exit 1
    fi

    echo ""
    echo "Step 2/5: Installing Python dependencies..."
    # Upgrade pip + setuptools first — Ubuntu 22.04's system setuptools (59.x) is
    # too old for PEP 660 editable installs (needs >=64).
    $PYTHON -m pip install --no-cache-dir --upgrade pip setuptools 2>&1 | tail -5
    # Install in two passes to avoid setuptools conflict between bittensor (~=70)
    # and vllm (>=77). Install vllm first (upgrades setuptools), then bittensor
    # (downgrades setuptools — but both work fine at runtime).
    echo "  Installing vLLM + inference deps (this downloads several GB, may take 5-10 min)..."
    $PYTHON -m pip install --no-cache-dir -e ".[api,hashing,vllm]" 2>&1 | tail -20
    echo "  Installing Bittensor + neuron deps..."
    $PYTHON -m pip install --no-cache-dir -e ".[neurons]" 2>&1 | tail -20
    fix_ld_library_path

    # Pin async-substrate-interface <2: ASI 2.0 breaks bittensor 10.x
    # (removes ScaleObj) and introduces scalecodec/cyscale namespace conflict.
    # Force downgrade if 2.x was pulled transitively.
    $PYTHON -m pip install --no-cache-dir 'async-substrate-interface>=1.6,<2' 2>&1 | tail -3

    # Defensive pin: kernels 0.13.0 (released 2026-04-11) ships a
    # `str | None` annotation in PythonPackage.import_name that breaks
    # huggingface_hub 0.36.2's strict dataclass validator at import time
    # — every fresh transformers/vllm import crashes with
    # StrictDataclassFieldValidationError.  Pin to 0.12.x until upstream
    # ships a fix.  pyproject.toml [vllm] extra also pins this, but pip
    # may resolve transitively to 0.13 if a transitive dep upgrades it,
    # so we force the downgrade here as a belt-and-suspenders.
    echo "  Pinning kernels<0.13 (upstream 0.13.0 import-time crash)..."
    $PYTHON -m pip install --no-cache-dir 'kernels>=0.12,<0.13' 2>&1 | tail -3

    echo ""
    echo "  Checking vLLM CUDA kernel compatibility..."
    if ! check_vllm_cuda_compat; then
        echo "  vLLM wheel's CUDA kernels don't support this GPU — rebuilding from source..."
        echo "  (This compiles Marlin/quantization kernels for your GPU. May take 10-20 min.)"
        if ! rebuild_vllm_from_source; then
            echo ""
            echo "  ERROR: vLLM source rebuild failed — cannot continue."
            echo "  The miner will not start without working CUDA kernels."
            exit 1
        fi
        fix_ld_library_path
    else
        echo "  vLLM CUDA kernels: compatible"
    fi

    echo ""
    echo "Step 3/5: Installing gptqmodel (GPTQ quantization support)..."
    echo "  This compiles CUDA kernels from source — may take 5-10 minutes."
    echo "  (Only needed for GPTQ models; AWQ/fp16/fp8 work without it.)"
    # gptqmodel needs setuptools>=77 (bittensor downgrades to ~70).
    # After install, pin back transformers/protobuf/huggingface_hub to
    # versions compatible with vLLM — gptqmodel works fine at runtime
    # with older versions despite its pip constraints.
    $PYTHON -m pip install --no-cache-dir "setuptools>=77,<81" 2>&1 | tail -3
    $PYTHON -m pip install --no-build-isolation --no-cache-dir "gptqmodel>=0.9,<6.0" 2>&1 | tail -10 || {
        echo "  WARNING: gptqmodel install failed — GPTQ models will not be available."
        echo "  (Non-fatal; AWQ/fp16/fp8 models still work. Retry manually if needed:"
        echo "   pip install --no-build-isolation 'gptqmodel>=0.9,<6.0')"
    }
    # Restore deps that gptqmodel upgraded beyond vLLM's constraints.
    # gptqmodel works fine at runtime with these pinned versions.
    $PYTHON -m pip install --no-cache-dir \
        "transformers>=4.56,<5" "protobuf>=5.0,<7" "huggingface_hub>=0.28,<1.0" \
        2>&1 | tail -3

    echo ""
    echo "Step 4/5: Verifying torch + CUDA..."
    $PYTHON -c "
import torch, subprocess
assert torch.cuda.is_available(), 'CUDA not available!'
p = torch.cuda.get_device_properties(0)
cc = torch.cuda.get_device_capability(0)
sm = cc[0] * 10 + cc[1]
print(f'  torch {torch.__version__}, CUDA {torch.version.cuda}')
print(f'  GPU: {p.name}')
print(f'  VRAM: {p.total_memory / (1024**3):.1f} GB')
print(f'  Compute: sm_{sm}')

# Driver check for RTX 5090 specifically (passes torch.cuda but fails at
# runtime with PTX errors on driver < 575).  Driver 570 had a PTX JIT
# bug for sm_120 that crashed vLLM during CUDA graph capture; 575 fixed
# it (verified 2026-04-11 with UID 91, RTX 5090, Qwen3.5-9B fp16, full
# warmup + 5/5 verified canaries on driver 575.57.08).  Other Blackwell
# GPUs (RTX PRO 6000, B200 etc.) work fine on driver 570.
if sm >= 120:
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=driver_version,name', '--format=csv,noheader'],
            text=True, timeout=5).strip()
        driver_str, gpu_name = [x.strip() for x in out.split(',', 1)]
        driver_major = int(driver_str.split('.')[0])
        is_rtx5090 = '5090' in gpu_name
        if is_rtx5090 and driver_major < 575:
            print(f'')
            print(f'  ERROR: RTX 5090 requires NVIDIA driver >= 575.')
            print(f'  Current driver: {driver_str}')
            print(f'  Update: https://developer.nvidia.com/cuda-downloads')
            import sys; sys.exit(1)
    except (subprocess.SubprocessError, ValueError):
        pass

# Sanity check: verify CUDA actually works
try:
    torch.zeros(1, device='cuda')
except Exception as e:
    print(f'')
    print(f'  ERROR: CUDA not functional: {e}')
    print(f'  Check that your NVIDIA driver supports this GPU.')
    import sys; sys.exit(1)
" || {
        echo "ERROR: torch cannot access CUDA."
        echo "  LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
        echo "  Check nvidia-smi and torch installation."
        exit 1
    }

    echo ""
    echo "Step 5/5: Verifying zkllm CUDA extension..."
    fix_ld_library_path
    # zkllm is installed as a wheel — no build needed.
    # The wheel auto-detects the correct torch version at import time.
    if $PYTHON -c "
from zkllm.cuda import zkllm_native, HAS_CUDA
assert zkllm_native is not None, 'zkllm_native not loaded — is the zkllm wheel installed?'
assert HAS_CUDA, 'Extension loaded but HAS_CUDA=False — CUDA not available in zkllm'
assert hasattr(zkllm_native, 'cuda_blake3_merkle_leaves'), 'Missing cuda_blake3_merkle_leaves kernel'
print(f'  HAS_CUDA: {HAS_CUDA}')
print(f'  Kernels: blake3_merkle, sumcheck, field_ops')
" 2>/dev/null; then
        echo "  CUDA extension: OK"
    else
        echo ""
        echo "WARNING: zkllm CUDA extension not available or missing GPU kernels."
        echo ""
        echo "  Ensure the zkllm wheel is installed:"
        echo "    pip install zkllm-*.whl"
        echo ""
        echo "  See INSTALL_ZKLLM.md for download links and troubleshooting."
        echo ""
        echo "  Without the CUDA extension, Merkle tree computation takes 10-50x longer"
        echo "  and proof generation falls back to slow CPU paths."
        if [ "${VERATHOS_ALLOW_CPU_FALLBACK:-}" != "1" ]; then
            exit 1
        fi
        echo "  Continuing with CPU fallback (VERATHOS_ALLOW_CPU_FALLBACK=1)..."
    fi

    echo ""
    echo "Installation complete."
else
    echo "  Skipping install (--skip-install)"
fi

# ── Persist environment for future SSH sessions ──────────────────────────────

ENV_FILE="${REPO_DIR}/.env.sh"
cat > "$ENV_FILE" <<ENVEOF
# Auto-generated by setup_miner.sh — source from .bashrc for persistent env
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}"
export HF_HOME="${HF_HOME}"
export VLLM_ENABLE_V1_MULTIPROCESSING=0
if [ -f "${VENV_DIR}/bin/activate" ]; then
    source "${VENV_DIR}/bin/activate"
fi
ENVEOF

BASHRC="${HOME}/.bashrc"
if ! grep -qF "verathos/.env.sh" "$BASHRC" 2>/dev/null; then
    echo "" >> "$BASHRC"
    echo "# Verathos environment (auto-added by setup_miner.sh)" >> "$BASHRC"
    echo "[ -f \"${ENV_FILE}\" ] && source \"${ENV_FILE}\"" >> "$BASHRC"
    echo "  Persisted env to $ENV_FILE (sourced from .bashrc)"
else
    echo "  Env already persisted in .bashrc"
fi

# ── Show model recommendations + next steps ──────────────────────────────────

echo ""
echo "============================================================"
echo "  Miner environment ready!"
echo "============================================================"
echo ""
$PYTHON -m verallm.registry --recommend --verified-only 2>&1 || true
echo ""
echo "============================================================"
echo "  Next steps"
echo "============================================================"
echo ""
echo "  0. Activate the environment:"
echo "     cd $(pwd)"
echo "     source .venv-vllm/bin/activate"
echo ""
echo "  1. Set up wallet (on your local machine, then copy to this server):"
echo "     btcli wallet create --wallet.name miner"
echo "     btcli subnet register --wallet.name miner --netuid 96 --subtensor.network finney"
echo ""
echo "     Copy to this server (coldkeypub + hotkey only — never the coldkey):"
echo "     btcli wallet regen-coldkeypub --wallet.name miner --ss58-address <SS58>"
echo "     btcli wallet regen-hotkey --wallet.name miner --hotkey default"
echo ""
echo "  2. Fund your EVM address (one-time, for gas):"
echo "     python scripts/show_evm_info.py --wallet miner --hotkey default"
echo "     btcli wallet transfer --dest <SS58_MIRROR> --amount 0.1 --subtensor.network finney"
echo ""
echo "  3. Run the setup wizard to configure HTTPS, model, and PM2:"
echo "     verathos setup"
echo ""
echo "  See docs/setup.md for full details."
echo "============================================================"
