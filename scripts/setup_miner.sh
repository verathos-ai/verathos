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
set -o pipefail

resolve_miner_runtime_stack() {
    local gpu_sm="${1:-}"
    local driver_major="${2:-0}"

    if ! [[ "$gpu_sm" =~ ^[0-9]+$ ]] ||
        ! [[ "$driver_major" =~ ^[0-9]+$ ]]; then
        echo "ERROR: GPU compute capability and driver major must be integers." >&2
        return 2
    fi

    if [ "$gpu_sm" -lt 89 ]; then
        printf '%s\n' 'ampere-cu128|0.19.1|2.10.0+cu128|12.8|cu128'
    elif { [ "$gpu_sm" -eq 90 ] || [ "$gpu_sm" -ge 120 ]; } &&
        [ "$driver_major" -ge 580 ]; then
        # vLLM 0.20.2's published binary is CUDA 13-linked. Hopper and
        # Blackwell hosts on a CUDA 13-capable driver must therefore use the
        # coherent torch/vLLM CUDA 13 stack instead of forcing cu128 and then
        # falling into an unnecessary source rebuild.
        printf '%s\n' 'hopper-blackwell-cu130|0.20.2|2.11.0+cu130|13.0|cu130'
    else
        printf '%s\n' 'ada-hopper-cu128|0.20.2|2.11.0+cu128|12.8|cu128'
    fi
}

# Deterministic diagnostic used by the installer regression suite. It runs no
# setup actions and keeps the architecture policy executable in one place.
if [ "${1:-}" = "--resolve-runtime-selection" ]; then
    if [ "$#" -ne 3 ]; then
        echo "Usage: $0 --resolve-runtime-selection <sm> <driver-major>" >&2
        exit 2
    fi
    resolve_miner_runtime_stack "$2" "$3"
    exit $?
fi

# ── Parse arguments ──────────────────────────────────────────────────────────

SKIP_INSTALL=false
FORCE_VLLM_SOURCE_REBUILD="${VERATHOS_FORCE_VLLM_SOURCE_REBUILD:-0}"
INSTALL_GPTQMODEL="${VERATHOS_INSTALL_GPTQMODEL:-0}"

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

can_run_privileged() {
    [ "$(id -u)" -eq 0 ] || { command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null; }
}

run_privileged() {
    if [ "$(id -u)" -eq 0 ]; then
        "$@"
    elif command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null; then
        sudo "$@"
    else
        return 127
    fi
}

# Detect FUSE / network filesystem on WORKSPACE.  Some RunPod-style pods
# expose /workspace as MooseFS-over-FUSE (mfs#…runpod.net:9421), which is
# fine for big sequential ops (model downloads, HF cache, compile cache)
# but DEADLOCKS pip's parallel build-isolation worker pool on the many
# small concurrent file operations in pip-build-env-*.  Symptom:
# `python -m pip install` hangs with wchan=request_wait_answer (FUSE
# userspace daemon wait), zero network activity.  Workaround: keep the
# big caches on $WORKSPACE (where they belong) but pin TMPDIR to /tmp
# (local overlay) so pip's transient build env stays off the network FS.
# Falls back to $WORKSPACE/tmp on non-FUSE hosts (existing behaviour).
WORKSPACE_FSTYPE=$(stat -f -c %T "$WORKSPACE" 2>/dev/null || echo "")
case "$WORKSPACE_FSTYPE" in
    fuseblk|fuse|fuse.*)
        WORKSPACE_IS_FUSE=1
        ;;
    *)
        # Fallback: check `mount` output for FUSE markers on a parent dir
        # (`stat -f` may report the FUSE backing fs incorrectly on older
        # libc; `mount` output is authoritative).
        if mount 2>/dev/null | awk -v p="$WORKSPACE" '
            $3 == p && $5 ~ /^fuse/ { found = 1 }
            END { exit !found }
        '; then
            WORKSPACE_IS_FUSE=1
        else
            WORKSPACE_IS_FUSE=0
        fi
        ;;
esac
if [ "${WORKSPACE_IS_FUSE:-0}" = "1" ]; then
    # Force TMPDIR to a local fs for the rest of this setup AND in the
    # persisted .env.sh, so future PM2 starts honour it too.  Also redirect
    # the torch.compile / triton compile caches: vLLM's inductor codegen
    # path writes thousands of tiny `.py` / `.cubin` / `.ptx` / `.json`
    # files during CUDA graph capture, and FUSE/MooseFS deadlocks the
    # same way pip's build-isolation worker pool does — server warmup
    # hangs in `request_wait_answer` and the dry_run hits its 30 min
    # ready-timeout without ever finishing.
    if [ -z "$TMPDIR" ]; then
        export TMPDIR=/tmp
    fi
    if [ -z "$TRITON_CACHE_DIR" ]; then
        export TRITON_CACHE_DIR=/tmp/.cache/triton
    fi
    if [ -z "$TORCHINDUCTOR_CACHE_DIR" ]; then
        export TORCHINDUCTOR_CACHE_DIR=/tmp/.cache/torchinductor
    fi
    if [ -z "$XDG_CACHE_HOME" ]; then
        export XDG_CACHE_HOME=/tmp/.cache
    fi
    mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$XDG_CACHE_HOME" 2>/dev/null || true
    echo "  WORKSPACE on FUSE/network FS — TMPDIR + triton/inductor caches pinned to /tmp"
    echo "  (avoids pip build-env deadlock AND torch.compile-on-FUSE warmup hang)"
fi

echo ""
echo "============================================================"
echo "  Verathos Miner Environment Setup"
echo "============================================================"
echo "  Workspace: $WORKSPACE"
echo "  HF cache:  $HF_HOME"
echo "============================================================"

detect_memory_gb() {
    local candidates=()
    local f val

    # cgroup v2 common path inside containers.
    if [ -r /sys/fs/cgroup/memory.max ]; then
        candidates+=("/sys/fs/cgroup/memory.max")
    fi
    # cgroup v1 fallback.
    if [ -r /sys/fs/cgroup/memory/memory.limit_in_bytes ]; then
        candidates+=("/sys/fs/cgroup/memory/memory.limit_in_bytes")
    fi

    for f in "${candidates[@]}"; do
        val=$(cat "$f" 2>/dev/null || true)
        case "$val" in
            ""|"max") continue ;;
        esac
        if [ "$val" -gt 0 ] 2>/dev/null; then
            # Ignore effectively unlimited cgroup values; use host MemTotal
            # below in that case.
            if [ "$val" -lt 9000000000000000000 ] 2>/dev/null; then
                echo $(( val / 1024 / 1024 / 1024 ))
                return 0
            fi
        fi
    done

    awk '/MemTotal/ {print int($2 / 1024 / 1024)}' /proc/meminfo 2>/dev/null || echo 0
}

# ── Early GPU check (fail fast before installing anything) ───────────────────
if ! command -v nvidia-smi &>/dev/null; then
    echo "  ERROR: nvidia-smi not found. NVIDIA GPU required."
    exit 1
fi
# Query the first visible device directly.  With pipefail enabled, piping a
# multi-GPU nvidia-smi result through head can terminate nvidia-smi with
# SIGPIPE and abort an otherwise healthy one-command install.
GPU_NAME=$(nvidia-smi -i 0 --query-gpu=name --format=csv,noheader 2>/dev/null)
GPU_VRAM=$(nvidia-smi -i 0 --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null)
if [ -z "$GPU_NAME" ]; then
    echo "  ERROR: No GPU detected."
    exit 1
fi
if [ "${GPU_VRAM:-0}" -lt 20000 ] 2>/dev/null; then
    echo "  ERROR: GPU has ${GPU_VRAM}MB VRAM. Minimum 24GB required."
    exit 1
fi
GPU_SM=$(nvidia-smi -i 0 --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | tr -d '.')
GPU_DRIVER=$(nvidia-smi -i 0 --query-gpu=driver_version --format=csv,noheader 2>/dev/null)
GPU_DRIVER_MAJOR=$(echo "$GPU_DRIVER" | cut -d. -f1)
RUNTIME_SELECTION=$(resolve_miner_runtime_stack "$GPU_SM" "${GPU_DRIVER_MAJOR:-0}")
IFS='|' read -r VLLM_RUNTIME_STACK VLLM_RUNTIME_VERSION \
    TORCH_RUNTIME_VERSION TORCH_CUDA_VERSION TORCH_CUDA_TAG \
    <<< "$RUNTIME_SELECTION"
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

# ── vLLM version policy & known issues ──────────────────────────────────────
# The default install selects a qualified architecture/runtime pair below:
# Ampere uses vLLM 0.19.1 + torch 2.10/cu128; Ada and older-driver Hopper use
# vLLM 0.20.2 + torch 2.11/cu128; CUDA-13-capable Hopper/Blackwell uses vLLM
# 0.20.2 + torch 2.11/cu130. Do not mix torch and vLLM across those pairs.
#
# KNOWN ISSUE: A small number of sm_89 operators (RTX 4090 / L4 / L40S
# / RTX 6000 Ada) hit `cudaErrorIllegalAddress` during model load
# (vLLM CUDA-graph capture phase).  The vLLM FLA kernel guard
# (`state_idx < 0` in 0.19) does NOT catch the NULL_BLOCK_ID=0 padding,
# so any combination that triggers a zero-padded state index reads
# uninitialised memory.  Hopper (sm_90) masks the corruption.
#
# RECOVERY:
# 1) Try a clean restart first — Triton autotune is non-deterministic, so
#    the next compile may pick different params and avoid the bad codegen.
#    Stop the miner, wipe caches, restart:
#       pm2 stop miner
#       rm -rf /tmp/torchinductor_* ~/.cache/vllm/torch_compile_cache ~/.triton/cache
#       pm2 restart miner
#    (miner.py auto-clears stale caches at startup, but only after the
#     vllm subprocess crash that creates the bad cache entry — manual
#     pre-restart wipe is more reliable.)
#
# 2) If clean restart still crashes, downgrade to vLLM 0.17.1 (uses an
#    older Triton kernel path that doesn't trip this bug):
#       source .venv-vllm/bin/activate
#       pip install --no-cache-dir 'vllm==0.17.1' \
#           'flashinfer-python==0.6.4' 'flashinfer-cubin==0.6.4'
#       pm2 restart miner
#    Trade-off: Qwen 3.6 / Gemma 4 require vLLM >= 0.19, so they will not
#    work on the 0.17.1 fallback.  Qwen 3.5 family works fine.

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

GPU_VRAM_GB=$(( (GPU_VRAM + 512) / 1024 ))
if ! "$PYTHON" scripts/check_capacity_audit_gpu.py --gpu-name "$GPU_NAME" --vram-gb "$GPU_VRAM_GB"; then
    exit 1
fi

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

install_proof_pcs_runtime() {
    local runtime_dir="${VENV_DIR}/lib/verathos"
    local runtime_library="${runtime_dir}/libverathos_pcs_v2.so"
    local python_tag
    local wheel

    python_tag=$(
        "$PYTHON" -c \
            "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')"
    )
    wheel=$(
        find "${REPO_DIR}/dist" -maxdepth 1 -type f \
            -name "zkllm-*-${python_tag}-${python_tag}-*.whl" \
            -print 2>/dev/null | sort | tail -1
    )
    if [ -z "$wheel" ]; then
        # The PCS library is Python-ABI independent. A release wheel for a
        # different supported Python version is therefore a valid source; the
        # ctypes loader still enforces the exact PCS ABI below.
        wheel=$(
            find "${REPO_DIR}/dist" -maxdepth 1 -type f \
                -name "zkllm-*.whl" -print 2>/dev/null | sort | tail -1
        )
    fi
    if [ -z "$wheel" ]; then
        echo "ERROR: bundled zkllm release wheel is missing; cannot install proof PCS."
        return 1
    fi

    mkdir -p "$runtime_dir"
    "$PYTHON" - "$wheel" "$runtime_library" <<'PY'
import os
import pathlib
import sys
import zipfile

wheel = pathlib.Path(sys.argv[1])
destination = pathlib.Path(sys.argv[2])
member = "zkllm/crypto/libverathos_pcs_v2.so"
with zipfile.ZipFile(wheel) as archive:
    try:
        payload = archive.read(member)
    except KeyError as exc:
        raise SystemExit(f"{wheel} does not contain {member}") from exc
temporary = destination.with_suffix(".so.tmp")
temporary.write_bytes(payload)
temporary.chmod(0o755)
os.replace(temporary, destination)
PY

    # This runs before the Torch/vLLM dependency step.  Validate the extracted
    # library directly so a clean Ubuntu image does not import the checkout's
    # torch-dependent zkllm package before Torch has been installed.
    if ! "$PYTHON" - "$runtime_library" <<'PY'
import ctypes
import pathlib
import sys

path = pathlib.Path(sys.argv[1]).resolve()
library = ctypes.CDLL(str(path))
library.verathos_pcs_v2_abi_version.argtypes = []
library.verathos_pcs_v2_abi_version.restype = ctypes.c_uint32
abi_version = int(library.verathos_pcs_v2_abi_version())
if abi_version != 10:
    raise SystemExit(f"proof PCS ABI mismatch: expected 10, got {abi_version}")
print(f"  proof PCS ABI {abi_version}: {path}")
PY
    then
        echo "ERROR: bundled proof PCS library failed its ABI/load check."
        return 1
    fi

    PROOF_PCS_RUNTIME_LIBRARY="$runtime_library"
    export VERATHOS_PCS_V2_LIB="$runtime_library"
}

# vLLM's Qwen3.6 GDN path compiles a small FlashInfer kernel during the first
# production warmup.  CUDA runtime wheels are sufficient for torch/vLLM, but
# they do not provide nvcc.  On the qualified CUDA 13 Hopper/Blackwell lane,
# keep the compiler in the miner venv so a clean cloud image needs no manual
# system-toolkit installation.
configure_runtime_cuda_compiler() {
    if [ "$VLLM_RUNTIME_STACK" != "hopper-blackwell-cu130" ]; then
        return 0
    fi

    local candidate=""
    local compiler_cuda=""
    local root
    for root in "${CUDA_HOME:-}" /usr/local/cuda /usr/local/cuda-*; do
        if [ -n "$root" ] && [ -x "$root/bin/nvcc" ]; then
            compiler_cuda=$(
                "$root/bin/nvcc" --version 2>/dev/null |
                    sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' |
                    head -1
            )
            if [ "$compiler_cuda" = "$TORCH_CUDA_VERSION" ]; then
                candidate="$root"
                break
            fi
        fi
    done

    if [ -z "$candidate" ]; then
        candidate=$($PYTHON - <<'PY'
import pathlib
import site

for site_root in site.getsitepackages():
    cuda_root = pathlib.Path(site_root) / "nvidia" / "cu13"
    if (cuda_root / "bin" / "nvcc").is_file():
        print(cuda_root)
        break
PY
        )
    fi

    if [ -z "$candidate" ]; then
        echo "  Installing the CUDA ${TORCH_CUDA_VERSION} compiler required by production warmup..."
        $PYTHON -m pip install --no-cache-dir \
            "nvidia-cuda-nvcc==${TORCH_CUDA_VERSION}.*" \
            "nvidia-cuda-cccl==${TORCH_CUDA_VERSION}.*" \
            "nvidia-nvvm==${TORCH_CUDA_VERSION}.*" \
            "nvidia-cuda-crt==${TORCH_CUDA_VERSION}.*" 2>&1 | tail -10
        candidate=$($PYTHON - <<'PY'
import pathlib
import site

for site_root in site.getsitepackages():
    cuda_root = pathlib.Path(site_root) / "nvidia" / "cu13"
    if (cuda_root / "bin" / "nvcc").is_file():
        print(cuda_root)
        break
PY
        )
    fi

    if [ -z "$candidate" ] || [ ! -x "$candidate/bin/nvcc" ]; then
        echo "  ERROR: CUDA ${TORCH_CUDA_VERSION} nvcc is required for the qualified H100/Blackwell warmup path."
        return 1
    fi

    compiler_cuda=$(
        "$candidate/bin/nvcc" --version 2>/dev/null |
            sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' |
            head -1
    )
    if [ "$compiler_cuda" != "$TORCH_CUDA_VERSION" ]; then
        echo "  ERROR: CUDA compiler $compiler_cuda does not match torch CUDA $TORCH_CUDA_VERSION."
        return 1
    fi

    export CUDA_HOME="$candidate"
    export PATH="$CUDA_HOME/bin:$PATH"
    if [ -d "$CUDA_HOME/lib" ]; then
        # The venv CUDA packages expose versioned runtime SONAMEs.  Extension
        # linkers request the unversioned development names.
        local library soname
        for library in libcudart libcublas libcublasLt; do
            if [ ! -e "$CUDA_HOME/lib/${library}.so" ]; then
                soname=$(
                    find "$CUDA_HOME/lib" -maxdepth 1 -type f \
                        -name "${library}.so.*" -printf '%f\n' \
                        2>/dev/null | sort -V | head -1
                )
                if [ -n "$soname" ]; then
                    ln -s "$soname" "$CUDA_HOME/lib/${library}.so"
                fi
            fi
        done
        export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
        export LIBRARY_PATH="$CUDA_HOME/lib:${LIBRARY_PATH:-}"
    elif [ -d "$CUDA_HOME/lib64" ]; then
        export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
        export LIBRARY_PATH="$CUDA_HOME/lib64:${LIBRARY_PATH:-}"
    fi
    echo "  CUDA compiler: $($CUDA_HOME/bin/nvcc --version | tail -1)"
}

warm_hopper_gdn_kernel() {
    if [ "$GPU_SM" != "90" ] ||
        [ "$VLLM_RUNTIME_STACK" != "hopper-blackwell-cu130" ]; then
        return 0
    fi

    echo "  Warming the FlashInfer H100 GDN kernel..."
    if ! timeout 600 env MAX_JOBS="${MAX_JOBS:-$(nproc)}" "$PYTHON" - <<'PY'
import torch
from flashinfer.gdn_prefill import get_gdn_prefill_module

module = get_gdn_prefill_module()
assert hasattr(module, "gdn_prefill"), "FlashInfer GDN kernel is incomplete"
torch.cuda.synchronize()
print("  FlashInfer H100 GDN kernel: OK")
PY
    then
        echo "  ERROR: the production H100 GDN warmup could not compile and load."
        return 1
    fi
}

ensure_proof_cuda12_libraries() {
    local missing_packages
    missing_packages=$($PYTHON - <<'PY'
import glob
import os
import site

def has_lib(name: str) -> bool:
    for sp in site.getsitepackages():
        if glob.glob(os.path.join(sp, "nvidia", "*", "lib", name)):
            return True
    for pattern in ("/usr/local/cuda*/targets/x86_64-linux/lib", "/usr/local/cuda*/lib64"):
        for d in glob.glob(pattern):
            if os.path.exists(os.path.join(d, name)):
                return True
    return False

packages = []
if not has_lib("libcudart.so.12"):
    packages.append("nvidia-cuda-runtime-cu12==12.8.90")
if not has_lib("libcublas.so.12"):
    packages.append("nvidia-cublas-cu12==12.8.4.1")
print(" ".join(packages))
PY
    )
    if [ -z "$missing_packages" ]; then
        return 0
    fi

    echo "  Installing CUDA 12 compatibility libraries for proof wheels..."
    # shellcheck disable=SC2086
    $PYTHON -m pip install --no-cache-dir $missing_packages 2>&1 | tail -5
    fix_ld_library_path
}

# ── vLLM CUDA kernel compatibility check ─────────────────────────────────────

check_vllm_cuda_compat() {
    $PYTHON -c "
import torch, subprocess, site, glob, os, sys

try:
    import vllm  # noqa: F401
    import vllm._C  # noqa: F401
except Exception as e:
    print(f'vLLM binary import failed: {e}')
    sys.exit(1)

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

# Find a working cuobjdump once.
cuobjdump = None
for cmd in ['/usr/local/cuda/bin/cuobjdump', 'cuobjdump']:
    try:
        if subprocess.run([cmd, '--help'], capture_output=True, timeout=5).returncode == 0:
            cuobjdump = cmd
            break
    except (FileNotFoundError, subprocess.TimeoutExpired):
        continue

if cuobjdump is not None:
    # vLLM ships a *set* of _*.so files (Marlin, MoE, attention etc.); not
    # every one carries every sm.  Scan ALL of them — if ANY contains the
    # current sm, the wheel is good.  Earlier code only inspected sos[0]
    # which falsely flagged 0.20.x as incompatible on sm_89 even though
    # other .so files in the same wheel did contain sm_89 cubins.
    sm_found = False
    base_sm = (sm // 10) * 10
    base_str = f'sm_{base_sm}'
    base_found = False
    for so in sos:
        try:
            r = subprocess.run([cuobjdump, '--list-elf', so],
                               capture_output=True, text=True, timeout=10)
        except subprocess.TimeoutExpired:
            continue
        if r.returncode != 0:
            continue
        if sm_str in r.stdout:
            sm_found = True
            break
        if base_sm != sm and base_str in r.stdout:
            base_found = True
    if not sm_found:
        if base_found:
            # PTX forward-compat fallback (e.g. sm_86 runs on sm_80 cubins).
            print(f'{sm_str} not in any wheel .so but {base_str} present — using PTX fallback')
        else:
            print(f'{sm_str} not found in any of {len(sos)} vLLM CUDA kernels')
            sys.exit(1)
else:
    # No cuobjdump available — fall back to known wheel arch list.
    known = {80, 86, 89, 90}
    cuda = str(getattr(torch.version, 'cuda', '') or '')
    # CUDA 13 wheels are the supported forward path for Blackwell. Do not
    # reject one solely because this host image lacks cuobjdump; the runtime
    # Marlin smoke below is the authoritative compatibility check.
    if sm >= 120 and cuda.startswith('13.'):
        pass
    elif sm not in known:
        print(f'{sm_str} not in known wheel archs {known}')
        sys.exit(1)

try:
    import vllm._custom_ops as vllm_ops
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        marlin_permute_scales,
    )
    s = torch.ones(1, 128, dtype=torch.float16, device='cuda')
    marlin_permute_scales(s, size_k=128, size_n=128, group_size=128)
    torch.cuda.synchronize()
    # Compressed-tensors INT4 models use the GPTQ Marlin repack op during
    # weight loading. Some binary wheels import and pass the scale-permute
    # smoke but fail here with max_shared_mem errors on Ada/Hopper hosts,
    # so exercise the exact kernel setup before declaring the wheel usable.
    q = torch.zeros((16, 128), dtype=torch.int32, device='cuda')
    vllm_ops.gptq_marlin_repack(
        q,
        perm=torch.empty(0, dtype=torch.int32, device='cuda'),
        size_k=128,
        size_n=128,
        num_bits=4,
        is_a_8bit=False,
    )
    torch.cuda.synchronize()
except Exception as e:
    err = str(e).lower()
    if (
        'ptx' in err
        or 'unsupported' in err
        or 'toolchain' in err
        or 'max_shared_mem' in err
    ):
        print(f'Marlin kernel broken on {sm_str}: {e}')
        sys.exit(1)

# For very new GPUs (sm_120+), older CUDA 12 wheels can carry PTX that imports
# but fails at runtime. CUDA 13 wheels are the intended Blackwell path and pass
# the import + Marlin smoke above, so do not force a source build there.
if sm >= 120:
    cuda = str(getattr(torch.version, 'cuda', '') or '')
    if not cuda.startswith('13.'):
        print(f'{sm_str} on CUDA {cuda or \"unknown\"} requires CUDA 13 wheel path or source rebuild')
        sys.exit(1)

sys.exit(0)
" 2>/dev/null
}

torch_pin_ready() {
    local expected="$1"
    local cuda_prefix="${2:-12.8}"
    $PYTHON - "$expected" "$cuda_prefix" <<'PY' >/dev/null 2>&1
import sys
expected = sys.argv[1]
cuda_prefix = sys.argv[2]
import torch
if torch.__version__ != expected:
    raise SystemExit(1)
if not torch.version.cuda or not torch.version.cuda.startswith(cuda_prefix):
    raise SystemExit(1)
if not torch.cuda.is_available():
    raise SystemExit(1)
torch.zeros(1, device="cuda")
torch.cuda.synchronize()
PY
}

vllm_pin_ready() {
    local expected="$1"
    $PYTHON - "$expected" <<'PY' >/dev/null 2>&1
import importlib.metadata
import sys
expected = sys.argv[1]
if importlib.metadata.version("vllm") != expected:
    raise SystemExit(1)
import vllm  # noqa: F401
PY
}

rebuild_vllm_from_source() {
    # Fail loudly on any error in this function — a silent half-built vLLM
    # would let the outer script exit 0 with a broken install.
    set -e

    local BUILD_PARENT="${WORKSPACE}"
    if [ "${WORKSPACE_IS_FUSE:-0}" = "1" ]; then
        BUILD_PARENT="${TMPDIR:-/tmp}"
    fi
    local BUILD_DIR="${BUILD_PARENT}/vllm-build"
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

    local MEM_GB
    MEM_GB=$(detect_memory_gb)
    local DEFAULT_JOBS=8
    if [ "${MEM_GB:-0}" -gt 0 ] && [ "$MEM_GB" -lt 96 ]; then
        DEFAULT_JOBS=4
    elif [ "${MEM_GB:-0}" -ge 192 ]; then
        DEFAULT_JOBS=16
    fi
    local NJOBS="${VERATHOS_VLLM_BUILD_JOBS:-$DEFAULT_JOBS}"

    echo "  CUDA_HOME=$CUDA_HOME_PATH"
    echo "  TORCH_CUDA_ARCH_LIST=$ARCH_LIST"
    echo "  MAX_JOBS=$NJOBS ($(nproc) CPUs, ${MEM_GB:-unknown}GB RAM available)"

    rm -rf "$BUILD_DIR"
    # Pin to the runtime-selected vLLM tag.  Avoids breaking changes on main
    # branch and keeps the rebuilt binary ABI-compatible with the torch pin.
    local VLLM_TAG="${VLLM_SOURCE_TAG:-v0.20.2}"
    git clone --depth 1 --branch "$VLLM_TAG" https://github.com/vllm-project/vllm.git "$BUILD_DIR"
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
VENV_ON_LOCAL=0
if [ "${WORKSPACE_IS_FUSE:-0}" = "1" ]; then
    VENV_ON_LOCAL=1
    VENV_LINK="$VENV_DIR"
    LOCAL_INSTALL_ROOT="${VERATHOS_LOCAL_INSTALL_ROOT:-/opt/verathos-local}"
    if ! mkdir -p "${LOCAL_INSTALL_ROOT}/venvs" 2>/dev/null; then
        LOCAL_INSTALL_ROOT="${TMPDIR:-/tmp}/verathos-local"
        mkdir -p "${LOCAL_INSTALL_ROOT}/venvs"
    fi
    REPO_HASH=$(printf '%s' "$REPO_DIR" | sha256sum | awk '{print substr($1, 1, 12)}')
    VENV_REAL="${LOCAL_INSTALL_ROOT}/venvs/$(basename "$REPO_DIR")-${REPO_HASH}-venv-vllm"
    if [ -e "$VENV_LINK" ] && [ ! -L "$VENV_LINK" ]; then
        echo "  Existing FUSE-backed venv directory found — moving venv to local disk"
        rm -rf "$VENV_LINK"
    fi
    mkdir -p "$VENV_REAL"
    ln -sfnT "$VENV_REAL" "$VENV_LINK"
    VENV_DIR="$VENV_REAL"
    echo "  WORKSPACE on FUSE/network FS — venv stored on local disk: $VENV_REAL"
fi

# ── System dependencies ────────────────────────────────────────────────────
# vLLM/Triton may compile small runtime launchers even when all Verathos native
# artifacts come from wheels.  A minimal cloud image therefore still needs a C
# compiler on the ordinary serving path.
if ! command -v cc &>/dev/null; then
    SYSTEM_PY_VER=$($PYTHON -c \
        "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    if can_run_privileged; then
        if command -v apt-get &>/dev/null; then
            run_privileged apt-get update -qq
            run_privileged apt-get install -y -qq \
                build-essential "python${SYSTEM_PY_VER}-dev" >/dev/null
        elif command -v dnf &>/dev/null; then
            run_privileged dnf install -y gcc gcc-c++ python3-devel >/dev/null
        fi
    fi
    if ! command -v cc &>/dev/null; then
        echo "  ERROR: a C compiler is required by the vLLM/Triton runtime."
        echo "  Install build-essential (Debian/Ubuntu) or gcc (RHEL/Fedora), then rerun setup."
        exit 1
    fi
fi

PYTHON_INCLUDE=$($PYTHON -c \
    "import sysconfig; print(sysconfig.get_path('include') or '')")
if [ -z "$PYTHON_INCLUDE" ] || [ ! -f "$PYTHON_INCLUDE/Python.h" ]; then
    SYSTEM_PY_VER=$($PYTHON -c \
        "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    if can_run_privileged && command -v apt-get &>/dev/null; then
        run_privileged apt-get update -qq
        run_privileged apt-get install -y -qq \
            "python${SYSTEM_PY_VER}-dev" >/dev/null
    elif can_run_privileged && command -v dnf &>/dev/null; then
        run_privileged dnf install -y python3-devel >/dev/null
    fi
    PYTHON_INCLUDE=$($PYTHON -c \
        "import sysconfig; print(sysconfig.get_path('include') or '')")
    if [ -z "$PYTHON_INCLUDE" ] || [ ! -f "$PYTHON_INCLUDE/Python.h" ]; then
        echo "  ERROR: Python development headers are required by the vLLM/Triton runtime."
        echo "  Install python${SYSTEM_PY_VER}-dev (Debian/Ubuntu) or python3-devel (RHEL/Fedora), then rerun setup."
        exit 1
    fi
fi

# ninja-build: required by FlashInfer JIT compilation on Hopper/Blackwell (sm_90+)
if ! command -v ninja &>/dev/null; then
    if can_run_privileged; then
        if command -v apt-get &>/dev/null; then
            run_privileged apt-get install -y -qq ninja-build >/dev/null 2>&1 || true
        elif command -v dnf &>/dev/null; then
            run_privileged dnf install -y ninja-build >/dev/null 2>&1 || true
        fi
    fi
    # If system install failed, pip fallback happens after venv activation below
fi

ensure_node_runtime() {
    local node_major=0
    if command -v node &>/dev/null; then
        node_major=$(node -p 'Number(process.versions.node.split(".")[0])' 2>/dev/null || echo 0)
    fi
    if [ "$node_major" -ge 18 ] 2>/dev/null && command -v npm &>/dev/null; then
        return 0
    fi

    local SUDO=""
    if [ "$(id -u)" -ne 0 ]; then
        if sudo -n true 2>/dev/null; then
            SUDO="sudo"
        else
            echo "  ERROR: Node.js >= 18 is required and sudo is unavailable."
            echo "  Install Node.js >= 18, then rerun this setup."
            exit 1
        fi
    fi

    echo "  Installing Node.js 20 runtime..."
    if command -v apt-get &>/dev/null; then
        $SUDO apt-get update -qq
        $SUDO apt-get install -y -qq ca-certificates curl gnupg 2>&1 | tail -3
        # Ubuntu 22.04 may have Node 12 plus libnode-dev preinstalled.
        # NodeSource's Node 20 package owns the same headers, so remove the
        # obsolete distro packages first instead of leaving dpkg to fail on
        # an overwrite conflict.
        if [ "$node_major" -gt 0 ] 2>/dev/null; then
            $SUDO apt-get remove -y -qq nodejs npm libnode-dev 2>&1 | tail -5
        fi
        local key_tmp
        key_tmp=$(mktemp)
        if ! curl -fsSL \
            https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key \
            -o "$key_tmp"; then
            rm -f "$key_tmp"
            echo "  ERROR: could not download the NodeSource signing key."
            exit 1
        fi
        $SUDO mkdir -p /etc/apt/keyrings
        $SUDO gpg --batch --yes --dearmor \
            -o /etc/apt/keyrings/nodesource.gpg "$key_tmp"
        rm -f "$key_tmp"
        printf '%s\n' \
            'deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_20.x nodistro main' \
            | $SUDO tee /etc/apt/sources.list.d/nodesource.list >/dev/null
        $SUDO apt-get update -qq
        $SUDO apt-get install -y -qq nodejs 2>&1 | tail -5
    elif command -v dnf &>/dev/null; then
        $SUDO dnf module enable -y nodejs:20 2>&1 | tail -3 || true
        $SUDO dnf install -y nodejs npm 2>&1 | tail -5
    else
        echo "  ERROR: no supported package manager can install Node.js >= 18."
        exit 1
    fi

    hash -r
    node_major=0
    if command -v node &>/dev/null; then
        node_major=$(node -p 'Number(process.versions.node.split(".")[0])' 2>/dev/null || echo 0)
    fi
    if [ "$node_major" -lt 18 ] 2>/dev/null || ! command -v npm &>/dev/null; then
        echo "  ERROR: setup did not produce a working Node.js >= 18 runtime."
        exit 1
    fi
    echo "  Node.js ready: $(node --version)"
}

install_pm2_if_missing() {
    ensure_node_runtime
    if command -v pm2 &>/dev/null; then
        return 0
    fi

    echo "  Installing PM2 process manager..."
    local SUDO=""
    if [ "$(id -u)" -ne 0 ]; then
        SUDO="sudo"
    fi
    $SUDO npm install -g pm2 >/dev/null
    if ! command -v pm2 &>/dev/null; then
        echo "  ERROR: PM2 installation completed but pm2 is not on PATH."
        exit 1
    fi
    echo "  PM2 installed: $(pm2 -v)"
}

install_pm2_if_missing

NEED_CREATE_VENV=0
if [ -d "$VENV_DIR" ] && { [ ! -f "$VENV_DIR/bin/activate" ] || [ ! -x "$VENV_DIR/bin/python" ]; }; then
    echo ""
    echo "  Existing venv is incomplete — recreating: $VENV_DIR"
    if [ "$VENV_ON_LOCAL" = "1" ]; then
        rm -rf "$VENV_REAL"
        mkdir -p "$VENV_REAL"
        ln -sfnT "$VENV_REAL" "$VENV_LINK"
    else
        rm -rf "$VENV_DIR"
    fi
    NEED_CREATE_VENV=1
fi

if [ ! -d "$VENV_DIR" ]; then
    NEED_CREATE_VENV=1
fi

if [ "$NEED_CREATE_VENV" = "1" ]; then
    echo ""
    echo "  Creating venv with --system-site-packages..."
    if ! $PYTHON -m venv "$VENV_DIR" --system-site-packages 2>/dev/null; then
        echo "  python3-venv not installed — installing..."
        PY_VER=$($PYTHON -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        if command -v apt-get &>/dev/null; then
            run_privileged apt-get update -qq && run_privileged apt-get install -y -qq "python${PY_VER}-venv" 2>&1 | tail -3
        elif command -v dnf &>/dev/null; then
            run_privileged dnf install -y "python${PY_VER}-venv" 2>&1 | tail -3
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
    echo "Step 1/6: Installing zkllm wheel..."
    if ls "$REPO_DIR/dist"/zkllm-*.whl &>/dev/null; then
        # Use --find-links so pip auto-selects the wheel matching this Python version
        $PYTHON -m pip install --no-cache-dir --force-reinstall --find-links "$REPO_DIR/dist" zkllm 2>&1 | tail -5
    else
        echo "  ERROR: No zkllm wheels found in dist/."
        echo "  The dist/ directory should contain pre-built zkllm wheels."
        exit 1
    fi
    if ! install_proof_pcs_runtime; then
        exit 1
    fi

    echo ""
    echo "Step 2/6: Installing Python dependencies..."
    # Upgrade pip + setuptools first — Ubuntu 22.04's system setuptools (59.x) is
    # too old for PEP 660 editable installs (needs >=64).
    $PYTHON -m pip install --no-cache-dir --upgrade pip setuptools 2>&1 | tail -5
    VLLM_RUNTIME_DEPS=(
        "bitsandbytes>=0.42"
        "streamlit>=1.30"
        "ninja>=1.11"
        "kernels>=0.12,<0.13"
    )

    # Install the arch-specific vLLM + torch pair before installing Verathos.
    # This avoids resolving a generic torch/vLLM stack that is replaced below.
    # Per-arch (vLLM, torch) pinning.  Each (sm, vLLM, torch) triple is a
    # known-good combination empirically validated end-to-end (boot +
    # proof verify) on real hardware.  Mixing versions across the gate
    # produces ABI errors (`undefined symbol _ZN3c10...`) or runtime
    # crashes (`max_shared_mem > 0`, `increment_version expects tensor`).
    #
    # Why per-arch:
    #  - vLLM 0.20.2 PyPI wheels are CUDA 13-linked. On Blackwell with a CUDA
    #    13-capable driver/image, use the coherent default cu130 torch/vLLM
    #    stack. Forcing cu128 there is unnecessary and can trigger a slow
    #    source rebuild path.
    #  - vLLM 0.19.1 was built against torch 2.10 (ABI in METADATA).  Pairing
    #    it with torch 2.11 produces undefined-symbol load errors.
    #  - vLLM 0.20.2 + Ampere (sm<89) Marlin FP8 path crashes with
    #    `max_shared_mem == 0` during process_weights_after_loading. vLLM
    #    0.19.1 + Marlin FP8 + Qwen3.6 is known good (verified A100 cross-
    #    backend match to Blackwell canonical, session 2026-05-13).
    #  - vLLM 0.19.1 + Hopper/Ada/Blackwell + FP8 + verallm capture crashes
    #    on first inference (`increment_version expects each element to be
    #    a tensor`); these archs need 0.20.2.
    #
    # Net selection:
    #  - sm < 89 (Ampere): vLLM 0.19.1 + torch 2.10.0 + cu128
    #  - sm 90 or sm >= 120 with driver >= 580: vLLM 0.20.2 + torch
    #    2.11.0 + cu130
    #  - sm >= 89 otherwise: vLLM 0.20.2 + torch 2.11.0 + cu128
    if [ "$VLLM_RUNTIME_STACK" = "ampere-cu128" ]; then
        VLLM_SOURCE_TAG="v0.19.1"
        echo "  Ampere GPU (sm_${GPU_SM}): installing vLLM 0.19.1 + torch 2.10.0+cu128..."
        TORCH_CONSTRAINT="$(mktemp)"
        {
            echo "torch==2.10.0+cu128"
            echo "torchvision==0.25.0+cu128"
            echo "torchaudio==2.10.0+cu128"
        } > "$TORCH_CONSTRAINT"
        $PYTHON -m pip install --no-cache-dir \
            --extra-index-url https://download.pytorch.org/whl/cu128 \
            -c "$TORCH_CONSTRAINT" \
            "vllm==0.19.1" "${VLLM_RUNTIME_DEPS[@]}" 2>&1 | tail -20
        rm -f "$TORCH_CONSTRAINT"
        echo "  Verifying Ampere torch/vLLM pins..."
        if torch_pin_ready "2.10.0+cu128" "12.8"; then
            echo "  torch 2.10.0+cu128 already installed"
        else
            $PYTHON -m pip install --no-cache-dir --force-reinstall --no-deps \
                "torch==2.10.0+cu128" "torchvision==0.25.0+cu128" "torchaudio==2.10.0+cu128" \
                --index-url https://download.pytorch.org/whl/cu128 2>&1 | tail -5
        fi
        echo "  Ampere GPU: pinning vLLM to 0.19.1 (Marlin FP8 known-good on this arch)..."
        if vllm_pin_ready "0.19.1"; then
            echo "  vLLM 0.19.1 already installed"
        else
            $PYTHON -m pip install --no-cache-dir --force-reinstall --no-deps 'vllm==0.19.1' 2>&1 | tail -5
        fi
    elif [ "$VLLM_RUNTIME_STACK" = "hopper-blackwell-cu130" ]; then
        VLLM_SOURCE_TAG="v0.20.2"
        echo "  Hopper/Blackwell GPU (sm_${GPU_SM}) on driver ${GPU_DRIVER}: installing vLLM 0.20.2 + torch 2.11.0+cu130..."
        $PYTHON -m pip install --no-cache-dir \
            "vllm==0.20.2" "${VLLM_RUNTIME_DEPS[@]}" 2>&1 | tail -20
        echo "  Verifying Hopper/Blackwell torch/vLLM pins..."
        if torch_pin_ready "2.11.0+cu130" "13."; then
            echo "  torch 2.11.0+cu130 already installed"
        else
            $PYTHON -m pip install --no-cache-dir --force-reinstall \
                "torch==2.11.0" "torchvision==0.26.0" "torchaudio==2.11.0" 2>&1 | tail -10
            if ! torch_pin_ready "2.11.0+cu130" "13."; then
                echo "  ERROR: expected torch 2.11.0+cu130 on Hopper/Blackwell CUDA 13, but verification failed."
                echo "  Use a CUDA 13-capable image/driver or retry on a newer host."
                exit 1
            fi
        fi
        if vllm_pin_ready "0.20.2"; then
            echo "  vLLM 0.20.2 already installed"
        else
            $PYTHON -m pip install --no-cache-dir --force-reinstall --no-deps 'vllm==0.20.2' 2>&1 | tail -5
        fi
    else
        VLLM_SOURCE_TAG="v0.20.2"
        echo "  Ada/Hopper/Blackwell (sm_${GPU_SM:-89+}): installing vLLM 0.20.2 + torch 2.11.0+cu128..."
        TORCH_CONSTRAINT="$(mktemp)"
        {
            echo "torch==2.11.0+cu128"
            echo "torchvision==0.26.0+cu128"
            echo "torchaudio==2.11.0+cu128"
        } > "$TORCH_CONSTRAINT"
        $PYTHON -m pip install --no-cache-dir \
            --extra-index-url https://download.pytorch.org/whl/cu128 \
            -c "$TORCH_CONSTRAINT" \
            "vllm==0.20.2" "${VLLM_RUNTIME_DEPS[@]}" 2>&1 | tail -20
        rm -f "$TORCH_CONSTRAINT"
        echo "  Verifying Ada/Hopper/Blackwell torch/vLLM pins..."
        if torch_pin_ready "2.11.0+cu128" "12.8"; then
            echo "  torch 2.11.0+cu128 already installed"
        else
            $PYTHON -m pip install --no-cache-dir --force-reinstall --no-deps \
                "torch==2.11.0+cu128" "torchvision==0.26.0+cu128" "torchaudio==2.11.0+cu128" \
                --index-url https://download.pytorch.org/whl/cu128 2>&1 | tail -5
        fi
        if vllm_pin_ready "0.20.2"; then
            echo "  vLLM 0.20.2 already installed"
        else
            $PYTHON -m pip install --no-cache-dir --force-reinstall --no-deps 'vllm==0.20.2' 2>&1 | tail -5
        fi
        if [ "${GPU_DRIVER_MAJOR:-0}" -lt 580 ] 2>/dev/null; then
            echo "  Driver ${GPU_DRIVER} is < 580: using cu128 torch and validating the vLLM wheel before any rebuild."
        fi
    fi

    if ! configure_runtime_cuda_compiler; then
        exit 1
    fi
    if ! warm_hopper_gdn_kernel; then
        exit 1
    fi

    echo "  Installing Verathos API deps..."
    $PYTHON -m pip install --no-cache-dir -e ".[api,hashing]" 2>&1 | tail -20

    echo "  Installing Bittensor + neuron deps..."
    $PYTHON -m pip install --no-cache-dir -e ".[neurons]" 2>&1 | tail -20
    fix_ld_library_path

    $PYTHON -c 'import neurons; import bittensor as bt; assert bt.__version__ == "10.3.2"'

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

    ensure_proof_cuda12_libraries

    echo ""
    echo "  Checking vLLM CUDA kernel compatibility..."
    if [ "${FORCE_VLLM_SOURCE_REBUILD:-0}" = "1" ]; then
        echo "  vLLM source rebuild forced for this GPU/driver combination."
        if ! rebuild_vllm_from_source; then
            echo ""
            echo "  ERROR: vLLM source rebuild failed — cannot continue."
            echo "  The miner will not start without working CUDA kernels."
            exit 1
        fi
        fix_ld_library_path
        if ! check_vllm_cuda_compat; then
            echo ""
            echo "  ERROR: rebuilt vLLM still failed CUDA compatibility check."
            echo "  The miner will not start without a working vLLM binary."
            exit 1
        fi
    elif ! check_vllm_cuda_compat; then
        echo "  vLLM wheel's CUDA kernels don't support this GPU — rebuilding from source..."
        echo "  (This compiles Marlin/quantization kernels for your GPU. May take 10-20 min.)"
        if ! rebuild_vllm_from_source; then
            echo ""
            echo "  ERROR: vLLM source rebuild failed — cannot continue."
            echo "  The miner will not start without working CUDA kernels."
            exit 1
        fi
        fix_ld_library_path
        if ! check_vllm_cuda_compat; then
            echo ""
            echo "  ERROR: rebuilt vLLM still failed CUDA compatibility check."
            echo "  The miner will not start without a working vLLM binary."
            exit 1
        fi
    else
        echo "  vLLM CUDA kernels: compatible"
    fi

    if ! $PYTHON - <<'PY'
from zkllm.crypto.pcs_v2 import (
    ABI_VERSION,
    fold_u31_linear_coefficients,
    native_library_path,
    prove_i64_linear_combination_linear,
)

assert ABI_VERSION == 10, f"expected proof PCS ABI 10, got {ABI_VERSION}"
assert callable(fold_u31_linear_coefficients)
assert callable(prove_i64_linear_combination_linear)
native_library_path()
PY
    then
        echo "  ERROR: zkllm wheel is stale or lacks the proof PCS ABI required by this release."
        echo "  Rebuild the release wheels with scripts/build_zkllm_wheel.sh."
        exit 1
    fi
    echo "  Proof PCS ABI 10: OK"

    echo ""
    echo "Step 3/6: GPTQ quantization support..."
    if [ "$INSTALL_GPTQMODEL" = "1" ]; then
        echo "  Installing gptqmodel (may compile CUDA kernels from source)."
        # gptqmodel needs setuptools>=77 (bittensor downgrades to ~70).
        # After install, pin back transformers/protobuf/huggingface_hub to
        # versions compatible with vLLM — gptqmodel works fine at runtime
        # with older versions despite its pip constraints.
        $PYTHON -m pip install --no-cache-dir "setuptools>=77,<81" 2>&1 | tail -3
        $PYTHON -m pip install --no-build-isolation --no-cache-dir "gptqmodel>=0.9,<6.0" 2>&1 | tail -10 || {
            echo "  WARNING: gptqmodel install failed — GPTQ models will not be available."
            echo "  (Non-fatal; AWQ/fp16/fp8 models still work. Retry manually if needed:"
            echo "   VERATHOS_INSTALL_GPTQMODEL=1 bash scripts/public_overlay/setup_miner.sh)"
        }
        # Restore deps that gptqmodel upgraded beyond vLLM's constraints.
        # gptqmodel works fine at runtime with these pinned versions.
        $PYTHON -m pip install --no-cache-dir \
            "transformers>=4.56,<5" "protobuf>=5.0,<7" "huggingface_hub>=0.28,<1.0" \
            2>&1 | tail -3
    else
        echo "  Skipping optional gptqmodel install."
        echo "  Set VERATHOS_INSTALL_GPTQMODEL=1 before setup only if serving GPTQ models."
    fi

    echo ""
    echo "Step 4/6: Verifying torch + CUDA..."
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
    echo "Step 5/6: Verifying proof CUDA extensions..."
    fix_ld_library_path
    # zkllm is installed as a wheel — no build needed.
    # The wheel auto-detects the correct torch version at import time.
    #
    # Defensive cleanup: if any zkllm_native .so sits in the *source* tree
    # (e.g. left over from a previous source-tree sync, or from an
    # aborted JIT build, or from a different Python version's build), it
    # will shadow the correctly-installed wheel in site-packages because
    # Python's implicit `sys.path[0]=""` puts the source tree first when
    # the import runs from `${REPO_DIR}`.  Any cpython-ABI mismatch
    # (e.g. .so built for py3.12 but loaded by py3.11) crashes the
    # verify with "Python version mismatch" even though the wheel install
    # is fine.  Wipe ALL source-tree .so AND .so.torch* alternates here —
    # the wheel install in site-packages is the authoritative source.
    # Public deploys won't have these (gitignored + excluded by
    # sync_to_public.sh), but private-repo rsyncs can carry them along.
    if [ -d "${REPO_DIR}/zkllm/cuda" ]; then
        find "${REPO_DIR}/zkllm/cuda" -maxdepth 1 \( \
            -name "zkllm_native.cpython-*.so" -o \
            -name "zkllm_native.cpython-*.so.torch*" \
        \) -delete 2>/dev/null || true
    fi
    TORCH_CUDA_RUNTIME=$($PYTHON -c '
import torch
cuda = str(torch.version.cuda or "")
if cuda.startswith("12.8"):
    print("cu128")
elif cuda.startswith("13.0"):
    print("cu130")
else:
    raise SystemExit(f"unsupported torch CUDA runtime: {cuda or None}")
') || {
        echo "  ERROR: unsupported torch CUDA runtime for proof-v3."
        exit 1
    }
    PYTHON_ABI=$($PYTHON -c \
        'import sys; print(f"cp{sys.version_info.major}{sys.version_info.minor}")')
    PROOF_V3_CUDA_WHEELS=(
        "$REPO_DIR"/dist/verathos_proof_v3_cuda-*+"$TORCH_CUDA_RUNTIME"-"$PYTHON_ABI"-"$PYTHON_ABI"-*.whl
    )
    if [ "${#PROOF_V3_CUDA_WHEELS[@]}" -ne 1 ] ||
        [ ! -f "${PROOF_V3_CUDA_WHEELS[0]}" ]; then
        echo "  ERROR: expected exactly one proof-v3 CUDA runtime wheel for"
        echo "    runtime=$TORCH_CUDA_RUNTIME python=$PYTHON_ABI"
        echo "  Available artifacts:"
        find "$REPO_DIR/dist" -maxdepth 1 -type f \
            -name 'verathos_proof_v3_cuda-*.whl' -printf '    %f\n' \
            2>/dev/null | sort
        exit 1
    fi
    echo "  Installing proof-v3 CUDA runtime: $(basename "${PROOF_V3_CUDA_WHEELS[0]}")"
    $PYTHON -m pip install --no-cache-dir --force-reinstall --no-deps \
        "${PROOF_V3_CUDA_WHEELS[0]}" 2>&1 | tail -5
    if ! $PYTHON -c "
import torch
from verathos_proof_v3_cuda import load_fused_kernels, load_tree_kernels
fold = load_fused_kernels()
tree = load_tree_kernels()
for name in ('round_partials', 'lerp_fold', 'product_round_partials', 'fs_round_v2'):
    assert hasattr(fold, name), f'Missing proof-v3 fold kernel: {name}'
for name in ('leaf_hash_w1', 'leaf_hash_wn_base', 'node_hash', 'node_hash_base'):
    assert hasattr(tree, name), f'Missing proof-v3 tree kernel: {name}'
folded = fold.lerp_fold(
    torch.tensor([1, 2], dtype=torch.int64, device='cuda'), 0
)
assert folded.cpu().tolist() == [1]
leaf = tree.leaf_hash_wn_base(
    torch.zeros(90, dtype=torch.uint8, device='cuda'),
    torch.tensor([1], dtype=torch.int64, device='cuda'),
    0,
    1,
)
assert tuple(leaf.shape) == (32,)
torch.cuda.synchronize()
print('  Proof-v3 CUDA runtime: OK')
" 2>/dev/null; then
        echo "ERROR: precompiled proof-v3 CUDA runtime is unavailable or incompatible."
        echo "  Reinstall the matching wheel from dist/verathos_proof_v3_cuda-*.whl."
        exit 1
    fi
    if $PYTHON -c "
from zkllm.cuda import zkllm_native, HAS_CUDA
from zkllm.crypto import pcs_v2
assert zkllm_native is not None, 'zkllm_native not loaded — is the zkllm wheel installed?'
assert HAS_CUDA, 'Extension loaded but HAS_CUDA=False — CUDA not available in zkllm'
assert hasattr(zkllm_native, 'cuda_blake3_merkle_leaves'), 'Missing cuda_blake3_merkle_leaves kernel'
assert hasattr(zkllm_native, 'cuda_blake3_activation_row_roots'), 'Missing proof-v3 activation row reducer'
assert hasattr(zkllm_native, 'cuda_blake3_runtime_row_roots_retain_into'), 'Missing proof-v3 retained-runtime activation reducer'
assert hasattr(zkllm_native, 'cuda_blake3_runtime_staged_row_roots_into'), 'Missing proof-v3 staged row reducer'
assert hasattr(zkllm_native, 'cuda_blake3_activation_update_peaks'), 'Missing proof-v3 activation frontier reducer'
assert hasattr(zkllm_native, 'cuda_blake3_activation_update_peaks_heterogeneous'), 'Missing proof-v3 heterogeneous frontier reducer'
assert hasattr(zkllm_native, 'cuda_blake3_runtime_row_roots_into'), 'Missing proof-v3 graph row reducer'
assert hasattr(zkllm_native, 'cuda_blake3_prehashed_update_peaks'), 'Missing proof-v3 prehashed frontier reducer'
assert hasattr(zkllm_native, 'cuda_blake3_prehashed_update_peaks_ranges'), 'Missing proof-v3 sparse range reducer'
for name in (
    'combine_registered_catalog_u31_batch',
    'prove_linear',
    'verify_linear',
):
    assert hasattr(pcs_v2, name), f'Missing proof-v3 PCS API: {name}'
print(f'  HAS_CUDA: {HAS_CUDA}')
print(f'  Kernels: blake3_merkle, proof_v3_anchor, sumcheck, field_ops, linear_pcs')
" 2>/dev/null; then
        echo "  CUDA extension: OK"
    else
        echo ""
        echo "WARNING: zkllm CUDA extension not available or missing GPU kernels."
        echo ""
        echo "  Ensure the zkllm wheel is installed:"
        echo "    pip install --force-reinstall zkllm-*.whl"
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
    echo "Step 6/6: Verifying hot-capacity audit workspace extension..."
    HOT_CAPACITY_READY=0
    CAPACITY_AUDIT_REQUESTED="${VERATHOS_SETUP_CAPACITY_AUDIT:-${VERATHOS_CAPACITY_AUDIT_ENABLED:-1}}"
    CAPACITY_AUDIT_MODE="${VERATHOS_SETUP_CAPACITY_AUDIT_MODE:-${VERATHOS_CAPACITY_AUDIT_MODE:-observe}}"
    CAPACITY_AUDIT_REQUIRES_WORKSPACE="${VERATHOS_REQUIRE_CAPACITY_AUDIT_WORKSPACE:-}"
    if [ -z "$CAPACITY_AUDIT_REQUIRES_WORKSPACE" ]; then
        case "$(printf '%s' "$CAPACITY_AUDIT_REQUESTED" | tr '[:upper:]' '[:lower:]')" in
            1|true|yes|on|score_gate|soft_gate|enforce)
                if [ "$(printf '%s' "$CAPACITY_AUDIT_MODE" | tr '[:upper:]' '[:lower:]')" != "observe" ]; then
                    CAPACITY_AUDIT_REQUIRES_WORKSPACE=1
                fi
                ;;
        esac
    fi
    if ls "$REPO_DIR/dist"/hot_capacity_workspace_cuda-*.whl &>/dev/null; then
        # Install after torch/vLLM pinning so the import smoke catches ABI issues
        # against the final runtime torch version.
        $PYTHON -m pip install --no-cache-dir --force-reinstall --find-links "$REPO_DIR/dist" hot_capacity_workspace_cuda 2>&1 | tail -5
    else
        echo "  WARNING: No hot-capacity workspace wheel found in dist/."
    fi
    if $PYTHON -c "
import hot_capacity_workspace_cuda
from hot_capacity_workspace.bench_combined import main
" 2>/dev/null; then
        HOT_CAPACITY_READY=1
        echo "  hot_capacity_workspace_cuda wheel: OK"
    else
        echo ""
        echo "WARNING: hot-capacity audit wheel is not importable."
        echo "  Capacity audits require the pre-built hot_capacity_workspace_cuda wheel."
        echo "  Rebuild private release wheels with:"
        echo "    bash scripts/build_hot_capacity_workspace_wheel.sh"
    fi
    if [ "$HOT_CAPACITY_READY" = "1" ]; then
        HOT_CAPACITY_SMOKE_DIR="${TMPDIR:-/tmp}/verathos-hot-capacity-smoke-$$"
        if timeout 60 "$PYTHON" -c "from hot_capacity_workspace.bench_combined import main; main()" \
            --capacity-passes 1 \
            --capacity-rounds 1 \
            --capacity-warmup-passes 0 \
            --capacity-tail-passes 0 \
            --fp64-passes 0 \
            --spot-checks 1 \
            --b-proof-hex 0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef \
            --out-dir "$HOT_CAPACITY_SMOKE_DIR" >/dev/null 2>&1; then
            rm -rf "$HOT_CAPACITY_SMOKE_DIR" 2>/dev/null || true
            echo "  Hot-capacity audit smoke: OK"
        else
            echo ""
            echo "WARNING: hot-capacity audit smoke failed."
            echo "  The miner can start in observe mode, but enforced capacity audits will fail."
            HOT_CAPACITY_READY=0
        fi
    fi
    if [ "$HOT_CAPACITY_READY" != "1" ]; then
        if [ "${CAPACITY_AUDIT_REQUIRES_WORKSPACE:-}" = "1" ]; then
            echo "ERROR: enforced hot-capacity audit mode requires a working hot_capacity_workspace_cuda wheel."
            exit 1
        fi
        echo "  Continuing because enforced hot-capacity audit mode is not enabled."
    fi

    echo ""
    echo "Installation complete."
else
    echo "  Skipping install (--skip-install)"
fi

# ── Persist environment for future SSH sessions ──────────────────────────────

# Pick runtime cache defaults based on the FUSE detection done above.
if [ "${WORKSPACE_IS_FUSE:-0}" = "1" ]; then
    TMPDIR_DEFAULT="${TMPDIR:-/tmp}"
    TRITON_CACHE_DEFAULT="${TRITON_CACHE_DIR:-/tmp/.cache/triton}"
    TORCHINDUCTOR_CACHE_DEFAULT="${TORCHINDUCTOR_CACHE_DIR:-/tmp/.cache/torchinductor}"
    XDG_CACHE_DEFAULT="${XDG_CACHE_HOME:-/tmp/.cache}"
    VERATHOS_MODEL_ROOT_CACHE_DEFAULT="${VERATHOS_MODEL_ROOT_CACHE_DIR:-${LOCAL_INSTALL_ROOT:-/tmp/verathos-local}/cache/model_root}"
    VERATHOS_MERKLE_TREE_CACHE_DEFAULT="${VERATHOS_MERKLE_TREE_CACHE_DIR:-${LOCAL_INSTALL_ROOT:-/tmp/verathos-local}/cache/merkle_tree}"
else
    TMPDIR_DEFAULT="${TMPDIR:-${WORKSPACE}/tmp}"
    TRITON_CACHE_DEFAULT="${TRITON_CACHE_DIR:-${WORKSPACE}/.cache/triton}"
    TORCHINDUCTOR_CACHE_DEFAULT="${TORCHINDUCTOR_CACHE_DIR:-${WORKSPACE}/.cache/torchinductor}"
    XDG_CACHE_DEFAULT="${XDG_CACHE_HOME:-${WORKSPACE}/.cache}"
    VERATHOS_MODEL_ROOT_CACHE_DEFAULT="${VERATHOS_MODEL_ROOT_CACHE_DIR:-${REPO_DIR}/.model_root_cache}"
    VERATHOS_MERKLE_TREE_CACHE_DEFAULT="${VERATHOS_MERKLE_TREE_CACHE_DIR:-${REPO_DIR}/.merkle_tree_cache}"
fi

ENV_FILE="${REPO_DIR}/.env.sh"
cat > "$ENV_FILE" <<ENVEOF
# Auto-generated by setup_miner.sh — source from .bashrc for persistent env
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}"
export LIBRARY_PATH="${LIBRARY_PATH:-}"
export CUDA_HOME="${CUDA_HOME:-}"
export HF_HOME="${HF_HOME}"
export PATH="${REPO_DIR}/.venv-vllm/bin${CUDA_HOME:+:${CUDA_HOME}/bin}:\${PATH}"
export VERATHOS_PCS_V2_LIB="${PROOF_PCS_RUNTIME_LIBRARY:-${VERATHOS_PCS_V2_LIB:-}}"
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export TORCHINDUCTOR_COMPILE_THREADS="\${TORCHINDUCTOR_COMPILE_THREADS:-\${VERATHOS_TORCHINDUCTOR_COMPILE_THREADS:-4}}"
# Runtime cache paths are persisted from setup-time detection.  On normal
# local filesystems they can live under the workspace.  On FUSE / network
# workspaces, setup pins TMPDIR plus Triton/Inductor/XDG caches to /tmp to
# avoid warmup stalls while vLLM compiles many small kernels.
export TMPDIR="\${TMPDIR:-${TMPDIR_DEFAULT}}"
export TRITON_CACHE_DIR="\${TRITON_CACHE_DIR:-${TRITON_CACHE_DEFAULT}}"
export TORCHINDUCTOR_CACHE_DIR="\${TORCHINDUCTOR_CACHE_DIR:-${TORCHINDUCTOR_CACHE_DEFAULT}}"
export XDG_CACHE_HOME="\${XDG_CACHE_HOME:-${XDG_CACHE_DEFAULT}}"
export VERATHOS_MODEL_ROOT_CACHE_DIR="\${VERATHOS_MODEL_ROOT_CACHE_DIR:-${VERATHOS_MODEL_ROOT_CACHE_DEFAULT}}"
export VERATHOS_MERKLE_TREE_CACHE_DIR="\${VERATHOS_MERKLE_TREE_CACHE_DIR:-${VERATHOS_MERKLE_TREE_CACHE_DEFAULT}}"
# sm_89+ (Ada / Hopper / Blackwell) FP8 path workarounds for vLLM 0.20.x:
#   - FlashInfer 0.6 ships without fp8_blockscale_gemm_sm90 cubins
#     (flashinfer#2527, vllm#33833); JIT fails during CUDA-graph capture
#     with "Assertion failed: !cubin.empty()".  Disable that path.
#   - DeepGEMM E8M0 requantization mutates layer.weight in place, which
#     desyncs the FP8 weight bytes from the committed Merkle root and
#     causes W spot-check failures.  Disable.
# Cheap to set for non-FP8 GPUs/models — both vars are no-ops outside
# the FP8 blockscale path.
export VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER="\${VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER:-0}"
export VLLM_USE_DEEP_GEMM="\${VLLM_USE_DEEP_GEMM:-0}"
mkdir -p "\$TMPDIR" "\$TRITON_CACHE_DIR" "\$TORCHINDUCTOR_CACHE_DIR" "\$XDG_CACHE_HOME" "\$VERATHOS_MODEL_ROOT_CACHE_DIR" "\$VERATHOS_MERKLE_TREE_CACHE_DIR" 2>/dev/null || true
if [ -f "${VENV_DIR}/bin/activate" ]; then
    source "${VENV_DIR}/bin/activate"
fi
ENVEOF

BASHRC="${HOME}/.bashrc"
if ! grep -qF "${ENV_FILE}" "$BASHRC" 2>/dev/null; then
    echo "" >> "$BASHRC"
    echo "# Verathos environment (auto-added by setup_miner.sh)" >> "$BASHRC"
    echo "[ -f \"${ENV_FILE}\" ] && source \"${ENV_FILE}\"" >> "$BASHRC"
    echo "  Persisted env to $ENV_FILE (sourced from .bashrc)"
else
    echo "  Env already persisted in .bashrc"
fi

# ── Show model recommendations + next steps ──────────────────────────────────

if [ "$SKIP_INSTALL" = true ]; then
    echo "  Verifying existing miner runtime (--skip-install)..."
    "$PYTHON" - <<'PY'
import torch
import vllm
import zstandard
import zkllm

assert torch.cuda.is_available(), "CUDA is unavailable"
print(f"  Existing miner runtime: torch={torch.__version__}, vllm={vllm.__version__}")
PY
fi

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
echo "     source .env.sh"
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
