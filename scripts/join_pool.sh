#!/usr/bin/env bash
# Verathos mesh: join a pool as a worker with ONE command.
# Usage: bash scripts/join_pool.sh --token-file ~/.verathos/pool-token.txt [--gpus 0,1,2+3] [--single-unit]
# Installs missing build deps (Linux/apt), builds the patched llama.cpp and the
# zkllm native extension if absent, creates a venv with CUDA torch, and
# registers PM2 worker units so this machine serves the pool and starts after
# reboots. Default: ONE unit owning every GPU (fastest; models tensor-split
# locally, RPC only between machines). --gpus splits the host: comma separates
# units, + groups GPUs into one unit ("0,1,2+3" = three units). Fatal exits
# remain stopped; network and manager reconnects are retried inside the worker.
#
# Every step here exists because a real fresh pod was missing it: CUDA toolkit
# (driver-only images), libssl-dev/ninja (zkllm build), python3-venv, CUDA
# torch (plain pip installs the CPU wheel), LD_LIBRARY_PATH for the llama
# binaries, and hf_xet crashes mid-download.
set -euo pipefail
# Every `set -e` abort names its culprit: without this trap a failing bare
# command (no stderr of its own) killed the script - and the wizard above
# it - with zero output, which read as a silent join death.
trap 'echo "join_pool.sh: aborted at line $LINENO (exit $?): $BASH_COMMAND" >&2' ERR
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# One fatal exit for the whole installer. A half-installed worker that keeps
# going is worse than one that stops here: it joins, gets scheduled, and then
# fails minutes later inside a build or a first drive.
_die() { echo "install aborted: $*" >&2; exit 1; }

# CUDA major.minor that this torch was BUILT against, and that nvcc provides.
# They must agree before anything native is compiled.
_torch_cuda() {
  "$1" -c "import torch; print(torch.version.cuda or '')" 2>/dev/null || true
}
_nvcc_cuda() {
  "${1:-nvcc}" --version 2>/dev/null \
    | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -1
}
TOKEN=""; TOKEN_FILE=""; GPU_NAME=""; VRAM_GB=""; WORKER_ID="${HOSTNAME:-worker}"; ADVERTISE=""
WALLET_NAME=""; WALLET_HOTKEY=""; VALIDATOR_ALLOWLIST=""; MEMBER_ONLY=0
SUBTENSOR_NETWORK=""; NETUID=""; CHAIN_CONFIG=""; ALLOW_VALIDATORS=""
MANAGER_CA_FILE=""
RPC_PORT=50052; PROOF_PORT=9402; MESH_PORT=9443
GPUS=""; SINGLE_UNIT=0
CATALOG="$HOME/.verathos/pool-catalog.json"
while [ $# -gt 0 ]; do case "$1" in
  --token-file) TOKEN_FILE="$2"; shift 2;;
  # Compatibility only: materialize the token into an owner-only file before
  # starting PM2 so the secret never remains in the managed process argv.
  --token) TOKEN="$2"; shift 2;;
  --gpu-name) GPU_NAME="$2"; shift 2;;
  --vram-gb) VRAM_GB="$2"; shift 2;;
  # --worker-id names the per-host base; each unit becomes <base>-gpu<i>.
  --worker-id|--worker-id-base) WORKER_ID="$2"; shift 2;;
  --gpus) GPUS="$2"; shift 2;;
  # Escape hatch: one legacy unit with the exact given id and ports.
  --single-unit) SINGLE_UNIT=1; shift;;
  --advertise-host) ADVERTISE="$2"; shift 2;;
  --catalog) CATALOG="$2"; shift 2;;
  --wallet-name) WALLET_NAME="$2"; shift 2;;
  --wallet-hotkey) WALLET_HOTKEY="$2"; shift 2;;
  --validator-allowlist-path) VALIDATOR_ALLOWLIST="$2"; shift 2;;
  # Standalone mesh miners: refresh the allowlist from the metagraph
  # (there is no vLLM miner process to keep it fresh).
  --subtensor-network) SUBTENSOR_NETWORK="$2"; shift 2;;
  --netuid) NETUID="$2"; shift 2;;
  --chain-config) CHAIN_CONFIG="$2"; shift 2;;
  --allow-validators) ALLOW_VALIDATORS="$2"; shift 2;;
  --manager-ca-file) MANAGER_CA_FILE="$2"; shift 2;;
  --rpc-port) RPC_PORT="$2"; shift 2;;
  --proof-port) PROOF_PORT="$2"; shift 2;;
  --mesh-port) MESH_PORT="$2"; shift 2;;
  --member-only) MEMBER_ONLY=1; shift;;
  *) echo "unknown arg: $1" >&2; exit 2;;
esac; done
if { [ -n "$WALLET_NAME" ] && [ -z "$WALLET_HOTKEY" ]; } \
  || { [ -z "$WALLET_NAME" ] && [ -n "$WALLET_HOTKEY" ]; }; then
  echo "--wallet-name and --wallet-hotkey must be provided together" >&2
  exit 2
fi
# The worker maintains the allowlist file itself: chain coordinates come
# either from --subtensor-network/--netuid or, for a token-only join, from
# the pool's join response. The path is an implementation detail — default
# it instead of demanding a flag, and never require the file to pre-exist
# (the refresher creates it).
if [ -z "$VALIDATOR_ALLOWLIST" ]; then
  VALIDATOR_ALLOWLIST="$HOME/.verathos/validator-allowlist.json"
fi
case "$VALIDATOR_ALLOWLIST" in /*) ;; *) VALIDATOR_ALLOWLIST="$PWD/$VALIDATOR_ALLOWLIST";; esac
if { [ -n "$SUBTENSOR_NETWORK" ] && [ -z "$NETUID" ]; } \
  || { [ -z "$SUBTENSOR_NETWORK" ] && [ -n "$NETUID" ]; }; then
  echo "--subtensor-network and --netuid must be provided together" >&2
  exit 2
fi
if [ -n "$MANAGER_CA_FILE" ]; then
  case "$MANAGER_CA_FILE" in /*) ;; *) MANAGER_CA_FILE="$PWD/$MANAGER_CA_FILE";; esac
  [ -f "$MANAGER_CA_FILE" ] || {
    echo "--manager-ca-file must name a regular CA bundle" >&2
    exit 2
  }
fi
[ -z "$TOKEN" ] || [ -z "$TOKEN_FILE" ] || { echo "use only one of --token-file or --token" >&2; exit 2; }
if [ -n "$TOKEN" ]; then
  TOKEN_DIR="$HOME/.verathos"
  mkdir -p "$TOKEN_DIR"
  chmod 700 "$TOKEN_DIR"
  TOKEN_FILE="$TOKEN_DIR/pool-token.txt"
  TMP_TOKEN_FILE="$(mktemp "$TOKEN_DIR/.pool-token.XXXXXX")"
  chmod 600 "$TMP_TOKEN_FILE"
  printf '%s\n' "$TOKEN" > "$TMP_TOKEN_FILE"
  mv -f "$TMP_TOKEN_FILE" "$TOKEN_FILE"
  unset TOKEN TMP_TOKEN_FILE
fi
[ -n "$TOKEN_FILE" ] || { echo "--token-file is required (legacy --token is also accepted)" >&2; exit 2; }
case "$TOKEN_FILE" in /*) ;; *) TOKEN_FILE="$PWD/$TOKEN_FILE";; esac
[ ! -L "$TOKEN_FILE" ] || { echo "--token-file must not be a symlink" >&2; exit 2; }
[ -f "$TOKEN_FILE" ] || { echo "--token-file must name a regular file" >&2; exit 2; }

# --- backend + defaults -----------------------------------------------------
if command -v nvidia-smi >/dev/null 2>&1; then
  BACKEND=cuda; RPC_DEV=CUDA0
  : "${GPU_NAME:=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)}"
  : "${VRAM_GB:=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | awk '{print int($1/1024)}')}"
elif [ "$(uname)" = "Darwin" ]; then
  BACKEND=metal; RPC_DEV=MTL0
  SINGLE_UNIT=1   # Apple silicon has one GPU; the legacy path fits it.
  : "${GPU_NAME:=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo 'Apple Silicon')}"
  : "${VRAM_GB:=$(( $(sysctl -n hw.memsize) / 1073741824 - 8 ))}"   # leave 8GB for the OS
else
  echo "no CUDA GPU and not macOS; unsupported backend" >&2; exit 1
fi
# hostname -I is Linux-only; on macOS use the primary interface. Under
# set -o pipefail the pipeline inherits hostname's failure even though awk
# exits 0, and set -e then kills the whole script silently on macOS - the
# failing stage must be neutralized explicitly.
if [ -z "$ADVERTISE" ]; then
  LAN_IP="$( (hostname -I 2>/dev/null || true) | awk '{print $1}')"
  [ -n "$LAN_IP" ] || LAN_IP="$(ipconfig getifaddr en0 2>/dev/null || true)"
  # How the POOL reaches this machine is the operator's call, so ask when a
  # terminal is available (curl|bash keeps /dev/tty even though stdin is the
  # script). IPs are detected, never typed.
  if [ -e /dev/tty ] && [ -t 1 ]; then
    PUB_IP="$(curl -fsS --max-time 5 https://api.ipify.org 2>/dev/null || true)"
    echo "how does the pool reach this machine?"
    echo "  1) same LAN as the pool (use ${LAN_IP:-unknown})"
    [ -n "$PUB_IP" ] && echo "  2) over the internet (use $PUB_IP; needs a router/firewall port forward)"
    echo "  3) it is not reachable (join anyway; it can host once the pool can reach it)"
    read -r REPLY < /dev/tty || REPLY=1
    case "${REPLY:-1}" in
      2)
        ADVERTISE="$PUB_IP"
        printf "forwarded port for mesh serving [%s]: " "$MESH_PORT" > /dev/tty
        read -r PORT_REPLY < /dev/tty || PORT_REPLY=""
        [ -z "$PORT_REPLY" ] || MESH_PORT="$PORT_REPLY"
        ;;
      3) ADVERTISE="${LAN_IP:-127.0.0.1}";;
      *) ADVERTISE="${LAN_IP:-127.0.0.1}";;
    esac
  else
    ADVERTISE="$LAN_IP"
  fi
  [ -n "$ADVERTISE" ] || ADVERTISE=127.0.0.1
fi

# --- build dependencies (Linux/apt; macOS ships its toolchain) ----------------
SUDO=""; [ "$(id -u)" -ne 0 ] && command -v sudo >/dev/null && SUDO="sudo"
if [ "$BACKEND" = cuda ] && command -v apt-get >/dev/null 2>&1; then
  MISSING=""
  command -v git   >/dev/null || MISSING="$MISSING git"
  command -v cmake >/dev/null || MISSING="$MISSING cmake"
  command -v gcc   >/dev/null || MISSING="$MISSING build-essential"
  command -v ninja >/dev/null || MISSING="$MISSING ninja-build"
  # The worker's drive path verifies port ownership with lsof before it
  # frees anything; a fresh container without it fails every first drive.
  command -v lsof  >/dev/null || MISSING="$MISSING lsof"
  command -v wget  >/dev/null || MISSING="$MISSING wget"
  # PM2 supervises every worker unit: `verathos mesh start/stop/logs` drive it
  # and units must survive a reboot. Without it a machine "joins" and then has
  # nothing supervising it, so install node/npm here rather than degrading to
  # a printed command line that nobody runs.
  command -v node  >/dev/null || MISSING="$MISSING nodejs"
  command -v npm   >/dev/null || MISSING="$MISSING npm"
  dpkg -s libssl-dev    >/dev/null 2>&1 || MISSING="$MISSING libssl-dev"
  # Probe the CAPABILITY, not the package name: python3.X-venv (or a
  # non-deb python) provides working venvs while `dpkg -s python3-venv`
  # fails, which forced a needless sudo apt run on such machines.
  python3 -c 'import venv, ensurepip' >/dev/null 2>&1 \
    || MISSING="$MISSING python3-venv"
  python3 -c 'import sysconfig, os, sys
sys.exit(0 if os.path.exists(sysconfig.get_paths()["include"] + "/Python.h") else 1)' \
    >/dev/null 2>&1 || dpkg -s python3-dev >/dev/null 2>&1 \
    || MISSING="$MISSING python3-dev"
  if [ -n "$MISSING" ]; then
    echo "installing build deps:$MISSING"
    $SUDO apt-get update -qq
    # shellcheck disable=SC2086
    $SUDO apt-get install -y -qq $MISSING >/dev/null \
      || _die "apt could not install:$MISSING"
  fi
  if ! command -v pm2 >/dev/null 2>&1; then
    echo "installing pm2 (worker unit supervisor)"
    $SUDO npm install -g pm2 >/dev/null 2>&1 \
      || _die "npm could not install pm2; install it and re-run"
  fi
  CUDA_HOME="$(ls -d /usr/local/cuda-* 2>/dev/null | sort -V | tail -1 || true)"
  [ -n "$CUDA_HOME" ] || CUDA_HOME=/usr/local/cuda
  export PATH="$CUDA_HOME/bin:$PATH"
fi

# The CUDA toolkit is needed ONLY to compile the patched llama.cpp (the
# zkllm prover and every other native piece ship as prebuilt wheels).
# Install it lazily from the build path instead of on every join: a
# driver-only image that reuses a cached runtime build never pays the
# ~3GB download and avoids racing an otherwise unnecessary toolkit install
# against concurrent package operations.
_ensure_cuda_toolkit() {
  if command -v nvcc >/dev/null 2>&1 || [ -x /usr/local/cuda/bin/nvcc ] \
     || ls /usr/local/cuda-*/bin/nvcc >/dev/null 2>&1; then
    return 0
  fi
  command -v apt-get >/dev/null 2>&1 \
    || _die "no nvcc and no apt-get; install the CUDA toolkit and re-run"
  . /etc/os-release
  UBU="ubuntu${VERSION_ID/./}"
  # Install the toolkit torch will need, not a fixed one: an image that
  # already ships torch dictates the version, and guessing here means
  # downloading ~3GB twice (once wrong, once right).
  WANT_CUDA="$(_torch_cuda "$REPO/.venv-mesh/bin/python")"
  [ -n "$WANT_CUDA" ] || WANT_CUDA="12.8"   # matches the cu128 wheel below
  echo "installing cuda-toolkit-${WANT_CUDA/./-} (one time, ~3GB; driver-only image detected)"
  wget -q "https://developer.download.nvidia.com/compute/cuda/repos/${UBU}/x86_64/cuda-keyring_1.1-1_all.deb" -O /tmp/cuda-keyring.deb \
    || _die "could not download the NVIDIA apt keyring"
  $SUDO dpkg -i /tmp/cuda-keyring.deb >/dev/null
  $SUDO apt-get update -qq
  $SUDO apt-get install -y -qq "cuda-toolkit-${WANT_CUDA/./-}" >/dev/null \
    || _die "could not install cuda-toolkit-${WANT_CUDA/./-}"
  CUDA_HOME="$(ls -d /usr/local/cuda-* 2>/dev/null | sort -V | tail -1 || true)"
  [ -n "$CUDA_HOME" ] || CUDA_HOME=/usr/local/cuda
  export PATH="$CUDA_HOME/bin:$PATH"
}
if [ "$(uname)" = "Darwin" ]; then
  # Non-interactive shells miss Homebrew paths, which is where cmake and
  # openssl live on most Macs; a login Mac worked interactively while the
  # scripted install died on "cmake: command not found".
  export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
  command -v cc >/dev/null 2>&1 || {
    echo "Xcode command line tools missing; run: xcode-select --install" >&2
    exit 1
  }
  if ! command -v cmake >/dev/null 2>&1; then
    if command -v brew >/dev/null 2>&1; then
      echo "installing cmake via Homebrew (one time)"
      brew install -q cmake
    else
      echo "cmake missing and no Homebrew; install https://brew.sh then re-run" >&2
      exit 1
    fi
  fi
fi

# --- patched llama.cpp ------------------------------------------------------
# Reuse is gated on the patch-series provenance, not bare existence: a cached
# build from an older patch set or upstream base may serve unpatched binaries.
BUILD_DIR="$HOME/.cache/verathos-mesh-runtime/llama.cpp-build/build-verathos-$BACKEND/bin"
_sha256_stream() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum | cut -d' ' -f1
  else shasum -a 256 | cut -d' ' -f1; fi
}
RUNTIME_STAMP_WANT="$(cat "$REPO"/patches/llama.cpp/*.patch \
  "$REPO/patches/llama.cpp/build.sh" \
  "$REPO/patches/llama.cpp/UPSTREAM_BASE.txt" 2>/dev/null | _sha256_stream)"
RUNTIME_STAMP_FILE="$BUILD_DIR/verathos-join-stamp.sha256"
if [ ! -x "$BUILD_DIR/verathos-rpc-server" ] \
   || [ "$(cat "$RUNTIME_STAMP_FILE" 2>/dev/null)" != "$RUNTIME_STAMP_WANT" ]; then
  [ "$BACKEND" = cuda ] && _ensure_cuda_toolkit
  # Reaching here means the cached build is absent, aborted, or from a
  # different patch series (the stamp is written only AFTER success): a
  # stale half-built dir makes later compiles fail on unwritable object
  # paths, so rebuild from a clean slate.
  rm -rf "$(dirname "$BUILD_DIR")"
  echo "building patched llama.cpp ($BACKEND); one time, several minutes..."
  bash "$REPO/patches/llama.cpp/build.sh" --backend "$BACKEND"
  echo "$RUNTIME_STAMP_WANT" > "$RUNTIME_STAMP_FILE"
fi

# --- python interpreter (>= 3.10, NATIVE arch) -------------------------------
# Macs routinely carry several pythons and the first "python3" on PATH is
# often an old or Rosetta x86_64 build: a venv made from it installs x86_64
# wheels whose native extensions (bittensor_wallet, torch) cannot load the
# arm64 proof stack. Pick the newest interpreter that is BOTH >= 3.10 and
# the machine's native architecture; MESH_PYTHON overrides.
_python_ok() {
  "$1" -c "import sys, platform; raise SystemExit(0 if sys.version_info >= (3, 10) and platform.machine() == '$(uname -m)' else 1)" >/dev/null 2>&1
}
if [ -z "${MESH_PYTHON:-}" ]; then
  for CAND in python3.13 python3.12 python3.11 python3.10 \
      /opt/homebrew/bin/python3 "$HOME/anaconda3/bin/python3" \
      "$HOME/miniconda3/bin/python3" python3; do
    P="$(command -v "$CAND" 2>/dev/null || true)"
    [ -n "$P" ] || P="$CAND"
    [ -x "$P" ] || continue
    if _python_ok "$P"; then MESH_PYTHON="$P"; break; fi
  done
fi
if [ -z "${MESH_PYTHON:-}" ]; then
  echo "no native $(uname -m) python >= 3.10 found; install one (macOS: brew install python@3.12) or set MESH_PYTHON" >&2
  exit 1
fi

# --- venv (CUDA torch on Linux; plain pip's default torch is CPU-only) -------
VENV="$REPO/.venv-mesh"
if [ -d "$VENV" ] && ! _python_ok "$VENV/bin/python"; then
  # A venv from a Rosetta/old python poisons every native wheel; rebuilding
  # it is the one-shot repair, not a manual debugging session.
  echo "existing $VENV is the wrong arch or python < 3.10; recreating it"
  rm -rf "$VENV"
fi
if [ ! -d "$VENV" ]; then
  "$MESH_PYTHON" -m venv "$VENV"
  "$VENV/bin/pip" install -q --upgrade pip
fi
# The coordinator bootstrap (setup_mesh.sh) may have created the venv without
# torch, so check torch itself, not just the venv directory: a GPU worker
# needs the CUDA wheel either way (macOS default wheels ARE the Metal/arm
# builds, so plain pip is correct there).
if [ "$BACKEND" = cuda ] \
  && ! "$VENV/bin/python" -c "import torch; assert torch.version.cuda" >/dev/null 2>&1; then
  # Pinned to the majors the shipped zkllm wheels carry native kernels for.
  "$VENV/bin/pip" install -q "torch>=2.10,<2.12" --index-url https://download.pytorch.org/whl/cu128 \
    || _die "could not install the CUDA torch wheel"
fi
# The CUDA toolkit must match the CUDA torch was built against, or every
# native build fails ("The detected CUDA version mismatches the version that
# was used to compile PyTorch"). Torch is the harder constraint (its wheel is
# prebuilt), so the TOOLKIT follows torch. This matters on rented pods, whose
# images often ship their own torch: installing a fixed toolkit version and
# assuming it matches is exactly how a box gets to the zkllm build and dies.
if [ "$BACKEND" = cuda ]; then
  TORCH_CUDA="$(_torch_cuda "$VENV/bin/python")"
  [ -n "$TORCH_CUDA" ] || _die "torch in $VENV has no CUDA build"
  # CUDA_HOME is only set by the apt block above; a non-apt Linux may still
  # have a perfectly good toolkit on PATH.
  CUDA_HOME="${CUDA_HOME:-}"
  if [ -z "$CUDA_HOME" ] && command -v nvcc >/dev/null 2>&1; then
    CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
  fi
  NVCC_CUDA="$(_nvcc_cuda "${CUDA_HOME:-/usr/local/cuda}/bin/nvcc")"
  if [ "$NVCC_CUDA" != "$TORCH_CUDA" ]; then
    WANT_DIR="/usr/local/cuda-${TORCH_CUDA}"
    if [ ! -x "$WANT_DIR/bin/nvcc" ] && command -v apt-get >/dev/null 2>&1; then
      echo "torch was built against CUDA $TORCH_CUDA but nvcc is ${NVCC_CUDA:-absent}"
      echo "installing cuda-toolkit-${TORCH_CUDA/./-} to match torch (one time)"
      . /etc/os-release
      UBU="ubuntu${VERSION_ID/./}"
      if [ ! -f /usr/share/keyrings/cuda-archive-keyring.gpg ]; then
        wget -q "https://developer.download.nvidia.com/compute/cuda/repos/${UBU}/x86_64/cuda-keyring_1.1-1_all.deb" -O /tmp/cuda-keyring.deb \
          && $SUDO dpkg -i /tmp/cuda-keyring.deb >/dev/null
      fi
      $SUDO apt-get update -qq
      $SUDO apt-get install -y -qq "cuda-toolkit-${TORCH_CUDA/./-}" >/dev/null \
        || _die "no cuda-toolkit-${TORCH_CUDA/./-} package for torch's CUDA $TORCH_CUDA"
    fi
    [ -x "$WANT_DIR/bin/nvcc" ] \
      || _die "torch needs CUDA $TORCH_CUDA; no nvcc at $WANT_DIR/bin/nvcc"
    CUDA_HOME="$WANT_DIR"
    export CUDA_HOME
    export PATH="$CUDA_HOME/bin:$PATH"
  fi
  NVCC_CUDA="$(_nvcc_cuda "$CUDA_HOME/bin/nvcc")"
  [ "$NVCC_CUDA" = "$TORCH_CUDA" ] \
    || _die "CUDA toolkit $NVCC_CUDA still does not match torch's $TORCH_CUDA"
  echo "toolchain: torch CUDA $TORCH_CUDA, nvcc $NVCC_CUDA ($CUDA_HOME)"
fi
if [ "$BACKEND" = metal ] \
  && ! "$VENV/bin/python" -c "import torch" >/dev/null 2>&1; then
  "$VENV/bin/pip" install -q "torch>=2.10,<2.12"
fi
"$VENV/bin/pip" install -q -e "$REPO[neurons]" huggingface_hub >/dev/null
# macOS: the bittensor_wallet native wheel links openssl@3 from the SAME
# architecture's Homebrew prefix (/opt/homebrew on Apple silicon). Macs with
# only Intel/Rosetta Homebrew have an x86_64 libssl that can never satisfy
# the arm64 wheel, so after trying brew, scan for any native-arch
# libssl.3.dylib already on the machine (conda package caches and envs
# carry them) and pin it via DYLD_FALLBACK_LIBRARY_PATH for the worker.
WORKER_DYLD_FALLBACK=""
_wallet_ok() {
  env ${WORKER_DYLD_FALLBACK:+DYLD_FALLBACK_LIBRARY_PATH="$WORKER_DYLD_FALLBACK"} \
    "$VENV/bin/python" -c "from bittensor_wallet import Keypair" >/dev/null 2>&1
}
if [ "$(uname)" = "Darwin" ] && ! _wallet_ok; then
  if [ -x /opt/homebrew/bin/brew ]; then
    echo "installing openssl@3 via Homebrew (bittensor_wallet runtime dependency)"
    /opt/homebrew/bin/brew install -q openssl@3 || true
  fi
  if ! _wallet_ok; then
    HOST_ARCH="$(uname -m)"
    for CAND_DIR in \
        /opt/homebrew/opt/openssl@3/lib \
        "$HOME"/anaconda3/pkgs/openssl-3*/lib \
        "$HOME"/miniconda3/pkgs/openssl-3*/lib \
        "$HOME"/anaconda3/envs/*/lib \
        "$HOME"/miniconda3/envs/*/lib; do
      [ -f "$CAND_DIR/libssl.3.dylib" ] || continue
      file -b "$CAND_DIR/libssl.3.dylib" | grep -q "$HOST_ARCH" || continue
      WORKER_DYLD_FALLBACK="$CAND_DIR"
      if _wallet_ok; then
        echo "using native libssl.3 from $CAND_DIR (DYLD fallback for the worker)"
        break
      fi
      WORKER_DYLD_FALLBACK=""
    done
  fi
fi

# Mesh-only boxes get the plain `verathos` command globally (fleet/status/
# pool probe without activating the venv). Never clobber a vLLM miner box,
# where .env.sh owns the name.
if [ ! -e /usr/local/bin/verathos ] && [ ! -d "$REPO/.venv-vllm" ]; then
  SUDO_LN=""
  [ "$(id -u)" -ne 0 ] && command -v sudo >/dev/null && SUDO_LN="sudo"
  $SUDO_LN ln -sfn "$VENV/bin/verathos" /usr/local/bin/verathos 2>/dev/null || true
fi

# --- zkllm native extension (proofs refuse to run on the NumPy fallback) -----
# Required on EVERY proof-producing worker: metal serving fail-closes on the
# NumPy fallback exactly like cuda, so the mac build (CPU field arithmetic,
# no CUDA kernels) is not optional either.
# The check is a real micro-proof, not an import: a stale extension built
# against an older torch LOADS cleanly but garbles every tensor that
# crosses the binding (a July .so under a newer torch turned plain 2D
# tensors into "must be 2D" runtime errors at first proof). Import-level
# checks pass that state; only exercising the binding catches it.
_zkllm_probe() {
  "$VENV/bin/python" - >/dev/null 2>&1 <<'ZKEOF'
import torch
import zkllm.cuda as z
assert z._load_native() is not None
from zkllm.prover.gemm_fast import GEMMProverFast
from zkllm.crypto.transcript import Transcript
X = torch.randint(-4, 4, (64, 128), dtype=torch.int64)
W = torch.randint(-4, 4, (128, 64), dtype=torch.int8)
GEMMProverFast().prove(
    X, W, X @ W.to(torch.int64), [(0, 0)], Transcript(b"install-probe")
)
# On a CUDA box the GPU kernels must actually LAUNCH for this GPU's arch:
# a wheel built without it imports and passes the CPU micro-proof, then
# fails every launch with "no kernel image", silently dropping the
# proof-weight cache to the FULL profile (near-model-sized cache + 6-200s
# CPU tree builds). Probing the launch here routes that state into the
# on-box rebuild below instead.
if torch.cuda.is_available():
    from zkllm.cuda import zkllm_native as native
    probe = torch.zeros(1, 256, dtype=torch.int8, device="cuda")
    native.cuda_blake3_merkle_leaves(probe.contiguous(), 128, 0)
ZKEOF
}
if ! _zkllm_probe; then
  if [ ! -d "$REPO/zkllm/cuda" ]; then
    # Public release trees ship the extension as prebuilt wheels only:
    # install (or reinstall) the wheel into the venv and re-probe.
    # --no-index: only dist/ may satisfy the name.
    echo "installing zkllm prover wheel (public release tree)..."
    "$VENV/bin/pip" install --no-cache-dir --force-reinstall --no-deps --no-index \
      --find-links "$REPO/dist" zkllm >/dev/null 2>&1 \
      || _die "no installable zkllm wheel in dist/ for $("$VENV/bin/python" -V 2>&1)"
    _zkllm_probe || _die "the shipped zkllm wheel failed its micro-proof on this box"
  else
  echo "building zkllm native extension; one time, a few minutes..."
  # A stale in-tree .so from an older torch shadows any fresh build AND
  # loads without error; remove it so the rebuild is what gets loaded.
  # The build/ cache must go too: setuptools sees unchanged sources and
  # silently copies the OLD artifact back out of build/lib.* (with its
  # original mtime), reproducing the broken state while claiming success.
  rm -rf "$REPO"/zkllm/cuda/zkllm_native.*.so "$REPO"/zkllm/cuda/build
  (cd "$REPO/zkllm/cuda" && "$VENV/bin/python" build.py)
  _zkllm_probe \
    || { echo "zkllm native build failed its micro-proof; see output above" >&2; exit 1; }
  fi
fi

# --- native PCS commitment library (gemm-v2 sidecar) --------------------------
# Serving refuses to start without it (VERATHOS_MESH_REQUIRE_GEMM_V2 defaults
# on). The zkllm wheel SHIPS it (zkllm/crypto/libverathos_pcs_v2.so), but the
# editable source tree shadows the wheel package, so materialize the wheel's
# copy into the tree first . dist/pcs prebuilts cover
# platforms without a wheel (macOS); the in-repo crate is the last fallback.
if ! "$VENV/bin/python" -c "from zkllm.crypto.pcs_v2 import native_library_path; native_library_path()" >/dev/null 2>&1; then
  PCS_TMP="$(mktemp -d)"
  if [ -d "$REPO/zkllm/crypto" ]; then
    PCS_DEST="$REPO/zkllm/crypto"
  else
    # Public tree: no source package shadows the venv, so the venv's own
    # zkllm/crypto is both the import target and the materialization target.
    PCS_DEST="$("$VENV/bin/python" -c 'import os, zkllm; print(os.path.join(os.path.dirname(zkllm.__file__), "crypto"))' 2>/dev/null || true)"
  fi
  if [ -n "$PCS_DEST" ] && "$VENV/bin/pip" install --no-cache-dir --no-deps --target "$PCS_TMP" \
       --find-links "$REPO/dist" zkllm >/dev/null 2>&1 \
     && ls "$PCS_TMP"/zkllm/crypto/libverathos_pcs_v2.* >/dev/null 2>&1; then
    cp "$PCS_TMP"/zkllm/crypto/libverathos_pcs_v2.* "$PCS_DEST/"
    echo "installed native PCS library from the shipped zkllm wheel"
  fi
  rm -rf "$PCS_TMP"
fi
if ! "$VENV/bin/python" -c "from zkllm.crypto.pcs_v2 import native_library_path; native_library_path()" >/dev/null 2>&1; then
  case "$(uname)-$(uname -m)" in
    Linux-x86_64)  PCS_PREBUILT="$REPO/dist/pcs/linux-x86_64/libverathos_pcs_v2.so";;
    Darwin-arm64)  PCS_PREBUILT="$REPO/dist/pcs/macos-arm64/libverathos_pcs_v2.dylib";;
    *)             PCS_PREBUILT="";;
  esac
  if [ -n "$PCS_PREBUILT" ] && [ -f "$PCS_PREBUILT" ] && [ -n "${PCS_DEST:-}" ]; then
    cp "$PCS_PREBUILT" "$PCS_DEST/"
    echo "installed prebuilt native PCS library ($(basename "$PCS_PREBUILT"))"
  elif command -v cargo >/dev/null 2>&1 && [ -d "$REPO/zkllm/pcs_native" ]; then
    echo "building native PCS library from source (one time)..."
    (cd "$REPO/zkllm/pcs_native" && cargo build --release -q)
  fi
  "$VENV/bin/python" -c "from zkllm.crypto.pcs_v2 import native_library_path; native_library_path()" >/dev/null 2>&1 \
    || { echo "native PCS library unavailable: the zkllm wheel in dist/ carries no PCS library for this platform and no dist/pcs prebuilt matches $(uname)-$(uname -m); install rust (https://rustup.rs) and re-run to build zkllm/pcs_native" >&2; exit 1; }
fi

# --- hot-capacity audit workspace wheel (CUDA workers only) -------------------
# A worker in a chain-bound subnet mesh proves GPU exclusivity with the same
# synthetic CUDA workload vLLM miners run. The extension ships as prebuilt
# per-Python-minor wheels like zkllm; never copy a bare .so. A missing wheel
# only disables this worker's audit openings (the daemon preflight logs it),
# but the slot then accumulates capacity strikes — so install it here.
if [ "$BACKEND" = cuda ]; then
  if ! "$VENV/bin/python" -c "import torch, hot_capacity_workspace_cuda" >/dev/null 2>&1; then
    PYTAG="$("$VENV/bin/python" -c 'import sys; print(f"cp{sys.version_info[0]}{sys.version_info[1]}")')"
    # A no-match `ls` exits non-zero; under `set -euo pipefail` the bare
    # assignment aborts with exit 2 and no message, so keep the lookup
    # itself tolerant and fail with a real explanation below. dist/ ships
    # in the repo AND in the manager worker bundle, so a missing wheel
    # means a broken or hand-pruned checkout, not a supported setup.
    HOTCAP_WHEEL="$(ls "$REPO"/dist/hot_capacity_workspace_cuda-*-"$PYTAG"-*.whl 2>/dev/null | head -1 || true)"
    if [ -n "$HOTCAP_WHEEL" ]; then
      "$VENV/bin/pip" install -q "$HOTCAP_WHEEL"
      "$VENV/bin/python" -c "import torch, hot_capacity_workspace_cuda" >/dev/null 2>&1 \
        || _die "hot-capacity workspace wheel installed but not importable"
      echo "installed hot-capacity workspace wheel ($(basename "$HOTCAP_WHEEL"))"
    elif [ "${VERATHOS_ALLOW_NO_CAPACITY_AUDITS:-}" = "1" ]; then
      echo "warning: no hot_capacity_workspace_cuda wheel for $PYTAG in $REPO/dist; capacity audits DISABLED (explicitly allowed via VERATHOS_ALLOW_NO_CAPACITY_AUDITS=1)" >&2
    else
      _die "no hot_capacity_workspace_cuda wheel for $PYTAG in $REPO/dist: a subnet worker without it serves audit-blind and probates. Restore dist/ in the checkout, or set VERATHOS_ALLOW_NO_CAPACITY_AUDITS=1 for a light-tier worker."
    fi
  fi
fi

# --- preflight: everything serving fail-closes on, checked BEFORE joining ----
# Each of these bit a real machine as a runtime failure minutes after an
# apparently successful install; a worker that cannot serve must fail HERE.
# The host tools first: every one of these is used by a path the worker takes
# on its own (supervision, stopping a mesh, native builds), so a machine
# missing any of them is not installed, whatever else succeeded.
# Supervision is launchd on macOS and pm2 on Linux; requiring pm2 on Darwin
# aborted the install after the whole build even though start_unit never
# touches pm2 there.
if [ "$(uname)" = "Darwin" ]; then
  PREFLIGHT_TOOLS="launchctl lsof"
else
  PREFLIGHT_TOOLS="pm2 lsof"
fi
for TOOL in $PREFLIGHT_TOOLS; do
  command -v "$TOOL" >/dev/null 2>&1 || _die "$TOOL is required but missing"
done
if [ "$BACKEND" = cuda ]; then
  [ "$(_nvcc_cuda "${CUDA_HOME:-/usr/local/cuda}/bin/nvcc")" = "$(_torch_cuda "$VENV/bin/python")" ] \
    || _die "CUDA toolkit does not match torch's CUDA"
fi
env ${WORKER_DYLD_FALLBACK:+DYLD_FALLBACK_LIBRARY_PATH="$WORKER_DYLD_FALLBACK"} \
  "$VENV/bin/python" - <<'PYEOF' || exit 1
failures = []
try:
    from verallm.mesh.receipt_signing import _stage_keypair_from_seed
    _stage_keypair_from_seed("ab" * 32)
except Exception as exc:  # noqa: BLE001
    hint = ""
    if "libssl" in str(exc) or "openssl" in str(exc).lower():
        hint = " (macOS: brew install openssl@3)"
    failures.append(f"stage receipt signer: {exc}{hint}")
try:
    import torch
    import zkllm.cuda as z
    assert z._load_native() is not None, "native extension did not load"
    from zkllm.prover.gemm_fast import GEMMProverFast
    from zkllm.crypto.transcript import Transcript
    _X = torch.randint(-4, 4, (64, 128), dtype=torch.int64)
    _W = torch.randint(-4, 4, (128, 64), dtype=torch.int8)
    GEMMProverFast().prove(
        _X, _W, _X @ _W.to(torch.int64), [(0, 0)], Transcript(b"preflight")
    )
except Exception as exc:  # noqa: BLE001
    failures.append(f"zkllm native prover: {exc}")
try:
    from zkllm.crypto.pcs_v2 import native_library_path
    native_library_path()
except Exception as exc:  # noqa: BLE001
    failures.append(f"native PCS library: {exc}")
if failures:
    print("preflight FAILED; this worker could join but never serve:")
    for failure in failures:
        print(f"  - {failure}")
    raise SystemExit(1)
print("preflight OK: signer + native prover + PCS library all load")
PYEOF

[ -f "$CATALOG" ] || echo "[]" > "$CATALOG"   # empty catalog = file-less member only

# --- runtime environment ------------------------------------------------------
# The llama binaries resolve libggml/libcudart at runtime; a worker started
# without this hangs every drive with a crash-looping rpc-server.
RUNTIME_LD="$BUILD_DIR"
[ "$BACKEND" = cuda ] && RUNTIME_LD="$RUNTIME_LD:${CUDA_HOME:-/usr/local/cuda}/lib64"
export LD_LIBRARY_PATH="$RUNTIME_LD${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HF_HUB_DISABLE_XET=1   # xet native downloader can crash the process mid-download

# --- shared worker flags -----------------------------------------------------
# Array, not a string: GPU names contain spaces ("NVIDIA GeForce RTX 4090")
# and an unquoted expansion would word-split them into stray argv.
COMMON=(--pool-token-file "$TOKEN_FILE" --advertise-host "$ADVERTISE"
  --llama-server-binary "$BUILD_DIR/llama-server"
  --rpc-worker-binary "$BUILD_DIR/verathos-rpc-server")
[ -z "$WALLET_NAME" ] || COMMON+=(--wallet-name "$WALLET_NAME" --wallet-hotkey "$WALLET_HOTKEY")
[ -z "$VALIDATOR_ALLOWLIST" ] || COMMON+=(--validator-allowlist-path "$VALIDATOR_ALLOWLIST")
[ -z "$SUBTENSOR_NETWORK" ] || COMMON+=(--subtensor-network "$SUBTENSOR_NETWORK" --netuid "$NETUID")
[ -z "$CHAIN_CONFIG" ] || COMMON+=(--chain-config "$CHAIN_CONFIG")
[ -z "$ALLOW_VALIDATORS" ] || COMMON+=(--allow-validators "$ALLOW_VALIDATORS")
[ -z "$MANAGER_CA_FILE" ] || COMMON+=(--manager-ca-file "$MANAGER_CA_FILE")
[ "$MEMBER_ONLY" -eq 0 ] || COMMON+=(--member-only)

STARTED_UNITS=()

confirm_joins() {
  # A started unit is NOT a joined unit: the daemon negotiates membership
  # with the manager after launch, and the manager can refuse (undialable
  # advertise address, bad token). Reporting success before that
  # confirmation can leave operators "joined" on boxes the pool rejected.
  # Wait for each unit's verdict in its logs.
  local deadline=$((SECONDS + 120)) pending name ok=0 refused=0
  local logs
  echo "waiting for the pool to accept the join (up to 120s)..."
  while [ $SECONDS -lt $deadline ]; do
    pending=0
    for name in "${STARTED_UNITS[@]}"; do
      if [ "$(uname)" = "Darwin" ]; then
        logs="$HOME/.verathos/logs/${name}.log"
      else
        logs="$HOME/.pm2/logs/${name}-out.log $HOME/.pm2/logs/${name}-error.log"
      fi
      # shellcheck disable=SC2086
      if grep -q "pool join accepted" $logs 2>/dev/null; then
        continue
      elif grep -q "pool join REFUSED" $logs 2>/dev/null; then
        continue
      fi
      pending=1
    done
    [ "$pending" -eq 0 ] && break
    sleep 3
  done
  for name in "${STARTED_UNITS[@]}"; do
    if [ "$(uname)" = "Darwin" ]; then
      logs="$HOME/.verathos/logs/${name}.log"
    else
      logs="$HOME/.pm2/logs/${name}-out.log $HOME/.pm2/logs/${name}-error.log"
    fi
    # shellcheck disable=SC2086
    if grep -q "pool join accepted" $logs 2>/dev/null; then
      echo "  $name: JOINED"
      ok=$((ok + 1))
    elif grep -q "pool join REFUSED" $logs 2>/dev/null; then
      echo "  $name: REFUSED by the pool manager:"
      # shellcheck disable=SC2086
      grep -h "pool join REFUSED" $logs 2>/dev/null | tail -1 | sed 's/^/      /'
      refused=$((refused + 1))
    else
      echo "  $name: no join confirmation yet — check logs: pm2 logs $name"
    fi
  done
  if [ "$refused" -gt 0 ] || [ "$ok" -lt "${#STARTED_UNITS[@]}" ]; then
    echo "JOIN INCOMPLETE: $ok/${#STARTED_UNITS[@]} unit(s) accepted."
    echo "Fix the reported reason and re-run this installer (safe to repeat)."
    return 1
  fi
  echo "all ${#STARTED_UNITS[@]} unit(s) joined the pool."
}

start_unit() {
  # start_unit NAME CUDA_VISIBLE WORKER_ARGS...
  local name="$1" cuda_visible="$2"; shift 2
  STARTED_UNITS+=("$name")
  if [ "$(uname)" = "Darwin" ]; then
    # macOS supervision = launchd, the native answer to "survive reboots":
    # no Node/pm2 dependency, and RunAtLoad + KeepAlive=false reproduces the
    # pm2 --no-autorestart contract (start at login, fatal exits stay down).
    local label="ai.verathos.${name}"
    local plist="$HOME/Library/LaunchAgents/${label}.plist"
    local log_dir="$HOME/.verathos/logs"
    mkdir -p "$HOME/Library/LaunchAgents" "$log_dir"
    # Same stale-verdict hazard as the pm2 flush below: start each run
    # with a fresh log so confirm_joins reads THIS run's outcome.
    : > "$log_dir/${name}.log"
    "$VENV/bin/python" - "$plist" "$label" "$VENV/bin/python" \
      "$log_dir/${name}.log" "$REPO" "${WORKER_DYLD_FALLBACK:-}" "$@" <<'PLISTEOF'
import plistlib
import sys

plist_path, label, python, log, repo, dyld_fallback, *worker_args = sys.argv[1:]
env = {"HF_HUB_DISABLE_XET": "1"}
if dyld_fallback:
    env["DYLD_FALLBACK_LIBRARY_PATH"] = dyld_fallback
payload = {
    "Label": label,
    "ProgramArguments": [
        python, "-m", "neurons.cli", "mesh", "pool", "worker", *worker_args
    ],
    "WorkingDirectory": repo,
    "RunAtLoad": True,
    "KeepAlive": False,
    "StandardOutPath": log,
    "StandardErrorPath": log,
    "EnvironmentVariables": env,
    # macOS launchd defaults the soft FD limit to 256; a serving worker
    # (llama slots + proof endpoint + mesh RPC + hub downloads) exhausts
    # that under load and starts throwing EMFILE as HTTP 500s on the
    # proof endpoint .
    "SoftResourceLimits": {"NumberOfFiles": 65536},
    "HardResourceLimits": {"NumberOfFiles": 65536},
}
with open(plist_path, "wb") as handle:
    plistlib.dump(payload, handle)
print(plist_path)
PLISTEOF
    launchctl bootout "gui/$(id -u)" "$plist" >/dev/null 2>&1 || true
    launchctl bootstrap "gui/$(id -u)" "$plist"
    launchctl kickstart "gui/$(id -u)/${label}" >/dev/null 2>&1 || true
    echo "started worker unit '${label}' via launchd (join pending)"
    echo "  log: $log_dir/${name}.log · stop: launchctl bootout gui/\$(id -u) $plist"
    return
  fi
  if command -v pm2 >/dev/null 2>&1; then
    pm2 delete "$name" >/dev/null 2>&1 || true
    # pm2 keeps a deleted unit's log files; without a flush, confirm_joins
    # reads a PREVIOUS run's join verdict after the operator fixes the flags.
    pm2 flush "$name" >/dev/null 2>&1 || true
    # pm2 snapshots the current env (LD_LIBRARY_PATH, HF_HUB_DISABLE_XET and
    # the per-unit CUDA_VISIBLE_DEVICES below). The worker retries manager and
    # network failures internally. A fatal top-level error must stay stopped
    # so PM2 cannot reset the in-process crash breaker and orphan an unbounded
    # sequence of GPU child sessions.
    CUDA_VISIBLE_DEVICES="$cuda_visible" pm2 start "$VENV/bin/python" \
      --name "$name" --interpreter none \
      --no-autorestart -- -m neurons.cli mesh pool worker "$@"
    echo "started worker unit '$name' (join pending); pm2 logs $name"
  else
    # Unreachable on a machine this installer set up (pm2 is installed and
    # preflighted above). Printing a command line here used to leave the box
    # "joined" with nothing supervising it, and the env in that line is easy
    # to get wrong, so refuse instead.
    _die "pm2 is not available to supervise unit '$name'"
  fi
}

if [ "$SINGLE_UNIT" -eq 1 ]; then
  # Legacy path: one unit spanning all GPUs, exact given id and ports.
  start_unit "verathos-mesh-$WORKER_ID" "${CUDA_VISIBLE_DEVICES-}" \
    --workdir "$HOME/.verathos/poolwork" \
    --rpc-port "$RPC_PORT" --proof-port "$PROOF_PORT" --mesh-port "$MESH_PORT" \
    --rpc-device "$RPC_DEV" --catalog "$CATALOG" \
    --worker-id "$WORKER_ID" --gpu-name "$GPU_NAME" --vram-gb "$VRAM_GB" \
    "${COMMON[@]}"
else
  # One worker unit per GPU. All derivation (ids, ports, workdirs, masks)
  # lives in verallm/mesh/units.py; this loop only renders it.
  UNITS_FILE="$HOME/.verathos/planned-units.json"
  PLAN=("$VENV/bin/python" -m neurons.cli mesh plan-units
    --worker-id-base "$WORKER_ID" --backend "$BACKEND"
    --rpc-port "$RPC_PORT" --proof-port "$PROOF_PORT" --mesh-port "$MESH_PORT")
  [ -z "$GPUS" ] || PLAN+=(--gpus "$GPUS")
  "${PLAN[@]}" > "$UNITS_FILE"
  UNIT_COUNT=$("$VENV/bin/python" -c \
    "import json,sys; print(len(json.load(open(sys.argv[1]))['units']))" "$UNITS_FILE")
  echo "planning $UNIT_COUNT worker unit(s) from $UNITS_FILE"
  for I in $(seq 0 $((UNIT_COUNT - 1))); do
    eval "$("$VENV/bin/python" - "$UNITS_FILE" "$I" <<'PYEOF'
import json, shlex, sys
unit = json.load(open(sys.argv[1]))["units"][int(sys.argv[2])]
for key in ("worker_id", "pm2_name", "gpu_name", "vram_gb",
            "cuda_visible_devices", "rpc_device", "rpc_port", "proof_port",
            "mesh_port", "workdir", "catalog"):
    print(f"U_{key.upper()}={shlex.quote(str(unit[key]))}")
PYEOF
)"
    mkdir -p "$U_WORKDIR"
    [ -f "$U_CATALOG" ] || echo "[]" > "$U_CATALOG"   # empty = file-less member
    start_unit "$U_PM2_NAME" "$U_CUDA_VISIBLE_DEVICES" \
      --workdir "$U_WORKDIR" \
      --rpc-port "$U_RPC_PORT" --proof-port "$U_PROOF_PORT" --mesh-port "$U_MESH_PORT" \
      --rpc-device "$U_RPC_DEVICE" --catalog "$U_CATALOG" \
      --worker-id "$U_WORKER_ID" --gpu-name "$U_GPU_NAME" --vram-gb "$U_VRAM_GB" \
      "${COMMON[@]}"
  done
  "$VENV/bin/python" -m neurons.cli mesh register-units \
    --units-file "$UNITS_FILE" --token-file "$TOKEN_FILE"
fi
if command -v pm2 >/dev/null 2>&1; then pm2 save; fi
confirm_joins
