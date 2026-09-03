#!/usr/bin/env bash
# Build patched llama.cpp (llama-server + verathos-rpc-server) for the mesh.
# Clones upstream at the pinned base, applies the platform patch, builds.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="$(cat "$HERE/UPSTREAM_BASE.txt")"
BACKEND="cuda"
SRC="${LLAMA_SRC:-$HOME/.cache/verathos-mesh-runtime/llama.cpp-build}"
CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES:-}"
while [ $# -gt 0 ]; do case "$1" in
  --backend) BACKEND="$2"; shift 2;;
  --src) SRC="$2"; shift 2;;
  --cuda-architectures) CUDA_ARCHITECTURES="$2"; shift 2;;
  *) echo "unknown arg: $1" >&2; exit 2;;
esac; done

_resolve_nvcc() {
  # This script is invoked both by join_pool.sh (which exports the torch-
  # matched toolkit PATH) and DIRECTLY by the pool auto-rebuild whenever the
  # patch stack changes. The direct path must find nvcc itself: driver-only
  # pod images ship /usr/local/cuda as headers without a compiler, and the
  # real toolkit lives in a versioned /usr/local/cuda-X.Y prefix off PATH.
  if [ -n "${CUDACXX:-}" ] && [ -x "${CUDACXX}" ]; then
    return 0
  fi
  if command -v nvcc >/dev/null 2>&1; then
    export CUDACXX="$(command -v nvcc)"
    return 0
  fi
  local best=""
  # Highest toolkit version wins (version-sort the PREFIX dirs, not the
  # nvcc paths: a trailing /bin/nvcc corrupts sort -V ordering). The plain
  # /usr/local/cuda symlink is checked first so any versioned prefix
  # overrides it.
  for prefix in /usr/local/cuda $(ls -d /usr/local/cuda-* 2>/dev/null | sort -V); do
    [ -x "$prefix/bin/nvcc" ] || continue
    best="$prefix/bin/nvcc"
  done
  if [ -n "$best" ]; then
    export CUDACXX="$best"
    export PATH="$(dirname "$best"):$PATH"
    echo "using nvcc: $best"
    return 0
  fi
  echo "no nvcc found (PATH, /usr/local/cuda*/bin); install the CUDA toolkit" >&2
  echo "matching your torch build (scripts/join_pool.sh does this automatically)" >&2
  exit 2
}

case "$BACKEND" in
  cuda)
    _resolve_nvcc
    PATCH="$HERE/0001-verathos-proof-capture-cuda-cpu-rpc-server.patch"
    # Streaming execution anchors ride on top of the capture patch. They are
    # inert unless VERATHOS_GGML_ANCHOR_TENSORS is set at runtime, so every
    # build carries them and no operator has to choose a variant.
    EXTRA_PATCHES=(
      "$HERE/0003-verathos-streaming-execution-anchors.patch"
      "$HERE/0005-verathos-mmid-decode-intra-parity.patch"
      "$HERE/0007-verathos-rpc-foreign-view-serialization.patch"
      "$HERE/0008-verathos-wildcard-op-arming.patch"
      "$HERE/0009-verathos-name-keyed-op-arming.patch"
      "$HERE/0010-verathos-decode-priority-interleave.patch"
      "$HERE/0011-verathos-prefill-fairness.patch"
    )
    if [ -z "$CUDA_ARCHITECTURES" ]; then
      CUDA_CAPABILITY="$(
        nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null |
          head -n 1 | tr -d '[:space:].' || true
      )"
      if ! [[ "$CUDA_CAPABILITY" =~ ^[0-9]+$ ]]; then
        echo "cannot detect the CUDA compute capability; pass --cuda-architectures (for example 89)" >&2
        exit 2
      fi
      CUDA_ARCHITECTURES="$CUDA_CAPABILITY"
    fi
    CMAKE_FLAGS=(
      -DGGML_CUDA=ON
      "-DCMAKE_CUDA_ARCHITECTURES=$CUDA_ARCHITECTURES"
    )
    ;;
  metal)
    # Metal builds carry the FULL Verathos stack: the 0001-rooted series
    # provides the cpu/rpc trace hooks and every tools/server change (batch
    # records, eager prompt tail, decode checkpoints) exactly as CUDA builds
    # get them; 0002 is a pure Metal backend ADD-ON applied on top. Without
    # 0001 the Metal llama-server has no capture support at all and a Mac
    # worker's mesh serve fails at trace-capture arming.
    PATCH="$HERE/0001-verathos-proof-capture-cuda-cpu-rpc-server.patch"
    EXTRA_PATCHES=(
      "$HERE/0003-verathos-streaming-execution-anchors.patch"
      "$HERE/0005-verathos-mmid-decode-intra-parity.patch"
      "$HERE/0007-verathos-rpc-foreign-view-serialization.patch"
      "$HERE/0008-verathos-wildcard-op-arming.patch"
      "$HERE/0009-verathos-name-keyed-op-arming.patch"
      "$HERE/0002-verathos-proof-capture-metal.patch"
      "$HERE/0010-verathos-decode-priority-interleave.patch"
      "$HERE/0011-verathos-prefill-fairness.patch"
    )
    CMAKE_FLAGS=(-DGGML_METAL=ON)
    ;;
  *) echo "backend must be cuda|metal" >&2; exit 2;;
esac

MANAGED_MARKER="$SRC/.git/verathos-mesh-managed"
if [ ! -d "$SRC/.git" ]; then
  if ! git clone https://github.com/ggml-org/llama.cpp "$SRC"; then
    echo "initial llama.cpp clone failed; retrying with Git HTTP/1.1" >&2
    git -c http.version=HTTP/1.1 clone \
      https://github.com/ggml-org/llama.cpp "$SRC"
  fi
  printf 'managed by patches/llama.cpp/build.sh\n' > "$MANAGED_MARKER"
elif [ ! -f "$MANAGED_MARKER" ]; then
  echo "refusing to reset unmanaged llama.cpp checkout: $SRC" >&2
  echo "choose an empty --src directory; this builder force-checkouts and cleans only its own managed cache" >&2
  exit 2
fi
cd "$SRC"
git fetch --depth 1 origin "$BASE" 2>/dev/null || \
  git -c http.version=HTTP/1.1 fetch --depth 1 origin "$BASE" 2>/dev/null || \
  git fetch origin || \
  git -c http.version=HTTP/1.1 fetch origin
git checkout -f "$BASE"
git clean -fdx >/dev/null 2>&1 || true
git apply --check "$PATCH" && git apply "$PATCH"
for extra in ${EXTRA_PATCHES+"${EXTRA_PATCHES[@]}"}; do
  git apply --check "$extra" && git apply "$extra"
done
BUILD="build-verathos-$BACKEND"
# GGML_RPC=ON is what creates the rpc-server target at all; without it the
# build stops at "No rule to make target 'rpc-server'" (mesh members ARE
# rpc servers, so this flag is not optional).
cmake -B "$BUILD" "${CMAKE_FLAGS[@]}" -DGGML_RPC=ON -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_SERVER=ON
# upstream renamed the target rpc-server -> ggml-rpc-server; support both by
# trying the new name first (target-help grepping proved unreliable on macOS).
DETECTED_JOBS="$(nproc 2>/dev/null || sysctl -n hw.ncpu)"
if [ -n "${VERATHOS_BUILD_JOBS:-}" ]; then
  JOBS="$VERATHOS_BUILD_JOBS"
else
  JOBS="$DETECTED_JOBS"
  case "$JOBS" in
    ''|*[!0-9]*|0)
      echo "could not detect a positive build job count" >&2
      exit 2
      ;;
  esac
  # Desktop workstations can expose dozens of logical cores; compiling every
  # C++/CUDA translation unit concurrently can exhaust RAM and destabilize the
  # operator machine. Operators may opt in to a different explicit value.
  [ "$JOBS" -le 8 ] || JOBS=8
fi
case "$JOBS" in
  ''|*[!0-9]*|0)
    echo "VERATHOS_BUILD_JOBS must be a positive integer" >&2
    exit 2
    ;;
esac
cmake --build "$BUILD" --target llama-server -j "$JOBS"
RPC_TARGET=ggml-rpc-server
cmake --build "$BUILD" --target "$RPC_TARGET" -j "$JOBS" || { RPC_TARGET=rpc-server; cmake --build "$BUILD" --target "$RPC_TARGET" -j "$JOBS"; }
cp "$BUILD/bin/$RPC_TARGET" "$BUILD/bin/verathos-rpc-server"
echo "BUILT: $SRC/$BUILD/bin/{llama-server,verathos-rpc-server}"
