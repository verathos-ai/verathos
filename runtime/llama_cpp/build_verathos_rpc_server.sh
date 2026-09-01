#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 /path/to/llama.cpp [build-dir]" >&2
  exit 2
fi

LLAMA_DIR="$(cd "$1" && pwd)"
BUILD_DIR="${2:-$LLAMA_DIR/build-verathos-rpc}"
PATCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ ! -f "$LLAMA_DIR/ggml/src/ggml-rpc/ggml-rpc.cpp" ]]; then
  echo "not a llama.cpp checkout with ggml RPC sources: $LLAMA_DIR" >&2
  exit 2
fi
if [[ ! -f "$LLAMA_DIR/ggml/src/ggml-cpu/ggml-cpu.c" ]]; then
  echo "not a llama.cpp checkout with ggml CPU sources: $LLAMA_DIR" >&2
  exit 2
fi

is_enabled() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

apply_patch_once() {
  local patch_file="$1"
  local source_file="$2"
  local marker="$3"

  if ! git -C "$LLAMA_DIR" apply --recount --check "$patch_file" >/dev/null 2>&1; then
    if ! grep -q "$marker" "$LLAMA_DIR/$source_file"; then
      echo "patch does not apply cleanly and source is not already patched: $patch_file" >&2
      exit 1
    fi
  else
    git -C "$LLAMA_DIR" apply --recount "$patch_file"
  fi
}

apply_patch_once \
  "$PATCH_DIR/verathos_ggml_rpc_trace.patch" \
  "ggml/src/ggml-rpc/ggml-rpc.cpp" \
  "VERATHOS_GGML_RPC_GRAPH_TRACE"
apply_patch_once \
  "$PATCH_DIR/verathos_ggml_cpu_trace.patch" \
  "ggml/src/ggml-cpu/ggml-cpu.c" \
  "VERATHOS_GGML_TRACE_ENABLE_FILE"

cmake_args=(-DGGML_RPC=ON)
if is_enabled "${VERATHOS_BUILD_CUDA:-0}"; then
  if [[ ! -f "$LLAMA_DIR/ggml/src/ggml-cuda/ggml-cuda.cu" ]]; then
    echo "VERATHOS_BUILD_CUDA=1 but llama.cpp has no ggml CUDA source: $LLAMA_DIR" >&2
    exit 2
  fi
  apply_patch_once \
    "$PATCH_DIR/verathos_ggml_cuda_trace.patch" \
    "ggml/src/ggml-cuda/ggml-cuda.cu" \
    "verathos_cuda_try_dump_mul_mat_trace"
  cmake_args+=(-DGGML_CUDA=ON -DGGML_CUDA_GRAPHS=OFF)
  if [[ -n "${VERATHOS_CUDA_ARCHITECTURES:-}" ]]; then
    cmake_args+=("-DCMAKE_CUDA_ARCHITECTURES=${VERATHOS_CUDA_ARCHITECTURES}")
  fi
fi
if is_enabled "${VERATHOS_BUILD_METAL:-0}"; then
  if [[ ! -f "$LLAMA_DIR/ggml/src/ggml-metal/ggml-metal.cpp" ]]; then
    echo "VERATHOS_BUILD_METAL=1 but llama.cpp has no ggml Metal source: $LLAMA_DIR" >&2
    exit 2
  fi
  if [[ ! -f "$LLAMA_DIR/ggml/src/ggml-metal/ggml-metal-context.m" ]]; then
    echo "VERATHOS_BUILD_METAL=1 but llama.cpp has no ggml Metal context source: $LLAMA_DIR" >&2
    exit 2
  fi
  apply_patch_once \
    "$PATCH_DIR/verathos_ggml_metal_trace.patch" \
    "ggml/src/ggml-metal/ggml-metal.cpp" \
    "verathos_metal_try_dump_mul_mat_trace"
  apply_patch_once \
    "$PATCH_DIR/verathos_ggml_metal_snapshot_trace.patch" \
    "ggml/src/ggml-metal/ggml-metal.cpp" \
    "verathos_metal_plan_graph_trace"
  cmake_args+=(-DGGML_METAL=ON)
fi
if is_enabled "${VERATHOS_BUILD_VULKAN:-0}"; then
  if [[ ! -f "$LLAMA_DIR/ggml/src/ggml-vulkan/ggml-vulkan.cpp" ]]; then
    echo "VERATHOS_BUILD_VULKAN=1 but llama.cpp has no ggml Vulkan source: $LLAMA_DIR" >&2
    exit 2
  fi
  apply_patch_once \
    "$PATCH_DIR/verathos_ggml_vulkan_trace.patch" \
    "ggml/src/ggml-vulkan/ggml-vulkan.cpp" \
    "verathos_vk_try_dump_mul_mat_trace"
  cmake_args+=(-DGGML_VULKAN=ON)
fi

if [[ "$(uname -s)" == "Darwin" ]]; then
  osx_architectures="${VERATHOS_CMAKE_OSX_ARCHITECTURES:-}"
  defaulted_osx_architectures=0
  if [[ -z "$osx_architectures" ]] && is_enabled "${VERATHOS_BUILD_METAL:-0}"; then
    if [[ "$(sysctl -n hw.optional.arm64 2>/dev/null || echo 0)" == "1" ]]; then
      osx_architectures="arm64"
      defaulted_osx_architectures=1
    fi
  fi
  if [[ -n "$osx_architectures" ]]; then
    cmake_args+=("-DCMAKE_OSX_ARCHITECTURES=${osx_architectures}")
  fi
  if is_enabled "${VERATHOS_BUILD_METAL:-0}" && [[ "$osx_architectures" == *"arm64"* ]]; then
    cmake_args+=("-DGGML_NATIVE=${VERATHOS_GGML_NATIVE:-OFF}")
  fi
  if [[ "$defaulted_osx_architectures" == "1" && "$(arch 2>/dev/null || echo unknown)" != "arm64" ]]; then
    cmake_args+=(-DGGML_NATIVE=OFF)
  fi
fi

build_targets=(rpc-server)
if is_enabled "${VERATHOS_BUILD_LLAMA_SERVER:-0}"; then
  if [[ ! -f "$LLAMA_DIR/tools/server/server-task.cpp" ]]; then
    echo "VERATHOS_BUILD_LLAMA_SERVER=1 but llama.cpp has no llama-server sources: $LLAMA_DIR" >&2
    exit 2
  fi
  apply_patch_once \
    "$PATCH_DIR/verathos_llama_server_tokens.patch" \
    "tools/server/server-task.cpp" \
    "verathos_generated_tokens"
  apply_patch_once \
    "$PATCH_DIR/verathos_llama_server_batch_trace.patch" \
    "tools/server/server-context.cpp" \
    "VERATHOS_LLAMA_BATCH_V1"
  build_targets+=(llama-server)
fi

cmake -S "$LLAMA_DIR" -B "$BUILD_DIR" "${cmake_args[@]}"
cmake --build "$BUILD_DIR" --target "${build_targets[@]}" -j "${VERATHOS_BUILD_JOBS:-4}"

rpc_server="$BUILD_DIR/bin/rpc-server"
rpc_suffix=""
if [[ ! -f "$rpc_server" && -f "$rpc_server.exe" ]]; then
  rpc_server="$rpc_server.exe"
  rpc_suffix=".exe"
fi
if [[ ! -f "$rpc_server" ]]; then
  echo "built rpc-server was not found under $BUILD_DIR/bin" >&2
  exit 1
fi
cp "$rpc_server" "$BUILD_DIR/bin/verathos-rpc-server$rpc_suffix"
echo "$BUILD_DIR/bin/verathos-rpc-server$rpc_suffix"
