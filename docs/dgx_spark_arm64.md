# NVIDIA DGX Spark / ARM64 Compatibility

DGX Spark systems use an ARM64/aarch64 CPU with an NVIDIA GB10 Blackwell GPU and
unified memory. They cannot load Linux x86_64 Python extension modules or wheels.

The miner setup now checks native artifacts before installing large dependencies
so unsupported ARM64 systems fail early with an actionable error instead of
partially installing vLLM, PyTorch, and Bittensor.

## Current blocker

The public repository currently ships only x86_64 native artifacts:

- `dist/zkllm-*-linux_x86_64.whl`
- `verallm/miner/*.cpython-*-x86_64-linux-gnu.so`
- `verallm/moe/*.cpython-*-x86_64-linux-gnu.so`
- `verallm/prover/*.cpython-*-x86_64-linux-gnu.so`
- `verallm/verifier/*.cpython-*-x86_64-linux-gnu.so`

On DGX Spark, setup requires matching aarch64 artifacts for the active Python
ABI, for example Python 3.12:

- `dist/zkllm-<version>-cp312-cp312-linux_aarch64.whl`
- `verallm/miner/*.cpython-312-aarch64-linux-gnu.so`
- `verallm/moe/*.cpython-312-aarch64-linux-gnu.so`
- `verallm/prover/*.cpython-312-aarch64-linux-gnu.so`
- `verallm/verifier/*.cpython-312-aarch64-linux-gnu.so`

Equivalent `manylinux_*_aarch64` wheels are also acceptable when they are built
against a compatible CUDA/PyTorch stack.

## Maintainer release checklist

1. Build and publish `zkllm` wheels for Linux aarch64 for each supported Python
   ABI (`cp310`, `cp311`, `cp312`).
2. Build and publish or source-build the `verallm` native extension packages for
   Linux aarch64.
3. Validate the CUDA extension on DGX Spark / GB10:

   ```bash
   python - <<'PY'
   import torch
   from zkllm.cuda import zkllm_native, HAS_CUDA

   assert torch.cuda.is_available()
   assert torch.cuda.get_device_capability(0) == (12, 1)
   assert zkllm_native is not None
   assert HAS_CUDA
   assert hasattr(zkllm_native, "cuda_blake3_merkle_leaves")
   print("DGX Spark CUDA extension OK")
   PY
   ```

4. Validate the miner setup on a clean DGX Spark:

   ```bash
   bash scripts/setup_miner.sh
   python -m verallm.registry --recommend --verified-only
   ```

5. Start the miner only after the environment validates and the operator has
   completed wallet registration and EVM funding intentionally.

## Notes

- x86_64 wheels must not be installed via emulation for CUDA workloads.
- DGX Spark reports unified memory differently from discrete GPUs. If
  `nvidia-smi --query-gpu=memory.total` returns `N/A`, setup may need to fall
  back to system memory checks for GB10-specific recommendations.
- vLLM may still require a source rebuild for `sm_121` kernels after ARM64
  Python artifacts are available.
