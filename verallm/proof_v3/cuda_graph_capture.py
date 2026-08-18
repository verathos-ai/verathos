"""CUDA-graph capture policy for proof work beside live inference."""

from __future__ import annotations

__all__ = ["proof_cuda_graph_capture_v3"]


def proof_cuda_graph_capture_v3(graph):
    """Capture proof kernels without globally blocking vLLM's CUDA thread.

    Post-nonce proof construction runs on a worker thread while inference
    that was admitted before the nonce continues on vLLM's serving thread.
    PyTorch's default ``global`` capture error mode makes unrelated event
    synchronization on that serving thread fail with
    ``operation not permitted when stream is capturing``.  Every operation
    recorded by the proof graph is issued by the current worker thread, so
    ``thread_local`` retains fail-closed capture checking for the prover while
    permitting the independent serving thread to finish its existing work.
    """

    import torch

    return torch.cuda.graph(
        graph,
        capture_error_mode="thread_local",
    )
