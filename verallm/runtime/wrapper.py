"""
Verifiable Runtime Wrapper.

Wraps any PyTorch model to capture activations and generate
commitments during inference. No model code changes required.
"""

from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
import time

import torch
import torch.nn as nn

from verallm.config import Config, get_config
from verallm.types import InferenceCommitment
from verallm.commitment.tensor import commit_tensor
from verallm.commitment.model import ModelCommitment
from verallm.runtime.witness import WitnessStore, LayerWitness


@dataclass
class HookConfig:
    """Configuration for which modules to hook."""

    # Layer patterns to capture (fnmatch style)
    capture_patterns: List[str]

    # Whether to capture inputs (in addition to outputs)
    capture_inputs: bool = True

    # Whether to capture detailed op-level witnesses
    capture_op_details: bool = False


def default_hook_config() -> HookConfig:
    """Default hook configuration for transformer models."""
    return HookConfig(
        capture_patterns=[
            # LLaMA, Mistral, Qwen style
            "*.layers.*",
            "model.layers.*",
            # GPT-2, GPT-J, GPT-Neo style
            "transformer.h.*",
            "*.h.*",
            # Encoder-decoder models
            "*.decoder.layers.*",
            "*.encoder.layers.*",
            # BERT style
            "*.layer.*",
            "encoder.layer.*",
        ],
        capture_inputs=True,
        capture_op_details=False,
    )


class VerifiableRuntime:
    """
    Wraps a PyTorch model for verifiable inference.

    Usage:
        model = load_model(...)
        model_commit = create_model_commitment(model, "my-model")
        runtime = VerifiableRuntime(model, model_commit)

        output, commitment = runtime.generate(input_ids)
        # commitment can be submitted for verification

        # If challenged:
        proofs = runtime.generate_proofs(commitment, beacon)
    """

    def __init__(
        self,
        model: nn.Module,
        model_commitment: ModelCommitment,
        config: Optional[Config] = None,
        hook_config: Optional[HookConfig] = None,
    ):
        """
        Initialize verifiable runtime.

        Args:
            model: PyTorch model to wrap
            model_commitment: Pre-computed model weight commitment
            config: VeraLLM configuration
            hook_config: Hook configuration (which layers to capture)
        """
        self.model = model
        self.model_commitment = model_commitment
        self.config = config or get_config()
        self.hook_config = hook_config or default_hook_config()

        # Witness storage
        self.witness_store = WitnessStore(
            retention_sec=self.config.witness_retention_sec,
            max_memory_gb=self.config.max_witness_memory_gb,
        )

        # Current inference state
        self._current_session: Optional[str] = None
        self._layer_commitments: List[bytes] = []
        self._layer_counter = 0

        # Hook handles (for cleanup)
        self._hook_handles: List[torch.utils.hooks.RemovableHandle] = []

        # Install hooks
        self._install_hooks()

    def _install_hooks(self) -> None:
        """Install forward hooks on relevant modules."""
        import fnmatch

        for name, module in self.model.named_modules():
            # Check if this module matches any capture pattern
            should_capture = any(
                fnmatch.fnmatch(name, pattern) for pattern in self.hook_config.capture_patterns
            )

            if should_capture:
                # Create hook for this module
                hook = self._make_capture_hook(name)
                handle = module.register_forward_hook(hook)
                self._hook_handles.append(handle)

    def _make_capture_hook(self, layer_name: str) -> Callable:
        """Create a forward hook for a layer."""

        def hook(
            module: nn.Module,
            input: Tuple[torch.Tensor, ...],
            output: torch.Tensor,
        ) -> None:
            if self._current_session is None:
                return

            # Get layer index
            layer_idx = self._layer_counter
            self._layer_counter += 1

            # Get input tensor (first argument typically)
            input_tensor = input[0] if isinstance(input, tuple) and len(input) > 0 else None

            # Get output tensor
            output_tensor = output if isinstance(output, torch.Tensor) else None
            if output_tensor is None and isinstance(output, tuple):
                output_tensor = output[0] if len(output) > 0 else None

            # Store witness
            if input_tensor is not None or output_tensor is not None:
                # Detach to avoid holding computation graph
                input_detached = input_tensor.detach() if input_tensor is not None else None
                output_detached = output_tensor.detach() if output_tensor is not None else None

                self.witness_store.store_layer_witness(
                    layer_idx=layer_idx,
                    layer_name=layer_name,
                    input_activation=input_detached if self.hook_config.capture_inputs else None,
                    output_activation=output_detached,
                    session_id=self._current_session,
                )

                # Build commitment for this layer's output
                if output_detached is not None:
                    commitment = commit_tensor(output_detached)
                    self._layer_commitments.append(commitment)

        return hook

    def _remove_hooks(self) -> None:
        """Remove all installed hooks."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()

    def generate(
        self,
        input_ids: torch.Tensor,
        **generation_kwargs,
    ) -> Tuple[torch.Tensor, InferenceCommitment]:
        """
        Run inference with commitment generation.

        Args:
            input_ids: Input token IDs
            **generation_kwargs: Arguments passed to model.generate()

        Returns:
            (output_ids, commitment)
        """
        # Start new session
        self._current_session = self.witness_store.new_session()
        self._layer_commitments = []
        self._layer_counter = 0

        # Commit to input
        input_commitment = commit_tensor(input_ids)

        # Run inference
        with torch.no_grad():
            # Check if model has generate method (causal LM)
            if hasattr(self.model, "generate"):
                output = self.model.generate(input_ids, **generation_kwargs)
            else:
                # Direct forward pass
                output = self.model(input_ids)
                if isinstance(output, tuple):
                    output = output[0]

        # Commit to output
        output_commitment = commit_tensor(output)

        # Finalize session
        self.witness_store.finalize_session(self._current_session)

        # Build inference commitment
        commitment = InferenceCommitment(
            session_id=self._current_session,
            model_id=self.model_commitment.model_id,
            model_commitment=self.model_commitment.root,
            input_commitment=input_commitment,
            output_commitment=output_commitment,
            layer_commitments=list(self._layer_commitments),
            timestamp=time.time(),
        )

        session_id = self._current_session
        self._current_session = None

        return output, commitment

    def forward(
        self,
        input_tensor: torch.Tensor,
        **forward_kwargs,
    ) -> Tuple[torch.Tensor, InferenceCommitment]:
        """
        Run forward pass with commitment generation.

        For models without .generate() method.

        Args:
            input_tensor: Input tensor
            **forward_kwargs: Arguments passed to model.forward()

        Returns:
            (output, commitment)
        """
        # Start new session
        self._current_session = self.witness_store.new_session()
        self._layer_commitments = []
        self._layer_counter = 0

        # Commit to input
        input_commitment = commit_tensor(input_tensor)

        # Run forward pass
        with torch.no_grad():
            output = self.model(input_tensor, **forward_kwargs)
            if isinstance(output, tuple):
                output = output[0]

        # Commit to output
        output_commitment = commit_tensor(output)

        # Finalize session
        self.witness_store.finalize_session(self._current_session)

        # Build inference commitment
        commitment = InferenceCommitment(
            session_id=self._current_session,
            model_id=self.model_commitment.model_id,
            model_commitment=self.model_commitment.root,
            input_commitment=input_commitment,
            output_commitment=output_commitment,
            layer_commitments=list(self._layer_commitments),
            timestamp=time.time(),
        )

        session_id = self._current_session
        self._current_session = None

        return output, commitment

    def get_layer_witnesses(
        self,
        session_id: str,
        layer_indices: List[int],
    ) -> Dict[int, LayerWitness]:
        """
        Get witnesses for specific layers.

        Used when generating proofs for challenged layers.

        Args:
            session_id: Inference session ID
            layer_indices: Which layers to get

        Returns:
            Dict mapping layer index to witness
        """
        result = {}
        for idx in layer_indices:
            witness = self.witness_store.get_layer_witness(session_id, idx)
            if witness is not None:
                result[idx] = witness
        return result

    def __del__(self):
        """Cleanup hooks on deletion."""
        self._remove_hooks()


def wrap_model(
    model: nn.Module,
    model_id: str,
    config: Optional[Config] = None,
) -> VerifiableRuntime:
    """
    Convenience function to wrap a model for verifiable inference.

    Args:
        model: PyTorch model
        model_id: Model identifier
        config: Optional configuration

    Returns:
        VerifiableRuntime instance
    """
    from verallm.commitment.model import create_model_commitment

    # Create model commitment
    model_commit = create_model_commitment(model, model_id, include_tree=True)

    # Create runtime
    return VerifiableRuntime(model, model_commit, config)
