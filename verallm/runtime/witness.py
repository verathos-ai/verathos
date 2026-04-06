"""
Witness storage for verifiable inference.

Manages storage of intermediate values needed for proof generation.
Implements tiered storage with automatic garbage collection.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import time
import threading
import uuid

import torch


@dataclass
class OpWitness:
    """Witness for a single operation."""

    op_name: str
    op_type: str  # "gemm", "softmax", "gelu", "silu", "rmsnorm", "layernorm"
    input_tensor: Optional[torch.Tensor] = None
    output_tensor: Optional[torch.Tensor] = None
    weight_tensor: Optional[torch.Tensor] = None  # For GEMM
    extra: Dict[str, Any] = field(default_factory=dict)  # gamma, beta, etc.


@dataclass
class LayerWitness:
    """Witness for a complete layer."""

    layer_idx: int
    layer_name: str
    input_activation: Optional[torch.Tensor] = None
    output_activation: Optional[torch.Tensor] = None
    input_commitment: Optional[bytes] = None
    output_commitment: Optional[bytes] = None
    op_witnesses: List[OpWitness] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


@dataclass
class InferenceSession:
    """A single inference session's witnesses."""

    session_id: str
    timestamp: float
    layer_witnesses: Dict[int, LayerWitness]  # layer_idx -> witness
    input_tensor: Optional[torch.Tensor] = None
    output_tensor: Optional[torch.Tensor] = None
    finalized: bool = False


class WitnessStore:
    """
    Tiered witness storage with garbage collection.

    Tier 1: Commitments only (kept forever)
    Tier 2: Full activations (kept for retention_sec)
    Tier 3: Op-level witnesses (regenerated on demand)
    """

    def __init__(
        self,
        retention_sec: float = 60.0,
        max_memory_gb: float = 4.0,
    ):
        """
        Initialize witness store.

        Args:
            retention_sec: How long to keep full witnesses (seconds)
            max_memory_gb: Maximum memory budget for witnesses
        """
        self.retention_sec = retention_sec
        self.max_memory_bytes = int(max_memory_gb * 1024 * 1024 * 1024)

        self._sessions: Dict[str, InferenceSession] = {}
        self._lock = threading.Lock()
        self._current_session: Optional[str] = None

        # Memory tracking (approximate)
        self._current_memory = 0

    def new_session(self) -> str:
        """Start a new inference session."""
        session_id = str(uuid.uuid4())

        with self._lock:
            # Garbage collect old sessions first
            self._gc_old_sessions()

            session = InferenceSession(
                session_id=session_id,
                timestamp=time.time(),
                layer_witnesses={},
            )
            self._sessions[session_id] = session
            self._current_session = session_id

        return session_id

    def finalize_session(self, session_id: Optional[str] = None) -> None:
        """Mark session as complete."""
        sid = session_id or self._current_session
        if sid is None:
            return

        with self._lock:
            if sid in self._sessions:
                self._sessions[sid].finalized = True

    def store_layer_witness(
        self,
        layer_idx: int,
        layer_name: str,
        input_activation: Optional[torch.Tensor] = None,
        output_activation: Optional[torch.Tensor] = None,
        session_id: Optional[str] = None,
    ) -> None:
        """
        Store witness for a layer.

        Args:
            layer_idx: Layer index
            layer_name: Layer name
            input_activation: Input tensor (optional, for full witness)
            output_activation: Output tensor (optional, for full witness)
            session_id: Session ID (uses current if not specified)
        """
        sid = session_id or self._current_session
        if sid is None:
            raise RuntimeError("No active session")

        from verallm.commitment.tensor import commit_tensor

        # Compute commitments
        input_commit = commit_tensor(input_activation) if input_activation is not None else None
        output_commit = (
            commit_tensor(output_activation) if output_activation is not None else None
        )

        witness = LayerWitness(
            layer_idx=layer_idx,
            layer_name=layer_name,
            input_activation=input_activation.clone() if input_activation is not None else None,
            output_activation=output_activation.clone() if output_activation is not None else None,
            input_commitment=input_commit,
            output_commitment=output_commit,
        )

        with self._lock:
            if sid not in self._sessions:
                raise RuntimeError(f"Session {sid} not found")

            self._sessions[sid].layer_witnesses[layer_idx] = witness

            # Update memory tracking
            if input_activation is not None:
                self._current_memory += input_activation.numel() * input_activation.element_size()
            if output_activation is not None:
                self._current_memory += output_activation.numel() * output_activation.element_size()

    def store_op_witness(
        self,
        layer_idx: int,
        op_name: str,
        op_type: str,
        input_tensor: Optional[torch.Tensor] = None,
        output_tensor: Optional[torch.Tensor] = None,
        weight_tensor: Optional[torch.Tensor] = None,
        session_id: Optional[str] = None,
        **extra,
    ) -> None:
        """Store witness for a specific operation within a layer."""
        sid = session_id or self._current_session
        if sid is None:
            raise RuntimeError("No active session")

        op_witness = OpWitness(
            op_name=op_name,
            op_type=op_type,
            input_tensor=input_tensor.clone() if input_tensor is not None else None,
            output_tensor=output_tensor.clone() if output_tensor is not None else None,
            weight_tensor=weight_tensor,  # Don't clone weights (they're shared)
            extra=extra,
        )

        with self._lock:
            if sid not in self._sessions:
                raise RuntimeError(f"Session {sid} not found")

            session = self._sessions[sid]
            if layer_idx not in session.layer_witnesses:
                session.layer_witnesses[layer_idx] = LayerWitness(
                    layer_idx=layer_idx,
                    layer_name=f"layer_{layer_idx}",
                )

            session.layer_witnesses[layer_idx].op_witnesses.append(op_witness)

    def get_layer_witness(
        self,
        session_id: str,
        layer_idx: int,
    ) -> Optional[LayerWitness]:
        """Get witness for a layer."""
        with self._lock:
            if session_id not in self._sessions:
                return None
            return self._sessions[session_id].layer_witnesses.get(layer_idx)

    def get_all_layer_commitments(
        self,
        session_id: str,
    ) -> List[bytes]:
        """Get all layer output commitments for a session."""
        with self._lock:
            if session_id not in self._sessions:
                return []

            session = self._sessions[session_id]
            commitments = []

            for layer_idx in sorted(session.layer_witnesses.keys()):
                witness = session.layer_witnesses[layer_idx]
                if witness.output_commitment is not None:
                    commitments.append(witness.output_commitment)

            return commitments

    def _gc_old_sessions(self) -> None:
        """Garbage collect old sessions."""
        now = time.time()
        to_delete = []

        for sid, session in self._sessions.items():
            if session.finalized and (now - session.timestamp) > self.retention_sec:
                to_delete.append(sid)

        for sid in to_delete:
            session = self._sessions[sid]

            # Update memory tracking
            for lw in session.layer_witnesses.values():
                if lw.input_activation is not None:
                    self._current_memory -= (
                        lw.input_activation.numel() * lw.input_activation.element_size()
                    )
                if lw.output_activation is not None:
                    self._current_memory -= (
                        lw.output_activation.numel() * lw.output_activation.element_size()
                    )

            del self._sessions[sid]

    def clear_session(self, session_id: str) -> None:
        """Manually clear a session."""
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]

    @property
    def memory_usage_gb(self) -> float:
        """Current memory usage in GB."""
        return self._current_memory / (1024 * 1024 * 1024)
