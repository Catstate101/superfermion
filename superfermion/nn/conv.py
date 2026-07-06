"""
Quantum Convolutional Neural Networks (QCNN) — Layers for spatial data.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
from flax import linen as nn
import superfermion as sf
from superfermion.nn.quantum_layer import QuantumLayer

class Conv(nn.Module):
    """Hybrid Quantum-Classical Convolution.
    
    A standard convolution followed by an optional quantum transformation.
    """
    features: int
    kernel_size: Union[int, Sequence[int]]
    strides: Union[int, Sequence[int]] = 1
    padding: str = "SAME"
    n_qubits: int = 4
    ansatz: Any = None
    backend: str = "jax"
    use_bias: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # 1. Classical Convolution (Feature extraction)
        # Using Flax's standard Conv
        x_conv = nn.Conv(
            features=self.features,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            use_bias=self.use_bias,
            name="classical_conv"
        )(x)
        
        # 2. Quantum Transformation (optional nonlinear refinement)
        # We project features to qubits, apply quantum layer, and project back.
        original_shape = x_conv.shape
        batch_size = original_shape[0]
        # Flatten spatial dimensions for quantum layer processing if needed
        # Or apply point-wise (1x1) quantum filter
        
        # To make it efficient, we process across the spatial dims as a batch
        # shape (batch, spatial_dim_1, ..., spatial_dim_n, features)
        x_flat = x_conv.reshape(-1, self.features)
        
        # Project to qubits
        x_proj = nn.Dense(self.n_qubits, name="feat_to_qubit")(x_flat)
        
        # Quantum Layer
        q_out = QuantumLayer(
            n_qubits=self.n_qubits,
            ansatz=self.ansatz,
            backend=self.backend
        )(x_proj)
        
        # Project back to features
        x_out_flat = nn.Dense(self.features, name="qubit_to_feat")(q_out)
        
        # Reshape back to original
        return x_out_flat.reshape(original_shape)

class Conv1D(Conv):
    """Quantum-Classical 1D Convolution."""
    kernel_size: int = 3

class Conv2D(Conv):
    """Quantum-Classical 2D Convolution."""
    kernel_size: Tuple[int, int] = (3, 3)

class Conv3D(Conv):
    """Quantum-Classical 3D Convolution."""
    kernel_size: Tuple[int, int, int] = (3, 3, 3)
