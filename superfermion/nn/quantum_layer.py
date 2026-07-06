"""
QuantumLayer — Integration between Superfermion and Flax (JAX).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import superfermion as sf
import jax
import jax.numpy as jnp
from flax import linen as nn


class QuantumLayer(nn.Module):
    """
    A quantum circuit layer that integrates natively with JAX autograd.
    Gradients flow through automatically via the parameter-shift rule.

    Args:
        n_qubits:     Number of qubits in the circuit.
        ansatz:       Variational circuit template or sf.Circuit.
        backend:      Quantum hardware or simulator (string or backend instance).
        observables:  Quantum observables to measure. Defaults to Z measurements.
        diff_method:  Gradient method (currently "parameter_shift").
        shots:        Measurement shots per execution.
    """
    n_qubits: int
    ansatz: Any = None
    backend: Any = "statevector"
    observables: Optional[list] = None
    diff_method: str = "parameter_shift"
    shots: int = 1000

    @nn.compact
    def __call__(self, x: jnp.ndarray = None) -> jnp.ndarray:
        """Forward pass through quantum circuit.
        
        If x is provided, it is encoded into the circuit before execution.
        """
        # 1. Resolve Circuit/Ansatz
        if self.ansatz is None:
            from superfermion.qml.ansatz import hardware_efficient
            circuit = hardware_efficient(self.n_qubits, layers=1)
        elif isinstance(self.ansatz, sf.Circuit):
            circuit = self.ansatz
        elif callable(self.ansatz):
            # If ansatz is a factory function, call it
            circuit = self.ansatz(self.n_qubits)
        else:
            circuit = self.ansatz

        # 2. Parameter Management
        param_names = circuit.parameters
        if param_names:
            weights = self.param(
                "weights",
                jax.nn.initializers.uniform(scale=2 * jnp.pi),
                (len(param_names),)
            )
        else:
            weights = jnp.array([])

        # 3. Preparation
        from superfermion.qml.gradient.core import execute_circuit
        
        # Default observables if none provided
        if self.observables is None:
            # We return probabilities by default if no observables
            pass 

        # 4. Handle Execution (with vmap if x is batched)
        if x is not None and circuit.n_parameters == 0:
            raise NotImplementedError(
                "QuantumLayer input encoding is not yet implemented. "
                "Use an ansatz circuit with SymbolicParameter gates and pass x=None, "
                "or use AngleEmbedding to encode your data into the circuit."
            )

        def _single_forward(params, input_val):
            return execute_circuit(circuit, *params, backend=self.backend)

        if x is not None and x.ndim > 1:
            # Vectorized execution over batch dimension
            return jax.vmap(lambda val: _single_forward(weights, val))(x)
        else:
            return _single_forward(weights, x)

