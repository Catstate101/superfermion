"""
TFQuantumLayer — TensorFlow/Keras ↔ Superfermion bridge.

Wraps a parameterized quantum circuit as a ``tf.keras.layers.Layer``.
Gradients flow via ``tf.custom_gradient`` calling ``sf.State.grad()``.

Requires: ``pip install tensorflow``
"""

from __future__ import annotations

from typing import Any, List, Optional

import numpy as np

try:
    import tensorflow as tf
except ImportError:
    raise ImportError(
        "TensorFlow is required for TFQuantumLayer. "
        "Install with: pip install tensorflow"
    )

import superfermion as sf

_PAULI_MAP = {'I': 0, 'X': 1, 'Y': 2, 'Z': 3}


def _obs_to_rust(observable) -> list:
    """Convert a SF observable to the Rust [(paulis, re, im)] format.

    Python Pauli strings use MSB-first ordering (e.g. "ZI" = Z on last qubit),
    while Rust expects LSB-indexed (paulis[0] = qubit 0), so we reverse.
    """
    from superfermion.observables.core import PauliString, SparsePauliOp, Hamiltonian
    terms = []
    if isinstance(observable, PauliString):
        paulis = [_PAULI_MAP[c] for c in reversed(observable.pauli_str)]
        terms.append((paulis, float(observable.coeffs.real), float(observable.coeffs.imag)))
    elif isinstance(observable, SparsePauliOp):
        for ps, coeff in observable._terms:
            paulis = [_PAULI_MAP[c] for c in reversed(ps)]
            terms.append((paulis, float(complex(coeff).real), float(complex(coeff).imag)))
    elif isinstance(observable, Hamiltonian):
        for t in observable.terms:
            paulis = [_PAULI_MAP[c] for c in reversed(t.pauli_str)]
            terms.append((paulis, float(t.coeffs.real), float(t.coeffs.imag)))
    elif isinstance(observable, list):
        return observable
    else:
        raise TypeError(f"Unsupported observable type: {type(observable)}")
    return terms


class TFQuantumLayer(tf.keras.layers.Layer):
    """Keras layer wrapping a Superfermion variational quantum circuit.

    Args:
        circuit: Parameterized ``sf.Circuit``.
        observable: Observable for expectation value measurement.
        device: Simulation device (``"cpu"`` or ``"gpu"``).
        method: Simulation method.
    """

    def __init__(
        self,
        circuit: sf.Circuit,
        observable,
        device: str = "cpu",
        method: str = "statevector",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._circuit = circuit
        self._observable = observable
        self._rust_obs = _obs_to_rust(observable)
        self._device = device
        self._method = method
        self._param_names = list(circuit.parameters) if circuit.parameters else []

    def build(self, input_shape):
        n_params = len(self._param_names)
        if n_params > 0:
            self.weights_var = self.add_weight(
                name="quantum_weights",
                shape=(n_params,),
                initializer=tf.keras.initializers.RandomUniform(0, 2 * np.pi),
                trainable=True,
            )
        else:
            self.weights_var = None

    def call(self, inputs=None, training=None):
        @tf.custom_gradient
        def _quantum_forward(w):
            w_np = w.numpy()
            p_dict = dict(zip(self._param_names, w_np.tolist()))
            bound = self._circuit.bind(p_dict)
            state = sf.simulate(bound, device=self._device, method=self._method)
            val = state.expectation(self._rust_obs)

            def _grad(upstream, variables=None):
                dag = self._circuit.to_ir()
                grads = state.grad(self._rust_obs, dag, p_dict)
                grad_np = np.array([grads.get(k, 0.0) for k in self._param_names])
                grad_tensor = upstream * tf.constant(grad_np, dtype=w.dtype)
                if variables is not None and variables:
                    # tf.custom_gradient requires gradients for watched
                    # variables in addition to the function inputs.
                    return grad_tensor, [grad_tensor for _ in variables]
                return grad_tensor

            return tf.constant(val, dtype=w.dtype), _grad

        if self.weights_var is not None:
            return _quantum_forward(self.weights_var)
        p_dict = {}
        bound = self._circuit.bind(p_dict) if self._param_names else self._circuit
        state = sf.simulate(bound, device=self._device, method=self._method)
        return tf.constant(state.expectation(self._rust_obs), dtype=tf.float32)
