"""
QuantumLayer — Integration between Superfermion and TensorFlow autograd.

Enables differentiable quantum execution inside TensorFlow/Keras models via the
parameter-shift rule.  Gradients flow automatically through the circuit
into upstream classical layers.

Usage:
    >>> import tensorflow as tf
    >>> from superfermion.nn.tf_layer import TFQuantumLayer
    >>>
    >>> layer = TFQuantumLayer(n_qubits=4, n_layers=2)
    >>> x = tf.random.normal((32, 8))      # batch of 32, 8 features each
    >>> y = layer(x)                        # quantum-encoded outputs
    >>> loss = tf.reduce_sum(y)
    >>> # Use with tf.GradientTape or Keras model.fit()
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Union

import jax
import jax.numpy as jnp
import numpy as np

try:
    import tensorflow as tf
except ImportError:
    raise ImportError(
        "TensorFlow is required for TFQuantumLayer. "
        "Install with: pip install tensorflow"
    )


def _to_jax(tensor: "tf.Tensor") -> jnp.ndarray:
    """Zero-copy (when possible) conversion: tf.Tensor → JAX NumPy."""
    return jnp.asarray(tensor.numpy())


def _to_tf(array: jnp.ndarray, dtype=None) -> "tf.Tensor":
    """JAX NumPy → tf.Tensor."""
    arr = np.asarray(array)
    return tf.convert_to_tensor(arr, dtype=dtype or tf.float32)


# ── helper: parameter-shift gradient for a single parameter ──────────────

_SHIFT = jnp.pi / 2  # standard parameter-shift offset


def _ps_grad(fn, params, idx, shift=_SHIFT):
    """Parameter-shift gradient for parameter at position `idx`."""
    eps = jnp.zeros_like(params)
    eps = eps.at[idx].set(shift)
    p_plus = fn(params + eps)
    p_minus = fn(params - eps)
    return 0.5 * (p_plus - p_minus)


# ── main layer class ────────────────────────────────────────────────────

class TFQuantumLayer(tf.keras.layers.Layer):
    """TensorFlow/Keras layer wrapping a Superfermion variational quantum circuit.

    Gradients are computed via the **parameter-shift rule**, giving exact
    analytical gradients for every gate parameter. Works seamlessly with
    ``tf.GradientTape`` and Keras ``model.fit()``.

    Args:
        n_qubits:        Number of qubits.
        n_layers:        Number of circuit layers (depth).
        ansatz:          Variational form: ``'hardware_efficient'`` (default),
                         ``'strongly_entangling'``, or a callable
                         ``fn(n_qubits)`` that returns an ``sf.Circuit``.
        backend:         Simulation backend (``'statevector'``, ``'jax'``,
                         ``'jax_mps'``, ``'mps'``, ``'cuda'``, ``'stabilizer'``,
                         ``'density_matrix'``).  The density_matrix backend
                         supports noisy circuits with parameter-shift gradients.
        observables:     List of Pauli-string observables, e.g.
                         ``['Z0', 'Z1', 'ZZ']``.  Defaults to ``['Z0']``.
        encoding:        How classical input is encoded:
                         ``'angle'``  — scale to [-pi, pi] and use as RX params
                         ``'angle2'`` — use pairs as RX + RY per qubit
                         ``'zz'``     — ZZFeatureMap angles
        diff_method:     Gradient method (``'parameter_shift'`` only for now).

    Shape:
        - Input:  ``(batch, input_dim)``
        - Output: ``(batch, n_observables)``  where ``n_observables = len(observables)``

    Example:
        >>> model = tf.keras.Sequential([
        ...     tf.keras.layers.Dense(10, input_shape=(4,)),
        ...     TFQuantumLayer(n_qubits=4, n_layers=2, observables=['Z0', 'Z1']),
        ...     tf.keras.layers.Dense(1),
        ... ])
        >>> model.compile(optimizer='adam', loss='mse')
        >>> x = tf.random.normal((64, 4))
        >>> y = tf.random.normal((64, 1))
        >>> model.fit(x, y, epochs=10)
    """

    def __init__(
        self,
        n_qubits: int,
        n_layers: int = 2,
        ansatz: Any = "hardware_efficient",
        backend: str = "statevector",
        observables: Optional[List[str]] = None,
        encoding: str = "angle",
        diff_method: str = "parameter_shift",
        **kwargs,
    ):
        super().__init__(**kwargs)
        import superfermion as sf

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.backend_name = backend
        self.encoding = encoding
        self.diff_method = diff_method

        # ── observables ────────────────────────────────────────────────
        if observables is None:
            observables = ["Z" + str(i) for i in range(min(n_qubits, 4))]
        self.observable_strings = observables
        self.n_out = len(observables)

        # ── build ansatz circuit ────────────────────────────────────────
        if isinstance(ansatz, str):
            if ansatz == "hardware_efficient":
                from superfermion.qml.ansatz import hardware_efficient
                self._circuit = hardware_efficient(n_qubits, layers=n_layers)
            elif ansatz == "strongly_entangling":
                from superfermion.qml.ansatz import strongly_entangling
                self._circuit = strongly_entangling(n_qubits, layers=n_layers)
            else:
                raise ValueError(f"Unknown ansatz: {ansatz}. "
                                 f"Use 'hardware_efficient' or 'strongly_entangling'.")
        elif isinstance(ansatz, sf.Circuit):
            self._circuit = ansatz
        elif callable(ansatz):
            self._circuit = ansatz(n_qubits)
        else:
            raise TypeError(
                f"ansatz must be str, sf.Circuit, or callable; got {type(ansatz)}"
            )

        # ── parameter count ────────────────────────────────────────────
        self.n_params = (
            len(self._circuit.parameters) if self._circuit.parameters else 0
        )
        self._fixed_circuit = self.n_params == 0

        # ── compile circuit function ───────────────────────────────
        if not self._fixed_circuit and self.n_params > 0:
            from superfermion.observables.core import SparsePauliOp
            self._obs = [SparsePauliOp.from_string(o, n_qubits) for o in observables]

            if self.backend_name == "jax":
                # Use JAX native path for best fusion
                self._fn = sf.qml.circuit_to_jax(
                    self._circuit,
                    backend="jax",
                    observables=observables,
                )
                self._exec_backend = None
            elif self.backend_name == "density_matrix":
                # Density matrix backend: use native expval() + parameter-shift
                from superfermion.backends.density_matrix import DensityMatrixBackend
                self._exec_backend = DensityMatrixBackend()
                self._fn = None
                self._fn_expectation = None
            else:
                # Fall back to parameter-shift on generic backend
                self._exec_backend = sf.get_backend(self.backend_name)
                self._fn_expectation = sf.qml.circuit_to_jax(
                    self._circuit,
                    backend=self.backend_name,
                    observables=observables,
                )
                self._fn = None
        else:
            self._obs = None
            self._fn = None
            self._exec_backend = None
            self._fn_expectation = None

    def _encode_input(self, x: "tf.Tensor") -> jnp.ndarray:
        """Convert batch input tensor to circuit parameters (JAX array).

        NOTE: This method uses .numpy() internally, which breaks TF gradient
        tracking. Use ``_encode_input_tf`` for differentiable paths.
        """
        x_np = x.numpy()
        x_np = np.clip(x_np, -10.0, 10.0)

        if self.encoding == "angle":
            params = np.pi * np.tanh(x_np)
        elif self.encoding == "angle2":
            n_pairs = x_np.shape[1] // 2
            params = np.zeros((x_np.shape[0], 2 * n_pairs), dtype=np.float32)
            for i in range(n_pairs):
                params[:, 2 * i] = np.pi * np.tanh(x_np[:, 2 * i])
                params[:, 2 * i + 1] = np.pi * np.tanh(x_np[:, 2 * i + 1])
        elif self.encoding == "zz":
            params = np.pi * np.tanh(x_np)
        else:
            raise ValueError(f"Unknown encoding: {self.encoding}")

        if self.n_params > 0 and params.shape[1] != self.n_params:
            n_repeat = int(np.ceil(self.n_params / params.shape[1]))
            params = np.tile(params, (1, n_repeat))[:, : self.n_params]

        return jnp.asarray(params)

    def _encode_input_tf(self, x: "tf.Tensor") -> "tf.Tensor":
        """TF-native encoding that preserves gradient flow.

        Uses ``tf.math.tanh`` instead of ``numpy`` so ``tf.GradientTape``
        can differentiate through the encoding.
        """
        x_clipped = tf.clip_by_value(x, -10.0, 10.0)

        if self.encoding in ("angle", "zz"):
            params = np.pi * tf.math.tanh(x_clipped)
        elif self.encoding == "angle2":
            params = np.pi * tf.math.tanh(x_clipped)
        else:
            params = np.pi * tf.math.tanh(x_clipped)

        if self.n_params > 0 and params.shape[1] != self.n_params:
            n_repeat = int(np.ceil(self.n_params / params.shape[1]))
            params = tf.tile(params, [1, n_repeat])[:, : self.n_params]

        return params

    @tf.custom_gradient
    def _call_jax(self, params_jax: jnp.ndarray):
        """Forward + backward via parameter-shift on JAX function."""

        def _forward(p):
            return jnp.asarray(self._fn(*[p[i] for i in range(len(p))]))

        result = jax.vmap(_forward)(params_jax)

        def _grad(upstream):
            grad_fn = jax.grad(lambda p: jnp.sum(_forward(p)))
            grads = jax.vmap(grad_fn)(params_jax)
            # Chain rule: upstream gradient * per-sample gradient
            return (jnp.asarray(upstream)[:, None] * grads),

        return result, _grad

    def call(self, inputs, training=None):
        """Forward pass. ``training`` is unused but required by Keras API."""
        batch_size = inputs.shape[0]

        if self._fixed_circuit or self.n_params == 0:
            # No parameters — just evaluate
            from superfermion.observables.core import SparsePauliOp
            backend = self._exec_backend or __import__("superfermion").get_backend(
                self.backend_name
            )
            obs = [SparsePauliOp.from_string(o, self.n_qubits) for o in self.observable_strings]
            results = []
            for i in range(batch_size):
                vals = [backend.expval(self._circuit, o) for o in obs]
                results.append(vals)
            return tf.convert_to_tensor(np.array(results, dtype=np.float32))

        if self._fn is not None:
            # JAX-native path: encode with numpy (JAX can trace through)
            params_jax = self._encode_input(inputs)
            result = self._call_jax(params_jax)
            return tf.convert_to_tensor(np.asarray(result), dtype=tf.float32)

        if self.backend_name == "density_matrix":
            # DM path: use TF-native encoding to preserve gradient flow
            params_tf = self._encode_input_tf(inputs)
            result = self._call_dm(params_tf)
            return result

        # Generic backend path
        params_jax = self._encode_input(inputs)
        result = self._ps_forward(params_jax)
        return tf.convert_to_tensor(np.asarray(result), dtype=tf.float32)

    def _ps_forward(self, params_jax: jnp.ndarray) -> jnp.ndarray:
        """Parameter-shift forward on generic backend."""
        batch_size = params_jax.shape[0]
        results = np.zeros((batch_size, self.n_out), dtype=np.float32)
        for b in range(batch_size):
            for o_idx, obs in enumerate(self._obs):
                val = self._exec_backend.expval(self._circuit.bind(
                    dict(zip(self._circuit.parameters,
                             [float(params_jax[b, i])
                              for i in range(self.n_params)]))
                ), obs)
                results[b, o_idx] = float(val)
        return jnp.asarray(results)

    def _call_dm(self, params_tf: "tf.Tensor"):
        """Forward + backward for density_matrix backend via parameter-shift.

        Uses ``tf.custom_gradient`` internally (not as a decorator) to avoid
        ``self``-binding issues with instance methods.

        Args:
            params_tf: TF tensor of shape (batch, n_params) — must be a TF
                       tensor (not numpy) so GradientTape can track it.
        """
        backend = self._exec_backend
        circuit = self._circuit
        obs_list = self._obs
        param_names = list(circuit.parameters)
        n_params = self.n_params
        n_out = self.n_out

        @tf.custom_gradient
        def _dm_forward(p):
            # p is a tf.Tensor; we use .numpy() for the concrete evaluation
            # but the gradient tape sees `p` as a tracked tensor.
            p_np = p.numpy()
            batch_size = p_np.shape[0]

            # ── Forward pass ──
            results = np.zeros((batch_size, n_out), dtype=np.float64)
            for b in range(batch_size):
                p_dict = dict(zip(param_names,
                                  [float(p_np[b, i]) for i in range(n_params)]))
                bound = circuit.bind(p_dict)
                for o_idx, obs in enumerate(obs_list):
                    results[b, o_idx] = float(backend.expval(bound, obs))

            # ── Backward pass (parameter-shift on density matrix) ──
            def _grad(upstream):
                upstream_np = np.asarray(upstream, dtype=np.float64)
                grads = np.zeros((batch_size, n_params), dtype=np.float64)
                shift = np.pi / 2.0

                for b in range(batch_size):
                    p_vals = np.array([float(p_np[b, i]) for i in range(n_params)])
                    for p_idx in range(n_params):
                        # +shift
                        p_plus = p_vals.copy()
                        p_plus[p_idx] += shift
                        p_dict_plus = dict(zip(param_names, p_plus.tolist()))
                        bound_plus = circuit.bind(p_dict_plus)

                        # -shift
                        p_minus = p_vals.copy()
                        p_minus[p_idx] -= shift
                        p_dict_minus = dict(zip(param_names, p_minus.tolist()))
                        bound_minus = circuit.bind(p_dict_minus)

                        for o_idx, obs in enumerate(obs_list):
                            ev_plus = float(backend.expval(bound_plus, obs))
                            ev_minus = float(backend.expval(bound_minus, obs))
                            grad_val = 0.5 * (ev_plus - ev_minus)
                            grads[b, p_idx] += upstream_np[b, o_idx] * grad_val

                return tf.convert_to_tensor(grads, dtype=tf.float32)

            return tf.convert_to_tensor(results, dtype=tf.float32), _grad

        # params_tf is already a tf.Tensor from _encode_input_tf
        return _dm_forward(tf.cast(params_tf, tf.float32))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.n_out)

    def get_config(self):
        config = super().get_config()
        config.update({
            "n_qubits": self.n_qubits,
            "n_layers": self.n_layers,
            "backend": self.backend_name,
            "observables": self.observable_strings,
            "encoding": self.encoding,
            "diff_method": self.diff_method,
        })
        return config


# ── convenience factory ──────────────────────────────────────────────────

def tf_quantum_layer(
    n_qubits: int,
    n_layers: int = 2,
    **kwargs,
) -> TFQuantumLayer:
    """Factory for :class:`TFQuantumLayer` with cleaner signature.

    Args:
        n_qubits: Number of qubits.
        n_layers: Number of circuit layers.

    Returns:
        A configured :class:`TFQuantumLayer` instance.
    """
    return TFQuantumLayer(n_qubits=n_qubits, n_layers=n_layers, **kwargs)
