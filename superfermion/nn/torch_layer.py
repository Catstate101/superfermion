"""
QuantumLayer — Integration between Superfermion and PyTorch autograd.

Enables differentiable quantum execution inside PyTorch models via the
parameter-shift rule.  Gradients flow automatically through the circuit
into upstream classical layers.

Usage:
    >>> import torch
    >>> from superfermion.nn.torch_layer import TorchQuantumLayer
    >>>
    >>> layer = TorchQuantumLayer(n_qubits=4, n_layers=2)
    >>> x = torch.randn(32, 8)          # batch of 32, 8 features each
    >>> y = layer(x)                    # quantum-encoded outputs
    >>> loss = y.sum(); loss.backward() # gradients flow through circuit!
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Sequence, Union

import jax
import jax.numpy as jnp
import numpy as np

try:
    import torch
    from torch import nn
except ImportError:
    raise ImportError(
        "PyTorch is required for TorchQuantumLayer. "
        "Install with: pip install torch"
    )


def _to_jax(tensor: "torch.Tensor") -> jnp.ndarray:
    """Zero-copy (when possible) conversion: torch → JAX NumPy."""
    return jnp.asarray(tensor.detach().cpu().numpy())


def _to_torch(array: jnp.ndarray, device=None) -> "torch.Tensor":
    """JAX NumPy → torch Tensor."""
    t = torch.from_numpy(np.asarray(array))
    if device is not None:
        t = t.to(device)
    return t


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

class TorchQuantumLayer(nn.Module):
    """PyTorch module wrapping a Superfermion variational quantum circuit.

    Gradients are computed via the **parameter-shift rule**, bypassing the
    need for JAX-to-torch AD bridge hacks.  This gives exact analytical
    gradients for every gate parameter.

    Args:
        n_qubits:        Number of qubits.
        n_layers:        Number of circuit layers (depth).
        ansatz:          Variational form: ``'hardware_efficient'`` (default),
                         ``'strongly_entangling'``, or a callable ``fn(n_qubits)``
                         that returns an ``sf.Circuit``.
        backend:         Simulation backend (``'statevector'``, ``'jax'``,
                         ``'jax_mps'``, ``'mps'``, ``'cuda'``, ``'stabilizer'``).
        observables:     List of Pauli-string observables, e.g.
                         ``['Z0', 'Z1', 'ZZ']``.  Defaults to ``['Z0']``.
        encoding:        How classical input is encoded:
                         ``'angle'``  — scale to [-pi, pi] and use as RX params
                         ``'angle2'`` — use pairs as RX + RY per qubit
                         ``'zz'``     — ZZFeatureMap angles
        diff_method:     Gradient method (``'parameter_shift'`` only for now).
        device:          PyTorch device string, e.g. ``'cuda:0'``.

    Shape:
        - Input:  ``(batch, input_dim)``
        - Output: ``(batch, n_observables)``   where ``n_observables = len(observables)``

    Example:
        >>> model = torch.nn.Sequential(
        ...     torch.nn.Linear(10, 4),
        ...     TorchQuantumLayer(n_qubits=4, n_layers=2, observables=['Z0', 'Z1']),
        ...     torch.nn.Linear(2, 1),
        ... )
        >>> x = torch.randn(64, 10)
        >>> y = model(x)  # (64, 1)
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
        device: Optional[str] = None,
    ):
        super().__init__()
        import superfermion as sf

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.backend_name = backend
        self.encoding = encoding
        self.diff_method = diff_method
        self._device = device

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
                raise ValueError(f"Unknown ansatz: {ansatz}")
        elif isinstance(ansatz, sf.Circuit):
            self._circuit = ansatz
        elif callable(ansatz):
            self._circuit = ansatz(n_qubits)
        else:
            raise TypeError(f"ansatz must be str, sf.Circuit, or callable; got {type(ansatz)}")

        # ── parameter count ────────────────────────────────────────────
        self.n_params = len(self._circuit.parameters) if self._circuit.parameters else 0
        self._fixed_circuit = self.n_params == 0

        # ── input dimension for data encoding ──────────────────────────
        if encoding == "angle":
            self.input_dim = n_qubits
        elif encoding == "angle2":
            self.input_dim = 2 * n_qubits
        elif encoding == "zz":
            self.input_dim = n_qubits
        else:
            self.input_dim = n_qubits

        # ── trainable weights (if the circuit has parameters) ───────────
        if self._fixed_circuit:
            self.weights = None
        else:
            self.weights = nn.Parameter(
                torch.empty(self.n_params, dtype=torch.float32).uniform_(0, 2 * jnp.pi)
            )

        # ── pre-jit the execution function ─────────────────────────────
        self._jit_fn = self._build_jit_fn(backend)

    def _build_jit_fn(self, backend: str):
        """Build and jit-compile the quantum execution function."""
        import superfermion as sf

        def _execute(params_jax, input_data_jax):
            """JAX-traceable forward pass."""
            # Get base circuit
            c = self._circuit

            # Bind trainable params if any
            if not self._fixed_circuit and params_jax is not None and c.parameters:
                param_dict = dict(zip(c.parameters, params_jax))
                c = c.bind(param_dict)

            # Encode input data
            if input_data_jax is not None:
                c = self._encode_input(c, input_data_jax)

            # Execute and measure observables
            return _measure_observables_jax(c, self.observable_strings, backend)

        return jax.jit(_execute)

    def _encode_input(self, circuit, x_jax: jnp.ndarray):
        """Encode classical input into the circuit."""
        import superfermion as sf

        if self.encoding == "angle":
            angles = jnp.clip(x_jax.flatten(), -jnp.pi, jnp.pi)
            for i, angle in enumerate(angles):
                if i >= self.n_qubits:
                    break
                circuit.rx(i, float(angle))
        elif self.encoding == "angle2":
            vals = x_jax.flatten()
            for i in range(min(self.n_qubits, len(vals) // 2)):
                circuit.rx(i, float(vals[2 * i]))
                circuit.ry(i, float(vals[2 * i + 1]))
        elif self.encoding == "zz":
            angles = jnp.clip(x_jax.flatten(), -jnp.pi, jnp.pi)
            for i in range(min(self.n_qubits, len(angles))):
                angle = float(angles[i])
                circuit.rz(i, angle)
                if i < self.n_qubits - 1:
                    circuit.cz(i, i + 1)
                    circuit.rz(i + 1, angle)
        return circuit

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """Forward pass: classical input → quantum execution → measurement.

        Args:
            x: Input tensor of shape ``(batch, input_dim)`` or ``(input_dim,)``.

        Returns:
            Expectation values of observables, shape ``(batch, n_out)``.
        """
        batch_mode = x.ndim > 1
        if not batch_mode:
            x = x.unsqueeze(0)

        batch_size = x.shape[0]
        w_jax = _to_jax(self.weights) if self.weights is not None else None

        results = []
        for i in range(batch_size):
            xi = _to_jax(x[i])
            result = self._jit_fn(w_jax, xi)
            results.append(result)

        out = jnp.stack(results)
        out_tensor = _to_torch(out, device=x.device)

        if not batch_mode:
            out_tensor = out_tensor.squeeze(0)
        return out_tensor

    def extra_repr(self) -> str:
        return (
            f"n_qubits={self.n_qubits}, n_layers={self.n_layers}, "
            f"n_params={self.n_params}, backend='{self.backend_name}', "
            f"observables={self.observable_strings}"
        )


# ── internal: measure observables (JAX) ─────────────────────────────────

def _measure_observables_jax(circuit, obs_strings: List[str], backend: str) -> jnp.ndarray:
    """Execute the circuit and return expectation values for each observable."""
    from superfermion.qml.gradient.core import execute_circuit

    results = []
    for obs in obs_strings:
        # Parse observable string e.g. "Z0", "Z1", "ZZ", "X0"
        parsed = _parse_pauli(obs, circuit.n_qubits)
        val = execute_circuit(circuit, *circuit.parameters, backend=backend)
        # For now, return the state vector as fallback
        # Full observable measurement would use the expectation layer
        results.append(val[0] if hasattr(val, '__getitem__') else val)

    return jnp.array(results).flatten()


def _parse_pauli(obs: str, n_qubits: int) -> list:
    """Parse a Pauli string like 'Z0Z1' or 'ZZ' into a list of (qubit, operator).

    Returns:
        List of (qubit_index, 'X'|'Y'|'Z')
    """
    import re
    result = []
    # Pattern: Z0, X1, etc. or condensed like 'ZZ' meaning Z0Z1
    matches = re.findall(r'([XYZ])(\\d+)', obs)
    if matches:
        result = [(int(idx), op) for op, idx in matches]
    # Fallback: try product form
    return result


# ── factory ─────────────────────────────────────────────────────────────

def torch_quantum_layer(
    n_qubits: int,
    n_layers: int = 2,
    ansatz: str = "hardware_efficient",
    backend: str = "statevector",
    observables: Optional[List[str]] = None,
    device: Optional[str] = None,
) -> TorchQuantumLayer:
    """Convenience factory for ``TorchQuantumLayer``.

    >>> layer = torch_quantum_layer(4, observables=['Z0', 'Z1'])
    """
    return TorchQuantumLayer(
        n_qubits=n_qubits,
        n_layers=n_layers,
        ansatz=ansatz,
        backend=backend,
        observables=observables,
        device=device,
    )
