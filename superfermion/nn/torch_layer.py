"""
TorchQuantumLayer — PyTorch ↔ Superfermion bridge.

Wraps a parameterized quantum circuit as a ``torch.nn.Module``.
Gradients flow via ``torch.autograd.Function`` calling ``sf.State.grad()``.

Requires: ``pip install torch``
"""

from __future__ import annotations

from typing import Any, List, Optional

import numpy as np

try:
    import torch
    from torch import nn
except ImportError:
    raise ImportError(
        "PyTorch is required for TorchQuantumLayer. "
        "Install with: pip install torch"
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


class _QuantumFunction(torch.autograd.Function):
    """Custom autograd function bridging sf.State.grad() into PyTorch."""

    @staticmethod
    def forward(ctx, params, circuit, rust_obs, device_str, method):
        ctx.circuit = circuit
        ctx.rust_obs = rust_obs
        ctx.device_str = device_str
        ctx.method = method
        ctx.save_for_backward(params)

        p_np = params.detach().cpu().numpy()
        p_dict = dict(zip(circuit.parameters, p_np.tolist()))
        bound = circuit.bind(p_dict)
        state = sf.simulate(bound, device=device_str, method=method)
        val = state.expectation(rust_obs)
        return torch.tensor(val, dtype=params.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (params,) = ctx.saved_tensors
        p_np = params.detach().cpu().numpy()
        p_dict = dict(zip(ctx.circuit.parameters, p_np.tolist()))
        bound = ctx.circuit.bind(p_dict)
        state = sf.simulate(bound, device=ctx.device_str, method=ctx.method)
        dag = ctx.circuit.to_ir()
        grads = state.grad(ctx.rust_obs, dag, p_dict)
        grad_np = np.array([grads.get(k, 0.0) for k in ctx.circuit.parameters])
        grad_tensor = torch.from_numpy(grad_np).to(params.dtype) * grad_output
        return grad_tensor, None, None, None, None


class TorchQuantumLayer(nn.Module):
    """PyTorch module wrapping a Superfermion variational quantum circuit.

    Args:
        circuit: Parameterized ``sf.Circuit``.
        observable: Observable for expectation value measurement.
        device: Simulation device (``"cpu"`` or ``"gpu"``).
        method: Simulation method (``"statevector"``, ``"mps"``, etc.).
    """

    def __init__(
        self,
        circuit: sf.Circuit,
        observable,
        device: str = "cpu",
        method: str = "statevector",
    ):
        super().__init__()
        self._circuit = circuit
        self._observable = observable
        self._rust_obs = _obs_to_rust(observable)
        self._device = device
        self._method = method

        n_params = len(circuit.parameters) if circuit.parameters else 0
        if n_params > 0:
            self.weights = nn.Parameter(
                torch.empty(n_params, dtype=torch.float64).uniform_(0, 2 * np.pi)
            )
        else:
            self.weights = None

    def forward(self, x: Optional[torch.Tensor] = None) -> torch.Tensor:
        params = self.weights if self.weights is not None else torch.zeros(0)
        return _QuantumFunction.apply(
            params, self._circuit, self._rust_obs, self._device, self._method
        )
