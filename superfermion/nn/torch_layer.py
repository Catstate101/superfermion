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

    SF Pauli strings put qubit 0 first (e.g. "ZI" = Z on qubit 0), matching
    the Rust engine (paulis[0] = qubit 0) and ``adjoint_grad`` — no reversal.
    (Verified: PauliString("ZI") on |01> gives +1 = <Z0>.)
    """
    from superfermion.observables.core import PauliString, SparsePauliOp, Hamiltonian
    terms = []
    if isinstance(observable, PauliString):
        paulis = [_PAULI_MAP[c] for c in observable.pauli_str]
        terms.append((paulis, float(observable.coeffs.real), float(observable.coeffs.imag)))
    elif isinstance(observable, SparsePauliOp):
        for ps, coeff in observable._terms:
            paulis = [_PAULI_MAP[c] for c in ps]
            terms.append((paulis, float(complex(coeff).real), float(complex(coeff).imag)))
    elif isinstance(observable, Hamiltonian):
        for t in observable.terms:
            paulis = [_PAULI_MAP[c] for c in t.pauli_str]
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
        feature_names: optional list of circuit parameter names fed from
            the layer input ``x`` at call time (data embedding); all other
            parameters become trainable weights. ``None`` (default) keeps
            the legacy behavior: every parameter is a weight and ``x`` is
            ignored. Gradients flow to both weights and ``x``.
    """

    def __init__(
        self,
        circuit: sf.Circuit,
        observable,
        device: str = "cpu",
        method: str = "statevector",
        feature_names: Optional[List[str]] = None,
    ):
        super().__init__()
        self._circuit = circuit
        self._observable = observable
        self._rust_obs = _obs_to_rust(observable)
        self._device = device
        self._method = method
        allp = list(circuit.parameters) if circuit.parameters else []
        self._feature_names = list(feature_names) if feature_names else []
        unknown = [f for f in self._feature_names if f not in allp]
        if unknown:
            raise ValueError(f"feature_names not in circuit.parameters: {unknown}")
        self._trainable = [p for p in allp if p not in self._feature_names]
        self._feat_idx = [allp.index(f) for f in self._feature_names]
        self._train_idx = [allp.index(p) for p in self._trainable]

        n_weights = len(self._trainable) if self._feature_names else len(allp)
        if n_weights > 0:
            self.weights = nn.Parameter(
                torch.empty(n_weights, dtype=torch.float64).uniform_(0, 2 * np.pi)
            )
        else:
            self.weights = None

    def forward(self, x: Optional[torch.Tensor] = None) -> torch.Tensor:
        if not self._feature_names:
            params = self.weights if self.weights is not None else torch.zeros(0)
            return _QuantumFunction.apply(
                params, self._circuit, self._rust_obs, self._device, self._method
            )
        if x is None:
            raise ValueError("feature_names is set but forward() got x=None")
        xf = torch.as_tensor(x, dtype=torch.float64).flatten()
        if xf.numel() != len(self._feature_names):
            raise ValueError(
                f"expected {len(self._feature_names)} features, got {xf.numel()}"
            )
        n_all = len(self._feat_idx) + len(self._train_idx)
        full = torch.zeros(n_all, dtype=torch.float64)
        # index_put preserves autograd flow into both x and weights.
        full[self._feat_idx] = xf
        if self.weights is not None:
            full[self._train_idx] = self.weights
        return _QuantumFunction.apply(
            full, self._circuit, self._rust_obs, self._device, self._method
        )
