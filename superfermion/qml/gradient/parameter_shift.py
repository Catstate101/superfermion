"""
Parameter-Shift Rule — gradient computation via sf.run().

Works with any device and simulation method. Supports all SF observables
(SparsePauliOp, Hamiltonian, PauliString).

The rule:  ∂⟨O⟩/∂θᵢ = ½ [ ⟨O⟩(θᵢ + π/2) − ⟨O⟩(θᵢ − π/2) ]

This is exact (not an approximation) for all gates whose generator G
satisfies G² = I, which covers all standard rotation gates Rₓ, Ry, Rz, P, etc.

Noisy gradients are supported via ``method="density_matrix"`` using Tr(O ρ).
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np

import superfermion as sf
from superfermion.observables.core import Observable, expval


# ── Public API ─────────────────────────────────────────────────────────────────


def parameter_shift_grad(
    circuit: sf.Circuit,
    observable: Observable,
    params: Dict[str, float],
    device: Any = "cpu",
    method: str = "statevector",
    shots: int = 0,
) -> Dict[str, float]:
    """Compute the gradient of ⟨O⟩ w.r.t. every circuit parameter.

    Uses the parameter-shift rule: exact for all standard rotation gates.

    Args:
        circuit:    A parametric SF circuit with ``sf.param(...)`` symbols.
        observable: Any SF observable (SparsePauliOp, Hamiltonian, PauliString).
        params:     Dict mapping parameter names → current values (floats).
        device:     Execution target — ``"cpu"``, ``"gpu"``, or ``DeviceExecutor``.
        method:     Simulation method — ``"statevector"``, ``"density_matrix"``, etc.
        shots:      0 → exact statevector expectation; > 0 → shot-based (noisy).

    Returns:
        Dict mapping parameter names → gradient values.

    Example:
        >>> circ = sf.Circuit(2)
        >>> theta = sf.param('theta')
        >>> circ.ry(theta, 0).cx(0, 1)
        >>> H = SparsePauliOp.from_dict({'ZZ': 1.0})
        >>> grad = parameter_shift_grad(circ, H, {'theta': 0.5})
        >>> grad['theta']   # ∂⟨ZZ⟩/∂theta
    """
    def _expval_at(p_dict: Dict[str, float]) -> float:
        bound = circuit.bind(p_dict)

        result = sf.run(bound, device=device, method=method, shots=shots)
        if result.statevector is not None:
            sv = np.asarray(result.statevector, dtype=np.complex128).ravel()
            return float(np.real(observable._fast_expval(sv)))

        if result.metadata and 'density_matrix' in result.metadata:
            rho = result.metadata['density_matrix']
            from superfermion.backends.density_matrix import _observable_to_matrix
            O_mat = _observable_to_matrix(observable, circuit.n_qubits)
            return float(np.real(np.trace(O_mat @ rho)))

        return _expval_from_counts(result.counts, observable, circuit.n_qubits)

    grad: Dict[str, float] = {}
    shift = math.pi / 2.0

    for name in params:
        p_plus = {**params, name: params[name] + shift}
        p_minus = {**params, name: params[name] - shift}
        grad[name] = 0.5 * (_expval_at(p_plus) - _expval_at(p_minus))

    return grad


def parameter_shift_grad_vector(
    circuit: sf.Circuit,
    observable: Observable,
    param_names: Sequence[str],
    param_values: np.ndarray,
    device: Any = "cpu",
    method: str = "statevector",
    shots: int = 0,
) -> np.ndarray:
    """Gradient as a 1-D numpy array aligned with ``param_names``.

    Convenience wrapper around :func:`parameter_shift_grad` for numerical
    optimizers (scipy.optimize, etc.) that expect array-in / array-out.

    Args:
        circuit:       Parametric SF circuit.
        observable:    SF observable.
        param_names:   Ordered list of parameter names.
        param_values:  Current parameter values (1-D array).
        device:        Execution target — ``"cpu"``, ``"gpu"``, or ``DeviceExecutor``.
        method:        Simulation method.
        shots:         0 for exact statevector.

    Returns:
        1-D numpy array of gradients, same length as ``param_names``.
    """
    params = {n: float(v) for n, v in zip(param_names, param_values)}
    grad_dict = parameter_shift_grad(circuit, observable, params, device=device, method=method, shots=shots)
    return np.array([grad_dict[n] for n in param_names])


def finite_diff_grad(
    circuit: sf.Circuit,
    observable: Observable,
    params: Dict[str, float],
    device: Any = "cpu",
    method: str = "statevector",
    eps: float = 1e-5,
) -> Dict[str, float]:
    """Finite-difference gradient (for cross-checking parameter-shift).

    Uses central differences: (f(θ+ε) − f(θ−ε)) / (2ε).
    Less accurate than parameter-shift but works for any gate type.
    """
    def _expval_at(p_dict):
        bound = circuit.bind(p_dict)
        result = sf.run(bound, device=device, method=method, shots=0)
        sv = np.asarray(result.statevector, dtype=np.complex128).ravel()
        return float(np.real(observable._fast_expval(sv)))

    grad: Dict[str, float] = {}
    for name in params:
        p_plus = {**params, name: params[name] + eps}
        p_minus = {**params, name: params[name] - eps}
        grad[name] = (_expval_at(p_plus) - _expval_at(p_minus)) / (2 * eps)

    return grad


# ── Internal helpers ───────────────────────────────────────────────────────────

def _expval_from_counts(
    counts: Dict[str, int],
    observable: Observable,
    n_qubits: int,
) -> float:
    """Estimate ⟨O⟩ from shot counts for diagonal (Z-basis) observables."""
    total = sum(counts.values())
    if total == 0:
        return 0.0

    result = 0.0
    for bitstring, count in counts.items():
        prob = count / total
        # Build a computational basis statevector with amplitude 1 at this basis state
        idx = int(bitstring, 2)
        sv = np.zeros(2 ** n_qubits, dtype=np.complex128)
        sv[idx] = 1.0
        result += prob * float(np.real(observable._fast_expval(sv)))

    return result
