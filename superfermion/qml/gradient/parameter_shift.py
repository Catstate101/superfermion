"""
Parameter-Shift Rule — gradient computation via sf.run().

Works with any device and simulation method.  Accepts every SF observable
form: ``str`` ("ZI"), ``dict``, ``SparsePauliOp``, ``Hamiltonian``,
``PauliString``, or a list of Pauli strings.

The rule is fixed by the eigenvalue spectrum of each gate's generator ``G``
(``∂U/∂θ = −i·G·U``), not by the gate's name:

* two distinct eigenvalues, unit gap — every plain rotation (Rx/Ry/Rz,
  Rxx/Ryy/Rzz), every phase gate (P, R1, CP) and the diagonal slots of
  U and Cu.  One shift is exact::

      ∂⟨O⟩/∂θ = ½ [ f(θ+π/2) − f(θ−π/2) ]

* three distinct eigenvalues {0, ±½} — the controlled rotations (CRx,
  CRy, CRz and Cu's θ).  The gap-2 component survives the θ-shift, so a
  second shift by π is needed::

      ∂⟨O⟩/∂θ = ½ [ f(θ+π/2) − f(θ−π/2) ]
              + ((1−√2)/4) [ f(θ+π) − f(θ−π) ]

Both are exact (not approximations).  A gate outside the table raises
``ValueError`` naming the gate and pointing at ``method="adjoint"`` — a
silently wrong number is never an acceptable outcome.

One parameter name may drive several gates (``ry(t); ry(t)``) or several
slots of one gate (``u(t, t, 0.1)``).  The derivative is the sum over
*occurrences*, each shifted on its own via ``Circuit.bind_occurrence``;
binding by name alone (the pre-0.1.24 behaviour) moves them all together
and reports ½·∂f instead of ∂f for a doubled gate.

Noisy gradients are supported via ``method="density_matrix"`` using Tr(O ρ).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

import superfermion as sf
from superfermion.observables.core import Observable, SparsePauliOp, expval

# ── Shift-rule table ───────────────────────────────────────────────────────────

_PI_2 = math.pi / 2.0

#: (shift in multiples of π/2, weight) — spectrum with a unit gap.
_RULE_GAP1 = ((1, 0.5),)

#: (shift in multiples of π/2, weight) — spectrum {0, ±½}; the π shift
#: cancels the ±1 difference that the ±π/2 shifts cannot see.
_RULE_GAP2 = ((1, 0.5), (2, (1.0 - math.sqrt(2.0)) / 4.0))

_SHIFT_RULES: dict[tuple[str, int], tuple[tuple[int, float], ...]] = {
    # gap 1: a single π/2 shift is exact
    ("RX", 0): _RULE_GAP1,
    ("RY", 0): _RULE_GAP1,
    ("RZ", 0): _RULE_GAP1,
    ("R1", 0): _RULE_GAP1,
    ("P", 0): _RULE_GAP1,
    ("RXX", 0): _RULE_GAP1,
    ("RYY", 0): _RULE_GAP1,
    ("RZZ", 0): _RULE_GAP1,
    ("CP", 0): _RULE_GAP1,
    # U(θ,φ,λ): the θ generator has eigenvalues ±½ (gap 1), φ and λ are
    # rank-one (eigenvalues {0,−1}).
    ("U", 0): _RULE_GAP1,
    ("U", 1): _RULE_GAP1,
    ("U", 2): _RULE_GAP1,
    ("U3", 0): _RULE_GAP1,
    ("U3", 1): _RULE_GAP1,
    ("U3", 2): _RULE_GAP1,
    # Cu(θ,φ,λ,γ): only θ sits in the ±½ block; φ, λ and γ are rank-one.
    ("CU", 0): _RULE_GAP2,
    ("CU", 1): _RULE_GAP1,
    ("CU", 2): _RULE_GAP1,
    ("CU", 3): _RULE_GAP1,
    # spectrum {0, ±½}: the controlled rotations
    ("CRX", 0): _RULE_GAP2,
    ("CRY", 0): _RULE_GAP2,
    ("CRZ", 0): _RULE_GAP2,
}


def _shift_rule(gate_name: str, slot: int) -> tuple[tuple[int, float], ...]:
    """Shift rule for parameter ``slot`` of ``gate_name``.

    Raises:
        ValueError: if the gate is not in the table — the alternative would
            be a wrong number, and ``method="adjoint"`` is always available
            on a statevector device.
    """
    rule = _SHIFT_RULES.get((gate_name.upper(), slot))
    if rule is None:
        raise ValueError(
            f"parameter-shift rule unavailable for gate '{gate_name}' slot "
            f"{slot} (its generator spectrum is not in the rule table). "
            f"Use method='adjoint' on a statevector device, or add the gate "
            f"to _SHIFT_RULES in "
            f"superfermion/qml/gradient/parameter_shift.py"
        )
    return rule


def _as_observable(observable: Any, n_qubits: int) -> Observable:
    """Coerce the user-facing observable forms into an ``Observable``.

    ``str``/``dict``/list-of-str used to fail with
    ``AttributeError: 'str' object has no attribute '_fast_expval'``; they
    are now the same thing as the equivalent ``SparsePauliOp``.

    The coercion itself is ``_observables.normalize_for_gradient``, which the
    adjoint and finite-difference engines and ``RunResult.grad`` also use, so
    the five entry points accept one spelling table instead of five
    (E/V-3: ``list[PauliString]`` used to raise here and succeed in the
    adjoint; E/V-4: the engine term form used to raise here and succeed in
    ``RunResult.grad``).  ``n_qubits`` is required because the width check is
    part of that shared contract: a label one character short used to act as
    identity on the missing wire (E/V-7).
    """
    from superfermion.qml.gradient._observables import (
        normalize_for_gradient,
        terms_to_dict,
    )

    # Every Observable subclass (PauliString, SparsePauliOp, Hamiltonian) is
    # handled by the shared spelling table, so there is no separate branch: an
    # already-built Observable goes through the same width and Hermiticity
    # checks as a bare "ZZ" string.
    terms = normalize_for_gradient(observable, n_qubits)
    return SparsePauliOp.from_dict(terms_to_dict(terms))


def _occurrences(circuit: sf.Circuit, name: str) -> list:
    """Occurrences of ``name``, failing loudly if it is not a parameter."""
    occurrences = circuit.param_occurrences(name)
    if not occurrences:
        raise ValueError(
            f"{name!r} is not a parameter of this circuit (circuit "
            f"parameters: {circuit.parameters}). Pass the unbound circuit "
            f"built with sf.param(...), not circuit.bind(...): binding folds "
            f"the parameters into constants."
        )
    return occurrences


# ── Public API ─────────────────────────────────────────────────────────────────


def parameter_shift_grad(
    circuit: sf.Circuit,
    observable: Observable,
    params: dict[str, float],
    device: Any = "cpu",
    method: str = "statevector",
    shots: int = 0,
) -> dict[str, float]:
    """Compute the gradient of ⟨O⟩ w.r.t. every circuit parameter.

    Exact for every gate in the shift-rule table (see the module docstring);
    raises for anything else rather than returning a wrong number.

    Args:
        circuit:    A parametric SF circuit with ``sf.param(...)`` symbols.
        observable: Any SF observable form (str, dict, SparsePauliOp,
                    Hamiltonian, PauliString, list of Pauli strings).
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
        >>> grad = parameter_shift_grad(circ, {'ZZ': 1.0}, {'theta': 0.5})
        >>> grad['theta']   # ∂⟨ZZ⟩/∂theta
    """
    obs = _as_observable(observable, circuit.n_qubits)

    def _expval_at(bound: sf.Circuit) -> float:
        result = sf.run(bound, device=device, method=method, shots=shots)
        if result.statevector is not None:
            sv = np.asarray(result.statevector, dtype=np.complex128).ravel()
            return expval(sv, obs)

        if result.metadata and 'density_matrix' in result.metadata:
            rho = result.metadata['density_matrix']
            from superfermion.backends.density_matrix import _observable_to_matrix
            o_mat = _observable_to_matrix(obs, circuit.n_qubits)
            return float(np.real(np.trace(o_mat @ rho)))

        return _expval_from_counts(result.counts, obs, circuit.n_qubits)

    grad: dict[str, float] = {}
    for name in params:
        total = 0.0
        for k, (gate_name, slot) in enumerate(_occurrences(circuit, name)):
            for mult, weight in _shift_rule(gate_name, slot):
                step = mult * _PI_2
                up = circuit.bind_occurrence(params, name, k, value=params[name] + step)
                down = circuit.bind_occurrence(
                    params, name, k, value=params[name] - step
                )
                total += weight * (_expval_at(up) - _expval_at(down))
        grad[name] = total

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
    grad_dict = parameter_shift_grad(
        circuit, observable, params, device=device, method=method, shots=shots
    )
    return np.array([grad_dict[n] for n in param_names])


def finite_diff_grad(
    circuit: sf.Circuit,
    observable: Observable,
    params: dict[str, float],
    device: Any = "cpu",
    method: str = "statevector",
    eps: float = 1e-5,
) -> dict[str, float]:
    """Finite-difference gradient (for cross-checking parameter-shift).

    Uses central differences: (f(θ+ε) − f(θ−ε)) / (2ε).  Less accurate than
    parameter-shift but works for any gate type.  Like the shift rule it
    differentiates one *occurrence* at a time, so it is a valid reference
    for circuits that reuse a parameter name.
    """
    obs = _as_observable(observable, circuit.n_qubits)

    def _expval_at(bound: sf.Circuit) -> float:
        result = sf.run(bound, device=device, method=method, shots=0)
        sv = np.asarray(result.statevector, dtype=np.complex128).ravel()
        return expval(sv, obs)

    grad: dict[str, float] = {}
    for name in params:
        total = 0.0
        for k, _ in enumerate(_occurrences(circuit, name)):
            up = circuit.bind_occurrence(params, name, k, value=params[name] + eps)
            down = circuit.bind_occurrence(params, name, k, value=params[name] - eps)
            total += (_expval_at(up) - _expval_at(down)) / (2 * eps)
        grad[name] = total

    return grad


# ── Internal helpers ───────────────────────────────────────────────────────────

def _expval_from_counts(
    counts: dict[str, int],
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
        result += prob * expval(sv, observable)

    return result
