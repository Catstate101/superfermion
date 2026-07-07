"""Adjoint differentiation — fast gradient for parameterised quantum circuits.

Uses the Rust sf-ir adjoint engine (Jones & Gacon 2020) via PyO3 bindings.
Cost: O((M + N) * 2^n) — one forward pass + one backward pass, regardless
of the number of parameters.  For N=20 parameters this is a 40x reduction
vs parameter-shift.

Public API:
    adjoint_grad_vector(circuit, observable, param_names, param_values)
    adjoint_grad(circuit, observable, params_dict)
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Union

import numpy as np

from superfermion.circuit import Circuit
from superfermion.parameters import SymbolicParameter


_PAULI_MAP = {"I": 0, "X": 1, "Y": 2, "Z": 3}


def _observable_to_rust_terms(
    obs, n_qubits: int
) -> List[tuple]:
    """Convert an observable to Rust's [(paulis_u8, coef_re, coef_im)] format.

    Supports:
      - str: single Pauli string like "ZZI"
      - dict: {"ZI": 0.5, "IX": 0.3}
      - Hamiltonian / SparsePauliOp with .terms attribute
      - Any object with .terms returning list of PauliString-like objects
    """
    from superfermion.observables.core import SparsePauliOp

    if isinstance(obs, str):
        paulis = [_PAULI_MAP[ch] for ch in reversed(obs.upper())]
        return [(paulis, 1.0, 0.0)]

    if isinstance(obs, dict):
        terms = []
        for pauli_str, coef in obs.items():
            paulis = [_PAULI_MAP[ch] for ch in reversed(pauli_str.upper())]
            c = complex(coef)
            terms.append((paulis, c.real, c.imag))
        return terms

    if isinstance(obs, SparsePauliOp):
        terms = []
        for pauli_str, coef in obs._terms:
            paulis = [_PAULI_MAP[ch] for ch in reversed(pauli_str.upper())]
            c = complex(coef)
            terms.append((paulis, c.real, c.imag))
        return terms

    if hasattr(obs, "terms"):
        terms = []
        for ps in obs.terms:
            paulis = [_PAULI_MAP[ch] for ch in reversed(ps.pauli_str.upper())]
            c = complex(ps.coeffs)
            terms.append((paulis, c.real, c.imag))
        return terms

    raise TypeError(
        f"Cannot convert observable of type {type(obs).__name__} to Rust format. "
        f"Expected str, dict, SparsePauliOp, or Hamiltonian."
    )


def adjoint_grad_vector(
    circuit: Circuit,
    observable,
    param_names: Sequence[str],
    param_values: np.ndarray,
) -> np.ndarray:
    """Adjoint differentiation gradient via the Rust sf-ir engine.

    Drop-in replacement for ``parameter_shift_grad_vector``.

    Returns an array of shape ``(len(param_names),)`` with d<O>/d(theta_k).
    """
    params_dict = {nm: float(v) for nm, v in zip(param_names, param_values)}
    rust_obs = _observable_to_rust_terms(observable, circuit.n_qubits)

    dag = circuit.to_ir()
    grad_dict = dag.adjoint_grad(rust_obs, params_dict)

    return np.array([grad_dict.get(nm, 0.0) for nm in param_names])


def adjoint_grad(
    circuit: Circuit,
    observable,
    params: Dict[str, float],
) -> Dict[str, float]:
    """Dict-form adjoint gradient (analogue of ``parameter_shift_grad``)."""
    names = list(params.keys())
    vals = np.array([params[nm] for nm in names])
    g = adjoint_grad_vector(circuit, observable, names, vals)
    return {nm: float(v) for nm, v in zip(names, g)}
