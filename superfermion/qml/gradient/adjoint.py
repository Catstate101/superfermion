"""Adjoint differentiation — fast gradient for parameterised quantum circuits.

For a circuit U(theta) = U_M ... U_1 acting on |0> and an observable O,
the standard parameter-shift rule needs 2 forward evaluations per
parameter, i.e. O(M) forward passes.  The adjoint method computes the
full gradient in a single forward pass plus a single backward pass —
O(M) gates of work total, regardless of the number of parameters.  That
gives a 2N-fold speed-up where N is the parameter count.

Algorithm (Jones & Gacon 2020, "Efficient calculation of gradients in
classical simulations of variational quantum algorithms"):

  1. Forward: evolve |psi> = U(theta)|0>.
  2. Build |phi> = O |psi>  (apply the observable to psi).
  3. Walk the gates in REVERSE.  For each gate U_k (with parameter
     theta_k and Pauli generator G_k = exp(-i*alpha*theta_k * G_k)):
       a. Compute gradient contribution
            d<O>/dtheta_k = 2*alpha * Im( <phi | G_k | psi_{k-1}> )
       b. Apply U_k^dagger to phi.

The cost is O(M * 2^n) regardless of N — vs O(M*N * 2^n) for
parameter-shift.  At N=20 parameters that's a 40x reduction in forward
passes.

Public API: ``adjoint_grad_vector(circuit, observable, param_names,
param_values)`` returns the same numpy array shape as
``parameter_shift_grad_vector`` so it's a drop-in replacement.
"""
from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import numpy as np

import superfermion as sf
from superfermion.observables.core import Observable, SparsePauliOp
from superfermion.parameters import SymbolicParameter
from superfermion.circuit import Circuit, GateRecord


# ─── Pauli matrices ──────────────────────────────────────────────────────────
_I2 = np.eye(2, dtype=np.complex128)
_PX = np.array([[0, 1], [1, 0]], dtype=np.complex128)
_PY = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
_PZ = np.array([[1, 0], [0, -1]], dtype=np.complex128)
_PAULI = {"I": _I2, "X": _PX, "Y": _PY, "Z": _PZ}


# ─── Statevector primitives (SF MSB convention) ─────────────────────────────
def _apply_1q(state: np.ndarray, U: np.ndarray, q: int, n: int) -> np.ndarray:
    tensor = state.reshape([2] * n)
    out = np.tensordot(U, tensor, axes=([1], [q]))
    out = np.moveaxis(out, 0, q)
    return out.reshape(-1)


def _apply_2q(state: np.ndarray, U: np.ndarray, q1: int, q2: int, n: int) -> np.ndarray:
    tensor = state.reshape([2] * n)
    Ut = U.reshape(2, 2, 2, 2)
    out = np.tensordot(Ut, tensor, axes=([2, 3], [q1, q2]))
    out = np.moveaxis(out, [0, 1], [q1, q2])
    return out.reshape(-1)


def _apply_3q(state: np.ndarray, U: np.ndarray, q1: int, q2: int, q3: int, n: int) -> np.ndarray:
    tensor = state.reshape([2] * n)
    Ut = U.reshape(2, 2, 2, 2, 2, 2)
    out = np.tensordot(Ut, tensor, axes=([3, 4, 5], [q1, q2, q3]))
    out = np.moveaxis(out, [0, 1, 2], [q1, q2, q3])
    return out.reshape(-1)


def _apply_gate(state: np.ndarray, U: np.ndarray, qs: List[int], n: int) -> np.ndarray:
    if len(qs) == 1: return _apply_1q(state, U, qs[0], n)
    if len(qs) == 2: return _apply_2q(state, U, qs[0], qs[1], n)
    if len(qs) == 3: return _apply_3q(state, U, qs[0], qs[1], qs[2], n)
    raise NotImplementedError(f"Gate on {len(qs)} qubits not handled by adjoint")


def _gate_matrix(name: str, p: List[float]) -> np.ndarray:
    """Return the unitary for gate ``name`` with numeric params ``p``."""
    if name == "H": return np.array([[1, 1], [1, -1]], dtype=np.complex128) / math.sqrt(2)
    if name == "X": return _PX.copy()
    if name == "Y": return _PY.copy()
    if name == "Z": return _PZ.copy()
    if name == "S": return np.array([[1, 0], [0, 1j]], dtype=np.complex128)
    if name == "SDG": return np.array([[1, 0], [0, -1j]], dtype=np.complex128)
    if name == "T": return np.array([[1, 0], [0, np.exp(1j * math.pi / 4)]], dtype=np.complex128)
    if name == "TDG": return np.array([[1, 0], [0, np.exp(-1j * math.pi / 4)]], dtype=np.complex128)
    if name in ("ID", "I"): return _I2.copy()
    if name == "RX":
        c, s = math.cos(p[0]/2), math.sin(p[0]/2)
        return np.array([[c, -1j*s], [-1j*s, c]], dtype=np.complex128)
    if name == "RY":
        c, s = math.cos(p[0]/2), math.sin(p[0]/2)
        return np.array([[c, -s], [s, c]], dtype=np.complex128)
    if name == "RZ":
        return np.array([[np.exp(-1j*p[0]/2), 0], [0, np.exp(1j*p[0]/2)]], dtype=np.complex128)
    if name in ("R1", "P"):
        return np.array([[1, 0], [0, np.exp(1j*p[0])]], dtype=np.complex128)
    if name in ("CX", "CNOT"):
        return np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=np.complex128)
    if name == "CZ":
        return np.diag([1, 1, 1, -1]).astype(np.complex128)
    if name == "SWAP":
        return np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=np.complex128)
    if name == "RZZ":
        em, ep = np.exp(-1j*p[0]/2), np.exp(1j*p[0]/2)
        return np.diag([em, ep, ep, em]).astype(np.complex128)
    if name == "CCX":
        m = np.eye(8, dtype=np.complex128)
        m[6, 6] = 0; m[7, 7] = 0; m[6, 7] = 1; m[7, 6] = 1
        return m
    raise NotImplementedError(f"Gate {name} not in adjoint kernel")


# ─── Generator metadata ──────────────────────────────────────────────────────
# Each rotation gate U(theta) = exp(-i*alpha*theta * G).  We need the
# Pauli generator G and the prefactor alpha so the gradient is
# d<O>/dtheta = 2*alpha * Im(<phi|G|psi_prev>).
_GENERATORS: Dict[str, Tuple[Tuple[str, ...], float]] = {
    "RX":  (("X",), 0.5),
    "RY":  (("Y",), 0.5),
    "RZ":  (("Z",), 0.5),
    "RZZ": (("Z", "Z"), 0.5),
    # P/R1: U(theta) = diag(1, e^{i*theta}) = e^{i*theta/2} * RZ(-theta) up to
    # global phase.  Locally generator is (I-Z)/2 — not pure Pauli — skip
    # for now (use parameter-shift fallback if the user has these).
}


def _observable_to_terms(obs, n: int) -> List[Tuple[str, complex]]:
    if isinstance(obs, str):
        return [(obs, 1.0 + 0.0j)]
    if isinstance(obs, dict):
        return [(k, complex(v)) for k, v in obs.items()]
    if isinstance(obs, SparsePauliOp):
        # SF SparsePauliOp stores ._terms as a list of (pauli_str, coef) tuples
        return [(k, complex(v)) for k, v in obs._terms]
    if hasattr(obs, "_terms"):
        terms = obs._terms
        if isinstance(terms, dict):
            return [(k, complex(v)) for k, v in terms.items()]
        return [(k, complex(v)) for k, v in terms]
    if hasattr(obs, "to_dict"):
        return [(k, complex(v)) for k, v in obs.to_dict().items()]
    raise TypeError(f"Unsupported observable type: {type(obs).__name__}")


def _apply_pauli_string(state: np.ndarray, pauli_str: str, n: int) -> np.ndarray:
    out = state.copy()
    for q, ch in enumerate(pauli_str.upper()):
        if ch == "I":
            continue
        out = _apply_1q(out, _PAULI[ch], q, n)
    return out


def _resolve_param(p, params: Dict[str, float]) -> float:
    if isinstance(p, SymbolicParameter):
        return float(params[p.name])
    if isinstance(p, str):
        return float(params[p])
    return float(p)


# ─── Public API ──────────────────────────────────────────────────────────────
def adjoint_grad_vector(
    circuit: Circuit,
    observable,
    param_names: Sequence[str],
    param_values: np.ndarray,
) -> np.ndarray:
    """Adjoint differentiation gradient — drop-in for parameter_shift_grad_vector.

    Returns an array of shape (len(param_names),) with d<O>/dtheta_k.

    Cost: 1 forward pass + 1 backward pass + 1 inner product per parameter
    = O((M + N) * 2^n) instead of parameter-shift's O(M * N * 2^n).

    For a 20-parameter VQE ansatz on n=10 qubits this is a 40x reduction
    in forward passes.
    """
    n = circuit.n_qubits
    params: Dict[str, float] = {nm: float(v) for nm, v in zip(param_names, param_values)}
    sym_to_idx: Dict[str, int] = {nm: i for i, nm in enumerate(param_names)}

    # Forward pass; cache intermediates
    state = np.zeros(2 ** n, dtype=np.complex128)
    state[0] = 1.0
    intermediates: List[np.ndarray] = [state.copy()]
    for g in circuit._gates:
        nm = g.name.upper()
        if nm in ("MEASURE", "BARRIER", "RESET"):
            intermediates.append(state.copy())
            continue
        p_vals = [_resolve_param(p, params) for p in g.params]
        U = _gate_matrix(nm, p_vals)
        state = _apply_gate(state, U, list(g.qubits), n)
        intermediates.append(state.copy())

    # Build phi = O |psi_M> as a sum of weighted Pauli applications
    psi_final = state
    phi = np.zeros_like(psi_final)
    for pauli_str, coef in _observable_to_terms(observable, n):
        phi = phi + coef * _apply_pauli_string(psi_final, pauli_str, n)

    # Backward pass
    grad = np.zeros(len(param_names))
    n_gates = len(circuit._gates)
    for k_rev in range(n_gates - 1, -1, -1):
        g = circuit._gates[k_rev]
        nm = g.name.upper()
        if nm in ("MEASURE", "BARRIER", "RESET"):
            continue

        # Gradient contribution of THIS gate (if learnable)
        # Derivation: with U(theta) = exp(-i*alpha*theta*G), we have
        #   d|psi_M>/dtheta_k = -i*alpha * U_{M-1}...U_{k+1} G |psi_after_k>
        # where |psi_after_k> = intermediates[k_rev + 1] (post-gate state).
        # Define |phi_k> = (U_{M-1}...U_{k+1})^dagger O |psi_M>; at this
        # point in the loop phi == phi_k (we have NOT yet back-walked
        # through gate k_rev).  Then
        #   d<O>/dtheta_k = 2*alpha * Im( <phi_k| G | psi_after_k> ).
        gen_info = _GENERATORS.get(nm)
        if gen_info is not None and len(g.params) >= 1:
            sym = g.params[0]
            sym_name = (sym.name if isinstance(sym, SymbolicParameter)
                        else (sym if isinstance(sym, str) else None))
            idx = sym_to_idx.get(sym_name) if sym_name is not None else None
            if idx is not None:
                gen_paulis, alpha = gen_info
                # post-gate state = intermediates[k_rev + 1]
                psi_after = intermediates[k_rev + 1]
                g_psi = psi_after
                for q_local, ch in enumerate(gen_paulis):
                    g_psi = _apply_1q(g_psi, _PAULI[ch], g.qubits[q_local], n)
                ip = np.vdot(phi, g_psi)
                grad[idx] += 2.0 * alpha * float(np.imag(ip))

        # Back-walk phi: phi <- U_k^dagger phi
        p_vals = [_resolve_param(p, params) for p in g.params]
        U = _gate_matrix(nm, p_vals)
        Udag = U.conj().T
        phi = _apply_gate(phi, Udag, list(g.qubits), n)

    return grad


def adjoint_grad(
    circuit: Circuit,
    observable,
    params: Dict[str, float],
) -> Dict[str, float]:
    """Dict-form adjoint gradient (analogue of parameter_shift_grad)."""
    names = list(params.keys())
    vals = np.array([params[nm] for nm in names])
    g = adjoint_grad_vector(circuit, observable, names, vals)
    return {nm: float(v) for nm, v in zip(names, g)}
