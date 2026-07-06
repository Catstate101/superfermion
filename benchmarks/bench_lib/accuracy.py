"""Accuracy metrics for comparing (counts, statevector) between frameworks.

All inputs are assumed to already be in the harness's BigEndian convention
(qubit 0 = leftmost / MSB). Qiskit outputs must be passed through
bench.normalize first.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional
import numpy as np

from bench.normalize import fidelity as _fidelity, l2_distance, tvd


def pairwise_fidelity(psi: np.ndarray, psi_ref: np.ndarray) -> float:
    """|<psi|psi_ref>|^2. Both in BigEndian ordering, equal length."""
    return _fidelity(psi, psi_ref)


def statevector_l2(psi: np.ndarray, psi_ref: np.ndarray) -> float:
    return l2_distance(psi, psi_ref)


def counts_tvd(p_probs: Dict[str, float], q_probs: Dict[str, float]) -> float:
    """TVD on probability dicts (already normalized)."""
    return tvd(p_probs, q_probs)


# ---------- observables from probability dict (computational-basis Pauli-Z strings) ----------

def expectation_Z(probs: Dict[str, float], qubit: int, n_qubits: int) -> float:
    """<Z_qubit> in BigEndian: bitstring[qubit] == '0' contributes +1, '1' -1."""
    total = 0.0
    for bs, p in probs.items():
        if len(bs) != n_qubits:
            bs = bs.zfill(n_qubits)
        total += (1.0 if bs[qubit] == "0" else -1.0) * p
    return total


def expectation_ZZ(probs: Dict[str, float], q1: int, q2: int, n_qubits: int) -> float:
    total = 0.0
    for bs, p in probs.items():
        if len(bs) != n_qubits:
            bs = bs.zfill(n_qubits)
        b1 = 1 if bs[q1] == "1" else 0
        b2 = 1 if bs[q2] == "1" else 0
        sign = 1.0 if (b1 ^ b2) == 0 else -1.0
        total += sign * p
    return total


def expectation_parity(probs: Dict[str, float], n_qubits: int) -> float:
    """<Prod_i Z_i>. +1 if even number of 1s."""
    total = 0.0
    for bs, p in probs.items():
        if len(bs) != n_qubits:
            bs = bs.zfill(n_qubits)
        ones = bs.count("1")
        total += (1.0 if ones % 2 == 0 else -1.0) * p
    return total


def observable_vector(probs: Dict[str, float], n_qubits: int) -> List[float]:
    """Full observable signature used for cross-framework comparison:
    [<Z_0>, <Z_1>, ..., <Z_{n-1}>,
     <Z_0 Z_1>, <Z_1 Z_2>, ..., <Z_{n-2} Z_{n-1}>,
     <Parity>]
    """
    out = [expectation_Z(probs, i, n_qubits) for i in range(n_qubits)]
    out += [expectation_ZZ(probs, i, i + 1, n_qubits) for i in range(n_qubits - 1)]
    out.append(expectation_parity(probs, n_qubits))
    return out


def observable_linf(a: List[float], b: List[float]) -> float:
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    if a_arr.shape != b_arr.shape:
        return float("inf")
    return float(np.max(np.abs(a_arr - b_arr)))
