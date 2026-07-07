"""
Density Matrix Utilities — embedding, Kraus application, sampling, and
observable-to-matrix conversion for the density_matrix simulation method.

The DensityMatrixBackend class has been removed. All density matrix simulation
flows through RustDevice._run_density_matrix() using Rust-native DM evolution.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from superfermion.circuit import Circuit, GateRecord
from superfermion.results import RunResult


_I2 = np.eye(2, dtype=np.complex128)
_X  = np.array([[0, 1], [1, 0]], dtype=np.complex128)
_Y  = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
_Z  = np.array([[1, 0], [0, -1]], dtype=np.complex128)


from superfermion.noise import NoiseModel, kraus_depolarizing_1q, kraus_depolarizing_2q, kraus_amplitude_damping, kraus_phase_damping, kraus_bit_flip, kraus_phase_flip, kraus_bit_phase_flip


# ── Density matrix operations ──────────────────────────────────────────────────

def _reverse_qubits_dm(rho: np.ndarray, n: int) -> np.ndarray:
    """Reverse qubit ordering in density matrix (big-endian ↔ little-endian).

    SF internally uses big-endian convention (qubit 0 = MSB, leftmost in
    the Kronecker product).  PennyLane and Qiskit use little-endian
    (qubit 0 = LSB, rightmost).  This function applies the bit-reversal
    permutation  ρ' = P ρ P†  so that the output matches the standard
    little-endian convention expected by external frameworks.
    """
    dim = 2 ** n
    perm = np.zeros((dim, dim), dtype=np.complex128)
    for i in range(dim):
        bits = format(i, f'0{n}b')
        i_rev = int(bits[::-1], 2)
        perm[i_rev, i] = 1.0
    return perm @ rho @ perm.T


def _embed_1q(gate_2x2: np.ndarray, qubit: int, n: int) -> np.ndarray:
    """Embed a 2×2 single-qubit unitary into the 2^n × 2^n Hilbert space."""
    ops = [gate_2x2 if i == qubit else _I2 for i in range(n)]
    result = ops[0]
    for op in ops[1:]:
        result = np.kron(result, op)
    return result


def _embed_2q(gate_4x4: np.ndarray, q0: int, q1: int, n: int) -> np.ndarray:
    """Embed a 4×4 two-qubit unitary into the 2^n space.

    Handles non-adjacent qubits via SWAP routing then SWAP-back.
    SF convention: qubit 0 = MSB (leftmost), so we must respect ordering.
    """
    # Build the full unitary by tensor product expansion
    # We use the kronecker embedding approach for small n
    dim = 2 ** n
    U_full = np.eye(dim, dtype=np.complex128)

    # Rearrange gate to match qubit order q0 < q1
    if q0 > q1:
        # Transpose operator to swap control/target ordering
        G = gate_4x4.reshape(2, 2, 2, 2).transpose(1, 0, 3, 2).reshape(4, 4)
        q0, q1 = q1, q0
    else:
        G = gate_4x4

    # Route q1 adjacent to q0 via SWAP permutations
    swaps: List[Tuple[int, int]] = []
    cur_q1 = q1
    while cur_q1 - q0 > 1:
        neigh = cur_q1 - 1
        swaps.append((neigh, cur_q1))
        U_full = _embed_swap(neigh, cur_q1, n) @ U_full
        cur_q1 = neigh

    # Embed the gate at (q0, cur_q1)
    G_full = _embed_2q_adjacent(G, q0, n)
    U_full = G_full @ U_full

    # Undo SWAPs
    for a, b in reversed(swaps):
        U_full = _embed_swap(a, b, n) @ U_full

    return U_full


def _embed_swap(q0: int, q1: int, n: int) -> np.ndarray:
    SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.complex128)
    return _embed_2q_adjacent(SWAP, min(q0, q1), n)


def _embed_2q_adjacent(gate_4x4: np.ndarray, q_low: int, n: int) -> np.ndarray:
    """Embed gate acting on adjacent qubits (q_low, q_low+1) into 2^n space."""
    ops: List[np.ndarray] = []
    i = 0
    while i < n:
        if i == q_low:
            ops.append(gate_4x4)
            i += 2
        else:
            ops.append(_I2)
            i += 1
    result = ops[0]
    for op in ops[1:]:
        result = np.kron(result, op)
    return result


def _apply_unitary_dm(rho: np.ndarray, U: np.ndarray) -> np.ndarray:
    """Apply unitary: ρ → U ρ U†"""
    return U @ rho @ U.conj().T


def _apply_kraus_1q(rho: np.ndarray, kraus_set: List[np.ndarray], qubit: int, n: int) -> np.ndarray:
    """Apply a 1-qubit Kraus set to the density matrix."""
    new_rho = np.zeros_like(rho)
    for K in kraus_set:
        K_full = _embed_1q(K, qubit, n)
        new_rho += K_full @ rho @ K_full.conj().T
    return new_rho


def _apply_kraus_2q(rho: np.ndarray, kraus_set: List[np.ndarray], q0: int, q1: int, n: int) -> np.ndarray:
    """Apply a 2-qubit Kraus set to the density matrix."""
    new_rho = np.zeros_like(rho)
    for K in kraus_set:
        K_full = _embed_2q(K, q0, q1, n)
        new_rho += K_full @ rho @ K_full.conj().T
    return new_rho


# ── Measurement from density matrix ───────────────────────────────────────────

def _dm_to_probs(rho: np.ndarray) -> np.ndarray:
    """Extract computational-basis probabilities from density matrix."""
    return np.real(np.diag(rho))


def _sample_dm(rho: np.ndarray, shots: int, rng: np.random.Generator) -> Dict[str, int]:
    n_qubits = int(round(math.log2(rho.shape[0])))
    probs = _dm_to_probs(rho)
    probs = np.maximum(probs, 0)
    probs /= probs.sum()
    indices = rng.choice(len(probs), size=shots, p=probs)
    counts: Dict[str, int] = {}
    for idx in indices:
        bs = format(idx, f'0{n_qubits}b')
        counts[bs] = counts.get(bs, 0) + 1
    return counts


def _apply_readout_noise(counts: Dict[str, int], p_readout: float, rng: np.random.Generator) -> Dict[str, int]:
    """Flip each measured bit independently with probability p_readout."""
    if p_readout <= 0.0:
        return counts
    noisy: Dict[str, int] = {}
    for bs, count in counts.items():
        for _ in range(count):
            bits = list(bs)
            for i in range(len(bits)):
                if rng.random() < p_readout:
                    bits[i] = '1' if bits[i] == '0' else '0'
            new_bs = ''.join(bits)
            noisy[new_bs] = noisy.get(new_bs, 0) + 1
    return noisy




def _observable_to_matrix(obs, n: int) -> np.ndarray:
    """Build the full 2^n × 2^n matrix for a SF observable."""
    from superfermion.observables.core import PauliString, SparsePauliOp, Hamiltonian

    _P = {
        'I': np.eye(2, dtype=np.complex128),
        'X': np.array([[0, 1], [1, 0]], dtype=np.complex128),
        'Y': np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
        'Z': np.array([[1, 0], [0, -1]], dtype=np.complex128),
    }

    def _ps_matrix(pauli_str: str, coeff: complex) -> np.ndarray:
        mat = coeff * np.array([[1.0]], dtype=np.complex128)
        for ch in pauli_str:
            mat = np.kron(mat, _P[ch])
        return mat

    if isinstance(obs, PauliString):
        return _ps_matrix(obs.pauli_str, obs.coeffs)
    elif isinstance(obs, SparsePauliOp):
        mat = np.zeros((2 ** n, 2 ** n), dtype=np.complex128)
        for ps, coeff in obs._terms:
            mat += _ps_matrix(ps, coeff)
        return mat
    elif isinstance(obs, Hamiltonian):
        mat = np.zeros((2 ** n, 2 ** n), dtype=np.complex128)
        for term in obs.terms:
            mat += _ps_matrix(term.pauli_str, term.coeffs)
        return mat
    else:
        raise TypeError(f"Unsupported observable type: {type(obs)}")
