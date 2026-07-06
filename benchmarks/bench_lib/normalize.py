"""Endianness normalization across frameworks.

Convention adopted by this harness: **Big-Endian** (qubit 0 is leftmost
character of bitstring, i.e. MSB). This matches SuperFermion and PennyLane.

Qiskit uses Little-Endian (qubit 0 rightmost); we reverse its outputs
before comparing. See superfermion/bridge/__init__.py:56-57 and :113-114
for the mirror convention SuperFermion already uses when importing from /
exporting to Qiskit.
"""
from __future__ import annotations

from typing import Dict, Iterable
import numpy as np


# ---------- counts ----------

def reverse_bitstring_keys(counts: Dict[str, int] | Dict[str, float]) -> Dict[str, float]:
    """Reverse every bitstring key (needed for Qiskit -> BigEndian)."""
    out: Dict[str, float] = {}
    for k, v in counts.items():
        out[k[::-1]] = out.get(k[::-1], 0.0) + float(v)
    return out


def normalize_counts(counts: Dict[str, int] | Dict[str, float], n_qubits: int) -> Dict[str, float]:
    """Zero-pad short keys, cast to float, return shot-probabilities."""
    total = sum(counts.values())
    if total == 0:
        return {}
    return {k.zfill(n_qubits): float(v) / float(total) for k, v in counts.items()}


# ---------- statevectors ----------

def reverse_statevector_endianness(sv: np.ndarray, n_qubits: int) -> np.ndarray:
    """Reverse the qubit-order convention of a statevector of shape (2**n,).

    If the original vector has index convention 'qubit 0 = LSB' (Qiskit),
    this returns it in 'qubit 0 = MSB' convention (SF / PennyLane).

    Implementation: reshape to a rank-n tensor, reverse the axis order,
    flatten. Avoids qiskit dependency.
    """
    sv = np.asarray(sv).reshape([2] * n_qubits)
    sv = np.transpose(sv, axes=list(range(n_qubits - 1, -1, -1)))
    return sv.reshape(2 ** n_qubits)


# ---------- fidelity / TVD ----------

def fidelity(psi: np.ndarray, phi: np.ndarray) -> float:
    """|<psi|phi>|^2 for two equal-length statevectors."""
    psi = np.asarray(psi, dtype=complex)
    phi = np.asarray(phi, dtype=complex)
    overlap = np.vdot(psi, phi)
    return float(np.abs(overlap) ** 2)


def l2_distance(psi: np.ndarray, phi: np.ndarray) -> float:
    """Global-phase-insensitive L2: min over phase e^{i theta} of |psi - e^{i theta} phi|."""
    psi = np.asarray(psi, dtype=complex)
    phi = np.asarray(phi, dtype=complex)
    overlap = np.vdot(phi, psi)
    if abs(overlap) < 1e-15:
        return float(np.linalg.norm(psi - phi))
    phase = overlap / abs(overlap)
    return float(np.linalg.norm(psi - phase * phi))


def tvd(p: Dict[str, float], q: Dict[str, float]) -> float:
    """Total-variation distance between two discrete distributions keyed on bitstrings.

    Both must already be normalized to sum to 1.
    """
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)


# ---------- sanity ----------

def sanity_check_bell() -> None:
    """Assert the convention is self-consistent: a Bell state from H(0)CX(0,1)
    must produce {'00': 0.5, '11': 0.5} when counts are normalized and
    Qiskit's output is reverse_bitstring_keys'd.

    Raises AssertionError if mis-handled. Called before the sweep starts.
    """
    # Simulate by hand: |psi> = (|00> + |11>)/sqrt(2) in BigEndian indexing
    expected = {"00": 0.5, "11": 0.5}
    # Pretend these are the Qiskit (LE) counts — they would also be '00'/'11' for Bell (symmetric).
    qiskit_counts = {"00": 500, "11": 500}
    got = normalize_counts(reverse_bitstring_keys(qiskit_counts), 2)
    assert abs(got["00"] - 0.5) < 1e-9 and abs(got["11"] - 0.5) < 1e-9, got

    # Asymmetric: state |01> (qubit 0 = 0, qubit 1 = 1) in BE is "01"; in LE it's "10".
    qiskit_counts_asym = {"10": 1000}  # Qiskit little-endian repr of q1=1, q0=0
    got = normalize_counts(reverse_bitstring_keys(qiskit_counts_asym), 2)
    assert got == {"01": 1.0}, got

    # Statevector reverse: |01>_BE = index 1, |01>_LE = index 2.
    sv_le = np.zeros(4, dtype=complex)
    sv_le[2] = 1.0  # LE index for q1=1, q0=0 --> binary "10" --> decimal 2
    sv_be = reverse_statevector_endianness(sv_le, 2)
    # In BE, q0=0, q1=1 --> binary "01" --> decimal 1
    assert np.argmax(np.abs(sv_be)) == 1, (sv_le, sv_be)
