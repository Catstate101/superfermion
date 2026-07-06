"""
Grover's Search — Quadratic speedup for unstructured database search.

Given a marking oracle U_f |x⟩ = (−1)^{f(x)} |x⟩, Grover's algorithm finds a
marked state in O(√N) iterations instead of the classical O(N).

Usage:
    >>> from superfermion.algorithms.grover import grover_search, GroverOracle
    >>>
    >>> # Mark state |101⟩ (index 5) in a 3-qubit register
    >>> oracle = GroverOracle.mark_state("101")
    >>> circuit, result = grover_search(oracle, n_qubits=3)
    >>> print(result["top_bitstring"])  # → "101"
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional

import numpy as np

import superfermion as sf


class GroverOracle:
    """Encodes the marking oracle for Grover's search.

    An oracle flips the phase of marked (solution) states:
        U_f |x⟩ = (−1)^{f(x)} |x⟩

    where f(x) = 1 for marked states, 0 otherwise.
    """

    def __init__(self, marked_bitstrings: List[str], n_qubits: int, label: str = ""):
        """
        Args:
            marked_bitstrings: List of binary-string solutions, e.g. ``["101", "110"]``.
            n_qubits: Number of qubits in the search register.
            label: Optional human-readable label.
        """
        self.marked = marked_bitstrings
        self.n_qubits = n_qubits
        self.label = label or f"Oracle({','.join(marked_bitstrings)})"

    @classmethod
    def mark_state(cls, bitstring: str) -> GroverOracle:
        """Convenience: create an oracle that marks a single bitstring."""
        return cls([bitstring], len(bitstring))

    def __call__(self, circuit: sf.Circuit) -> sf.Circuit:
        """Apply the phase-flip oracle to `circuit` (mutates in place)."""
        _apply_oracle(circuit, self.marked, self.n_qubits)
        return circuit


# ── Oracle application (Z-controlled multi-controlled-Z strategy) ──────

def _apply_oracle(circuit: sf.Circuit, marked: List[str], data_qubits: int):
    """Phase-flip the marked states using multi-controlled Z gates."""
    n = data_qubits
    for bitstring in marked:
        if len(bitstring) != n:
            raise ValueError(f"Bitstring '{bitstring}' length ≠ n_qubits={n}")
        # Convert 0s to X gates on control qubits → toggle to |1⟩ for MCZ
        for i, bit in enumerate(bitstring):
            if bit == "0":
                circuit.x(i)
        # Multi-controlled Z on all n qubits
        _mcz(circuit, list(range(n)), ancilla_base=n)
        # Undo the per-qubit X gates
        for i, bit in enumerate(bitstring):
            if bit == "0":
                circuit.x(i)


def _mcz(circuit: sf.Circuit, qubits: List[int], ancilla_base: int = 0):
    """Multi-controlled-Z gate using standard Toffoli cascade + ancilla."""
    n = len(qubits)
    if n == 1:
        circuit.z(qubits[0])
    elif n == 2:
        circuit.cz(qubits[0], qubits[1])
    else:
        # n-qubit controlled-Z: = H on target · MCX(ctl..., target=last) · H
        target = qubits[-1]
        ctrls = qubits[:-1]
        circuit.h(target)
        _mcx(circuit, ctrls, target, ancilla_base)
        circuit.h(target)


def _mcx(circuit: sf.Circuit, controls: List[int], target: int, ancilla_base: int = 0):
    """Multi-controlled X using linear chain of Toffolis."""
    if len(controls) == 1:
        circuit.cx(controls[0], target)
        return
    # ancilla_base must be provided by caller (data qubit count)
    if ancilla_base == 0:
        ancilla_base = circuit.n_qubits  # fallback: total qubits (caller should always pass explicitly)
    # Linear-depth implementation: cascade Toffolis
    n_ctrl = len(controls)
    for i in range(n_ctrl - 1):
        a = controls[i] if i == 0 else ancilla_base + i - 1
        b = controls[i + 1]
        c = ancilla_base + i
        circuit.toffoli(a, b, c)
    # Last Toffoli: last ancilla + last control → target
    circuit.toffoli(ancilla_base + n_ctrl - 2, controls[-1], target)
    # Uncompute ancillas
    for i in range(n_ctrl - 2, -1, -1):
        a = controls[i] if i == 0 else ancilla_base + i - 1
        b = controls[i + 1]
        c = ancilla_base + i
        circuit.toffoli(a, b, c)


# ── Diffusion operator ─────────────────────────────────────────────────

def _diffusion(circuit: sf.Circuit, n_data: int):
    """Apply the Grover diffusion operator U_s = 2|s⟩⟨s| − I.

    Implemented as H^⊗n · X^⊗n · MCZ · X^⊗n · H^⊗n.
    """
    n = n_data
    for q in range(n):
        circuit.h(q)
    for q in range(n):
        circuit.x(q)
    _mcz(circuit, list(range(n)), ancilla_base=n)
    for q in range(n):
        circuit.x(q)
    for q in range(n):
        circuit.h(q)


# ── Main algorithm ─────────────────────────────────────────────────────

def grover_search(
    oracle: GroverOracle,
    n_qubits: Optional[int] = None,
    iterations: Optional[int] = None,
    backend: str = "statevector",
    shots: int = 0,
) -> Dict[str, Any]:
    """Run Grover's search algorithm.

    Args:
        oracle: The marking oracle.
        n_qubits: Qubit count (inferred from oracle if not given).
        iterations: Number of Grover iterations. Auto-calculated if None:
                    ⌊π/4 · √(N/M)⌋ ≈ ⌊π/4 · √(2^n)⌋ for single marked state.
        backend: Simulation backend.
        shots: If > 0, sample results; if 0, return full statevector.

    Returns:
        Dict with keys: ``"top_bitstring"``, ``"probability"``,
        ``"statevector"`` (if shots=0), ``"counts"`` (if shots>0),
        ``"iterations"``, ``"n_qubits"``.
    """
    n = n_qubits or oracle.n_qubits

    # Build circuit with n qubits + extra ancillas for Toffoli decomposition
    # For n≤5 we can use a single ancilla-free approach; for larger n we need
    # 1 ancilla per 3+ controlled gate (worst-case n−2 ancillas).
    ancilla_qubits = max(0, n - 2)
    total_qubits = n + ancilla_qubits
    circuit = sf.Circuit(total_qubits, name=f"Grover(n={n})")

    # Initialize to uniform superposition
    for q in range(n):
        circuit.h(q)

    # Calculate optimal number of iterations
    if iterations is None:
        M = len(oracle.marked)
        iterations = math.floor((math.pi / 4) * math.sqrt(2 ** n / max(M, 1)))

    # Grover iterations
    for _ in range(iterations):
        oracle(circuit)      # Oracle: phase-flip marked states
        _diffusion(circuit, n)  # Diffusion: amplify

    # Run
    sim = sf.get_backend(backend)
    result = sim.run(circuit, shots=shots)

    # Extract top bitstring from statevector (only data-register qubits)
    if shots == 0 and result.statevector is not None:
        sv = np.asarray(result.statevector).flatten()
        probs = np.abs(sv) ** 2
        # Trace out ancillas: sum over ancilla subspace
        if ancilla_qubits > 0:
            probs = probs.reshape([2] * total_qubits)
            probs = probs.sum(axis=tuple(range(n, total_qubits))).flatten()

        top_idx = int(np.argmax(probs))
        top_bitstring = format(top_idx, f"0{n}b")
        top_prob = float(probs[top_idx])

        return {
            "top_bitstring": top_bitstring,
            "probability": top_prob,
            "statevector": sv if ancilla_qubits == 0 else None,
            "iterations": iterations,
            "n_qubits": n,
            "result": result,
        }

    # Sampling mode
    if result.counts:
        top_bitstring = max(result.counts, key=result.counts.get)
        total = sum(result.counts.values())
        top_prob = result.counts[top_bitstring] / total
        return {
            "top_bitstring": top_bitstring,
            "probability": top_prob,
            "counts": result.counts,
            "iterations": iterations,
            "n_qubits": n,
            "result": result,
        }

    return {
        "top_bitstring": "",
        "probability": 0.0,
        "iterations": iterations,
        "n_qubits": n,
        "result": result,
    }
