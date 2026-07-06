"""
Statevector quantum simulator — pure Python implementation.

Simulates quantum circuits by direct statevector evolution.
Each gate is applied as a unitary matrix multiplication on the full state.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from numpy.typing import NDArray

from superfermion.backends.base import Backend
from superfermion.circuit import Circuit, GateRecord
from superfermion.parameters import SymbolicParameter
from superfermion.results import RunResult


# Type alias for complex statevector
StateVector = NDArray[np.complex128]


def _circuit_has_complex_params(circuit: "Circuit") -> bool:
    """Return True iff any bound parameter in the circuit is a Python
    ``complex`` instance.  Used by ``simulate`` to enable the complex-step
    gradient code path without slowing the real-valued hot path.

    Note: we accept any ``complex`` instance, even one with ``imag == 0``,
    because ``float(complex(0.3, 0.0))`` raises ``TypeError`` and would
    crash the float-cast cache-key fast path.  Routing through the
    complex-aware path is correct (and almost free) when the user
    explicitly passed a complex value.
    """
    for g in circuit._gates:
        for p in g.params:
            if isinstance(p, complex):
                return True
    return False


class StatevectorBackend(Backend):
    """Pure Python statevector simulator backend."""

    def __init__(self, name: str = "statevector", options: Optional[Dict[str, Any]] = None):
        super().__init__(name, options)
        self._n_max_qubits = 25  # Practical limit for pure Python

    @property
    def n_qubits(self) -> int:
        return self._n_max_qubits

    @property
    def supported_gates(self) -> List[str]:
        return [
            "H", "X", "Y", "Z", "S", "Sdg", "T", "Tdg", "SX", "Id",
            "Rx", "Ry", "Rz", "R1", "P", "U", "CNOT", "CZ", "SWAP", "Rzz", "CCX"
        ]

    def run(self, circuit: Circuit, shots: int = 1000, **kwargs: Any) -> RunResult:
        """Run simulation on the circuit.

        For n>=18 we *do not* materialise the full probability dictionary
        (2^n entries — calling ``format()`` 1M+ times is gratuitously slow
        and rarely useful since callers usually only need ``statevector`` or
        ``counts``).  We still compute ``probs_vec`` and pass it through to
        ``RunResult`` via the statevector path; downstream consumers can
        re-derive probs as needed.
        """
        # NOTE: We deliberately do NOT auto-dispatch Clifford circuits to
        # the StabilizerBackend here — callers using StatevectorBackend
        # explicitly want a dense 2^n statevector returned via
        # ``.statevector``, which the tableau cannot produce.  Use
        # SingularityBackend (auto-router) or call StabilizerBackend
        # directly if you want the Clifford fast path.

        # 1. Simulate statevector
        final_state = self.simulate(circuit)
        n = circuit.n_qubits

        # 2. Compute probabilities
        probs_vec = np.abs(final_state) ** 2
        # Normalize to handle floating point drift
        total_p = probs_vec.sum()
        if total_p > 0:
            probs_vec = probs_vec / total_p

        # 3. Sample counts if shots > 0 (vectorised — np.bincount + format only
        # the unique outcomes that actually appeared)
        counts: Dict[str, int] = {}
        if shots > 0:
            seed = kwargs.get("seed")
            rng = np.random.default_rng(seed)
            indices = rng.choice(len(final_state), size=shots, p=probs_vec)
            unique, freq = np.unique(indices, return_counts=True)
            counts = {format(int(idx), f"0{n}b"): int(c) for idx, c in zip(unique, freq)}

        # 4. Probability dict — only build for small n to avoid the 2^n
        # ``format`` blowup on n>=18 (a 20-qubit circuit otherwise spends
        # >0.5 s formatting bitstrings nobody asked for).
        probs_dict: Dict[str, float] = {}
        if n <= 16:
            for i, p in enumerate(probs_vec):
                if p > 0:
                    probs_dict[format(i, f"0{n}b")] = float(p)

        # 5. Return RunResult
        return RunResult(
            counts=counts,
            probabilities=probs_dict,
            statevector=final_state,
            shots=shots,
            circuit=circuit,
            metadata={"backend": self.name}
        )

    def simulate(self, circuit: Circuit, initial_state: Optional[StateVector] = None) -> StateVector:
        """Core simulation loop (turbo edition).

        Optimisations vs. the naive per-gate loop:

        1. Single-qubit gate fusion via ``turbo.fuse_single_qubit_gates``.
        2. State kept in shape ``[2]*n`` throughout — no per-gate
           ``reshape(-1)`` (which forces a full state-vector copy when the
           buffer is non-contiguous after ``moveaxis``).
        3. Per-gate matrix cache keyed on ``(name, params)`` — avoids
           re-constructing identical 2x2 / 4x4 ndarrays.

        On QAOA n=20 this brings runtime from ~19 s to <1 s (>20x).

        Complex-step support: if any bound parameter is a Python ``complex``
        with non-zero imaginary part, we disable both fusion (which casts
        to float internally) and the cache key float-cast, and route the
        gate matrices through a complex-aware path.  This is the path the
        ``test_psr_gradient_accuracy`` complex-step reference relies on.
        """
        n = circuit.n_qubits
        dim = 2 ** n

        if initial_state is not None:
            state = initial_state.copy().astype(np.complex128)
        else:
            state = np.zeros(dim, dtype=np.complex128)
            state[0] = 1.0  # |0...0⟩
        # Work in tensor form: shape (2,)*n. All apply-helpers consume and
        # return the tensor form; we flatten only at the very end.
        state_t = state.reshape([2] * n) if n > 0 else state

        # Detect complex-valued bound parameters.  Used by complex-step
        # gradient checks where a single parameter is set to ``a + i*h`` so
        # ``Im(<H>) / h`` gives an exact derivative.
        has_complex = _circuit_has_complex_params(circuit)

        # ---- gate fusion (best-effort) ----
        # Fusion uses ``float(p)`` internally; skip it when complex params
        # are present so we don't lose the imaginary component.
        if has_complex:
            gates_iter = circuit._gates
        else:
            try:
                from superfermion.backends.turbo import fuse_single_qubit_gates
                gates_iter = fuse_single_qubit_gates(circuit)._gates
            except Exception:
                gates_iter = circuit._gates

        # Per-call matrix cache: (name, tuple(params)) -> np.ndarray.
        # When complex params are in play we use ``complex(p)`` for the key;
        # otherwise keep the original ``float(p)`` cast for byte-for-byte
        # backwards compatibility on the hot path.
        mat_cache: Dict[tuple, NDArray[np.complex128]] = {}

        for gate in gates_iter:
            if gate.name in ("Measure", "Barrier", "Reset"):
                continue

            # Cache key: gate name + numeric params.
            if has_complex:
                key = (gate.name, tuple(
                    complex(p) if not isinstance(p, (SymbolicParameter, str)) else 0.0+0j
                    for p in gate.params
                ))
            else:
                key = (gate.name, tuple(
                    float(p) if not isinstance(p, (SymbolicParameter, str)) else 0.0
                    for p in gate.params
                ))
            matrix = mat_cache.get(key)
            if matrix is None:
                matrix = self._get_gate_matrix(gate, allow_complex=has_complex)
                mat_cache[key] = matrix
            n_gate_qubits = len(gate.qubits)

            if n_gate_qubits == 1:
                state_t = self._apply_1q_t(state_t, matrix, gate.qubits[0])
            elif n_gate_qubits == 2:
                state_t = self._apply_2q_t(state_t, matrix, gate.qubits[0], gate.qubits[1])
            elif n_gate_qubits == 3:
                state_t = self._apply_3q_t(state_t, matrix,
                                           gate.qubits[0], gate.qubits[1], gate.qubits[2])

        # Flatten once at the end. ``reshape(-1)`` may copy if the final
        # tensor is non-contiguous — that's OK since it happens once, not
        # per gate.
        return np.ascontiguousarray(state_t).reshape(dim)

    # ------------------------------------------------------------------
    # Tensor-form apply helpers (state stays as shape (2,)*n)
    # ------------------------------------------------------------------
    @staticmethod
    def _apply_1q_t(tensor: NDArray, matrix: NDArray, qubit: int) -> NDArray:
        result = np.tensordot(matrix, tensor, axes=([1], [qubit]))
        return np.moveaxis(result, 0, qubit)

    @staticmethod
    def _apply_2q_t(tensor: NDArray, matrix: NDArray, q1: int, q2: int) -> NDArray:
        gate_ten = matrix.reshape(2, 2, 2, 2)
        result = np.tensordot(gate_ten, tensor, axes=([2, 3], [q1, q2]))
        return np.moveaxis(result, [0, 1], [q1, q2])

    @staticmethod
    def _apply_3q_t(tensor: NDArray, matrix: NDArray, q1: int, q2: int, q3: int) -> NDArray:
        gate_ten = matrix.reshape(2, 2, 2, 2, 2, 2)
        result = np.tensordot(gate_ten, tensor, axes=([3, 4, 5], [q1, q2, q3]))
        return np.moveaxis(result, [0, 1, 2], [q1, q2, q3])

    def _get_gate_matrix(self, gate: GateRecord, allow_complex: bool = False) -> NDArray[np.complex128]:
        """Unitary matrix for a gate.

        Delegates to ``gate_unitary_matrix`` — the single source of truth.
        """
        from superfermion.gates.matrices import gate_unitary_matrix

        params = [
            (complex(p) if not isinstance(p, (SymbolicParameter, str)) else 0.0 + 0j)
            if allow_complex
            else (float(p) if not isinstance(p, (SymbolicParameter, str)) else 0.0)
            for p in gate.params
        ]

        try:
            return gate_unitary_matrix(gate.name, params, use_complex_trig=allow_complex)
        except ValueError:
            # Fall back to circuit.py's universal to_unitary() for unknown gates
            return gate.to_unitary().astype(np.complex128)

    def _apply_1q(self, state: StateVector, matrix: NDArray, qubit: int, n: int) -> StateVector:
        tensor = state.reshape([2] * n)
        result = np.tensordot(matrix, tensor, axes=([1], [qubit]))
        result = np.moveaxis(result, 0, qubit)
        return result.reshape(-1)

    def _apply_2q(self, state: StateVector, matrix: NDArray, q1: int, q2: int, n: int) -> StateVector:
        tensor = state.reshape([2] * n)
        gate_ten = matrix.reshape(2, 2, 2, 2)
        result = np.tensordot(gate_ten, tensor, axes=([2, 3], [q1, q2]))
        result = np.moveaxis(result, [0, 1], [q1, q2])
        return result.reshape(-1)

    def _apply_3q(self, state: StateVector, matrix: NDArray, q1: int, q2: int, q3: int, n: int) -> StateVector:
        tensor = state.reshape([2] * n)
        gate_ten = matrix.reshape(2, 2, 2, 2, 2, 2)
        result = np.tensordot(gate_ten, tensor, axes=([3, 4, 5], [q1, q2, q3]))
        result = np.moveaxis(result, [0, 1, 2], [q1, q2, q3])
        return result.reshape(-1)
