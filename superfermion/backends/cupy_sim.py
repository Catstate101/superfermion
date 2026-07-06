"""
GPU-accelerated Statevector simulator using CuPy (CUDA-native).

This backend provides direct GPU simulation on Windows/Linux systems 
with NVIDIA hardware, bypassing the complexities of native JAX-GPU 
configuration on Windows.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

try:
    import cupy as cp
except ImportError:
    cp = None

from superfermion.backends.base import Backend
from superfermion.circuit import Circuit, GateRecord
from superfermion.parameters import SymbolicParameter
from superfermion.results import RunResult


class CupyBackend(Backend):
    """GPU statevector simulator backend using CuPy."""

    def __init__(self, name: str = "cuda", options: Optional[Dict[str, Any]] = None):
        super().__init__(name, options)
        if cp is None:
            raise ImportError(
                "CuPy is not installed. To use GPU-accelerated simulation, "
                "install it with: pip install cupy-cudaXXx"
            )
        self._n_max_qubits = 32  # GPU can handle more qubits than CPU

    @property
    def n_qubits(self) -> int:
        return self._n_max_qubits

    @property
    def supported_gates(self) -> List[str]:
        return [
            "H", "X", "Y", "Z", "S", "Sdg", "T", "Tdg", "SX", "Id",
            "Rx", "Ry", "Rz", "R1", "P", "U", "U3", "CNOT", "CZ", "SWAP", "Rzz", "CCX", "CU", "CU3", "CP"
        ]

    # GPU VRAM guard: complex64 statevector uses 8 bytes per amplitude.
    # Allow up to ~60% of available GPU memory (leaves room for gate matrices,
    # scratch buffers, and CuPy overhead).
    _GPU_MEM_HEADROOM = 0.60

    @staticmethod
    def _sv_fits_on_gpu(n: int) -> bool:
        """Return True if a complex64 SV for n qubits fits in available GPU memory."""
        try:
            mem_info = cp.cuda.Device(0).mem_info  # (free_bytes, total_bytes)
            free_bytes = mem_info[0]
            required = (2 ** n) * 8  # complex64 = 8 bytes
            return required < CupyBackend._GPU_MEM_HEADROOM * free_bytes
        except Exception:
            return n <= 26  # static fallback for 2 GB VRAM

    def run(self, circuit: Circuit, shots: int = 1000, **kwargs: Any) -> RunResult:
        """Run GPU simulation on the circuit."""
        n = circuit.n_qubits

        # NOTE: no Clifford auto-dispatch — sf.singularity routes Clifford
        # circuits explicitly.

        if n > 32:
            raise ValueError(f"Statevector backend 'cuda' cannot handle {n} qubits. Use 'cuda_mps' for industrial scale.")
        if not self._sv_fits_on_gpu(n):
            raise MemoryError(
                f"[sf.cuda] Insufficient GPU VRAM for {n}-qubit statevector. "
                f"Use 'cuda_mps' or reduce circuit size."
            )

        seed = kwargs.get("seed", 42)

        # 1. Simulate statevector on GPU
        final_state_gpu = self.simulate(circuit)

        # 2. Sample counts
        probs_vec_gpu = cp.abs(final_state_gpu) ** 2
        counts = {}
        if shots > 0:
            cp.random.seed(seed)
            indices_gpu = cp.random.choice(len(final_state_gpu), size=shots, p=probs_vec_gpu)
            indices_np = cp.asnumpy(indices_gpu)
            fmt = f"0{n}b"
            for idx in indices_np:
                bitstring = format(int(idx), fmt)
                counts[bitstring] = counts.get(bitstring, 0) + 1

        return RunResult(
            counts=counts,
            probabilities={},
            statevector=cp.asnumpy(final_state_gpu),
            shots=shots,
            circuit=circuit,
            metadata={"backend": self.name, "device": "GPU"}
        )

    def simulate(self, circuit: Circuit, initial_state: Optional[cp.ndarray] = None) -> cp.ndarray:
        """Core simulation loop on GPU."""
        n = circuit.n_qubits
        dim = 2 ** n

        if initial_state is not None:
            state = cp.array(initial_state).astype(cp.complex64)
        else:
            state = cp.zeros(dim, dtype=cp.complex64)
            state[0] = 1.0  # |0...0⟩

        for gate in circuit._gates:
            if gate.name in ("MEASURE", "BARRIER", "RESET"):
                continue

            matrix = cp.array(gate.to_unitary(), dtype=cp.complex64)
            n_gate_qubits = len(gate.qubits)

            if n_gate_qubits == 1:
                state = self._apply_1q(state, matrix, gate.qubits[0], n)
            elif n_gate_qubits == 2:
                state = self._apply_2q(state, matrix, gate.qubits[0], gate.qubits[1], n)
            elif n_gate_qubits == 3:
                state = self._apply_3q(state, matrix, gate.qubits[0], gate.qubits[1], gate.qubits[2], n)

        return state

    def _get_gate_matrix(self, gate: GateRecord) -> cp.ndarray:
        """Generate unitary matrix for a gate (uses math/np then transfers to cp)."""
        import numpy as np
        name = gate.name
        params = [
            float(p) if not isinstance(p, (SymbolicParameter, str)) else 0.0
            for p in gate.params
        ]

        if name == "ID": return np.eye(2, dtype=np.complex64)
        elif name == "H": return np.array([[1, 1], [1, -1]], dtype=np.complex64) / math.sqrt(2)
        elif name == "X": return np.array([[0, 1], [1, 0]], dtype=np.complex64)
        elif name == "Y": return np.array([[0, -1j], [1j, 0]], dtype=np.complex64)
        elif name == "Z": return np.array([[1, 0], [0, -1]], dtype=np.complex64)
        elif name == "S": return np.array([[1, 0], [0, 1j]], dtype=np.complex64)
        elif name == "SDG": return np.array([[1, 0], [0, -1j]], dtype=np.complex64)
        elif name == "T": return np.array([[1, 0], [0, np.exp(1j * math.pi / 4)]], dtype=np.complex64)
        elif name == "TDG": return np.array([[1, 0], [0, np.exp(-1j * math.pi / 4)]], dtype=np.complex64)
        elif name == "SX": return np.array([[0.5 + 0.5j, 0.5 - 0.5j], [0.5 - 0.5j, 0.5 + 0.5j]], dtype=np.complex64)
        elif name == "RX":
            theta = params[0]
            c, s = math.cos(theta/2), math.sin(theta/2)
            return np.array([[c, -1j * s], [-1j * s, c]], dtype=np.complex64)
        elif name == "RY":
            theta = params[0]
            c, s = math.cos(theta/2), math.sin(theta/2)
            return np.array([[c, -s], [s, c]], dtype=np.complex64)
        elif name == "RZ":
            theta = params[0]
            return np.array([[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]], dtype=np.complex64)
        elif name == "CNOT":
            return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=np.complex64)
        elif name == "CZ":
            return np.diag([1, 1, 1, -1]).astype(np.complex64)
        elif name == "SWAP":
            return np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.complex64)
        elif name == "RZZ":
            theta = params[0]
            em, ep = np.exp(-1j * theta / 2), np.exp(1j * theta / 2)
            return np.diag([em, ep, ep, em]).astype(np.complex64)
        elif name == "CCX":
            m = np.eye(8, dtype=np.complex64)
            m[6, 6] = 0; m[7, 7] = 0; m[6, 7] = 1; m[7, 6] = 1
            return m
        return np.eye(2, dtype=np.complex64)

    def _apply_1q(self, state: cp.ndarray, matrix: cp.ndarray, qubit: int, n: int) -> cp.ndarray:
        tensor = state.reshape([2] * n)
        result = cp.tensordot(matrix, tensor, axes=([1], [qubit]))
        result = cp.moveaxis(result, 0, qubit)
        return result.reshape(-1)

    def _apply_2q(self, state: cp.ndarray, matrix: cp.ndarray, q1: int, q2: int, n: int) -> cp.ndarray:
        tensor = state.reshape([2] * n)
        gate_ten = matrix.reshape(2, 2, 2, 2)
        result = cp.tensordot(gate_ten, tensor, axes=([2, 3], [q1, q2]))
        result = cp.moveaxis(result, [0, 1], [q1, q2])
        return result.reshape(-1)

    def _apply_3q(self, state: cp.ndarray, matrix: cp.ndarray, q1: int, q2: int, q3: int, n: int) -> cp.ndarray:
        tensor = state.reshape([2] * n)
        gate_ten = matrix.reshape(2, 2, 2, 2, 2, 2)
        result = cp.tensordot(gate_ten, tensor, axes=([3, 4, 5], [q1, q2, q3]))
        result = cp.moveaxis(result, [0, 1, 2], [q1, q2, q3])
        return result.reshape(-1)
