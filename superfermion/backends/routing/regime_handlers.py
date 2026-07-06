"""Regime strategies for Singularity — one class per simulation regime.

Pattern: Strategy
Each concrete strategy wraps one ``_run_*`` method previously on
``SingularityBackend``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from superfermion.circuit import Circuit
from superfermion.results import RunResult


class RegimeStrategy(ABC):
    """Abstract base for a simulation regime handler."""

    @abstractmethod
    def run(self, circuit: Circuit, n: int, shots: int, seed: int, **kwargs) -> RunResult:
        ...


class NumpyTurboStrategy(RegimeStrategy):
    """Pure numpy tensordot — fastest cold-start for n <= 10."""

    def run(self, circuit: Circuit, n: int, shots: int, seed: int, **kwargs) -> RunResult:
        from superfermion.backends.turbo import simulate_statevector_turbo, sample_from_statevector

        sv = simulate_statevector_turbo(circuit, seed)
        counts = sample_from_statevector(sv, n, shots, seed) if shots > 0 else {}

        return RunResult(
            counts=counts, statevector=sv,
            shots=shots, circuit=circuit,
            metadata={
                "backend": "singularity-turbo-numpy",
                "regime": "statevector-tensordot",
                "n_qubits": n,
                "gate_count_after_fusion": len(circuit._gates),
            }
        )


class RustRayonStrategy(RegimeStrategy):
    """Rust Rayon-parallel statevector for 11 <= n <= 32."""

    # Gates the dense Rust core doesn't parse natively
    _RUST_DECOMP_GATES = ("CP", "CR1", "CPHASE", "CRY", "CH", "U1", "U2", "U3")

    def run(self, circuit: Circuit, n: int, shots: int, seed: int, **kwargs) -> RunResult:
        from superfermion.backends.turbo import fuse_single_qubit_gates, decompose_for_rust, sample_from_statevector

        if any(g.name.upper() in self._RUST_DECOMP_GATES for g in circuit._gates):
            decomposed = Circuit(circuit.n_qubits)
            decomposed._gates = decompose_for_rust(circuit._gates)
            circuit = decomposed

        fused_circ = fuse_single_qubit_gates(circuit)
        dag = fused_circ.to_ir()
        sv = np.asarray(dag.simulate(), dtype=np.complex128)
        # Rust core is q0=LSB; SF is q0=MSB
        sv = sv.reshape([2] * n).transpose(list(range(n))[::-1]).flatten()

        counts = sample_from_statevector(sv, n, shots, seed) if shots > 0 else {}

        return RunResult(
            counts=counts, statevector=sv,
            shots=shots, circuit=circuit,
            metadata={
                "backend": "singularity-turbo-rust",
                "regime": "rayon-parallel-sv",
                "n_qubits": n,
            }
        )


class StabilizerStrategy(RegimeStrategy):
    """Aaronson-Gottesman tableau simulator for Clifford-only circuits."""

    def run(self, circuit: Circuit, n: int, shots: int, seed: int, **kwargs) -> RunResult:
        from superfermion.backends.stabilizer import StabilizerBackend
        sb = StabilizerBackend()
        result = sb.run(circuit, shots=shots, seed=seed)
        result.metadata = {
            **(result.metadata or {}),
            "backend": "singularity-stabilizer",
            "regime": "tableau",
            "n_qubits": n,
        }
        return result


class MPSDirectStrategy(RegimeStrategy):
    """MPS with direct sampling for n > 32 — no 2^n memory."""

    def run(self, circuit: Circuit, n: int, shots: int, seed: int, **kwargs) -> RunResult:
        from superfermion.backends.mps import MPSSimulatorBackend
        bond_dim = kwargs.get("bond_dim", 64)
        mps = MPSSimulatorBackend(options={"max_bond_dim": bond_dim})
        return mps.run(circuit, shots=shots, **kwargs)
