"""Data containers for quantum execution results."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from superfermion.circuit import Circuit

logger = logging.getLogger(__name__)


@dataclass
class RunResult:
    """Result of executing a quantum circuit.

    Attributes:
        counts: Measurement outcome counts (bitstring -> count).
        state: Rust-native quantum state handle (sf.State). Available for
            simulators, None for QPU hardware results.
        statevector: Legacy: raw statevector numpy array (deprecated, use .state).
        shots: Number of shots executed.
        circuit: The circuit that was executed.
        metadata: Additional execution metadata.
    """
    counts: Dict[str, int] = field(default_factory=dict)
    probabilities: Dict[str, float] = field(default_factory=dict)
    state: Optional[Any] = None
    statevector: Optional[Any] = None
    shots: int = 0
    circuit: Optional[Circuit] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def expectation(self, observable) -> float:
        """Compute expectation value of an observable.

        Uses exact computation from state (if available) or estimates
        from counts.

        Args:
            observable: Pauli observable terms as list of (paulis, coef_re, coef_im).

        Returns:
            Expectation value as float.
        """
        if self.state is not None:
            return self.state.expectation(observable)
        if self.counts:
            return _estimate_expval_from_counts(observable, self.counts)
        return 0.0

    def grad(self, observable, dag=None, param_values=None) -> dict:
        """Compute gradient of an observable.

        Uses adjoint differentiation from state (if available).

        Args:
            observable: Pauli observable terms.
            dag: QuantumDAG for gradient computation.
            param_values: Parameter values dict.

        Returns:
            Dict mapping parameter name to gradient value.
        """
        if self.state is not None:
            if dag is None or param_values is None:
                raise ValueError("grad() requires dag and param_values arguments")
            return self.state.grad(observable, dag, param_values)
        raise RuntimeError("grad() not available: no state (QPU result?)")

    def get_probabilities(self) -> Dict[str, float]:
        """Probability distribution from explicit value, statevector, or counts.

        Returns the explicitly set probabilities if non-empty, otherwise
        computes from the statevector (exact) or counts (empirical).
        """
        if self.probabilities:
            return self.probabilities
        if self.statevector is not None:
            sv = np.asarray(self.statevector, dtype=np.complex128)
            n_qubits = int(np.log2(len(sv)))
            probs = np.abs(sv) ** 2
            return {
                format(i, f"0{n_qubits}b"): float(p)
                for i, p in enumerate(probs) if p > 1e-15
            }
        if self.counts:
            total = sum(self.counts.values())
            if total > 0:
                return {k: v / total for k, v in self.counts.items()}
        return {}

    @property
    def probabilities_array(self) -> NDArray[np.float64]:
        """Probabilities as a flat numpy array in lexicographical order."""
        probs = self.get_probabilities()
        if not probs:
            return np.array([])
        n_qubits = len(next(iter(probs)))
        arr = np.zeros(2**n_qubits, dtype=np.float64)
        for b, p in probs.items():
            arr[int(b, 2)] = p
        return arr

    def plot(self, save_path: Optional[str] = None):
        """Plot the measurement counts/probabilities as a bar chart."""
        try:
            import matplotlib.pyplot as plt
            data = self.counts or self.get_probabilities()
            if not data:
                logger.warning("No data to plot.")
                return

            labels = list(data.keys())
            values = list(data.values())

            plt.figure(figsize=(10, 6))
            plt.bar(labels, values, color="#673ab7")
            plt.xlabel("Bitstrings")
            plt.ylabel("Counts" if self.counts else "Probability")
            plt.title("Quantum Execution Results")
            plt.xticks(rotation=45)

            if save_path:
                plt.savefig(save_path)
            else:
                plt.show()
        except ImportError:
            logger.warning("Matplotlib not installed. Summary:")
            logger.info("%s", self.counts or self.get_probabilities())

    def to_dict(self) -> dict:
        """Serialize the result to a plain dictionary."""
        d: dict = {
            "counts": dict(self.counts) if self.counts else {},
            "probabilities": dict(self.probabilities) if self.probabilities else {},
            "shots": self.shots,
            "metadata": dict(self.metadata) if self.metadata else {},
        }
        if self.statevector is not None:
            sv = np.asarray(self.statevector, dtype=np.complex128)
            d["statevector_real"] = sv.real.tolist()
            d["statevector_imag"] = sv.imag.tolist()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "RunResult":
        """Reconstruct a ``RunResult`` from a dictionary."""
        sv = None
        if "statevector_real" in d:
            real = np.array(d["statevector_real"], dtype=np.float64)
            imag = np.array(d.get("statevector_imag", np.zeros_like(real)), dtype=np.float64)
            sv = real + 1j * imag
        return cls(
            counts=d.get("counts", {}),
            probabilities=d.get("probabilities", {}),
            statevector=sv,
            shots=d.get("shots", 0),
            metadata=d.get("metadata", {}),
        )

    def __repr__(self) -> str:
        has_state = self.state is not None
        return (
            f"RunResult(shots={self.shots}, "
            f"outcomes={len(self.counts or self.get_probabilities())}, "
            f"has_state={has_state})"
        )


def _estimate_expval_from_counts(
    observable: list,
    counts: Dict[str, int],
) -> float:
    """Estimate expectation value from measurement counts.

    For each Pauli term, compute the parity of the measured bits
    for non-identity Pauli operators and accumulate the weighted sum.
    """
    total_shots = sum(counts.values())
    if total_shots == 0:
        return 0.0

    result = 0.0
    for paulis, coef_re, _coef_im in observable:
        term_val = 0.0
        for bitstring, count in counts.items():
            n = len(bitstring)
            parity = 0
            for q, p in enumerate(paulis):
                if p in (1, 2, 3):  # X, Y, Z all flip parity
                    if q < n:
                        parity ^= int(bitstring[q])
            eigenvalue = 1.0 - 2.0 * parity
            term_val += eigenvalue * count
        result += coef_re * term_val / total_shots
    return result
