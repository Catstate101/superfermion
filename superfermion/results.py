"""Data containers for quantum execution results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from superfermion.circuit import Circuit


@dataclass
class RunResult:
    """Result of executing a quantum circuit.

    Attributes:
        counts: Measurement outcome counts (bitstring → count).
        probabilities: Probability distribution (bitstring → float).
        statevector: Final statevector (if available from simulator).
        shots: Number of shots executed.
        circuit: The circuit that was executed.
        metadata: Additional execution metadata.
    """
    counts: Dict[str, int] = field(default_factory=dict)
    probabilities: Dict[str, float] = field(default_factory=dict)
    statevector: Optional[Any] = None # Support JAX or NumPy
    shots: int = 0
    circuit: Optional[Circuit] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def probabilities_array(self) -> NDArray[np.float64]:
        """Returns probabilities as a flat numpy array in lexicographical order."""
        if not self.circuit:
            raise ValueError("Circuit metadata missing in result.")
        n = self.circuit.n_qubits
        arr = np.zeros(2**n, dtype=np.float64)
        for b, p in self.probabilities.items():
            arr[int(b, 2)] = p
        return arr

    def plot(self, save_path: Optional[str] = None):
        """Plot the measurement counts/probabilities as a bar chart."""
        try:
            import matplotlib.pyplot as plt
            data = self.counts or self.probabilities
            if not data:
                print("No data to plot.")
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
            print("Matplotlib not installed. Summary:")
            print(self.counts or self.probabilities)

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
        return (
            f"RunResult(shots={self.shots}, "
            f"outcomes={len(self.counts or self.probabilities)})"
        )
