"""SingularityRouter — decides numpy/rust/mps/stabilizer regime for a circuit."""

from __future__ import annotations

from enum import Enum, auto

from superfermion.circuit import Circuit


class Regime(Enum):
    """Simulation regime for Singularity."""
    NUMPY_TURBO = auto()
    RUST_RAYON = auto()
    MPS_DIRECT = auto()
    STABILIZER = auto()


class SingularityRouter:
    """Routes circuits to the optimal simulation regime based on qubit count and RAM.

    Replaces the inline routing logic scattered across ``SingularityBackend.run``.
    """

    # Thresholds
    NUMPY_MAX = 10
    RUST_MAX = 32
    CLIFFORD_FASTPATH_N = 22  # above this, Clifford -> stabilizer even with shots==0

    def __init__(self, headroom: float = 0.6):
        self._headroom = headroom

    @staticmethod
    def dense_sv_fits(n: int, headroom: float = 0.6) -> bool:
        """True iff a complex128 statevector of 2^n amplitudes fits within
        ``headroom`` of currently-available RAM. Budgets 3x the raw SV size
        (1 for SV, 1 for Rust workspace, 1 for Python round-trip).
        """
        try:
            import psutil
            avail = psutil.virtual_memory().available
        except Exception:
            return n <= 28
        required = 3 * (1 << n) * 16  # bytes
        return required <= headroom * avail

    def decide(self, circuit: Circuit) -> Regime:
        """Determine the optimal simulation regime for this circuit."""
        n = circuit.n_qubits

        # Clifford fast path: poly-time tableau
        if n > self.CLIFFORD_FASTPATH_N:
            from superfermion.backends.stabilizer import is_clifford_circuit
            if is_clifford_circuit(circuit):
                return Regime.STABILIZER

        if n <= self.NUMPY_MAX:
            return Regime.NUMPY_TURBO
        elif n <= self.RUST_MAX and self.dense_sv_fits(n, self._headroom):
            return Regime.RUST_RAYON
        else:
            return Regime.MPS_DIRECT
