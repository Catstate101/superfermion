"""
SuperFermion Singularity — thin Facade composing Router + Cache + RegimeStrategies.

Was 315 lines with 10 responsibilities. Now delegates to:
  - SingularityRouter   → regime decision
  - TopologyCache       → statevector caching
  - RegimeStrategy      → one class per simulation path
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from superfermion.backends.base import Backend
from superfermion.circuit import Circuit
from superfermion.results import RunResult
from superfermion.backends.routing.topology_cache import TopologyCache
from superfermion.backends.routing.singularity_router import SingularityRouter, Regime
from superfermion.backends.routing.regime_handlers import (
    NumpyTurboStrategy,
    RustRayonStrategy,
    StabilizerStrategy,
    MPSDirectStrategy,
)


class SingularityBackend(Backend):
    """Master backend — routes circuits to the optimal simulation regime."""

    def __init__(self, name: str = "singularity", options: Optional[Dict[str, Any]] = None):
        super().__init__(name, options)
        self._n_max_qubits = 1000
        self._router = SingularityRouter()
        self._cache = TopologyCache()
        self._strategies = {
            Regime.NUMPY_TURBO: NumpyTurboStrategy(),
            Regime.RUST_RAYON: RustRayonStrategy(),
            Regime.STABILIZER: StabilizerStrategy(),
            Regime.MPS_DIRECT: MPSDirectStrategy(),
        }

    @property
    def n_qubits(self) -> int:
        return self._n_max_qubits

    @property
    def supported_gates(self) -> List[str]:
        return ["H", "X", "Y", "Z", "S", "SDG", "T", "TDG", "SX",
                "RX", "RY", "RZ", "CX", "CNOT", "CZ", "SWAP", "RZZ",
                "U", "P", "CP", "CCX"]

    def run(self, circuit: Circuit, shots: int = 1000, **kwargs: Any) -> RunResult:
        n = circuit.n_qubits
        seed = kwargs.pop("seed", 42)

        # 1. Check topology cache
        fp = TopologyCache.fingerprint(circuit)
        cached = self._cache.get(fp)
        if cached is not None:
            statevector, meta = cached
            if shots > 0:
                from superfermion.backends.turbo import sample_from_statevector
                counts = sample_from_statevector(statevector, n, shots, seed)
            else:
                counts = {}
            return RunResult(
                counts=counts, statevector=statevector,
                shots=shots, circuit=circuit,
                metadata={**meta, "cache_hit": True}
            )

        # 2. Decide regime
        regime = self._router.decide(circuit)

        # 3. Fuse gates for dense regimes only.
        # Skip MPS (fusion breaks native fast paths) and stabilizer
        # (fusion creates U gates that aren't Clifford-recognizable).
        if regime not in (Regime.MPS_DIRECT, Regime.STABILIZER):
            from superfermion.backends.turbo import fuse_single_qubit_gates
            circuit = fuse_single_qubit_gates(circuit)

        # 4. Execute via strategy
        strategy = self._strategies[regime]
        result = strategy.run(circuit, n, shots, seed, **kwargs)

        # 5. Cache dense statevectors (not MPS placeholders)
        sv = getattr(result, "statevector", None)
        if (
            sv is not None
            and hasattr(sv, "size")
            and sv.size == (1 << n)
            and n <= 24
        ):
            self._cache.put(fp, np.asarray(sv, dtype=np.complex128), result.metadata)

        return result

    def run_dag(self, dag: Any, shots: int = 0, seed: int = 42, **kwargs) -> RunResult:
        """High-speed path for pre-compiled Rust IR (QuantumDAG)."""
        n = dag.n_qubits()
        sv_raw = np.asarray(dag.simulate(), dtype=np.complex128)
        sv = sv_raw.reshape([2] * n).transpose(list(range(n))[::-1]).flatten()

        from superfermion.backends.turbo import sample_from_statevector
        counts = sample_from_statevector(sv, n, shots, seed) if shots > 0 else {}

        return RunResult(statevector=sv, counts=counts, metadata={"backend": "singularity_rust_dag"})

    def expval(self, circuit: Circuit, observable, **kwargs):
        """Compute <psi|O|psi> via the fastest path available."""
        from superfermion.backends.stabilizer import StabilizerBackend, is_clifford_circuit
        if is_clifford_circuit(circuit):
            return StabilizerBackend().expval(circuit, observable)

        from superfermion.backends.mps import MPSSimulatorBackend
        try:
            mps = MPSSimulatorBackend(options={"max_bond_dim": kwargs.get("bond_dim", 64)})
            return mps.expval(circuit, observable, max_bond=kwargs.get("bond_dim", 64))
        except (ValueError, NotImplementedError, RuntimeError):
            import logging
            logging.getLogger(__name__).debug("MPS expval failed, falling back to dense", exc_info=True)

        from superfermion.observables.core import SparsePauliOp
        sv = self.run(circuit, shots=0).statevector
        if isinstance(observable, str):
            observable = SparsePauliOp.from_dict({observable: 1.0})
        return float(np.real(observable._fast_expval(np.asarray(sv, dtype=np.complex128))))

    def grad(self, circuit: Circuit, observable, param_names, param_values, **kwargs):
        """Adjoint differentiation — ~30x faster than parameter-shift."""
        from superfermion.qml.gradient.adjoint import adjoint_grad_vector
        return adjoint_grad_vector(circuit, observable, param_names, param_values)

    def pre_bake(self, circuit: Circuit):
        """Pre-compute and cache the circuit simulation for instant re-runs."""
        fp = TopologyCache.fingerprint(circuit)
        if self._cache.get(fp) is None:
            self.run(circuit, shots=0)


def register_singularity():
    from superfermion.backends.registry import BackendRegistry
    backend = SingularityBackend()
    BackendRegistry.register("singularity", backend)
