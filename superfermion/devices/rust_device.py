"""
RustDevice — the primary local simulation device.

Wraps sf-ir's Rust statevector simulation (CPU via Rayon+AVX or GPU via CUDA).
Handles gate decomposition, fusion, caching, and sampling.

This replaces the old LocalDevice + factory.py + multiple backend classes.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import hashlib
import numpy as np

from superfermion.circuit import Circuit
from superfermion.devices import DeviceCapabilities
from superfermion.results import RunResult


def _lsb_to_msb(sv: np.ndarray, n_qubits: int) -> np.ndarray:
    """Reorder statevector from LSB to MSB bit ordering via reshape+transpose."""
    if n_qubits <= 1:
        return sv
    tensor = sv.reshape([2] * n_qubits)
    tensor = tensor.transpose(list(range(n_qubits - 1, -1, -1)))
    return tensor.reshape(-1)


_CACHE_MAX_ENTRIES = 32

class RustDevice:
    """High-performance local simulator backed entirely by Rust.

    Args:
        hardware: ``"cpu"`` or ``"gpu"``. Controls where the statevector
            computation happens. CPU uses multi-threaded Rayon + AVX/NEON.
            GPU uses CUDA via cudarc (requires sm_75+ GPU).
        method: Simulation method. ``"statevector"`` (default), ``"mps"``,
            or ``"stabilizer"``. Only statevector supports GPU.
    """

    _result_cache: Dict[str, np.ndarray] = {}
    _cache_order: List[str] = []

    def __init__(
        self,
        hardware: str = "cpu",
        method: str = "statevector",
    ) -> None:
        if hardware not in ("cpu", "gpu"):
            raise ValueError(f"hardware must be 'cpu' or 'gpu', got '{hardware}'")
        if method not in ("statevector", "mps", "stabilizer"):
            raise ValueError(f"method must be 'statevector', 'mps', or 'stabilizer', got '{method}'")
        if hardware == "gpu" and method != "statevector":
            raise ValueError(f"GPU only supports method='statevector', got '{method}'")
        if hardware == "gpu":
            from superfermion._sf_core import gpu_available
            if not gpu_available():
                from superfermion._sf_core import gpu_diagnose
                raise RuntimeError(
                    f"GPU not available. Diagnostic: {gpu_diagnose()}\n"
                    f"  Use device='cpu' for local simulation."
                )
        self._hardware = hardware
        self._method = method

    def execute(self, circuit: Circuit, shots: int = 1000, **kwargs: Any) -> RunResult:
        """Execute a circuit on this device."""
        if self._method == "stabilizer":
            return self._run_stabilizer(circuit, shots, **kwargs)
        if self._method == "mps":
            return self._run_mps(circuit, shots, **kwargs)
        return self._run_statevector(circuit, shots, **kwargs)

    def capabilities(self) -> DeviceCapabilities:
        return DeviceCapabilities(
            max_qubits=32 if self._hardware == "cpu" else 30,
            native_gates=["all"],
            skip_fusion=(self._method in ("stabilizer", "mps")),
            supports_statevector=(self._method == "statevector"),
            is_simulator=True,
        )

    def _run_statevector(self, circuit: Circuit, shots: int, **kwargs: Any) -> RunResult:
        """Statevector simulation on CPU or GPU."""
        n_qubits = circuit.n_qubits
        seed = kwargs.get("seed", 42)
        return_statevector = kwargs.get("return_statevector", True)

        # Fast path: shots>0, CPU, no statevector needed — sample entirely in Rust
        # (avoids copying the full 2^n statevector from Rust to Python)
        if shots > 0 and self._hardware == "cpu" and not return_statevector:
            dag = self._prepare_dag(circuit)
            counts = dag.simulate_and_sample(shots, seed)
            return RunResult(
                counts=counts,
                statevector=None,
                shots=shots,
                circuit=circuit,
                metadata={
                    "backend": "rust-cpu",
                    "n_qubits": n_qubits,
                    "method": "statevector",
                    "sample_path": "rust-native",
                },
            )

        # Full statevector path (needed when shots=0 or statevector requested)
        fp = self._circuit_fingerprint(circuit)
        final_state = RustDevice._result_cache.get(fp)
        if final_state is None:
            dag = self._prepare_dag(circuit)

            if self._hardware == "gpu":
                sv_lsb = np.asarray(dag.simulate_on("gpu"), dtype=np.complex128)
                final_state = _lsb_to_msb(sv_lsb, n_qubits)
            else:
                final_state = np.asarray(dag.simulate_msb(), dtype=np.complex128)

            # Bounded LRU-style cache: evict oldest when full
            if len(RustDevice._result_cache) >= _CACHE_MAX_ENTRIES:
                if RustDevice._cache_order:
                    evict_key = RustDevice._cache_order.pop(0)
                    RustDevice._result_cache.pop(evict_key, None)
            RustDevice._result_cache[fp] = final_state
            RustDevice._cache_order.append(fp)

        if shots > 0:
            from superfermion.backends.turbo import sample_from_statevector
            counts = sample_from_statevector(final_state, n_qubits, shots, seed)
        else:
            counts = {}

        return RunResult(
            counts=counts,
            statevector=final_state,
            shots=shots,
            circuit=circuit,
            metadata={
                "backend": f"rust-{self._hardware}",
                "n_qubits": n_qubits,
                "method": "statevector",
            },
        )

    def _run_mps(self, circuit: Circuit, shots: int, **kwargs: Any) -> RunResult:
        """MPS simulation (CPU only, tensor network for large circuits)."""
        bond_dim = kwargs.get("bond_dim", 64)
        seed = kwargs.get("seed", 42)
        dag = circuit.to_ir()

        if shots > 0:
            counts = dag.sample_mps(bond_dim, shots, seed)
            return RunResult(
                counts=counts,
                statevector=None,
                shots=shots,
                circuit=circuit,
                metadata={
                    "backend": "rust-cpu",
                    "method": "mps",
                    "bond_dim": bond_dim,
                },
            )
        else:
            sv = dag.simulate_mps(bond_dim)
            return RunResult(
                counts={},
                statevector=np.asarray(sv, dtype=np.complex128),
                shots=0,
                circuit=circuit,
                metadata={
                    "backend": "rust-cpu",
                    "method": "mps",
                    "bond_dim": bond_dim,
                },
            )

    def _run_stabilizer(self, circuit: Circuit, shots: int, **kwargs: Any) -> RunResult:
        """Stabilizer simulation for Clifford circuits."""
        from superfermion.backends.stabilizer import maybe_clifford_dispatch
        seed = kwargs.get("seed", 42)
        result = maybe_clifford_dispatch(circuit, shots, seed=seed, require_statevector=False)
        if result is not None:
            return result
        raise RuntimeError(
            "Circuit contains non-Clifford gates — cannot use method='stabilizer'.\n"
            "  Use method='statevector' or method='mps' instead."
        )

    def _prepare_dag(self, circuit: Circuit):
        """Decompose unsupported gates and build the Rust IR DAG."""
        _RUST_DECOMP_GATES = (
            "CP", "CR1", "CPHASE", "CRY", "CH", "U1", "U2", "U3",
        )
        needs_decomp = any(
            g.name.upper() in _RUST_DECOMP_GATES
            for g in circuit._gates
        )

        if needs_decomp:
            from superfermion.backends.turbo import fuse_all_gates, decompose_for_rust
            decomposed = Circuit(circuit.n_qubits)
            decomposed._gates = decompose_for_rust(circuit._gates)
            fused = fuse_all_gates(decomposed)
            return fused.to_ir()
        else:
            return circuit.to_ir()

    @staticmethod
    def _circuit_fingerprint(circuit: Circuit) -> str:
        """Fast hash of circuit structure + parameters for caching."""
        parts = [str(circuit.n_qubits)]
        for g in circuit._gates:
            parts.append(f"{g.name}:{g.qubits}:{g.params}")
        return hashlib.md5("|".join(parts).encode()).hexdigest()

    def __repr__(self) -> str:
        return f"RustDevice(hardware='{self._hardware}', method='{self._method}')"
