"""
RustDevice — the primary local simulation device.

Wraps sf-ir's Rust statevector simulation (CPU via Rayon+AVX or GPU via CUDA).
Handles gate decomposition, fusion, caching, and sampling.

This replaces the old LocalDevice + factory.py + multiple backend classes.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
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


class RustDevice:
    """High-performance local simulator backed entirely by Rust.

    Args:
        hardware: ``"cpu"`` or ``"gpu"``. Controls where the statevector
            computation happens. CPU uses multi-threaded Rayon + AVX/NEON.
            GPU uses CUDA via cudarc (requires sm_75+ GPU).
        method: Simulation method. ``"statevector"`` (default), ``"mps"``,
            ``"stabilizer"``, or ``"density_matrix"``. Only statevector
            supports GPU.
    """

    _VALID_METHODS = ("statevector", "mps", "stabilizer", "density_matrix")

    def __init__(
        self,
        hardware: str = "cpu",
        method: str = "statevector",
    ) -> None:
        if hardware not in ("cpu", "gpu"):
            raise ValueError(f"hardware must be 'cpu' or 'gpu', got '{hardware}'")
        if method not in self._VALID_METHODS:
            raise ValueError(
                f"method must be one of {self._VALID_METHODS}, got '{method}'"
            )
        if hardware == "gpu" and method != "statevector":
            raise ValueError(
                f"GPU currently only has CUDA kernels for method='statevector', "
                f"got '{method}'. Use device='cpu' for MPS/stabilizer/density_matrix."
            )
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
        if self._method == "density_matrix":
            return self._run_density_matrix(circuit, shots, **kwargs)
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
        if shots > 0 and self._hardware == "cpu" and not return_statevector:
            dag = self._prepare_dag(circuit)
            counts = dag.simulate_and_sample(shots, seed)
            return RunResult(
                counts=counts,
                state=None,
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

        dag = self._prepare_dag(circuit)
        state = dag.simulate_to_state("statevector", self._hardware)

        # Also get the legacy numpy statevector for backward compat
        final_state = np.asarray(state.numpy(), dtype=np.complex128)

        if shots > 0:
            counts = state.sample(shots, seed)
        else:
            counts = {}

        return RunResult(
            counts=counts,
            state=state,
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

        state = dag.simulate_to_state("mps", "cpu", bond_dim)

        if shots > 0:
            counts = state.sample(shots, seed)
        else:
            counts = {}

        sv = None
        if shots == 0:
            try:
                sv = state.numpy()
            except Exception:
                pass

        return RunResult(
            counts=counts,
            state=state,
            statevector=sv,
            shots=shots,
            circuit=circuit,
            metadata={
                "backend": "rust-cpu",
                "method": "mps",
                "bond_dim": bond_dim,
            },
        )

    def _run_density_matrix(self, circuit: Circuit, shots: int, **kwargs: Any) -> RunResult:
        """Density matrix simulation — Rust core for both noiseless and noisy paths."""
        from superfermion.backends.density_matrix import (
            _reverse_qubits_dm,
            _dm_to_probs,
            _sample_dm,
            _apply_readout_noise,
        )
        noise_model = kwargs.get("noise_model", None)
        seed = kwargs.get("seed", 42)
        n = circuit.n_qubits
        rng = np.random.default_rng(seed)

        dag = circuit.to_ir()

        if noise_model is not None and noise_model.has_noise:
            noise_ops = noise_model.to_rust_kraus_ops(n)
            rho_vec = dag.simulate_dm_noisy(noise_ops)
            rho = rho_vec.reshape(2**n, 2**n).conj()
            rho = _reverse_qubits_dm(rho, n)
        else:
            rho_vec = dag.simulate_dm()
            rho = rho_vec.reshape(2**n, 2**n).conj()
            rho = _reverse_qubits_dm(rho, n)

        state = dag.simulate_to_state("density_matrix")

        probs = _dm_to_probs(rho)
        purity = float(np.real(np.trace(rho @ rho)))

        counts: Dict[str, int] = {}
        if shots > 0:
            counts = _sample_dm(rho, shots, rng)
            if noise_model is not None and noise_model._readout_p > 0:
                counts = _apply_readout_noise(counts, noise_model._readout_p, rng)

        return RunResult(
            counts=counts,
            state=state,
            statevector=None,
            shots=shots,
            circuit=circuit,
            metadata={
                "backend": "rust-cpu",
                "method": "density_matrix",
                "purity": purity,
                "density_matrix": rho,
                "n_qubits": n,
                "probabilities": {
                    format(i, f'0{n}b'): float(p)
                    for i, p in enumerate(probs) if p > 1e-12
                },
            },
        )

    def _run_stabilizer(self, circuit: Circuit, shots: int, **kwargs: Any) -> RunResult:
        """Stabilizer simulation for Clifford circuits."""
        seed = kwargs.get("seed", 42)
        dag = circuit.to_ir()

        try:
            state = dag.simulate_to_state("stabilizer")
        except (ValueError, RuntimeError) as e:
            raise RuntimeError(
                "Circuit contains non-Clifford gates — cannot use method='stabilizer'.\n"
                "  Use method='statevector' or method='mps' instead."
            ) from e

        if shots > 0:
            counts = state.sample(shots, seed)
        else:
            counts = {}

        return RunResult(
            counts=counts,
            state=state,
            statevector=None,
            shots=shots,
            circuit=circuit,
            metadata={
                "backend": "rust-cpu",
                "method": "stabilizer",
                "n_qubits": circuit.n_qubits,
            },
        )

    def _prepare_dag(self, circuit: Circuit):
        """Decompose unsupported gates and build the Rust IR DAG."""
        _RUST_DECOMP_GATES = (
            "CRY", "CH", "U1", "U2", "U3",
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

    def __repr__(self) -> str:
        return f"RustDevice(hardware='{self._hardware}', method='{self._method}')"
