"""
RustDevice — the primary local simulation device.

Wraps sf-ir's Rust statevector simulation (CPU via Rayon+AVX or GPU via CUDA).
Handles gate decomposition, fusion, caching, and sampling.

This replaces the old LocalDevice + factory.py + multiple backend classes.
"""

from __future__ import annotations

import os
import warnings
from typing import Any, Dict, List, Optional
import numpy as np

from superfermion.circuit import Circuit
from superfermion.devices import DeviceCapabilities
from superfermion.results import LazyDict, RunResult

# Above this qubit count the exact |c|² probability table costs more to
# build than typical runs save by reading it — defer it until first use.
_EAGER_PROBS_MAX_QUBITS = 12

# MPS runs warn when the discarded-weight fraction exceeds this threshold
# (override with SF_MPS_WARN_WEIGHT); SF_MPS_STRICT=1 turns it into an error.
_MPS_WARN_WEIGHT_DEFAULT = 1e-10


def _check_mps_truncation(state: Any, bond_dim: int) -> None:
    """Surface MPS bond-dimension truncation instead of failing silently.

    The Rust core reports the discarded-weight fraction Σε for every step in
    which the QR bond cap removed real weight.  Warn when it exceeds
    ``SF_MPS_WARN_WEIGHT`` (default 1e-10), or raise when ``SF_MPS_STRICT=1``
    — results from a truncated MPS are approximate.
    """
    get_report = getattr(state, "truncation_report", None)
    if get_report is None:
        return
    try:
        info = get_report()
    except Exception:
        return
    if not info:
        return
    discarded = float(info.get("discarded_weight", 0.0))
    threshold = float(os.environ.get("SF_MPS_WARN_WEIGHT", _MPS_WARN_WEIGHT_DEFAULT))
    if discarded <= threshold:
        return
    msg = (
        f"MPS simulation truncated {discarded:.3e} of the state weight at "
        f"bond_dim={bond_dim} ({info.get('truncation_events', 0)} events, max "
        f"single-step {float(info.get('max_discarded_weight', 0.0)):.3e}); "
        f"results are approximate (fidelity lower bound "
        f"{float(info.get('fidelity_lower_bound', 1.0)):.6f}). "
        "Increase bond_dim for a more accurate state."
    )
    if os.environ.get("SF_MPS_STRICT") == "1":
        raise RuntimeError(msg)
    warnings.warn(msg, RuntimeWarning, stacklevel=3)


def _lsb_to_msb(sv: np.ndarray, n_qubits: int) -> np.ndarray:
    """Reorder statevector from LSB to MSB bit ordering via reshape+transpose."""
    if n_qubits <= 1:
        return sv
    tensor = sv.reshape([2] * n_qubits)
    tensor = tensor.transpose(list(range(n_qubits - 1, -1, -1)))
    return tensor.reshape(-1)


def _call_dm_noisy_state(dag: Any, noise_ops: Any, noise_ops_2q: Any) -> Any:
    """Native noisy-DM call; forwards 2q channels when present.

    Tolerates builds whose binding predates the optional 2q parameter
    (TypeError -> documented RuntimeWarning + 1q-only call) so a source
    checkout never silently drops 2q noise on an older extension.
    """
    if noise_ops_2q:
        try:
            return dag.simulate_dm_noisy_state(noise_ops, noise_ops_2q)
        except TypeError:
            warnings.warn(
                "This superfermion build does not support two-qubit noise "
                "channels on the native density-matrix path; they are ignored "
                "for this run (rebuild the extension to apply them).",
                RuntimeWarning,
                stacklevel=3,
            )
    return dag.simulate_dm_noisy_state(noise_ops)


def _call_dm_noisy_vec(dag: Any, noise_ops: Any, noise_ops_2q: Any) -> Any:
    """Legacy vec-returning fallback, 2q-aware in the same way."""
    if noise_ops_2q:
        try:
            return dag.simulate_dm_noisy(noise_ops, noise_ops_2q)
        except TypeError:
            warnings.warn(
                "This superfermion build does not support two-qubit noise "
                "channels on the native density-matrix path; they are ignored "
                "for this run (rebuild the extension to apply them).",
                RuntimeWarning,
                stacklevel=3,
            )
    return dag.simulate_dm_noisy(noise_ops)


def _is_dynamic_circuit(circuit: Circuit) -> bool:
    """True when a circuit needs mid-circuit (dynamic) semantics.

    Triggers on: any ``RESET``, any classically conditioned gate
    (``c_if``), or a ``MEASURE`` whose qubit is touched again later
    (gate, reset, or second measurement) — i.e. a measure that is not
    terminal. Purely terminal-measure circuits keep the fast exact path.
    """
    circuit._ensure_gates()
    gates = circuit._gates
    for i, g in enumerate(gates):
        name = g.name.upper()
        if g.condition is not None:
            return True
        if name == "RESET":
            return True
        if name == "MEASURE":
            q = g.qubits[0]
            for later in gates[i + 1:]:
                if q in later.qubits:
                    return True
    return False


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
        # Gate/readout noise is only applied on the density-matrix path;
        # warn instead of silently returning noiseless results elsewhere.
        noise_model = kwargs.get("noise_model", None)
        if (
            noise_model is not None
            and getattr(noise_model, "has_noise", False)
            and self._method != "density_matrix"
        ):
            warnings.warn(
                f"noise_model is ignored by method='{self._method}': gate and "
                "readout noise are only applied on the density-matrix path. "
                "Use method='density_matrix' for noisy simulations.",
                RuntimeWarning,
                stacklevel=2,
            )
        # Mid-circuit (dynamic) circuits: measure/reset/feed-forward change
        # the state during the circuit, so they need per-shot trajectory
        # replay instead of a single unitary evolution + terminal sampling.
        if _is_dynamic_circuit(circuit):
            return self._run_dynamic(circuit, shots, **kwargs)
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

    def _run_dynamic(self, circuit: Circuit, shots: int, **kwargs: Any) -> RunResult:
        """Mid-circuit (dynamic-circuit) simulation via per-shot trajectories.

        Each shot replays the circuit in the Rust trajectory engine: every
        Measure op samples + collapses its qubit and stores the outcome in
        the classical register; conditioned gates (``c_if``) execute only
        when their register test passes; Reset ops collapse the qubit back
        to |0>. Final bitstrings are sampled over all qubits.

        Requires ``shots > 0`` and ``method='statevector'`` on CPU — the
        collapsed mixture cannot be represented as a pure state, matching
        PennyLane's one-shot semantics for finite shots.
        """
        if self._hardware != "cpu":
            raise RuntimeError(
                "Mid-circuit measurement circuits (measure + feed-forward / "
                "reset) currently require device='cpu' — the per-shot "
                "trajectory engine is CPU-only.\n"
                "  Fix: sf.run(circuit, device='cpu', shots=...)"
            )
        if self._method != "statevector":
            raise RuntimeError(
                "Mid-circuit measurement circuits (measure + feed-forward / "
                "reset) currently require method='statevector'.\n"
                f"  Got method={self._method!r}. "
                "Fix: sf.run(circuit, method='statevector', shots=...)"
            )
        if shots <= 0:
            raise RuntimeError(
                "Mid-circuit measurement circuits (measure + feed-forward / "
                "reset) require finite shots: the collapsed per-shot "
                "mixture cannot be represented by a single pure state.\n"
                "  Fix: sf.run(circuit, shots>0) "
                "(PennyLane parity: finite-shots one-shot mode)"
            )
        seed = kwargs.get("seed", 42)
        dag = self._prepare_dag(circuit)
        counts = dag.simulate_dynamic(shots, seed)
        probabilities = {k: v / shots for k, v in counts.items()}
        return RunResult(
            counts=counts,
            probabilities=probabilities,
            state=None,
            statevector=None,
            shots=shots,
            circuit=circuit,
            metadata={
                "backend": "rust-cpu",
                "n_qubits": circuit.n_qubits,
                "method": "statevector",
                "sample_path": "rust-dynamic-trajectories",
                "dynamic": True,
            },
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
                probabilities={k: v / shots for k, v in counts.items()},
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

        def _exact_probabilities() -> Dict[str, float]:
            probs = np.abs(final_state) ** 2
            return {
                format(i, f"0{n_qubits}b"): float(p)
                for i, p in enumerate(probs)
                if p > 1e-15
            }

        probabilities: Dict[str, float] = (
            _exact_probabilities()
            if n_qubits <= _EAGER_PROBS_MAX_QUBITS
            else LazyDict(_exact_probabilities)
        )

        return RunResult(
            counts=counts,
            probabilities=probabilities,
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
        _check_mps_truncation(state, bond_dim)

        if shots > 0:
            counts = state.sample(shots, seed)
        else:
            counts = {}

        sv = None
        if shots == 0:
            if circuit.n_qubits > 26:
                raise MemoryError(
                    "MPS simulation with shots=0 densifies the state into a full "
                    f"2^{circuit.n_qubits}-amplitude statevector "
                    f"(≈ {2 ** circuit.n_qubits * 16 / 2**30:.1f} GiB). "
                    "Use shots>0 to sample directly from the tensor network instead."
                )
            try:
                sv = state.numpy()
            except Exception:
                pass
            if sv is not None:
                # The MPS handle returns the raw survivor vector: after
                # truncation its norm² is Π(1−ε) < 1.  Present the physical
                # state (sample() normalizes before measuring); the raw
                # truncation truth stays in the RuntimeWarning above and in
                # ``state.truncation_report()``.
                nrm = float(np.sqrt(np.real(np.vdot(sv, sv))))
                if nrm > 0.0:
                    sv = sv / nrm

        if shots > 0:
            probabilities = {k: v / shots for k, v in counts.items()}
        elif sv is not None:
            probabilities = {
                format(i, f"0{circuit.n_qubits}b"): float(p)
                for i, p in enumerate(np.abs(sv) ** 2)
                if p > 1e-15
            }
        else:
            probabilities = {}

        return RunResult(
            counts=counts,
            probabilities=probabilities,
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
            noise_ops_2q = (
                noise_model.to_rust_kraus_ops_2q()
                if hasattr(noise_model, "to_rust_kraus_ops_2q")
                else []
            )
            if noise_ops or noise_ops_2q:
                # Native handle: the noisy DM data stays in Rust and is moved
                # into the State handle (zero-copy), avoiding the numpy
                # round-trip (simulate_dm_noisy -> from_dm). Older wheels
                # without the binding fall back to the vec + Python proxy.
                try:
                    # Native handle: the noisy DM data stays in Rust and is
                    # moved into the State handle; the binding also returns
                    # the public row-major rho (one in-Rust gather) so the
                    # Python numpy() + _reverse_qubits_dm round-trip is
                    # skipped entirely. Optional 2q channels ride along on
                    # builds that support the parameter (helpers warn +
                    # degrade instead of silently dropping them otherwise).
                    state, rho_flat = _call_dm_noisy_state(dag, noise_ops, noise_ops_2q)
                    rho = np.asarray(rho_flat, dtype=np.complex128).reshape(2**n, 2**n)
                except AttributeError:
                    rho_vec = _call_dm_noisy_vec(dag, noise_ops, noise_ops_2q)
                    rho_le = rho_vec.reshape(2**n, 2**n).conj()
                    rho = _reverse_qubits_dm(rho_le, n)
                    # State handle backed by the noisy rho — no duplicate
                    # noiseless simulate_to_state("density_matrix") call.
                    from superfermion.devices.noisy_dm_state import NoisyDensityMatrixState
                    state = NoisyDensityMatrixState(rho_le, n)
                noisy_state_ready = True
            else:
                # Readout-only noise: the quantum state is unchanged, so the
                # Rust handle is kept (unchanged-path contract).
                rho_vec = dag.simulate_dm()
                rho = rho_vec.reshape(2**n, 2**n).conj()
                rho = _reverse_qubits_dm(rho, n)
                state = dag.simulate_to_state("density_matrix")
                noisy_state_ready = False
        else:
            rho_vec = dag.simulate_dm()
            rho = rho_vec.reshape(2**n, 2**n).conj()
            rho = _reverse_qubits_dm(rho, n)
            state = dag.simulate_to_state("density_matrix")
            noisy_state_ready = False

        probs = _dm_to_probs(rho)
        # Hermitian rho: Tr(rho²) = Σ|rho_ij|² — Frobenius norm squared,
        # O(d²) instead of the O(d³) rho @ rho matmul. np.vdot conjugates
        # its first argument, so one BLAS pass replaces abs+square+sum.
        purity = float(np.vdot(rho, rho).real)
        probabilities = {
            format(i, f'0{n}b'): float(p)
            for i, p in enumerate(probs) if p > 1e-12
        }

        counts: Dict[str, int] = {}
        if shots > 0:
            counts = _sample_dm(rho, shots, rng)
            if noise_model is not None and noise_model._readout_p > 0:
                counts = _apply_readout_noise(counts, noise_model._readout_p, rng)

        return RunResult(
            counts=counts,
            probabilities=probabilities,
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
                "probabilities": probabilities,
                # Private marker: tells _execute_with_noisy_state that the
                # state handle is already backed by the noisy rho, so its
                # rebuild (from_dm round-trip) is skipped. Popped by the
                # wrapper and never user-visible.
                **({"_sf_noisy_state_ready": True} if noisy_state_ready else {}),
            },
        )

    def _run_stabilizer(self, circuit: Circuit, shots: int, **kwargs: Any) -> RunResult:
        """Stabilizer simulation for Clifford circuits."""
        seed = kwargs.get("seed", 42)
        # Hard cap of the Rust tableau: raise a clean ValueError instead of
        # the pyo3 PanicException (a BaseException that `except Exception`
        # cannot catch) surfacing from crates/sf-ir/src/stabilizer.rs.
        if circuit.n_qubits > 1024:
            raise ValueError(
                "method='stabilizer' supports at most 1024 qubits "
                f"(got {circuit.n_qubits}): the Rust tableau has a hard cap.\n"
                "  Use method='mps' or method='statevector' for larger "
                "circuits."
            )
        dag = circuit.to_ir()

        try:
            state = dag.simulate_to_state("stabilizer")
        except (ValueError, RuntimeError) as e:
            raise RuntimeError(
                "Circuit contains non-Clifford gates — cannot use method='stabilizer'.\n"
                "  Use method='statevector' or method='mps' instead."
            ) from e
        except BaseException as e:  # noqa: BLE001 - PanicException escapes Exception
            if type(e).__name__ == "PanicException":
                raise ValueError(
                    f"method='stabilizer' failed in the Rust tableau: {e}.\n"
                    "  The tableau supports at most 1024 qubits; use "
                    "method='mps' or 'statevector' for larger circuits."
                ) from e
            raise

        if shots > 0:
            counts = state.sample(shots, seed)
        else:
            counts = {}

        return RunResult(
            counts=counts,
            probabilities={k: v / shots for k, v in counts.items()} if shots > 0 else {},
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


# ─────────────────────────────────────────────────────────────────────────
# Additive fix: noisy density-matrix state handle.
#
# `_run_density_matrix` builds `RunResult.state` via
# `dag.simulate_to_state("density_matrix")`, which has no noise path, so
# the state handle (and therefore `RunResult.expectation()`/`variance()`,
# `state.purity()`, ...) silently reflected the *noiseless* state whenever
# a gate-noise model was applied.  The noisy rho is already computed
# correctly and stored in `result.metadata["density_matrix"]`; when gate
# noise is present, swap the handle for one backed by that noisy rho.
# Preferred: the native `State.from_dm` binding (rebuilt wheels only).
# Fallback: `NoisyDensityMatrixState`, a pure-Python drop-in handle.
# No code above is modified.
# ─────────────────────────────────────────────────────────────────────────
def _execute_with_noisy_state(execute):
    """Wrap ``RustDevice.execute`` so noisy density-matrix runs return a
    state handle backed by the noisy rho (see module-level note)."""

    def wrapper(self, circuit: Circuit, shots: int = 1000, **kwargs: Any) -> RunResult:
        result = execute(self, circuit, shots, **kwargs)
        noise_model = kwargs.get("noise_model", None)
        has_gate_noise = bool(
            noise_model is not None
            and (getattr(noise_model, "_1q_kraus", None) or getattr(noise_model, "_2q_kraus", None))
        )
        if (
            has_gate_noise
            and self._method == "density_matrix"
            and result.state is not None
            and "density_matrix" in result.metadata
            and not result.metadata.pop("_sf_noisy_state_ready", False)
        ):
            n_qubits = result.metadata.get("n_qubits", circuit.n_qubits)
            from superfermion.devices.noisy_dm_state import (
                NoisyDensityMatrixState,
                _to_little_endian,
            )
            rho_le = _to_little_endian(
                np.asarray(result.metadata["density_matrix"]), n_qubits
            )
            try:
                # Native handle (Design B): `State.from_dm` exists only on
                # wheels rebuilt with the additive Rust binding; older
                # wheels fall back to the pure-Python proxy below.
                from superfermion._sf_core import State as _RustState
                result.state = _RustState.from_dm(
                    np.ascontiguousarray(rho_le.ravel(), dtype=np.complex128),
                    n_qubits,
                )
            except (ImportError, AttributeError):
                result.state = NoisyDensityMatrixState(rho_le, n_qubits, device="cpu")
        return result

    return wrapper


RustDevice.execute = _execute_with_noisy_state(RustDevice.execute)
