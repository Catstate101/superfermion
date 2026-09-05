"""
Superfermion — One framework. Every qubit. Every gradient. Every model.

A hardware-agnostic quantum computation framework that makes quantum circuits
differentiable, compilable to any hardware, and trainable end-to-end.

Quick Start::

    import superfermion as sf

    circuit = sf.Circuit(2).h(0).cnot(0, 1)
    result = sf.run(circuit, device="cpu", shots=1000)
    print(result.counts)   # {'00': 503, '11': 497}

Core surface (``sf.*``):
    Circuit, run, compile, param, RunResult, DeviceExecutor, experiment,
    PauliString, SparsePauliOp, Hamiltonian, expval, ClassicalShadow,
    classical_shadow, shadow_expval

Application modules (importable but not promoted to ``sf.*``):
    superfermion.algorithms   — VQE, QAOA
    superfermion.chemistry    — molecular Hamiltonians, UCCSD, PySCF
    superfermion.qml          — gradient methods, templates, measurements
    superfermion.nn           — quantum neural network layers
    superfermion.qec          — quantum error correction
    superfermion.mitigation   — error mitigation
    superfermion.bridge       — Qiskit / PennyLane interop
    superfermion.noise        — noise models
    superfermion.pulse        — pulse-level control
    superfermion.viz          — circuit visualization
"""

__version__ = "0.1.9"

# ── Core imports (only numpy required) ────────────────────────────────
from superfermion.circuit import Circuit
from superfermion.runner import run, simulate
from superfermion.compiler.manager import compile
from superfermion.parameters import param
from superfermion.results import RunResult
try:
    from superfermion._sf_core import State as _RustState
except ImportError as e:
    raise ImportError(
        "The Rust extension module `_sf_core` is not built.\n"
        "Build it with:\n"
        "    pip install maturin\n"
        "    cd crates/sf-bindings && maturin develop --release\n"
        "Or install the pre-built wheel: pip install superfermion\n"
    ) from e

# SUP-6: State.grad()/qfim() silently returned {} / [] for a bound or
# parameterless DAG (no symbolic parameters). Wrap them to emit a
# UserWarning instead, so callers learn to keep the dag symbolic.
from superfermion.utils.grad_checks import warn_if_bound_dag as _warn_if_bound_dag

_rust_state_grad = _RustState.grad
_rust_state_qfim = _RustState.qfim


def _state_grad_with_check(self, observable, dag, param_values):
    _warn_if_bound_dag(dag, param_values, what="State.grad()")
    return _rust_state_grad(self, observable, dag, param_values)


def _state_qfim_with_check(self, dag, param_values):
    _warn_if_bound_dag(dag, param_values, what="State.qfim()")
    return _rust_state_qfim(self, dag, param_values)


_RustState.grad = _state_grad_with_check
_RustState.qfim = _state_qfim_with_check
State = _RustState

# SUP-22: expose classical-shadow utilities on State (pure states only; the
# snapshots are sampled from the exact statevector returned by numpy()).
def _state_classical_shadow(self, shots, seed=None):
    from superfermion.mitigation.classical_shadow import ClassicalShadow
    return ClassicalShadow.from_statevector(self.numpy(), shots, seed=seed)


def _state_shadow_expval(self, observable, shots=1000, seed=None, k=1):
    from superfermion.mitigation.classical_shadow import ClassicalShadow
    return ClassicalShadow.from_statevector(
        self.numpy(), shots, seed=seed
    ).expval(observable, k=k)


_RustState.classical_shadow = _state_classical_shadow
_RustState.shadow_expval = _state_shadow_expval
from superfermion.utils.exceptions import MethodError

from superfermion.devices import (
    DeviceExecutor, DeviceCapabilities,
    Provider, Job, Algorithm, AlgorithmResult,
)
from superfermion.experiment.protocols import TrackerProtocol
from superfermion.experiment.context import experiment
from superfermion.experiment.local_tracker import LocalTracker

from superfermion.observables.core import (
    PauliString,
    SparsePauliOp,
    Hamiltonian,
    expval,
    variance,
)
from superfermion.noise import NoiseModel

# SUP-22: classical shadow protocol (randomized Pauli-basis snapshots) and
# shadow expectation values. Pure-Python/numpy; snapshots sampled from the
# exact statevector. See superfermion.mitigation.classical_shadow.
from superfermion.mitigation import ClassicalShadow, classical_shadow, shadow_expval


__all__ = [
    # Core
    "Circuit", "run", "simulate", "compile", "param",
    "RunResult", "State", "MethodError",
    # Device/Provider protocols
    "DeviceExecutor", "DeviceCapabilities",
    "Provider", "Job", "Algorithm", "AlgorithmResult",
    # Noise
    "NoiseModel",
    # Experiment tracking
    "TrackerProtocol", "experiment", "LocalTracker",
    # Observables
    "PauliString", "SparsePauliOp", "Hamiltonian", "expval", "variance",
    # Classical shadows (SUP-22)
    "ClassicalShadow", "classical_shadow", "shadow_expval",
    # Version
    "__version__",
]
