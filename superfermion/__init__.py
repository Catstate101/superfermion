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
    classical_shadow, shadow_expval, mps_expval (MPS-native, no densifying)

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

__version__ = "0.1.28"

# ── Core imports (only numpy required) ────────────────────────────────
from superfermion.circuit import Circuit
from superfermion.compiler.manager import compile
from superfermion.parameters import param
from superfermion.prepared import PreparedCircuit, prepare
from superfermion.results import RunResult
from superfermion.runner import run, simulate

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
    """\u2202\u27e8O\u27e9/\u2202\u03b8 for every symbol of ``dag``, by adjoint differentiation.

    Reverse mode: the state is already at the end of the circuit, so the bra
    is walked backwards through the gates once and every parameter's
    derivative falls out of that single sweep.  Cost is one forward plus one
    backward pass for the *whole* gradient vector, independent of how many
    parameters there are.  PennyLane analogue: ``qml.grad`` with
    ``diff_method="adjoint"``.

    Args:
        observable: the engine's own term list
            ``[(per-qubit Pauli codes, re, im)]`` with one code per wire
            (0=I, 1=X, 2=Y, 3=Z).  This is the internal form that every
            Python entry point coerces other spellings to; the label must be
            as wide as the register (``self.n_qubits``) and the summed
            operator must be Hermitian.
        dag: the parameterised IR (a ``QuantumDAG``, or a ``Circuit``, which
            ``RunResult.grad`` converts with ``to_ir()``).  Its symbols are
            the keys of the result, in ``dag.parameter_names()`` order.
        param_values: ``{name: value}`` for the symbols of ``dag``.

    Returns:
        ``{parameter_name: float}``.  A ``dag`` with no symbolic parameters
        -- what a bound circuit reports, because ``.bind()`` erases the
        symbols -- returns ``{}`` and emits a ``UserWarning`` rather than
        looking like a gradient that happens to be zero.

    Raises:
        ValueError: the observable is malformed.  A label whose width is not
            ``self.n_qubits`` used to be padded with ``I`` and came back as
            a plausible zero (E/V-7), and a non-Hermitian operator used to be
            differentiated three different ways by three engines (E/V-5);
            both are refused here on the same rule the four standalone
            engines use (``adjoint_grad``, ``parameter_shift_grad``,
            ``finite_diff_grad``, ``RunResult.grad``).

    Example:
        >>> import superfermion as sf
        >>> c = sf.Circuit(2)
        >>> t = sf.param("t")
        >>> c.h(0); c.ry(t, 1); c.cx(0, 1)
        >>> st = sf.run(c, params={"t": 0.3}).state
        >>> st.grad([([3, 3], 1.0, 0.0)], c.to_ir(), {"t": 0.3})
        {'t': -0.2955202066613395}
    """
    from superfermion.qml.gradient._observables import normalize_for_gradient

    terms = normalize_for_gradient(
        observable, getattr(self, "n_qubits", 0), what="State.grad() observable"
    )
    _warn_if_bound_dag(dag, param_values, what="State.grad()")
    return _rust_state_grad(self, terms, dag, param_values)


def _state_qfim_with_check(self, dag, param_values):
    """Quantum Fisher information matrix of this state over ``dag``'s symbols.

    ``F_ij = 4 Re[ \u27e8\u2202_i\u03c8|\u2202_j\u03c8\u27e9 - \u27e8\u2202_i\u03c8|\u03c8\u27e9\u27e8\u03c8|\u2202_j\u03c8\u27e9 ]``
    -- the Bures / Fubini-Study metric, i.e. the classical Fisher information
    of the pure state itself.  It is real, symmetric and positive
    semi-definite, and it is the metric tensor that a natural-gradient step
    (``qng_step``) inverts.  Verified against Qiskit's
    ``QFI(ReverseQGT)``/``QFI(LinCombQGT)`` to 5e-11 in the xq6 battery.

    Args:
        dag: the parameterised IR (a ``QuantumDAG``).  The matrix is laid out
            in ``dag.parameter_names()`` order -- which is the gate insertion
            order, so a caller holding its own ``param_names`` must not
            assume the two agree.
        param_values: ``{name: value}`` for the symbols of ``dag``.

    Returns:
        ``(k, k)`` float array with ``k = len(dag.parameter_names())``.  To
        read it in a different parameter order, permute **rows and columns**:
        ``F[np.ix_(perm, perm)]``.  A metric tensor is a matrix of
        coordinates, so permuting only the gradient mixes two bases and moves
        the step by 0.024420 on the F/V-10 circuit.

    Returns (degenerate):
        A rank-deficient ``F`` (for example two rotations of the same qubit
        that enter the state symmetrically) has no inverse.  ``qfim`` still
        reports it honestly; consumers fall back to ``np.linalg.pinv``
        instead of solving a singular system (F/V-11).

    Returns (no parameters):
        An empty array plus a ``UserWarning`` when ``dag`` carries no
        symbolic parameters, the same contract as ``State.grad``.
    """
    _warn_if_bound_dag(dag, param_values, what="State.qfim()")
    return _rust_state_qfim(self, dag, param_values)


_RustState.grad = _state_grad_with_check
_RustState.qfim = _state_qfim_with_check
State = _RustState
from superfermion._sf_core import DiagonalObservable


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
# Additive 0.1.15 compatibility shims (MPS output order; >3-qubit
# statevector fallback proxy).  Importing the module installs the MPS
# wrapper; no compiled-core symbol is replaced.
import superfermion._fixes as _fixes  # noqa: F401,E402

# cat backend: compile-once execution handle (statevector wired first).
from superfermion.cat import cat
from superfermion.devices import (
    Algorithm,
    AlgorithmResult,
    DeviceCapabilities,
    DeviceExecutor,
    Job,
    Provider,
)
from superfermion.experiment.context import experiment
from superfermion.experiment.local_tracker import LocalTracker
from superfermion.experiment.protocols import TrackerProtocol

# SUP-22: classical shadow protocol (randomized Pauli-basis snapshots) and
# shadow expectation values. Pure-Python/numpy; snapshots sampled from the
# exact statevector. See superfermion.mitigation.classical_shadow.
from superfermion.mitigation import ClassicalShadow, classical_shadow, shadow_expval
from superfermion.noise import ErrorChannel, NoiseModel
from superfermion.observables.core import (
    Hamiltonian,
    PauliString,
    SparsePauliOp,
    expval,
    mps_expval,
    mps_todense,
    variance,
)
from superfermion.utils.exceptions import MethodError

__all__ = [
    # Core
    "Circuit", "run", "simulate", "compile", "param", "cat",
    "RunResult", "State", "MethodError", "PreparedCircuit", "prepare", "DiagonalObservable",
    # Device/Provider protocols
    "DeviceExecutor", "DeviceCapabilities",
    "Provider", "Job", "Algorithm", "AlgorithmResult",
    # Noise
    "NoiseModel", "ErrorChannel",
    # Experiment tracking
    "TrackerProtocol", "experiment", "LocalTracker",
    # Observables
    "PauliString", "SparsePauliOp", "Hamiltonian", "expval", "variance",
    "mps_expval", "mps_todense",
    # Classical shadows (SUP-22)
    "ClassicalShadow", "classical_shadow", "shadow_expval",
    # Version
    "__version__",
]


# Announce a newer release on PyPI, if there is one.  Best-effort by design:
# cached for a day, runs in a daemon thread so import never waits on the
# network, and silent in CI/pytest or when SF_NO_UPDATE_CHECK is set.
try:
    from superfermion.update_check import notify_if_outdated as _notify_if_outdated

    _notify_if_outdated(__version__)
except Exception:  # a notice must never be able to break an import
    pass
