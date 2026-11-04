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


class LazyDict(dict):
    """A ``dict`` whose entries are built on first access.

    Large result payloads (e.g. the ``2^n``-entry exact probability table
    of a statevector run) are frequently never read — consumers usually
    only use ``counts``. ``LazyDict`` defers the construction cost until
    the first read operation, at which point it materializes in-place
    (storage grows via ``dict.update``), so callers can never observe a
    partially built table.

    All read, mutation, comparison, repr, copy/pickle operations trigger
    materialization. Only C-level dict fast paths that bypass Python
    dispatch (e.g. ``dict(obj)``'s storage merge, ``PyDict_Next``) are
    not intercepted — construct from ``obj.items()`` or iterate instead.

    Args:
        build: Zero-argument callable returning a plain ``dict``.
    """

    __slots__ = ("_build",)

    def __init__(self, build) -> None:
        super().__init__()
        self._build = build

    def _materialize(self) -> None:
        build = self._build
        if build is not None:
            self._build = None
            super().update(build())

    # ── read operations ──────────────────────────────────────────
    def __len__(self) -> int:
        self._materialize()
        return super().__len__()

    def __iter__(self):
        self._materialize()
        return super().__iter__()

    def __contains__(self, key) -> bool:
        self._materialize()
        return super().__contains__(key)

    def __getitem__(self, key):
        self._materialize()
        return super().__getitem__(key)

    def __bool__(self) -> bool:
        self._materialize()
        return super().__len__() > 0

    def __repr__(self) -> str:
        self._materialize()
        return super().__repr__()

    def __eq__(self, other):
        self._materialize()
        return super().__eq__(other)

    def __ne__(self, other):
        self._materialize()
        return super().__ne__(other)

    def __or__(self, other):
        self._materialize()
        return super().__or__(other)

    def __ror__(self, other):
        self._materialize()
        return super().__ror__(other)

    def get(self, key, default=None):
        self._materialize()
        return super().get(key, default)

    def keys(self):
        self._materialize()
        return super().keys()

    def values(self):
        self._materialize()
        return super().values()

    def items(self):
        self._materialize()
        return super().items()

    def copy(self):
        self._materialize()
        return super().copy()

    def __copy__(self):
        self._materialize()
        return dict(self)

    def __reduce__(self):
        # Pickle/deepcopy as a plain dict (avoids reviving the thunk).
        self._materialize()
        return (dict, (dict(self),))

    # ── mutation operations ──────────────────────────────────────
    def __setitem__(self, key, value) -> None:
        self._materialize()
        super().__setitem__(key, value)

    def __delitem__(self, key) -> None:
        self._materialize()
        super().__delitem__(key)

    def update(self, *args, **kwargs) -> None:
        self._materialize()
        super().update(*args, **kwargs)

    def setdefault(self, key, default=None):
        self._materialize()
        return super().setdefault(key, default)

    def pop(self, key, *args):
        self._materialize()
        return super().pop(key, *args)

    def popitem(self):
        self._materialize()
        return super().popitem()

    def clear(self) -> None:
        self._materialize()
        super().clear()

    def __ior__(self, other):
        self._materialize()
        return super().__ior__(other)


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
            observable: Pauli observable — str (one character per qubit,
                character 0 = qubit 0), dict, SparsePauliOp, PauliString,
                Hamiltonian, Sequence[str], Sequence[PauliString], or Rust term
                list
                [(paulis, coef_re, coef_im)].

        Returns:
            Expectation value as float.
        """
        observable = _coerce_observable(observable, _result_n_qubits(self))
        if self.state is not None:
            return self.state.expectation(observable)
        if self.counts:
            return _estimate_expval_from_counts(
                observable, self.counts, _outcome_char_index(self)
            )
        return 0.0

    def variance(self, observable) -> float:
        """Compute variance Var(O) = ⟨O²⟩ − ⟨O⟩² of an observable.

        Uses exact computation from state (if available) or estimates
        from counts (same parity convention as ``expectation``).
        PennyLane analogue: ``qml.var``.

        Args:
            observable: Pauli observable — str (one character per qubit,
                character 0 = qubit 0), dict, SparsePauliOp, PauliString,
                Hamiltonian, Sequence[str], Sequence[PauliString], or Rust term
                list
                [(paulis, coef_re, coef_im)].

        Returns:
            Variance as float.
        """
        observable = _coerce_observable(observable, _result_n_qubits(self))
        if self.state is not None:
            return self.state.variance(observable)
        if self.counts:
            return _estimate_var_from_counts(
                observable, self.counts, _outcome_char_index(self)
            )
        return 0.0

    def mutual_info(self, wires_a, wires_b) -> float:
        """Mutual information I(A;B) between two wire sets.

        Exact, computed from partial traces of the state (natural-log
        units). PennyLane analogue: ``qml.mutual_info``.

        Args:
            wires_a: First wire list.
            wires_b: Second (disjoint) wire list.

        Returns:
            Mutual information in nats.
        """
        if self.state is None:
            raise RuntimeError("mutual_info() not available: no state (QPU result?)")
        return self.state.mutual_info(wires_a, wires_b)

    def grad(self, observable, dag=None, param_values=None) -> dict:
        """Compute gradient of an observable.

        Uses adjoint differentiation from state (if available).

        Args:
            observable: Pauli observable — str (one character per qubit,
                character 0 = qubit 0), dict, SparsePauliOp, PauliString,
                Hamiltonian, Sequence[str], Sequence[PauliString], or Rust term
                list
                [(paulis, coef_re, coef_im)].
            dag: Parameterised IR (a ``QuantumDAG``, or a ``Circuit`` which is
                converted with ``to_ir()``).  Optional: a result returned by
                ``sf.run(circuit, params=...)`` remembers the symbolic circuit
                and the values it bound, so the dag is rebuilt for you (RES-2).
            param_values: Parameter values — a ``{name: value}`` dict, or a
                sequence aligned with the circuit's parameter names.
                Optional for the same reason as ``dag``.

        Returns:
            Dict mapping parameter name to gradient value.

        Raises:
            ValueError: the result has no dag to differentiate, the dag carries
                no symbolic parameters (a bound circuit erases them), the
                values were not recorded, or the observable is not a valid
                observable for a gradient (``grad`` refuses a label whose
                width is not the circuit's qubit count, and a non-Hermitian
                operator, exactly like the four standalone engines).
        """
        from superfermion.qml.gradient._observables import normalize_for_gradient
        from superfermion.qml.gradient._params import dag_parameter_names

        observable = normalize_for_gradient(
            observable, _result_n_qubits(self) or 0, what="grad() observable"
        )
        if self.state is None:
            raise RuntimeError("grad() not available: no state (QPU result?)")
        if dag is not None and not hasattr(dag, "adjoint_grad") and hasattr(
            dag, "to_ir"
        ):
            dag = dag.to_ir()
        if dag is None or param_values is None:
            fallback_dag, fallback_values = _grad_fallbacks(self)
            if dag is None:
                dag = fallback_dag
            if param_values is None:
                param_values = fallback_values
        if param_values is not None and not isinstance(param_values, dict):
            param_values = _as_param_dict(param_values, self)
        if dag is None:
            raise ValueError(
                "grad() needs a parameterised dag for this result and none "
                "could be rebuilt (the circuit carries no free parameters).\n"
                "  Fix: pass dag=circuit.to_ir() for the circuit that produced "
                "this state"
            )
        if param_values is None:
            raise ValueError(
                "grad() needs the values of the circuit parameters; the run "
                "did not record them.\n"
                "  Fix: run with sf.run(circuit, params={'theta': 0.3, ...}) "
                "or pass param_values={'theta': 0.3, ...}"
            )
        if not dag_parameter_names(dag):
            # A bound circuit has no symbols left, so the Rust adjoint engine
            # can only answer {}.  Returning that silently made a bound-by-
            # accident training loop read a zero gradient and stop making
            # progress (F/V-6).
            raise ValueError(
                "grad() has nothing to differentiate: the dag for this result "
                "carries no symbolic parameters.\n"
                "  Cause: the circuit was already bound — circuit.bind({...}) "
                "replaces every sf.param(...) symbol with a number, so "
                ".to_ir() reports no parameters.\n"
                "  Fix: run the unbound circuit and let this result bind the "
                "values: sf.run(circuit, params={'theta': 0.3, ...}).grad('ZZ')"
            )
        return self.state.grad(observable, dag, param_values)

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


def _coerce_observable(observable, n_qubits=None):
    """Accept user-friendly observable forms; convert to Rust term list.

    Accepted (verified against parameter-shift ground truth to ~1e-16):
      - str, e.g. "ZIII": one character per qubit, **character 0 = qubit 0**
        (q0-leftmost, matching the Pauli-index encoding the engines consume
        and ``RunResult.expectation``'s own counts parity).
      - dict, e.g. {"ZI": 0.5, "IX": 0.3}.
      - SparsePauliOp / Hamiltonian-like objects.
      - PauliString: a single weighted Pauli string (CH-7).
      - Sequence[PauliString]: the same sum written as a list of weighted
        terms; each element is converted on its own (CH-7).
      - Sequence[str]: several Pauli strings, summed with unit weight.
      - Rust term list [(paulis, coef_re, coef_im)]: passed through unchanged,
        and so are the ``(label, coef)`` / ``(label, coef_re, coef_im)``
        spellings a user might reasonably write (RES-2).

    Anything unrecognized is passed through unchanged (legacy behavior).
    """
    if isinstance(observable, (list, tuple)):
        terms = _term_sequence(observable)
        return observable if terms is None else terms
    try:
        from superfermion.qml.gradient.adjoint import _observable_to_rust_terms
        return _observable_to_rust_terms(observable, n_qubits or 0)
    except (TypeError, ValueError, ImportError, AttributeError):
        return observable


def _term_sequence(observable):
    """Rust terms for a user-written sequence, or ``None`` when unrecognised.

    Only shapes whose meaning is unambiguous are converted.  The engine's own
    form carries ``paulis`` as integer codes, so anything whose first element
    is already a list/tuple of ints is left exactly as it is; a string label
    (or a label plus coefficient) is plainly a user spelling and becomes
    ``(paulis, coef_re, coef_im)``.
    """
    try:
        items = list(observable)
    except TypeError:
        return None
    if not items:
        return None
    try:
        from superfermion.qml.gradient.adjoint import _pauli_indices
    except ImportError:  # pragma: no cover - adjoint module always ships
        return None
    if all(isinstance(item, str) for item in items):
        return [(_pauli_indices(item), 1.0, 0.0) for item in items]
    converted = []
    for item in items:
        if not isinstance(item, (tuple, list)) or len(item) not in (2, 3):
            return None
        label = item[0]
        if isinstance(label, str):
            paulis = _pauli_indices(label)
        else:
            return None  # already the engine form (or unknown): do not touch
        if len(item) == 3:
            try:
                coef_re = float(item[1])
                coef_im = float(item[2])
            except (TypeError, ValueError):
                return None
        else:
            coefficient = item[1]
            if isinstance(coefficient, dict):
                try:
                    coef_re = float(
                        coefficient.get("real", coefficient.get("re", 0.0))
                    )
                    coef_im = float(
                        coefficient.get("imag", coefficient.get("im", 0.0))
                    )
                except (TypeError, ValueError):
                    return None
            else:
                try:
                    value = complex(coefficient)
                except (TypeError, ValueError):
                    return None
                coef_re, coef_im = value.real, value.imag
        converted.append((paulis, coef_re, coef_im))
    return converted


def _outcome_char_index(result) -> "Optional[Dict[int, int]]":
    """Character position of each qubit in ``result.counts``, if not default.

    ``None`` means the historical qubit-keyed layout (character ``n-1-q``).
    Results whose counts are keyed by classical bit carry the map in
    ``metadata['qubit_char_index']``.
    """
    metadata = getattr(result, "metadata", None)
    if not isinstance(metadata, dict):
        return None
    index = metadata.get("qubit_char_index")
    return index if isinstance(index, dict) else None


def _grad_fallbacks(result):
    """Best-effort ``(dag, param_values)`` for :meth:`RunResult.grad`.

    ``sf.run`` stashes the *symbolic* circuit it bound and the values it used,
    which is the only place those names survive: ``bind()`` returns a circuit
    with the numbers substituted in.  Without that context the stored circuit
    is used, which differentiates only if it still carries free parameters.
    """
    context = getattr(result, "_sf_grad_context", None)
    if not isinstance(context, dict):
        context = {}
    candidate = context.get("circuit")
    if candidate is None:
        candidate = getattr(result, "circuit", None)
    dag = None
    if candidate is not None:
        try:
            dag = candidate.to_ir()
        except Exception:
            dag = None
    params = context.get("params")
    if params is not None:
        return dag, dict(params)
    free = tuple(getattr(candidate, "parameters", ()) or ())
    if free:
        return dag, None  # names known, values lost -> caller must supply
    return dag, {}


def _as_param_dict(values, result) -> dict:
    """Align a sequence of parameter values with the circuit's names."""
    names: List[str] = []
    context = getattr(result, "_sf_grad_context", None)
    if isinstance(context, dict) and isinstance(context.get("params"), dict):
        names = list(context["params"])
    if not names:
        circuit = getattr(result, "circuit", None)
        names = list(getattr(circuit, "parameters", ()) or ())
    values = list(values)
    if not names:
        raise ValueError(
            "grad() cannot align a sequence of parameter values: this result "
            "records no parameter names.\n"
            "  Fix: pass param_values={'theta': 0.3, ...}"
        )
    if len(names) != len(values):
        raise ValueError(
            f"grad() got {len(values)} parameter value(s) for {len(names)} "
            f"parameter(s) {names}.\n"
            "  Fix: pass a dict, or one value per parameter"
        )
    return {name: float(value) for name, value in zip(names, values)}


def _result_n_qubits(result) -> "Optional[int]":
    """Best-effort qubit count for observable coercion (None if unknown)."""
    try:
        return result.circuit.n_qubits
    except Exception:
        return None


def _estimate_expval_from_counts(
    observable: list,
    counts: Dict[str, int],
    qubit_char_index: "Optional[Dict[int, int]]" = None,
) -> float:
    """Estimate expectation value from measurement counts.

    For each Pauli term, compute the parity of the measured bits
    for non-identity Pauli operators and accumulate the weighted sum.

    ``qubit_char_index`` maps qubit ``q`` onto its character position; ``None``
    is the default qubit-keyed layout, where character ``i`` is qubit
    ``n-1-i``.  A result whose counts are keyed by classical bit (RES-1) holds
    a non-default map in ``metadata['qubit_char_index']``.
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
                    index = _char_position(q, n, qubit_char_index)
                    if 0 <= index < n:
                        parity ^= int(bitstring[index])
            eigenvalue = 1.0 - 2.0 * parity
            term_val += eigenvalue * count
        result += coef_re * term_val / total_shots
    return result


def _char_position(
    qubit: int, n: int, qubit_char_index: "Optional[Dict[int, int]]"
) -> int:
    """Character position of ``qubit`` in an outcome string of width ``n``."""
    if qubit_char_index is not None:
        position = qubit_char_index.get(qubit)
        if position is not None:
            return int(position)
    return n - 1 - qubit


def _estimate_var_from_counts(
    observable: list,
    counts: Dict[str, int],
    qubit_char_index: "Optional[Dict[int, int]]" = None,
) -> float:
    """Estimate Var(O) = E[o²] − E[o]² from measurement counts.

    Uses the same per-bitstring eigenvalue convention as
    ``_estimate_expval_from_counts``: the eigenvalue of the observable
    on a measured bitstring is the parity-weighted coefficient sum.
    """
    total_shots = sum(counts.values())
    if total_shots == 0:
        return 0.0

    mean = 0.0
    mean_sq = 0.0
    for bitstring, count in counts.items():
        n = len(bitstring)
        eigen = 0.0
        for paulis, coef_re, _coef_im in observable:
            parity = 0
            for q, p in enumerate(paulis):
                if p in (1, 2, 3):  # X, Y, Z all flip parity
                    index = _char_position(q, n, qubit_char_index)
                    if 0 <= index < n:
                        parity ^= int(bitstring[index])
            eigen += coef_re * (1.0 - 2.0 * parity)
        mean += eigen * count
        mean_sq += eigen * eigen * count
    mean /= total_shots
    mean_sq /= total_shots
    return mean_sq - mean * mean
