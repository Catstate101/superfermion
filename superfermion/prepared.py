"""Prepared, immutable CPU statevector execution with explicit f64 arithmetic."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from superfermion.circuit import Circuit


class PreparedCircuit:
    """Snapshot a unitary circuit once for repeated parameterized execution.

    This opt-in CPU API always uses complex128. It caches gate ordering and
    constant matrices, while rebuilding angle-dependent matrices and fusion
    plans for each parameter set. It does not mutate the original circuit.
    Measurement, reset and conditioned circuits are rejected.

    Batch execution is sequential across circuits and releases the Python GIL.
    Within-circuit parallelism follows RAYON_NUM_THREADS. State banks are
    limited to 1 GiB; split larger requests into batches. Returned State.numpy()
    arrays still own a copy; use states_batch() to avoid that extra export copy.
    """

    def __init__(self, circuit: Circuit) -> None:
        from superfermion._sf_core import PreparedCircuit as NativePreparedCircuit

        from superfermion._fixes import reject_compat_dag

        dag = circuit.to_ir()
        reject_compat_dag(dag, "sf.prepare")
        self._native = NativePreparedCircuit(dag)

    @property
    def n_qubits(self) -> int:
        return self._native.n_qubits

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(self._native.parameter_names)

    def simulate(self, values: Mapping[str, float] | None = None):
        """Return a Rust State without sampling or exporting its statevector."""
        return self._native.simulate(dict(values or {}))

    def states_batch(self, values: Iterable[Mapping[str, float]]):
        """Return complex128 states as rows, in the existing core's LSB order."""
        return self._native.states_batch([dict(row) for row in values])

    def mean_z_batch(self, values: Iterable[Mapping[str, float]]):
        """Return the mean of single-wire Z expectations for each circuit."""
        return self._native.mean_z_batch([dict(row) for row in values])

    def diagonal_batch(self, values: Iterable[Mapping[str, float]], observable):
        """Return diagonal expectations without exporting states or probabilities.

        Construct sf.DiagonalObservable(weights.tolist()) once. Its weights
        follow the statevector index order, where qubit q occupies bit q.
        """
        return self._native.diagonal_batch([dict(row) for row in values], observable)


def prepare(circuit: Circuit) -> PreparedCircuit:
    """Prepare an immutable CPU/f64 snapshot; see PreparedCircuit."""
    return PreparedCircuit(circuit)
