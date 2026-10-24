"""The ``cat`` backend — a compile-once execution handle.

``sf.cat(circuit, mode="statevector")`` lowers a circuit into a reusable
:class:`CatCircuit` handle *once*: topological ordering, constant gate
matrices and fusion metadata are cached, so repeated parameter rebinds skip
the DAG rebuild and gate re-parse that the per-call ``sf.run`` path pays on
every invocation.

Availability is optional: wheels that predate the ``cat`` crate do not export
``CatCircuit`` from ``_sf_core``, and this module raises a clear ImportError
only when ``cat`` is actually requested.
"""

from __future__ import annotations

from typing import Any

try:
    from superfermion._sf_core import CatCircuit as _CatCircuit

    _HAS_CAT = True
except ImportError:  # pragma: no cover - depends on built extension
    _CatCircuit = None
    _HAS_CAT = False

CatCircuit = _CatCircuit


def _to_dag(obj: Any) -> Any:
    """Accept a Circuit (via ``to_ir``) or an already-built QuantumDAG."""
    if hasattr(obj, "simulate_to_state"):  # raw Rust QuantumDAG
        return obj
    to_ir = getattr(obj, "to_ir", None)
    if callable(to_ir):
        return to_ir()
    raise TypeError("sf.cat expects a Circuit or a QuantumDAG")


def cat(circuit_or_dag: Any, mode: str = "statevector") -> Any:
    """Compile ``circuit_or_dag`` once into a reusable ``cat`` handle.

    Parameters
    ----------
    circuit_or_dag
        A :class:`superfermion.Circuit` (converted via ``to_ir``) or an
        already-built Rust ``QuantumDAG``.
    mode
        One of ``statevector`` (default), ``density_matrix``, ``adjoint``,
        ``stabilizer``, ``clifford``. ``statevector`` executes via
        :meth:`CatCircuit.simulate`; ``adjoint`` via
        :meth:`CatCircuit.run_gradient` (Pauli-string observable, coefficient
        1.0); ``stabilizer``/``clifford`` via :meth:`CatCircuit.sample`
        (little-endian counts). ``density_matrix`` is not implemented and
        raises ``ValueError`` rather than returning a wrong result.
    """
    if not _HAS_CAT:
        raise ImportError(
            "The 'cat' backend is not present in this build of _sf_core. "
            "Rebuild the extension (maturin) to enable it."
        )
    # A >3-qubit opaque unitary makes to_ir() return the D1 compatibility
    # proxy, which the PyO3 constructor cannot accept; refuse it by name.
    from superfermion._fixes import reject_compat_dag

    dag = _to_dag(circuit_or_dag)
    reject_compat_dag(dag, "sf.cat")
    return _CatCircuit(dag, mode)


__all__ = ["cat", "CatCircuit"]
