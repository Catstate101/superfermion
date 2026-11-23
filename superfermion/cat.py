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


def cat(circuit_or_dag: Any, mode: str = "statevector", bond_dim: int = 64) -> Any:
    """Compile ``circuit_or_dag`` once into a reusable ``cat`` handle.

    Parameters
    ----------
    circuit_or_dag
        A :class:`superfermion.Circuit` (converted via ``to_ir``) or an
        already-built Rust ``QuantumDAG``.
    mode
        One of ``statevector`` (default), ``density_matrix``, ``adjoint``,
        ``stabilizer``, ``clifford``, ``mps``. ``statevector`` and
        ``density_matrix`` both execute via :meth:`CatCircuit.simulate`: the
        first returns a state vector, the second a ``rho`` of shape
        ``(2**n, 2**n)``, which is the only lane that can carry a ``Reset``
        after use or a mid-circuit readout without a later conditioned gate. A
        ``rho`` cannot represent a mixture over measurement branches, so
        :meth:`CatCircuit.sample` on this mode raises ``ValueError``.
        ``adjoint`` executes via :meth:`CatCircuit.run_gradient` (Pauli-string
        observable, coefficient 1.0) or :meth:`CatCircuit.run_gradient_terms`
        (any number of terms for one forward and one reverse sweep);
        ``stabilizer``/``clifford`` via :meth:`CatCircuit.sample`
        (little-endian counts); ``mps`` via :meth:`CatCircuit.simulate`,
        which returns a ``State`` holding the tensor network itself rather
        than a flat ``2**n`` vector, and -- like ``adjoint`` -- via
        :meth:`CatCircuit.run_gradient` / :meth:`CatCircuit.run_gradient_terms`,
        which answer with the shift rule on that chain instead of a reverse
        sweep.
    bond_dim
        Schmidt-rank cap for ``mode="mps"`` (default 64). Large enough is
        exact -- agreement with ``mode="statevector"`` down to rounding --
        and small enough truncates, which the state's
        ``truncation_report()`` reports instead of hiding. The returned state
        keeps the *surviving* weight rather than renormalizing, so its squared
        norm is ``1 - discarded_weight`` and the report says exactly how much
        was dropped. The pure lanes ignore ``bond_dim``.

    Notes
    -----
    ``mode="mps"`` compiles the *symbolic* circuit: gate order, wire routing
    and parameter names are fixed once, and each :meth:`CatCircuit.simulate`
    re-binds in place and re-evolves. It takes the statevector lane's guards
    (a trailing readout is dropped, a leading reset is ``|0> -> |0>``, a
    reset after use and every feed-forward construct are refused), because a
    single tensor network is still one pure state. A ``simulate`` call that
    omits a name from ``parameter_names()`` raises ``ValueError`` naming the
    missing parameters rather than evaluating a half-bound circuit. Sampling
    goes through that state --
    ``cat(qc, mode="mps").simulate(values).sample(shots)`` -- so
    :meth:`CatCircuit.sample` on an ``mps`` handle raises with that route
    instead of guessing parameter values.

    The ``adjoint`` lane differentiates ``RY``/``RZ``/``RX``, ``P``/``R1``,
    ``U``/``U3``, the controlled rotations ``CRX``/``CRZ``/``CRY``/``CP``/
    ``CU`` (and ``CH``, which is a ``CU``), and the Pauli entanglers ``RXX``/
    ``RYY``/``RZZ``. It applies ``H``/``X``/``Y``/``Z``/``S``/``T``/``SX`` (and
    their daggers), ``CZ``/``CY``/``SWAP`` and ``CNOT`` as constants. ``CRY``
    is carried as ``CU(theta, 0, 0)`` because the IR has no separate opcode.
    An angle that is already bound is baked into the plan rather than refused.
    ``ECR`` and opaque ``Unitary`` gates remain outside the lane.  The reverse
    sweep either replays a stored post-state per parameter gate (``O(P*2**n)``
    memory) or re-walks the state (``O(2**n)``).  The cache is kept while it
    stays L3-resident and, past that, for gate-dense circuits that carry few
    parameters (e.g. long CZ ladders); when it would spill *and* most gates
    carry parameters the plan streams, which measured 1.14-1.25x faster on the
    all-parameter two-qubit lanes at n=16.

    The ``mps`` lane's gradient is the shift rule on the chain that
    :meth:`CatCircuit.simulate` returns, not a reverse sweep: differentiating
    through the SVD that truncates would need the discarded subspace to be
    fixed with respect to the parameters, and it is not. So the two lanes
    answer the same question with different cost profiles -- the adjoint lane
    pays one forward plus one reverse pass and is exact on the dense state,
    while the chain pays ``2*P`` evolutions at an untruncated bond and ``4*P``
    once the cap discards weight (a two-term rule is only the derivative of
    the reported functional while nothing is dropped) and buys memory instead
    of passes. At a bond large enough to be exact the two agree to ~1e-12.
    """
    if not _HAS_CAT:
        raise ImportError(
            "The 'cat' backend is not present in this build of _sf_core. "
            "Rebuild the extension (maturin) to enable it."
        )
    # A >3-qubit opaque unitary makes to_ir() return the D1 compatibility
    # proxy, which the PyO3 constructor cannot accept; refuse it by name.
    from superfermion._fixes import reject_compat_dag

    bond_dim = int(bond_dim)
    if bond_dim < 1:
        raise ValueError(
            "bond_dim must be >= 1 (an MPS with zero bond holds no state); "
            f"got {bond_dim}"
        )
    dag = _to_dag(circuit_or_dag)
    reject_compat_dag(dag, "sf.cat")
    try:
        return _CatCircuit(dag, mode, bond_dim)
    except TypeError as exc:
        # A build from before the MPS lane takes two constructor arguments.
        # Falling back keeps every earlier mode working and names the real
        # cause only when the missing lane is the one that was asked for.
        if str(mode).lower() == "mps" and "argument" in str(exc):
            raise ImportError(
                "This _sf_core predates cat mode='mps'; rebuild the extension "
                "(maturin) to enable it."
            ) from exc
        return _CatCircuit(dag, mode)


__all__ = ["cat", "CatCircuit"]
