"""Classical-register ordering for measurement outcomes.

``circuit.measure(qubit, cbit)`` stores the sampled bit for ``qubit`` into
classical bit ``cbit``, but every backend samples *qubits* and keys the
outcome string by qubit, so an explicit ``cbit`` never reached the result::

    c = sf.Circuit(2, n_cbits=2)
    c.x(0); c.measure(0, cbit=1); c.measure(1, cbit=0)
    sf.run(c, shots=200).counts          # {'01': 200}, classical bits '10'

Character ``i`` of a superfermion outcome string is qubit ``n-1-i`` (the
statevector index is ``q0 + 2*q1 + ...``), so relabelling is a permutation of
characters.  :func:`measured_classical_map` returns that permutation only
when it is lossless -- every qubit measured exactly once, each into a
distinct classical bit, with ``n_cbits == n_qubits`` -- because only then is
the string width unchanged.  Every other circuit -- a partial measurement, a
qubit measured twice, ``n_cbits != n_qubits``, or a gate *after* the
measurement (where the sampled string is the qubits' final value rather than
what the register latched) -- is left byte-for-byte as it was, so nothing
that works today changes.

When the map does fire, the result records it: ``metadata`` gains
``{"counts_keyed_by": "classical_bits", "qubit_char_index": {q: i}}`` and
:class:`~superfermion.results.RunResult` uses ``qubit_char_index`` so
``expectation``/``variance`` estimated from counts stay correct.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

__all__ = [
    "apply_to_result",
    "char_index_for_qubit",
    "measured_classical_map",
    "rekey",
]

#: Gate names a measurement can be stored under.
MEASURE_NAMES = frozenset({"MEASURE", "M", "MEASURE_Z", "MEASURE_ALL"})

#: Gates that carry no state change, so they do not make a circuit dynamic.
_TRANSPARENT_NAMES = frozenset({"BARRIER"})


def _gate_records(circuit: Any):
    """Gate records of ``circuit`` **without** forcing a Rust conversion.

    ``Circuit._ensure_gates()`` converts a Rust-backed gate list into Python
    ``GateRecord`` objects and drops the Rust buffer; it is deferred on purpose
    so that conversion lands outside measured regions.  Calling it from every
    completed run would undo that, so the Rust records are read directly here.
    They carry no classical bits, which simply means the identity map -- the
    untouched path -- for circuits that only exist in Rust form.
    """
    rust = getattr(circuit, "_gates_rust", None)
    if bool(getattr(circuit, "_use_rust", False)) and rust is not None:
        try:
            return tuple(rust.to_gate_records())
        except Exception:
            return ()
    try:
        return tuple(getattr(circuit, "_gates", ()) or ())
    except Exception:
        return ()


def _unpack(gate: Any):
    """``(name, qubits, classical_bits)`` from a record or a Rust 3-tuple."""
    if isinstance(gate, (tuple, list)):
        name = gate[0] if len(gate) > 0 else ""
        qubits = gate[1] if len(gate) > 1 else ()
        return str(name).upper(), tuple(qubits or ()), ()
    return (
        str(getattr(gate, "name", "")).upper(),
        tuple(getattr(gate, "qubits", ()) or ()),
        tuple(getattr(gate, "classical_bits", ()) or ()),
    )


def measured_classical_map(circuit: Any) -> Optional[Dict[int, int]]:
    """``{qubit: cbit}`` when outcomes can be keyed by classical bit.

    Returns ``None`` -- meaning "leave the counts alone" -- unless the
    declared measurement targets form a permutation of ``0..n-1`` over
    every qubit, because only a permutation keeps the string width and
    therefore the meaning of every other character.
    """
    try:
        n = int(getattr(circuit, "n_qubits", 0) or 0)
    except (TypeError, ValueError):
        return None
    if n <= 0:
        return None
    try:
        n_cbits = int(getattr(circuit, "n_cbits", n) or n)
    except (TypeError, ValueError):
        return None
    if n_cbits != n:
        # The register is wider (or narrower) than the qubits, so the outcome
        # string is not the classical register and the widths would disagree.
        return None
    mapping: Dict[int, int] = {}
    measured = False
    for gate in _gate_records(circuit):
        name, qubits, classical = _unpack(gate)
        if name in MEASURE_NAMES:
            measured = True
        elif measured and name not in _TRANSPARENT_NAMES:
            # A gate after a measurement makes the circuit dynamic: the
            # sampled string then describes each qubit's *final* value, not
            # what the classical register latched mid-circuit.  Relabelling
            # by cbit would be a different claim, so leave it alone.
            return None
        else:
            continue
        if len(qubits) != 1:
            return None
        try:
            qubit = int(qubits[0])
        except (TypeError, ValueError):
            return None
        try:
            cbit = int(classical[0]) if classical else qubit
        except (TypeError, ValueError):
            return None
        if qubit in mapping or not 0 <= qubit < n or not 0 <= cbit < n:
            return None
        mapping[qubit] = cbit
    if len(mapping) != n:
        return None
    if len(set(mapping.values())) != n:
        return None
    if all(qubit == cbit for qubit, cbit in mapping.items()):
        return None  # identity: the current qubit-keyed output is already it
    return mapping


def char_index_for_qubit(
    qubit_to_cbit: Dict[int, int], n_qubits: int
) -> Dict[int, int]:
    """Character position of each qubit once keys are classical-bit ordered."""
    n = int(n_qubits)
    return {
        int(qubit): n - 1 - int(cbit)
        for qubit, cbit in qubit_to_cbit.items()
    }


def rekey(
    outcomes: Any,
    qubit_to_cbit: Dict[int, int],
    n_qubits: int,
) -> Any:
    """Relabel outcome strings from qubit order to classical-bit order.

    ``outcomes`` is a ``{bitstring: value}`` mapping (counts or
    probabilities).  Anything with an unexpected key width is returned
    unchanged rather than mangled.
    """
    if not outcomes or not qubit_to_cbit:
        return outcomes
    n = int(n_qubits)
    try:
        entries = list(outcomes.items())
    except AttributeError:
        return outcomes
    out: Dict[str, Any] = {}
    for key, value in entries:
        text = str(key)
        if len(text) < n:
            text = text.zfill(n)
        if len(text) != n:
            return outcomes
        bits = text
        relabelled = list(text)
        for qubit, cbit in qubit_to_cbit.items():
            relabelled[n - 1 - int(cbit)] = bits[n - 1 - int(qubit)]
        new_key = "".join(relabelled)
        out[new_key] = out.get(new_key, 0) + value
    return out


def apply_to_result(circuit: Any, result: Any) -> Any:
    """Rewrite ``result``'s outcome keys when the circuit declares a map.

    Returns the same object; untouched results are returned immediately, so
    a circuit that always measured ``qubit`` into ``cbit=qubit`` -- every
    existing circuit -- is not modified at all.
    """
    mapping = measured_classical_map(circuit)
    if mapping is None:
        return result
    try:
        n = int(circuit.n_qubits)
    except (TypeError, ValueError):
        return result

    counts = getattr(result, "counts", None)
    if counts:
        result.counts = rekey(counts, mapping, n)

    probabilities = getattr(result, "probabilities", None)
    if probabilities is not None:
        from superfermion.results import LazyDict

        # ``isinstance`` before ``bool``: ``LazyDict.__bool__`` materialises
        # the whole 2^n table, which is precisely what it exists to avoid.
        if isinstance(probabilities, LazyDict):
            result.probabilities = LazyDict(
                lambda table=probabilities: rekey(table, mapping, n)
            )
        elif probabilities:
            result.probabilities = rekey(probabilities, mapping, n)

    metadata = getattr(result, "metadata", None)
    if isinstance(metadata, dict):
        metadata["counts_keyed_by"] = "classical_bits"
        metadata["qubit_char_index"] = char_index_for_qubit(mapping, n)
    return result
