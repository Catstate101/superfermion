"""Exact Python-side repairs for wrong and missing Rust basis-translation arms.
=============================================================================

``crates/sf-compiler/src/decompose/basis.rs`` translates gates the target does
not implement.  Two of its arms are numerically wrong, and four gate types have
no arm at all; both problems are repaired here.

``OpType::Ry`` on an ``SX`` device
    emits ``Rz(pi/2), SX, Rz(theta - pi), SX, Rz(-3 pi/2)``.  That is not a
    global phase of ``Ry(theta)``: the unitary overlap is 0.977668245 for
    every angle, i.e. a genuinely different rotation.

``cx_decomp`` on an ``ECR`` device
    emits ``Rz(pi/2)_c, Rx(pi/2)_c, ECR, Rx(pi/2)_t``, which leaves a
    spurious local ``pi/2`` rotation behind.  The overlap with ``CX`` is
    0.707106781 -- ``|cos(pi/4)|``, the signature of a half-turn error.

Both are repaired here, additively.  The Rust pass is untouched, so every
branch that is *not* broken -- and every target that does not select these
arms -- keeps the exact behaviour it had before.  The identities used are

.. code-block:: text

    Ry(theta)           -> Rz(-pi/2), Rx(theta), Rz(pi/2)          (exact)
    CX(control, target) -> Rz(pi)_t, SX_t, Rz(pi)_t,
                           ECR(control, target), X_c, Rz(-pi/2)_c   (up to phase)

``Ry`` is lowered *before* the Rust call, and only when the basis cannot
express it (``sx`` present, ``ry`` absent): the arm is selected on ``SX`` and
is checked before the ``RZ``/``RX`` arm, so ``Rigetti``/``IonQ``/``linear``
targets -- which already take the correct ``Rz(-pi/2) Rx(theta) Rz(pi/2)``
branch -- are left alone.

``CX`` is lowered *after* the Rust call.  By then the Rust decomposer has
already turned every two-qubit gate (``CZ``, ``SWAP``, ``CCX``, ``CP``,
``ISWAP``, ``RZZ``, ``RXX``, ``RYY``, ``CY``) into ``CX`` gates through its
correct ``has("CX")`` arm, so a single rewrite of one gate type is enough and
none of the Rust identities has to be duplicated in Python.

A third gap is repaired here as well, and it is a different kind of gap: ``CU``,
``CRX``, ``CRZ`` and ``CSWAP`` have **no translator arm at all**, so compiling
against a constrained target returns them untouched and the "compiled" circuit
still is not in the target gate set.  Measured on all eight presets and every
optimisation level: ``sf.compile(Circuit(3).cu(...), target=ibm_eagle)`` emits a
single ``CU``.  :meth:`Circuit.ch` and :meth:`Circuit.cry` lower to ``CU``, so
they are affected too.  ``lower_untranslated`` rewrites them *before* the Rust
call, into forms the existing passes already handle:

* ``CU``/``CRX``/``CRZ`` are decomposed by :class:
  ``superfermion.compiler.passes.UnitaryDecompositionPass`` -- the same public
  pass ``sf.compile()`` already uses for opaque unitaries -- into ``RZ``/``RY``
  /``CNOT``, which is inside the Rust compiler's input vocabulary.
  They cannot be handed over as an ``OpType::Unitary``: the Rust ``Compiler``
  has no arm that synthesises a matrix, leaves it untouched, and
  ``Dag::to_gate_records()`` has no name for it, so the "compiled" circuit came
  back containing a gate literally called ``unknown`` and raised
  ``ValueError: Unknown gate: 'unknown'`` on the first ``to_ir()``.
* ``CSWAP`` becomes ``CX, CCX, CX`` -- exact, not up to phase -- because the
  decomposer refuses 3-qubit unitaries by design.

Both rewrites are pinned by tests that compare the full circuit unitary before
and after ``compile()``.  The controlled-U rewrite matches the input to
abs(trace(U_in^dag U_out))/d = 1.000000000 and differs from the input matrix by
a global phase for ``CU``/``CH`` (their controlled block has a non-trivial
determinant phase) -- exactly as the existing public unitary path does.

``CSWAP`` is additionally reduced to the Rust input vocabulary, because the
Rust ``Compiler`` does *not* decompose ``CCX``: it validates the output against
the target gate set and raises
``RuntimeError: Encountered unsupported gate CCX/Toffoli after decomposition``.
A *classically conditioned* ``CSWAP`` is left exactly as it was.  The Rust
``Compiler`` has no notion of classical conditions end-to-end, so lowering it
would only move the silent guard drop one stage later; ``sf.compile()`` now
rejects conditioned circuits up front with a clear error, and the pass-through
keeps direct ``prepare()`` users from getting an uncontrolled sequence.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from superfermion.circuit import Circuit, GateRecord

__all__ = [
    "CX_VIA_ECR",
    "ECR_LOWERING_KEY",
    "RZ_LOWERING_KEY",
    "RUST_INPUT_VOCABULARY",
    "UNTRANSLATED_OPCODES",
    "UNTRANSLATED_THREE_QUBIT",
    "UNTRANSLATED_TWO_QUBIT",
    "cx_via_ecr",
    "cswap_via_ccx",
    "finalize",
    "lower_cx_to_ecr",
    "lower_ry",
    "lower_untranslated",
    "needs_ecr_lowering",
    "needs_ry_lowering",
    "needs_untranslated_lowering",
    "prepare",
]

PI = math.pi
HALF_PI = PI / 2.0

#: ``Circuit._metadata`` keys describing what the lowerings did.
RZ_LOWERING_KEY = "ry_lowering"
ECR_LOWERING_KEY = "ecr_lowering"

#: Names the Rust pipeline accepts for a controlled-X.
CX_NAMES = frozenset({"CX", "CNOT"})

#: Two-qubit controlled opcodes with no ``BasisTranslationPass`` arm.  They are
#: handed to ``UnitaryDecompositionPass`` as their own matrix instead.
UNTRANSLATED_TWO_QUBIT = frozenset({"CU", "CRX", "CRZ"})

#: Three-qubit opcodes with no arm.  ``UnitaryDecompositionPass`` only accepts
#: 1Q/2Q blocks, so these need a gate identity rather than a matrix.
UNTRANSLATED_THREE_QUBIT = frozenset({"CSWAP"})

UNTRANSLATED_OPCODES = UNTRANSLATED_TWO_QUBIT | UNTRANSLATED_THREE_QUBIT

#: Gate names the Rust ``Compiler`` accepts as input.  Mirrors
#: ``superfermion.compiler.manager._RUST_KNOWN_GATES``; a test pins the two
#: sets against each other so they cannot drift apart silently.
RUST_INPUT_VOCABULARY = frozenset({
    "CX", "CNOT", "CZ", "SWAP", "H", "X", "Y", "Z", "S", "SDG", "T", "TDG",
    "ID", "RX", "RY", "RZ", "SX", "P",
})


# ────────────────────────────────────────────────────────────────────────
# helpers
# ────────────────────────────────────────────────────────────────────────
def _normalise(names: Optional[Iterable[str]]) -> set:
    """Lower-case, de-duplicated gate names; ``{"all"}`` is kept meaningful."""
    return {str(name).strip().lower() for name in (names or ())}


def _compile_basis(target: Any) -> List[str]:
    """The gate set the target hands to the Rust translator."""
    basis = getattr(target, "compile_basis", None)
    if basis is None:
        basis = getattr(target, "native_gates", [])
    return list(basis)


def _make(
    name: str,
    qubits: Sequence[int],
    params: Sequence[Any] = (),
    condition: Optional[Tuple[int, int]] = None,
) -> GateRecord:
    """Build one uppercase ``GateRecord``."""
    return GateRecord(
        name=name,
        qubits=[int(qubit) for qubit in qubits],
        params=list(params),
        condition=condition,
    )


def _rebuild(circuit: Circuit, records: List[GateRecord]) -> Circuit:
    """Return a copy of ``circuit`` whose gate list is ``records``."""
    new = Circuit(
        int(circuit.n_qubits),
        n_cbits=int(circuit.n_cbits),
        name=getattr(circuit, "name", None),
    )
    new.extend_raw(records)
    metadata = getattr(circuit, "_metadata", None)
    if metadata:
        new._metadata.update(metadata)
    return new


# ────────────────────────────────────────────────────────────────────────
# detection
# ────────────────────────────────────────────────────────────────────────
def needs_ry_lowering(basis: Optional[Iterable[str]]) -> bool:
    """Whether the (wrong) ``SX`` arm of ``OpType::Ry`` will be taken.

    The arm is guarded by ``has("SX")`` and is evaluated *before* the
    ``RZ``/``RX`` arm, so ``rx`` in the basis does not avoid it -- only an
    explicit ``ry`` (or the absence of ``sx``) does.
    """
    names = _normalise(basis)
    return "sx" in names and "ry" not in names and "all" not in names


def needs_ecr_lowering(
    basis: Optional[Iterable[str]],
    native_gates: Optional[Iterable[str]] = None,
) -> bool:
    """Whether ``CX`` must be rewritten into an ``ECR`` sequence.

    True only for targets that implement ``ECR`` and have no native
    ``CX``/``CNOT``/``CZ`` -- exactly the case in which ``cx_decomp`` takes
    its broken ``ECR`` arm.  The test is made against ``native_gates``, not
    against the translator basis: the basis deliberately contains ``cx`` so
    that the Rust decomposer uses its *correct* controlled-X arm.
    """
    if "all" in _normalise(basis) or "all" in _normalise(native_gates):
        return False
    native = _normalise(native_gates if native_gates is not None else basis)
    if "ecr" not in native:
        return False
    if {"cx", "cnot", "cz"} & native:
        return False
    return True


def needs_untranslated_lowering(basis: Optional[Iterable[str]]) -> bool:
    """Whether opcodes with no translator arm must be rewritten up front.

    Only when a constrained basis was actually requested.  A ``"all"`` target
    (the simulator presets) accepts everything, and passing ``CU`` straight
    through is both cheaper and exactly what the user asked for; an empty basis
    means no target was given at all.
    """
    names = _normalise(basis)
    return bool(names) and "all" not in names


# ────────────────────────────────────────────────────────────────────────
# CX -> native ECR
# ────────────────────────────────────────────────────────────────────────
def cx_via_ecr(
    control: int,
    target: int,
    condition: Optional[Tuple[int, int]] = None,
) -> List[GateRecord]:
    """``CX(control, target)`` as ``{rz, sx, x, ecr}`` only, up to a global phase.

    Verified against ``Circuit.to_unitary()``: overlap 1.000000000 for both
    ``CX(0, 1)`` and ``CX(1, 0)`` (residual phase ``-pi/2``, a scalar that is
    common to the whole circuit, exactly like the compiler's existing
    ``H = Rz(pi/2) SX Rz(pi/2)`` identity).
    """
    control = int(control)
    target = int(target)
    return [
        _make("RZ", [target], [PI], condition),
        _make("SX", [target], (), condition),
        _make("RZ", [target], [PI], condition),
        _make("ECR", [control, target], (), condition),
        _make("X", [control], (), condition),
        _make("RZ", [control], [-HALF_PI], condition),
    ]


CX_VIA_ECR: Tuple[str, ...] = ("rz", "sx", "x", "ecr")


def lower_cx_to_ecr(circuit: Circuit) -> Tuple[Circuit, int]:
    """Replace every ``CX``/``CNOT`` gate with the native ``ECR`` sequence.

    Returns the (possibly new) circuit and the number of gates rewritten.
    ``circuit`` is returned unchanged -- the very same object -- when there is
    nothing to do, so callers can cheaply detect a no-op.
    """
    circuit._ensure_gates()
    records: List[GateRecord] = []
    rewritten = 0
    for record in circuit._gates:
        if record.name.upper() in CX_NAMES and len(record.qubits) == 2:
            records.extend(
                cx_via_ecr(record.qubits[0], record.qubits[1], record.condition)
            )
            rewritten += 1
        else:
            records.append(record)
    if not rewritten:
        return circuit, 0
    return _rebuild(circuit, records), rewritten


# ────────────────────────────────────────────────────────────────────────
# RY -> RZ/RX
# ────────────────────────────────────────────────────────────────────────
def lower_ry(circuit: Circuit) -> Tuple[Circuit, int]:
    """Replace ``RY(theta)`` with ``RZ(-pi/2), RX(theta), RZ(pi/2)``.

    This is the ``RZ``/``RX`` arm the Rust pass already uses for targets
    without ``SX``, and it is exact: ``Circuit(1).ry(t, 0).to_unitary()``
    equals ``RZ(pi/2) @ RX(t) @ RZ(-pi/2)`` bit for bit (matrix products read
    right to left, hence the emission order).
    """
    circuit._ensure_gates()
    records: List[GateRecord] = []
    rewritten = 0
    for record in circuit._gates:
        if (
            record.name.upper() == "RY"
            and len(record.qubits) == 1
            and len(record.params) == 1
        ):
            qubit = int(record.qubits[0])
            theta = record.params[0]
            condition = record.condition
            records.append(_make("RZ", [qubit], [-HALF_PI], condition))
            records.append(_make("RX", [qubit], [theta], condition))
            records.append(_make("RZ", [qubit], [HALF_PI], condition))
            rewritten += 1
        else:
            records.append(record)
    if not rewritten:
        return circuit, 0
    return _rebuild(circuit, records), rewritten


# ────────────────────────────────────────────────────────────────────────
# armless opcodes -> something the existing passes already handle
# ────────────────────────────────────────────────────────────────────────
def cswap_via_ccx(
    control: int,
    qubit1: int,
    qubit2: int,
    condition: Optional[Tuple[int, int]] = None,
) -> List[GateRecord]:
    """``CSWAP(c, a, b)`` as ``CX(a, b), CCX(c, b, a), CX(a, b)``.

    Checked against ``Circuit.to_unitary()`` for several wire assignments: the
    product reproduces the Fredkin matrix bit for bit (overlap 1.000000000),
    not merely up to a global phase.  ``CCX`` is deliberately used rather than
    an 8x8 ``UNITARY`` because ``UnitaryDecompositionPass`` refuses 3-qubit
    blocks outright.
    """
    control, qubit1, qubit2 = int(control), int(qubit1), int(qubit2)
    return [
        _make("CX", [qubit1, qubit2], (), condition),
        _make("CCX", [control, qubit2, qubit1], (), condition),
        _make("CX", [qubit1, qubit2], (), condition),
    ]


def _decompose_two_qubit(record: GateRecord, n_qubits: int) -> List[GateRecord]:
    """The armless 2-qubit gate expressed in the Rust input vocabulary.

    The record matrix (``GateRecord.to_unitary()``, ``qubits[0]`` most
    significant) is loaded through :meth:`Circuit.unitary`, which expects that
    very convention, and decomposed by the public
    ``UnitaryDecompositionPass``.  The result is ``RZ``/``RY``/``CNOT`` only.

    A classical ``condition`` on the original gate is carried to every emitted
    record: distributing one condition over a sequence that implements the
    gate is exactly the same controlled operation.
    """
    from superfermion.compiler.passes import UnitaryDecompositionPass

    sub = Circuit(int(n_qubits))
    sub.unitary(
        np.asarray(record.to_unitary(), dtype=complex),
        [int(q) for q in record.qubits],
    )
    decomposed = UnitaryDecompositionPass().run(sub)
    decomposed._ensure_gates()
    return [
        _make(gate.name.upper(), gate.qubits, gate.params, record.condition)
        for gate in decomposed._gates
    ]


def _to_rust_vocabulary(records: List[GateRecord], n_qubits: int) -> List[GateRecord]:
    """Reduce ``records`` to gates the Rust ``Compiler`` accepts as input.

    Reuses :class:`superfermion.compiler.passes.BasisTranslationPass` with the
    Rust input vocabulary -- the very translation ``sf.compile()`` already
    applies before handing a circuit to Rust.  Conditions are preserved by
    refusing to translate a conditioned record (see the module docstring).
    """
    from superfermion.compiler.passes import BasisTranslationPass

    sub = Circuit(int(n_qubits))
    sub.extend_raw(list(records))
    translated = BasisTranslationPass(sorted(RUST_INPUT_VOCABULARY)).run(sub)
    translated._ensure_gates()
    return list(translated._gates)


def lower_untranslated(circuit: Circuit) -> Tuple[Circuit, int]:
    """Rewrite the armless opcodes into forms the Rust passes do handle.

    ``CU``/``CRX``/``CRZ`` become ``RZ``/``RY``/``CNOT`` sequences and
    ``CSWAP`` becomes ``CX, CCX, CX``.  Every other record is passed through
    untouched, and the circuit object itself is returned unchanged when there
    is nothing to rewrite.

    Returns the (possibly new) circuit and the number of gates rewritten.
    """
    circuit._ensure_gates()
    records: List[GateRecord] = []
    rewritten = 0
    for record in circuit._gates:
        name = record.name.upper()
        if (
            name in UNTRANSLATED_THREE_QUBIT
            and len(record.qubits) == 3
            and record.condition is None
        ):
            control, qubit1, qubit2 = (int(q) for q in record.qubits)
            records.extend(
                _to_rust_vocabulary(
                    cswap_via_ccx(control, qubit1, qubit2), circuit.n_qubits
                )
            )
            rewritten += 1
        elif name in UNTRANSLATED_TWO_QUBIT and len(record.qubits) == 2:
            records.extend(_decompose_two_qubit(record, circuit.n_qubits))
            rewritten += 1
        else:
            records.append(record)
    if not rewritten:
        return circuit, 0
    return _rebuild(circuit, records), rewritten


# ────────────────────────────────────────────────────────────────────────
# pipeline hooks
# ────────────────────────────────────────────────────────────────────────
def _empty_report() -> Dict[str, Any]:
    return {"ry_lowered": 0, "cx_lowered": 0, "untranslated_lowered": 0, "reasons": []}


def prepare(circuit: Circuit, target: Any = None) -> Tuple[Circuit, Dict[str, Any]]:
    """Lower gates the Rust pass would translate wrongly or not translate at all."""
    report = _empty_report()
    if target is None:
        return circuit, report
    basis = _compile_basis(target)
    if needs_untranslated_lowering(basis):
        circuit, count = lower_untranslated(circuit)
        if count:
            report["untranslated_lowered"] = count
            report["reasons"].append("armless_opcode")
    if needs_ry_lowering(basis):
        circuit, count = lower_ry(circuit)
        if count:
            report["ry_lowered"] = count
            report["reasons"].append("ry_sx_arm")
    return circuit, report


def finalize(
    circuit: Circuit, target: Any = None, report: Optional[Dict[str, Any]] = None
) -> Tuple[Circuit, Dict[str, Any]]:
    """Rewrite ``CX`` into the native ``ECR`` sequence (post-pass)."""
    if report is None:
        report = _empty_report()
    if target is None:
        return circuit, report
    basis = _compile_basis(target)
    if needs_ecr_lowering(basis, getattr(target, "native_gates", None)):
        circuit, count = lower_cx_to_ecr(circuit)
        if count:
            report["cx_lowered"] = count
            report["reasons"].append("cx_ecr_arm")
    return circuit, report
