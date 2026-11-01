"""Regression tests for the 0.1.24 gradient-correctness fixes.

Every expected number is either the exact analytic value or a central finite
difference computed from SuperFermion's own ``expectation()`` — no external
simulator is involved, so the file runs on every CI row (pure numpy).

Defect ledger these tests pin down:

A. adjoint returned exactly 0.0 for ``p``/``u``/``cp``/``cu`` parameters
   (the Rust generator table stopped at Rx/Ry/Rz/Rxx/Ryy/Rzz and had no
   error branch for the rest).
B. adjoint returned 0.0 for a pre-bound circuit instead of raising.
C. parameter-shift halved the gradient when one name drove several gates
   (``bind()`` is name-global, so one shift moved every occurrence).
D. parameter-shift was wrong for controlled rotations (``cu``), which need
   the second, π-shift term of the {0, ±½} spectrum rule.
E. parameter-shift rejected ``str`` observables with an AttributeError.
F. ``r1``/``crx``/``cry``/``crz`` existed in the IR (and in both rule
   tables) but had no ``Circuit`` builder, so that coverage was unreachable
   from the Python API.
"""

from __future__ import annotations

import numpy as np
import pytest

import superfermion as sf
from superfermion.observables.core import SparsePauliOp
from superfermion.qml.gradient.adjoint import adjoint_grad
from superfermion.qml.gradient.parameter_shift import (
    finite_diff_grad,
    parameter_shift_grad,
)

pytestmark = pytest.mark.domain

FD_EPS = 1e-5


# ── Helpers ───────────────────────────────────────────────────────────────────


def finite_difference(circuit, observable, params, name, eps=FD_EPS):
    """Central difference of ⟨O⟩ over *every* occurrence of ``name``.

    The circuit's ``bind()`` is name-global, so a shared symbol is moved one
    occurrence at a time with ``bind_occurrence``; summing the per-occurrence
    differences gives the true total derivative.
    """
    total = 0.0
    for k in range(len(circuit.param_occurrences(name))):
        up = circuit.bind_occurrence(params, name, k, value=params[name] + eps)
        down = circuit.bind_occurrence(params, name, k, value=params[name] - eps)
        e_up = sf.run(up, shots=0).expectation(observable)
        e_down = sf.run(down, shots=0).expectation(observable)
        total += (e_up - e_down) / (2.0 * eps)
    return total


def reviewer_circuit(gate: str):
    """The circuit from the bug report, with ``p`` or ``rz`` on parameter ``a``."""
    a, b, c = sf.param("a"), sf.param("b"), sf.param("c")
    circ = sf.Circuit(2).ry(0.7, 0).ry(1.1, 1).cnot(0, 1)
    circ = circ.p(a, 0) if gate == "p" else circ.rz(a, 0)
    return circ.rx(b, 0).ry(c, 1).cnot(0, 1)


REVIEWER_PARAMS = {"a": 0.9, "b": -0.4, "c": 1.3}
#: Non-symmetric so that a qubit-order mistake cannot hide behind symmetry.
REVIEWER_OBS = {"YI": 1.0, "IZ": 0.5, "XX": 0.3}


# ── A: adjoint gate coverage ──────────────────────────────────────────────────


def test_adjoint_phase_gate_equals_rz():
    """``p(a)`` and ``rz(a)`` differ by a global phase, so the gradients match."""
    g_p = adjoint_grad(reviewer_circuit("p"), REVIEWER_OBS, REVIEWER_PARAMS)["a"]
    g_rz = adjoint_grad(reviewer_circuit("rz"), REVIEWER_OBS, REVIEWER_PARAMS)["a"]

    assert g_p != 0.0, "adjoint still drops the P gate (defect A)"
    assert abs(g_p - g_rz) < 1e-12
    # The value from the bug report, and the FD anchor on the rz variant.
    assert abs(g_p - 0.03887496) < 1e-8
    fd = finite_difference(
        reviewer_circuit("p"), REVIEWER_OBS, REVIEWER_PARAMS, "a"
    )
    assert abs(g_p - fd) < 1e-8


def test_adjoint_u_gate_all_three_slots():
    """Every angle of ``u(θ, φ, λ)`` gets its own generator (defect A)."""
    t1, t2, t3 = sf.param("t1"), sf.param("t2"), sf.param("t3")
    circuit = sf.Circuit(1).u(t1, t2, t3, 0)
    observable = {"X": 0.5, "Y": -0.3, "Z": 0.8}
    params = {"t1": 0.61, "t2": -0.42, "t3": 1.17}

    grad = adjoint_grad(circuit, observable, params)
    for name in params:
        fd = finite_difference(circuit, observable, params, name)
        assert grad[name] != 0.0, f"{name}: adjoint returned the old silent zero"
        assert abs(grad[name] - fd) < 1e-8, (
            f"{name}: adjoint {grad[name]!r} vs FD {fd!r}"
        )


def test_adjoint_cp_and_cu():
    """Controlled phase and controlled-U gradients against FD (defect A)."""
    t = sf.param("t")

    cp_circ = sf.Circuit(2).h(0).h(1).cp(t, 0, 1)
    cp_grad = adjoint_grad(cp_circ, {"XX": 1.0}, {"t": 0.7})["t"]
    assert abs(cp_grad - (-0.3221088436)) < 1e-8
    assert abs(cp_grad - finite_difference(cp_circ, {"XX": 1.0}, {"t": 0.7}, "t")) < 1e-8

    cu_circ = sf.Circuit(2).h(0).cu(t, 0.0, 0.0, 0, 1)
    cu_grad = adjoint_grad(cu_circ, {"XI": 1.0}, {"t": 0.7})["t"]
    assert abs(cu_grad - (-0.1714489037)) < 1e-8
    assert abs(cu_grad - finite_difference(cu_circ, {"XI": 1.0}, {"t": 0.7}, "t")) < 1e-8


# ── B: pre-bound circuit must raise, not return zero ─────────────────────────


def test_adjoint_raises_on_prebound_circuit():
    t = sf.param("t")
    circuit = sf.Circuit(1).ry(t, 0).ry(t, 0)
    bound = circuit.bind({"t": 0.35})

    with pytest.raises(ValueError) as excinfo:
        adjoint_grad(bound, {"Z": 1.0}, {"t": 0.35})
    message = str(excinfo.value)
    assert "to_ir()" in message
    assert "'t'" in message

    # The unbound circuit still works, so the guard is not hiding a bug.
    assert abs(
        adjoint_grad(circuit, {"Z": 1.0}, {"t": 0.35})["t"] + 1.2884353743969346
    ) < 1e-9


# ── A/B: every gradient entry point ──────────────────────────────────────────


def test_grad_entry_points_agree_on_p_gate():
    """``res.grad``, ``adjoint_grad`` and ``State.grad`` are one implementation."""
    from superfermion.results import _coerce_observable

    circuit = reviewer_circuit("p")
    dag = circuit.to_ir()
    result = sf.run(circuit.bind(REVIEWER_PARAMS), shots=0)

    from_result = result.grad(REVIEWER_OBS, dag, REVIEWER_PARAMS)["a"]
    from_adjoint = adjoint_grad(circuit, REVIEWER_OBS, REVIEWER_PARAMS)["a"]
    from_state = result.state.grad(
        _coerce_observable(REVIEWER_OBS, 2), dag, REVIEWER_PARAMS
    )["a"]

    assert from_result != 0.0
    assert abs(from_result - from_adjoint) < 1e-12
    assert abs(from_result - from_state) < 1e-12


# ── C/D/E: parameter-shift ───────────────────────────────────────────────────


def test_parameter_shift_shared_parameter():
    """``ry(t); ry(t)`` is two occurrences: the rule must sum both (defect C)."""
    t = sf.param("t")
    circuit = sf.Circuit(1).ry(t, 0).ry(t, 0)
    params = {"t": 0.35}

    grad = parameter_shift_grad(circuit, {"Z": 1.0}, params)["t"]
    assert abs(grad - (-1.2884353743969346)) < 1e-9
    assert abs(grad - finite_difference(circuit, {"Z": 1.0}, params, "t")) < 1e-8
    # The pre-0.1.24 name-global shift reported half of this (0.0 for t=0.35).
    assert abs(grad - finite_diff_grad(circuit, {"Z": 1.0}, params)["t"]) < 1e-6


def test_parameter_shift_controlled_rotation():
    """All four ``cu`` slots against FD — θ needs the π-shift term (defect D)."""
    names = ["theta", "phi", "lam", "gamma"]
    values = [0.7, 0.3, -0.5, 0.2]
    syms = [sf.param(n) for n in names]
    # Target starts in |+y> so that φ and λ also move the expectation value;
    # a circuit where a slot is trivially zero could hide a wrong rule.
    circuit = sf.Circuit(2).h(0).ry(0.6, 1).cu(syms[0], syms[1], syms[2], 0, 1, syms[3])
    observable = {"YI": 0.7, "XI": 0.3, "IZ": 0.5, "XX": 0.2}
    params = dict(zip(names, values))

    grad = parameter_shift_grad(circuit, observable, params)
    for name in names:
        fd = finite_difference(circuit, observable, params, name)
        assert grad[name] != 0.0, f"cu {name}: parameter-shift returned a silent zero"
        assert abs(grad[name] - fd) < 1e-8, (
            f"cu {name}: parameter-shift {grad[name]!r} vs FD {fd!r}"
        )

    # The exact number from the report, on the slot that used to be wrong.
    cu_circ = sf.Circuit(2).h(0).cu(sf.param("u"), 0.0, 0.0, 0, 1)
    cu_grad = parameter_shift_grad(cu_circ, {"XI": 1.0}, {"u": 0.7})["u"]
    assert abs(cu_grad - (-0.1714489037)) < 1e-8


def test_parameter_shift_string_and_dict_observables():
    """``str``/``dict``/``SparsePauliOp`` are the same observable (defect E)."""
    t = sf.param("t")
    circuit = sf.Circuit(2).ry(t, 0).cnot(0, 1)
    params = {"t": 0.4}

    as_str = parameter_shift_grad(circuit, "ZI", params)["t"]
    as_dict = parameter_shift_grad(circuit, {"ZI": 1.0}, params)["t"]
    as_spo = parameter_shift_grad(circuit, SparsePauliOp.from_dict({"ZI": 1.0}), params)[
        "t"
    ]
    as_list = parameter_shift_grad(circuit, ["ZI"], params)["t"]

    assert as_str != 0.0
    assert abs(as_str - as_dict) < 1e-12
    assert abs(as_str - as_spo) < 1e-12
    assert abs(as_str - as_list) < 1e-12


# ── End-to-end: the optimiser that used to stall ─────────────────────────────


def test_vqe_u_ansatz_converges():
    """``u_ansatz`` is the package's own U-gate template — it must optimise."""
    from superfermion.algorithms.variational import VQE
    from superfermion.qml.ansatz import u_ansatz

    vqe = VQE(
        u_ansatz(1, 1),
        SparsePauliOp.from_dict({"X": 1.0}),
        diff_method="adjoint",
    )
    result = vqe.minimize(iterations=200, seed=7)
    assert abs(result.optimal_value - (-1.0)) < 1e-6, (
        f"VQE stalled at {result.optimal_value!r}; all-zero adjoint gradients "
        f"leave the parameters where they started"
    )


def test_deep_shared_parameter_adjoint_matches_fd():
    """5q ring, one symbol per layer, reused by every qubit in the layer."""
    t = sf.param("t")
    circuit = sf.Circuit(5)
    for layer in range(3):
        for q in range(5):
            circuit.ry(t, q)
            circuit.rz(0.31 * (q + 1) - 0.2 * layer, q)
        for q in range(5):
            circuit.cnot(q, (q + 1) % 5)
    observable = {"ZIIIZ": 0.7, "XIIXZ": 0.3, "IZIZI": -0.5, "YIIYI": 0.25}
    params = {"t": -0.37}

    grad = adjoint_grad(circuit, observable, params)["t"]
    fd = finite_difference(circuit, observable, params, "t")
    assert np.all(np.isfinite([grad, fd]))
    assert abs(grad - fd) < 1e-8, f"adjoint {grad!r} vs FD {fd!r}"


# ── F: r1 / crx / cry / crz builders ─────────────────────────────────────────


def new_gate_circuit(gate: str, symbol):
    """Probe circuit with *gate* wired to *symbol* (control 0, target 1)."""
    circ = sf.Circuit(2).ry(0.7, 0).ry(1.1, 1).cnot(0, 1)
    if gate == "r1":
        circ = circ.r1(symbol, 0)
    else:
        circ = getattr(circ, gate)(symbol, 0, 1)
    return circ.rx(0.3, 0).ry(-0.4, 1)


#: Adjoint gradient from the cross-framework run (Qiskit/PennyLane agreed to
#: 8.3e-17), so a silent regression in the builder wiring cannot hide behind FD.
NEW_GATE_GRAD = {
    "r1": 0.0444400102,
    "crx": -0.2376428529,
    "cry": -0.1123078840,
    "crz": -0.0737024720,
}
NEW_GATE_OBS = {"YI": 0.7, "XI": 0.3, "IZ": 0.5, "XX": 0.2}


@pytest.mark.parametrize("gate", ["r1", "crx", "cry", "crz"])
def test_new_gate_builders_are_parametric(gate):
    """The builders reach the symbolic-parameter machinery like any other gate."""
    t = sf.param("t")
    circuit = new_gate_circuit(gate, t)
    assert len(circuit.param_occurrences("t")) == 1
    # A float angle must also work: these are ordinary fluent gates.
    bound = new_gate_circuit(gate, 0.83)
    assert abs(float(sf.run(bound, shots=0).expectation(NEW_GATE_OBS))) <= 1.0 + 1e-9


@pytest.mark.parametrize("gate", ["r1", "crx", "cry", "crz"])
def test_new_gate_gradients_match_fd(gate):
    """Adjoint and parameter-shift both differentiate the new builders (defect F)."""
    circuit = new_gate_circuit(gate, sf.param("t"))
    params = {"t": 0.83}

    fd = finite_difference(circuit, NEW_GATE_OBS, params, "t")
    assert abs(fd) > 1e-3, "probe circuit is insensitive to t (test would be vacuous)"

    adjoint = adjoint_grad(circuit, NEW_GATE_OBS, params)["t"]
    shifted = parameter_shift_grad(circuit, NEW_GATE_OBS, params)["t"]

    assert adjoint != 0.0, f"{gate}: adjoint returned a silent zero"
    assert abs(adjoint - NEW_GATE_GRAD[gate]) < 1e-9
    assert abs(adjoint - fd) < 1e-8, f"{gate}: adjoint {adjoint!r} vs FD {fd!r}"
    assert abs(shifted - fd) < 1e-8, f"{gate}: parameter-shift {shifted!r} vs FD {fd!r}"


def test_new_gate_semantics_through_the_simulator():
    """Control-on branch only, and the exact amplitudes each gate implies.

    SF's statevector is little-endian (qubit 0 is the least significant bit),
    the same order as ``Circuit.to_unitary()``. ``x(0)`` therefore prepares
    index 1 and ``x(1)`` index 2.
    """
    theta = 0.62
    cos, sin = np.cos(theta / 2), np.sin(theta / 2)

    def state(circuit):
        return np.asarray(sf.run(circuit, shots=0).statevector, dtype=complex)

    # Control in |0>: every controlled rotation must be a no-op.
    for gate in ("crx", "cry", "crz"):
        got = state(getattr(sf.Circuit(2).x(1), gate)(theta, 0, 1))
        assert np.allclose(got, [0, 0, 1, 0], atol=1e-12), f"{gate} touched the off branch"

    # Control in |1>: Rx, Ry, Rz act on the target, which starts in |0>.
    assert np.allclose(state(sf.Circuit(2).x(0).crx(theta, 0, 1)), [0, cos, 0, -1j * sin])
    assert np.allclose(state(sf.Circuit(2).x(0).cry(theta, 0, 1)), [0, cos, 0, sin])
    assert np.allclose(
        state(sf.Circuit(2).x(0).crz(theta, 0, 1)), [0, np.exp(-0.5j * theta), 0, 0]
    )

    # r1(φ) is a phase: on |+> it leaves |0> alone and multiplies |1> by e^{iφ}.
    plus = sf.Circuit(1).h(0).r1(theta, 0)
    assert np.allclose(state(plus), np.array([1, np.exp(1j * theta)]) / np.sqrt(2))


def test_new_gate_aliases_are_exact():
    """``r1 == p`` and ``cry == cu(θ,0,0)`` as matrices, not up to a phase."""
    theta = 0.5
    assert np.allclose(
        sf.Circuit(1).r1(theta, 0).to_unitary(),
        sf.Circuit(1).p(theta, 0).to_unitary(),
        atol=1e-14,
    )
    assert np.allclose(
        sf.Circuit(2).cry(theta, 0, 1).to_unitary(),
        sf.Circuit(2).cu(theta, 0.0, 0.0, 0, 1).to_unitary(),
        atol=1e-14,
    )
    # And crz is not cp — a control-branch phase, not a target-branch one.
    assert not np.allclose(
        sf.Circuit(2).crz(theta, 0, 1).to_unitary(),
        sf.Circuit(2).cp(theta, 0, 1).to_unitary(),
        atol=1e-2,
    )
