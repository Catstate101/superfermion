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


# ── E/F: the 0.1.25 observable and metric fixes ──────────────────────────────
#
# Defect ledger for this section (numbers measured in _xq6_grad.py against
# Qiskit 2.6.1, which refuses the same inputs and inverts the same metric):
#
# E. the five gradient entry points each coerced observables their own way --
#    ``list[PauliString]`` reached only the adjoint and the engine term list
#    only ``RunResult.grad``;
# V. a label whose width was not the register's was accepted and meant a
#    different operator (``'Z'`` on two wires acted as ``II``), and a
#    non-Hermitian operator was differentiated three different ways;
# W. ``qng_step`` assembled the gradient in the caller's order but left the
#    qfim in the dag's, which moved the step by 0.024420, and used
#    ``np.linalg.solve`` on a rank-deficient metric, which returned
#    |dtheta| = 532 instead of raising.


def _f_metric_circuit():
    """Three symbols on two qubits, ``zz`` repeated across an RY and a P."""
    zz, aa, mm = sf.param("zz"), sf.param("aa"), sf.param("mm")
    circuit = sf.Circuit(2).h(0).ry(zz, 1).cx(0, 1).rz(aa, 0).rx(mm, 1).p(zz, 0)
    return circuit, {"zz": 0.37, "aa": 0.4, "mm": 0.9}


def test_state_grad_documents_itself():
    """``help(sf.State.grad)`` must explain the term form and the ordering."""
    import inspect

    for method, needles in (
        (sf.State.grad, ("adjoint", "parameter_names", "Hermitian", "ValueError")),
        (sf.State.qfim, ("Bures", "parameter_names", "permute", "pinv")),
    ):
        doc = inspect.getdoc(method)
        assert doc, f"{method.__name__} has no docstring"
        missing = [n for n in needles if n not in doc]
        assert not missing, f"{method.__name__} docstring omits {missing}"


def test_state_grad_refuses_a_label_narrower_than_the_register():
    """``'Z'`` on two wires used to mean ``II`` and return a plausible zero."""
    circuit, values = _f_metric_circuit()
    state = sf.run(circuit, params=values).state
    dag = circuit.to_ir()

    with pytest.raises(ValueError, match="wire"):
        state.grad([([3], 1.0, 0.0)], dag, values)
    with pytest.raises(ValueError, match="wire"):
        state.grad("IZI", dag, values)


def test_state_grad_refuses_a_non_hermitian_operator():
    """A complex coefficient has no real derivative; Qiskit refuses it too."""
    circuit, values = _f_metric_circuit()
    state = sf.run(circuit, params=values).state
    dag = circuit.to_ir()

    with pytest.raises(ValueError, match="Hermitian"):
        state.grad([([3, 3], 1.0, 0.5)], dag, values)
    # A Hermitian complex sum is still accepted.
    assert state.grad({"ZZ": 1.0 + 0.0j}, dag, values)["aa"] == pytest.approx(
        state.grad("ZZ", dag, values)["aa"], abs=1e-12
    )


def test_state_grad_accepts_a_label_string_and_the_engine_form_alike():
    """Both spellings reach the same engine and must not disagree."""
    circuit, values = _f_metric_circuit()
    state = sf.run(circuit, params=values).state
    dag = circuit.to_ir()

    from_terms = state.grad([([3, 0], 1.0, 0.0), ([0, 3], 0.5, 0.0)], dag, values)
    from_dict = state.grad({"ZI": 1.0, "IZ": 0.5}, dag, values)
    for name in values:
        assert from_terms[name] == pytest.approx(from_dict[name], abs=1e-12)


def test_qng_step_solves_in_the_callers_parameter_order():
    """F/V-10: a metric tensor is a matrix of coordinates -- permute both axes."""
    from superfermion.qml.gradient.qng import qng_step

    circuit, values = _f_metric_circuit()
    dag = circuit.to_ir()
    names = list(dag.parameter_names())
    state = sf.run(circuit, params=values).state
    observable = "ZZ"
    lr, reg = 0.05, 1e-6

    metric = np.asarray(state.qfim(dag, values), dtype=float)
    grad = adjoint_grad(circuit, observable, values)

    shuffled = list(reversed(names))
    perm = [names.index(name) for name in shuffled]
    g_shuffled = np.array([grad[name] for name in shuffled])
    m_shuffled = metric[np.ix_(perm, perm)]

    step = qng_step(state, dag, observable, shuffled, values,
                    learning_rate=lr, regularization=reg)
    solved = np.linalg.solve(m_shuffled + reg * np.eye(len(names)), g_shuffled)
    for name, delta in zip(shuffled, solved):
        assert step[name] == pytest.approx(values[name] - lr * delta, abs=1e-10)

    # ... and the old mixed-basis answer is measurably a different step, so
    # the row above is not vacuous.
    mixed = np.linalg.solve(metric + reg * np.eye(len(names)), g_shuffled)
    gap = max(abs((values[name] - lr * m) - step[name])
              for name, m in zip(shuffled, mixed))
    assert gap > 1e-3, f"the two bases coincide here (gap {gap:.3e}); test is vacuous"


def test_qng_step_refuses_an_unknown_parameter_name():
    """An unknown name used to be filled with 0.0 and silently ignored."""
    from superfermion.qml.gradient.qng import qng_step

    circuit, values = _f_metric_circuit()
    dag = circuit.to_ir()
    state = sf.run(circuit, params=values).state

    with pytest.raises(ValueError, match="not a parameter"):
        qng_step(state, dag, "ZZ", ["zz", "nope"], values)


def test_qng_step_survives_a_rank_deficient_metric():
    """F/V-11: ``solve`` on a singular metric returned |dtheta| = 532, not an error.

    ``ry(a) ry(b) = ry(a + b)``, so the Fubini-Study tensor is exactly
    ``[[1,1],[1,1]]``: rank 1 in a 2-dimensional space.  The unregularized
    step must stay in the parameter domain and must be the same minimum-norm
    step ``sr_step`` (which has always used the pseudo-inverse) takes.
    """
    from superfermion.qml.gradient.qng import qng_step
    from superfermion.qml.gradient.stochastic_reconfig import sr_step

    a, b = sf.param("a"), sf.param("b")
    circuit = sf.Circuit(1).ry(a, 0).ry(b, 0)
    values = {"a": 0.4, "b": 0.9}
    dag = circuit.to_ir()
    names = list(dag.parameter_names())
    state = sf.run(circuit, params=values).state
    lr = 0.05

    metric = np.asarray(state.qfim(dag, values), dtype=float)
    assert np.linalg.matrix_rank(metric) == 1, f"expected a rank-1 metric, got {metric}"

    step = qng_step(state, dag, "X", names, values, learning_rate=lr,
                    regularization=0.0)
    sane = sr_step(state, dag, "X", names, values, learning_rate=lr,
                   regularization=0.0)

    worst = max(abs(step[name] - values[name]) for name in names)
    assert worst < 1.0, f"the step left the parameter domain: |dtheta|={worst:.3f}"
    assert worst > 1e-6, "the deficient metric silently did nothing"
    for name in names:
        assert step[name] == pytest.approx(sane[name], abs=1e-12)
