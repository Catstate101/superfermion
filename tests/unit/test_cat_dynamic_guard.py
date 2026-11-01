"""cat: dynamic-circuit guard + thread-cap regression tests.

``sf.cat(circuit, mode="stabilizer")`` samples the *final* stabilizer state.
Before this guard, a mid-circuit measurement, a reset after use, or a
classically-conditioned gate was skipped silently, so the caller received the
unconditional distribution — a valid answer to a different question.  These
tests pin the loud rejection, the patterns that must stay legal, and the
``SF_CAT_THREADS`` cap (identical counts, different thread count).
"""

import json
import os
import subprocess
import sys

import numpy as np
import pytest

import superfermion as sf

pytestmark = pytest.mark.unit


def _ghz(n=2):
    circuit = sf.Circuit(n)
    circuit.h(0)
    for q in range(1, n):
        circuit.cx(0, q)
    return circuit


def _reject(circuit, mode="stabilizer"):
    with pytest.raises(ValueError) as excinfo:
        sf.cat(circuit, mode=mode)
    return str(excinfo.value)


def test_terminal_measure_all_is_accepted():
    circuit = _ghz(2)
    circuit.measure_all()
    counts = sf.cat(circuit, mode="stabilizer").sample(500, 7)
    assert sum(counts.values()) == 500
    assert set(counts) == {"00", "11"}


def test_terminal_single_measurement_is_accepted():
    circuit = _ghz(2)
    circuit.measure(0, 0)
    counts = sf.cat(circuit, mode="stabilizer").sample(200, 3)
    assert sum(counts.values()) == 200


def test_default_mode_also_rejects_a_gate_after_measurement():
    circuit = _ghz(2)
    circuit.measure(0, 0)
    circuit.x(0)
    message = _reject(circuit)
    assert "measured" in message
    assert "simulate_dynamic" in message


def test_feed_forward_is_rejected():
    circuit = sf.Circuit(2, n_cbits=1)
    circuit.h(0)
    circuit.measure(0, 0)
    circuit.c_if(0, 1).x(1)
    message = _reject(circuit)
    assert "conditioned" in message
    assert "simulate_dynamic" in message


def test_reset_after_use_is_rejected():
    circuit = _ghz(2)
    circuit.reset(0)
    message = _reject(circuit)
    assert "Reset on qubit 0" in message


def test_leading_reset_is_accepted():
    circuit = sf.Circuit(2)
    circuit.reset(0)
    circuit.h(0)
    circuit.cx(0, 1)
    counts = sf.cat(circuit, mode="stabilizer").sample(100, 5)
    assert sum(counts.values()) == 100


def test_clifford_mode_rejects_the_same_circuit():
    circuit = _ghz(2)
    circuit.measure(0, 0)
    circuit.h(0)
    _reject(circuit, mode="clifford")


def test_statevector_mode_keeps_its_own_guard():
    """The state lane refuses a collapse, and drops only a *trailing* readout.

    `prepare()` used to refuse every measurement; the lane now runs the same
    guard as the tableau modes first, so a gate on a measured qubit is reported
    with the shared, lane-labelled message.
    """
    circuit = _ghz(2)
    circuit.measure(0, 0)
    circuit.x(0)
    with pytest.raises(ValueError) as excinfo:
        sf.cat(circuit, mode="statevector")
    assert "after it was measured" in str(excinfo.value)


def test_statevector_mode_accepts_a_trailing_readout():
    """A readout nothing follows cannot change the state, so it is dropped."""
    bare = _ghz(2)
    measured = _ghz(2)
    measured.measure_all()
    a = sf.cat(bare, mode="statevector").simulate({}).numpy()
    b = sf.cat(measured, mode="statevector").simulate({}).numpy()
    np.testing.assert_array_equal(a, b)


def test_statevector_mode_refuses_a_readout_that_is_not_trailing():
    """A gate on a *different* qubit still makes the readout droppable-wrong.

    Dropping it would hand back a state in which qubit 0 is still coherent,
    which is not the state of the circuit that was asked for.
    """
    circuit = _ghz(2)
    circuit.measure(0, 0)
    circuit.x(1)
    with pytest.raises(ValueError) as excinfo:
        sf.cat(circuit, mode="statevector")
    assert "runs after a measurement" in str(excinfo.value)


def test_statevector_mode_accepts_a_leading_reset():
    """reset(0) on a fresh qubit is |0> -> |0>: the state must not move.

    Qiskit's ``Statevector.from_instruction`` gives the same answer for this
    shape.  The statevector lane used to refuse the whole circuit, because
    ``prepare()`` builds a unitary and cannot host a reset; the leading reset is
    now dropped before ``prepare()`` ever sees the DAG.
    """
    bare = _ghz(2)
    with_reset = sf.Circuit(2)
    with_reset.reset(0)
    with_reset.h(0)
    with_reset.cx(0, 1)
    a = sf.cat(bare, mode="statevector").simulate({}).numpy()
    b = sf.cat(with_reset, mode="statevector").simulate({}).numpy()
    np.testing.assert_array_equal(a, b)


def test_statevector_mode_still_refuses_a_reset_after_use():
    """A real reset yields a mixed state, which one pure vector cannot hold.

    This is deliberately stricter than Qiskit's pure-state path, which
    post-selects and returns a definite state instead of the channel output.
    """
    circuit = _ghz(2)
    circuit.reset(0)
    with pytest.raises(ValueError) as excinfo:
        sf.cat(circuit, mode="statevector")
    assert "Reset on qubit 0" in str(excinfo.value)


def test_adjoint_accepts_a_leading_reset():
    """reset(0) on a fresh qubit is |0> -> |0>: the gradient must not move.

    The stabilizer/clifford lanes already skipped these; the adjoint lane used
    to refuse the whole circuit, which made hardware-style circuits (which
    start with resets) impossible to differentiate.
    """
    t = sf.param("t")

    def build(with_reset: bool):
        c = sf.Circuit(2)
        if with_reset:
            c.reset(0)
        c.ry(t, 0)
        c.cx(0, 1)
        return c

    plain = sf.cat(build(False), mode="adjoint").run_gradient("XX", [0.7])
    with_reset = sf.cat(build(True), mode="adjoint").run_gradient("XX", [0.7])
    assert with_reset == plain
    # cos(0.7), so the probe is not comparing zeros.
    assert abs(plain[0] - 0.7648421872844884) < 1e-12


def test_adjoint_rejects_a_collapse_it_cannot_differentiate():
    """ry(theta) ; measure ; rz(phi) -> <X> == 0, so d<X>/d(th,ph) == 0.

    The adjoint program differentiates the unitary.  Before the guard this
    returned cos(theta)cos(phi) = 0.7307 at (0.7, 0.3): a gradient that does
    not exist, returned without complaint.
    """
    theta, phi = sf.param("theta"), sf.param("phi")
    circuit = sf.Circuit(1, n_cbits=1)
    circuit.ry(theta, 0)
    circuit.measure(0, 0)
    circuit.rz(phi, 0)
    with pytest.raises(ValueError) as excinfo:
        sf.cat(circuit, mode="adjoint")
    message = str(excinfo.value)
    assert "measured" in message
    assert "simulate_dynamic" in message


def test_adjoint_rejects_feed_forward():
    t0, t1 = sf.param("t0"), sf.param("t1")
    circuit = sf.Circuit(2, n_cbits=1)
    circuit.ry(t0, 0)
    circuit.measure(0, 0)
    circuit.c_if(0, 1).ry(t1, 1)
    with pytest.raises(ValueError) as excinfo:
        sf.cat(circuit, mode="adjoint")
    assert "conditioned" in str(excinfo.value)


def test_adjoint_still_accepts_a_terminal_measurement():
    """A terminal measurement cannot change a unitary: keep it legal."""
    theta = sf.param("theta")
    circuit = sf.Circuit(2, n_cbits=2)
    circuit.ry(theta, 0)
    circuit.cx(0, 1)
    circuit.measure_all()
    handle = sf.cat(circuit, mode="adjoint")
    assert list(handle.parameter_names) == ["theta"]


_SAMPLE_SCRIPT = """
import json
import superfermion as sf

n = 32
circuit = sf.Circuit(n)
for q in range(n):
    circuit.h(q)
for q in range(n - 1):
    circuit.cx(q, q + 1)
counts = sf.cat(circuit, mode="stabilizer").sample(2_000, 7)
print(json.dumps(sorted(counts.items())))
"""


def test_thread_cap_keeps_samples_bit_identical():
    """SF_CAT_THREADS only changes width, never a sampled bitstring."""
    results = []
    for threads in (None, "1", "2"):
        env = dict(os.environ)
        env.pop("SF_CAT_THREADS", None)
        env.pop("SF_THREADS", None)
        if threads is not None:
            env["SF_CAT_THREADS"] = threads
        proc = subprocess.run(
            [sys.executable, "-c", _SAMPLE_SCRIPT],
            capture_output=True,
            text=True,
            env=env,
            timeout=600,
        )
        assert proc.returncode == 0, proc.stderr[-800:]
        results.append(json.loads(proc.stdout.strip().splitlines()[-1]))
    assert results[0] == results[1] == results[2]
