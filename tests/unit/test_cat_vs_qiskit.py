"""Every `sf.cat` lane against Qiskit, on the same circuit.

Qiskit is the external referee here, so no number in this file is produced by
Superfermion itself.  Skipped when Qiskit is absent (the interop CI lane installs
it); the assertions are exact comparisons, except the sampling one, which is a
total-variation bound at a fixed shot count.

Two conventions differ between the libraries and are asserted rather than
assumed:

* Observable strings: SF indexes from the left (``"ZI"`` = Z on qubit 0), Qiskit
  Pauli strings put the leftmost character on the *highest* qubit.  The test
  reverses one side.
* Amplitude arrays need no translation: both put qubit ``q`` in bit ``q``.
"""

import numpy as np
import pytest

import superfermion as sf

pytestmark = pytest.mark.unit

qiskit = pytest.importorskip("qiskit")
from qiskit import QuantumCircuit  # noqa: E402
from qiskit.circuit import Parameter  # noqa: E402
from qiskit.quantum_info import DensityMatrix, Pauli, Statevector  # noqa: E402

N = 4
DIM = 2**N


def _sf_circuit():
    """Non-Clifford, and free of global-phase ambiguity (no Z-phase gates)."""
    c = sf.Circuit(N)
    c.h(0)
    c.cx(0, 1)
    c.ry(0.37, 2)
    c.rx(-0.81, 1)
    c.t(3)
    c.cz(2, 3)
    c.s(1)
    c.cx(3, 0)
    c.rx(0.29, 3)
    return c


def _qiskit_circuit():
    qc = QuantumCircuit(N)
    qc.h(0)
    qc.cx(0, 1)
    qc.ry(0.37, 2)
    qc.rx(-0.81, 1)
    qc.t(3)
    qc.cz(2, 3)
    qc.s(1)
    qc.cx(3, 0)
    qc.rx(0.29, 3)
    return qc


def _rho(state):
    flat = np.asarray(state.numpy())
    return flat.reshape((DIM, DIM), order="C")


def test_statevector_matches_qiskit():
    got = np.asarray(sf.cat(_sf_circuit(), mode="statevector").simulate({}).numpy())
    want = np.asarray(Statevector(_qiskit_circuit()))
    np.testing.assert_allclose(got.ravel(), want, atol=1e-12)


def test_density_matrix_matches_qiskit():
    got = _rho(sf.cat(_sf_circuit(), mode="density_matrix").simulate({}))
    want = np.asarray(DensityMatrix(_qiskit_circuit()))
    np.testing.assert_allclose(got, want, atol=1e-12)


def test_density_matrix_matches_its_own_statevector():
    """A pure rho is |psi><psi|; a transposed read-out is |psi*><psi*|."""
    circuit = _sf_circuit()
    rho = _rho(sf.cat(circuit, mode="density_matrix").simulate({}))
    sv = np.asarray(sf.cat(circuit, mode="statevector").simulate({}).numpy()).ravel()
    np.testing.assert_allclose(rho, np.outer(sv, sv.conj()), atol=1e-12)


def test_pauli_string_is_indexed_from_the_left():
    """SF ``"ZI"`` is Z on qubit 0; Qiskit's ``"ZI"`` is Z on qubit 1."""
    circuit = _sf_circuit()
    got = float(np.real(sf.cat(circuit, mode="statevector").simulate({}).expectation("ZIII")))
    qk = Statevector(_qiskit_circuit())
    assert got == pytest.approx(float(np.real(qk.expectation_value(Pauli("IIIZ")))), abs=1e-12)
    assert got != pytest.approx(float(np.real(qk.expectation_value(Pauli("ZIII")))), abs=1e-6)


def test_adjoint_gradient_matches_finite_difference():
    t = sf.param("t")
    tq = Parameter("t")

    def sf_circuit():
        c = sf.Circuit(N)
        c.ry(t, 3)
        c.cx(3, 2)
        c.cx(2, 1)
        return c

    def qk_circuit():
        qc = QuantumCircuit(N)
        qc.ry(tq, 3)
        qc.cx(3, 2)
        qc.cx(2, 1)
        return qc

    theta = 0.63

    def expect(value):
        qc = qk_circuit().assign_parameters({tq: value})
        return float(np.real(Statevector(qc).expectation_value(Pauli("IIZZ"))))

    h = 1e-6
    want = (expect(theta + h) - expect(theta - h)) / (2 * h)
    got = sf.cat(sf_circuit(), mode="adjoint").run_gradient("ZZII", [theta])[0]
    assert got == pytest.approx(want, abs=1e-8)
    assert got == pytest.approx(-np.sin(theta), abs=1e-9)


def test_clifford_sampling_matches_exact_distribution():
    def sf_circuit():
        c = sf.Circuit(N)
        c.h(0)
        c.s(1)
        c.cx(0, 2)
        c.cz(1, 3)
        c.x(2)
        c.h(3)
        c.swap(0, 1)
        return c

    def qk_circuit():
        qc = QuantumCircuit(N)
        qc.h(0)
        qc.s(1)
        qc.cx(0, 2)
        qc.cz(1, 3)
        qc.x(2)
        qc.h(3)
        qc.swap(0, 1)
        return qc

    shots, seed = 20000, 11
    counts = sf.cat(sf_circuit(), mode="clifford").sample(shots, seed=seed)
    assert sum(counts.values()) == shots

    sampled = np.zeros(DIM)
    for key, count in counts.items():
        # little-endian keys: leftmost character is the highest qubit, which is
        # exactly how a bitstring becomes an array index
        index = int(key, 2)
        sampled[index] = count

    exact = np.abs(np.asarray(Statevector(qk_circuit()))) ** 2
    tv = 0.5 * np.abs(sampled / shots - exact).sum()
    assert tv < 0.02, f"total variation {tv:.4f} vs exact distribution"
