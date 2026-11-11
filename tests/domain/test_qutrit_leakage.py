"""Qutrit / leakage channels -- 3-level reference and Qiskit Aer parity.

A transmon's second excited level does not fit on one wire, so the model
embeds the qutrit on a ``(data, flag)`` qubit pair::

    |0> -> |data=0, flag=0>
    |1> -> |data=1, flag=0>
    |2> -> |data=0, flag=1>

(the matrix is written in Qiskit wire order, first wire = least significant;
in the little-endian density matrix used below the level index *is* the
2-bit integer, so the embedding is the top-left 3x3 block).

Every assertion is pinned against an independent 3x3 model, and the Aer
tests build the same embedding by hand with ``kraus_error`` -- Aer has no
native leakage or qutrit support, which is exactly the manual work
``add_leakage`` / ``add_qutrit_error`` replace here.
"""

import numpy as np
import pytest

import superfermion as sf
from superfermion.devices.noisy_dm_state import _to_little_endian
from superfermion.noise import ErrorChannel, kraus_leakage

pytestmark = pytest.mark.domain

qiskit = pytest.importorskip("qiskit")
AerSimulator = pytest.importorskip("qiskit_aer").AerSimulator

from qiskit import QuantumCircuit  # noqa: E402
from qiskit.quantum_info import Kraus  # noqa: E402

PARITY = 1e-10

H2 = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)


# ── harness ──────────────────────────────────────────────────────────
def _sf_rho(circ, model, n):
    """SF density matrix in the little-endian (Qiskit) basis."""
    res = sf.run(circ, method="density_matrix", shots=0, noise_model=model)
    return _to_little_endian(np.asarray(res.metadata["density_matrix"]), n)


def _diff(a, b):
    return float(np.max(np.abs(a - b)))


def _embed3(rho3):
    """Pad a 3x3 qutrit state into the 4x4 (data, flag) little-endian space."""
    out = np.zeros((4, 4), dtype=np.complex128)
    out[:3, :3] = rho3
    return out


def _apply(rho, kraus):
    return sum(k @ rho @ k.conj().T for k in kraus)


def _rho3_after_h(p):
    """h |0> on the data wire, then one leakage kick of strength ``p``."""
    psi = H2 @ np.array([1, 0], dtype=np.complex128)
    rho3 = np.zeros((3, 3), dtype=np.complex128)
    rho3[:2, :2] = np.outer(psi, psi.conj())
    return _apply(rho3, kraus_leakage(p))


def _aer_kraus_rho(qc, kraus_ops, qubits=(0, 1)):
    """Run ``qc`` with a Kraus channel appended by hand.

    Aer 0.18 rejects a two-qubit ``QuantumError`` on a one-qubit instruction
    (``h``), so the channel goes in as a circuit instruction -- the manual
    construction this API replaces.
    """
    qc = qc.copy()
    qc.append(Kraus(kraus_ops), list(qubits))
    qc.save_density_matrix()
    sim = AerSimulator(method="density_matrix")
    return np.asarray(sim.run(qc).result().data()["density_matrix"])


def _qutrit_swap():
    """3-level unitary |1> <-> |2>, |0> untouched."""
    return np.array(
        [[1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=np.complex128
    )


# ── leakage channel ──────────────────────────────────────────────────
class TestLeakage:
    def test_matches_three_level_model(self):
        p = 0.2
        circ = sf.Circuit(2).h(0)
        model = sf.NoiseModel().add_leakage(p, qubit=0, flag=1)
        got = _sf_rho(circ, model, 2)
        ref = _embed3(_rho3_after_h(p))
        assert _diff(got, ref) < PARITY
        # the effect is real: p of the population left the subspace
        assert abs(float(np.real(got[2, 2])) - p) < 1e-12
        assert abs(float(np.real(got[0, 0] + got[1, 1])) - (1 - p)) < 1e-12

    def test_matches_aer_manual_embedding(self):
        p = 0.2
        qc = QuantumCircuit(2)
        qc.h(0)
        got = _sf_rho(
            sf.Circuit(2).h(0),
            sf.NoiseModel().add_leakage(p, qubit=0, flag=1),
            2,
        )
        embedded = ErrorChannel.from_qutrit_kraus(kraus_leakage(p))
        ref = _aer_kraus_rho(qc, embedded.kraus)
        assert _diff(got, ref) < PARITY
        # guard against a vacuous comparison with the clean state
        clean = _aer_kraus_rho(qc, [np.eye(4, dtype=np.complex128)])
        assert _diff(ref, clean) > 1e-3

    def test_zero_probability_is_noop(self):
        circ = sf.Circuit(2).h(0)
        noisy = _sf_rho(circ, sf.NoiseModel().add_leakage(0.0, 0, 1), 2)
        clean = _sf_rho(circ, None, 2)
        assert np.array_equal(noisy, clean)

    def test_measurement_does_not_fire(self):
        # A terminal measurement is not a drive: the default instruction
        # filter must leave the state untouched by a post-measurement kick.
        model = sf.NoiseModel().add_leakage(0.25, qubit=0, flag=1)
        with_measure = _sf_rho(sf.Circuit(2).h(0).measure(0), model, 2)
        without = _sf_rho(sf.Circuit(2).h(0), model, 2)
        assert _diff(with_measure, without) < PARITY

    def test_two_qubit_gate_fires_once(self):
        # cx(0, 2) drives the data wire once: the flag is idle, so the cycle
        # rule fires the channel exactly one extra time.  Reference built in
        # the little-endian basis (q0 = rightmost factor).
        p = 0.2
        model = sf.NoiseModel().add_leakage(p, qubit=0, flag=1)
        got = _sf_rho(sf.Circuit(3).h(0).cnot(0, 2), model, 3)

        ops4 = []
        for k in kraus_leakage(p):
            emb = np.zeros((4, 4), dtype=np.complex128)
            emb[:3, :3] = k
            ops4.append(emb)
        completion = np.zeros((4, 4), dtype=np.complex128)
        completion[0, 3] = 1.0
        ops4.append(completion)
        full = [np.kron(np.eye(2, dtype=np.complex128), k) for k in ops4]

        psi = np.zeros(8, dtype=np.complex128)
        psi[0] = 1.0
        psi = np.kron(np.eye(4, dtype=np.complex128), H2) @ psi  # h on q0
        rho = _apply(np.outer(psi, psi.conj()), full)  # kick after h(0)
        # cx(0 -> 2): q0 is the least significant bit, q2 the most
        p0 = np.diag([1, 0]).astype(np.complex128)
        p1 = np.diag([0, 1]).astype(np.complex128)
        x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        u = np.kron(np.eye(4, dtype=np.complex128), p0) + np.kron(
            np.kron(x, np.eye(2, dtype=np.complex128)), p1
        )
        rho = u @ rho @ u.conj().T
        ref = _apply(rho, full)  # kick after cx(0, 2)
        assert _diff(got, ref) < PARITY


# ── generic qutrit channels ──────────────────────────────────────────
class TestQutritError:
    def test_swap_channel_marks_the_flag(self):
        # x(0) puts the transmon in |1>; the 3-level swap moves it to |2>,
        # which the embedding stores as data=0, flag=1 (little-endian index 2).
        got = _sf_rho(
            sf.Circuit(2).x(0),
            sf.NoiseModel().add_qutrit_error(_qutrit_swap(), qubits=(0, 1)),
            2,
        )
        assert abs(float(np.real(got[2, 2])) - 1.0) < PARITY
        assert abs(float(np.real(got[1, 1]))) < PARITY  # data bit cleared

        qc = QuantumCircuit(2)
        qc.x(0)
        ref = _aer_kraus_rho(
            qc, ErrorChannel.from_qutrit_kraus(_qutrit_swap()).kraus
        )
        assert _diff(got, ref) < PARITY

    def test_validation(self):
        with pytest.raises(ValueError):
            ErrorChannel.from_qutrit_kraus([np.eye(2)])
        with pytest.raises(ValueError):
            ErrorChannel.from_qutrit_kraus([np.zeros((3, 3))])
        with pytest.raises(ValueError):
            sf.NoiseModel().add_leakage(-0.1, 0, 1)
        with pytest.raises(ValueError):
            sf.NoiseModel().add_leakage(1.5, 0, 1)
        with pytest.raises(ValueError):
            sf.NoiseModel().add_qutrit_error(_qutrit_swap(), qubits=None)
        with pytest.raises(ValueError):
            sf.NoiseModel().add_qutrit_error(_qutrit_swap(), qubits=(0, 0))

    def test_serialization_round_trips(self):
        model = sf.NoiseModel().add_leakage(0.15, qubit=0, flag=1)
        data = model.to_dict()
        assert data["two_qubit_channels"][-1]["mode"] == "spectator"
        rebuilt = sf.NoiseModel.from_dict(data)
        assert rebuilt.two_qubit_channels[-1].mode == "spectator"
        assert rebuilt.to_rust_kraus_2q_modes() == [1]
        circ = sf.Circuit(2).h(0)
        assert _diff(_sf_rho(circ, model, 2), _sf_rho(circ, rebuilt, 2)) < PARITY
