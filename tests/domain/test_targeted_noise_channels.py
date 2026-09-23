"""Targeted noise channels + T1/T2 thermal relaxation — engine ground truth.

Pins the additive noise-API extensions end to end on the native
density-matrix path; every expected value is an independent hand-built
numpy ground truth (never read back from the API under test):

  * ``add_*(..., qubits=...)`` — 1-qubit channels fire only on their wires
    (works on every build: the native core has always keyed 1q channels by
    qubit).
  * 2-qubit ``qubits=(a, b)`` — instruction-order pair matching, the
    Qiskit ``add_quantum_error`` semantics (a channel on ``(0, 1)`` is
    silent on a gate emitted as ``cx(1, 0)``); needs the extension argument
    carrying the pair list, so those tests skip on older builds.
  * ``add_thermal_relaxation(t1, t2, gate_time)`` — populations decay as
    ``exp(-t/T1)``, coherences as ``exp(-t/T2)``.

Layout note: ground truths are built little-endian (qubit q = bit q of the
basis index); the public ``metadata`` rho is the q0-first permutation,
converted with ``_to_little_endian``.
"""

import math

import numpy as np
import pytest

import superfermion as sf
from superfermion.circuit import Circuit
from superfermion.devices.noisy_dm_state import _to_little_endian

pytestmark = pytest.mark.domain

I2 = np.eye(2, dtype=np.complex128)
X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
P0 = np.array([[1, 0], [0, 0]], dtype=np.complex128)
P1 = np.array([[0, 0], [0, 1]], dtype=np.complex128)
_PAULI = (I2, X, Y, Z)


def _supports_targeted_2q() -> bool:
    """Capability probe: does the extension accept the 2q pair-list argument?"""
    try:
        Circuit(1).to_ir().simulate_dm_noisy_state_view([], [], None, [])
    except (AttributeError, TypeError):
        return False
    return True


TARGETED_2Q = _supports_targeted_2q()


def _full_le(op: np.ndarray, q: int, n: int) -> np.ndarray:
    """Embed a 1q operator on wire q of an LE-indexed operator (q = bit q)."""
    full = np.eye(1, dtype=np.complex128)
    for i in range(n - 1, -1, -1):  # descending: i = 0 is the rightmost (LSB) factor
        full = np.kron(full, op if i == q else I2)
    return full


def _apply_kraus_le(rho, kraus, q, n):
    acc = np.zeros_like(rho)
    for k in kraus:
        kf = _full_le(k, q, n)
        acc += kf @ rho @ kf.conj().T
    return acc


def _amplitude_damping(g):
    return [
        np.array([[1, 0], [0, np.sqrt(1 - g)]], dtype=np.complex128),
        np.array([[0, np.sqrt(g)], [0, 0]], dtype=np.complex128),
    ]


def _pauli_channel_2q(rho, p):
    """15-Pauli 2q depolarizing channel in the LE 2-qubit basis.

    The pair sum is symmetric under swapping the two wires' Pauli labels,
    so the kron order is immaterial for the sum.
    """
    acc = (1 - p) * rho
    for a in _PAULI:
        for b in _PAULI:
            if a is I2 and b is I2:
                continue
            full = np.kron(b, a)  # b on wire 1 (high factor), a on wire 0 (low)
            acc += (p / 15) * (full @ rho @ full.conj().T)
    return acc


def _cnot_control0_le() -> np.ndarray:
    """CNOT with control wire 0 (LSB) and target wire 1, LE basis.

    Verified against the engine: ``x(0).cnot(0, 1)`` yields '11' (the
    instruction's first qubit is the high bit of its local 4-dim block).
    """
    return np.kron(I2, P0) + np.kron(X, P1)


def _cnot_control1_le() -> np.ndarray:
    """CNOT with control wire 1 (high factor) and target wire 0, LE basis."""
    return np.kron(P0, I2) + np.kron(P1, X)


def _zero_state(n):
    rho = np.zeros((2**n, 2**n), dtype=np.complex128)
    rho[0, 0] = 1.0
    return rho


def _basis_le(index: int, n: int) -> np.ndarray:
    rho = np.zeros((2**n, 2**n), dtype=np.complex128)
    rho[index, index] = 1.0
    return rho


def _public(res) -> np.ndarray:
    return np.asarray(res.metadata["density_matrix"])


class TestTargetedSingleQubit:
    """1q channels fire only on their target wires."""

    def test_only_targeted_wire_is_noisy(self):
        g = 0.3
        qc = Circuit(2).x(0).x(1)
        nm = sf.NoiseModel().add_amplitude_damping(g, qubits=0)
        res = sf.run(qc, method="density_matrix", shots=0, noise_model=nm)
        # key convention: qubit q = bit q of int(key, 2); x(0).x(1) -> index 3
        assert res.probabilities.get("11", 0.0) == pytest.approx(1 - g, abs=1e-12)
        assert res.probabilities.get("10", 0.0) == pytest.approx(g, abs=1e-12)
        truth = _zero_state(2)
        for u in (_full_le(X, 0, 2), _full_le(X, 1, 2)):
            truth = u @ truth @ u.conj().T
        truth = _apply_kraus_le(truth, _amplitude_damping(g), 0, 2)
        assert np.max(np.abs(_public(res) - _to_little_endian(truth, 2))) < 1e-12

    def test_full_damping_leaves_the_companion_wire_intact(self):
        qc = Circuit(2).x(0).x(1)
        nm = sf.NoiseModel().add_amplitude_damping(1.0, qubits=0)
        res = sf.run(qc, method="density_matrix", shots=0, noise_model=nm)
        assert res.probabilities.get("10", 0.0) == pytest.approx(1.0, abs=1e-12)
        assert res.probabilities.get("11", 0.0) == pytest.approx(0.0, abs=1e-12)

    def test_targeted_matches_untargeted_on_the_only_wire(self):
        qc = Circuit(1).h(0).rz(0.37, 0)
        kw = dict(method="density_matrix", shots=0)
        a = sf.run(
            qc, **kw, noise_model=sf.NoiseModel().add_depolarizing(0.07, qubits=0)
        )
        b = sf.run(qc, **kw, noise_model=sf.NoiseModel().add_depolarizing(0.07))
        assert np.array_equal(_public(a), _public(b))

    def test_per_wire_channels_are_independent(self):
        qc = Circuit(2).x(0).x(1)
        nm = (
            sf.NoiseModel()
            .add_amplitude_damping(0.5, qubits=0)
            .add_amplitude_damping(1.0, qubits=1)
        )
        res = sf.run(qc, method="density_matrix", shots=0, noise_model=nm)
        # wire 1 fully decays and wire 0 keeps P(1) = 1 - 0.5
        assert res.probabilities.get("01", 0.0) == pytest.approx(0.5, abs=1e-12)


class TestTargetedTwoQubitPairs:
    """2q channels with an explicit ordered pair (Qiskit parity)."""

    pytestmark = pytest.mark.skipif(
        not TARGETED_2Q,
        reason="extension predates the per-pair 2q noise targets",
    )

    def test_matching_pair_fires_with_ground_truth(self):
        p = 0.2
        qc = Circuit(2).x(0).cnot(0, 1)
        nm = sf.NoiseModel().add_two_qubit_depolarizing(p, qubits=(0, 1))
        res = sf.run(qc, method="density_matrix", shots=0, noise_model=nm)
        ucx = _cnot_control0_le()
        truth = ucx @ _basis_le(1, 2) @ ucx.conj().T  # |q0=1, q1=0> -> Bell
        truth = _pauli_channel_2q(truth, p)
        assert np.max(np.abs(_public(res) - _to_little_endian(truth, 2))) < 1e-12

    def test_reversed_gate_is_silent(self):
        qc = Circuit(2).x(1).cnot(1, 0)  # emitted as cx(1, 0)
        nm = sf.NoiseModel().add_two_qubit_depolarizing(0.2, qubits=(0, 1))
        res = sf.run(qc, method="density_matrix", shots=0, noise_model=nm)
        clean = sf.run(qc, method="density_matrix", shots=0)
        assert np.array_equal(_public(res), _public(clean))

    def test_reversed_pair_fires_on_the_reversed_gate(self):
        p = 0.2
        qc = Circuit(2).x(1).cnot(1, 0)
        nm = sf.NoiseModel().add_two_qubit_depolarizing(p, qubits=(1, 0))
        res = sf.run(qc, method="density_matrix", shots=0, noise_model=nm)
        ucx = _cnot_control1_le()
        truth = ucx @ _basis_le(2, 2) @ ucx.conj().T  # |q0=0, q1=1> -> Bell
        truth = _pauli_channel_2q(truth, p)
        assert np.max(np.abs(_public(res) - _to_little_endian(truth, 2))) < 1e-12

    def test_untargeted_2q_channel_fires_on_both_orders(self):
        qc = Circuit(2).x(1).cnot(1, 0)
        targeted_away = sf.run(
            qc,
            method="density_matrix",
            shots=0,
            noise_model=sf.NoiseModel().add_two_qubit_depolarizing(0.2, qubits=(0, 1)),
        )
        untargeted = sf.run(
            qc,
            method="density_matrix",
            shots=0,
            noise_model=sf.NoiseModel().add_two_qubit_depolarizing(0.2),
        )
        assert targeted_away.metadata["purity"] == pytest.approx(1.0, abs=1e-12)
        assert untargeted.metadata["purity"] < 1.0 - 1e-6


class TestThermalRelaxationGroundTruth:
    """``add_thermal_relaxation`` against closed-form T1/T2 decay."""

    T1, T2, T = 40.0, 55.0, 12.0

    def _run(self, qc, noise_model):
        return sf.run(qc, method="density_matrix", shots=0, noise_model=noise_model)

    def test_populations_decay_as_exp_minus_t_over_t1(self):
        res = self._run(
            Circuit(1).x(0),
            sf.NoiseModel().add_thermal_relaxation(
                self.T1, self.T2, self.T, qubits=0
            ),
        )
        p1 = math.exp(-self.T / self.T1)
        assert res.probabilities.get("1", 0.0) == pytest.approx(p1, abs=1e-12)
        assert res.probabilities.get("0", 0.0) == pytest.approx(1 - p1, abs=1e-12)

    def test_coherences_decay_as_exp_minus_t_over_t2(self):
        res = self._run(
            Circuit(1).h(0),
            sf.NoiseModel().add_thermal_relaxation(
                self.T1, self.T2, self.T, qubits=0
            ),
        )
        rho = _public(res)
        # |+> under the zero-temperature bath: populations relax toward |0>
        # (p1 = exp(-t/T1)/2) while the coherence keeps exp(-t/T2)/2.
        p1 = 0.5 * math.exp(-self.T / self.T1)
        assert rho[0, 0].real == pytest.approx(1.0 - p1, abs=1e-12)
        assert rho[1, 1].real == pytest.approx(p1, abs=1e-12)
        assert rho[0, 1].real == pytest.approx(
            0.5 * math.exp(-self.T / self.T2), abs=1e-12
        )
        assert abs(rho[0, 1].imag) < 1e-15

    def test_hadamard_decay_laws_on_more_parameter_sets(self):
        """Two more (T1, T2, t) sets pin both exp decay laws."""
        for t1, t2, t in ((80.0, 160.0, 8.0), (80.0, 120.0, 20.0)):
            res = self._run(
                Circuit(1).h(0),
                sf.NoiseModel().add_thermal_relaxation(t1, t2, t, qubits=0),
            )
            rho = _public(res)
            assert rho[0, 1].real == pytest.approx(
                0.5 * math.exp(-t / t2), abs=1e-12
            ), (t1, t2, t)
            assert rho[1, 1].real == pytest.approx(
                0.5 * math.exp(-t / t1), abs=1e-12
            ), (t1, t2, t)

    def test_matches_explicit_damping_then_dephasing_recipe(self):
        """AD(g1) then PD(g_phi) — the analytic T1/T2 decomposition.

        The two runs must agree bit-for-bit: both routes build the same
        composed Kraus product in the same order (C2 ∘ C1).
        """
        g1 = 1.0 - math.exp(-self.T / self.T1)
        g_phi = 1.0 - math.exp(-2.0 * self.T / self.T2 + self.T / self.T1)
        qc = Circuit(1).x(0).h(0)
        thermal = self._run(
            qc,
            sf.NoiseModel().add_thermal_relaxation(
                self.T1, self.T2, self.T, qubits=0
            ),
        )
        recipe = self._run(
            qc,
            sf.NoiseModel()
            .add_amplitude_damping(g1, qubits=0)
            .add_phase_damping(g_phi, qubits=0),
        )
        assert np.array_equal(_public(thermal), _public(recipe))

    def test_per_wire_list_targets_each_qubit_once(self):
        res = self._run(
            Circuit(2).x(0).x(1),
            sf.NoiseModel().add_thermal_relaxation(
                self.T1, self.T2, self.T, qubits=[0, 1]
            ),
        )
        p = math.exp(-self.T / self.T1)
        assert res.probabilities.get("11", 0.0) == pytest.approx(p * p, abs=1e-12)
        assert res.probabilities.get("10", 0.0) == pytest.approx(
            p * (1 - p), abs=1e-12
        )
        assert res.probabilities.get("01", 0.0) == pytest.approx(
            (1 - p) * p, abs=1e-12
        )
        assert res.probabilities.get("00", 0.0) == pytest.approx(
            (1 - p) ** 2, abs=1e-12
        )
