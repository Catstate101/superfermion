"""Regression tests: noisy density-matrix state handle.

Baseline (pre-fix, recorded 2026-09-07): with method='density_matrix' and a
gate-noise model, ``RunResult.probabilities``/``counts``/``metadata`` were
correctly noisy, but ``RunResult.state`` (hence ``.expectation()``,
``.variance()``, ``state.purity()``, ...) silently reflected the noiseless
state because the handle is built via ``dag.simulate_to_state``, which has
no noise path.  The additive fix swaps in a handle backed by the correct
noisy ``metadata["density_matrix"]``: a native ``State`` built through the
``State.from_dm`` Rust binding when the rebuilt extension is present, and
``NoisyDensityMatrixState`` (pure-Python drop-in) otherwise.

Every expected value below is an independent ground truth (manual Kraus
application with numpy), not derived from the state API under test.
"""

import math

import numpy as np
import pytest
from superfermion._sf_core import State as _RustState

import superfermion as sf
from superfermion.circuit import Circuit
from superfermion.devices.noisy_dm_state import NoisyDensityMatrixState
from superfermion.noise import NoiseModel
from superfermion.utils.exceptions import MethodError

# Design B: the native ``State.from_dm`` binding exists only on extensions
# rebuilt with the additive Rust change; older wheels fall back to the proxy.
NATIVE_FROM_DM = hasattr(_RustState, "from_dm")

pytestmark = pytest.mark.domain

Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
I2 = np.eye(2, dtype=np.complex128)


def amp_damping_kraus(gamma):
    return [
        np.array([[1, 0], [0, math.sqrt(1 - gamma)]], dtype=np.complex128),
        np.array([[0, math.sqrt(gamma)], [0, 0]], dtype=np.complex128),
    ]


def apply_kraus_1q(rho, kraus, qubit, n):
    """Apply a Kraus set on qubit q of an n-qubit rho (little-endian)."""
    new_rho = np.zeros_like(rho)
    for k in kraus:
        kfull = 1.0
        for i in range(n):
            kfull = np.kron(kfull, k if i == qubit else I2)
        new_rho += kfull @ rho @ kfull.conj().T
    return new_rho


def depolarizing_kraus(p):
    return [
        math.sqrt(1 - p) * I2,
        math.sqrt(p / 3) * X,
        math.sqrt(p / 3) * Y,
        math.sqrt(p / 3) * Z,
    ]


def bitflip_kraus(p):
    return [math.sqrt(1 - p) * I2, math.sqrt(p) * X]


def purity(rho):
    return float(np.real(np.trace(rho @ rho)))


def expval_z1(rho):
    return float(np.real(np.trace(rho @ Z)))


def _noisy_rho_1q(gate_2x2, kraus_set):
    """Noisy rho for 1-qubit circuit: unitary gate then Kraus channel."""
    rho = gate_2x2 @ np.array([[1.0, 0], [0, 0]], dtype=np.complex128) @ gate_2x2.conj().T
    return apply_kraus_1q(rho, kraus_set, 0, 1)


OBS_Z = [([3], 1.0, 0.0)]


class TestReportedReproduction:
    """The exact reproduction from the bug report (|1>, amp damping 0.8)."""

    def test_expectation_purity_variance_match_noisy_truth(self):
        c = Circuit(1).x(0)
        nm = NoiseModel().add_amplitude_damping(0.8)
        res = sf.run(c, method="density_matrix", noise_model=nm, shots=0)
        rho_th = _noisy_rho_1q(X, amp_damping_kraus(0.8))  # diag(0.8, 0.2)
        assert res.probabilities == pytest.approx({"0": 0.8, "1": 0.2})
        assert res.metadata["purity"] == pytest.approx(0.68, abs=1e-9)
        assert expval_z1(rho_th) == pytest.approx(0.6)
        assert purity(rho_th) == pytest.approx(0.68)
        # the reported bug: these used to return -1.0 / 1.0 (noiseless)
        assert res.expectation(OBS_Z) == pytest.approx(0.6, abs=1e-9)
        assert res.state.purity() == pytest.approx(0.68, abs=1e-9)
        assert res.variance(OBS_Z) == pytest.approx(1 - 0.6**2, abs=1e-9)

    def test_entropy_matches_analytic(self):
        c = Circuit(1).x(0)
        nm = NoiseModel().add_amplitude_damping(0.8)
        res = sf.run(c, method="density_matrix", noise_model=nm, shots=0)
        s = -0.8 * math.log(0.8) - 0.2 * math.log(0.2)
        assert res.state.entropy() == pytest.approx(s, abs=1e-6)

    def test_state_handle_is_noisy_dm_handle(self):
        """Noisy handle is a density-matrix handle: native ``State`` when
        the rebuilt extension exposes ``State.from_dm``, proxy otherwise."""
        c = Circuit(1).x(0)
        nm = NoiseModel().add_amplitude_damping(0.8)
        res = sf.run(c, method="density_matrix", noise_model=nm, shots=0)
        if NATIVE_FROM_DM:
            assert type(res.state).__name__ == "State"
            assert not isinstance(res.state, NoisyDensityMatrixState)
        else:
            assert isinstance(res.state, NoisyDensityMatrixState)
        assert res.state.method == "density_matrix"
        assert res.state.device == "cpu"
        assert res.state.n_qubits == 1
        assert res.state.shape == [2, 2]

    def test_metadata_rho_and_state_agree(self):
        c = Circuit(1).x(0)
        nm = NoiseModel().add_amplitude_damping(0.8)
        res = sf.run(c, method="density_matrix", noise_model=nm, shots=0)
        # proxy numpy() is the little-endian rho; metadata rho is public order
        from superfermion.backends.density_matrix import _reverse_qubits_dm
        rho_le = np.asarray(res.state.numpy()).reshape(2, 2)
        assert np.allclose(_reverse_qubits_dm(rho_le, 1), res.metadata["density_matrix"])
        assert np.real(np.diag(rho_le)).sum() == pytest.approx(1.0, abs=1e-9)

    def test_counts_still_noisy(self):
        c = Circuit(1).x(0)
        nm = NoiseModel().add_amplitude_damping(0.8)
        res = sf.run(c, method="density_matrix", noise_model=nm, shots=20000, seed=7)
        total = sum(res.counts.values())
        ratio = res.counts.get("0", 0) / total
        assert 0.78 < ratio < 0.82


class TestOtherChannels:
    def test_depolarizing_full(self):
        c = Circuit(1).x(0)
        nm = NoiseModel().add_depolarizing(1.0)
        res = sf.run(c, method="density_matrix", noise_model=nm, shots=0)
        # channel: (X rho X + Y rho Y + Z rho Z)/3 -> diag(2/3, 1/3)
        assert res.expectation(OBS_Z) == pytest.approx(1 / 3, abs=1e-9)
        assert res.state.purity() == pytest.approx(5 / 9, abs=1e-9)
        assert res.probabilities == pytest.approx({"0": 2 / 3, "1": 1 / 3})

    def test_bitflip_full_flips_state(self):
        c = Circuit(1).x(0)
        nm = NoiseModel().add_bit_flip(1.0)
        res = sf.run(c, method="density_matrix", noise_model=nm, shots=0)
        assert res.probabilities == {"0": 1.0}
        assert res.expectation(OBS_Z) == pytest.approx(1.0, abs=1e-9)
        assert res.state.purity() == pytest.approx(1.0, abs=1e-9)

    def test_bitflip_half_maximally_mixed(self):
        c = Circuit(1).x(0)
        nm = NoiseModel().add_bit_flip(0.5)
        res = sf.run(c, method="density_matrix", noise_model=nm, shots=0)
        assert res.expectation(OBS_Z) == pytest.approx(0.0, abs=1e-9)
        assert res.state.purity() == pytest.approx(0.5, abs=1e-9)

    def test_rotated_state_off_diagonal_observables(self):
        """h(0) + depolarizing p=0.3: X/Y/Z expvals vs manual Kraus truth."""
        p = 0.3
        c = Circuit(1).h(0)
        nm = NoiseModel().add_depolarizing(p)
        res = sf.run(c, method="density_matrix", noise_model=nm, shots=0)
        h = np.array([[1, 1], [1, -1]], dtype=np.complex128) / math.sqrt(2)
        rho_th = _noisy_rho_1q(h, depolarizing_kraus(p))
        for mat, code, name in [(X, 1, "X"), (Y, 2, "Y"), (Z, 3, "Z")]:
            expected = float(np.real(np.trace(rho_th @ mat)))
            got = res.expectation([([code], 1.0, 0.0)])
            assert got == pytest.approx(expected, abs=1e-9), name

    def test_phase_discriminating_y(self):
        """rx(pi/2)|0> has <Y> = 1: guards the Y phase convention."""
        c = Circuit(1).rx(math.pi / 2, 0)
        nm = NoiseModel().add_amplitude_damping(0.4)
        res = sf.run(c, method="density_matrix", noise_model=nm, shots=0)
        rx = np.array([[math.cos(math.pi / 4), -1j * math.sin(math.pi / 4)],
                       [-1j * math.sin(math.pi / 4), math.cos(math.pi / 4)]],
                      dtype=np.complex128)
        rho_th = _noisy_rho_1q(rx, amp_damping_kraus(0.4))
        got_y = res.expectation([([2], 1.0, 0.0)])
        got_x = res.expectation([([1], 1.0, 0.0)])
        assert got_y == pytest.approx(float(np.real(np.trace(rho_th @ Y))), abs=1e-9)
        assert got_x == pytest.approx(float(np.real(np.trace(rho_th @ X))), abs=1e-9)


class TestTwoQubit:
    def test_two_qubit_damping(self):
        c = Circuit(2).x(0).x(1)
        nm = NoiseModel().add_amplitude_damping(0.5)
        res = sf.run(c, method="density_matrix", noise_model=nm, shots=0)
        # each qubit -> I/2: purity 0.25, all Z-type expvals 0
        assert res.state.purity() == pytest.approx(0.25, abs=1e-9)
        assert res.metadata["purity"] == pytest.approx(0.25, abs=1e-9)
        assert res.expectation([([3, 3], 1.0, 0.0)]) == pytest.approx(0.0, abs=1e-9)
        assert res.expectation([([3, 0], 1.0, 0.0)]) == pytest.approx(0.0, abs=1e-9)
        assert res.expectation([([0, 3], 1.0, 0.0)]) == pytest.approx(0.0, abs=1e-9)
        assert res.probabilities == pytest.approx(
            {"00": 0.25, "01": 0.25, "10": 0.25, "11": 0.25}
        )

    def test_partial_trace_and_mutual_info(self):
        c = Circuit(2).x(0).x(1)
        nm = NoiseModel().add_amplitude_damping(0.5)
        res = sf.run(c, method="density_matrix", noise_model=nm, shots=0)
        # each reduced qubit is I/2 (entropy ln2); product state: I = 0
        st = res.state
        assert st.partial_trace([0]).purity() == pytest.approx(0.5, abs=1e-9)
        assert st.partial_trace([1]).entropy() == pytest.approx(math.log(2), abs=1e-6)
        assert st.mutual_info([0], [1]) == pytest.approx(0.0, abs=1e-9)


class TestUnchangedPaths:
    """Noiseless and readout-only runs must keep the Rust handle."""

    def _rust_state_type_name(self):
        return "State"

    def test_noiseless_keeps_rust_handle(self):
        c = Circuit(1).x(0)
        res = sf.run(c, method="density_matrix", shots=0)
        assert type(res.state).__name__ == self._rust_state_type_name()
        assert not isinstance(res.state, NoisyDensityMatrixState)
        assert res.expectation(OBS_Z) == pytest.approx(-1.0, abs=1e-9)
        assert res.state.purity() == pytest.approx(1.0, abs=1e-9)

    def test_noiseless_metadata_purity_unchanged(self):
        c = Circuit(1).x(0)
        res = sf.run(c, method="density_matrix", shots=0)
        assert res.metadata["purity"] == pytest.approx(1.0, abs=1e-9)
        assert res.probabilities == {"1": 1.0}

    def test_readout_only_keeps_rust_handle(self):
        c = Circuit(1).x(0)
        nm = NoiseModel().add_readout_error(0.25)
        res = sf.run(c, method="density_matrix", noise_model=nm, shots=20000, seed=3)
        assert type(res.state).__name__ == self._rust_state_type_name()
        total = sum(res.counts.values())
        assert 0.22 < res.counts.get("0", 0) / total < 0.28


class TestHandleEquivalence:
    """Proxy built from a NOISELESS rho must match the Rust handle numbers."""

    def test_asymmetric_state_observables(self):
        c = Circuit(2).x(1)  # physical: q1=|1>, q0=|0>
        res = sf.run(c, method="density_matrix", shots=0)
        st = res.state
        proxy = NoisyDensityMatrixState.from_public_rho(res.metadata["density_matrix"], 2)
        for paulis in ([3, 0], [0, 3], [3, 3], [1, 0], [0, 1], [2, 0], [0, 2]):
            obs = [(paulis, 1.0, 0.0)]
            assert proxy.expectation(obs) == pytest.approx(st.expectation(obs), abs=1e-9)
            assert proxy.variance(obs) == pytest.approx(st.variance(obs), abs=1e-9)
        assert proxy.purity() == pytest.approx(st.purity(), abs=1e-9)
        assert proxy.entropy() == pytest.approx(st.entropy(), abs=1e-9)
        assert proxy.mutual_info([0], [1]) == pytest.approx(st.mutual_info([0], [1]), abs=1e-9)
        assert proxy.partial_trace([0]).purity() == pytest.approx(
            st.partial_trace([0]).purity(), abs=1e-9
        )

    def test_numpy_layout_matches_rust_handle(self):
        c = Circuit(2).x(1)
        res = sf.run(c, method="density_matrix", shots=0)
        proxy = NoisyDensityMatrixState.from_public_rho(res.metadata["density_matrix"], 2)
        assert np.allclose(np.asarray(proxy.numpy()), np.asarray(res.state.numpy()))

    def test_phase_discriminating_equivalence(self):
        # rx(pi/2)|0> = (|0> - i|1>)/sqrt(2) with the package RX convention
        # [[c, -i s], [-i s, c]] -> <X>=0, <Y>=-1, <Z>=0
        c = Circuit(1).rx(math.pi / 2, 0)
        res = sf.run(c, method="density_matrix", shots=0)
        proxy = NoisyDensityMatrixState.from_public_rho(res.metadata["density_matrix"], 1)
        for code, expected in [(1, 0.0), (2, -1.0), (3, 0.0)]:
            assert proxy.expectation([([code], 1.0, 0.0)]) == pytest.approx(
                res.state.expectation([([code], 1.0, 0.0)]), abs=1e-9
            )
            assert proxy.expectation([([code], 1.0, 0.0)]) == pytest.approx(
                expected, abs=1e-9
            )


class TestProxySurface:
    def test_sample_matches_distribution(self):
        c = Circuit(1).x(0)
        nm = NoiseModel().add_amplitude_damping(0.8)
        res = sf.run(c, method="density_matrix", noise_model=nm, shots=0)
        counts = res.state.sample(40000, seed=11)
        total = sum(counts.values())
        assert abs(counts.get("0", 0) / total - 0.8) < 0.02
        assert res.state.sample(40000, seed=11) == counts  # deterministic

    def test_probabilities_array(self):
        c = Circuit(1).x(0)
        nm = NoiseModel().add_amplitude_damping(0.8)
        res = sf.run(c, method="density_matrix", noise_model=nm, shots=0)
        probs = np.asarray(res.state.probabilities())
        assert probs.sum() == pytest.approx(1.0, abs=1e-9)
        # little-endian diagonal of the damped |1> state: index 0 = |0> (0.8)
        assert np.max(np.abs(probs - np.array([0.8, 0.2]))) < 1e-9

    def test_fidelity_is_trace_product(self):
        c = Circuit(1).x(0)
        nm = NoiseModel().add_amplitude_damping(0.8)
        res = sf.run(c, method="density_matrix", noise_model=nm, shots=0)
        st = res.state
        assert st.fidelity(st) == pytest.approx(0.68, abs=1e-9)  # |Tr(rho rho)| = purity

    def test_grad_qfim_raise_method_error(self):
        c = Circuit(1).x(0)
        nm = NoiseModel().add_amplitude_damping(0.8)
        res = sf.run(c, method="density_matrix", noise_model=nm, shots=0)
        st = res.state
        if isinstance(st, NoisyDensityMatrixState):
            with pytest.raises(MethodError):
                st.grad(OBS_Z, None, None)
            with pytest.raises(MethodError):
                st.qfim(None, None)
            with pytest.raises(MethodError):
                st.fidelity(object())
            return
        # Native handle: grad/qfim are unsupported on the density-matrix
        # method (Rust MethodError); non-state args fail pyo3 conversion.
        dag = c.to_ir()
        with pytest.raises(MethodError):
            st.grad(OBS_Z, dag, {})
        with pytest.raises(MethodError):
            st.qfim(dag, {})
        with pytest.raises(TypeError):
            st.fidelity(object())

    def test_sample_zero_shots(self):
        c = Circuit(1).x(0)
        nm = NoiseModel().add_amplitude_damping(0.8)
        res = sf.run(c, method="density_matrix", noise_model=nm, shots=0)
        assert res.state.sample(0) == {}
