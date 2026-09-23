"""Regression tests: single-buffer density-matrix handle view + noise placement.

Two additive fixes are pinned here; every expected value is an independent
ground truth (hand-built Kraus/unitary numpy), never taken from the API
under test.

Fix B — single buffer (memory).  The native ``density_matrix`` path returns
``metadata["density_matrix"]`` as a read-only 2-D numpy view of the *state
handle's own* 4^n buffer (the public, qubit-0-first layout is permuted in
place in Rust), so the handle and the array no longer double the peak
allocation.  The two-buffer binding (``simulate_dm_noisy_state``) and the
flat-vector binding (``simulate_dm_noisy``) must stay bit-identical.

Fix A — placement (quality).  ``NoiseModel(placement="gate")`` applies
1-qubit channels only after 1-qubit gates (Qiskit/Aer placement) while the
default ``"touch"`` also applies them after every instruction touching the
qubit; ``convention="qiskit"`` rescales depolarizing lambdas so the channel
matches ``qiskit_aer`` at the same nominal number (unit-level checks live in
``tests/unit/test_noise.py``).

Layout note: all ground-truth matrices here are built in the engine's
little-endian basis (qubit q = bit q of the basis index); the public
``metadata`` rho is the bit-reversed (q0-first) permutation, converted with
``_to_little_endian`` (whose permutation is an involution).
"""

import numpy as np
import pytest

import superfermion as sf
from superfermion._sf_core import State as _RustState
from superfermion.circuit import Circuit
from superfermion.devices.noisy_dm_state import _to_little_endian

# The single-buffer view binding exists only on extensions rebuilt with the
# additive Rust change; older wheels keep the two-buffer binding (plus a
# writable rho copy), so the pinning tests below are skipped there.
VIEW_BINDING = hasattr(Circuit(1).to_ir(), "simulate_dm_noisy_state_view")

pytestmark = [
    pytest.mark.domain,
    pytest.mark.skipif(
        not VIEW_BINDING,
        reason="extension predates the single-buffer density-matrix view binding",
    ),
]

I2 = np.eye(2, dtype=np.complex128)
X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
H = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
_PAULI = (I2, X, Y, Z)


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


def _depolarizing_1q(p):
    return [
        np.sqrt(1 - p) * I2,
        np.sqrt(p / 3) * X,
        np.sqrt(p / 3) * Y,
        np.sqrt(p / 3) * Z,
    ]


def _pauli_channel_2q(rho, p):
    """15-Pauli 2q depolarizing channel in the LE 2-qubit basis."""
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
    p0 = np.array([[1, 0], [0, 0]], dtype=np.complex128)
    p1 = np.array([[0, 0], [0, 1]], dtype=np.complex128)
    return np.kron(I2, p0) + np.kron(X, p1)


def _expectation_le(rho, paulis, n):
    """Tr(rho * P) for per-wire Pauli codes (1=X, 2=Y, 3=Z), wire q = bit q."""
    full = np.eye(1, dtype=np.complex128)
    for i in range(n - 1, -1, -1):
        full = np.kron(full, _PAULI[paulis[i]])
    return float(np.real(np.trace(rho @ full)))


def _purity(rho):
    return float(np.real(np.trace(rho @ rho)))


def _zero_state(n):
    rho = np.zeros((2**n, 2**n), dtype=np.complex128)
    rho[0, 0] = 1.0
    return rho


def _noisy_ops(n, model):
    return model.to_rust_kraus_ops(n), model.to_rust_kraus_ops_2q()


class TestSingleBufferView:
    def test_noisy_rho_is_readonly_view_of_the_handle(self):
        qc = Circuit(2).h(0).cnot(0, 1)
        nm = sf.NoiseModel().add_depolarizing(0.05)
        res = sf.run(qc, method="density_matrix", shots=0, noise_model=nm)
        rho = res.metadata["density_matrix"]
        assert isinstance(rho, np.ndarray)
        assert rho.shape == (4, 4) and rho.dtype == np.complex128
        assert rho.flags.writeable is False
        assert rho.flags.owndata is False
        assert type(rho.base).__name__ == "State"
        assert type(res.state).__name__ == "State"

    def test_clean_rho_is_also_a_view(self):
        res = sf.run(Circuit(2).h(0).cnot(0, 1), method="density_matrix", shots=0)
        rho = res.metadata["density_matrix"]
        assert rho.flags.writeable is False
        assert type(rho.base).__name__ == "State"

    def test_write_is_rejected_and_handle_stays_intact(self):
        qc = Circuit(2).x(0)
        nm = sf.NoiseModel().add_amplitude_damping(0.3)
        res = sf.run(qc, method="density_matrix", shots=0, noise_model=nm)
        rho = res.metadata["density_matrix"]
        diag_before = np.array(np.real(np.diag(rho)))
        purity_before = res.state.purity()
        with pytest.raises(ValueError):
            rho[0, 0] = 5.0
        with pytest.raises(ValueError):
            rho.fill(0.0)
        assert np.array_equal(np.real(np.diag(rho)), diag_before)
        assert res.state.purity() == pytest.approx(purity_before, abs=1e-15)
        assert abs(float(np.real(np.trace(rho))) - 1.0) < 1e-12

    def test_view_bit_identical_to_two_buffer_binding(self):
        qc = Circuit(3).h(0).cnot(0, 1).ry(0.37, 2)
        nm = sf.NoiseModel().add_depolarizing(0.05).add_depolarizing(0.02, n_qubits=2)
        dag = qc.to_ir()
        ops, ops2 = _noisy_ops(3, nm)
        _, flat = dag.simulate_dm_noisy_state(ops, ops2)
        _, view = dag.simulate_dm_noisy_state_view(ops, ops2)
        public = np.asarray(view)
        assert np.array_equal(np.asarray(flat).reshape(8, 8), public)
        # the flat-vec binding carries the engine-layout (LE) matrix
        vec = np.asarray(dag.simulate_dm_noisy(ops, ops2))
        assert np.array_equal(vec, np.asarray(_to_little_endian(public, 3)).ravel())

    def test_handle_ops_parity_across_layouts(self):
        qc = Circuit(3).h(0).cnot(0, 1).rx(0.7, 2)
        nm = sf.NoiseModel().add_depolarizing(0.04)
        dag = qc.to_ir()
        ops, ops2 = _noisy_ops(3, nm)
        st_old, _ = dag.simulate_dm_noisy_state(ops, ops2)
        st_new, _ = dag.simulate_dm_noisy_state_view(ops, ops2)
        assert st_old.purity() == pytest.approx(st_new.purity(), abs=1e-15)
        assert st_old.entropy() == pytest.approx(st_new.entropy(), abs=1e-14)
        assert np.array_equal(np.asarray(st_old.numpy()), np.asarray(st_new.numpy()))
        assert np.array_equal(
            np.asarray(st_old.probabilities()), np.asarray(st_new.probabilities())
        )
        assert st_old.sample(200, 42) == st_new.sample(200, 42)
        assert np.array_equal(
            np.asarray(st_old.partial_trace([0]).numpy()),
            np.asarray(st_new.partial_trace([0]).numpy()),
        )
        for q in range(3):
            for code in (1, 2, 3):
                paulis = [0, 0, 0]
                paulis[q] = code
                obs = [(paulis, 1.0, 0.0)]
                assert st_old.expectation(obs) == st_new.expectation(obs)
                assert st_old.variance(obs) == pytest.approx(
                    st_new.variance(obs), abs=1e-15
                )

    def test_phase_and_wire_parity_on_clean_state(self):
        """A transpose/conjugate accessor slip flips imaginary-Pauli terms."""
        qc = Circuit(2).rx(np.pi / 2, 0).x(1)
        dag = qc.to_ir()
        st_old, _ = dag.simulate_dm_noisy_state([], [])
        st_new, _ = dag.simulate_dm_noisy_state_view([], [])
        # rx(pi/2)|0> = (|0> - i|1>)/sqrt(2) -> <Y> = -1 on wire 0
        assert st_new.expectation([([2, 0], 1.0, 0.0)]) == pytest.approx(-1.0, abs=1e-12)
        for paulis in ([2, 0], [0, 2], [1, 0], [0, 1], [3, 0], [0, 3], [1, 1]):
            obs = [(list(paulis), 1.0, 0.0)]
            assert st_old.expectation(obs) == pytest.approx(
                st_new.expectation(obs), abs=1e-15
            ), paulis

    def test_numpy_is_the_le_flatten_on_both_layouts(self):
        qc = Circuit(3).x(1).h(2)  # asymmetric: LE matrix != public matrix
        nm = sf.NoiseModel().add_depolarizing(0.03)
        dag = qc.to_ir()
        ops, ops2 = _noisy_ops(3, nm)
        st, view = dag.simulate_dm_noisy_state_view(ops, ops2)
        public = np.asarray(view)
        assert np.array_equal(
            np.asarray(st.numpy()), np.asarray(_to_little_endian(public, 3)).ravel()
        )
        assert not np.array_equal(
            np.asarray(st.numpy()).reshape(8, 8), public
        ), "test state must break the LE <-> public symmetry"

    def test_from_dm_round_trip_from_the_view(self):
        if not hasattr(_RustState, "from_dm"):
            pytest.skip("extension predates the State.from_dm binding")
        qc = Circuit(2).x(1)
        nm = sf.NoiseModel().add_amplitude_damping(0.3)
        dag = qc.to_ir()
        ops, _ = _noisy_ops(2, nm)
        st, view = dag.simulate_dm_noisy_state_view(ops, [])
        le = np.ascontiguousarray(
            _to_little_endian(np.asarray(view), 2).ravel(), dtype=np.complex128
        )
        st_rt = _RustState.from_dm(le, 2)
        assert np.array_equal(np.asarray(st_rt.numpy()), np.asarray(st.numpy()))

    def test_metadata_survives_a_dict_copy(self):
        qc = Circuit(2).h(0).cnot(0, 1)
        nm = sf.NoiseModel().add_depolarizing(0.05)
        res = sf.run(qc, method="density_matrix", shots=0, noise_model=nm)
        rho = res.metadata["density_matrix"]
        snap = dict(res.metadata)
        assert "density_matrix" in snap
        assert snap["density_matrix"] is rho
        assert not [k for k in res.metadata if k.startswith("_sf")]


class TestNoisePlacementGroundTruth:
    """placement="gate" vs "touch" against hand-applied Kraus truth (n=2)."""

    def test_ground_truth_cnot_matrix_convention(self):
        """Anchor for ``_cnot_control0_le``: x(0).cnot(0,1) must give '11'.

        The engine indexes an instruction's first qubit as the high bit of
        its local 4-dim block, so the *other* kron order is a CNOT with
        flipped control/target (and would silently shift every truth built
        on top of it).
        """
        res = sf.run(Circuit(2).x(0).cnot(0, 1), method="density_matrix", shots=0)
        e1 = np.zeros(4, dtype=np.complex128)
        e1[1] = 1.0  # |q0=1, q1=0>
        v = _cnot_control0_le() @ e1
        le = np.asarray(res.state.numpy()).reshape(4, 4)
        assert np.max(np.abs(np.outer(v, v.conj()) - le)) < 1e-12
        assert set(res.probabilities) == {"11"}

    def _bell_then_depolarizing(self, p):
        """H(0), CNOT(0,1) with a 1q depolarizing channel on every touch."""
        rho = _zero_state(2)
        uh = _full_le(H, 0, 2)
        ucx = _cnot_control0_le()
        dep = _depolarizing_1q(p)
        rho = _apply_kraus_le(uh @ rho @ uh.conj().T, dep, 0, 2)
        rho = ucx @ rho @ ucx.conj().T
        touch = _apply_kraus_le(_apply_kraus_le(rho, dep, 0, 2), dep, 1, 2)
        return rho, touch

    def test_gate_placement_skips_1q_channels_after_2q_gate(self):
        p = 0.2
        qc = Circuit(2).h(0).cnot(0, 1)
        res_gate = sf.run(
            qc,
            method="density_matrix",
            shots=0,
            noise_model=sf.NoiseModel(placement="gate").add_depolarizing(p),
        )
        res_touch = sf.run(
            qc,
            method="density_matrix",
            shots=0,
            noise_model=sf.NoiseModel().add_depolarizing(p),
        )
        truth_gate, truth_touch = self._bell_then_depolarizing(p)
        assert res_gate.metadata["purity"] == pytest.approx(_purity(truth_gate), abs=1e-12)
        assert res_touch.metadata["purity"] == pytest.approx(_purity(truth_touch), abs=1e-12)
        assert res_gate.metadata["purity"] > res_touch.metadata["purity"]
        for paulis in ([3, 0], [0, 3], [3, 3], [1, 0], [0, 1], [1, 1]):
            obs = [(list(paulis), 1.0, 0.0)]
            assert res_gate.expectation(obs) == pytest.approx(
                _expectation_le(truth_gate, paulis, 2), abs=1e-12
            ), ("gate", paulis)
            assert res_touch.expectation(obs) == pytest.approx(
                _expectation_le(truth_touch, paulis, 2), abs=1e-12
            ), ("touch", paulis)
        for i in range(4):
            assert res_gate.probabilities.get(format(i, "02b"), 0.0) == pytest.approx(
                float(np.real(truth_gate[i, i])), abs=1e-12
            )
            assert res_touch.probabilities.get(format(i, "02b"), 0.0) == pytest.approx(
                float(np.real(truth_touch[i, i])), abs=1e-12
            )

    def test_gate_placement_1q_only_circuit_is_unchanged(self):
        qc = Circuit(2).ry(0.4, 0).rz(0.3, 1)
        kw = dict(method="density_matrix", shots=0)
        p = 0.05
        r_gate = sf.run(qc, **kw, noise_model=sf.NoiseModel(placement="gate").add_depolarizing(p))
        r_touch = sf.run(qc, **kw, noise_model=sf.NoiseModel().add_depolarizing(p))
        assert r_gate.metadata["purity"] == pytest.approx(
            r_touch.metadata["purity"], abs=1e-15
        )
        assert r_gate.probabilities == pytest.approx(r_touch.probabilities, abs=1e-15)

    def test_gate_placement_still_applies_the_2q_channel(self):
        p = 0.21
        qc = Circuit(2).cnot(0, 1).x(0)  # asymmetric after the 2q gate
        ucx = _cnot_control0_le()
        ux = _full_le(X, 0, 2)
        rho = ucx @ _zero_state(2) @ ucx.conj().T
        rho = _pauli_channel_2q(rho, p)
        truth = ux @ rho @ ux.conj().T
        res = sf.run(
            qc,
            method="density_matrix",
            shots=0,
            noise_model=sf.NoiseModel(placement="gate").add_two_qubit_depolarizing(p),
        )
        public_truth = _to_little_endian(truth, 2)
        rho_meta = np.asarray(res.metadata["density_matrix"])
        assert np.max(np.abs(rho_meta - public_truth)) < 1e-12
        assert abs(float(np.real(np.trace(rho_meta))) - 1.0) < 1e-12
