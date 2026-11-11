"""Regression tests: single-buffer density-matrix handle view + noise placement.

Three additive fixes are pinned here; every expected value is an independent
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

Fix C — reset (capability).  ``RESET`` is a deterministic CPTP map, so
``method="density_matrix"`` now evolves it exactly instead of dispatching the
whole circuit to the per-shot trajectory engine (which rejects the DM method).
That is what lets a channel bound to ``"reset"`` — the default of
``NoiseModel.add_reset_error`` — fire at all; a reset used to be a hard error
on this path.
"""

import numpy as np
import pytest
from superfermion._sf_core import State as _RustState

import superfermion as sf
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


# --------------------------------------------------------------------------
# Fix C — RESET as an exact CPTP map on the density-matrix path
# --------------------------------------------------------------------------


def _reset_le(rho, q, n):
    """Independent ground truth: ρ → |0⟩⟨0|_q ⊗ Tr_q(ρ), LE basis.

    Pure numpy: reshape to a rank-2n tensor, ``np.trace`` over the ket/bra
    axes of qubit q (partial trace), then write the result back with that
    qubit fixed to 0 on both sides.  No Kraus set, no engine helper.
    """
    tensor = rho.reshape([2] * (2 * n))
    ket_axis = n - 1 - q  # LE: index = Σ b_q 2^q, so axis n-1 is the LSB
    bra_axis = 2 * n - 1 - q
    traced = np.trace(tensor, axis1=ket_axis, axis2=bra_axis)
    out = np.zeros([2] * (2 * n), dtype=np.complex128)
    where = [slice(None)] * (2 * n)
    where[ket_axis] = 0
    where[bra_axis] = 0
    out[tuple(where)] = traced.reshape([2] * (2 * n - 2))
    return out.reshape(2**n, 2**n)


def _reset_dm_supported():
    """Skip gate for extensions that predate the exact DM reset."""
    try:
        res = sf.run(Circuit(2).x(0).reset(0), method="density_matrix", shots=0)
    except Exception:  # noqa: BLE001 - any failure means "not supported yet"
        return False
    rho = np.asarray(res.metadata["density_matrix"])
    return abs(complex(rho[0, 0]) - 1.0) < 1e-12


RESET_ON_DM = _reset_dm_supported()


@pytest.mark.skipif(RESET_ON_DM is False, reason="extension predates the exact DM reset")
class TestResetOnDensityMatrixPath:
    """RESET on ``method="density_matrix"`` is the exact reset channel.

    Expected values are an independent numpy partial trace (``_reset_le``)
    plus two analytic anchors, so a wrong qubit, a wrong fold or a missing
    discard half all fail.
    """

    @staticmethod
    def _rho_le(circuit, model=None):
        n = circuit.n_qubits
        res = sf.run(circuit, method="density_matrix", shots=0, noise_model=model)
        assert res.metadata["method"] == "density_matrix"
        return _to_little_endian(np.asarray(res.metadata["density_matrix"]), n), res

    def test_reset_after_x_is_the_ground_state(self):
        rho, res = self._rho_le(Circuit(2).x(0).reset(0))
        assert np.max(np.abs(rho - _zero_state(2))) < 1e-12
        assert res.metadata["purity"] == pytest.approx(1.0, abs=1e-12)

    def test_reset_matches_an_independent_partial_trace(self):
        cases = [
            (Circuit(2).h(1).cnot(1, 0), 0),
            (Circuit(2).h(0).cnot(0, 1), 1),
            (Circuit(2).ry(0.7, 0).rx(-0.3, 1), 1),
            (Circuit(3).h(0).cnot(0, 1).ry(0.4, 2), 0),
            (Circuit(3).h(2).cnot(2, 1).rz(0.2, 0), 1),
            (Circuit(3).h(1).cnot(1, 0).cnot(0, 2), 2),
        ]
        for qc, q in cases:
            n = qc.n_qubits
            clean, _ = self._rho_le(qc)
            truth = _reset_le(clean, q, n)
            got, _ = self._rho_le(qc.reset(q))
            assert np.max(np.abs(got - truth)) < 1e-12, (n, q)

    def test_reset_of_an_entangled_pair_is_the_classical_mixture(self):
        rho, res = self._rho_le(Circuit(2).h(1).cnot(1, 0).reset(0))
        expected = np.zeros((4, 4), dtype=np.complex128)
        expected[0, 0] = expected[2, 2] = 0.5
        assert np.max(np.abs(rho - expected)) < 1e-12
        assert res.metadata["purity"] == pytest.approx(0.5, abs=1e-12)
        assert float(np.real(np.trace(rho))) == pytest.approx(1.0, abs=1e-12)

    def test_reset_on_every_qubit_returns_the_ground_state(self):
        rho, _ = self._rho_le(Circuit(3).h(0).cnot(0, 1).cnot(1, 2).reset(0).reset(1).reset(2))
        assert np.max(np.abs(rho - _zero_state(3))) < 1e-12

    def test_reset_keeps_rho_hermitian_and_trace_preserving(self):
        qc = Circuit(3).h(0).cnot(0, 1).ry(0.9, 1).cnot(1, 2).reset(1)
        rho, _ = self._rho_le(qc)
        assert np.max(np.abs(rho - rho.conj().T)) < 1e-12
        assert float(np.real(np.trace(rho))) == pytest.approx(1.0, abs=1e-12)
        assert min(float(v) for v in np.linalg.eigvalsh(rho)) > -1e-12

    def test_probabilities_follow_the_post_reset_diagonal(self):
        rho, res = self._rho_le(Circuit(2).x(0).cnot(0, 1).reset(1))
        for i in range(1 << 2):
            assert res.probabilities.get(format(i, "02b"), 0.0) == pytest.approx(
                float(np.real(rho[i, i])), abs=1e-12
            )

    def test_reset_error_binds_to_reset_and_fires(self):
        p1 = 0.4
        qc = Circuit(1).x(0).reset(0)
        rho, _ = self._rho_le(qc, sf.NoiseModel().add_reset_error(0.0, p1))
        # Reset leaves |0>; the error channel then excites it with prob p1.
        assert float(np.real(rho[1, 1])) == pytest.approx(p1, abs=1e-12)
        assert float(np.real(np.trace(rho))) == pytest.approx(1.0, abs=1e-12)

    def test_reset_error_zero_is_the_ideal_reset(self):
        qc = Circuit(2).h(0).cnot(0, 1).reset(0)
        ideal, _ = self._rho_le(qc)
        noisy, _ = self._rho_le(qc, sf.NoiseModel().add_reset_error(0.0, 0.0))
        assert np.max(np.abs(noisy - ideal)) < 1e-12

    def test_reset_error_does_not_fire_on_an_untargeted_qubit(self):
        qc = Circuit(2).h(0).cnot(0, 1).reset(0)
        ideal, _ = self._rho_le(qc)
        # Channel bound to qubit 1 while only qubit 0 is reset: never fires.
        other, _ = self._rho_le(qc, sf.NoiseModel().add_reset_error(0.3, 0.3, qubits=[1]))
        assert np.max(np.abs(other - ideal)) < 1e-12
        # Bound to qubit 0 it does fire, so the check above is not vacuous.
        targeted, _ = self._rho_le(qc, sf.NoiseModel().add_reset_error(0.3, 0.3, qubits=[0]))
        assert np.max(np.abs(targeted - ideal)) > 1e-3

    def test_reset_error_is_trace_preserving_and_positive(self):
        qc = Circuit(2).h(0).cnot(0, 1).reset(0).reset(1)
        rho, _ = self._rho_le(qc, sf.NoiseModel().add_reset_error(0.25, 0.15))
        assert np.max(np.abs(rho - rho.conj().T)) < 1e-12
        assert float(np.real(np.trace(rho))) == pytest.approx(1.0, abs=1e-12)
        assert min(float(v) for v in np.linalg.eigvalsh(rho)) > -1e-12

    def test_reset_still_samples_on_the_dm_path(self):
        qc = Circuit(2).x(0).cnot(0, 1).reset(1).measure_all()
        res = sf.run(qc, method="density_matrix", shots=300, seed=1)
        assert res.counts == {"01": 300}

    def test_pure_state_methods_keep_their_trajectory_replay(self):
        for method in ("statevector", "mps"):
            res = sf.run(Circuit(2).x(0).reset(0), method=method, shots=100, seed=7)
            assert res.counts == {"00": 100}


class TestFlatKrausValidation:
    """Malformed flat Kraus sets raise ValueError, never a Rust panic.

    The bindings take one 2x2 matrix per 8 floats (four complex re/im
    pairs) and one 4x4 per 32; a short final ``chunks`` slice used to
    reach an out-of-bounds index and surface as ``PanicException`` (a
    ``BaseException``), which bypassed ``except Exception``.
    """

    def test_short_1q_set_raises(self):
        dag = Circuit(1).x(0).to_ir()
        with pytest.raises(ValueError, match="positive multiple of 8"):
            dag.simulate_dm_noisy([(0, [1.0, 0.0, 0.0, 0.0])])

    def test_short_2q_set_raises(self):
        dag = Circuit(2).cnot(0, 1).to_ir()
        with pytest.raises(ValueError, match="positive multiple of 32"):
            dag.simulate_dm_noisy([], [[1.0] * 16])

    def test_empty_set_raises(self):
        dag = Circuit(1).x(0).to_ir()
        with pytest.raises(ValueError, match="positive multiple of 8"):
            dag.simulate_dm_noisy([(0, [])])

    def test_state_and_view_bindings_share_the_check(self):
        dag = Circuit(1).x(0).to_ir()
        with pytest.raises(ValueError, match="positive multiple of 8"):
            dag.simulate_dm_noisy_state([(0, [1.0, 0.0, 0.0, 0.0])])
        with pytest.raises(ValueError, match="positive multiple of 8"):
            dag.simulate_dm_noisy_state_view([(0, [1.0, 0.0, 0.0, 0.0])])

    def test_valid_identity_set_is_accepted(self):
        dag = Circuit(1).x(0).to_ir()
        vec = np.asarray(
            dag.simulate_dm_noisy([(0, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0])])
        ).reshape(2, 2)
        assert np.allclose(np.diag(vec).real, [0.0, 1.0], atol=1e-12)
