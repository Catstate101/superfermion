"""Unit tests for observable construction and expectation values."""

import math

import numpy as np
import pytest

from superfermion.observables.core import (
    Hamiltonian,
    PauliString,
    SparsePauliOp,
    expval,
    mps_expval,
    mps_todense,
)


pytestmark = pytest.mark.unit


class TestPauliString:
    def test_construction_and_properties(self):
        ps = PauliString("XZ", coeff=0.5)
        assert ps.pauli_str == "XZ"
        assert ps.coeffs == 0.5
        assert "PauliString('XZ'" in repr(ps)

    def test_uppercases_pauli_string(self):
        ps = PauliString("xi")
        assert ps.pauli_str == "XI"


class TestSparsePauliOp:
    def test_from_dict(self):
        op = SparsePauliOp.from_dict({"ZZ": -1.0, "XX": 0.5})
        assert len(op._terms) == 2
        assert op._terms[0] == ("ZZ", -1.0 + 0j)

    def test_from_string_indexed(self):
        op = SparsePauliOp.from_string("Z0Z1")
        assert op._terms[0][0] == "ZZ"

    def test_list_constructor(self):
        op = SparsePauliOp(["Z", "X"], coeffs=[1.0, -0.5])
        assert len(op._terms) == 2


class TestHamiltonian:
    def test_construction_from_pauli_strings(self):
        terms = [PauliString("Z"), PauliString("X", coeff=0.5)]
        ham = Hamiltonian(terms)
        assert len(ham.terms) == 2
        assert ham.to_sparse_pauli_op()._terms[1][1] == 0.5


class TestExpval:
    def test_z_on_zero_state(self):
        sv = np.array([1.0, 0.0], dtype=np.complex128)
        assert expval(sv, PauliString("Z")) == pytest.approx(1.0)

    def test_x_on_plus_state(self):
        sv = np.array([1.0, 1.0], dtype=np.complex128) / math.sqrt(2)
        assert expval(sv, PauliString("X")) == pytest.approx(1.0)

    def test_bell_state_zz_expectation(self):
        sv = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.complex128) / math.sqrt(2)
        assert expval(sv, PauliString("ZZ")) == pytest.approx(1.0)

    def test_sparse_pauli_op_expectation(self):
        sv = np.array([1.0, 0.0], dtype=np.complex128)
        op = SparsePauliOp.from_dict({"Z": 2.0, "I": 0.5})
        assert expval(sv, op) == pytest.approx(2.5)

    def test_hamiltonian_expectation(self):
        sv = np.array([1.0, 0.0], dtype=np.complex128)
        ham = Hamiltonian([PauliString("Z"), PauliString("X", coeff=0.0)])
        assert expval(sv, ham) == pytest.approx(1.0)


def _dense_expval(state, pauli):
    """Independent ground truth: contract the state with a dense Pauli matrix.

    SuperFermion amplitude arrays are little-endian (qubit q is bit q of the
    index), while the observable strings are written qubit 0 first, so the
    string is contracted last-qubit-first here.
    """
    mats = {
        "I": np.eye(2, dtype=np.complex128),
        "X": np.array([[0, 1], [1, 0]], dtype=np.complex128),
        "Y": np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
        "Z": np.array([[1, 0], [0, -1]], dtype=np.complex128),
    }
    op = np.array([[1.0]], dtype=np.complex128)
    for ch in reversed(pauli):
        op = np.kron(op, mats[ch])
    return float(np.real(np.vdot(state, op @ state)))


def _su2_result(n, theta):
    """Run one SU(2) layer (RY, RZ, ring of CX) with fixed angles."""
    import superfermion as sf

    circ = sf.Circuit(n)
    for q in range(n):
        circ = circ.ry(sf.param(f"ry_{q}"), q)
        circ = circ.rz(sf.param(f"rz_{q}"), q)
    for q in range(n):
        circ = circ.cx(q, (q + 1) % n)
    values = {f"ry_{q}": theta[q] for q in range(n)}
    values.update({f"rz_{q}": theta[n + q] for q in range(n)})
    return sf.run(circ.bind(values), device="cpu", shots=0)


class TestRunResultExpectation:
    """RunResult.expectation(obs): the native path used in the benchmark tables."""

    THETA = (0.3, 1.1, -0.8, 0.7, -0.4, 2.0)

    @pytest.fixture(scope="class")
    def result(self):
        return _su2_result(3, self.THETA)

    def test_statevector_is_little_endian(self):
        """x(0) must land at index 1, x(2) at index 4 (qubit q is bit q)."""
        for q, idx in ((0, 1), (2, 4)):
            import superfermion as sf

            sv = np.asarray(
                sf.run(sf.Circuit(3).x(q), device="cpu", shots=0).statevector,
                dtype=np.complex128,
            ).ravel()
            assert int(np.argmax(np.abs(sv))) == idx

    def test_pauli_string_matches_dense(self, result):
        sv = np.asarray(result.statevector, dtype=np.complex128).ravel()
        assert result.expectation("ZII") == pytest.approx(
            _dense_expval(sv, "ZII"), abs=1e-12
        )
        assert result.expectation("IIZ") == pytest.approx(
            _dense_expval(sv, "IIZ"), abs=1e-12
        )

    def test_dict_matches_dense(self, result):
        sv = np.asarray(result.statevector, dtype=np.complex128).ravel()
        terms = {"ZII": 1.0, "IXI": -0.5, "IIZ": 0.25, "XXX": 0.75}
        expected = sum(c * _dense_expval(sv, p) for p, c in terms.items())
        assert result.expectation(terms) == pytest.approx(expected, abs=1e-12)

    def test_sparse_pauli_op_matches_dense(self, result):
        sv = np.asarray(result.statevector, dtype=np.complex128).ravel()
        terms = {"ZII": 1.0, "IIZ": 1.0, "XIX": 1.0}
        op = SparsePauliOp.from_dict(terms)
        expected = sum(c * _dense_expval(sv, p) for p, c in terms.items())
        assert result.expectation(op) == pytest.approx(expected, abs=1e-12)

    def test_matches_numpy_roundtrip(self, result):
        """The native handle and Observable.expectation(statevector) agree."""
        op = SparsePauliOp.from_dict({"ZII": 1.0, "IIZ": 1.0, "XIX": 0.3})
        native = result.expectation(op)
        roundtrip = op.expectation(result.statevector)
        assert native == pytest.approx(roundtrip, abs=1e-12)


class TestMpsExpval:
    def test_ghz_zz_exact(self):
        import superfermion as sf

        c = sf.Circuit(4)
        c.h(0)
        for i in range(3):
            c.cx(i, i + 1)
        assert mps_expval(c, "ZZII") == pytest.approx(1.0)
        assert mps_expval(c, {"ZZII": 1.0}) == pytest.approx(1.0)
        assert mps_expval(c, SparsePauliOp.from_string("ZZII")) == pytest.approx(1.0)
        assert mps_expval(c, "XIII") == pytest.approx(0.0, abs=1e-9)

    def test_matches_dense_ground_truth(self):
        import superfermion as sf

        rng = np.random.default_rng(7)
        n = 8
        c = sf.Circuit(n)
        for i in range(n):
            c.ry(float(rng.uniform(0, 2 * np.pi)), i)
        for i in range(0, n - 1, 2):
            c.cx(i, i + 1)
        sv = sf.simulate(c).numpy()
        probs = np.abs(sv) ** 2
        bits = np.arange(len(probs))
        want = float(np.sum(probs * (1 - 2 * ((bits & 1) ^ ((bits >> 1) & 1)))))
        assert mps_expval(c, "ZZ" + "I" * (n - 2), bond_dim=32) == pytest.approx(want, abs=1e-6)

    def test_param_binding(self):
        import superfermion as sf

        th = sf.param("th")
        c = sf.Circuit(2)
        c.rx(th, 0)
        c.cx(0, 1)
        assert mps_expval(c, "ZI", params={"th": 0.5}) == pytest.approx(
            math.cos(0.5)
        )

    def test_bad_length_raises(self):
        import superfermion as sf

        c = sf.Circuit(2)
        c.h(0)
        with pytest.raises(ValueError, match="n_qubits"):
            mps_expval(c, "ZZZ")


class TestMpsTodense:
    def test_asymmetric_state_layout(self):
        # HEA (asymmetric) catches bit-ordering bugs that GHZ-symmetric
        # states cannot (both i*2+s and i+s*2^p coincide on all-0/all-1).
        import superfermion as sf

        rng = np.random.default_rng(11)
        n = 6
        c = sf.Circuit(n)
        for i in range(n):
            c.ry(float(rng.uniform(0, 2 * np.pi)), i)
        for i in range(n - 1):
            c.cx(i, i + 1)
        sv = mps_todense(c, bond_dim=16)
        ref = sf.simulate(c).numpy()
        assert np.max(np.abs(sv - ref)) < 1e-9


class TestFfiErrorPropagation:
    """The FFI entry point must raise, never panic.

    A Rust panic crosses the boundary as ``pyo3_runtime.PanicException``, which
    derives from ``BaseException`` -- so it escapes ``except Exception`` in
    caller code and cannot be handled at all. These pin that each rejected
    input arrives as an ordinary, catchable Python exception.
    """

    def test_a_pauli_term_wider_than_the_state_raises(self):
        core = pytest.importorskip("superfermion._sf_core")
        sv = np.zeros(4, dtype=np.complex128)
        sv[0] = 1.0
        # 3-character term on a 2-qubit state: bit 2 indexes past the end.
        with pytest.raises(ValueError, match="only 2 qubits"):
            core.hamiltonian_expval(sv, [([0, 0, 1], 1.0, 0.0)])

    def test_an_over_long_term_does_not_shift_out_of_range(self):
        core = pytest.importorskip("superfermion._sf_core")
        sv = np.zeros(4, dtype=np.complex128)
        sv[0] = 1.0
        with pytest.raises(ValueError, match="only 2 qubits"):
            core.hamiltonian_expval(sv, [([0] * 69 + [1], 1.0, 0.0)])

    def test_a_code_outside_the_pauli_alphabet_raises(self):
        core = pytest.importorskip("superfermion._sf_core")
        sv = np.zeros(4, dtype=np.complex128)
        sv[0] = 1.0
        with pytest.raises(ValueError, match="invalid Pauli code 9"):
            core.hamiltonian_expval(sv, [([9], 1.0, 0.0)])

    def test_a_strided_view_raises_typeerror_and_is_catchable(self):
        core = pytest.importorskip("superfermion._sf_core")
        strided = np.zeros(8, dtype=np.complex128)[::2]
        with pytest.raises(TypeError, match="not contiguous"):
            core.hamiltonian_expval(strided, [([1], 1.0, 0.0)])

    def test_valid_input_is_unchanged(self):
        core = pytest.importorskip("superfermion._sf_core")
        sv = np.zeros(4, dtype=np.complex128)
        sv[0] = 1.0
        assert core.hamiltonian_expval(sv, [([0, 1], 1.0, 0.0)]) == 0.0
        # Identity padding wider than the state is legal: those bits are unused.
        assert core.hamiltonian_expval(sv, [([0, 0, 0, 0], 2.0, 0.0)]) == 2.0


class TestNumpyConstructorBounds:
    """Same FFI boundary, different symptom: a silently wrong object.

    Both constructors size-check the array against ``1 << n_qubits``. Written as
    a bare shift, ``n_qubits = 64`` masks the shift amount to 0, so the expected
    length becomes 1 and a one-amplitude array is accepted as a 64-qubit state --
    no exception, just a ``State`` whose ``n_qubits`` disagrees with its data.
    """

    def test_from_numpy_rejects_n_qubits_64(self):
        core = pytest.importorskip("superfermion._sf_core")
        with pytest.raises(ValueError, match="doesn't match n_qubits=64"):
            core.State.from_numpy(np.zeros(1, dtype=np.complex128), 64)

    def test_from_dm_rejects_n_qubits_64(self):
        core = pytest.importorskip("superfermion._sf_core")
        with pytest.raises(ValueError, match="does not fit in a usize"):
            core.State.from_dm(np.zeros(1, dtype=np.complex128), 64)

    def test_from_dm_rejects_a_squared_dimension_that_overflows(self):
        core = pytest.importorskip("superfermion._sf_core")
        # 2^32 amplitudes is a representable dim, but 4^32 is not.
        with pytest.raises(ValueError, match="4\\^32 does not fit"):
            core.State.from_dm(np.zeros(1, dtype=np.complex128), 32)

    def test_valid_sizes_still_round_trip(self):
        core = pytest.importorskip("superfermion._sf_core")
        sv = np.zeros(4, dtype=np.complex128)
        sv[0] = 1.0
        assert core.State.from_numpy(sv, 2).n_qubits == 2
        rho = np.zeros(16, dtype=np.complex128)
        rho[0] = 1.0
        assert core.State.from_dm(rho, 2).n_qubits == 2
        with pytest.raises(ValueError, match="doesn't match n_qubits=3"):
            core.State.from_numpy(sv, 3)
