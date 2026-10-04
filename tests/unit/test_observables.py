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
