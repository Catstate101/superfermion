"""Regression tests for the SUP-5…SUP-14 fix batch (superfermion 0.1.5).

Each class mirrors one per-fix verification script (see the repo-root
``_fix_verify_*.py`` files) and pins the corrected contract:

(a) MPS sampling TV vs exact on a random 10-qubit circuit      (SUP-13a)
(b) cross-backend counts / vector / probability order          (SUP-13b)
(c) grad vs finite-diff on asymmetric observables              (SUP-14)
(d) expval vs dense reference for ``from_string("Z0")``        (SUP-14)
(e) MethodError catchability as RuntimeError                   (SUP-7)
(f) probabilities populated on every backend                   (SUP-5/11)
(g) MPS short-Pauli padding == statevector                     (SUP-9)
(h) MPS shots=0 MemoryError guard                              (SUP-10)
(i) compiler U-convention fidelity suite                       (SUP-12)
(j) bound/parameterless DAG grad/qfim warning (SUP-6, not silent)
(k) missing param_values raise catchable MethodError (no Rust panic)
(l) compile() rejects unbound parameters with a clean ValueError (SUP-19)
(m) CU/CU3 native IR: matrix convention, sim, qfim (SUP-20)
(n) DAG JSON roundtrip preserves CP/CU gates (SUP-21)
(o) classical shadow + shadow expval entry points & accuracy (SUP-22)
(p) variance + mutual_info measurement types, Rust-backed (SUP-23)
(q) mid-circuit measure + c_if feed-forward + reset (SUP-24)

Conventions assumed (documented in guides/execution.mdx):
- numpy statevectors are little-endian: qubit q lives at bit q.
- bitstrings / probabilities keys are q0-last ("001" = q0 is |1>).
- observable labels are q0-leftmost ("ZI" = Z on qubit 0).
"""

import functools
import warnings

import numpy as np
import pytest

import superfermion as sf
from superfermion.circuit import Circuit
from superfermion.observables.core import _PAULI_ENCODE, SparsePauliOp
from superfermion.utils.exceptions import MethodError, SuperfermionError


pytestmark = pytest.mark.domain

compile_circuit = pytest.importorskip(
    "superfermion.compiler.manager",
    reason="compiler module unavailable",
).compile

EPS = 1e-6
SEED = 42


# ── shared helpers ────────────────────────────────────────────────────────────

def to_raw(op) -> list:
    """Convert a SparsePauliOp to the raw FFI term format."""
    return [
        ([_PAULI_ENCODE[ch] for ch in label], complex(c).real, complex(c).imag)
        for label, c in op._terms
    ]


_PAULI_MAT = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def dense_pauli(label: str) -> np.ndarray:
    """Dense 2^n × 2^n matrix for a q0-leftmost Pauli label.

    Little-endian: qubit 0 is the least significant bit, i.e. the last
    (rightmost) tensor factor, so the label is consumed right-to-left.
    """
    return functools.reduce(
        np.kron, [_PAULI_MAT[ch] for ch in reversed(label)]
    )


def dense_expval(sv: np.ndarray, label: str) -> float:
    """⟨ψ|O|ψ⟩ computed with an explicit dense matrix (independent reference)."""
    sv = np.asarray(sv, dtype=np.complex128).ravel()
    return float(np.real(np.vdot(sv, dense_pauli(label) @ sv)))


# ── (a) + (b) SUP-13: MPS sampling & ordering == statevector ─────────────────

class TestMpsSamplingAndOrdering:
    def test_mps_sampling_tv_vs_exact_10q(self):
        """(a) MPS sampling TV vs exact probabilities on a random 10-qubit circuit."""
        rng = np.random.default_rng(7)
        qc = Circuit(10)
        for _ in range(5):
            for i in range(10):
                qc.ry(float(rng.uniform(0, 3.1416)), i)
            for i in range(0, 9, 2):
                qc.cnot(i, i + 1)
            for i in range(1, 9, 2):
                qc.cnot(i, i + 1)
        p_sv = np.abs(np.asarray(sf.run(qc, method="statevector", shots=0).statevector)) ** 2
        n = 40000
        res = sf.run(qc, method="mps", shots=n, seed=SEED)
        tv = 0.5 * sum(abs(c / n - p_sv[int(bs, 2)]) for bs, c in res.counts.items())
        noise = np.sqrt(2**10 / n) / 2
        assert tv < 3 * noise + 0.01, f"TV={tv:.4f} noise~{noise:.4f}"

    def test_mps_vector_equals_statevector_directly(self):
        """(b) MPS to_statevector must equal the statevector backend (no bitrev)."""
        qc = Circuit(10)
        rng = np.random.default_rng(7)
        for _ in range(5):
            for i in range(10):
                qc.ry(float(rng.uniform(0, 3.1416)), i)
            for i in range(0, 9, 2):
                qc.cnot(i, i + 1)
            for i in range(1, 9, 2):
                qc.cnot(i, i + 1)
        p_sv = np.abs(np.asarray(sf.run(qc, method="statevector", shots=0).statevector)) ** 2
        p_mps = np.abs(np.asarray(sf.run(qc, method="mps", shots=0).statevector)) ** 2
        assert float(np.max(np.abs(p_mps - p_sv))) < 1e-12

    def test_mps_counts_q0_last(self):
        """(b) H(0) probe: MPS counts keys are q0-last ('001'), like statevector."""
        qc = Circuit(3).h(0)
        p_sv = np.abs(np.asarray(sf.run(qc, method="statevector", shots=0).statevector)) ** 2
        res = sf.run(qc, method="mps", shots=4000, seed=SEED)
        assert res.counts.get("001", 0) > 0, f"keys={sorted(res.counts)}"
        p_mps = np.abs(np.asarray(sf.run(qc, method="mps", shots=0).statevector)) ** 2
        assert float(np.max(np.abs(p_mps - p_sv))) < 1e-12
        sv_counts = sf.run(qc, method="statevector", shots=4000, seed=SEED).counts
        assert set(sv_counts) == set(res.counts)

    def test_sample_mps_api_q0_last(self):
        """(b) dag.sample_mps also emits q0-last bitstrings."""
        dag = Circuit(3).h(0).to_ir()
        d = dag.sample_mps(bond_dim=64, shots=4000, seed=SEED)
        assert d.get("001", 0) > 0, f"keys={sorted(d)}"

    def test_ghz_cross_backend_support(self):
        """(b) statevector / MPS / stabilizer all sample the same support."""
        qc = Circuit(3).h(0).cnot(0, 1).cnot(1, 2)  # GHZ3: {000, 111}
        for method in ("statevector", "mps", "stabilizer"):
            res = sf.run(qc, method=method, shots=4000, seed=SEED)
            keys = set(res.counts)
            assert keys == {"000", "111"}, f"{method}: {sorted(keys)}"


# ── (c) + (d) SUP-14: asymmetric observables — labels are q0-leftmost ────────

class TestAsymmetricObservables:
    _NAMES = ["t0", "t1", "t2", "t3"]
    _X = np.array([1.72131662, -0.38403809, 2.25313718, 1.24009990])

    @staticmethod
    def _energy(xx, names, circ, H):
        pv = dict(zip(names, xx))
        res = sf.run(circ.bind(pv), method="statevector", shots=1)
        return sf.expval(res.state.numpy(), H)

    def _fd_grad(self, xx, names, circ, H):
        return np.array([
            (self._energy(xx + EPS * e, names, circ, H)
             - self._energy(xx - EPS * e, names, circ, H)) / (2 * EPS)
            for e in np.eye(len(names))
        ])

    def _adjoint_grad(self, xx, names, circ, dag, raw_terms):
        pv = dict(zip(names, xx))
        res = sf.run(circ.bind(pv), method="statevector", shots=1)
        g = res.grad(raw_terms, dag=dag, param_values=pv)
        return np.array([g[n] for n in names])

    def _check_fd_vs_adjoint(self, H, circ, dag, names, xx):
        fd = self._fd_grad(xx, names, circ, H)
        ad = self._adjoint_grad(xx, names, circ, dag, to_raw(H))
        assert np.max(np.abs(fd - ad)) < 1e-6, f"max|fd-adj|={np.max(np.abs(fd - ad)):.3e}"

    def test_grad_matches_fd_asymmetric_iz(self):
        """(c) Single asymmetric term 0.39*IZ: adjoint == finite-diff."""
        p = [sf.param(n) for n in self._NAMES]
        circ = Circuit(2).ry(p[0], 0).ry(p[1], 1).cnot(0, 1).ry(p[2], 0).ry(p[3], 1)
        self._check_fd_vs_adjoint(
            SparsePauliOp.from_dict({"IZ": 0.39}), circ, circ.to_ir(),
            self._NAMES, self._X,
        )

    def test_grad_matches_fd_h2_like(self):
        """(c) Multi-term asymmetric H2-like observable: adjoint == finite-diff."""
        p = [sf.param(n) for n in self._NAMES]
        circ = Circuit(2).ry(p[0], 0).ry(p[1], 1).cnot(0, 1).ry(p[2], 0).ry(p[3], 1)
        H = SparsePauliOp.from_dict(
            {"II": -1.05, "IZ": 0.39, "ZI": -0.39, "ZZ": -0.01, "XX": 0.18}
        )
        self._check_fd_vs_adjoint(H, circ, circ.to_ir(), self._NAMES, self._X)

    def test_grad_matches_fd_3q_mixed(self):
        """(c) Independent 3-qubit mixed-term observable."""
        rng = np.random.default_rng(777)
        names = ["a0", "a1", "a2", "a3"]
        p = [sf.param(n) for n in names]
        qc3 = Circuit(3)
        qc3.ry(p[0], 0).rz(p[1], 1).ry(p[2], 2).cnot(0, 1).cnot(1, 2).ry(p[3], 0)
        x3 = rng.uniform(0, 3.1416, size=4)
        H3 = SparsePauliOp.from_dict({"XIZ": 0.7, "IZY": -0.4, "ZXZ": 0.25})
        self._check_fd_vs_adjoint(H3, qc3, qc3.to_ir(), names, x3)

    def test_expval_anchor_ry(self):
        """(d) Anchor: <Z0> = cos(theta), <Z1> = 1 for Ry(theta) on qubit 0."""
        theta = np.pi / 2
        qc = Circuit(2).ry(sf.param("t0"), 0)
        sv = np.asarray(sf.run(qc.bind({"t0": theta}), method="statevector", shots=0).statevector)
        assert abs(sf.expval(sv, SparsePauliOp.from_string("Z0", n_qubits=2)) - 0.0) < 1e-12
        assert abs(sf.expval(sv, SparsePauliOp.from_string("Z1", n_qubits=2)) - 1.0) < 1e-12

    def test_expval_from_string_matches_dense(self):
        """(d) from_string labels vs explicit dense-matrix reference (SUP-14)."""
        rng = np.random.default_rng(11)
        qc = Circuit(2)
        qc.ry(float(rng.uniform(0, np.pi)), 0).rz(float(rng.uniform(0, 2 * np.pi)), 1)
        qc.cnot(0, 1).rx(float(rng.uniform(0, np.pi)), 1)
        sv = np.asarray(sf.run(qc, method="statevector", shots=0).statevector)
        for label in ("Z0", "Z1", "X0", "X1", "Z0Z1", "Z1X0"):
            op = SparsePauliOp.from_string(label, n_qubits=2)
            v_ffi = sf.expval(sv, op)
            v_dense = dense_expval(sv, op._terms[0][0])
            assert abs(v_ffi - v_dense) < 1e-9, f"{label}: ffi={v_ffi:.12f} dense={v_dense:.12f}"

    def test_cross_path_expval_consistent(self):
        """(d) FFI expval == Rust state.expectation == counts estimate (q0-leftmost)."""
        p = [sf.param(n) for n in self._NAMES]
        circ = Circuit(2).ry(p[0], 0).ry(p[1], 1).cnot(0, 1).ry(p[2], 0).ry(p[3], 1)
        H = SparsePauliOp.from_dict(
            {"II": -1.05, "IZ": 0.39, "ZI": -0.39, "ZZ": -0.01, "XX": 0.18}
        )
        pv = dict(zip(self._NAMES, self._X))
        res = sf.run(circ.bind(pv), method="statevector", shots=20000, seed=SEED)
        v1 = sf.expval(res.state.numpy(), H)
        v2 = res.state.expectation(to_raw(H))
        v3 = res.expectation(to_raw(H))
        assert abs(v1 - v2) < 1e-9
        assert abs(v1 - v3) < 0.03, f"counts estimate off: {v1:.6f} vs {v3:.6f}"


# ── (e) SUP-7: MethodError is a RuntimeError subclass ────────────────────────

class TestMethodErrorMapping:
    def test_method_error_hierarchy(self):
        """(e) MethodError subclasses SuperfermionError and RuntimeError."""
        assert issubclass(MethodError, SuperfermionError)
        assert issubclass(MethodError, RuntimeError)

    def test_stabilizer_unsupported_methods_raise_method_error(self):
        """(e) Stabilizer state unsupported ops raise MethodError…"""
        st = sf.run(Circuit(3).h(0).cnot(0, 1), method="stabilizer", shots=0).state
        calls = [
            ("numpy()", lambda: st.numpy()),
            ("partial_trace()", lambda: st.partial_trace([0])),
            ("fidelity()", lambda: st.fidelity(st)),
            ("entropy()", lambda: st.entropy()),
            ("purity()", lambda: st.purity()),
            ("probabilities()", lambda: st.probabilities()),
        ]
        for name, call in calls:
            with pytest.raises(MethodError, match="not supported"):
                call()

    def test_stabilizer_errors_catchable_as_runtime_error(self):
        """(e) …and are still caught by ``except RuntimeError`` (back-compat)."""
        st = sf.run(Circuit(3).h(0).cnot(0, 1), method="stabilizer", shots=0).state
        try:
            st.numpy()
        except RuntimeError:
            pass
        else:
            pytest.fail("stabilizer numpy() must be catchable as RuntimeError")


# ── (f) SUP-5/11: probabilities populated on every backend ───────────────────

class TestProbabilitiesPopulated:
    _C3 = Circuit(3).h(0).cnot(0, 1).cnot(1, 2)

    def test_statevector_paths(self):
        """(f) statevector fast path (counts) and full path (exact |c|²)."""
        r = sf.run(self._C3, method="statevector", shots=1000, seed=7,
                   return_statevector=False)
        assert r.probabilities
        assert r.probabilities == {k: v / 1000 for k, v in r.counts.items()}
        assert abs(sum(r.probabilities.values()) - 1.0) < 1e-12

        r = sf.run(self._C3, method="statevector", shots=0, seed=7)
        sv = np.asarray(r.statevector, dtype=np.complex128)
        exact = {
            format(i, "03b"): float(abs(sv[i]) ** 2)
            for i in range(8) if abs(sv[i]) ** 2 > 1e-15
        }
        assert set(r.probabilities) == set(exact)
        assert all(abs(r.probabilities[k] - exact[k]) < 1e-15 for k in exact)
        assert abs(sum(r.probabilities.values()) - 1.0) < 1e-12

    def test_mps_paths(self):
        """(f) MPS shots>0 from counts; shots=0 (small) exact from state."""
        r = sf.run(self._C3, method="mps", shots=1000, seed=7)
        assert r.probabilities == {k: v / 1000 for k, v in r.counts.items()}
        r = sf.run(self._C3, method="mps", shots=0, seed=7)
        assert r.probabilities
        assert abs(sum(r.probabilities.values()) - 1.0) < 1e-12

    def test_density_matrix_paths(self):
        """(f) density_matrix: exact from rho (pure-state == statevector)."""
        r0 = sf.run(self._C3, method="density_matrix", shots=0, seed=7)
        r1 = sf.run(self._C3, method="density_matrix", shots=1000, seed=7)
        assert r0.probabilities and r1.probabilities
        assert abs(sum(r0.probabilities.values()) - 1.0) < 1e-12
        assert r0.metadata.get("probabilities") == r0.probabilities
        sv = np.asarray(sf.run(self._C3, method="statevector", shots=0).statevector)
        for k, p in r0.probabilities.items():
            assert abs(p - abs(sv[int(k, 2)]) ** 2) < 1e-8

    def test_stabilizer_path(self):
        """(f) stabilizer shots>0 from counts; shots=0 stays empty (no counts)."""
        r = sf.run(self._C3, method="stabilizer", shots=1000, seed=7)
        assert r.probabilities == {k: v / 1000 for k, v in r.counts.items()}
        r = sf.run(self._C3, method="stabilizer", shots=0, seed=7)
        assert r.probabilities == {}

    def test_get_probabilities_unchanged(self):
        """(f) get_probabilities() semantics preserved."""
        r = sf.run(self._C3, method="statevector", shots=5000, seed=7,
                   return_statevector=False)
        assert r.get_probabilities() == r.probabilities
        assert abs(float(np.sum(r.probabilities_array)) - 1.0) < 1e-9
        rp = sf.run(self._C3, method="statevector", shots=0, seed=7).get_probabilities()
        assert abs(sum(rp.values()) - 1.0) < 1e-12


# ── (g) SUP-9: MPS short-Pauli padding ───────────────────────────────────────

class TestMpsShortPauli:
    def test_short_pauli_equals_statevector(self):
        """(g) MPS expval('Z') and ('ZI') == statevector; full-length unchanged."""
        qc = Circuit(2).ry(0.7, 0).ry(1.1, 1)
        for label in ("Z", "ZI", "ZZ"):
            e_sv = sf.run(qc, method="statevector", shots=0).state.expectation(
                to_raw(SparsePauliOp.from_string(label, n_qubits=2)))
            e_mps = sf.run(qc, method="mps", shots=0).state.expectation(
                to_raw(SparsePauliOp.from_string(label, n_qubits=2)))
            assert abs(e_sv - e_mps) < 1e-12, f"{label}: sv={e_sv:.16f} mps={e_mps:.16f}"

    def test_short_pauli_anchor(self):
        """(g) |00> anchor: MPS expval('Z') == 1.0 (padding with identity)."""
        e0 = sf.run(Circuit(2), method="mps", shots=0).state.expectation(
            to_raw(SparsePauliOp.from_string("Z", n_qubits=2)))
        assert abs(e0 - 1.0) < 1e-12


# ── (h) SUP-10: MPS shots=0 MemoryError guard ────────────────────────────────

class TestMpsShotsZeroGuard:
    def test_large_circuit_raises_memory_error(self):
        """(h) n=50 MPS shots=0 raises MemoryError with guidance before densifying."""
        c50 = Circuit(50)
        for q in range(50):
            c50.h(q)
        with pytest.raises(MemoryError, match="shots>0"):
            sf.run(c50, method="mps", shots=0, seed=7)

    def test_small_circuit_still_densifies(self):
        """(h) n=16 MPS shots=0 still returns the exact statevector."""
        c16 = Circuit(16)
        for q in range(16):
            c16.h(q)
        r = sf.run(c16, method="mps", shots=0, seed=7)
        assert r.statevector is not None and len(r.statevector) == 2**16
        assert r.probabilities


# ── (i) SUP-12: compiler U-convention fidelity suite ─────────────────────────

class TestCompilerFidelity:
    _PATTERNS = {
        "lone CZ": Circuit(2).cz(0, 1),
        "CX-CZ-CX": Circuit(2).cx(0, 1).cz(0, 1).cx(0, 1),
        "CX-CX (control)": Circuit(2).cx(0, 1).cx(0, 1),
        "CX-CZ (control)": Circuit(2).cx(0, 1).cz(0, 1),
        "CZ-CX (control)": Circuit(2).cz(0, 1).cx(0, 1),
        "lone CX (control)": Circuit(2).cx(0, 1),
        "lone H (control)": Circuit(2).h(0),
    }
    _PREP_OK = {"U", "CX", "CNOT", "CZ", "RZ", "RY", "RX", "R1", "P", "X", "Y", "Z",
                "H", "S", "SDG", "T", "SX", "ID", "BARRIER"}

    @staticmethod
    def _unit(c):
        U = np.asarray(c.to_unitary())
        if U.ndim == 1:
            U = U.reshape(2 ** c.n_qubits, 2 ** c.n_qubits)
        return np.asarray(U, dtype=complex)

    @staticmethod
    def _hsfid(U, V):
        d = U.shape[0]
        return float(np.abs(np.vdot(U.flatten(), V.flatten())) ** 2 / d**2)

    def _sim_fid(self, c_ref, c_comp, prep_qubits):
        def with_prep(c):
            out = Circuit(c.n_qubits)
            for q in prep_qubits:
                out.x(q)
            for g in c.to_gate_list():
                name, qs = g["name"], g["qubits"]
                if name not in self._PREP_OK:
                    raise KeyError(f"cannot replay gate {name}")
                if name in ("BARRIER", "ID"):
                    continue
                method = name.lower()
                if name == "CNOT":
                    method = "cnot"
                elif name == "CX":
                    method = "cx"
                elif name == "SDG":
                    method = "s"
                getattr(out, method)(*g.get("params", []), *qs)
            return out

        sv_ref = np.asarray(sf.simulate(with_prep(c_ref), method="statevector").numpy())
        sv_cmp = np.asarray(sf.simulate(with_prep(c_comp), method="statevector").numpy())
        return float(np.abs(np.vdot(sv_ref, sv_cmp)) ** 2)

    def test_explicit_patterns_preserve_unitary(self):
        """(i) Lone-CZ / CX-CZ-CX / controls compile to the same unitary (L1, L2)."""
        for name, qc in self._PATTERNS.items():
            U_ref = self._unit(qc)
            for level in (1, 2):
                compiled = compile_circuit(qc, level=level)
                U_c = self._unit(compiled)
                md = float(np.max(np.abs(U_c - U_ref)))
                assert md < 1e-6, f"{name} L{level}: maxdiff={md:.2e}"
                for prep in ([], [0]):
                    fid = self._sim_fid(qc, compiled, prep)
                    assert abs(fid - 1.0) < 1e-9, (
                        f"{name} L{level} prep={prep}: simulator fid={fid:.10f}"
                    )

    def test_random_circuits_preserve_unitary(self):
        """(i) Random circuits compile to the same unitary up to global phase."""
        rng = np.random.default_rng(20260830)
        gate_pool = ["h", "x", "z", "rz", "rx", "ry", "u", "cz", "cx"]
        for i in range(6):
            n = int(rng.integers(2, 5))
            qc = Circuit(n)
            for _ in range(int(rng.integers(8, 20))):
                g = gate_pool[rng.integers(len(gate_pool))]
                if g == "u":
                    q = int(rng.integers(n))
                    qc.u(float(rng.uniform(0, 2 * np.pi)),
                         float(rng.uniform(0, 2 * np.pi)),
                         float(rng.uniform(0, 2 * np.pi)), q)
                elif g in ("cz", "cx"):
                    a, b = int(rng.integers(n)), int(rng.integers(n))
                    while b == a:
                        b = int(rng.integers(n))
                    (qc.cz if g == "cz" else qc.cx)(a, b)
                else:
                    q = int(rng.integers(n))
                    if g == "h":
                        qc.h(q)
                    elif g == "x":
                        qc.x(q)
                    elif g == "z":
                        qc.z(q)
                    else:
                        getattr(qc, g)(float(rng.uniform(0, 2 * np.pi)), q)
            U_ref = self._unit(qc)
            for level in (1, 2):
                compiled = compile_circuit(qc, level=level)
                U_c = self._unit(compiled)
                # KAK/magic-basis path is exact up to a global phase; remove it
                # (fidelity is the phase-insensitive metric).
                t = np.trace(U_ref.conj().T @ U_c)
                ph = np.exp(1j * np.angle(t)) if abs(t) > 0 else 1.0
                md_pc = float(np.max(np.abs(U_c - ph * U_ref)))
                fid = self._hsfid(U_ref, U_c)
                assert md_pc < 1e-6 and abs(fid - 1.0) < 1e-9, (
                    f"random[{i}] n={n} L{level}: phase-corrected maxdiff={md_pc:.2e} "
                    f"fid={fid:.10f}"
                )


# ── SUP-6: bound/parameterless DAG must not silently return {} ───────────────
class TestBoundDagWarning:
    """A DAG without symbolic parameters yields {} — now with a UserWarning."""

    _OBS = [([3, 3], 1.0, 0.0)]  # ZZ

    def _state(self, circuit):
        return sf.simulate(circuit, params={"t": 0.5})

    def test_bound_dag_grad_warns_and_returns_empty(self):
        qc = Circuit(2).ry(sf.param("t"), 0).cnot(0, 1)
        st = self._state(qc)
        bound_dag = qc.bind({"t": 0.5}).to_ir()
        with pytest.warns(UserWarning, match="symbolic"):
            g = st.grad(self._OBS, bound_dag, {"t": 0.5})
        assert g == {}

    def test_unbound_dag_grad_stays_silent_and_correct(self):
        qc = Circuit(2).ry(sf.param("t"), 0).cnot(0, 1)
        st = self._state(qc)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            g = st.grad(self._OBS, qc.to_ir(), {"t": 0.5})
        assert g == {"t": 0.0}  # <ZZ> identically 1.0 on this state

    def test_parameterless_circuit_with_values_warns(self):
        qc = Circuit(2).h(0).cnot(0, 1)
        st = sf.simulate(qc)
        with pytest.warns(UserWarning, match="no symbolic parameters"):
            g = st.grad(self._OBS, qc.to_ir(), {"t": 0.5})
        assert g == {}

    def test_empty_param_values_stays_silent(self):
        qc = Circuit(2).ry(sf.param("t"), 0).cnot(0, 1)
        st = self._state(qc)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            g = st.grad(self._OBS, qc.bind({"t": 0.5}).to_ir(), {})
        assert g == {}

    def test_bound_dag_qfim_warns(self):
        qc = Circuit(2).ry(sf.param("t"), 0).cnot(0, 1)
        st = self._state(qc)
        with pytest.warns(UserWarning, match="symbolic"):
            qfim = st.qfim(qc.bind({"t": 0.5}).to_ir(), {"t": 0.5})
        assert qfim.shape == (0, 0)

    def test_runresult_grad_warns_via_state(self):
        qc = Circuit(2).ry(sf.param("t"), 0).cnot(0, 1)
        res = sf.run(qc.bind({"t": 0.5}), shots=0)
        with pytest.warns(UserWarning, match="symbolic"):
            g = res.grad(self._OBS, qc.bind({"t": 0.5}).to_ir(), {"t": 0.5})
        assert g == {}


# ── SUP-6: missing param_values raise a catchable error, not a panic ──────────
class TestMissingParamValues:
    """Missing values raise MethodError instead of a pyo3 PanicException."""

    _OBS = [([3, 3], 1.0, 0.0)]  # ZZ

    def _state(self, circuit):
        return sf.simulate(circuit, params={"t": 0.5})

    def test_grad_empty_values_raises_method_error(self):
        qc = Circuit(2).ry(sf.param("t"), 0).cnot(0, 1)
        st = self._state(qc)
        with pytest.raises(MethodError, match="no value provided"):
            st.grad(self._OBS, qc.to_ir(), {})

    def test_grad_wrong_name_raises_method_error(self):
        qc = Circuit(2).ry(sf.param("t"), 0).cnot(0, 1)
        st = self._state(qc)
        with pytest.raises(MethodError, match="no value provided"):
            st.grad(self._OBS, qc.to_ir(), {"x": 1.0})

    def test_grad_partial_values_names_missing_parameter(self):
        qc = Circuit(1).ry(sf.param("t0"), 0).rz(sf.param("t1"), 0)
        st = sf.simulate(qc, params={"t0": 0.2, "t1": 0.3})
        with pytest.raises(MethodError, match="t1"):
            st.grad([([3], 1.0, 0.0)], qc.to_ir(), {"t0": 0.2})

    def test_missing_values_catchable_as_runtime_error(self):
        # Regression: used to raise an uncatchable pyo3 PanicException
        # ("Cannot evaluate unbound variable") from the Rust adjoint engine.
        qc = Circuit(2).ry(sf.param("t"), 0).cnot(0, 1)
        st = self._state(qc)
        assert issubclass(MethodError, RuntimeError)
        with pytest.raises(RuntimeError):
            st.grad(self._OBS, qc.to_ir(), {})

    def test_qfim_missing_values_raises_method_error(self):
        qc = Circuit(2).ry(sf.param("t"), 0).cnot(0, 1)
        st = self._state(qc)
        with pytest.raises(MethodError, match="not found in param_values"):
            st.qfim(qc.to_ir(), {})


# ── (l) SUP-19: compile() rejects unbound params (no Rust panic) ─────────────
class TestCompileUnboundParameters:
    """compile(level>0) on symbolic parameters used to panic inside the Rust
    parameter evaluator (PanicException at ops.rs). It must raise a catchable
    ValueError that names the offending parameters."""

    @staticmethod
    def _unbound_circuit():
        return (
            Circuit(2)
            .h(0)
            .ry(sf.param("t0"), 1)
            .rz(sf.param("t1"), 1)
            .cnot(0, 1)
        )

    def test_level1_unbound_raises_value_error(self):
        with pytest.raises(ValueError, match="requires bound parameter values"):
            compile_circuit(self._unbound_circuit(), level=1)

    def test_level2_unbound_raises_value_error(self):
        with pytest.raises(ValueError, match="requires bound parameter values"):
            compile_circuit(self._unbound_circuit(), level=2)

    def test_error_names_unbound_parameters(self):
        with pytest.raises(ValueError, match="t0.*t1"):
            compile_circuit(self._unbound_circuit(), level=1)

    def test_level0_unbound_is_passthrough(self):
        c = self._unbound_circuit()
        assert compile_circuit(c, level=0) is c

    def test_bound_circuit_still_compiles(self):
        c = self._unbound_circuit().bind({"t0": 0.3, "t1": 0.4})
        result = compile_circuit(c, level=1)
        assert isinstance(result, Circuit)
        assert result.gate_count >= 1


# ── (m) SUP-20: CU/CU3 native IR support ─────────────────────────────────────
class TestControlledU3Native:
    """cu/cu3 used to raise "Unknown gate: 'cu'" from to_ir(). They are now
    native OpType::Cu gates: control-first, diag(I, U3) matrix convention."""

    _P = (0.6, 0.9, 1.3)  # (theta, phi, lam)

    @staticmethod
    def _sv(circuit):
        sv = np.asarray(sf.simulate(circuit, method="statevector").numpy())
        return np.asarray(sv, dtype=np.complex128).ravel()

    @staticmethod
    def _exact_fs_metric(circuit, names, vals, eps=1e-6):
        """Exact FS metric via central-difference state derivatives (SUP-18)."""
        def sv(v):
            out = sf.run(circuit.bind(v), device="cpu", shots=0).statevector
            return np.asarray(out, dtype=np.complex128).ravel()

        psi0 = sv(vals)
        dpsi = np.zeros((len(names), psi0.size), dtype=np.complex128)
        for k, nm in enumerate(names):
            vp, vm = dict(vals), dict(vals)
            vp[nm] += eps
            vm[nm] -= eps
            dpsi[k] = (sv(vp) - sv(vm)) / (2 * eps)
        gram = dpsi @ dpsi.conj().T
        conn = psi0 @ dpsi.conj().T            # <dpsi_k|psi>
        corr = np.outer(conn, conn.conj())     # <dpsi_i|psi><psi|dpsi_j>
        return 4.0 * np.real(gram - corr)

    def test_cu3_aliases_cu(self):
        np.testing.assert_allclose(
            self._sv(Circuit(2).x(0).cu3(*self._P, 0, 1)),
            self._sv(Circuit(2).x(0).cu(*self._P, 0, 1)),
            atol=1e-10,
        )

    def test_cu_equals_u_when_control_is_one(self):
        # x(0) pins the control to |1>: cu must act exactly like an open u.
        np.testing.assert_allclose(
            self._sv(Circuit(2).x(0).cu(*self._P, 0, 1)),
            self._sv(Circuit(2).x(0).u(*self._P, 1)),
            atol=1e-10,
        )

    def test_cu_is_identity_when_control_is_zero(self):
        # Control q0 stays |0>: neither |00> nor x(1)|00> is touched.
        sv0 = self._sv(Circuit(2).cu(*self._P, 0, 1))
        np.testing.assert_allclose(sv0, [1.0, 0, 0, 0], atol=1e-10)
        sv1 = self._sv(Circuit(2).x(1).cu(*self._P, 0, 1))
        np.testing.assert_allclose(sv1, [0, 0, 1.0, 0], atol=1e-10)

    def test_cu_acts_as_u3_with_phases(self):
        # x(0) -> index 1 (q0=|1>, q1=|0>); U3|0> = (cos(θ/2), e^{iφ} sin(θ/2))
        # lands on the {q0=1} subspace: indices 1 and 3.
        theta, phi, _ = self._P
        sv = self._sv(Circuit(2).x(0).cu(*self._P, 0, 1))
        np.testing.assert_allclose(sv[1], np.cos(theta / 2), atol=1e-10)
        np.testing.assert_allclose(
            sv[3], np.exp(1j * phi) * np.sin(theta / 2), atol=1e-10
        )
        np.testing.assert_allclose(sv[[0, 2]], 0.0, atol=1e-10)

    def test_cu_to_ir_preserves_symbolic_parameters(self):
        c = Circuit(2).cu(sf.param("t0"), sf.param("t1"), sf.param("t2"), 0, 1)
        assert c.to_ir().parameter_names() == ["t0", "t1", "t2"]

    def test_qfim_through_cu_matches_exact_metric(self):
        # cu3 on an excited control (q0), entangled via CX with q2: the qfim
        # engine must see Cu parameters and match the exact FS metric.
        c = Circuit(3).x(0).cu3(sf.param("p0"), sf.param("p1"), sf.param("p2"), 0, 1)
        c.cx(1, 2)
        names = c.to_ir().parameter_names()
        assert names == ["p0", "p1", "p2"]
        vals = {"p0": 0.6, "p1": 0.9, "p2": 1.3}
        st = sf.run(c.bind(vals), device="cpu", shots=0).state
        g = np.array(st.qfim(c.to_ir(), vals))
        exact = self._exact_fs_metric(c, names, vals)
        np.testing.assert_allclose(g, exact, atol=1e-5, rtol=1e-5)


# ── (n) SUP-21: DAG JSON roundtrip keeps parameterized controlled gates ─────
class TestDagJsonRoundtrip:
    """QuantumDAG to_json/from_json used to silently drop CP gates
    (rebuild_op_type had no "CP" arm - same bug class as Cu before SUP-20).
    Both gates must survive a roundtrip unchanged."""

    @staticmethod
    def _roundtrip_sv(circuit):
        dag = circuit.to_ir()
        dag2 = type(dag).from_json(dag.to_json())
        return (
            np.asarray(dag.simulate(), dtype=np.complex128).ravel(),
            np.asarray(dag2.simulate(), dtype=np.complex128).ravel(),
        )

    def test_cp_survives_roundtrip(self):
        before, after = self._roundtrip_sv(
            Circuit(2).x(0).x(1).cp(0.6, 0, 1)
        )
        np.testing.assert_allclose(after, before, atol=1e-10)

    def test_cu_survives_roundtrip(self):
        before, after = self._roundtrip_sv(
            Circuit(2).x(0).cu3(0.6, 0.9, 1.3, 0, 1)
        )
        np.testing.assert_allclose(after, before, atol=1e-10)


# ── SUP-22: classical shadow + shadow expval ────────────────────────────────

class TestClassicalShadow:
    """(o) SUP-22: classical-shadow estimators (additive feature).

    Entry points sf.classical_shadow / sf.shadow_expval / sf.ClassicalShadow
    (and State.classical_shadow / State.shadow_expval) sample randomized
    single-qubit Pauli-basis snapshots from the exact statevector and
    estimate expectation values with the unbiased shadow estimator
    (median of k chunk means for k > 1).  Verified against exact expval;
    PennyLane cross-checks live in the repo-root _verify_shadow.py.
    """

    @staticmethod
    def _exact(circuit, label):
        sv = np.asarray(
            sf.run(circuit, device="cpu", method="statevector", shots=0)
            .statevector,
            dtype=np.complex128,
        ).ravel()
        op = SparsePauliOp.from_string(label, n_qubits=circuit.n_qubits)
        return float(np.real(op._fast_expval(sv)))

    def test_entry_points_present(self):
        for name in ("classical_shadow", "shadow_expval", "ClassicalShadow"):
            assert callable(getattr(sf, name, None)), name
        assert callable(getattr(sf.State, "classical_shadow", None))
        assert callable(getattr(sf.State, "shadow_expval", None))
        import superfermion.mitigation as sfm

        assert callable(sfm.classical_shadow)
        assert callable(sfm.shadow_expval)
        assert callable(sfm.ClassicalShadow)

    def test_single_qubit_matches_exact(self):
        circuit = Circuit(1).ry(0.3, 0)
        ex = self._exact(circuit, "Z")
        est = sf.shadow_expval(circuit, "Z", shots=12000, seed=3)
        assert abs(est - ex) < 0.05, f"est={est} exact={ex}"
        # X expectation of RY(0.3)|0> = sin(0.3)
        exx = self._exact(circuit, "X")
        estx = sf.shadow_expval(circuit, "X", shots=12000, seed=4)
        assert abs(estx - exx) < 0.05, f"est={estx} exact={exx}"

    def test_bell_correlators(self):
        bell = Circuit(2).h(0).cnot(0, 1)
        for label in ("ZZ", "XX", "YY"):
            ex = self._exact(bell, label)
            est = sf.shadow_expval(bell, label, shots=16000, seed=5)
            assert abs(est - ex) < 0.12, f"{label}: est={est} exact={ex}"

    def test_three_qubit_term(self):
        ghz = Circuit(3).h(0).cnot(0, 1).cnot(1, 2)
        ex = self._exact(ghz, "ZZZ")
        est = sf.shadow_expval(ghz, "ZZZ", shots=24000, seed=6)
        assert abs(est - ex) < 0.25, f"est={est} exact={ex}"

    def test_local_single_qubit_terms_match_exact_sup25(self):
        """(r) SUP-25: single-qubit terms on n=2 used to collapse to the
        wire's X/Y/Z mean because _rotate measured the reflected wire."""
        circuit = Circuit(2).ry(1.2, 0).rz(0.6, 0).cnot(0, 1).ry(0.4, 1).rz(0.9, 1)
        for label in ("Z0", "X0", "Z1", "X1"):
            ex = self._exact(circuit, label)
            est = sf.shadow_expval(circuit, label, shots=30000, seed=100)
            assert abs(est - ex) < 0.10, f"{label}: est={est} exact={ex}"

    def test_asymmetric_two_qubit_terms_match_exact_sup25(self):
        """(r) SUP-25: n=2 asymmetric pairs used to estimate the reflected
        operator (est of X0Z1 tracked exact Z0X1 instead of X0Z1)."""
        circuit = Circuit(2).ry(1.2, 0).rz(0.6, 0).cnot(0, 1).ry(0.4, 1).rz(0.9, 1)
        for label in ("X0Z1", "Z0X1", "Y0Z1", "Z0Y1"):
            ex = self._exact(circuit, label)
            est = sf.shadow_expval(circuit, label, shots=40000, seed=101)
            assert abs(est - ex) < 0.15, f"{label}: est={est} exact={ex}"

    def test_three_qubit_local_and_asymmetric_terms_sup25(self):
        """(r) SUP-25: on n=3 only the middle wire and reflection-closed
        Z0Z2 survived the reflected-wire rotation; local/asymmetric terms
        were biased."""
        circuit = (Circuit(3).ry(1.0, 0).cnot(0, 1).ry(0.7, 1).cnot(1, 2)
                   .rz(0.5, 0).rz(0.3, 2))
        for label, tol in (("Z0", 0.10), ("Z1", 0.10), ("Z2", 0.10),
                           ("Z0Z1", 0.15), ("X0Z1", 0.15), ("Z0Z2", 0.15)):
            ex = self._exact(circuit, label)
            est = sf.shadow_expval(circuit, label, shots=40000, seed=102)
            assert abs(est - ex) < tol, f"{label}: est={est} exact={ex}"

    def test_observable_forms_agree(self):
        circuit = Circuit(2).ry(0.6, 0).ry(1.2, 1).cnot(0, 1)
        shadow = sf.classical_shadow(circuit, shots=8000, seed=7)
        base = shadow.expval(([1, 1], 1.0, 0.0))
        assert abs(shadow.expval(SparsePauliOp.from_string("X0X1", n_qubits=2))
                   - base) < 1e-9
        assert abs(shadow.expval(sf.PauliString("XX")) - base) < 1e-9
        assert abs(shadow.expval(
            sf.Hamiltonian([sf.PauliString("XX", coeffs=1.0)])) - base) < 1e-9
        assert abs(shadow.expval("X0X1") - base) < 1e-9
        # identity term is exact; coefficients scale linearly
        assert abs(shadow.expval(([0, 0], 2.0, 0.0)) - 2.0) < 1e-9
        assert abs(shadow.expval(([1, 1], 3.0, 0.0)) - 3.0 * base) < 1e-9

    def test_seed_reproducibility(self):
        circuit = Circuit(2).ry(0.6, 0).ry(1.2, 1).cnot(0, 1)
        a = sf.classical_shadow(circuit, shots=400, seed=42)
        b = sf.classical_shadow(circuit, shots=400, seed=42)
        assert np.array_equal(a.bits, b.bits)
        assert np.array_equal(a.recipes, b.recipes)
        c_ = sf.classical_shadow(circuit, shots=400, seed=43)
        assert not (np.array_equal(a.bits, c_.bits)
                    and np.array_equal(a.recipes, c_.recipes))
        assert a.n_qubits == 2 and a.n_snapshots == 400
        assert set(np.unique(a.recipes)).issubset({1, 2, 3})

    def test_median_of_means(self):
        circuit = Circuit(2).ry(0.6, 0).ry(1.2, 1).cnot(0, 1)
        ex = self._exact(circuit, "ZZ")
        for k in (2, 8, 25):
            est = sf.shadow_expval(circuit, "ZZ", shots=20000, seed=8, k=k)
            assert abs(est - ex) < 0.12, f"k={k}: est={est} exact={ex}"

    def test_state_methods(self):
        circuit = Circuit(1).ry(0.3, 0)
        st = sf.simulate(circuit, method="statevector")
        ex = self._exact(circuit, "Z")
        shadow = st.classical_shadow(shots=12000, seed=9)
        assert shadow.n_snapshots == 12000
        assert abs(shadow.expval(([3], 1.0, 0.0)) - ex) < 0.05
        assert abs(st.shadow_expval(([3], 1.0, 0.0), shots=12000, seed=10)
                   - ex) < 0.05

    def test_validation(self):
        circuit = Circuit(1).h(0)
        with pytest.raises(ValueError):
            sf.classical_shadow(circuit, shots=0)
        with pytest.raises(ValueError):
            sf.ClassicalShadow([[0, 2]], [[1, 2]])   # bits must be 0/1
        with pytest.raises(ValueError):
            sf.ClassicalShadow([[0, 1]], [[1, 7]])   # bad basis code
        with pytest.raises(ValueError):
            sf.ClassicalShadow([[0, 1]], [[1]])      # shape mismatch
        shadow = sf.classical_shadow(circuit, shots=100, seed=1)
        with pytest.raises(ValueError):
            shadow.expval(([1, 1], 1.0, 0.0))      # term beyond n_qubits
        with pytest.raises(ValueError):
            shadow.expval("Z0", k=0)
        with pytest.raises(ValueError):
            shadow.expval("Z0", k=101)              # k > snapshots
        with pytest.raises(ValueError):
            sf.classical_shadow(np.ones(3), shots=10)  # not a 2**n state


class TestVarianceMutualInfo:
    """(p) variance + mutual_info measurement types (SUP-23)."""

    @staticmethod
    def _raw(label, coef=1.0, n=None):
        codes = [_PAULI_ENCODE[ch] for ch in label]
        if n is not None:
            codes = codes + [0] * (n - len(codes))
        return (codes, float(coef), 0.0)

    @staticmethod
    def _state(circuit):
        return sf.run(circuit, device="cpu", method="statevector",
                      shots=0).state

    def test_entry_points(self):
        assert callable(sf.variance)
        assert "variance" in sf.__all__
        assert hasattr(sf.State, "variance")
        assert hasattr(sf.State, "mutual_info")
        assert hasattr(sf.RunResult, "variance")
        assert hasattr(sf.RunResult, "mutual_info")

    def test_variance_bell_analytic(self):
        circuit = Circuit(2).h(0).cnot(0, 1)
        st = self._state(circuit)
        assert abs(st.variance([self._raw("ZI", n=2)]) - 1.0) < 1e-9
        assert abs(st.variance([self._raw("ZZ", n=2)])) < 1e-12
        assert abs(st.variance([self._raw("ZI", 0.5, 2),
                                self._raw("IX", 0.3, 2)]) - 0.34) < 1e-9
        assert abs(st.variance([self._raw("I", n=2)])) < 1e-12

    def test_variance_numpy_reference(self):
        # 3-qubit RY/CNOT; State.variance (Rust) vs pure-numpy O|psi>
        circuit = (Circuit(3).ry(0.7, 0).ry(0.4, 1).cnot(0, 1)
                   .cnot(1, 2).rx(0.9, 2))
        st = self._state(circuit)
        sv = np.asarray(st.numpy(), dtype=np.complex128)
        obs = [sf.PauliString("ZII", coeffs=0.5),
               sf.PauliString("IZI", coeffs=0.3),
               sf.PauliString("IIX", coeffs=0.2)]
        opsi = sum(t._apply(sv) for t in obs)
        mean = float(np.vdot(opsi, sv).real)
        mean_sq = float(np.vdot(opsi, opsi).real)
        expected = mean_sq - mean * mean
        raw = [self._raw("ZII", 0.5, 3), self._raw("IZI", 0.3, 3),
               self._raw("IIX", 0.2, 3)]
        assert abs(st.variance(raw) - expected) < 1e-8
        spo = SparsePauliOp(["ZII", "IZI", "IIX"], coeffs=[0.5, 0.3, 0.2])
        assert abs(sf.variance(sv, spo) - st.variance(raw)) < 1e-9

    def test_variance_density_matrix(self):
        circuit = Circuit(2).h(0).cnot(0, 1)
        st = self._state(circuit)
        rho0 = st.partial_trace([0])  # Bell reduced state = I/2
        assert rho0.method == "density_matrix"
        for label in ("Z", "X", "Y"):
            assert abs(rho0.variance([self._raw(label)]) - 1.0) < 1e-9
        assert abs(rho0.entropy() - np.log(2.0)) < 1e-9
        assert abs(rho0.purity() - 0.5) < 1e-9

    def test_variance_edges(self):
        st0 = self._state(Circuit(1))
        assert abs(st0.variance([self._raw("Z")])) < 1e-12
        stp = self._state(Circuit(1).h(0))
        assert abs(stp.variance([self._raw("Z")]) - 1.0) < 1e-9
        assert abs(stp.variance([self._raw("X")])) < 1e-12
        assert abs(stp.variance([self._raw("Z", 2.0)])
                   - 4.0 * stp.variance([self._raw("Z")])) < 1e-9

    def test_mutual_info_bell(self):
        circuit = Circuit(2).h(0).cnot(0, 1)
        st = self._state(circuit)
        mi = st.mutual_info([0], [1])
        assert abs(mi - 2.0 * np.log(2.0)) < 1e-9
        assert abs(st.mutual_info([1], [0]) - mi) < 1e-12
        comp = (st.partial_trace([0]).entropy()
                + st.partial_trace([1]).entropy()
                - st.partial_trace([0, 1]).entropy())
        assert abs(mi - comp) < 1e-9
        prod = self._state(Circuit(2).h(0))
        assert abs(prod.mutual_info([0], [1])) < 1e-12

    def test_mutual_info_ry3_pl_constants(self):
        # PennyLane 0.45 exact-state anchors (probe _probe_var_mi.py)
        circuit = (Circuit(3).ry(0.7, 0).ry(0.4, 1).cnot(0, 1)
                   .cnot(1, 2).rx(0.9, 2))
        st = self._state(circuit)
        assert abs(st.mutual_info([0, 1], [2]) - 0.83763164) < 1e-6
        assert abs(st.mutual_info([0], [1]) - 0.31962789) < 1e-6

    def test_validation(self):
        st = self._state(Circuit(2).h(0).cnot(0, 1))
        with pytest.raises(MethodError):
            st.mutual_info([0], [5])
        with pytest.raises(MethodError):
            st.mutual_info([0, 1], [1])
        res = sf.RunResult(counts={"00": 2})
        with pytest.raises(RuntimeError):
            res.mutual_info([0], [1])

    def test_runresult_paths(self):
        circuit = Circuit(2).h(0).cnot(0, 1)
        res = sf.run(circuit, device="cpu", method="statevector", shots=0)
        assert abs(res.variance([self._raw("ZI", n=2)]) - 1.0) < 1e-9
        assert abs(res.mutual_info([0], [1]) - 2.0 * np.log(2.0)) < 1e-9
        counts_res = sf.run(circuit, device="cpu", method="statevector",
                            shots=20000, seed=42)
        assert abs(counts_res.variance([self._raw("ZI", n=2)]) - 1.0) < 0.1


# ── (q) SUP-24: mid-circuit measure + c_if feed-forward + reset ──────────────

class TestMidCircuit:
    """(q) mid-circuit measure / classical feed-forward / reset (SUP-24).

    Measure + c_if used to be terminal-only: the Rust simulator dropped
    Measure ops, Reset was an identity, and no conditional-execution API
    existed. Dynamic circuits (reused wire after a measure, any reset,
    any c_if gate) now run per-shot trajectories (dag.simulate_dynamic)
    with finite shots on CPU statevector; purely terminal-measure
    circuits keep the exact fast path unchanged.

    SF count keys are q0-last (char j = qubit n-1-j), so P(q1=1) in a
    2-qubit circuit = (counts["10"] + counts["11"]) / shots.
    """

    @staticmethod
    def _p1(counts, shots):
        return (counts.get("10", 0) + counts.get("11", 0)) / shots

    def test_c_if_feedforward_correlation(self):
        """h then measure: X(1) iff m == 1 -> q1 = m, P(q1=1) = 0.5,
        keys 00/11 only (q1 is |0> until the conditional X fires)."""
        circuit = Circuit(2).h(0).measure(0, 0).c_if(0, 1).x(1)
        res = sf.run(circuit, shots=40000, seed=7)
        assert set(res.counts) <= {"00", "11"}
        assert sum(res.counts.values()) == 40000
        assert abs(self._p1(res.counts, 40000) - 0.5) < 0.01
        assert res.metadata["dynamic"] is True
        assert res.state is None

    def test_c_if_deterministic_flip(self):
        """m = 1 with certainty (X before measure) -> X(1) fires every shot."""
        circuit = Circuit(2).x(0).measure(0, 0).c_if(0, 1).x(1)
        res = sf.run(circuit, shots=5000, seed=1)
        assert self._p1(res.counts, 5000) == 1.0

    def test_c_if_value_zero_branch(self):
        """c_if(0, 0): X fires only when m == 0 -> q1 == not m, keys 01/10."""
        circuit = Circuit(2).h(0).measure(0, 0).c_if(0, 0).x(1)
        res = sf.run(circuit, shots=40000, seed=7)
        assert set(res.counts) <= {"01", "10"}
        assert abs(self._p1(res.counts, 40000) - 0.5) < 0.01

    def test_reset_mid_circuit(self):
        """Reset collapses to |0> (was identity): X;reset -> P(q0=1) = 0."""
        res = sf.run(Circuit(1).x(0).reset(0), shots=20000, seed=7)
        assert res.counts.get("1", 0) == 0
        # measure+reset then X -> |1> with certainty every shot.
        res2 = sf.run(Circuit(1).h(0).measure(0, 0).reset(0).x(0),
                      shots=20000, seed=7)
        assert res2.counts.get("1", 0) == 20000

    def test_measure_then_reuse_collapses(self):
        """Measure then reuse the wire: collapse (not identity) -> uniform."""
        res = sf.run(Circuit(1).h(0).measure(0, 0).h(0), shots=40000, seed=7)
        assert abs(res.counts.get("1", 0) / 40000 - 0.5) < 0.01

    def test_chained_feed_forward_ghz(self):
        """GHZ: X(1) iff m0 makes q1 = 0 always, so the m1 branch never
        fires and q2 = q0 -> P(q2=1) = 0.5 (char 0 of a 3-qubit key)."""
        circuit = (Circuit(3).h(0).cnot(0, 1).cnot(0, 2)
                   .measure(0, 0).c_if(0, 1).x(1)
                   .measure(1, 1).c_if(1, 1).x(2))
        res = sf.run(circuit, shots=40000, seed=7)
        p2 = sum(v for k, v in res.counts.items() if k[0] == "1") / 40000
        assert abs(p2 - 0.5) < 0.01

    def test_seed_determinism(self):
        """Same seed -> identical trajectory sample; different seed differs."""
        circuit = Circuit(2).h(0).measure(0, 0).c_if(0, 1).x(1)
        a = sf.run(circuit, shots=20000, seed=123)
        b = sf.run(circuit, shots=20000, seed=123)
        assert a.counts == b.counts
        c = sf.run(circuit, shots=20000, seed=124)
        assert a.counts != c.counts

    def test_surface_roundtrip_preserves_condition(self):
        """to_gate_list / to_qasm3 / bind() all preserve the c_if condition."""
        circuit = Circuit(2).h(0).measure(0, 0).c_if(0, 1).x(1)
        cond = [g for g in circuit.to_gate_list()
                if g.get("condition") == [0, 1]]
        assert len(cond) == 1 and cond[0]["name"] == "X"
        q3 = circuit.to_qasm3()
        assert "if (c[0] == 1) x q[1];" in q3
        assert "c[0] = measure q[0];" in q3
        bound = circuit.bind({})
        assert any(g.condition == (0, 1) for g in bound._gates)

    def test_requires_finite_shots_statevector_cpu(self):
        """Dynamic circuits need shots > 0, statevector, CPU (clean errors)."""
        circuit = Circuit(1).h(0).measure(0, 0).h(0)
        with pytest.raises(RuntimeError):
            sf.run(circuit, shots=0)
        with pytest.raises(RuntimeError):
            sf.run(circuit, shots=10, method="mps")
        with pytest.raises(RuntimeError):
            sf.run(circuit, shots=10, method="stabilizer")
        with pytest.raises(RuntimeError):
            sf.run(circuit, shots=10, method="density_matrix")

    def test_c_if_validation(self):
        """c_if guards: cbit range, value 0/1, no double condition, no
        pending condition on measure/unitary."""
        with pytest.raises(ValueError):
            Circuit(1).measure(0, 0).c_if(5, 1)
        with pytest.raises(ValueError):
            Circuit(1).c_if(0, 2)
        with pytest.raises(ValueError):
            Circuit(1).c_if(0, 1).c_if(0, 1)
        with pytest.raises(ValueError):
            Circuit(1).c_if(0, 1).measure(0, 0)
        with pytest.raises(ValueError):
            Circuit(1).c_if(0, 1).unitary(np.eye(2), [0])

    def test_terminal_measure_path_unchanged(self):
        """Non-regression: purely terminal measures keep the exact path."""
        res = sf.run(Circuit(2).h(0).cnot(0, 1).measure(0, 0).measure(1, 1),
                     shots=20000, seed=42)
        assert set(res.counts) == {"00", "11"}
        assert sum(res.counts.values()) == 20000
        assert res.metadata.get("dynamic") is not True
        assert res.state is not None
