"""Comprehensive gradient method domain tests."""

import math
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

import superfermion as sf  # noqa: E402
from superfermion.observables.core import SparsePauliOp  # noqa: E402


pytestmark = pytest.mark.domain

parameter_shift = pytest.importorskip(
    "superfermion.qml.gradient.parameter_shift",
    reason="parameter-shift gradient module unavailable",
)
adjoint = pytest.importorskip(
    "superfermion.qml.gradient.adjoint",
    reason="adjoint gradient module unavailable",
)
spsa = pytest.importorskip(
    "superfermion.qml.gradient.spsa",
    reason="spsa gradient module unavailable",
)
qng = pytest.importorskip(
    "superfermion.qml.gradient.qng",
    reason="qng gradient module unavailable",
)


class TestFiniteDiffVsAdjoint:
    def test_finite_diff_agrees_with_adjoint_on_ry(self):
        theta = sf.param("theta")
        circuit = sf.Circuit(1).ry(theta, 0)
        observable = SparsePauliOp.from_dict({"Z": 1.0})
        params = {"theta": 0.5}
        names = ["theta"]
        values = np.array([0.5])

        fd_grad = parameter_shift.finite_diff_grad(
            circuit, observable, params, device="cpu", eps=1e-4
        )
        adj_grad = adjoint.adjoint_grad_vector(circuit, observable, names, values)

        assert abs(fd_grad["theta"] - adj_grad[0]) < 1e-3

    def test_finite_diff_agrees_with_adjoint_two_params(self):
        theta = sf.param("theta")
        phi = sf.param("phi")
        circuit = sf.Circuit(2).ry(theta, 0).ry(phi, 1).cnot(0, 1)
        observable = SparsePauliOp.from_dict({"ZZ": 1.0})
        names = ["theta", "phi"]
        values = np.array([0.3, 0.7])
        params = dict(zip(names, values))

        fd_grad = parameter_shift.finite_diff_grad(
            circuit, observable, params, device="cpu", eps=1e-4
        )
        adj_grad = adjoint.adjoint_grad_vector(circuit, observable, names, values)

        for name, adj_val in zip(names, adj_grad):
            assert abs(fd_grad[name] - adj_val) < 1e-2


class TestSU2LayerGrid:
    """Adjoint vs parameter-shift vs finite difference on SU(2) grids.

    Two cases above check a bare RY and a two-qubit pair. These sweep qubit
    count and depth together: RY + RZ on every qubit per layer, ring CX
    between layers, so ``2 * n * layers`` parameters, and a ring TFIM
    observable. Adjoint must equal the exact shift rule at every point and
    track central differences to their truncation error.
    """

    GRID = [(2, 1), (2, 2), (3, 1), (3, 2), (4, 1), (4, 2), (4, 4),
            (6, 1), (6, 2), (8, 1), (8, 2), (10, 1), (10, 2), (12, 1)]

    @staticmethod
    def _circuit(n, layers):
        c = sf.Circuit(n)
        names = []
        k = 0
        for layer in range(layers):
            for q in range(n):
                c = c.ry(sf.param(f"ry_{k}"), q)
                names.append(f"ry_{k}")
                k += 1
                c = c.rz(sf.param(f"rz_{k}"), q)
                names.append(f"rz_{k}")
                k += 1
            if layer < layers - 1:
                for q in range(n):
                    c = c.cx(q, (q + 1) % n)
        return c, names

    @staticmethod
    def _observable(n):
        terms = {}
        for i in range(n):
            z = ["I"] * n
            z[i] = "Z"
            terms["".join(z)] = 1.0
            x = ["I"] * n
            x[i] = "X"
            x[(i + 1) % n] = "X"
            terms["".join(x)] = 1.0
        return SparsePauliOp.from_dict(terms)

    @pytest.mark.parametrize("n,layers", GRID)
    def test_adjoint_matches_parameter_shift(self, n, layers):
        circuit, names = self._circuit(n, layers)
        obs = self._observable(n)
        vals = np.linspace(-1.1, 1.3, len(names))

        g_adj = adjoint.adjoint_grad_vector(circuit, obs, names, vals)
        g_ps = parameter_shift.parameter_shift_grad_vector(circuit, obs, names, vals)

        assert g_adj.shape == (2 * n * layers,)
        np.testing.assert_allclose(g_adj, g_ps, atol=1e-10, rtol=0)

    @pytest.mark.parametrize("n,layers", GRID)
    def test_adjoint_matches_finite_difference(self, n, layers):
        circuit, names = self._circuit(n, layers)
        obs = self._observable(n)
        vals = np.linspace(-1.1, 1.3, len(names))
        g_adj = adjoint.adjoint_grad_vector(circuit, obs, names, vals)

        eps = 1e-5
        for idx in (0, len(names) // 2, len(names) - 1):
            shifted = []
            for sign in (1.0, -1.0):
                v = vals.copy()
                v[idx] += sign * eps
                bound = circuit.bind(dict(zip(names, v)))
                sv = np.asarray(
                    sf.run(bound, device="cpu", shots=0).statevector,
                    dtype=np.complex128,
                ).ravel()
                shifted.append(float(np.real(obs.expectation(sv))))
            fd = (shifted[0] - shifted[1]) / (2 * eps)
            assert abs(fd - g_adj[idx]) < 1e-6, (
                f"n={n} layers={layers} param {names[idx]}: fd={fd} adj={g_adj[idx]}"
            )


class TestSPSA:
    def test_spsa_approximates_gradient(self):
        theta = sf.param("theta")
        circuit = sf.Circuit(1).ry(theta, 0)
        observable = SparsePauliOp.from_dict({"Z": 1.0})

        def loss_fn(p):
            bound = circuit.bind({"theta": float(p[0])})
            result = sf.run(bound, device="cpu", shots=0)
            sv = np.asarray(result.statevector, dtype=np.complex128).ravel()
            return jnp.array(float(np.real(observable._fast_expval(sv))))

        params = jnp.array([0.5])
        key = jax.random.PRNGKey(0)
        grad = spsa.spsa_grad(loss_fn, params, key, delta=0.05)

        expected = -math.sin(0.5)
        assert grad.shape == (1,)
        assert abs(float(grad[0]) - expected) < 0.3


class TestQNG:
    def test_qfim_positive_semidefinite(self):
        theta = sf.param("theta")
        circuit = sf.Circuit(1).ry(theta, 0)
        param_vals = {"theta": 0.4}
        bound = circuit.bind(param_vals)
        state = sf.simulate(bound, device="cpu")
        dag = circuit.to_ir()
        g = np.array(state.qfim(dag, param_vals))
        assert g.shape == (1, 1)
        assert float(g[0, 0]) >= 0

    def test_qng_step_updates_parameters(self):
        theta = sf.param("theta")
        circuit = sf.Circuit(1).ry(theta, 0)
        observable_terms = [([3], 1.0, 0.0)]

        param_vals = {"theta": 0.5}
        bound = circuit.bind(param_vals)
        state = sf.simulate(bound, device="cpu")
        dag = circuit.to_ir()

        new_params = qng.qng_step(
            state, dag, observable_terms,
            param_names=["theta"],
            param_values=param_vals,
            learning_rate=0.01,
        )
        assert "theta" in new_params
        assert new_params["theta"] != 0.5


class TestQfimFubiniStudy:
    """Value-level regression tests for State.qfim() (SUP-18).

    The metric must be the Fubini-Study metric, i.e. the covariance
    (gauge) term must be subtracted: for a Pauli rotation e^{-i theta P/2}
    the diagonal is Var(P) in the pre-gate state, not <P^2> = 1.
    """

    def test_rz_on_zero_is_zero(self):
        """n=1 RZ from |0>: pure global phase -> metric must be ~0."""
        c = sf.Circuit(1).rz(sf.param("p0"), 0)
        dag = c.to_ir()
        st = sf.run(c.bind({"p0": 0.7}), device="cpu", shots=0).state
        g = np.array(st.qfim(dag, {"p0": 0.7}))
        assert g.shape == (1, 1)
        assert abs(float(g[0, 0])) < 1e-6, f"RZ on |0> qfim = {g[0, 0]}"

    def test_rx_ry_on_zero_are_one(self):
        """n=1 RX/RY from |0>: Var(P) = 1, unchanged by the fix."""
        for gate in ("rx", "ry"):
            c = sf.Circuit(1)
            getattr(c, gate)(sf.param("p0"), 0)
            dag = c.to_ir()
            st = sf.run(c.bind({"p0": 0.7}), device="cpu", shots=0).state
            val = float(np.asarray(st.qfim(dag, {"p0": 0.7}))[0, 0])
            assert abs(val - 1.0) < 1e-6, f"{gate} on |0> qfim = {val}"

    @staticmethod
    def _sv(circuit, vals):
        return np.asarray(
            sf.run(circuit.bind(vals), device="cpu", shots=0).statevector,
            dtype=np.complex128,
        ).ravel()

    @classmethod
    def _exact_fs_metric(cls, circuit, names, vals, eps=1e-6):
        """Exact FS metric via central-difference state derivatives (SUP-18)."""
        psi0 = cls._sv(circuit, vals)
        dpsi = np.zeros((len(names), psi0.size), dtype=np.complex128)
        for k, nm in enumerate(names):
            vp, vm = dict(vals), dict(vals)
            vp[nm] += eps
            vm[nm] -= eps
            dpsi[k] = (cls._sv(circuit, vp) - cls._sv(circuit, vm)) / (2 * eps)
        gram = dpsi @ dpsi.conj().T
        conn = psi0 @ dpsi.conj().T            # <dpsi_k|psi>
        corr = np.outer(conn, conn.conj())     # <dpsi_i|psi><psi|dpsi_j>
        return 4.0 * np.real(gram - corr)

    @staticmethod
    def _ladder(n):
        c = sf.Circuit(n)
        for i in range(n):
            c.ry(sf.param(f"p{2 * i}"), i)
            c.rz(sf.param(f"p{2 * i + 1}"), i)
        for i in range(n - 1):
            c.cx(i, i + 1)
        return c

    def test_entangled_matches_exact_metric(self):
        """Ladder + RZZ circuits: qfim == exact FS metric (was off by O(1))."""
        cases = [self._ladder(2), self._ladder(3)]
        rzz = sf.Circuit(2).ry(sf.param("p0"), 0)
        rzz.rz(sf.param("p1"), 1)
        rzz.rzz(sf.param("p2"), 0, 1)
        cases.append(rzz)
        rng = np.random.default_rng(11)
        for circuit in cases:
            names = circuit.to_ir().parameter_names()
            vals = dict(zip(names, rng.uniform(-np.pi, np.pi, len(names))))
            st = sf.run(circuit.bind(vals), device="cpu", shots=0).state
            g = np.array(st.qfim(circuit.to_ir(), vals))
            exact = self._exact_fs_metric(circuit, names, vals)
            np.testing.assert_allclose(g, exact, atol=1e-5, rtol=1e-5)


class TestRiemannianGradient:
    def test_riemannian_gradient(self):
        riemannian = pytest.importorskip("superfermion.qml.gradient.riemannian")
        grad = np.array([1.0, 0.5])
        metric = np.eye(2)
        nat_grad = riemannian.riemannian_gradient(grad, metric)
        np.testing.assert_allclose(nat_grad, grad)


class TestStochasticReconfig:
    def test_sr_update(self):
        sr = pytest.importorskip("superfermion.qml.gradient.stochastic_reconfig")
        params = np.array([0.1, 0.2])
        grad = np.array([0.5, -0.3])
        qfim = np.eye(2)
        updated = sr.sr_update(params, grad, qfim, learning_rate=0.01)
        assert updated.shape == (2,)
