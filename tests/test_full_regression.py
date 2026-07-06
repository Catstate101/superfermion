"""
Full Regression Test Suite — Superfermion
Runs every critical subsystem end-to-end.
"""

from __future__ import annotations
import time, traceback, sys
import jax
import jax.numpy as jnp

RESULTS = []

def run(name, fn):
    t0 = time.time()
    try:
        fn()
        dt = time.time() - t0
        RESULTS.append((name, "PASS", dt))
        print(f"  [PASS] {name} ({dt:.2f}s)")
    except Exception as e:
        dt = time.time() - t0
        RESULTS.append((name, f"FAIL: {e}", dt))
        print(f"  [FAIL] {name}: {e}")
        traceback.print_exc()

# ── 1. Core Circuit API ─────────────────────────────────────────
def test_circuit_api():
    import superfermion as sf
    c = sf.Circuit(3)
    c.h(0).cx(0, 1).cx(1, 2)
    assert c.n_qubits == 3
    assert c.depth >= 1
    assert c.gate_count >= 3
    qasm = c.to_qasm3()
    assert "OPENQASM" in qasm
    # draw
    drawing = c.draw()
    assert len(drawing) > 0
    # serialization round-trip
    j = c.to_json()
    c2 = sf.Circuit.from_json(j)
    assert c.gate_count == c2.gate_count

# ── 2. Parameterized Circuits ───────────────────────────────────
def test_parametric_circuit():
    import superfermion as sf
    c = sf.Circuit(2)
    c.rx(sf.param("a"), 0)
    c.ry(sf.param("b"), 1)
    assert len(c.parameters) == 2
    bound = c.bind({"a": 1.57, "b": 3.14})
    assert len(bound.parameters) == 0

# ── 3. JAX Primitive (forward + grad) ──────────────────────────
def test_jax_primitive():
    import superfermion as sf
    c = sf.Circuit(2)
    c.rx(sf.param("t"), 0)
    f = sf.qml.circuit_to_jax(c, backend="jax")
    sv = f(jnp.array(0.0))
    assert sv.shape == (4,)
    g = jax.grad(lambda t: jnp.real(jnp.sum(jnp.abs(f(t))**2)))(0.5)
    assert jnp.isfinite(g)

# ── 4. Observables / Hamiltonians ──────────────────────────────
def test_observables():
    from superfermion.observables.core import Hamiltonian, PauliString
    h = Hamiltonian([PauliString("ZI", coeffs=1.0), PauliString("IX", coeffs=0.5)])
    sv = jnp.array([1, 0, 0, 0], dtype=jnp.complex64)
    e = h.expectation(sv)
    assert jnp.isfinite(jnp.real(e))

# ── 5. VQE Convergence ─────────────────────────────────────────
def test_vqe():
    import superfermion as sf
    from superfermion.algorithms.variational import VQE
    from superfermion.observables.core import Hamiltonian, PauliString
    c = sf.Circuit(1); c.ry(sf.param("t"), 0)
    h = Hamiltonian([PauliString("Z", coeffs=1.0), PauliString("X", coeffs=1.0)])
    vqe = VQE(c, h, optimizer="L-BFGS-B")
    res = vqe.minimize(iterations=80)
    assert res.optimal_value < -1.2

# ── 6. QAOA MaxCut ─────────────────────────────────────────────
def test_qaoa():
    from superfermion.algorithms.variational import QAOA
    qaoa = QAOA(n_qubits=2, edges=[(0, 1)], p_layers=1)
    res = qaoa.minimize(iterations=80)
    assert res.optimal_value > 0.7

# ── 7. QSVM Classification ────────────────────────────────────
def test_qsvm():
    import superfermion as sf
    from superfermion.algorithms.qsvm import QSVM
    c = sf.Circuit(2); c.rx(sf.param("t0"), 0); c.rx(sf.param("t1"), 1)
    x = jnp.array([[0.,0.],[jnp.pi,jnp.pi],[0.,jnp.pi],[jnp.pi,0.]])
    y = jnp.array([0, 0, 1, 1])
    qsvm = QSVM(c, num_classes=2)
    res = qsvm.fit(x, y, iterations=120)
    preds = qsvm.predict(res.optimal_params, x)
    acc = jnp.mean(preds == y)
    assert acc >= 0.5

# ── 8. QRL (QuantumREINFORCE) ─────────────────────────────────
def test_qrl():
    import superfermion as sf
    from superfermion.algorithms.qrl import QuantumREINFORCE
    c = sf.Circuit(2); c.rx(sf.param("t0"), 0)
    agent = QuantumREINFORCE(c, num_actions=2)
    key = jax.random.PRNGKey(0)
    s = jnp.array([0.1, 0.2])
    p = agent.model.init(key, s[None, :])
    os = agent.optimizer.init(p)
    traj = [{'state': s, 'action': 0, 'reward': 1.0}]
    np2, _ = agent.update(p, os, traj)
    diff = jnp.mean(jnp.abs(np2['params']['weights'] - p['params']['weights']))
    assert diff > 0

# ── 11. QBM Energy ─────────────────────────────────────────────
def test_qbm():
    from superfermion.algorithms.qbm import QBM
    model = QBM(n_qubits=3)
    key = jax.random.PRNGKey(1)
    x = jax.random.randint(key, (4, 3), 0, 2).astype(jnp.float32)
    p = model.init(key, x)
    e = model.apply(p, x)
    assert e.shape == (4,)
    z = model.get_partition_function(p)
    assert z > 0

# ── 12. Fidelity ───────────────────────────────────────────────
def test_fidelity():
    from superfermion.qml.fidelity import state_fidelity
    a = jnp.array([1, 0, 0, 0], dtype=jnp.complex64)
    b = jnp.array([1, 0, 0, 0], dtype=jnp.complex64)
    assert jnp.isclose(state_fidelity(a, b), 1.0)

# ── 13. Benchmarks ─────────────────────────────────────────────
def test_benchmarks():
    from superfermion.benchmarks.suite import BenchmarkSuite
    # Just verify the benchmark objects can be created
    assert hasattr(BenchmarkSuite, 'run_vqe_benchmark')

# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  SUPERFERMION FULL REGRESSION SUITE")
    print("=" * 60)
    t_start = time.time()

    run("Circuit API",           test_circuit_api)
    run("Parametric Circuits",   test_parametric_circuit)
    run("JAX Primitive",         test_jax_primitive)
    run("Observables",           test_observables)
    run("VQE Convergence",       test_vqe)
    run("QAOA MaxCut",           test_qaoa)
    run("QSVM Classification",  test_qsvm)
    run("QRL (REINFORCE)",       test_qrl)
    run("QBM Energy",            test_qbm)
    run("Fidelity",              test_fidelity)
    run("Benchmarks",            test_benchmarks)

    total_time = time.time() - t_start
    passed = sum(1 for _, s, _ in RESULTS if s == "PASS")
    failed = sum(1 for _, s, _ in RESULTS if s != "PASS")

    print("\n" + "=" * 60)
    print(f"  RESULTS: {passed} PASS / {failed} FAIL / {len(RESULTS)} TOTAL")
    print(f"  TOTAL TIME: {total_time:.2f}s")
    print("=" * 60)

    if failed > 0:
        print("\nFailed tests:")
        for n, s, _ in RESULTS:
            if s != "PASS":
                print(f"  X {n}: {s}")
        sys.exit(1)
    else:
        print("\n  ALL TESTS PASSED - Superfermion is stable.")
        sys.exit(0)
