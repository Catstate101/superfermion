"""
Physics Validation Suite — Scientific Correctness Tests.

These tests verify that Superfermion produces physically correct results.
They are the gold standard for quantum platform validation.

Tier 1: Algebraic gate identities
Tier 2: Entanglement verification
Tier 3: Quantum information bounds
Tier 4: VQE benchmarks
"""

from __future__ import annotations

import math
import time
import sys
import jax
import jax.numpy as jnp
import superfermion as sf
from superfermion.qml.fidelity import state_fidelity

RESULTS = []

def run(name, fn):
    t0 = time.time()
    try:
        fn()
        dt = time.time() - t0
        RESULTS.append((name, "PASS", dt))
        print(f"  [PASS] {name} ({dt:.3f}s)")
    except Exception as e:
        dt = time.time() - t0
        RESULTS.append((name, f"FAIL: {e}", dt))
        print(f"  [FAIL] {name}: {e}")
        import traceback
        traceback.print_exc()


# ===================================================================
# TIER 1: Algebraic Gate Identities
# ===================================================================

def test_hh_identity():
    """H * H = I"""
    c = sf.Circuit(1)
    c.h(0).h(0)
    f = sf.qml.circuit_to_jax(c, backend="jax")
    sv = f()  # no params
    # Should be |0> = [1, 0]
    fid = state_fidelity(sv, jnp.array([1, 0], dtype=jnp.complex64))
    assert fid > 0.999, f"H*H != I, fidelity={fid}"

def test_xx_identity():
    """X * X = I"""
    c = sf.Circuit(1)
    c.x(0).x(0)
    f = sf.qml.circuit_to_jax(c, backend="jax")
    sv = f()
    fid = state_fidelity(sv, jnp.array([1, 0], dtype=jnp.complex64))
    assert fid > 0.999

def test_zz_identity():
    """Z * Z = I"""
    c = sf.Circuit(1)
    c.z(0).z(0)
    f = sf.qml.circuit_to_jax(c, backend="jax")
    sv = f()
    fid = state_fidelity(sv, jnp.array([1, 0], dtype=jnp.complex64))
    assert fid > 0.999

def test_rx_2pi_identity():
    """Rx(2*pi) = -I (global phase)"""
    c = sf.Circuit(1)
    c.rx(sf.param("t"), 0)
    f = sf.qml.circuit_to_jax(c, backend="jax")
    sv = f(jnp.array(2 * math.pi))
    # Should be -|0> = [-1, 0], which has fidelity 1 with |0>
    fid = state_fidelity(sv, jnp.array([1, 0], dtype=jnp.complex64))
    assert fid > 0.999, f"Rx(2pi) failed, fidelity={fid}"

def test_ry_pi_creates_one():
    """Ry(pi)|0> = |1>"""
    c = sf.Circuit(1)
    c.ry(sf.param("t"), 0)
    f = sf.qml.circuit_to_jax(c, backend="jax")
    sv = f(jnp.array(math.pi))
    prob_1 = float(jnp.abs(sv[1])**2)
    assert prob_1 > 0.999, f"|1> probability = {prob_1}"

def test_h_creates_superposition():
    """H|0> = |+>"""
    c = sf.Circuit(1)
    c.h(0)
    f = sf.qml.circuit_to_jax(c, backend="jax")
    sv = f()
    # Both probabilities should be 0.5
    p0 = float(jnp.abs(sv[0])**2)
    p1 = float(jnp.abs(sv[1])**2)
    assert abs(p0 - 0.5) < 0.01, f"P(0) = {p0}"
    assert abs(p1 - 0.5) < 0.01, f"P(1) = {p1}"


# ===================================================================
# TIER 2: Entanglement Verification
# ===================================================================

def test_bell_state():
    """CNOT(H|0>, |0>) creates a Bell state |Phi+>"""
    c = sf.Circuit(2)
    c.h(0).cx(0, 1)
    f = sf.qml.circuit_to_jax(c, backend="jax")
    sv = f()
    
    # Bell state: (|00> + |11>) / sqrt(2)
    bell = jnp.array([1/jnp.sqrt(2), 0, 0, 1/jnp.sqrt(2)], dtype=jnp.complex64)
    fid = state_fidelity(sv, bell)
    assert fid > 0.999, f"Bell state fidelity = {fid}"

def test_ghz_state():
    """GHZ state |000> + |111> / sqrt(2)"""
    c = sf.Circuit(3)
    c.h(0).cx(0, 1).cx(0, 2)
    f = sf.qml.circuit_to_jax(c, backend="jax")
    sv = f()
    
    # Only |000> and |111> should have nonzero amplitude
    p000 = float(jnp.abs(sv[0])**2)
    p111 = float(jnp.abs(sv[7])**2)
    p_rest = 1.0 - p000 - p111
    assert abs(p000 - 0.5) < 0.01, f"P(000) = {p000}"
    assert abs(p111 - 0.5) < 0.01, f"P(111) = {p111}"
    assert p_rest < 0.01, f"Leakage = {p_rest}"

def test_non_entangled_product():
    """Two independent H gates create a product state, not entangled."""
    c = sf.Circuit(2)
    c.h(0).h(1)
    f = sf.qml.circuit_to_jax(c, backend="jax")
    sv = f()
    
    # All 4 amplitudes should be 1/2
    probs = jnp.abs(sv)**2
    for i in range(4):
        assert abs(float(probs[i]) - 0.25) < 0.01, f"P({i}) = {probs[i]}"


# ===================================================================
# TIER 3: Quantum Information Bounds
# ===================================================================

def test_unitarity():
    """Verify that circuit execution preserves state norm."""
    c = sf.Circuit(3)
    c.h(0).rx(sf.param("t"), 1).cx(0, 2).rz(sf.param("p"), 0)
    f = sf.qml.circuit_to_jax(c, backend="jax")
    sv = f(jnp.array(1.23), jnp.array(0.45))
    
    norm = float(jnp.sum(jnp.abs(sv)**2))
    assert abs(norm - 1.0) < 1e-5, f"Norm = {norm} (should be 1.0)"

def test_no_cloning():
    """No-cloning theorem: can't copy a quantum state with unitary ops.
    Verify that after CNOT, the two qubits are NOT in the same state 
    (they're entangled, not cloned).
    """
    c = sf.Circuit(2)
    c.h(0)  # Create |+>
    c.cx(0, 1)  # "Clone" attempt
    f = sf.qml.circuit_to_jax(c, backend="jax")
    sv = f()
    
    # If cloning worked, we'd have |+>|+> = (|00>+|01>+|10>+|11>)/2
    # But we actually get Bell state (|00>+|11>)/sqrt(2)
    # Verify it's NOT a product state by checking |01> and |10> have zero prob
    p01 = float(jnp.abs(sv[1])**2)
    p10 = float(jnp.abs(sv[2])**2)
    assert p01 < 0.01, f"P(01) = {p01}, should be ~0 (entangled, not cloned)"
    assert p10 < 0.01, f"P(10) = {p10}, should be ~0 (entangled, not cloned)"

def test_born_rule():
    """Verify Born rule: probabilities sum to 1."""
    for n in [1, 2, 3, 4]:
        c = sf.Circuit(n)
        for i in range(n):
            c.h(i)
            if i > 0:
                c.cx(0, i)
        f = sf.qml.circuit_to_jax(c, backend="jax")
        sv = f()
        total_prob = float(jnp.sum(jnp.abs(sv)**2))
        assert abs(total_prob - 1.0) < 1e-5, f"{n}-qubit: P_total = {total_prob}"


# ===================================================================
# TIER 4: VQE Physical Benchmarks
# ===================================================================

def test_vqe_single_qubit_z():
    """VQE for H = Z: ground state is |0>, energy = -1."""
    from superfermion.algorithms.variational import VQE
    from superfermion.observables.core import Hamiltonian, PauliString
    
    c = sf.Circuit(1)
    c.ry(sf.param("t"), 0)
    
    h = Hamiltonian([PauliString("Z", coeffs=1.0)])
    vqe = VQE(c, h)
    result = vqe.minimize(iterations=60)
    
    assert result.optimal_value < -0.95, f"E_0 = {result.optimal_value} (expected -1.0)"

def test_vqe_single_qubit_x():
    """VQE for H = X: ground state is |->, energy = -1."""
    from superfermion.algorithms.variational import VQE
    from superfermion.observables.core import Hamiltonian, PauliString
    
    c = sf.Circuit(1)
    c.ry(sf.param("t"), 0)
    
    h = Hamiltonian([PauliString("X", coeffs=1.0)])
    vqe = VQE(c, h)
    result = vqe.minimize(iterations=60)
    
    assert result.optimal_value < -0.95, f"E_0 = {result.optimal_value} (expected -1.0)"

def test_vqe_heisenberg():
    """VQE for H = Z + X (Heisenberg-like): E_gs = -sqrt(2)."""
    from superfermion.algorithms.variational import VQE
    from superfermion.observables.core import Hamiltonian, PauliString
    
    c = sf.Circuit(1)
    c.ry(sf.param("t"), 0)
    
    h = Hamiltonian([
        PauliString("Z", coeffs=1.0),
        PauliString("X", coeffs=1.0),
    ])
    vqe = VQE(c, h)
    result = vqe.minimize(iterations=100)
    
    expected = -math.sqrt(2)
    assert result.optimal_value < expected + 0.05, \
        f"E_0 = {result.optimal_value} (expected {expected:.4f})"


# ===================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  SUPERFERMION PHYSICS VALIDATION SUITE")
    print("=" * 60)
    t_start = time.time()

    print("\n  TIER 1: Algebraic Gate Identities")
    run("H*H = I",               test_hh_identity)
    run("X*X = I",               test_xx_identity)
    run("Z*Z = I",               test_zz_identity)
    run("Rx(2pi) = -I",          test_rx_2pi_identity)
    run("Ry(pi)|0> = |1>",       test_ry_pi_creates_one)
    run("H|0> = |+>",            test_h_creates_superposition)

    print("\n  TIER 2: Entanglement Verification")
    run("Bell state",            test_bell_state)
    run("GHZ state (3q)",        test_ghz_state)
    run("Product state",         test_non_entangled_product)

    print("\n  TIER 3: Quantum Information Bounds")
    run("Unitarity (norm=1)",    test_unitarity)
    run("No-cloning theorem",    test_no_cloning)
    run("Born rule (1-4q)",      test_born_rule)

    print("\n  TIER 4: VQE Physical Benchmarks")
    run("VQE: H=Z, E=-1",       test_vqe_single_qubit_z)
    run("VQE: H=X, E=-1",       test_vqe_single_qubit_x)
    run("VQE: H=Z+X, E=-sqrt2", test_vqe_heisenberg)

    total_time = time.time() - t_start
    passed = sum(1 for _, s, _ in RESULTS if s == "PASS")
    failed = sum(1 for _, s, _ in RESULTS if s != "PASS")

    print(f"\n{'='*60}")
    print(f"  PHYSICS VALIDATION: {passed} PASS / {failed} FAIL / {len(RESULTS)} TOTAL")
    print(f"  TIME: {total_time:.2f}s")
    
    if failed == 0:
        print("  STATUS: SCIENTIFICALLY VALIDATED")
    else:
        print("  STATUS: VALIDATION FAILED")
        for n, s, _ in RESULTS:
            if s != "PASS":
                print(f"    X {n}: {s}")
    print(f"{'='*60}")
    
    sys.exit(1 if failed > 0 else 0)
