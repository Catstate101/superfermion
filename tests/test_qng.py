"""
Test Quantum Natural Gradient (QNG).
"""

from __future__ import annotations
import jax
import jax.numpy as jnp
import superfermion as sf
from superfermion.qml.gradient.qng import calculate_qfim, qng_step


def test_qfim_single_qubit():
    print("Testing QFIM calculation...")
    
    # Circuit: RY(theta)|0> = cos(theta/2)|0> + sin(theta/2)|1>
    def circuit_fn(params):
        theta = params[0]
        c = sf.Circuit(1).ry(theta, 0)
        f = sf.qml.circuit_to_jax(c, backend="jax")
        return f() # The circuit_to_jax returns a function that expects params, but c is already bound?
        # No, f(p) is what we need if we use sf.param.
        # Let's do it properly.
    
    c = sf.Circuit(1).ry(sf.param("t"), 0)
    f_jax = sf.qml.circuit_to_jax(c, backend="jax")
    
    # QFIM for 1-qubit RY rotation is constant 0.25 (1/4)
    # G = Re( <∂_theta psi| ∂_theta psi> - |<∂_theta psi|psi>|^2 )
    # |psi> = [cos(t/2), sin(t/2)]
    # |∂_theta psi> = [-1/2 sin(t/2), 1/2 cos(t/2)]
    # <∂_theta psi| ∂_theta psi> = 1/4 (sin^2 + cos^2) = 1/4
    # <∂_theta psi|psi> = -1/2 sin cos + 1/2 sin cos = 0
    # G = 1/4 = 0.25
    
    g = calculate_qfim(f_jax, jnp.array([1.0]))
    print(f"  QFIM at t=1.0: {g}")
    assert jnp.abs(g[0, 0] - 0.25) < 1e-5
    
    print("[PASS] QFIM calculation verified.")


def test_qng_optimization():
    print("\nTesting QNG optimization...")
    
    c = sf.Circuit(1).ry(sf.param("t"), 0)
    f_jax = sf.qml.circuit_to_jax(c, backend="jax")
    
    def loss_fn(params):
        sv = f_jax(params)
        # Goal: reach |1>, so minimize <0|psi>^2
        return jnp.abs(sv[0])**2
    
    params = jnp.array([0.5])
    print(f"  Initial loss: {loss_fn(params):.6f}")
    
    # Take 5 QNG steps
    for i in range(5):
        params = qng_step(loss_fn, f_jax, params, learning_rate=0.5)
        print(f"  Step {i+1}, loss: {loss_fn(params):.6f}, params: {params}")
        
    assert loss_fn(params) < 0.1
    print("[PASS] QNG optimization verified.")


if __name__ == "__main__":
    try:
        test_qfim_single_qubit()
        test_qng_optimization()
        print("\nQNG Module Verified.")
    except Exception as e:
        print(f"\nTests failed: {e}")
        import traceback
        traceback.print_exc()
