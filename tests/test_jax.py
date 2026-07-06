import jax
import jax.numpy as jnp
import superfermion as sf
from superfermion.qml.gradient.core import circuit_to_jax
import numpy as np

def test_jax_grad():
    print("Testing JAX Gradients (Parameter Shift Rule)...")
    
    # 1. Define a simple parameterized circuit
    # An Rx rotation followed by measurement
    theta_sym = sf.param("theta")
    circuit = sf.Circuit(1).rx(theta_sym, 0).measure(0)
    
    # 2. Convert to JAX function
    # Note: f(theta) = |<1|Rx(theta)|0>|^2 = sin(theta/2)^2 = (1 - cos(theta)) / 2
    # df/dtheta = sin(theta) / 2
    f_jax = circuit_to_jax(circuit)
    
    # 3. Compute gradient using JAX
    grad_fn = jax.grad(lambda x: f_jax(x)[1]) # Gradient of probability of state '1'
    
    # 4. Test at theta = pi/2
    # f(pi/2) = 0.5
    # f'(pi/2) = sin(pi/2) / 2 = 0.5
    theta = jnp.array(np.pi / 2)
    grad_val = grad_fn(theta)
    
    print(f"Theta: {theta}")
    print(f"Gradient at pi/2: {grad_val}")
    
    expected_grad = 0.5
    assert jnp.abs(grad_val - expected_grad) < 0.1 # Tolerance for 1000 shots
    print("[PASS] JAX Gradient test passed!")

def test_jax_jit():
    print("\nTesting JAX JIT...")
    theta_sym = sf.param("theta")
    circuit = sf.Circuit(1).rx(theta_sym, 0).measure(0)
    f_jax = circuit_to_jax(circuit)
    
    jit_f = jax.jit(f_jax)
    res = jit_f(jnp.array(np.pi))
    print(f"JIT Result at pi: {res}")
    assert jnp.abs(res[1] - 1.0) < 0.05
    print("[PASS] JAX JIT test passed!")

def test_jax_vmap():
    print("\nTesting JAX VMap (Batching)...")
    theta_sym = sf.param("theta")
    circuit = sf.Circuit(1).rx(theta_sym, 0).measure(0)
    f_jax = circuit_to_jax(circuit)
    
    # Batch of 5 angles
    thetas = jnp.array([0.0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
    vmap_f = jax.vmap(f_jax)
    
    results = vmap_f(thetas)
    print(f"VMap Results (Probabilities of |1>): {results[:, 1]}")
    
    expected = (1 - jnp.cos(thetas)) / 2
    print(f"Expected probabilities: {expected}")
    
    assert jnp.all(jnp.abs(results[:, 1] - expected) < 0.1)
    print("[PASS] JAX VMap test passed!")

if __name__ == "__main__":
    try:
        test_jax_grad()
        test_jax_jit()
        test_jax_vmap()
        print("\nAll JAX Integration tests passed! Superfermion is now differentiable.")
    except Exception as e:
        print(f"\nTests failed: {e}")
        import traceback
        traceback.print_exc()
