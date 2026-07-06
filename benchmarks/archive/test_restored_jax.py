
import superfermion as sf
from superfermion.circuit import Circuit
import time
import jax.numpy as jnp
import numpy as np

def test_restored_jax():
    print("=== Testing Restored JAXBackend Implementation ===")
    
    # 1. Create a 4-qubit circuit with various gates
    n = 4
    c = Circuit(n)
    c.h(0).h(1).h(2).h(3)
    c.cx(0, 1).cx(2, 3)
    c.rz(0.5, 0).rz(0.5, 1)
    c.cz(1, 2)
    c.measure_all()
    
    # 2. Run with JAX Backend
    print("\nRunning on JAX backend...")
    try:
        t0 = time.time()
        res_jax = sf.run(c, backend="jax", shots=1000)
        t_jax = time.time() - t0
        print(f"JAX Run successful in {t_jax:.4f}s")
        print(f"Top counts: {dict(list(res_jax.counts.items())[:5])}")
    except Exception as e:
        print(f"JAX backend failed: {e}")
        return

    # 3. Run with Statevector Backend (Reference)
    print("\nRunning on Statevector backend...")
    res_sv = sf.run(c, backend="statevector", shots=1000)
    print(f"Statevector Run successful.")

    # 4. Compare Statevectors (Fidelity)
    # Note: sf.run(..., shots=1000) results may differ in counts due to random sampling, 
    # but the statevector should be identical.
    # We'll run a simulate() call directly for fidelity
    print("\nComparing Statevectors for Fidelity...")
    from superfermion.backends.registry import get_backend
    jax_backend = get_backend("jax")
    sv_backend = get_backend("statevector")
    
    state_jax = jax_backend.simulate(c)
    state_sv = sv_backend.simulate(c)
    
    # Fidelity calculation
    fidelity = np.abs(np.vdot(np.array(state_jax), state_sv))**2
    print(f"Fidelity (JAX vs SV): {fidelity:.10f}")
    
    if fidelity > 0.9999:
        print("RESULT: SUCCESS - Restored JAX implementation is mathematically accurate.")
    else:
        print("RESULT: FAILURE - Fidelity mismatch.")

if __name__ == "__main__":
    test_restored_jax()
