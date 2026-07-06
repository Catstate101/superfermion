import numpy as np
import superfermion as sf
from superfermion.circuit import Circuit
from superfermion.backends.jax_sim import JAXBackend
from superfermion.simulator import simulate_statevector

def validate():
    n = 4
    c = Circuit(n).h(0)
    for i in range(n-1):
        c.cnot(i, i+1)
    
    # Truth (NumPy)
    sv_np = simulate_statevector(c)
    
    # JAX
    backend = JAXBackend()
    sv_jax = backend.simulate(c)
    
    # Compare
    fidelity = np.abs(np.vdot(sv_np, sv_jax))**2
    print(f"Fidelity (NP vs JAX): {fidelity:.10f}")
    
    # Check 16q timing more precisely
    n16 = 16
    c16 = Circuit(n16)
    for i in range(n16): c16.h(i)
    
    import time
    t0 = time.perf_counter()
    backend.simulate(c16)
    t_jax = time.perf_counter() - t0
    print(f"JAX 16q time: {t_jax:.6f}s")

if __name__ == "__main__":
    validate()
