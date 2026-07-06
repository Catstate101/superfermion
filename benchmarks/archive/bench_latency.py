import jax, time
import numpy as np
import superfermion as sf
from superfermion.backends.jax_sim import JAXBackend

def bench_jax_hot():
    c = sf.Circuit(10)
    c.h(0)
    for i in range(9): c.cx(i, i+1)
    
    sim = JAXBackend()
    # 1. Cold Start
    t0 = time.perf_counter_ns()
    sim.run(c, shots=0)
    lat_cold = time.perf_counter_ns() - t0
    
    # 2. Hot Start (Actually using the cache in JAXBackend)
    t0 = time.perf_counter_ns()
    sim.run(c, shots=0)
    lat_hot = time.perf_counter_ns() - t0
    
    print(f"JAX GHZ-10 Cold: {lat_cold:>12,} ns")
    print(f"JAX GHZ-10 Hot:  {lat_hot:>12,} ns")

if __name__ == "__main__":
    bench_jax_hot()
