"""
Performance Stress Test — Benchmarking Superfermion JAX backend on large circuits.
"""

from __future__ import annotations
import time
import jax
import jax.numpy as jnp
import superfermion as sf


def benchmark_ghz(min_qubits: int = 10, max_qubits: int = 22):
    print(f"{'Qubits':<10} | {'JIT Time':<12} | {'Exec Time':<12} | {'Mem (MB)':<10}")
    print("-" * 55)
    
    for n in range(min_qubits, max_qubits + 1, 2):
        # Build GHZ circuit
        c = sf.Circuit(n)
        c.h(0)
        for i in range(n - 1):
            c.cx(i, i + 1)
            
        f_jax = sf.qml.circuit_to_jax(c, backend="jax")
        
        # 1. Measure JIT Time
        t0 = time.time()
        sv = f_jax().block_until_ready()
        jit_time = time.time() - t0
        
        # 2. Measure Exec Time (mean of 5 runs for speed)
        times = []
        for _ in range(5):
            t0 = time.time()
            sv = f_jax().block_until_ready()
            times.append(time.time() - t0)
        exec_time = sum(times) / len(times)
        
        # 3. Estimate Memory
        mem_mb = (2**n * 8) / (1024 * 1024)
        print(f"{n:<10} | {jit_time*1000:>8.2f} ms | {exec_time*1000:>8.2f} ms | {mem_mb:>8.2f}")


def benchmark_vqe_scaling(qubits: int = 10):
    print(f"\nBenchmarking VQE Gradient Scaling ({qubits} qubits)...")
    
    layers = 1
    c = sf.Circuit(qubits)
    for l in range(layers):
        for i in range(qubits):
            c.ry(sf.param(f"p{l}_{i}"), i)
            
    f_jax = sf.qml.circuit_to_jax(c, backend="jax")
    params = jnp.zeros(len(c.parameters))
    
    def loss_fn(p):
        sv = f_jax(p)
        return jnp.real(sv[0])
        
    grad_fn = jax.grad(loss_fn)
    
    # Warmup
    grad_fn(params).block_until_ready()
    
    t0 = time.time()
    for _ in range(5):
        grad_fn(params).block_until_ready()
    avg_grad_time = (time.time() - t0) / 5
    
    print(f"  Avg Gradient Time: {avg_grad_time*1000:.2f} ms")


if __name__ == "__main__":
    print("=" * 60)
    print("  SUPERFERMION PERFORMANCE STRESS TEST")
    print("=" * 60)
    
    try:
        benchmark_ghz(10, 24)
        benchmark_vqe_scaling(12)
    except Exception as e:
        print(f"Stress test error: {e}")
