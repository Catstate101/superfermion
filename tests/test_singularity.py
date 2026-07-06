
import time
import superfermion as sf

def test_singularity_scaling():
    print("="*80)
    print(f"{'SUPERFERMION SINGULARITY: THE HYPER-ADAPTIVE TEST':^80}")
    print("="*80)
    print(f"{'N':<5} | {'Recommended Mode':<25} | {'Latency (ms)':<15}")
    print("-" * 80)

    for n in [20, 40, 100]:
        c = sf.Circuit(n).h(0)
        for i in range(n-1): c.cx(i, i+1)
        
        t0 = time.perf_counter_ns()
        # Singularity will choice JAX for 20, and Rust for 40/100
        res = sf.run(c, backend="singularity", shots=0)
        lat = (time.perf_counter_ns() - t0) / 1e6
        
        mode = res.metadata.get("singularity_mode", "UNKNOWN")
        print(f"{n:<5} | {mode:<25} | {lat:<15.2f}")
    
    print("="*80)

if __name__ == "__main__":
    test_singularity_scaling()
