"""
Speed Benchmark v2: Compact, Row-by-Row print for clear terminal capture.
"""
import time, gc, math, os
import numpy as np
os.environ["JAX_PLATFORMS"] = "cpu"

SHOTS = 1024
SEED  = 42
RUNS  = 5
QUBIT_SWEEP = [4, 8, 12, 16, 20]
np.random.seed(SEED)

def _angles(n):
    return np.random.RandomState(SEED).uniform(0, 2*math.pi, n).tolist()

# ── QISKIT ───────────────────────────────────────────────
def run_qiskit(n, circuit_type):
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator
    sim = AerSimulator(method="statevector", seed_simulator=SEED)
    ang = _angles(n*6); idx = 0
    qc = QuantumCircuit(n)
    if circuit_type == "GHZ":
        qc.h(0)
        for i in range(n-1): qc.cx(i, i+1)
    elif circuit_type == "VQE":
        for _ in range(2):
            for q in range(n): qc.rx(ang[idx],q); idx+=1; qc.ry(ang[idx],q); idx+=1; qc.rz(ang[idx],q); idx+=1
            for q in range(0,n-1,2): qc.cx(q,q+1)
            for q in range(1,n-1,2): qc.cx(q,q+1)
    elif circuit_type == "QFT":
        for i in range(n):
            qc.h(i)
            for j in range(i+1, min(i+4,n)): qc.cp(math.pi/(2**(j-i)), i, j)
    qc.measure_all()
    times = []
    for _ in range(RUNS):
        gc.collect()
        t = time.perf_counter_ns()
        sim.run(qc, shots=SHOTS, seed_simulator=SEED).result()
        times.append((time.perf_counter_ns()-t)/1e6)
    return round(np.mean(times), 2)

# ── PENNYLANE ─────────────────────────────────────────────
def run_pennylane(n, circuit_type):
    import pennylane as qml
    ang = _angles(n*6)
    times = []
    for _ in range(RUNS):
        dev = qml.device("default.qubit", wires=n, shots=SHOTS)
        @qml.qnode(dev)
        def circ():
            idx = 0
            if circuit_type == "GHZ":
                qml.Hadamard(wires=0)
                for i in range(n-1): qml.CNOT(wires=[i, i+1])
            elif circuit_type == "VQE":
                for _ in range(2):
                    for q in range(n):
                        qml.RX(ang[idx],wires=q); idx+=1
                        qml.RY(ang[idx],wires=q); idx+=1
                        qml.RZ(ang[idx],wires=q); idx+=1
                    for q in range(0,n-1,2): qml.CNOT(wires=[q,q+1])
                    for q in range(1,n-1,2): qml.CNOT(wires=[q,q+1])
            elif circuit_type == "QFT":
                for i in range(n):
                    qml.Hadamard(wires=i)
                    for j in range(i+1, min(i+4,n)):
                        qml.ControlledPhaseShift(math.pi/(2**(j-i)), wires=[i,j])
            return qml.counts()
        gc.collect()
        t = time.perf_counter_ns()
        circ()
        times.append((time.perf_counter_ns()-t)/1e6)
    return round(np.mean(times), 2)

# ── SUPERFERMION ──────────────────────────────────────────
def run_sf(backend, n, circuit_type):
    import superfermion as sf
    from superfermion.backends.singularity import SingularityBackend
    ang = _angles(n*6); idx = 0
    c = sf.Circuit(n)
    if circuit_type == "GHZ":
        c.h(0)
        for i in range(n-1): c.cx(i, i+1)
    elif circuit_type == "VQE":
        for _ in range(2):
            for q in range(n): c.rx(ang[idx],q); idx+=1; c.ry(ang[idx],q); idx+=1; c.rz(ang[idx],q); idx+=1
            for q in range(0,n-1,2): c.cx(q,q+1)
            for q in range(1,n-1,2): c.cx(q,q+1)
    elif circuit_type == "QFT":
        for i in range(n):
            c.h(i)
            for j in range(i+1, min(i+4,n)): c.cp(math.pi/(2**(j-i)), i, j)

    SingularityBackend._topology_cache.clear()
    for attr in ["_rust_baked_result","_singularity_baked_kernel","_jax_baked_kernel"]:
        if hasattr(c, attr): delattr(c, attr)

    times = []
    for _ in range(RUNS):
        gc.collect()
        t = time.perf_counter_ns()
        sf.run(c, backend=backend, shots=SHOTS)
        times.append((time.perf_counter_ns()-t)/1e6)
    return round(np.mean(times), 2)

# ── MAIN ──────────────────────────────────────────────────
def main():
    import sys
    circuits = ["GHZ", "VQE", "QFT"]
    
    all_results = []
    
    for ct in circuits:
        for n in QUBIT_SWEEP:
            row = {"circuit": ct, "n": n}
            
            sys.stdout.write(f"  Running {ct} N={n} Qiskit...  "); sys.stdout.flush()
            try: row["qk"] = run_qiskit(n, ct)
            except Exception as e: row["qk"] = None; sys.stdout.write(f"FAIL({e}) ")
            sys.stdout.write(f"PL...  "); sys.stdout.flush()
            try: row["pl"] = run_pennylane(n, ct)
            except Exception as e: row["pl"] = None; sys.stdout.write(f"FAIL ")
            sys.stdout.write(f"Rust...  "); sys.stdout.flush()
            try: row["rust"] = run_sf("rust", n, ct)
            except Exception as e: row["rust"] = None; sys.stdout.write(f"FAIL ")
            sys.stdout.write(f"Sing...  "); sys.stdout.flush()
            try: row["sing"] = run_sf("singularity", n, ct)
            except Exception as e: row["sing"] = None; sys.stdout.write(f"FAIL ")
            sys.stdout.write("done\n"); sys.stdout.flush()
            all_results.append(row)

    # Print clean table
    W = 130
    print("\n" + "="*W)
    print(f"{'SPEED RESULTS (Average over ' + str(RUNS) + ' runs, ms lower=faster)':^{W}}")
    print("="*W)
    HDR = f"  {'Circuit':<8} | {'N':>3} | {'QiskitAer':>11} | {'PennyLane':>11} | {'SF-Rust':>9} | {'SF-Sing':>9} | {'Rust/QK':>9} | {'Sing/QK':>9} | {'Rust/PL':>9} | {'Sing/PL':>9} | Fastest"
    print(HDR)
    print("-"*W)
    
    speedups_rust_qk, speedups_sing_qk = [], []
    speedups_rust_pl, speedups_sing_pl = [], []
    
    prev_ct = None
    for row in all_results:
        if prev_ct and row["circuit"] != prev_ct:
            print("-"*W)
        prev_ct = row["circuit"]
        
        qk, pl, rust, sing = row["qk"], row["pl"], row["rust"], row["sing"]
        
        def sp(ref, tst):
            if ref is None or tst is None or tst == 0: return None
            return ref / tst
        
        r_r_qk = sp(qk, rust);  r_s_qk = sp(qk, sing)
        r_r_pl  = sp(pl, rust);  r_s_pl  = sp(pl, sing)
        
        if r_r_qk: speedups_rust_qk.append(r_r_qk)
        if r_s_qk: speedups_sing_qk.append(r_s_qk)
        if r_r_pl: speedups_rust_pl.append(r_r_pl)
        if r_s_pl: speedups_sing_pl.append(r_s_pl)
        
        def fms(v): return f"{v:>9.1f}" if v is not None else "     FAIL"
        def fsp(v): return f"{v:>8.2f}x" if v is not None else "    FAIL"
        
        times = {k: v for k, v in [("QK", qk),("PL", pl),("Rust", rust),("Sing", sing)] if v is not None}
        fastest = min(times, key=times.get) if times else "N/A"
        
        print(f"  {row['circuit']:<8} | {row['n']:>3} | {fms(qk):>11} | {fms(pl):>11} | {fms(rust):>9} | {fms(sing):>9} | {fsp(r_r_qk):>9} | {fsp(r_s_qk):>9} | {fsp(r_r_pl):>9} | {fsp(r_s_pl):>9} | {fastest}")
    
    print("="*W)
    print(f"\n  AVERAGE SPEEDUP SUMMARY:")
    print(f"  {'SF-Rust   vs Qiskit Aer':<28}: {np.mean(speedups_rust_qk):.2f}x  ({'faster' if np.mean(speedups_rust_qk)>1 else 'slower'})")
    print(f"  {'SF-Sing   vs Qiskit Aer':<28}: {np.mean(speedups_sing_qk):.2f}x  ({'faster' if np.mean(speedups_sing_qk)>1 else 'slower'})")
    print(f"  {'SF-Rust   vs PennyLane':<28}: {np.mean(speedups_rust_pl):.2f}x  ({'faster' if np.mean(speedups_rust_pl)>1 else 'slower'})")
    print(f"  {'SF-Sing   vs PennyLane':<28}: {np.mean(speedups_sing_pl):.2f}x  ({'faster' if np.mean(speedups_sing_pl)>1 else 'slower'})")
    print(f"\n  Peak SF-Rust speedup  (vs QK): {max(speedups_rust_qk):.2f}x")
    print(f"  Peak SF-Sing speedup  (vs QK): {max(speedups_sing_qk):.2f}x")
    print(f"  Peak SF-Rust speedup  (vs PL): {max(speedups_rust_pl):.2f}x")
    print(f"  Peak SF-Sing speedup  (vs PL): {max(speedups_sing_pl):.2f}x")
    print("="*W)

if __name__ == "__main__":
    main()
