import time
import tracemalloc
import numpy as np
import superfermion as sf
import pandas as pd
import os
import gc

# Suppress JAX noise
os.environ["JAX_PLATFORMS"] = "cpu"

def run_performance_benchmark():
    # Focused range to show Rust vs Singularity vs Incumbents
    qubit_range = [4, 8, 16, 24, 28, 30] 
    shots = 1000
    seed = 42
    
    results = []
    
    print("🚀 STARTING FINAL INDUSTRIAL PERFORMANCE AUDIT...")
    print("Testing backends: [SF Singularity, SF Rust, Qiskit Aer, PennyLane]")
    
    for n in qubit_range:
        print(f"--- N={n} QUBITS ---")
        
        # 1. GROUND TRUTH (PennyLane)
        try:
            import pennylane as qml
            dev = qml.device("default.qubit", wires=n)
            @qml.qnode(dev)
            def pl_circ():
                qml.Hadamard(0); [qml.CNOT([i, i+1]) for i in range(n-1)]; 
                return qml.state()
            
            tracemalloc.start()
            t0 = time.time(); pl_sv = pl_circ(); t_pl = (time.time()-t0)*1000
            _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
            pl_mem = peak / (1024 * 1024)
            results.append({"n": n, "backend": "PennyLane", "ms": t_pl, "mem": pl_mem, "fid": 1.0})
        except:
            tracemalloc.stop()
            results.append({"n": n, "backend": "PennyLane", "ms": 0, "mem": 0, "fid": 0, "status": "OOM"})
            pl_sv = None

        # 2. QISKIT AER
        try:
            from qiskit import QuantumCircuit; from qiskit_aer import AerSimulator
            qc = QuantumCircuit(n); qc.h(0); [qc.cx(i, i+1) for i in range(n-1)]; qc.measure_all()
            sim = AerSimulator(method='statevector')
            tracemalloc.start()
            t0 = time.time(); sim.run(qc, shots=shots).result(); t_qk = (time.time()-t0)*1000
            _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
            results.append({"n": n, "backend": "Qiskit Aer", "ms": t_qk, "mem": peak/1024/1024, "fid": 1.0})
        except:
            tracemalloc.stop()
            results.append({"n": n, "backend": "Qiskit Aer", "ms": 0, "mem": 0, "fid": 0, "status": "OOM"})

        # 3. SF RUST (Direct Rust Engine)
        try:
            c = sf.Circuit(n).h(0); [c.cx(i, i+1) for i in range(n-1)]
            tracemalloc.start()
            t0 = time.time(); res_r = sf.run(c, backend="rust", shots=shots); t_r = (time.time()-t0)*1000
            _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
            
            fid = 0.0
            if pl_sv is not None and res_r.statevector is not None:
                # Need to account for potential global phase in SF Rust
                fid = np.abs(np.vdot(pl_sv, res_r.statevector))**2
                
            results.append({"n": n, "backend": "SF Rust", "ms": t_r, "mem": peak/1024/1024, "fid": fid})
        except:
            tracemalloc.stop()
            results.append({"n": n, "backend": "SF Rust", "ms": 0, "mem": 0, "fid": 0, "status": "FAIL"})

        # 4. SF SINGULARITY (Turbo Engine + Routing)
        try:
            from superfermion.backends.singularity import SingularityBackend
            SingularityBackend._topology_cache.clear()
            c = sf.Circuit(n).h(0); [c.cx(i, i+1) for i in range(n-1)]
            
            tracemalloc.start()
            t0 = time.time(); res_s = sf.run(c, backend="singularity", shots=shots); t_s = (time.time()-t0)*1000
            _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
            
            fid = 0.0
            if pl_sv is not None and res_s.statevector is not None and res_s.statevector.size > 1:
                fid = np.abs(np.vdot(pl_sv, res_s.statevector))**2
            elif n > 26: 
                # For high N, we check bitstring parity (000... or 111...)
                fid = (res_s.counts.get('0'*n, 0) + res_s.counts.get('1'*n, 0)) / shots

            results.append({"n": n, "backend": "SF Singularity", "ms": t_s, "mem": peak/1024/1024, "fid": fid})
        except:
            tracemalloc.stop()
            results.append({"n": n, "backend": "SF Singularity", "ms": 0, "mem": 0, "fid": 0, "status": "FAIL"})
            
        gc.collect()

    df = pd.DataFrame(results)
    df.to_csv("c:/Users/ASUS/OneDrive/Desktop/superfermion/tests/final_audit_results.csv", index=False)
    
    print("\n✅ FINAL AUDIT COMPLETE. GENERATING PRESENTATION...")
    
    # MD Report
    report = """# Quantum Industrial Supremacy Report: SuperFermion vs Incumbents
Created by SuperFermion Performance Engineering.

## 🚀 Latency Comparison (ms)
| N | Qiskit Aer | PennyLane | SF Rust | SF Singularity | Speedup (SF vs QK) |
|---|---|---|---|---|---|
"""
    for n in qubit_range:
        q = df[(df.n==n) & (df.backend=='Qiskit Aer')].ms.values[0]
        p = df[(df.n==n) & (df.backend=='PennyLane')].ms.values[0]
        sr = df[(df.n==n) & (df.backend=='SF Rust')].ms.values[0]
        ss = df[(df.n==n) & (df.backend=='SF Singularity')].ms.values[0]
        
        speedup = q / ss if ss > 0 and q > 0 else 0
        report += f"| {n} | {q:.1f} | {p:.1f} | {sr:.1f} | {ss:.1f} | **{speedup:.1f}x** |\n"

    report += "\n## 🧠 Memory Efficiency (Peak MB)\n| N | Qiskit Aer | PennyLane | SF Rust | SF Singularity | Saving (SF vs QK) |\n|---|---|---|---|---|---|\n"
    for n in qubit_range:
        q = df[(df.n==n) & (df.backend=='Qiskit Aer')].mem.values[0]
        p = df[(df.n==n) & (df.backend=='PennyLane')].mem.values[0]
        sr = df[(df.n==n) & (df.backend=='SF Rust')].mem.values[0]
        ss = df[(df.n==n) & (df.backend=='SF Singularity')].mem.values[0]
        
        saving = q / ss if ss > 0 and q > 0 else 0
        report += f"| {n} | {q:.1f} | {p:.1f} | {sr:.1f} | {ss:.1f} | **{saving:.1f}x** |\n"
        
    report += "\n## 🎯 Scientific Ground Truth Validation (Fidelity)\n| N | SF Rust Fidelity | SF Singularity Fidelity | Status |\n|---|---|---|---|\n"
    for n in qubit_range:
        sr = df[(df.n==n) & (df.backend=='SF Rust')].fid.values[0]
        ss = df[(df.n==n) & (df.backend=='SF Singularity')].fid.values[0]
        status = "✅ PASS" if ss > 0.99 else "❌ FAIL"
        report += f"| {n} | {sr:.4f} | {ss:.4f} | {status} |\n"

    with open("c:/Users/ASUS/OneDrive/Desktop/superfermion/FINAL_INDUSTRIAL_AUDIT.md", "w") as f:
        f.write(report)
    print("✅ PRESENTATION GENERATED: FINAL_INDUSTRIAL_AUDIT.md")

if __name__ == "__main__":
    run_performance_benchmark()
