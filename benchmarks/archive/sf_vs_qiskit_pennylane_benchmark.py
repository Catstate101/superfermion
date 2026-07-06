"""
SF Singularity & Rust vs Qiskit Aer & PennyLane
=================================================
Comprehensive benchmark comparing:
  1. Memory efficiency (peak RSS via tracemalloc)
  2. Scientific accuracy (Qiskit Aer & PennyLane as ground truth)
  
Test Circuits:
  A. GHZ (H + CNOT chain) — entanglement
  B. Hardware-style (RX/RY/RZ + CX layers) — VQE-like
  C. QFT-like (H + controlled-phase ladder) — Fourier analysis

Metrics:
  - TVD (Total Variation Distance): statistical closeness
  - Top-K Overlap: dominant bitstrings match
  - Memory: peak MB
  - Time: wall-clock ms
"""

import time
import tracemalloc
import sys
import os
import math
import gc
import numpy as np

# ─── suppress JAX/info noise ───
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ.setdefault("JAX_LOG_COMPILES", "0")

# ────────────────────────────────────────────────────────
# CONFIGURATION
# ────────────────────────────────────────────────────────
SHOTS   = 4096
SEED    = 42
QUBIT_SWEEP = [4, 8, 12, 16, 20]   # full sweep range
np.random.seed(SEED)

# Fixed VQE-style angles (deterministic)
def _angles(n):
    rng = np.random.RandomState(SEED)
    return rng.uniform(0, 2 * math.pi, size=n).tolist()


# ════════════════════════════════════════════════════════
#  GROUND-TRUTH: Qiskit Aer
# ════════════════════════════════════════════════════════

def run_qiskit_aer(circuit_fn, n, label):
    """Build & run a circuit on Qiskit Aer statevector sim."""
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator

    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter_ns()

    qc = circuit_fn("qiskit", n)
    sim = AerSimulator(method="statevector", seed_simulator=SEED)
    res = sim.run(qc, shots=SHOTS, seed_simulator=SEED).result()
    counts_raw = res.get_counts()
    
    fmt = f"0{n}b"
    counts = {format(int(k, 2), fmt)[::-1]: v for k, v in counts_raw.items()}

    elapsed_ms = (time.perf_counter_ns() - t0) / 1e6
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "counts": counts,
        "time_ms": elapsed_ms,
        "mem_mb": peak / (1024 * 1024),
        "label": f"Qiskit-Aer ({label})"
    }


# ════════════════════════════════════════════════════════
#  GROUND-TRUTH: PennyLane
# ════════════════════════════════════════════════════════

def run_pennylane(circuit_fn, n, label):
    """Build & run on PennyLane default.qubit."""
    import pennylane as qml

    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter_ns()

    dev = qml.device("default.qubit", wires=n, shots=SHOTS)
    qfunc = circuit_fn("pennylane", n, dev=dev)
    counts_raw = qfunc()

    # PennyLane returns dict with integer/bitstring keys. 
    # default.qubit on PL uses MSB-first [0,1,2] convention internally like SF.
    fmt = f"0{n}b"
    counts = {}
    if hasattr(counts_raw, "items"):
        for k, v in counts_raw.items():
            # Ensure bitstring format
            if isinstance(k, str):
                # If k is like '110' (only 0s and 1s), it's already a bitstring.
                # If it's like '6' (decimal), we convert.
                if all(c in '01' for c in k) and len(k) == n:
                    k_str = k
                elif k.isdigit():
                    k_str = format(int(k), fmt)
                else:
                    k_str = k
            else:
                k_str = format(int(k), fmt)
            counts[k_str] = int(v)
    else:
        # fallback: array of samples
        for s in np.array(counts_raw).flatten():
            bs = format(int(s), fmt)
            counts[bs] = counts.get(bs, 0) + 1

    elapsed_ms = (time.perf_counter_ns() - t0) / 1e6
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "counts": counts,
        "time_ms": elapsed_ms,
        "mem_mb": peak / (1024 * 1024),
        "label": f"PennyLane ({label})"
    }


# ════════════════════════════════════════════════════════
#  UNDER TEST: SF Singularity
# ════════════════════════════════════════════════════════

def run_sf(circuit_fn, n, backend_name, label):
    """Build & run on SuperFermion backend."""
    import superfermion as sf
    from superfermion.backends.singularity import SingularityBackend

    gc.collect()
    # Clear singleton caches to get a fair cold-start measurement
    SingularityBackend._topology_cache.clear()

    tracemalloc.start()
    t0 = time.perf_counter_ns()

    circ = circuit_fn("sf", n)
    res = sf.run(circ, backend=backend_name, shots=SHOTS)
    counts = res.counts

    elapsed_ms = (time.perf_counter_ns() - t0) / 1e6
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    meta_mode = res.metadata.get("singularity_mode", res.metadata.get("method", backend_name))
    return {
        "counts": counts,
        "time_ms": elapsed_ms,
        "mem_mb": peak / (1024 * 1024),
        "label": f"SF-{backend_name} ({label})",
        "mode": meta_mode,
    }


# ════════════════════════════════════════════════════════
#  CIRCUIT FACTORIES (multi-framework)
# ════════════════════════════════════════════════════════

def ghz_circuit(framework, n, **kw):
    """GHZ state: H(0) then CNOT chain."""
    if framework == "qiskit":
        from qiskit import QuantumCircuit
        qc = QuantumCircuit(n)
        qc.h(0)
        for i in range(n - 1):
            qc.cx(i, i + 1)
        qc.measure_all()
        return qc
    elif framework == "pennylane":
        import pennylane as qml
        dev = kw["dev"]
        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(wires=0)
            for i in range(n - 1):
                qml.CNOT(wires=[i, i + 1])
            return qml.counts()
        return circuit
    else:  # sf
        import superfermion as sf
        c = sf.Circuit(n).h(0)
        for i in range(n - 1):
            c.cx(i, i + 1)
        return c


def hardware_vqe_circuit(framework, n, **kw):
    """VQE-like: RX-RY-RZ layer + entangling CX layer, repeated twice."""
    angles = _angles(n * 6)  # 3 rotations * 2 layers * n qubits
    idx = 0

    if framework == "qiskit":
        from qiskit import QuantumCircuit
        qc = QuantumCircuit(n)
        for _layer in range(2):
            for q in range(n):
                qc.rx(angles[idx], q); idx += 1
                qc.ry(angles[idx], q); idx += 1
                qc.rz(angles[idx], q); idx += 1
            for q in range(0, n - 1, 2):
                qc.cx(q, q + 1)
            for q in range(1, n - 1, 2):
                qc.cx(q, q + 1)
        qc.measure_all()
        return qc
    elif framework == "pennylane":
        import pennylane as qml
        dev = kw["dev"]
        @qml.qnode(dev)
        def circuit():
            nonlocal idx
            idx = 0
            for _layer in range(2):
                for q in range(n):
                    qml.RX(angles[idx], wires=q); idx += 1
                    qml.RY(angles[idx], wires=q); idx += 1
                    qml.RZ(angles[idx], wires=q); idx += 1
                for q in range(0, n - 1, 2):
                    qml.CNOT(wires=[q, q + 1])
                for q in range(1, n - 1, 2):
                    qml.CNOT(wires=[q, q + 1])
            return qml.counts()
        return circuit
    else:
        import superfermion as sf
        c = sf.Circuit(n)
        for _layer in range(2):
            for q in range(n):
                c.rx(angles[idx], q); idx += 1
                c.ry(angles[idx], q); idx += 1
                c.rz(angles[idx], q); idx += 1
            for q in range(0, n - 1, 2):
                c.cx(q, q + 1)
            for q in range(1, n - 1, 2):
                c.cx(q, q + 1)
        return c


def qft_like_circuit(framework, n, **kw):
    """QFT-inspired: H + controlled-phase ladder."""
    if framework == "qiskit":
        from qiskit import QuantumCircuit
        qc = QuantumCircuit(n)
        for i in range(n):
            qc.h(i)
            for j in range(i + 1, min(i + 4, n)):  # limited range to keep it tractable
                angle = math.pi / (2 ** (j - i))
                qc.cp(angle, i, j)
        qc.measure_all()
        return qc
    elif framework == "pennylane":
        import pennylane as qml
        dev = kw["dev"]
        @qml.qnode(dev)
        def circuit():
            for i in range(n):
                qml.Hadamard(wires=i)
                for j in range(i + 1, min(i + 4, n)):
                    angle = math.pi / (2 ** (j - i))
                    qml.ControlledPhaseShift(angle, wires=[i, j])
            return qml.counts()
        return circuit
    else:
        import superfermion as sf
        c = sf.Circuit(n)
        for i in range(n):
            c.h(i)
            for j in range(i + 1, min(i + 4, n)):
                angle = math.pi / (2 ** (j - i))
                c.cp(angle, i, j)
        return c


CIRCUITS = [
    ("GHZ",      ghz_circuit),
    ("VQE-HW",   hardware_vqe_circuit),
    ("QFT-like", qft_like_circuit),
]


# ════════════════════════════════════════════════════════
#  METRICS
# ════════════════════════════════════════════════════════

def tvd(counts_a, counts_b, total_shots):
    """Total Variation Distance between two count dicts."""
    all_keys = set(counts_a.keys()) | set(counts_b.keys())
    diff = sum(abs(counts_a.get(k, 0) - counts_b.get(k, 0)) for k in all_keys)
    return diff / (2 * total_shots)


def top_k_overlap(counts_a, counts_b, k=5):
    """Fraction of top-k bitstrings from A that appear in top-k of B."""
    top_a = set(sorted(counts_a, key=counts_a.get, reverse=True)[:k])
    top_b = set(sorted(counts_b, key=counts_b.get, reverse=True)[:k])
    return len(top_a & top_b) / k


def fidelity_score(tvd_val):
    """Convert TVD to a fidelity-like score in [0, 1]."""
    return 1.0 - tvd_val


# ════════════════════════════════════════════════════════
#  MAIN BENCHMARK
# ════════════════════════════════════════════════════════

def main():
    W = 160
    results_table = []

    print("=" * W)
    print(f"{'SF SINGULARITY & RUST  vs  QISKIT AER & PENNYLANE':^{W}}")
    print(f"{'Memory Efficiency + Scientific Accuracy Benchmark':^{W}}")
    print(f"{'Shots=' + str(SHOTS) + '  Seed=' + str(SEED):^{W}}")
    print("=" * W)

    hdr = (
        f"{'Circuit':<10} | {'N':<3} | {'Framework':<22} | "
        f"{'Time(ms)':>10} | {'Mem(MB)':>9} | "
        f"{'vs QiskitAer TVD':>16} | {'vs PL TVD':>12} | "
        f"{'Fidelity(QK)':>13} | {'Top5(QK)':>9} | {'Top5(PL)':>9} | "
        f"{'Dominant Bitstring':<22}"
    )
    print(hdr)
    print("-" * W)

    for circ_name, circ_fn in CIRCUITS:
        for n in QUBIT_SWEEP:
            row_data = {}

            # ── Ground Truth: Qiskit Aer ──
            try:
                r_qk = run_qiskit_aer(circ_fn, n, circ_name)
                row_data["qiskit"] = r_qk
            except Exception as e:
                print(f"{circ_name:<10} | {n:<3} | {'Qiskit-Aer':<22} | {'FAIL: ' + str(e)[:60]}")
                row_data["qiskit"] = None

            # ── Ground Truth: PennyLane ──
            try:
                r_pl = run_pennylane(circ_fn, n, circ_name)
                row_data["pennylane"] = r_pl
            except Exception as e:
                print(f"{circ_name:<10} | {n:<3} | {'PennyLane':<22} | {'FAIL: ' + str(e)[:60]}")
                row_data["pennylane"] = None

            # ── SF Singularity ──
            try:
                r_sing = run_sf(circ_fn, n, "singularity", circ_name)
                row_data["singularity"] = r_sing
            except Exception as e:
                print(f"{circ_name:<10} | {n:<3} | {'SF-singularity':<22} | {'FAIL: ' + str(e)[:60]}")
                row_data["singularity"] = None

            # ── SF Rust ──
            try:
                r_rust = run_sf(circ_fn, n, "rust", circ_name)
                row_data["rust"] = r_rust
            except Exception as e:
                print(f"{circ_name:<10} | {n:<3} | {'SF-rust':<22} | {'FAIL: ' + str(e)[:60]}")
                row_data["rust"] = None

            # ── Print rows ──
            ref_qk = row_data.get("qiskit")
            ref_pl = row_data.get("pennylane")

            for key in ["qiskit", "pennylane", "singularity", "rust"]:
                r = row_data.get(key)
                if r is None:
                    continue

                # Calculate TVD against ground truths
                tvd_qk = tvd(r["counts"], ref_qk["counts"], SHOTS) if ref_qk else float("nan")
                tvd_pl = tvd(r["counts"], ref_pl["counts"], SHOTS) if ref_pl else float("nan")
                fid_qk = fidelity_score(tvd_qk)
                t5_qk  = top_k_overlap(r["counts"], ref_qk["counts"]) if ref_qk else float("nan")
                t5_pl  = top_k_overlap(r["counts"], ref_pl["counts"]) if ref_pl else float("nan")

                # Dominant bitstring
                dominant = max(r["counts"], key=r["counts"].get) if r["counts"] else "N/A"
                dom_count = r["counts"].get(dominant, 0)

                label = r["label"] if "label" in r else key
                line = (
                    f"{circ_name:<10} | {n:<3} | {label:<22} | "
                    f"{r['time_ms']:>10.2f} | {r['mem_mb']:>9.3f} | "
                    f"{tvd_qk:>16.6f} | {tvd_pl:>12.6f} | "
                    f"{fid_qk:>13.6f} | {t5_qk:>9.2f} | {t5_pl:>9.2f} | "
                    f"{dominant:<14} ({dom_count})"
                )
                print(line)

                results_table.append({
                    "circuit": circ_name, "n": n, "framework": key,
                    "time_ms": r["time_ms"], "mem_mb": r["mem_mb"],
                    "tvd_qk": tvd_qk, "tvd_pl": tvd_pl,
                    "fidelity_qk": fid_qk, "top5_qk": t5_qk, "top5_pl": t5_pl,
                    "dominant": dominant
                })

            print("-" * W)

    # ════════════════════════════════════════════════════════
    #  SUMMARY REPORT
    # ════════════════════════════════════════════════════════
    print("\n" + "=" * W)
    print(f"{'SUMMARY: MEMORY EFFICIENCY & SCIENTIFIC ACCURACY':^{W}}")
    print("=" * W)

    # Aggregate by framework
    from collections import defaultdict
    agg = defaultdict(lambda: {"tvd_qk": [], "mem_mb": [], "time_ms": [], "fid": [], "t5_qk": []})

    for row in results_table:
        fw = row["framework"]
        if not math.isnan(row["tvd_qk"]):
            agg[fw]["tvd_qk"].append(row["tvd_qk"])
        if not math.isnan(row.get("fidelity_qk", float("nan"))):
            agg[fw]["fid"].append(row["fidelity_qk"])
        if not math.isnan(row.get("top5_qk", float("nan"))):
            agg[fw]["t5_qk"].append(row["top5_qk"])
        agg[fw]["mem_mb"].append(row["mem_mb"])
        agg[fw]["time_ms"].append(row["time_ms"])

    print(f"\n{'Framework':<22} | {'Avg TVD(QK)':>12} | {'Avg Fidelity':>13} | {'Avg Top5':>9} | "
          f"{'Avg Mem(MB)':>12} | {'Total Mem(MB)':>14} | {'Avg Time(ms)':>13}")
    print("-" * 120)

    for fw in ["qiskit", "pennylane", "singularity", "rust"]:
        if fw not in agg:
            continue
        d = agg[fw]
        avg_tvd = np.mean(d["tvd_qk"]) if d["tvd_qk"] else float("nan")
        avg_fid = np.mean(d["fid"]) if d["fid"] else float("nan")
        avg_t5  = np.mean(d["t5_qk"]) if d["t5_qk"] else float("nan")
        avg_mem = np.mean(d["mem_mb"])
        tot_mem = np.sum(d["mem_mb"])
        avg_time = np.mean(d["time_ms"])
        print(f"{fw:<22} | {avg_tvd:>12.6f} | {avg_fid:>13.6f} | {avg_t5:>9.4f} | "
              f"{avg_mem:>12.3f} | {tot_mem:>14.3f} | {avg_time:>13.2f}")

    # Memory ratio
    print("\n" + "-" * 80)
    if "singularity" in agg and "qiskit" in agg:
        mem_ratio_sing = np.mean(agg["singularity"]["mem_mb"]) / max(np.mean(agg["qiskit"]["mem_mb"]), 1e-9)
        print(f"  SF Singularity / Qiskit Aer memory ratio:  {mem_ratio_sing:.3f}x")
    if "rust" in agg and "qiskit" in agg:
        mem_ratio_rust = np.mean(agg["rust"]["mem_mb"]) / max(np.mean(agg["qiskit"]["mem_mb"]), 1e-9)
        print(f"  SF Rust / Qiskit Aer memory ratio:         {mem_ratio_rust:.3f}x")
    if "singularity" in agg and "pennylane" in agg:
        mem_ratio_sing_pl = np.mean(agg["singularity"]["mem_mb"]) / max(np.mean(agg["pennylane"]["mem_mb"]), 1e-9)
        print(f"  SF Singularity / PennyLane memory ratio:    {mem_ratio_sing_pl:.3f}x")
    if "rust" in agg and "pennylane" in agg:
        mem_ratio_rust_pl = np.mean(agg["rust"]["mem_mb"]) / max(np.mean(agg["pennylane"]["mem_mb"]), 1e-9)
        print(f"  SF Rust / PennyLane memory ratio:           {mem_ratio_rust_pl:.3f}x")

    # Scientific accuracy verdict
    print("\n" + "=" * W)
    print(f"{'SCIENTIFIC ACCURACY VERDICT':^{W}}")
    print("=" * W)

    for fw in ["singularity", "rust"]:
        if fw not in agg or not agg[fw]["tvd_qk"]:
            continue
        avg_tvd_val = np.mean(agg[fw]["tvd_qk"])
        avg_fid_val = np.mean(agg[fw]["fid"])
        max_tvd_val = np.max(agg[fw]["tvd_qk"])

        if avg_fid_val >= 0.95 and max_tvd_val < 0.10:
            verdict = "PASS — Scientifically Accurate"
        elif avg_fid_val >= 0.85:
            verdict = "MARGINAL — Acceptable for statistical sampling"
        else:
            verdict = "FAIL — Significant deviation from ground truth"

        print(f"  SF {fw.upper():12s}:  avg_fidelity={avg_fid_val:.4f}  max_tvd={max_tvd_val:.4f}  => {verdict}")

    print("\n" + "=" * W)
    print("Benchmark complete.")


if __name__ == "__main__":
    main()
