"""
Benchpress SF vs Qiskit comparison.
Runs construction, manipulation, and transpilation workouts,
then prints a side-by-side comparison table.
"""
import time
import json
import random
import math
import multiprocessing as mp
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Optional, Dict, Any

SEED = 12345
N_QUBITS = 100
TIMEOUT = 180

# ─── helpers ──────────────────────────────────────────────

def heavy_hex_edges(n: int) -> List[Tuple[int, int]]:
    from superfermion._sf_core import CouplingMap as CM
    return CM.heavy_hex(n).edges()


@dataclass
class Result:
    category: str
    test: str
    sf_time_s: float = -1.0
    qk_time_s: float = -1.0
    sf_extra: Dict[str, Any] = field(default_factory=dict)
    qk_extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def winner(self):
        if self.sf_time_s < 0 and self.qk_time_s < 0:
            return "Both failed"
        if self.sf_time_s < 0:
            return "Qiskit"
        if self.qk_time_s < 0:
            return "SF"
        return "SF" if self.sf_time_s <= self.qk_time_s else "Qiskit"

    @property
    def ratio(self):
        if self.sf_time_s < 0 or self.qk_time_s < 0:
            return "n/a"
        if self.sf_time_s <= self.qk_time_s:
            return f"{self.qk_time_s / max(self.sf_time_s, 1e-9):.1f}x"
        return f"{self.sf_time_s / max(self.qk_time_s, 1e-9):.1f}x"


# ─── circuit builders ────────────────────────────────────

def build_qft_sf(n):
    import superfermion as sf
    c = sf.Circuit(n)
    for i in range(n):
        c.h(i)
        for j in range(i + 1, n):
            angle = math.pi / (2 ** (j - i))
            c.cp(angle, i, j)
    return c

def build_qft_qk(n):
    from qiskit.circuit.library import QFT
    return QFT(n)

def build_qv_sf(n, depth=10, seed=SEED):
    import superfermion as sf
    rng = random.Random(seed)
    c = sf.Circuit(n)
    for _ in range(depth):
        perm = list(range(n))
        rng.shuffle(perm)
        for k in range(0, n - 1, 2):
            q0, q1 = perm[k], perm[k + 1]
            c.h(q0)
            c.cx(q0, q1)
            c.rz(rng.uniform(0, 2 * math.pi), q0)
            c.ry(rng.uniform(0, 2 * math.pi), q1)
    return c

def build_qv_qk(n, depth=10, seed=SEED):
    from qiskit.circuit import QuantumCircuit
    import numpy as np
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(n)
    for _ in range(depth):
        perm = rng.permutation(n).tolist()
        for k in range(0, n - 1, 2):
            q0, q1 = perm[k], perm[k + 1]
            qc.h(q0)
            qc.cx(q0, q1)
            qc.rz(float(rng.uniform(0, 2 * math.pi)), q0)
            qc.ry(float(rng.uniform(0, 2 * math.pi)), q1)
    return qc

def build_su2_sf(n, reps=1, seed=SEED):
    import superfermion as sf
    rng = random.Random(seed)
    c = sf.Circuit(n)
    for _ in range(reps + 1):
        for i in range(n):
            c.ry(rng.uniform(0, 2 * math.pi), i)
            c.rz(rng.uniform(0, 2 * math.pi), i)
        for i in range(n - 1):
            c.cx(i, i + 1)
    return c

def build_su2_qk(n, reps=1, seed=SEED):
    from qiskit.circuit import QuantumCircuit
    rng = random.Random(seed)
    qc = QuantumCircuit(n)
    for _ in range(reps + 1):
        for i in range(n):
            qc.ry(rng.uniform(0, 2 * math.pi), i)
            qc.rz(rng.uniform(0, 2 * math.pi), i)
        for i in range(n - 1):
            qc.cx(i, i + 1)
    return qc

def build_bv_sf(n, seed=SEED):
    import superfermion as sf
    rng = random.Random(seed)
    secret = [rng.randint(0, 1) for _ in range(n - 1)]
    c = sf.Circuit(n)
    c.h(n - 1)
    c.z(n - 1)
    for i in range(n - 1):
        c.h(i)
    for i, bit in enumerate(secret):
        if bit:
            c.cx(i, n - 1)
    for i in range(n - 1):
        c.h(i)
    return c

def build_bv_qk(n, seed=SEED):
    from qiskit.circuit import QuantumCircuit
    rng = random.Random(seed)
    secret = [rng.randint(0, 1) for _ in range(n - 1)]
    qc = QuantumCircuit(n)
    qc.h(n - 1)
    qc.z(n - 1)
    for i in range(n - 1):
        qc.h(i)
    for i, bit in enumerate(secret):
        if bit:
            qc.cx(i, n - 1)
    for i in range(n - 1):
        qc.h(i)
    return qc

def build_heisenberg_sf(n, seed=SEED):
    import superfermion as sf
    rng = random.Random(seed)
    c = sf.Circuit(n)
    dt = 0.1
    for i in range(n - 1):
        jx = rng.uniform(0.5, 1.5)
        c.cx(i, i + 1)
        c.rz(2 * jx * dt, i + 1)
        c.cx(i, i + 1)
    for i in range(n):
        c.rz(rng.uniform(0, 0.2) * dt, i)
    return c

def build_heisenberg_qk(n, seed=SEED):
    from qiskit.circuit import QuantumCircuit
    rng = random.Random(seed)
    qc = QuantumCircuit(n)
    dt = 0.1
    for i in range(n - 1):
        jx = rng.uniform(0.5, 1.5)
        qc.cx(i, i + 1)
        qc.rz(2 * jx * dt, i + 1)
        qc.cx(i, i + 1)
    for i in range(n):
        qc.rz(rng.uniform(0, 0.2) * dt, i)
    return qc

def build_qaoa_sf(n, seed=SEED):
    import superfermion as sf
    rng = random.Random(seed)
    c = sf.Circuit(n)
    gamma = rng.uniform(0.1, 1.0)
    beta = rng.uniform(0.1, 1.0)
    for i in range(n):
        c.h(i)
    for i in range(n - 1):
        c.cx(i, i + 1)
        c.rz(2 * gamma, i + 1)
        c.cx(i, i + 1)
    for i in range(0, n - 2, 2):
        c.cx(i, i + 2)
        c.rz(2 * gamma, i + 2)
        c.cx(i, i + 2)
    for i in range(n):
        c.rx(2 * beta, i)
    return c

def build_qaoa_qk(n, seed=SEED):
    from qiskit.circuit import QuantumCircuit
    rng = random.Random(seed)
    qc = QuantumCircuit(n)
    gamma = rng.uniform(0.1, 1.0)
    beta = rng.uniform(0.1, 1.0)
    for i in range(n):
        qc.h(i)
    for i in range(n - 1):
        qc.cx(i, i + 1)
        qc.rz(2 * gamma, i + 1)
        qc.cx(i, i + 1)
    for i in range(0, n - 2, 2):
        qc.cx(i, i + 2)
        qc.rz(2 * gamma, i + 2)
        qc.cx(i, i + 2)
    for i in range(n):
        qc.rx(2 * beta, i)
    return qc

def build_simplification_sf(n):
    import superfermion as sf
    c = sf.Circuit(n)
    for i in range(n):
        c.h(i)
        c.h(i)
        c.x(i)
    return c

def build_simplification_qk(n):
    from qiskit.circuit import QuantumCircuit
    qc = QuantumCircuit(n)
    for i in range(n):
        qc.h(i)
        qc.h(i)
        qc.x(i)
    return qc

def build_clifford_sf(n, depth=100, seed=SEED):
    import superfermion as sf
    rng = random.Random(seed)
    c = sf.Circuit(n)
    cliff_1q = ['h', 's', 'x', 'y', 'z']
    for _ in range(depth):
        for i in range(n):
            gate = rng.choice(cliff_1q)
            getattr(c, gate)(i)
        pairs = list(range(n))
        rng.shuffle(pairs)
        for k in range(0, n - 1, 2):
            c.cx(pairs[k], pairs[k + 1])
    return c

def build_clifford_qk(n, depth=100, seed=SEED):
    from qiskit.circuit import QuantumCircuit
    rng = random.Random(seed)
    qc = QuantumCircuit(n)
    cliff_1q = ['h', 's', 'x', 'y', 'z']
    for _ in range(depth):
        for i in range(n):
            gate = rng.choice(cliff_1q)
            getattr(qc, gate)(i)
        pairs = list(range(n))
        rng.shuffle(pairs)
        for k in range(0, n - 1, 2):
            qc.cx(pairs[k], pairs[k + 1])
    return qc


# ─── transpilation runners ───────────────────────────────

def _run_sf_transpile(circuit_builder, n, backend_name, result_queue):
    """Run SF transpilation in a subprocess."""
    try:
        import superfermion as sf
        from superfermion.compiler.specs import SPECS
        hw = SPECS[backend_name]
        circ = circuit_builder(n)
        t0 = time.perf_counter()
        compiled = sf.compile(circ, target=hw)
        t1 = time.perf_counter()
        compiled._ensure_gates()
        gates = compiled._gates
        n_total = len(gates)
        gate_names = [g.name.upper() for g in gates]
        n_2q = sum(1 for g in gates if len(g.qubits) == 2)
        result_queue.put({
            "time": t1 - t0,
            "gate_count": n_total,
            "gate_count_2q": n_2q,
        })
    except Exception as e:
        result_queue.put({"error": str(e)})

def _run_qk_transpile(circuit_builder, n, edges, native_gates, result_queue):
    """Run Qiskit transpilation in a subprocess."""
    try:
        from qiskit import transpile
        from qiskit.transpiler import CouplingMap
        circ = circuit_builder(n)
        cm = CouplingMap(couplinglist=edges)
        t0 = time.perf_counter()
        compiled = transpile(circ, coupling_map=cm, basis_gates=native_gates, optimization_level=2)
        t1 = time.perf_counter()
        ops = compiled.count_ops()
        n_total = sum(ops.values())
        two_q_names = {'cx', 'cz', 'ecr', 'cy', 'swap', 'iswap', 'rzz', 'rxx', 'ryy', 'cp', 'cnot'}
        n_2q = sum(v for k, v in ops.items() if k in two_q_names)
        result_queue.put({
            "time": t1 - t0,
            "gate_count": n_total,
            "gate_count_2q": n_2q,
        })
    except Exception as e:
        result_queue.put({"error": str(e)})

def run_with_timeout(target, args, timeout):
    q = mp.Queue()
    p = mp.Process(target=target, args=(*args, q))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.kill()
        p.join()
        return {"error": f"Timeout ({timeout}s)"}
    if q.empty():
        return {"error": "No result returned"}
    return q.get()


# ─── manipulation benchmarks ─────────────────────────────

def bench_basis_change():
    """Test basis translation speed: CX basis → CZ basis."""
    r = Result("Manipulation", "bench_basis_change")
    n = N_QUBITS
    target_basis_sf = ['SX', 'X', 'RZ', 'CZ']
    target_basis_qk = ['sx', 'x', 'rz', 'cz']

    # SF
    try:
        import superfermion as sf
        from superfermion._sf_core import Compiler
        circ = build_qv_sf(n, depth=10, seed=SEED)
        ir = circ.to_ir()
        compiler = Compiler(
            name='basis_bench',
            native_gates=target_basis_sf,
            n_qubits=n,
            connectivity=[],
            optimization_level=1,
        )
        t0 = time.perf_counter()
        compiled = compiler.compile(ir)
        t1 = time.perf_counter()
        records = compiled.to_gate_records()
        n_2q = sum(1 for name, _, _ in records if name.lower() in ('cz', 'cx', 'cnot', 'ecr'))
        r.sf_time_s = t1 - t0
        r.sf_extra = {"gate_count_2q": n_2q, "target_basis": target_basis_sf}
    except Exception as e:
        r.sf_extra = {"error": str(e)}

    # Qiskit
    try:
        from qiskit import transpile
        circ_qk = build_qv_qk(n, depth=10, seed=SEED)
        t0 = time.perf_counter()
        compiled_qk = transpile(circ_qk, basis_gates=target_basis_qk, optimization_level=0)
        t1 = time.perf_counter()
        ops = compiled_qk.count_ops()
        n_2q = sum(v for k, v in ops.items() if k in ('cz', 'cx', 'ecr'))
        r.qk_time_s = t1 - t0
        r.qk_extra = {"gate_count_2q": n_2q, "target_basis": target_basis_qk}
    except Exception as e:
        r.qk_extra = {"error": str(e)}

    return r


# ─── transpilation benchmark suite ───────────────────────

def _transpile_test(name, sf_builder, qk_builder, n, backend_name, edges, native_gates):
    r = Result("Transpilation", name)

    sf_res = run_with_timeout(_run_sf_transpile, (sf_builder, n, backend_name), TIMEOUT)
    if "error" in sf_res:
        r.sf_extra = {"error": sf_res["error"]}
    else:
        r.sf_time_s = sf_res["time"]
        r.sf_extra = {k: v for k, v in sf_res.items() if k != "time"}

    qk_res = run_with_timeout(_run_qk_transpile, (qk_builder, n, edges, native_gates), TIMEOUT)
    if "error" in qk_res:
        r.qk_extra = {"error": qk_res["error"]}
    else:
        r.qk_time_s = qk_res["time"]
        r.qk_extra = {k: v for k, v in qk_res.items() if k != "time"}

    return r


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    edges_127 = heavy_hex_edges(127)
    native_ecr = ['rz', 'sx', 'x', 'ecr']

    results: List[Result] = []

    print("=" * 72)
    print("  BENCHPRESS: SF vs Qiskit  (post gap-closure)")
    print("=" * 72)

    # ── Manipulation ──────────────────────────────────────
    print("\n▸ Manipulation benchmarks...")

    print("  bench_basis_change ...", end=" ", flush=True)
    r = bench_basis_change()
    results.append(r)
    print(f"SF={r.sf_time_s*1000:.1f}ms  Qiskit={r.qk_time_s*1000:.1f}ms  winner={r.winner} ({r.ratio})")

    # ── Transpilation ─────────────────────────────────────
    print("\n▸ Transpilation benchmarks (100Q heavy-hex-127, basis={rz,sx,x,ecr})...")

    tests = [
        ("bench_qft_transpile",            build_qft_sf,            build_qft_qk,            N_QUBITS),
        ("bench_qv_transpile",             build_qv_sf,             build_qv_qk,             N_QUBITS),
        ("bench_su2_transpile",            build_su2_sf,            build_su2_qk,             N_QUBITS),
        ("bench_bv_transpile",             build_bv_sf,             build_bv_qk,              N_QUBITS),
        ("bench_heisenberg_transpile",     build_heisenberg_sf,     build_heisenberg_qk,      N_QUBITS),
        ("bench_qaoa_transpile",           build_qaoa_sf,           build_qaoa_qk,            N_QUBITS),
        ("bench_simplification_transpile", build_simplification_sf, build_simplification_qk,  N_QUBITS),
        ("bench_clifford_transpile",       build_clifford_sf,       build_clifford_qk,        N_QUBITS),
    ]

    for name, sf_b, qk_b, n in tests:
        print(f"  {name} ...", end=" ", flush=True)
        r = _transpile_test(name, sf_b, qk_b, n, "ibm_eagle", edges_127, native_ecr)
        results.append(r)

        sf_str = f"SF={r.sf_time_s:.2f}s" if r.sf_time_s > 0 else f"SF=FAIL"
        qk_str = f"Qk={r.qk_time_s:.2f}s" if r.qk_time_s > 0 else f"Qk=FAIL"

        sf_2q = r.sf_extra.get("gate_count_2q", "?")
        qk_2q = r.qk_extra.get("gate_count_2q", "?")
        sf_err = r.sf_extra.get("error", "")
        qk_err = r.qk_extra.get("error", "")

        detail = f"  winner={r.winner} ({r.ratio})  2Q: SF={sf_2q} Qk={qk_2q}"
        if sf_err:
            detail += f"  [SF err: {sf_err[:60]}]"
        if qk_err:
            detail += f"  [Qk err: {qk_err[:60]}]"
        print(f"{sf_str}  {qk_str}{detail}")

    # ── Summary table ─────────────────────────────────────
    print("\n" + "=" * 72)
    print("  SUMMARY TABLE")
    print("=" * 72)
    print(f"{'Test':<35} {'SF Time':>10} {'Qk Time':>10} {'Winner':>8} {'Ratio':>8}  {'SF 2Q':>7} {'Qk 2Q':>7}  {'2Q Ratio':>9}")
    print("-" * 110)
    for r in results:
        sf_t = f"{r.sf_time_s*1000:.0f}ms" if r.sf_time_s > 0 else "FAIL"
        qk_t = f"{r.qk_time_s*1000:.0f}ms" if r.qk_time_s > 0 else "FAIL"
        sf_2q = r.sf_extra.get("gate_count_2q", "")
        qk_2q = r.qk_extra.get("gate_count_2q", "")
        q_ratio = ""
        if isinstance(sf_2q, int) and isinstance(qk_2q, int) and qk_2q > 0:
            q_ratio = f"{sf_2q/qk_2q:.2f}x"
        print(f"{r.test:<35} {sf_t:>10} {qk_t:>10} {r.winner:>8} {r.ratio:>8}  {str(sf_2q):>7} {str(qk_2q):>7}  {q_ratio:>9}")

    # ── Comparison with previous ──────────────────────────
    print("\n" + "=" * 72)
    print("  BEFORE vs AFTER (key gaps)")
    print("=" * 72)

    old_data = {
        "bench_basis_change":        {"sf": 82.3, "qk": 20.2, "winner": "Qiskit", "ratio": "4.1x"},
        "bench_clifford_transpile":  {"sf": -1,   "qk": 16934, "winner": "Qiskit", "ratio": "FAIL"},
        "bench_qft_transpile":       {"sf_2q": 35892, "qk_2q": 8018},
        "bench_heisenberg_transpile":{"sf_2q": 858,   "qk_2q": 528},
    }

    for r in results:
        if r.test in old_data:
            old = old_data[r.test]
            if "sf" in old:
                old_sf = f"{old['sf']:.0f}ms" if old['sf'] > 0 else "FAIL"
                new_sf = f"{r.sf_time_s*1000:.0f}ms" if r.sf_time_s > 0 else "FAIL"
                print(f"  {r.test}:")
                print(f"    BEFORE: SF={old_sf}, winner={old['winner']} ({old['ratio']})")
                print(f"    AFTER:  SF={new_sf}, winner={r.winner} ({r.ratio})")
            if "sf_2q" in old:
                new_sf_2q = r.sf_extra.get("gate_count_2q", "?")
                print(f"  {r.test} (2Q gate quality):")
                print(f"    BEFORE: SF={old['sf_2q']} Qk={old['qk_2q']} (ratio={old['sf_2q']/old['qk_2q']:.1f}x)")
                print(f"    AFTER:  SF={new_sf_2q} Qk={r.qk_extra.get('gate_count_2q', '?')}")

    # ── Save results ──────────────────────────────────────
    out = []
    for r in results:
        out.append({
            "category": r.category,
            "test": r.test,
            "sf_time_s": r.sf_time_s,
            "qk_time_s": r.qk_time_s,
            "winner": r.winner,
            "ratio": r.ratio,
            "sf_extra": r.sf_extra,
            "qk_extra": r.qk_extra,
        })
    with open("benchpress_post_gaps.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to benchpress_post_gaps.json")
