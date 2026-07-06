"""
Claim Verification v2 — ALL BACKENDS
"""
import time, gc, tracemalloc, sys, os, traceback
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import superfermion as sf
from superfermion.backends.registry import BackendRegistry

try:
    import qiskit
    from qiskit_aer import AerSimulator
    from qiskit import transpile
    HAS_QISKIT = True
except ImportError:
    HAS_QISKIT = False

SEED = 12345
ALL_BACKENDS = ["statevector","rust","jax","singularity","supremacy","mps","jax_mps","stabilizer","density_matrix"]

def bench_lat(fn, nw=2, nr=5):
    for _ in range(nw): fn()
    ts=[]
    for _ in range(nr):
        t0=time.perf_counter(); fn(); ts.append((time.perf_counter()-t0)*1000)
    return np.mean(ts),np.min(ts),np.max(ts)

def bench_mem(fn, nw=1, nr=3):
    for _ in range(nw): fn()
    gc.collect(); tracemalloc.start()
    for _ in range(nr): fn()
    _,p=tracemalloc.get_traced_memory(); tracemalloc.stop()
    return p/(1024*1024)

def probe():
    c=sf.Circuit(2).h(0).cx(0,1); w=[]
    for n in ALL_BACKENDS:
        try:
            BackendRegistry.get_backend(n); sf.run(c,backend=n,shots=128); w.append(n)
        except: pass
    return w

def build_qv_sf(n,d,seed=SEED):
    rng=np.random.default_rng(seed); c=sf.Circuit(n)
    for _ in range(d):
        p=rng.permutation(n)
        for i in range(0,n-1,2):
            q0,q1=int(p[i]),int(p[i+1]); a=rng.uniform(0,2*np.pi,4)
            c.ry(float(a[0]),q0).ry(float(a[1]),q1).cx(q0,q1).ry(float(a[2]),q0).ry(float(a[3]),q1)
    return c

def build_qv_qk(n,d,seed=SEED):
    from qiskit import QuantumCircuit
    rng=np.random.default_rng(seed); qc=QuantumCircuit(n)
    for _ in range(d):
        p=rng.permutation(n)
        for i in range(0,n-1,2):
            q0,q1=int(p[i]),int(p[i+1]); a=rng.uniform(0,2*np.pi,4)
            qc.ry(float(a[0]),q0); qc.ry(float(a[1]),q1); qc.cx(q0,q1); qc.ry(float(a[2]),q0); qc.ry(float(a[3]),q1)
    return qc

def build_cliff_sf(n,seed=SEED):
    from superfermion.circuit import GateRecord
    rng=np.random.default_rng(seed); c=sf.Circuit(n); ng=10*n*n
    G=["CX","CZ","CY","SWAP","X","Y","Z","S","SDG","H"]
    ch=rng.choice(len(G),size=ng); tw=ch<4; ntw=int(tw.sum())
    qa=rng.integers(0,n,size=ntw,dtype=np.int64); qb=rng.integers(0,n-1,size=ntw,dtype=np.int64)
    qb[qb>=qa]+=1; noq=ng-ntw; q1=rng.integers(0,n,size=noq,dtype=np.int64)
    recs=[None]*ng; ti=oi=0
    for i in range(ng):
        gi=ch[i]; gn=G[gi]
        if gi<4: recs[i]=GateRecord(name=gn,qubits=[int(qa[ti]),int(qb[ti])]); ti+=1
        else: recs[i]=GateRecord(name=gn,qubits=[int(q1[oi])]); oi+=1
    c._gates.extend(recs); return c

def gen_qv100_qasm():
    from qiskit.circuit.library import quantum_volume; from qiskit.qasm2 import dumps
    qc=quantum_volume(100,100,seed=SEED); qc2=transpile(qc,basis_gates=['rx','ry','rz','cx'])
    return dumps(qc2)

def qasm2sf(q): 
    from superfermion.bridge import from_qasm; return from_qasm(q)

def main():
    print("="*80); print("  CLAIM VERIFICATION v2: ALL BACKENDS"); print("="*80)
    print(f"  Qiskit: {HAS_QISKIT}  SF: {sf.__version__}")
    W=probe(); print(f"  Working backends ({len(W)}): {W}")
    R={}

    # A1 QV100 memory
    print("\n"+"-"*80+"\nA1. QV100 CIRCUIT MEMORY (claim 67.7x)")
    msf=bench_mem(lambda: build_qv_sf(100,100))
    print(f"  SF: {msf:.2f} MB")
    if HAS_QISKIT:
        from qiskit.circuit.library import quantum_volume
        mqk=bench_mem(lambda: quantum_volume(100,100,seed=SEED))
        r1=msf/mqk if mqk>0 else 999
        print(f"  Qiskit: {mqk:.2f} MB  RATIO: {r1:.1f}x")
        R["A1"]={"sf":msf,"qk":mqk,"r":r1}

    # A2 DTC100 twirl memory
    print("\n"+"-"*80+"\nA2. DTC100 TWIRLING MEMORY (claim 17.0x)")
    try:
        from superfermion.compiler.advanced import PauliTwirlingPass
        if HAS_QISKIT:
            qa=gen_qv100_qasm()
            def sf_tw():
                c=qasm2sf(qa); return PauliTwirlingPass(seed=SEED).run(c)
            def qk_tw():
                from qiskit.circuit import pauli_twirl_2q_gates; from qiskit.qasm2 import loads
                return pauli_twirl_2q_gates(loads(qa))
            ms=bench_mem(sf_tw); mq=bench_mem(qk_tw)
            r2=ms/mq if mq>0 else 999
            print(f"  SF: {ms:.2f} MB  Qiskit: {mq:.2f} MB  RATIO: {r2:.1f}x")
            R["A2"]={"sf":ms,"qk":mq,"r":r2}
    except Exception as e:
        print(f"  ERR: {e}"); R["A2"]={"err":str(e)}

    # A3 Clifford decompose memory
    print("\n"+"-"*80+"\nA3. CLIFFORD DECOMPOSE MEMORY (claim 16.0x)")
    try:
        from superfermion.runtime.specs import HardwareSpec
        from superfermion.compiler.rust_bridge import compile_rust
        from superfermion.backends.stabilizer import simplify_clifford
        csf=build_cliff_sf(20); simp=simplify_clifford(csf)
        sp=HardwareSpec(name="c",n_qubits=20,native_gates=["rz","sx","x","cz"],coupling_map=[])
        def sf_cd(): return compile_rust(simp,level=1,target=sp,pre_simplified=True)
        ms=bench_mem(sf_cd); print(f"  SF: {ms:.2f} MB")
        if HAS_QISKIT:
            from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
            from qiskit.passmanager import PropertySet
            from qiskit.quantum_info import Clifford
            from qiskit.circuit.random import random_clifford_circuit
            cq=random_clifford_circuit(20,gates=["cx","cz","cy","swap","x","y","z","s","sdg","h"],num_gates=10*20*20,seed=SEED)
            co=Clifford(cq); cc=co.to_circuit()
            tr=generate_preset_pass_manager(1,basis_gates=["rz","sx","x","cz"]).translation
            def qk_cd():
                tr.property_set=PropertySet(); return tr.run(cc)
            mq=bench_mem(qk_cd); r3=ms/mq if mq>0 else 999
            print(f"  Qiskit: {mq:.2f} MB  RATIO: {r3:.1f}x")
            R["A3"]={"sf":ms,"qk":mq,"r":r3}
    except Exception as e:
        print(f"  ERR: {e}"); traceback.print_exc(); R["A3"]={"err":str(e)}

    # A4 DTC100 twirl latency
    print("\n"+"-"*80+"\nA4. DTC100 TWIRLING LATENCY (claim 3.5x)")
    try:
        from superfermion.compiler.advanced import PauliTwirlingPass
        if HAS_QISKIT:
            qa=gen_qv100_qasm(); csf=qasm2sf(qa)
            from qiskit.circuit import pauli_twirl_2q_gates; from qiskit.qasm2 import loads
            cqk=loads(qa)
            ls,_,_=bench_lat(lambda: PauliTwirlingPass(seed=SEED).run(csf))
            lq,_,_=bench_lat(lambda: pauli_twirl_2q_gates(cqk))
            r4=ls/lq if lq>0 else 999
            print(f"  SF: {ls:.1f}ms  Qiskit: {lq:.1f}ms  RATIO: {r4:.1f}x")
            R["A4"]={"sf":ls,"qk":lq,"r":r4}
    except Exception as e:
        print(f"  ERR: {e}"); R["A4"]={"err":str(e)}

    # A5 QV100 basis change
    print("\n"+"-"*80+"\nA5. QV100 BASIS CHANGE (claim 1.5x)")
    try:
        from superfermion.runtime.specs import HardwareSpec
        from superfermion.compiler.rust_bridge import compile_rust
        if HAS_QISKIT:
            qa=gen_qv100_qasm(); csf=qasm2sf(qa)
            sp=HardwareSpec(name="q",n_qubits=csf.n_qubits,native_gates=["sx","x","rz","cz"],coupling_map=[])
            from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
            from qiskit.passmanager import PropertySet; from qiskit.qasm2 import loads
            cqk=loads(qa)
            tr=generate_preset_pass_manager(1,basis_gates=["sx","x","rz","cz"]).translation
            ls,_,_=bench_lat(lambda: compile_rust(csf,level=1,target=sp))
            def _qb():
                tr.property_set=PropertySet(); return tr.run(cqk)
            lq,_,_=bench_lat(_qb)
            r5=ls/lq if lq>0 else 999
            print(f"  SF: {ls:.1f}ms  Qiskit: {lq:.1f}ms  RATIO: {r5:.1f}x")
            R["A5"]={"sf":ls,"qk":lq,"r":r5}
    except Exception as e:
        print(f"  ERR: {e}"); R["A5"]={"err":str(e)}

    # A6 QV100 build
    print("\n"+"-"*80+"\nA6. QV100 CIRCUIT BUILD (claim 1.2x)")
    ls,_,_=bench_lat(lambda: build_qv_sf(100,100))
    print(f"  SF: {ls:.1f}ms")
    if HAS_QISKIT:
        lq,_,_=bench_lat(lambda: build_qv_qk(100,100))
        r6=ls/lq if lq>0 else 999
        print(f"  Qiskit: {lq:.1f}ms  RATIO: {r6:.1f}x")
        R["A6"]={"sf":ls,"qk":lq,"r":r6}

    # B1 QV sim ALL BACKENDS
    print("\n"+"="*80+"\nB1. QV SIMULATION ALL BACKENDS (claim 1.7-3.2x)")
    print("="*80)
    R["B1"]={}
    bk=[b for b in ALL_BACKENDS if b in W]
    print(f"  Backends: {bk}")
    for n in [10,12,14]:
        print(f"\n  n={n}:")
        csf=build_qv_sf(n,n)
        st={}
        for b in bk:
            if b in ("density_matrix",) and n>=12: print(f"    {b:14s} SKIP (DM too large for {n}q)"); continue
            if b in ("supremacy","jax","jax_mps") and n>=14: print(f"    {b:14s} SKIP (too slow for {n}q)"); continue
            try:
                l,_,_=bench_lat(lambda b=b: sf.run(csf,backend=b,shots=1024),nw=1,nr=3)
                st[b]=l; print(f"    {b:14s} {l:.1f}ms")
            except Exception as e: print(f"    {b:14s} ERR: {str(e)[:50]}")
        if HAS_QISKIT:
            qc=build_qv_qk(n,n); qc.measure_all()
            sim=AerSimulator(method="statevector"); tqc=transpile(qc,sim)
            try:
                lq,_,_=bench_lat(lambda: sim.run(tqc,shots=1024).result(),nw=1,nr=3)
                print(f"    {'Qiskit':14s} {lq:.1f}ms")
                for b,sm in st.items():
                    rr=sm/lq if lq>0 else 999
                    R["B1"][f"n{n}_{b}"]={"sf":sm,"qk":lq,"r":rr}
                    print(f"    -> {b}/Qiskit: {rr:.1f}x")
            except Exception as e: print(f"    Qiskit ERR: {str(e)[:50]}")

    # B2 MCX sim ALL BACKENDS
    print("\n"+"="*80+"\nB2. MCX SIMULATION ALL BACKENDS (claim 37x at n=12, crash n=16)")
    print("="*80)
    R["B2"]={}
    from tests.benchpress.conftest import build_multi_control_circuit_sf, build_multi_control_circuit_qiskit
    try:
        from superfermion._sf_core import QuantumDAG
        hm=hasattr(QuantumDAG,'simulate_mps')
        print(f"  QuantumDAG.simulate_mps: {hm}")
        if not hm:
            ms=[m for m in dir(QuantumDAG) if not m.startswith('__')]
            print(f"  Methods: {ms[:15]}")
    except Exception as e: print(f"  DAG probe: {e}")

    for n in [10, 12, 14, 16]:
        print(f"\n  n={n}:")
        try:
            csf=build_multi_control_circuit_sf(n)
            print(f"    circuit: {csf.n_qubits}q {csf.gate_count}g")
            st={}
            for b in bk:
                if b=="density_matrix": print(f"    {b:14s} SKIP (OOM on MCX circuits)"); continue
                if b in ("supremacy","jax","jax_mps","mps"): print(f"    {b:14s} SKIP (MPS saturates on MCX {csf.n_qubits}q)"); continue
                try:
                    l,_,_=bench_lat(lambda b=b: sf.run(csf,backend=b,shots=1024),nw=1,nr=2)
                    st[b]=l; print(f"    {b:14s} {l:.1f}ms")
                except Exception as e: print(f"    {b:14s} ERR: {str(e)[:60]}"); R["B2"][f"n{n}_{b}"]={"err":str(e)[:60]}
        except Exception as e: print(f"    build ERR: {e}"); st={}
        if HAS_QISKIT:
            try:
                cqk=build_multi_control_circuit_qiskit(n); cqk.measure_all()
                sim=AerSimulator(method="statevector"); tqc=transpile(cqk,sim)
                lq,_,_=bench_lat(lambda: sim.run(tqc,shots=1024).result(),nw=1,nr=2)
                print(f"    {'Qiskit':14s} {lq:.1f}ms")
                R["B2"][f"n{n}_qiskit"]={"qk":lq}
                for b,sm in st.items():
                    rr=sm/lq if lq>0 else 999
                    R["B2"][f"n{n}_{b}"]={"sf":sm,"qk":lq,"r":rr}
                    print(f"    -> {b}/Qiskit: {rr:.1f}x")
            except Exception as e: print(f"    Qiskit ERR: {str(e)[:60]}")

    # SUMMARY
    print("\n\n"+"="*80+"\n  SUMMARY\n"+"="*80)
    print("\nCLAIM SET A:")
    for k,lbl,cl in [("A1","QV100 memory","67.7x"),("A2","DTC100 twirl mem","17.0x"),
                       ("A3","Cliff decomp mem","16.0x"),("A4","DTC100 twirl lat","3.5x"),
                       ("A5","QV100 basis","1.5x"),("A6","QV100 build","1.2x")]:
        d=R.get(k,{})
        if "r" in d: print(f"  {lbl:20s} claim={cl:8s} actual={d['r']:.1f}x")
        elif "err" in d: print(f"  {lbl:20s} ERR")
        else: print(f"  {lbl:20s} N/A")

    print("\nCLAIM SET B (all backends):")
    for sec,cl in [("B1","1.7-3.2x"),("B2","37x")]:
        d=R.get(sec,{})
        for k2,v2 in sorted(d.items()):
            if "r" in v2: print(f"  {sec} {k2:25s} claim={cl:8s} actual={v2['r']:.1f}x")
            elif "err" in v2: print(f"  {sec} {k2:25s} ERR: {v2['err'][:40]}")

    import json
    safe={}
    for k,v in R.items():
        if isinstance(v,dict):
            safe[k]={sk:(sv if isinstance(sv,(int,float,str,bool,type(None))) else str(sv)) for sk,sv in v.items()}
        else: safe[k]=str(v)
    print("\n"+json.dumps(safe,indent=2))

if __name__=="__main__":
    main()
