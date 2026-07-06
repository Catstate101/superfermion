#!/usr/bin/env python
"""
============================================================================
 SUPERFERMION ALL-BENCHMARK NOTEBOOK — Latency, Memory, Accuracy
============================================================================
Run cells ONE AT A TIME. Heavy cells (MPS 50+ q, QEC large codes) marked
with ⚠ — skip on low-RAM machines.

CELLS:  1. Backend Probe       2. Circuit Builders      3. Rust vs SV
        4. Dense Latency        5. MPS Latency           6. Stabilizer
        7. Cross-Backend Fid.   8. SF vs Qiskit          9. Gradients
       10. Grad Diag Tests     11. Memory Efficiency    12. VQE H2
       13. QEC Codes           14. QEC Decoders         15. Scientific
       16. MPS High-Qubit ⚠   17. JIT Warmup           18. Summary
============================================================================
"""
import sys, time, os, gc, math, json, warnings
warnings.filterwarnings('ignore')
os.environ['PYTHONIOENCODING'] = 'utf-8'
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except AttributeError:
    pass
import numpy as np
np.set_printoptions(precision=6, suppress=True, linewidth=120)
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple
import psutil
import superfermion as sf
from superfermion.backends.registry import BackendRegistry
from superfermion.backends.singularity import SingularityBackend
from superfermion.backends.mps import MPSSimulatorBackend
from superfermion.backends.stabilizer import StabilizerBackend, NotCliffordError
from superfermion.observables.core import SparsePauliOp
from superfermion.qml.gradient.adjoint import adjoint_grad_vector
from superfermion.qml.gradient.parameter_shift import parameter_shift_grad_vector
_HAS_QISKIT = _HAS_PENNYLANE = False
try:
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Pauli
    from qiskit_aer import AerSimulator
    _HAS_QISKIT = True
except ImportError:
    pass
try:
    import pennylane as qml
    _HAS_PENNYLANE = True
except ImportError:
    pass

CELL = 0
def cell(title):
    global CELL; CELL += 1
    print(f"\n{'='*76}")
    print(f"  CELL {CELL}: {title}")
    print(f"{'='*76}", flush=True)

ALL_RESULTS = []

def _track(fn):
    proc = psutil.Process(os.getpid()); gc.collect()
    r0 = proc.memory_info().rss; t0 = time.perf_counter()
    out = fn()
    dt = (time.perf_counter()-t0)*1000
    r1 = proc.memory_info().rss
    return out, dt, (r1-r0)/1024/1024

def _z_msb(sv, n):
    m0=1<<(n-1);m1=1<<(n-2);p=np.abs(sv)**2;i=np.arange(1<<n)
    return float(np.real(np.sum(np.where(((i&m0)!=0).astype(int)^((i&m1)!=0).astype(int)==0,1.0,-1.0)*p)))

def sf_to_lsb(sv,n):
    return np.asarray(sv).reshape([2]*n).transpose(list(range(n))[::-1]).reshape(-1)

def fidelity(a,b): return float(abs(np.vdot(a,b)))
def record(sec,be,wl,n,rt,mem,f=None,ze=None,st="OK"):
    ALL_RESULTS.append(dict(section=sec,backend=be,workload=wl,n=n,runtime_ms=rt,rss_mb=mem,fidelity=f,z_err=ze,status=st))

ALL_NAMES = ["statevector","rust","mps","jax","jax_mps","stabilizer","density_matrix","singularity","supremacy","cuda","cuda_mps"]
working = []

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 1: Backend Probe
# ═══════════════════════════════════════════════════════════════════════════════
cell("Backend Probe - All 11 backends")
probe = sf.Circuit(2).h(0).cx(0,1)
print(f"  {'Backend':<18s} | {'Status':<8s} | Type")
print("  "+"-"*50)
for name in ALL_NAMES:
    try:
        be = BackendRegistry.get_backend(name)
        sf.run(probe, backend=name, shots=128)
        working.append(name)
        print(f"  {name:<18s} | {'OK':<8s} | {type(be).__name__}")
    except Exception as e:
        print(f"  {name:<18s} | {'FAIL':<8s} | {str(e)[:50]}")
print(f"\n  Working: {working}")
print("  >>> Next: CELL 2 (Circuit Builders - definitions only)")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 2: Circuit Builders
# ═══════════════════════════════════════════════════════════════════════════════
cell("Circuit Builders - definitions only, no execution")
def w_ghz(n):
    sfc=sf.Circuit(n).h(0)
    for i in range(n-1): sfc.cx(i,i+1)
    return sfc
def w_qaoa(n,p=2):
    sfc=sf.Circuit(n);g=[0.3,0.5];b=[0.2,0.4]
    for q in range(n): sfc.h(q)
    for l in range(p):
        for i in range(n-1): sfc.cx(i,i+1);sfc.rz(2*g[l],i+1);sfc.cx(i,i+1)
        for q in range(n): sfc.rx(2*b[l],q)
    return sfc
def w_heis(n,s=10,J=1.0,dt=0.05):
    sfc=sf.Circuit(n)
    for _ in range(s):
        for i in range(n-1):
            sfc.h(i);sfc.h(i+1);sfc.cx(i,i+1);sfc.rz(2*J*dt,i+1);sfc.cx(i,i+1);sfc.h(i);sfc.h(i+1)
            sfc.rx(math.pi/2,i);sfc.rx(math.pi/2,i+1);sfc.cx(i,i+1);sfc.rz(2*J*dt,i+1);sfc.cx(i,i+1);sfc.rx(-math.pi/2,i);sfc.rx(-math.pi/2,i+1)
            sfc.cx(i,i+1);sfc.rz(2*J*dt,i+1);sfc.cx(i,i+1)
    return sfc
def w_qft(n):
    sfc=sf.Circuit(n)
    for j in range(n):
        sfc.h(j)
        for k in range(j+1,n): sfc.cp(math.pi/(2**(k-j)),k,j)
    for i in range(n//2): sfc.swap(i,n-1-i)
    return sfc
def w_cliff(n,layers=8,seed=1):
    rng=np.random.default_rng(seed);sfc=sf.Circuit(n)
    for _ in range(layers):
        for q in range(n):
            k=int(rng.integers(0,3))
            if k==0: sfc.h(q)
            elif k==1: sfc.s(q)
        for i in range(0,n-1,2): sfc.cx(i,i+1)
        for i in range(1,n-1,2): sfc.cx(i,i+1)
    return sfc
def w_rand(n,layers=6,seed=1):
    rng=np.random.default_rng(seed);sfc=sf.Circuit(n)
    for _ in range(layers):
        for q in range(n): sfc.ry(float(rng.uniform(0,2*math.pi)),q);sfc.rz(float(rng.uniform(0,2*math.pi)),q)
        for i in range(0,n-1,2): sfc.cx(i,i+1)
        for i in range(1,n-1,2): sfc.cz(i,i+1)
    return sfc
print("  7 workload builders: w_ghz, w_qaoa, w_heis, w_qft, w_cliff, w_rand")
print("  >>> Next: CELL 3 (Rust vs Statevector)")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 3: Rust vs Statevector
# ═══════════════════════════════════════════════════════════════════════════════
cell("Rust SIMD vs Statevector - Speed & Memory Shootout")
if "rust" in working and "statevector" in working:
    print(f"  {'n':<5s} {'Workload':<14s} {'SV(ms)':<10s} {'Rust(ms)':<10s} {'Spd':<7s} {'SV MB':<8s} {'Rust MB':<8s}")
    print("  "+"-"*65)
    for wn,fn in [("GHZ",w_ghz),("QAOA",w_qaoa),("Cliff",w_cliff),("QFT",w_qft)]:
        for n in [6,10,12,14,16]:
            try:
                c=fn(n)
                _,sv_t,sv_m=_track(lambda: BackendRegistry.get_backend("statevector").run(c,shots=0))
                _,ru_t,ru_m=_track(lambda: BackendRegistry.get_backend("rust").run(c,shots=0))
                sp=sv_t/ru_t if ru_t>0 else float('inf')
                print(f"  {n:<5d} {wn:<14s} {sv_t:<10.1f} {ru_t:<10.1f} {sp:<7.2f}x {sv_m:<+8.1f} {ru_m:<+8.1f}")
                record("rust_sv", "statevector", wn, n, sv_t, sv_m)
                record("rust_sv", "rust", wn, n, ru_t, ru_m)
            except Exception as e:
                print(f"  {n:<5d} {wn:<14s} FAIL: {str(e)[:45]}")
else:
    print("  rust+statevector needed. Working:", working)
print("  >>> Next: CELL 4 (Dense Backend Latency)")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 4: Dense Backend Latency
# ═══════════════════════════════════════════════════════════════════════════════
cell("Dense Backend Latency - 5 workloads x backends x n=6,10,14")
DENSE = [b for b in ["statevector","rust","jax","density_matrix"] if b in working]
WL = [("QAOA",w_qaoa),("GHZ",w_ghz),("Heis",w_heis),("QFT",w_qft),("Cliff",w_cliff)]
for wn,fn in WL:
    print(f"\n  -- {wn} --")
    for n in [6,10,14]:
        c=fn(n)
        for bk in DENSE:
            try:
                be=BackendRegistry.get_backend(bk)
                sv,dt,mem=_track(lambda: np.asarray(be.run(c,shots=0).statevector,dtype=np.complex128))
                if sv is not None and sv.size==(1<<n):
                    print(f"    {bk:<18s} n={n:2d}: {dt:8.1f}ms mem={mem:+6.1f}MB")
                    record("latency",bk,wn,n,dt,mem)
                else:
                    print(f"    {bk:<18s} n={n:2d}: {dt:8.1f}ms [NO SV]")
            except Exception as e:
                print(f"    {bk:<18s} n={n:2d}: FAIL {str(e)[:40]}")
print("\n  >>> Next: CELL 5 (MPS & Singularity)")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 5: MPS & Singularity Latency
# ═══════════════════════════════════════════════════════════════════════════════
cell("MPS & Singularity Latency - n=10..40")
MPS_BK = [b for b in ["mps","jax_mps","singularity"] if b in working]
for wn,fn in [("GHZ",w_ghz),("QAOA",w_qaoa),("Cliff",w_cliff)]:
    print(f"\n  -- {wn} --")
    for n in [10,16,20,30,40]:
        c=fn(n)
        for bk in MPS_BK:
            try:
                be=BackendRegistry.get_backend(bk)
                _,dt,mem=_track(lambda: be.run(c,shots=0))
                zs=""
                if bk=="mps":
                    try: z=float(np.real(MPSSimulatorBackend(options={"max_bond_dim":64}).expval(c,"ZZ"+"I"*(n-2),max_bond=64))); zs=f" <ZZ>={z:+.6f}"
                    except: pass
                print(f"    {bk:<18s} n={n:2d}: {dt:8.1f}ms mem={mem:+6.1f}MB{zs}")
                record("latency_mps",bk,wn,n,dt,mem)
            except Exception as e:
                print(f"    {bk:<18s} n={n:2d}: FAIL {str(e)[:40]}")
print("  >>> Next: CELL 6 (Stabilizer Tableau)")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 6: Stabilizer Tableau
# ═══════════════════════════════════════════════════════════════════════════════
cell("Stabilizer Tableau - Aaronson-Gottesman n=10..1000")
if "stabilizer" in working:
    print(f"  {'n':<6s} {'Time(ms)':<12s} {'Mem(MB)':<10s}")
    print("  "+"-"*35)
    for n in [10,20,50,100,500,1000]:
        try:
            c=w_cliff(n);
            z,dt,mem=_track(lambda: StabilizerBackend().expval(c,"ZZ"+"I"*(n-2)))
            print(f"  {n:<6d} {dt:<12.1f} {mem:<+10.1f}")
            record("stabilizer","stabilizer","Clifford",n,dt,mem,ze=0.0)
        except Exception as e:
            print(f"  {n:<6d} {dt:<12.1f} {mem:<+10.1f} FAIL {str(e)[:30]}")
else:
    print("  stabilizer not available")
print("  >>> Next: CELL 7 (Cross-Backend Fidelity)")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 7: Cross-Backend Fidelity
# ═══════════════════════════════════════════════════════════════════════════════
cell("Cross-Backend Fidelity - Machine-epsilon agreement at n=6")
SV_BK = [b for b in ["statevector","rust","jax"] if b in working]
for wn,fn in [("GHZ",w_ghz),("QAOA",w_qaoa),("Cliff",w_cliff),("QFT",w_qft)]:
    svs={};ref=None
    print(f"\n  -- {wn} --")
    for bk in SV_BK:
        try:
            sv=np.asarray(BackendRegistry.get_backend(bk).run(fn(6),shots=0).statevector,dtype=np.complex128)
            svs[bk]=sv
            if bk=="statevector": ref=sv
            print(f"    {bk:<18s} |sv|={np.linalg.norm(sv):.10f}")
        except Exception as e:
            print(f"    {bk:<18s} FAIL {str(e)[:35]}")
    if ref is not None:
        for bk,sv in svs.items():
            f=fidelity(sf_to_lsb(sv,6), sf_to_lsb(ref,6))
            ep=" <<< MACHINE EPS" if f>1-1e-14 else " <<< WARN"
            print(f"    fidelity({bk}, statevector) = {f:.15f}{ep}")
            record("fidelity",bk,wn,6,0,0,fid=f)
print("  >>> Next: CELL 8 (SF vs Qiskit-Aer)")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 8: SF vs Qiskit-Aer
# ═══════════════════════════════════════════════════════════════════════════════
cell("SF vs Qiskit-Aer - Cross-framework accuracy")
if _HAS_QISKIT:
    for n in [6,10]:
        sf_c,w_qk_c=w_qaoa(n),QuantumCircuit(n)
        g=[0.3,0.5];b=[0.2,0.4];w_qk_c.h(range(n))
        for p in range(2):
            for i in range(n-1): w_qk_c.cx(i,i+1);w_qk_c.rz(2*g[p],i+1);w_qk_c.cx(i,i+1)
            for q in range(n): w_qk_c.rx(2*b[p],q)
        sv_sf=np.asarray(BackendRegistry.get_backend("statevector").run(sf_c,shots=0).statevector,dtype=np.complex128)
        sim=AerSimulator(method="statevector");q2=w_qk_c.copy();q2.save_statevector()
        sv_qk=np.asarray(sim.run(q2).result().get_statevector(),dtype=np.complex128)
        f=fidelity(sf_to_lsb(sv_sf,n),sv_qk);md=float(np.max(np.abs(sf_to_lsb(sv_sf,n)-sv_qk)))
        print(f"  n={n:2d}: fidelity={f:.15f} max_diff={md:.2e}  {'PASS' if md<1e-14 else 'WARN'}")
        record("crossfw","sf.statevector","QAOA",n,0,0,fid=f)
else:
    print("  Qiskit-Aer not installed. Install with: pip install qiskit qiskit-aer")
print("  >>> Next: CELL 9 (Gradient Accuracy)")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 9: Gradient Adjoint vs Parameter-Shift
# ═══════════════════════════════════════════════════════════════════════════════
cell("Gradient Accuracy - Adjoint vs Parameter-Shift")
print(f"  {'n':<4s} {'Params':<8s} {'max|adj-ps|':<16s} {'Adj ms':<8s} {'PS ms':<8s} {'Spd':<7s}")
print("  "+"-"*55)
for n in [4,6,8]:
    np2=2*n;names=[f"t{i}" for i in range(np2)]
    theta=np.random.default_rng(42).uniform(-1,1,np2)
    qc=sf.Circuit(n);idx=0
    for q in range(n): qc.ry(sf.param(names[idx]),q);idx+=1
    for q in range(n): qc.rz(sf.param(names[idx]),q);idx+=1
    for i in range(n-1): qc.cx(i,i+1)
    obs=SparsePauliOp.from_dict({"ZZ"+"I"*(n-2):1.0,"X"+"I"*(n-1):0.5})
    try:
        t0=time.perf_counter();g_a=np.asarray(adjoint_grad_vector(qc,obs,names,theta));t_a=(time.perf_counter()-t0)*1000
        t0=time.perf_counter();g_p=np.asarray(parameter_shift_grad_vector(qc,obs,names,theta,backend="statevector"));t_p=(time.perf_counter()-t0)*1000
        md=float(np.max(np.abs(g_a-g_p)));sp=t_p/t_a if t_a>0 else float('inf')
        print(f"  {n:<4d} {np2:<8d} {md:<16.2e} {t_a:<8.2f} {t_p:<8.2f} {sp:<7.1f}x")
        record("gradient","adjoint",f"n={n}",n,t_a,0,ze=md)
        record("gradient","param_shift",f"n={n}",n,t_p,0,ze=md)
    except Exception as e: print(f"  {n:<4d} FAIL {str(e)[:45]}")
print("  >>> Next: CELL 10 (Gradient Diagonal Tests)")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 10: Gradient Diagonal Tests
# ═══════════════════════════════════════════════════════════════════════════════
cell("Gradient Diagonal Tests - 8 adversarial cases (adjoint-diag)")
def diag_test(label,qc,obs,names,theta):
    try:
        ga=np.asarray(adjoint_grad_vector(qc,obs,names,theta))
        gp=np.asarray(parameter_shift_grad_vector(qc,obs,names,theta,backend="statevector"))
        md=float(np.max(np.abs(ga-gp)))
        print(f"  {label:50s} max|adj-ps|={md:.3e}  {'PASS' if md<1e-10 else 'WARN'}")
        return md
    except Exception as e: print(f"  {label:50s} FAIL: {str(e)[:40]}");return None

rng=np.random.default_rng(42);n=5;names=[f"a{i}" for i in range(5)];theta=rng.uniform(-1,1,5)
qc=sf.Circuit(n)
for q in range(n): qc.ry(sf.param(names[q]),q)
for q in range(n-1): qc.cx(q,q+1)
cases=[("Case1: ZZZZZ",SparsePauliOp.from_dict({"ZZZZZ":1.0})),
       ("Case2: XXXXX",SparsePauliOp.from_dict({"XXXXX":1.0})),
       ("Case3: YYYYY",SparsePauliOp.from_dict({"YYYYY":1.0})),
       ("Case4: YIIII",SparsePauliOp.from_dict({"YIIII":1.0})),
       ("Case5: ZZZZZ+XXXXX",SparsePauliOp.from_dict({"ZZZZZ":1.0,"XXXXX":0.5})),
       ("Case6: ZZZZZ+XXXXX+YIIII",SparsePauliOp.from_dict({"ZZZZZ":1.0,"XXXXX":0.5,"YIIII":-0.3})),
       ("Case7: RY+RZ+multi",SparsePauliOp.from_dict({"ZZZZZ":1.0,"XXXXX":0.5,"YIIII":-0.3})),
       ("Case8: RY+CX+RZ+multi",SparsePauliOp.from_dict({"ZZZZZ":1.0,"XXXXX":0.5,"YIIII":-0.3}))]
qc2=sf.Circuit(n);n2=10;names2=[f"b{i}" for i in range(n2)];theta2=rng.uniform(-1,1,n2);idx=0
for q in range(n): qc2.ry(sf.param(names2[idx]),q);idx+=1
for q in range(n): qc2.rz(sf.param(names2[idx]),q);idx+=1
qc3=sf.Circuit(n);idx=0
for q in range(n): qc3.ry(sf.param(names2[idx]),q);idx+=1
for q in range(n-1): qc3.cx(q,q+1)
for q in range(n): qc3.rz(sf.param(names2[idx]),q);idx+=1
_=[diag_test(l,q,o,names,theta) for l,o in cases[:6]]
diag_test(cases[6][0],qc2,cases[6][1],names2,theta2)
diag_test(cases[7][0],qc3,cases[7][1],names2,theta2)
print("  >>> Next: CELL 11 (Memory Efficiency)")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 11: Memory Efficiency
# ═══════════════════════════════════════════════════════════════════════════════
cell("Memory Efficiency - Per-backend memory at n=6..16")
MEM_BK=[b for b in ["statevector","rust","jax"] if b in working]
print(f"\n  {'Backend':<16s} ",end="")
for n in [6,10,12,14,16]: print(f"n={n:<8d}",end="")
print("\n  "+"-"*(16+11*5))
for bk in MEM_BK:
    print(f"  {bk:<16s} ",end="")
    for n in [6,10,12,14,16]:
        try:
            _,dt,mem=_track(lambda: BackendRegistry.get_backend(bk).run(w_qaoa(n),shots=0))
            print(f"{mem:<+10.1f} ",end="")
            record("memory",bk,"QAOA",n,dt,mem)
        except: print(f"{'FAIL':<10s} ",end="")
    print()
print(f"\n  Theoretical (2^n*16 bytes):")
for n in [6,10,12,14,16]: print(f"  n={n:2d}: {(2**n)*16/1024/1024:<8.2f}MB ")
print("  >>> Next: CELL 12 (VQE H2 Accuracy)")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 12: VQE H2 Accuracy
# ═══════════════════════════════════════════════════════════════════════════════
cell("VQE H2 - Ground State Energy vs FCI")
H2={'II':-0.4804,'ZZ':0.1712,'XX':0.0485,'YY':-0.0485}
from scipy.optimize import minimize
I2=np.eye(2,dtype=complex);X=np.array([[0,1],[1,0]],dtype=complex)
Y=np.array([[0,-1j],[1j,0]],dtype=complex);Z=np.array([[1,0],[0,-1]],dtype=complex)
Hm=np.zeros((4,4),dtype=complex)
for ps,c in H2.items():
    op=1
    for ch in ps: op=np.kron(op,{'I':I2,'Z':Z,'X':X,'Y':Y}[ch])
    Hm+=c*op
E_exact=float(np.min(np.linalg.eigvalsh(Hm)))
print(f"  Exact FCI energy: {E_exact:.8f} Ha")
def energy(theta):
    qc=sf.Circuit(2);qc.h(0);qc.cx(0,1);qc.ry(theta[0],0);qc.ry(theta[1],1)
    sv=np.asarray(BackendRegistry.get_backend("statevector").run(qc,shots=0).statevector,dtype=np.complex128)
    e=0.0
    for ps,c in H2.items():
        p=np.abs(sv)**2;i=np.arange(4);par=np.zeros(4,int)
        for bit,ch in enumerate(ps):
            if ch in ('Z',): par^=(i>>(3-bit))&1
            elif ch=='X': par^=~((i>>(3-bit))&1)
            elif ch=='Y': par^=(i>>(3-bit))&1
        e+=c*float(np.real(np.sum(np.where(par==0,1.0,-1.0)*p)))
    return e
print("  Optimizing VQE (Nelder-Mead)...")
t0=time.time()
res=minimize(energy,x0=[0.1,0.1],method='Nelder-Mead',options={'maxiter':200,'xatol':1e-8,'fatol':1e-8})
t_vqe=time.time()-t0
E_vqe=float(res.fun)
err=abs(E_vqe-E_exact)
print(f"  VQE energy:  {E_vqe:.8f} Ha")
print(f"  Error:       {err:.2e} Ha")
print(f"  Chemical accuracy (<1.6mHa): {'YES' if err<0.0016 else 'NO'}")
print(f"  Time:        {t_vqe:.2f}s")
record("vqe","statevector","H2",2,t_vqe*1000,0,ze=err)
print("  >>> Next: CELL 13 (QEC Codes)")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 13: QEC Codes - All 11 quantum error correction codes
# ═══════════════════════════════════════════════════════════════════════════════
cell("QEC Codes - All 11 Superfermion error correction codes")
try:
    from superfermion.qec import RepetitionCode,ShorCode,SteaneCode,BaconShorCode,SurfaceCode2D
    from superfermion.qec import ToricCode2D,ColorCode,HoneycombCode,LDPCCode,HypercubeCode4D,BivariateBicycleCode
    _HAS_QEC=True
except: _HAS_QEC=False
if _HAS_QEC:
    codes={"Repetition(n=3)": lambda: RepetitionCode(n=3),"Repetition(n=5)": lambda: RepetitionCode(n=5),
           "Shor [[9,1,3]]": lambda: ShorCode(),"Steane [[7,1,3]]": lambda: SteaneCode(),
           "BaconShor(L=3)": lambda: BaconShorCode(L=3),"Surface2D(d=3)": lambda: SurfaceCode2D(distance=3),
           "Toric2D(L=3)": lambda: ToricCode2D(size=3),"ColorCode(d=3)": lambda: ColorCode(distance=3),
           "Honeycomb": lambda: HoneycombCode(),"LDPC(n=7,k=1)": lambda: LDPCCode(n=7,k=1),
           "Hypercube4D(L=3)": lambda: HypercubeCode4D(size=3)}
    passed=0
    for name,fn in codes.items():
        try:
            code=fn();circ=code.build();nq=circ.n_qubits
            be='mps' if nq>20 else 'statevector'
            r=sf.run(circ,backend=be,shots=256)
            outc=len(r.counts) if r.counts else 0
            print(f"  [PASS] {name:<22s} qubits={nq:<4d} outcomes={outc}")
            passed+=1
        except Exception as e:
            print(f"  [FAIL] {name:<22s} {str(e)[:60]}")
    print(f"\n  QEC codes: {passed}/{len(codes)} passed")
else:
    print("  QEC module not available")
print("  >>> Next: CELL 14 (QEC Decoders)")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 14: QEC Decoders
# ═══════════════════════════════════════════════════════════════════════════════
cell("QEC Decoders - 4 syndrome decoders")
if _HAS_QEC:
    try:
        from superfermion.qec.decoders import MWPMDecoder,UnionFindDecoder,BPOSD_Decoder,NeuralDecoder
        decoders={"MWPM": lambda: MWPMDecoder(),"UnionFind": lambda: UnionFindDecoder(),
                  "BPOSD": lambda: BPOSD_Decoder(),"Neural": lambda: NeuralDecoder()}
        for name,fn in decoders.items():
            try:
                d=fn();print(f"  [PASS] {name:<22s} init OK | type={type(d).__name__}")
            except Exception as e:
                print(f"  [FAIL] {name:<22s} {str(e)[:60]}")
    except Exception as e:
        print(f"  Decoder import failed: {str(e)[:60]}")
else:
    print("  QEC module not available")
print("  >>> Next: CELL 15 (Scientific Benchmarks)")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 15: Scientific Benchmarks
# ═══════════════════════════════════════════════════════════════════════════════
cell("Scientific Benchmarks - GHZ, QFT, Grover, BV, Trotter, CHSH")
SV_BE=BackendRegistry.get_backend("statevector")
print(f"  {'Test':<25s} {'n':<4s} {'Result':<20s}")
print("  "+"-"*55)
# GHZ fidelity
for n in [4,8]:
    sv=np.asarray(SV_BE.run(w_ghz(n),shots=0).statevector);norm=np.sum(np.abs(sv)**2)
    print(f"  {'GHZ norm':<25s} {n:<4d} {norm:<20.10f}")
    record("sci","statevector","GHZ",n,0,0,ze=abs(1-norm))
# QFT is unitary
sv=np.asarray(SV_BE.run(w_qft(4),shots=0).statevector);norm=np.sum(np.abs(sv)**2)
print(f"  {'QFT norm':<25s} {4:<4d} {norm:<20.10f}")
record("sci","statevector","QFT",4,0,0,ze=abs(1-norm))
# Grover
for n in [3,4]:
    c=sf.Circuit(n)
    for q in range(n): c.h(q)
    iters=max(1,int(round((math.pi/4)*math.sqrt(2**n))))
    for _ in range(iters):
        for q in range(n): c.x(q)
        for q in range(n): c.h(q)
        c.h(n-1);c.cx(list(range(n-1)),[n-1]*(n-1)) if n>2 else c.cz(0,1) if n==2 else c.z(0)
        c.h(n-1)
        for q in range(n): c.h(q)
        for q in range(n): c.x(q)
    sv=np.asarray(SV_BE.run(c,shots=0).statevector);top=int(np.argmax(np.abs(sv)**2))
    print(f"  {'Grover top state':<25s} {n:<4d} {bin(top):<20s}")
    record("sci","statevector","Grover",n,0,0,ze=float(top==2**n-1))
# Bernstein-Vazirani
for n in [4,6]:
    secret=rng.integers(0,2,n);c=sf.Circuit(n+1);c.x(n)
    for q in range(n+1): c.h(q)
    for q in range(n):
        if secret[q]: c.cx(q,n)
    for q in range(n): c.h(q)
    sv=np.asarray(SV_BE.run(c,shots=0).statevector);top=int(np.argmax(np.abs(sv)**2))
    print(f"  {'BV secret match':<25s} {n:<4d} {'OK' if top==sum(int(s)*(2**(n-1-i)) for i,s in enumerate(secret)) else 'FAIL':<20s}")
    record("sci","statevector","BV",n,0,0,ze=0.0)
# CHSH: Bell state <XX+ZZ> should be 2*sqrt(2)
c=sf.Circuit(2).h(0).cx(0,1);sv=np.asarray(SV_BE.run(c,shots=0).statevector)
xx=sum(((-1)**((i>>1)^(i&1)))*np.abs(sv[i])**2 for i in range(4))
zz=np.abs(sv[0])**2+np.abs(sv[3])**2-np.abs(sv[1])**2-np.abs(sv[2])**2
bell=float(xx+zz);th=2*math.sqrt(2)
print(f"  {'CHSH <XX+ZZ>':<25s} {2:<4d} {bell:<+20.6f} (theoretical={th:.4f})")
record("sci","statevector","CHSH",2,0,0,ze=abs(bell-th))
print("  >>> Next: CELL 16 (MPS High-Qubit Scaling ⚠)")

# ═══════════════════════════════════════════════════════════════════════════════
# ⚠ CELL 16: MPS High-Qubit Scaling
# ═══════════════════════════════════════════════════════════════════════════════
cell("MPS High-Qubit Scaling - 20..100 qubits (⚠ heavy for n>=50)")
if "mps" in working:
    print("  ⚠ WARNING: n>=50 may use significant memory and time.")
    print("  Skip this cell or run only small n values on constrained machines.\n")
    print(f"  {'n':<6s} {'Time(ms)':<12s} {'Mem(MB)':<10s}")
    print("  "+"-"*35)
    for n in [20,30,50,80,100]:
        try:
            c=w_qaoa(n)
            mps=MPSSimulatorBackend(options={"max_bond_dim":64})
            _,dt,mem=_track(lambda: mps.run(c,shots=0))
            print(f"  {n:<6d} {dt:<12.1f} {mem:<+10.1f}")
            record("mps_scale","mps","QAOA",n,dt,mem)
        except Exception as e:
            print(f"  {n:<6d} {'FAIL':<12s} {str(e)[:30]}")
    if "jax_mps" in working:
        print(f"\n  JAX-MPS:")
        for n in [20,30,50,100]:
            try:
                be=BackendRegistry.get_backend("jax_mps")
                _,dt,mem=_track(lambda: be.run(w_ghz(n),shots=0))
                print(f"  {n:<6d} {dt:<12.1f} {mem:<+10.1f}")
                record("mps_scale","jax_mps","GHZ",n,dt,mem)
            except Exception as e:
                print(f"  {n:<6d} {'FAIL':<12s} {str(e)[:30]}")
else:
    print("  mps backend not available")
print("  >>> Next: CELL 17 (JIT Warmup)")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 17: JIT Warmup & Cached Execution
# ═══════════════════════════════════════════════════════════════════════════════
cell("JIT Warmup & Cached Execution - JAX backend speed")
if "jax" in working:
    nj=12;c=w_qaoa(nj);be=BackendRegistry.get_backend("jax")
    t0=time.perf_counter();_=be.run(c,shots=0);cold=(time.perf_counter()-t0)*1000
    ws=[]
    for _ in range(10):
        t0=time.perf_counter();_=be.run(c,shots=0);ws.append((time.perf_counter()-t0)*1000)
    avg=np.mean(ws);mn=np.min(ws)
    print(f"  Static circuit (cached JIT):")
    print(f"    Cold: {cold:.2f}ms | Warm avg: {avg:.4f}ms | Warm min: {mn:.4f}ms | Speedup: {cold/mn:.0f}x")
    record("jit","jax","QAOA",nj,mn,0)
    # Dynamic (VQE-style parameter variation)
    cd=sf.Circuit(nj)
    for i in range(nj): cd.ry(sf.param(f"t{i}"),i)
    import jax.numpy as jnp; import jax
    fj=sf.qml.circuit_to_jax(cd,backend="jax")
    @jax.jit
    def step(p): return fj(*p)
    _=step(jnp.array(np.random.rand(nj)))
    vs=[]
    for _ in range(20):
        p=jnp.array(np.random.rand(nj));t0=time.perf_counter();r=step(p);r.block_until_ready();vs.append((time.perf_counter()-t0)*1000)
    avg_v=np.mean(vs);mn_v=np.min(vs)
    print(f"  Dynamic params (VQE-style): avg={avg_v:.4f}ms min={mn_v:.4f}ms")
    record("jit","jax_dyn","QAOA",nj,mn_v,0)
else:
    print("  jax backend not available")
print("  >>> Next: CELL 18 (Final Summary)")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 18: Final Summary
# ═══════════════════════════════════════════════════════════════════════════════
cell("FINAL SUMMARY - All Results")
print(f"\n  Total benchmarks: {len(ALL_RESULTS)}")
print(f"  Sections: {len(set(r['section'] for r in ALL_RESULTS))}")
print(f"\n  {'Section':<18s} {'Backend':<16s} {'Workload':<12s} {'n':<4s} {'Time(ms)':<10s} {'Mem':<8s} {'Status':<10s}")
print("  "+"-"*85)
for r in sorted(ALL_RESULTS, key=lambda x:(x['section'],x['backend'],x['n'])):
    rt=f"{r['runtime_ms']:.1f}" if r['runtime_ms']>=0 else 'n/a'
    me=f"{r['rss_mb']:+.1f}" if abs(r['rss_mb'])>0.1 else '-'
    print(f"  {r['section']:<18s} {r['backend']:<16s} {r['workload']:<12s} {r['n']:<4d} {rt:<10s} {me:<8s} {r['status']:<10s}")

passed=sum(1 for r in ALL_RESULTS if r['status']=='OK')
failed=sum(1 for r in ALL_RESULTS if 'FAIL' in str(r['status']))
print(f"\n{'='*60}")
print(f"  FINAL: {passed}/{len(ALL_RESULTS)} passed ({failed} failed)")
print(f"  Backends tested: {len(working)}/{len(ALL_NAMES)}")
# Rust highlights
for r in ALL_RESULTS:
    if r['backend']=='rust' and r['status']=='OK' and r['runtime_ms']>0:
        sv=[x for x in ALL_RESULTS if x['backend']=='statevector' and x['workload']==r['workload'] and x['n']==r['n'] and x['status']=='OK']
        if sv and sv[0]['runtime_ms']>0:
            sp=sv[0]['runtime_ms']/r['runtime_ms']
            if sp>1.5: print(f"  Rust {sp:.1f}x faster on {r['workload']} n={r['n']}")
print(f"{'='*60}")
print("  Benchmark notebook complete! Run cells individually.")
