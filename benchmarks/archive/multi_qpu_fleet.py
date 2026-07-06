# %% [markdown]
# Superfermion Multi-QPU Quantum Algorithm Fleet
# IonQ - IBM Quantum - OpenQuantum (Rigetti - IQM - AQT)
#
# 20+ Qubit Algorithms: VQE - QAOA - QFT - Bernstein-Vazirani - Deutsch-Jozsa - QML - Grover
# All orchestrated exclusively through Superfermion - no external SDKs

# %% [markdown]
# Setup: Load Credentials and Connect All Providers

# %%
import os, sys, time, json, math
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime

ROOT = Path(os.getcwd()).parent if Path(os.getcwd()).name == 'notebooks' else Path(os.getcwd())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / '.env')

IBM_TOKEN   = os.getenv('IBM_QUANTUM_TOKEN', '')
IONQ_KEY    = os.getenv('IONQ_API_KEY', '')
OQ_CID      = os.getenv('OPENQUANTUM_CLIENT_ID', '')
OQ_SECRET   = os.getenv('OPENQUANTUM_CLIENT_SECRET', '')

print(f'IBM Quantum:       {"✓" if IBM_TOKEN else "✗"}')
print(f'IonQ:              {"✓" if IONQ_KEY else "✗"}')
print(f'OpenQuantum:       {"✓" if (OQ_CID and OQ_SECRET) else "✗"}')

import superfermion as sf
print(f'\nsuperfermion v{sf.__version__ if hasattr(sf, "__version__") else "dev"}')

# Connect all providers via superfermion runtime
sf.runtime.connect('ibm', token=IBM_TOKEN)
sf.runtime.connect('ionq', api_key=IONQ_KEY)
sf.runtime.connect('openquantum', client_id=OQ_CID, client_secret=OQ_SECRET)

RESULTS = {}
FLEET_START = time.perf_counter()

print('\n✓ All providers connected via sf.runtime')

# Timeout wrapper for QPU submissions (prevents hanging)
def qpu_submit(fn, timeout_s=30):
    """Run a QPU submission with timeout. Returns job_id or None."""
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(fn)
        try:
            return fut.result(timeout=timeout_s)
        except FutureTimeout:
            print(f'    (timeout after {timeout_s}s)')
            return None
        except Exception as e:
            print(f'    Error: {e}')
            return None

# %% [markdown]
# 1. Provider Backend Discovery and Calibration

# %%
print('=' * 70)
print('PROVIDER BACKEND DISCOVERY')
print('=' * 70)

# ── IBM: List backends + noise data ──
print('\n[IBM QUANTUM]')
try:
    from superfermion.runtime.providers.ibm import IBMProvider
    ibm_prov = IBMProvider(token=IBM_TOKEN)
    if ibm_prov._service:
        backends = ibm_prov._service.backends()
        print(f'  Backends: {len(backends)}')
        backend_list = []
        for b in backends[:10]:
            st = b.status()
            print(f'    {b.name:24s}  qubits={b.num_qubits:3d}  pending={st.pending_jobs:4d}  op={st.operational}')
            backend_list.append({
                'name': b.name, 'qubits': b.num_qubits,
                'pending_jobs': st.pending_jobs, 'operational': st.operational
            })
        RESULTS['ibm_backends'] = backend_list
        
        if backends:
            noise = ibm_prov.get_noise_data(backends[0].name)
            if noise:
                t1_sample = [round(v/1e3, 1) for v in noise.get('t1', [])[:5]]
                ro_sample = [f'{v*100:.2f}%' for v in noise.get('readout_error', [])[:5]]
                print(f'\n  Noise from {backends[0].name}:')
                print(f'    T1 (us): {t1_sample}')
                print(f'    Readout err: {ro_sample}')
                RESULTS['ibm_noise'] = noise
except Exception as e:
    print(f'  IBM ERROR: {e}')

# ── IonQ ──
print('\n[IONQ]')
try:
    from superfermion.runtime.providers.ionq import IonQProvider
    ionq_prov = IonQProvider(api_key=IONQ_KEY)
    
    import requests
    headers = {'Authorization': f'apiKey {IONQ_KEY}'}
    resp = requests.get('https://api.ionq.co/v0.3/backends', headers=headers)
    if resp.status_code == 200:
        ionq_backends = resp.json()
        print(f'  Backends: {len(ionq_backends)}')
        for b in ionq_backends[:10]:
            print(f'    {b.get("backend","?")}  qubits={b.get("qubits","?")}  status={b.get("status","?")}')
        RESULTS['ionq_backends'] = ionq_backends
    else:
        print(f'  Backend listing: {resp.status_code}')
    
    c = ionq_prov.get_characterization()
    if c:
        fids = c.get('fidelities', {})
        print(f'\n  Characterization: {len(fids)} fidelity metrics')
        for k, v in list(fids.items())[:4]:
            print(f'    {k}: {v*100:.3f}%')
        RESULTS['ionq_char'] = c
    else:
        print('  No characterization (may need QPU access tier)')
except Exception as e:
    print(f'  IonQ ERROR: {e}')

# ── OpenQuantum ──
print('\n[OPENQUANTUM]')
try:
    from superfermion.runtime.providers.openquantum import OpenQuantumProvider
    oq_prov = OpenQuantumProvider(client_id=OQ_CID, client_secret=OQ_SECRET)
    backend_map = oq_prov._backend_map
    print(f'  Backends: {len(backend_map)}')
    for name, bid in list(backend_map.items())[:10]:
        print(f'    {name:24s}  id={bid}')
    RESULTS['oq_backends'] = {k: v for k, v in backend_map.items()}
    
    orgs = oq_prov.management.list_user_organizations().organizations
    if orgs:
        print(f'\n  Organization: {orgs[0].name}')
        RESULTS['oq_org'] = orgs[0].name
except Exception as e:
    print(f'  OpenQuantum ERROR: {e}')

print('\n✓ Provider discovery complete')

# %% [markdown]
# 2. Bernstein-Vazirani Algorithm - 22 Qubits
# Classic oracle: detects secret bitstring in 1 query vs N classical

# %%
N_BV = 22
SECRET = np.random.randint(0, 2, N_BV).tolist()
print(f'Bernstein-Vazirani: {N_BV} qubits')
print(f'Secret: {"".join(map(str, SECRET))}')

bv_circuit = sf.Circuit(N_BV + 1)
bv_circuit.x(N_BV)
for i in range(N_BV + 1):
    bv_circuit.h(i)
for i, bit in enumerate(SECRET):
    if bit == 1:
        bv_circuit.cx(i, N_BV)
for i in range(N_BV):
    bv_circuit.h(i)
bv_circuit.measure_all()
print(f'Gates: {bv_circuit.gate_count()}')

t0 = time.perf_counter()
bv_result = sf.run(bv_circuit, backend='statevector', shots=2048)
bv_dt = (time.perf_counter() - t0) * 1000
top_bv = max(bv_result.counts, key=bv_result.counts.get) if bv_result.counts else 'N/A'
measured_secret = top_bv[:-1]
match = measured_secret == ''.join(map(str, SECRET))
print(f'  Local SV: {bv_dt:.1f} ms  |  Match: {"✓" if match else "✗"}')
print(f'  Measured: {measured_secret}')
RESULTS['bv'] = {'secret': ''.join(map(str,SECRET)), 'measured': measured_secret, 'match': match, 'qubits': N_BV}

# Submit to QPUs
bv_jobs = {}
print('\n  Submitting to QPUs...')
def _bv_oq(): j = sf.runtime.run(bv_circuit, backend='oq.ionq', shots=256); return j.job_id
jid = qpu_submit(_bv_oq, timeout_s=45)
if jid: print(f'  OQ/IonQ:   job_id={jid}'); bv_jobs['oq_ionq'] = jid
def _bv_ibm(): j = sf.runtime.run(bv_circuit, backend='ibm_brisbane', shots=256); return j.job_id
jid = qpu_submit(_bv_ibm, timeout_s=30)
if jid: print(f'  IBM Bris:  job_id={jid}'); bv_jobs['ibm'] = jid
RESULTS['bv_jobs'] = bv_jobs

# %% [markdown]
# 3. Deutsch-Jozsa Algorithm - 22 Qubits
# Determines if boolean function is constant or balanced in 1 query

# %%
N_DJ = 22
print(f'Deutsch-Jozsa: {N_DJ} qubits (balanced oracle)')

dj_circuit = sf.Circuit(N_DJ + 1)
dj_circuit.x(N_DJ)
for i in range(N_DJ + 1):
    dj_circuit.h(i)
for i in range(N_DJ // 2):
    dj_circuit.cx(i, N_DJ)
for i in range(N_DJ):
    dj_circuit.h(i)
dj_circuit.measure_all()
print(f'Gates: {dj_circuit.gate_count()}')

t0 = time.perf_counter()
dj_result = sf.run(dj_circuit, backend='statevector', shots=2048)
dj_dt = (time.perf_counter() - t0) * 1000
top_dj = max(dj_result.counts, key=dj_result.counts.get) if dj_result.counts else 'N/A'
data_bits = top_dj[:-1]
is_balanced = any(b == '1' for b in data_bits)
print(f'  Local SV: {dj_dt:.1f} ms  |  Balanced: {"✓" if is_balanced else "✗"}')
print(f'  Top: {top_dj}')
RESULTS['dj'] = {'result': top_dj, 'balanced': is_balanced, 'qubits': N_DJ}

# Submit to QPUs
dj_jobs = {}
print('\n  Submitting to QPUs...')
def _dj_oq(): j = sf.runtime.run(dj_circuit, backend='oq.iqm', shots=256); return j.job_id
jid = qpu_submit(_dj_oq, timeout_s=45)
if jid: print(f'  OQ/IQM:    job_id={jid}'); dj_jobs['oq_iqm'] = jid
def _dj_ionq(): j = sf.runtime.run(dj_circuit, backend='ionq.simulator', shots=256); return j.job_id
jid = qpu_submit(_dj_ionq, timeout_s=30)
if jid: print(f'  IonQ:      job_id={jid}'); dj_jobs['ionq'] = jid
RESULTS['dj_jobs'] = dj_jobs

# %% [markdown]
# 4. Quantum Fourier Transform - 20+ Qubits
# Core subroutine for Shor's algorithm and phase estimation

# %%
N_QFT = 22
print(f'QFT: {N_QFT} qubits')

qft_circuit = sf.Circuit(N_QFT)
for i in range(N_QFT):
    qft_circuit.h(i)
for i in range(N_QFT):
    qft_circuit.h(i)
    for j in range(i + 1, N_QFT):
        angle = math.pi / (2 ** (j - i))
        qft_circuit.cp(angle, j, i)
qft_circuit.measure_all()
print(f'Gates: {qft_circuit.gate_count()}')

t0 = time.perf_counter()
qft_result = sf.run(qft_circuit, backend='jax_mps', shots=2048, max_bond_dim=128)
qft_dt = (time.perf_counter() - t0) * 1000
top_qft = max(qft_result.counts, key=qft_result.counts.get) if qft_result.counts else 'N/A'
n_unique = len(qft_result.counts) if qft_result.counts else 0
print(f'  Local MPS: {qft_dt:.1f} ms  |  unique: {n_unique}')
print(f'  Top: {top_qft}')
RESULTS['qft'] = {'top': top_qft, 'unique': n_unique, 'qubits': N_QFT, 'time_ms': qft_dt}

# Submit 10-qubit QFT to QPUs (deep circuit)
qft10 = sf.Circuit(10)
for i in range(10):
    qft10.h(i)
for i in range(10):
    qft10.h(i)
    for j in range(i + 1, 10):
        qft10.cp(math.pi / (2 ** (j - i)), j, i)
qft10.measure_all()

qft_jobs = {}
print('\n  Submitting 10-qubit QFT to QPUs...')
def _qft_oq(): j = sf.runtime.run(qft10, backend='oq.rigetti', shots=256); return j.job_id
jid = qpu_submit(_qft_oq, timeout_s=45)
if jid: print(f'  OQ/Rigetti: job_id={jid}'); qft_jobs['oq_rigetti'] = jid
def _qft_ionq(): j = sf.runtime.run(qft10, backend='ionq.simulator', shots=256); return j.job_id
jid = qpu_submit(_qft_ionq, timeout_s=30)
if jid: print(f'  IonQ:       job_id={jid}'); qft_jobs['ionq'] = jid
RESULTS['qft_jobs'] = qft_jobs

# %% [markdown]
## 5. VQE — 20-Qubit TFIM Ground State
### Variational Quantum Eigensolver via scipy.optimize

# %%
from superfermion.algorithms.variational import VQE
from superfermion.observables.core import SparsePauliOp
from superfermion.qml.templates import HardwareEfficientAnsatz

N_VQE = 20
print(f'VQE: {N_VQE}-qubit Transverse-Field Ising Model')

terms = {}
J, hfield = 1.0, 0.5
for i in range(N_VQE - 1):
    zz = list('I' * N_VQE)
    zz[i] = 'Z'; zz[i+1] = 'Z'
    terms[''.join(zz)] = -J
for i in range(N_VQE):
    xx = list('I' * N_VQE)
    xx[i] = 'X'
    terms[''.join(xx)] = -hfield
hamiltonian = SparsePauliOp.from_dict(terms)
print(f'  Hamiltonian terms: {len(terms)}')

ansatz = HardwareEfficientAnsatz(N_VQE, n_layers=2)
print(f'  Parameters: {len(ansatz.parameters)}')

vqe = VQE(ansatz, hamiltonian, backend='statevector', optimizer='L-BFGS-B')
t0 = time.perf_counter()
vqe_result = vqe.minimize(iterations=200, verbose=True, callback_freq=50)
vqe_dt = time.perf_counter() - t0

print(f'\n  VQE complete: {vqe_dt:.1f}s')
print(f'  Ground state energy: {vqe_result.optimal_value:+.8f}')
print(f'  Converged: {vqe_result.metadata.get("scipy_success")}')
RESULTS['vqe'] = {
    'energy': vqe_result.optimal_value, 'qubits': N_VQE,
    'params': len(ansatz.parameters), 'time_s': vqe_dt,
    'history': vqe_result.history
}

# %% [markdown]
## 6. QAOA — MaxCut on 20-Node Graph
### Quantum Approximate Optimization for combinatorial problems

# %%
from superfermion.algorithms.variational import QAOA

N_QAOA = 20
print(f'QAOA MaxCut: {N_QAOA} nodes, 3-regular graph')

rng_qaoa = np.random.default_rng(42)
edges = []
for i in range(N_QAOA):
    for d in range(1, 4):
        j = (i + d) % N_QAOA
        if i < j:
            edges.append((i, j, 1.0))
print(f'  Edges: {len(edges)}')

qaoa = QAOA(N_QAOA, edges, p_layers=2, backend='statevector', optimizer='COBYLA')
t0 = time.perf_counter()
qaoa_result = qaoa.minimize(iterations=150, verbose=True)
qaoa_dt = time.perf_counter() - t0

print(f'\n  QAOA complete: {qaoa_dt:.1f}s')
print(f'  Max cut: {qaoa_result.metadata.get("max_cut_value")} / {len(edges)}')
print(f'  Best bitstring: {qaoa_result.metadata.get("best_bitstring")}')
RESULTS['qaoa'] = {
    'max_cut': qaoa_result.metadata.get('max_cut_value'),
    'total_edges': len(edges), 'qubits': N_QAOA, 'time_s': qaoa_dt,
    'best_bitstring': qaoa_result.metadata.get('best_bitstring'),
    'history': qaoa_result.history
}

# Build optimized QAOA circuit for QPU sampling
qaoa_circuit = sf.Circuit(N_QAOA)
for i in range(N_QAOA):
    qaoa_circuit.h(i)
gamma_opt = qaoa_result.optimal_params['gamma']
beta_opt = qaoa_result.optimal_params['beta']
for p in range(2):
    g, b = float(gamma_opt[p]), float(beta_opt[p])
    for qi, qj, w in edges:
        qaoa_circuit.cx(qi, qj)
        qaoa_circuit.rz(2.0 * g * w, qj)
        qaoa_circuit.cx(qi, qj)
    for i in range(N_QAOA):
        qaoa_circuit.rx(2.0 * b, i)
qaoa_circuit.measure_all()

qaoa_jobs = {}
print('\n  Submitting optimized QAOA to QPUs...')
def _qaoa_oq(): j = sf.runtime.run(qaoa_circuit, backend='oq.ionq', shots=256); return j.job_id
jid = qpu_submit(_qaoa_oq, timeout_s=45)
if jid: print(f'  OQ/IonQ:   job_id={jid}'); qaoa_jobs['oq_ionq'] = jid
def _qaoa_ibm(): j = sf.runtime.run(qaoa_circuit, backend='ibm_brisbane', shots=256); return j.job_id
jid = qpu_submit(_qaoa_ibm, timeout_s=30)
if jid: print(f'  IBM Bris:  job_id={jid}'); qaoa_jobs['ibm'] = jid
RESULTS['qaoa_jobs'] = qaoa_jobs

# %% [markdown]
## 7. QML — 20-Qubit Variational Classifier + Grover + GitHub Codes

# %%
# ═══ QML: 20-qubit hardware-efficient variational classifier ═══
N_QML = 20
print(f'QML: {N_QML} qubits variational classifier')

qml_circuit = sf.Circuit(N_QML)
for i in range(N_QML):
    qml_circuit.ry(sf.param(f'x_{i}'), i)
    qml_circuit.rz(sf.param(f'z_{i}'), i)
for layer in range(2):
    for i in range(N_QML - 1):
        qml_circuit.cx(i, i + 1)
    qml_circuit.cx(N_QML - 1, 0)
    for i in range(N_QML):
        qml_circuit.ry(sf.param(f'ry_{layer}_{i}'), i)
        qml_circuit.rz(sf.param(f'rz_{layer}_{i}'), i)
qml_circuit.measure_all()
print(f'  Gates: {qml_circuit.gate_count()}  |  Params: {len(qml_circuit.parameters)}')

params_qml = {p: float(np.random.uniform(0, 2*math.pi)) for p in qml_circuit.parameters}
bound_qml = qml_circuit.bind(params_qml)
t0 = time.perf_counter()
qml_result = sf.run(bound_qml, backend='statevector', shots=4096)
qml_dt = (time.perf_counter() - t0) * 1000
top_qml = max(qml_result.counts, key=qml_result.counts.get) if qml_result.counts else 'N/A'
print(f'  Local: {qml_dt:.1f} ms  |  unique={len(qml_result.counts)}  |  top={top_qml}')
RESULTS['qml'] = {'top': top_qml, 'unique': len(qml_result.counts), 'qubits': N_QML}

qml_jobs = {}
print('  Submitting to QPUs...')
def _qml_ionq(): j = sf.runtime.run(bound_qml, backend='ionq.simulator', shots=512); return j.job_id
jid = qpu_submit(_qml_ionq, timeout_s=30)
if jid: print(f'  IonQ:      job_id={jid}'); qml_jobs['ionq'] = jid
def _qml_oq(): j = sf.runtime.run(bound_qml, backend='oq.rigetti', shots=256); return j.job_id
jid = qpu_submit(_qml_oq, timeout_s=45)
if jid: print(f'  OQ/Rigetti: job_id={jid}'); qml_jobs['oq_rigetti'] = jid
RESULTS['qml_jobs'] = qml_jobs

# ═══ Grover Search: 15 qubits ═══
print(f'\nGrover: 15 qubits, target=12345')
N_GRV, TGT = 15, 12345
n_iter = int(math.pi / 4 * math.sqrt(2**N_GRV))
grv = sf.Circuit(N_GRV)
for i in range(N_GRV):
    grv.h(i)
tgt_bits = format(TGT, f'0{N_GRV}b')
zq = [i for i, b in enumerate(tgt_bits) if b == '0']
for _ in range(n_iter):
    for q in zq: grv.x(q)
    grv.h(N_GRV-1)
    for i in range(N_GRV-1): grv.cx(i, N_GRV-1)
    grv.h(N_GRV-1)
    for q in zq: grv.x(q)
    for i in range(N_GRV): grv.h(i); grv.x(i)
    grv.h(N_GRV-1)
    for i in range(N_GRV-1): grv.cx(i, N_GRV-1)
    grv.h(N_GRV-1)
    for i in range(N_GRV): grv.x(i); grv.h(i)
grv.measure_all()
t0 = time.perf_counter()
grv_result = sf.run(grv, backend='statevector', shots=4096)
grv_dt = (time.perf_counter() - t0) * 1000
top_grv = max(grv_result.counts, key=grv_result.counts.get) if grv_result.counts else 'N/A'
found = top_grv == format(TGT, f'0{N_GRV}b')
print(f'  Local: {grv_dt:.1f} ms  |  found: {top_grv}  |  {"✓" if found else "✗"}')
RESULTS['grover'] = {'found': top_grv, 'target': tgt_bits, 'success': found}

# ═══ QPE: 12 qubits ═══
print(f'\nQPE: 12 qubits, phase=5/16')
N_PE, ph = 12, 0.3125
qpe = sf.Circuit(N_PE + 1)
for i in range(N_PE): qpe.h(i)
for i in range(N_PE):
    rep = 2**i
    qpe.cp(ph * rep * 2 * math.pi, i, N_PE)
for i in range(N_PE//2): qpe.swap(i, N_PE-1-i)
for i in range(N_PE):
    qpe.h(i)
    for j in range(i+1, N_PE): qpe.cp(-math.pi/(2**(j-i)), j, i)
qpe.measure_all()
t0 = time.perf_counter()
qpe_result = sf.run(qpe, backend='statevector', shots=2048)
qpe_dt = (time.perf_counter() - t0) * 1000
top_pe = max(qpe_result.counts, key=qpe_result.counts.get) if qpe_result.counts else 'N/A'
meas_ph = int(top_pe[:N_PE], 2) / (2**N_PE) if top_pe != 'N/A' else 0
print(f'  Local: {qpe_dt:.1f} ms  |  expected={ph}  measured={meas_ph:.6f}')
RESULTS['qpe'] = {'expected': ph, 'measured': meas_ph, 'top': top_pe}

# ═══ Random Circuit (Google-supremacy style): 15 qubits depth 20 ═══
print(f'\nRandom Circuit: 15 qubits, depth 20')
rng_rc = np.random.default_rng(99)
rc = sf.Circuit(15)
for d in range(20):
    for i in range(15):
        g = rng_rc.choice(['h','rx','ry','rz'])
        if g == 'h': rc.h(i)
        elif g == 'rx': rc.rx(rng_rc.uniform(0, 2*math.pi), i)
        elif g == 'ry': rc.ry(rng_rc.uniform(0, 2*math.pi), i)
        elif g == 'rz': rc.rz(rng_rc.uniform(0, 2*math.pi), i)
    for i in range(0, 14, 2): rc.cx(i, i+1)
    for i in range(1, 14, 2): rc.cx(i, i+1)
rc.measure_all()
t0 = time.perf_counter()
rc_result = sf.run(rc, backend='statevector', shots=8192)
rc_dt = (time.perf_counter() - t0) * 1000
print(f'  Local: {rc_dt:.1f} ms  |  unique bitstrings: {len(rc_result.counts)}')
RESULTS['random_circuit'] = {'unique': len(rc_result.counts), 'qubits': 15, 'depth': 20}

print('\n✓ QML + Grover + QPE + Random circuit complete')

# %% [markdown]
## 8. QEC + Job Polling + Final Results

# %%
# ═══ QEC: Repetition Code ═══
print('QEC: Repetition Code (3 qubits)')
from superfermion.qec import RepetitionCode
rep = RepetitionCode(n=3)
qec_c = rep.build()
qec_r = sf.run(qec_c, backend='statevector', shots=256)
print(f'  Syndrome counts: {qec_r.counts}')
RESULTS['qec'] = {'counts': qec_r.counts}

# ═══ QPU Job Polling ═══
print('\n' + '=' * 70)
print('QPU JOB POLLING')
print('=' * 70)

JOB_RESULTS = {}

all_jobs = {}
for k in ['bv_jobs','dj_jobs','qft_jobs','qaoa_jobs','qml_jobs']:
    if k in RESULTS:
        for prov, jid in RESULTS[k].items():
            all_jobs[f'{k.replace("_jobs","")}/{prov}'] = (prov, jid)

print(f'Total QPU jobs submitted: {len(all_jobs)}')

for label, (provider, job_id) in all_jobs.items():
    print(f'\n[{label}] {job_id[:20]}...')
    try:
        if provider == 'ibm':
            from superfermion.runtime.providers.ibm import IBMProvider
            p = IBMProvider(token=IBM_TOKEN)
            job = p.retrieve_job(job_id)
        elif provider in ('ionq',):
            from superfermion.runtime.providers.ionq import IonQProvider
            p = IonQProvider(api_key=IONQ_KEY)
            job = p.retrieve_job(job_id)
        elif provider.startswith('oq'):
            from superfermion.runtime.providers.openquantum import OpenQuantumProvider
            p = OpenQuantumProvider(client_id=OQ_CID, client_secret=OQ_SECRET)
            job = p.retrieve_job(job_id)
        else:
            print(f'  Unknown provider')
            continue
        
        st = job.status
        st_name = st.name if hasattr(st, 'name') else str(st)
        print(f'  Status: {st_name}')
        
        if st_name.upper() in ('DONE', 'COMPLETED'):
            try:
                result = job.result()
                counts = result.counts if hasattr(result, 'counts') else result.get('counts', {})
                if counts:
                    top = max(counts, key=counts.get)
                    print(f'  Counts: {len(counts)} unique  |  Top: {top} ({counts[top]})')
                    JOB_RESULTS[label] = {'status': 'DONE', 'top': top, 'n_unique': len(counts)}
                else:
                    print(f'  No counts in result')
                    JOB_RESULTS[label] = {'status': 'DONE', 'counts': {}}
            except Exception as e2:
                print(f'  Result fetch error: {e2}')
                JOB_RESULTS[label] = {'status': 'ERROR', 'error': str(e2)}
        else:
            JOB_RESULTS[label] = {'status': st_name, 'job_id': job_id}
    except Exception as e:
        print(f'  Error: {e}')
        JOB_RESULTS[label] = {'status': 'ERROR', 'error': str(e)}

RESULTS['qpu_results'] = JOB_RESULTS

# ═══ Summary ═══
FLEET_DT = time.perf_counter() - FLEET_START
print('\n' + '=' * 70)
print('SUPERFERMION MULTI-QPU FLEET — FINAL RESULTS')
print('=' * 70)
print(f'Duration: {FLEET_DT:.1f}s  |  {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
print(f'Providers: IBM Quantum  |  IonQ  |  OpenQuantum (Rigetti/IQM/AQT)')
print()

checks = [
    ('Bernstein-Vazirani', 'bv', lambda r: f'secret matched: {r.get("match")}', lambda r: r.get('match', False)),
    ('Deutsch-Jozsa', 'dj', lambda r: f'balanced: {r.get("balanced")}', lambda r: r.get('balanced', False)),
    ('QFT (22 qubit)', 'qft', lambda r: f'unique={r.get("unique",0)}', lambda r: r.get('unique', 0) > 1),
    ('VQE (TFIM 20q)', 'vqe', lambda r: f'E0={r.get("energy",0):+.6f}', lambda r: 'energy' in r),
    ('QAOA (MaxCut)', 'qaoa', lambda r: f'cut={r.get("max_cut",0)}/{r.get("total_edges","?")}', lambda r: 'max_cut' in r),
    ('QML Classifier', 'qml', lambda r: f'unique={r.get("unique",0)}', lambda r: r.get('unique', 0) > 0),
    ('Grover Search', 'grover', lambda r: f'found={r.get("success")}', lambda r: r.get('success', False)),
    ('QPE', 'qpe', lambda r: f'phase={r.get("measured",0):.4f}', lambda r: r.get('measured', 0) > 0),
    ('Random Circuit', 'random_circuit', lambda r: f'unique={r.get("unique",0)}', lambda r: r.get('unique', 0) > 1),
    ('QEC RepCode', 'qec', lambda r: f'syndromes={len(r.get("counts",{}))}', lambda r: len(r.get('counts',{})) > 0),
]

passed = 0
for name, key, desc_fn, check_fn in checks:
    r = RESULTS.get(key, {})
    ok = check_fn(r)
    desc = desc_fn(r)
    status = 'PASS' if ok else 'FAIL'
    if ok: passed += 1
    print(f'  [{status}] {name:28s}  {desc}')

qpu_total = len(JOB_RESULTS)
qpu_done = sum(1 for v in JOB_RESULTS.values() if v.get('status') == 'DONE')
print(f'\n  QPU Jobs: {qpu_total} submitted, {qpu_done} completed')
print(f'  Tests: {passed}/{len(checks)} passed')

# Export
RESULTS['_timestamp'] = datetime.now().isoformat()
RESULTS['_duration_s'] = FLEET_DT
try:
    with open(str(ROOT / 'superfermion_fleet_results.json'), 'w') as f:
        clean = {}
        for k, v in RESULTS.items():
            try:
                json.dumps({k: str(v)[:500]})
                clean[k] = v
            except:
                clean[k] = str(v)[:500]
        json.dump(clean, f, indent=2, default=str)
    print(f'\n  Results exported: superfermion_fleet_results.json')
except Exception as e:
    print(f'\n  Export error: {e}')

print('\n' + '=' * 70)
print('FLEET CAMPAIGN COMPLETE')
print('=' * 70)
