"""Fetch QPU results: IBM job from previous test + IonQ historical results."""
import os, sys, time, json
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / '.env')

IBM_TOKEN = os.getenv('IBM_QUANTUM_TOKEN', '')
IONQ_KEY  = os.getenv('IONQ_API_KEY', '')
OQ_CID    = os.getenv('OPENQUANTUM_CLIENT_ID', '')
OQ_SECRET = os.getenv('OPENQUANTUM_CLIENT_SECRET', '')

RESULTS = {}

print("=" * 70)
print("  QPU RESULTS FETCH")
print("=" * 70)

# ============================================================================
# 1. IBM: Fetch Bell circuit job result (submitted earlier)
# ============================================================================
print("\n--- [IBM] Fetch Previous Bell Circuit Job ---")
IBM_JOB_ID = "d8gu4ms2upec739kbr60"  # from previous test run

try:
    from superfermion.runtime.providers.ibm import IBMProvider
    ibm = IBMProvider(token=IBM_TOKEN)
    if ibm._service:
        print(f"  Retrieving job {IBM_JOB_ID}...")
        job = ibm.retrieve_job(IBM_JOB_ID)
        status = job.status
        print(f"  Status: {status.value}")
        
        if status.value == "completed":
            result = job.result()
            counts = result.counts
            total = sum(counts.values())
            print(f"  Total shots: {total}")
            print(f"  Counts:")
            for bs, cnt in sorted(counts.items(), key=lambda x: -x[1]):
                print(f"    {bs}: {cnt} ({cnt/total*100:.1f}%)")
            
            # Check Bell state fidelity: expect |00> and |11> ~50% each
            bell_states = counts.get('00', 0) + counts.get('11', 0)
            bell_fid = bell_states / total if total > 0 else 0
            print(f"\n  Bell state fidelity (|00>+|11>): {bell_fid*100:.1f}%")
            print(f"  Expected: ~95-99% on real hardware")
            RESULTS['ibm_bell_job'] = {
                'job_id': IBM_JOB_ID,
                'status': 'completed',
                'counts': counts,
                'bell_fidelity': bell_fid
            }
        else:
            print(f"  Job not completed yet (status={status.value})")
            RESULTS['ibm_bell_job'] = {'job_id': IBM_JOB_ID, 'status': status.value}
except Exception as e:
    print(f"  Error: {e}")
    RESULTS['ibm_bell_job'] = {'error': str(e)[:120]}

# ============================================================================
# 2. IBM: List recent jobs
# ============================================================================
print("\n--- [IBM] Recent Jobs ---")
try:
    if ibm._service:
        recent = ibm._service.jobs(limit=10)
        print(f"  Last 10 IBM jobs:")
        for j in recent:
            jid = j.job_id()
            try:
                st = str(j.status())
                bn = j.backend().name if hasattr(j, 'backend') and j.backend() else '?'
            except:
                st = '?'
                bn = '?'
            print(f"    {jid}  status={st:20s} backend={bn}")
            RESULTS[f'ibm_job_{jid[:8]}'] = {'id': jid, 'status': st, 'backend': bn}
except Exception as e:
    print(f"  Error: {e}")

# ============================================================================
# 3. IBM: Submit & run a new GHZ-3 circuit (quick result)
# ============================================================================
print("\n--- [IBM] Submit New GHZ-3 Circuit ---")
try:
    import superfermion as sf
    ghz = sf.Circuit(3)
    ghz.h(0); ghz.cx(0, 1); ghz.cx(1, 2)
    
    backends = ibm._service.backends()
    be_name = backends[0].name if backends else 'ibm_fez'
    print(f"  Submitting GHZ-3 to {be_name}...")
    job_ghz = ibm.run(ghz, backend=be_name, shots=1024)
    print(f"  Job ID: {job_ghz.job_id}")
    print(f"  Status: {job_ghz.status.value}")
    
    # Try to wait for result (up to 120s)
    print("  Waiting for result (up to 120s)...")
    t0 = time.perf_counter()
    result_ghz = job_ghz.result(timeout=120)
    dt = time.perf_counter() - t0
    
    counts_ghz = result_ghz.counts
    total_ghz = sum(counts_ghz.values())
    print(f"  Result received in {dt:.1f}s")
    print(f"  Counts:")
    for bs, cnt in sorted(counts_ghz.items(), key=lambda x: -x[1]):
        print(f"    {bs}: {cnt} ({cnt/total_ghz*100:.1f}%)")
    
    ghz_fid = (counts_ghz.get('000', 0) + counts_ghz.get('111', 0)) / total_ghz
    print(f"  GHZ fidelity (|000>+|111>): {ghz_fid*100:.1f}%")
    RESULTS['ibm_ghz3'] = {
        'job_id': job_ghz.job_id,
        'backend': be_name,
        'counts': counts_ghz,
        'ghz_fidelity': ghz_fid,
        'latency_s': dt
    }
except Exception as e:
    print(f"  Error: {e}")
    RESULTS['ibm_ghz3'] = {'error': str(e)[:120]}

# ============================================================================
# 4. IonQ: List recent jobs & results
# ============================================================================
print("\n--- [IonQ] Recent Jobs ---")
try:
    from superfermion.runtime.providers.ionq import IonQProvider
    ionq = IonQProvider(api_key=IONQ_KEY)
    
    jobs = ionq.list_jobs(limit=10)
    print(f"  Last 10 IonQ jobs:")
    for j in jobs:
        jid = j.job_id if hasattr(j, 'job_id') else str(j)
        st = j.status.name if hasattr(j, 'status') and hasattr(j.status, 'name') else str(getattr(j, 'status', '?'))
        tgt = getattr(j, 'target', '?')
        print(f"    {jid[:30]:30s}  status={st:15s} target={tgt}")
        
        # Try to get result for completed jobs
        if 'completed' in st.lower() or 'done' in st.lower():
            try:
                res = j.result()
                if hasattr(res, 'counts') and res.counts:
                    top3 = sorted(res.counts.items(), key=lambda x: -x[1])[:3]
                    print(f"      Top 3: {top3}")
            except:
                pass
        RESULTS[f'ionq_job_{jid[:8]}'] = {'id': jid, 'status': st, 'target': str(tgt)}
except Exception as e:
    print(f"  Error: {e}")

# ============================================================================
# 5. IonQ: Submit a new Bell circuit to simulator
# ============================================================================
print("\n--- [IonQ] Submit Bell Circuit to Simulator ---")
try:
    bell_ionq = sf.Circuit(2)
    bell_ionq.h(0); bell_ionq.cx(0, 1)
    
    t0 = time.perf_counter()
    job_ionq = ionq.run(bell_ionq, backend="ionq.simulator", shots=1024)
    print(f"  Job ID: {job_ionq.job_id}")
    print(f"  Waiting for result...")
    
    result_ionq = job_ionq.result(timeout=60)
    dt = time.perf_counter() - t0
    
    counts_ionq = result_ionq.counts
    total_ionq = sum(counts_ionq.values()) if counts_ionq else 0
    print(f"  Result in {dt:.1f}s, {total_ionq} shots")
    if counts_ionq:
        for bs, cnt in sorted(counts_ionq.items(), key=lambda x: -x[1]):
            print(f"    {bs}: {cnt} ({cnt/total_ionq*100:.1f}%)")
    RESULTS['ionq_bell'] = {
        'job_id': job_ionq.job_id,
        'counts': counts_ionq,
        'latency_s': dt
    }
except Exception as e:
    print(f"  Error: {e}")
    RESULTS['ionq_bell'] = {'error': str(e)[:120]}

# ============================================================================
# 6. OpenQuantum: List jobs
# ============================================================================
print("\n--- [OpenQuantum] Recent Jobs ---")
try:
    from superfermion.runtime.providers.openquantum import OpenQuantumProvider
    oq = OpenQuantumProvider(client_id=OQ_CID, client_secret=OQ_SECRET)
    
    jobs_oq = oq.list_jobs(limit=5)
    print(f"  Last 5 OpenQuantum jobs:")
    for j in jobs_oq:
        jid = j.job_id if hasattr(j, 'job_id') else str(j)
        st = j.status.name if hasattr(j, 'status') and hasattr(j.status, 'name') else str(getattr(j, 'status', '?'))
        print(f"    {jid[:30]:30s}  status={st}")
        RESULTS[f'oq_job_{jid[:8]}'] = {'id': jid, 'status': st}
except Exception as e:
    print(f"  Error: {e}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("  SUMMARY")
print("=" * 70)
for key, val in RESULTS.items():
    if isinstance(val, dict) and 'error' in val:
        print(f"  {key}: ERROR - {val['error'][:60]}")
    elif isinstance(val, dict) and 'counts' in val:
        n = sum(val['counts'].values()) if val['counts'] else 0
        print(f"  {key}: {n} shots")
    elif isinstance(val, dict):
        print(f"  {key}: {val.get('status', '?')}")
    else:
        print(f"  {key}: {val}")

out = ROOT / 'notebooks' / 'qpu_results.json'
with open(out, 'w') as f:
    json.dump(RESULTS, f, indent=2, default=str)
print(f"\nResults saved to: {out}")
print("DONE.")
