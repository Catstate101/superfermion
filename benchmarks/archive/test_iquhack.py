
import json
import numpy as np
import superfermion as sf
from superfermion.circuit import Circuit
from superfermion.simulator import simulate_statevector

def validate_protocols():
    print("=== Superfermion iQuHACK 2026 Validation ===\n")
    
    # 1. Validate BBPSSW Bridge
    print("Testing BBPSSW Circuit Generation...")
    c = Circuit(4, 3)
    c.cnot(1, 0).cnot(2, 3)
    c.measure(0, 0).measure(3, 1)
    
    print("ASCII Drawing:")
    print(c.draw())
    
    print("\nQASM 3.0 Export (Targeting server compatibility):")
    qasm = c.to_qasm3()
    print(qasm)
    
    # 2. Local Simulation (Ideal case verification)
    print("\nSimulating Ideal Distillation Circuit...")
    # Add prep to see non-trivial results
    sim_c = Circuit(4, 3)
    sim_c.h(1).cnot(1, 2) # Prep data Bell pair
    sim_c.h(0).cnot(0, 3) # Prep ancilla Bell pair
    sim_c.cnot(1, 0).cnot(2, 3)
    sim_c.measure(0, 0).measure(3, 1)
    
    sv = simulate_statevector(sim_c)
    # Probs of basic states
    probs = np.abs(sv)**2
    top_indices = np.argsort(probs)[-4:]
    print("Top simulation outcomes (ideal prep):")
    for i in top_indices:
        if probs[i] > 0.01:
            print(f"  |{i:04b}> : {probs[i]:.4f}")

    # 3. IBM Job Verification (Snapshot of current status)
    print("\nChecking IBM Job Status...")
    TOKEN = "l6ujymqMxhG7N6mO1-Vq4H0uh7Fkc-RyxA79kf74_jKX"
    from superfermion.runtime.providers.ibm import IBMProvider
    try:
        provider = IBMProvider(token=TOKEN)
        job = provider.retrieve_job("d6m9ahk3pels73a05mng")
        print(f"Job ID: {job.job_id}")
        print(f"Backend: {job.backend_name}")
        print(f"Status: {job.status}")
    except Exception as e:
        print(f"Could not reach IBM for status check: {e}")

if __name__ == "__main__":
    validate_protocols()
