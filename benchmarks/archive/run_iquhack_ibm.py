
import superfermion as sf
from superfermion.runtime.providers.ibm import IBMProvider
from superfermion.circuit import Circuit

TOKEN = "l6ujymqMxhG7N6mO1-Vq4H0uh7Fkc-RyxA79kf74_jKX"

def run_distillation():
    # Load provider
    provider = IBMProvider(token=TOKEN)
    
    # Build the iQuHACK distillation circuit
    # This circuit corresponds to the logic in the notebook
    # We add Bell pair prep so we're not just measuring zero
    c = Circuit(4, 3)
    
    # --- Bell Pair Preparation (The 'entanglement' being distilled) ---
    c.h(1).cnot(1, 2) # Data pair (q1, q2)
    c.h(0).cnot(0, 3) # Ancilla pair (q0, q3)
    c.barrier()
    
    # --- Distillation Logic (BBPSSW) ---
    c.cnot(1, 0)  # Alice: local CNOT data -> ancilla
    c.cnot(2, 3)  # Bob: local CNOT data -> ancilla
    c.barrier()
    
    # --- Measurement ---
    c.measure(0, 0) # Measure Alice's ancilla
    c.measure(3, 1) # Measure Bob's ancilla
    
    # Note: On IBM, we don't do real-time XOR in the circuit for success check 
    # unless using dynamic circuits, so we process it in the results.
    
    print("Submitting to IBM Quantum (ibm_fez)...")
    try:
        job = provider.run(c, backend="ibm_fez")
        print(f"Job ID: {job.job_id}")
        print(f"Current Status: {job.status}")
        
        # We'll wait a few seconds to see if it moves past 'CREATED'
        import time
        time.sleep(5)
        print(f"Updated Status: {job.status}")
        
    except Exception as e:
        print(f"Failed to submit: {e}")

if __name__ == "__main__":
    run_distillation()
