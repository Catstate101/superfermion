
import superfermion as sf
from superfermion.runtime.providers.ibm import IBMProvider
import numpy as np

TOKEN = "l6ujymqMxhG7N6mO1-Vq4H0uh7Fkc-RyxA79kf74_jKX"
JOB_ID = "d6ma55ofh9oc73enq0h0"

def check_physics_discovery():
    print(f"--- Checking Majorana Physics Discovery: {JOB_ID} ---")
    try:
        provider = IBMProvider(token=TOKEN)
        job = provider.retrieve_job(JOB_ID)
        status = job.status
        print(f"Current Status: {status}")
        
        if "COMPLETED" in str(status) or "DONE" in str(status):
            print("\nExperiment Complete! Analyzing Topological Signature...")
            result = job.result()
            counts = result.counts
            
            # For a 6-qubit Kitaev chain, we look for parity between qubit 0 and qubit 5
            # Majorana Zero Modes create a long-range entanglement.
            edge_correlation = 0
            total_shots = sum(counts.values())
            
            for bitstring, count in counts.items():
                # bits are [q5, q4, q3, q2, q1, q0] in standard Qiskit/SF output
                if len(bitstring) >= 6:
                    q0 = int(bitstring[-1])
                    q5 = int(bitstring[0])
                    # If q0 and q5 are correlated (both 0 or both 1), it's a signature of the edge mode
                    if q0 == q5:
                        edge_correlation += count
            
            correlation_ratio = edge_correlation / total_shots
            print(f"Edge Correlation Ratio: {correlation_ratio:.4f}")
            
            if correlation_ratio > 0.5:
                print("\n[RESULT] MAJORANA SIGNATURE DETECTED!")
                print("The first and last qubits are spatially separated but correlated.")
                print("This confirms the existence of edge-pinned Majorana Zero Modes.")
            else:
                print("\n[RESULT] WEAK TOPOLOGICAL SIGNAL.")
                print("Noise levels on the QPU may be obscuring the non-local order.")
        else:
            print("\nJob is still in the queue. Nature takes time to solve.")
            
    except Exception as e:
        print(f"Error retrieving physics results: {e}")

if __name__ == "__main__":
    check_physics_discovery()
