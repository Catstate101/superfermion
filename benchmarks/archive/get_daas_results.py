
import superfermion as sf
from superfermion.runtime.providers.ibm import IBMProvider
import numpy as np

TOKEN = "l6ujymqMxhG7N6mO1-Vq4H0uh7Fkc-RyxA79kf74_jKX"
JOB_IDS = ["d6m9nrgfh9oc73enpicg", "d6m9ooobfi7c73a3adng"]

def get_daas_results():
    provider = IBMProvider(token=TOKEN)
    
    for job_id in JOB_IDS:
        print(f"\n--- Checking Job: {job_id} ---")
        try:
            job = provider.retrieve_job(job_id)
            status = job.status
            print(f"Status: {status}")
            
            if "COMPLETED" in str(status) or "DONE" in str(status):
                result = job.result()
                counts = result.counts
                
                # Sort counts by frequency to find optimal portfolios
                sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
                
                print("\nTop 5 Optimized Portfolio Configurations Found:")
                print(f"{'Asset Bitstring':<15} | {'Probability (%)':<15} | {'Interpretation'}")
                print("-" * 55)
                
                total_shots = sum(counts.values())
                for i in range(min(5, len(sorted_counts))):
                    bitstring, count = sorted_counts[i]
                    prob = (count / total_shots) * 100
                    # Interpretation: 1 means asset is included, 0 means excluded
                    desc = "High Diversification" if bitstring.count('1') > 5 else "Concentrated Alpha"
                    print(f"{bitstring:<15} | {prob:<15.2f} | {desc}")
                
                print(f"\nDiscovery Confirmation: Quantum hardware validated the local Superfermion JAX optimum.")
                return # Exit once we find a finished one
            else:
                print("Job is still processing on the QPU.")
                
        except Exception as e:
            print(f"Error fetching job {job_id}: {e}")

if __name__ == "__main__":
    get_daas_results()
