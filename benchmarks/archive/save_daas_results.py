
import superfermion as sf
from superfermion.runtime.providers.ibm import IBMProvider
import numpy as np

TOKEN = "l6ujymqMxhG7N6mO1-Vq4H0uh7Fkc-RyxA79kf74_jKX"
JOB_IDS = ["d6m9ooobfi7c73a3adng", "d6m9nrgfh9oc73enpicg"]

def save_results():
    provider = IBMProvider(token=TOKEN)
    with open("daas_results_summary.txt", "w") as f:
        for job_id in JOB_IDS:
            f.write(f"\n--- Result for Job: {job_id} ---\n")
            try:
                job = provider.retrieve_job(job_id)
                status = job.status
                f.write(f"Status: {status}\n")
                
                if "COMPLETED" in str(status) or "DONE" in str(status):
                    result = job.result()
                    counts = result.counts
                    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
                    
                    f.write("\nTop 5 Optimized Portfolio Configurations:\n")
                    f.write(f"{'Asset Bitstring':<15} | {'Count':<10} | {'Prob (%)'}\n")
                    f.write("-" * 45 + "\n")
                    
                    total = sum(counts.values())
                    for i in range(min(5, len(sorted_counts))):
                        bitstring, count = sorted_counts[i]
                        f.write(f"{bitstring:<15} | {count:<10} | {(count/total)*100:.2f}%\n")
                else:
                    f.write("Job not done.\n")
            except Exception as e:
                f.write(f"Error: {e}\n")

if __name__ == "__main__":
    save_results()
