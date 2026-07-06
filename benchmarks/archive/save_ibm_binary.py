
import superfermion as sf
from superfermion.runtime.providers.ibm import IBMProvider

TOKEN = "l6ujymqMxhG7N6mO1-Vq4H0uh7Fkc-RyxA79kf74_jKX"
JOB_ID = "d6ma55ofh9oc73enq0h0"

def save_binary():
    provider = IBMProvider(token=TOKEN)
    job = provider.retrieve_job(JOB_ID)
    result = job.result()
    counts = result.counts
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    
    with open("ibm_binary_output.txt", "w") as f:
        f.write(f"{'Bitstring':<30} | {'Counts':<10}\n")
        f.write("-" * 45 + "\n")
        total = sum(counts.values())
        for bits, count in sorted_counts[:30]:
            f.write(f"{bits:<30} | {count:<10}\n")

if __name__ == "__main__":
    save_binary()
