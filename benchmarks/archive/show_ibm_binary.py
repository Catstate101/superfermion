
import superfermion as sf
from superfermion.runtime.providers.ibm import IBMProvider

TOKEN = "l6ujymqMxhG7N6mO1-Vq4H0uh7Fkc-RyxA79kf74_jKX"
JOB_ID = "d6ma55ofh9oc73enq0h0" # Majorana experiment

def show_binary_results():
    print(f"--- Fecthing RAW Binary/Bitstring results from IBM: {JOB_ID} ---")
    try:
        provider = IBMProvider(token=TOKEN)
        job = provider.retrieve_job(JOB_ID)
        
        if "COMPLETED" in str(job.status) or "DONE" in str(job.status):
            result = job.result()
            counts = result.counts
            
            # Sort by frequency
            sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
            
            print(f"\n{'Bitstring [q5 q4 q3 q2 q1 q0]':<30} | {'Counts':<10} | {'% Percentage'}")
            print("-" * 65)
            total = sum(counts.values())
            
            # Show top 20 binary results
            for i in range(min(20, len(sorted_counts))):
                bits, count = sorted_counts[i]
                perc = (count / total) * 100
                print(f"{bits:<30} | {count:<10} | {perc:.2f}%")
                
            print("\nNote: In the Majorana experiment, we look for correlation between the first (rightmost) and last (leftmost) bits.")
        else:
            print("Job is not finished.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    show_binary_results()
