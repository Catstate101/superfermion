
import superfermion as sf
from superfermion.runtime.providers.ibm import IBMProvider

TOKEN = "l6ujymqMxhG7N6mO1-Vq4H0uh7Fkc-RyxA79kf74_jKX"
JOB_ID = "d6m9ahk3pels73a05mng"

def check_job():
    print(f"Checking status for Job ID: {JOB_ID}...")
    try:
        provider = IBMProvider(token=TOKEN)
        job = provider.retrieve_job(JOB_ID)
        
        status = job.status
        print(f"Current Status: {status}")
        
        if "COMPLETED" in str(status) or "DONE" in str(status):
            print("\nJob is finished! Retrieving results...")
            result = job.result()
            print("Measurement Counts:")
            print(result.counts)
            
            # Analysis of success
            # Class bit format on IBM for this circuit: [c2, c1, c0]
            # Success is c2 = 1 (if we used XOR) or c0 == c1 (manual check)
            # In our BBPSSW script, we measured q0 -> c0 and q3 -> c1
            successes = 0
            total = sum(result.counts.values())
            for bits, count in result.counts.items():
                # bits is usually a string like '00', '01' etc. (c1, c0)
                if len(bits) >= 2:
                    if bits[-1] == bits[-2]: # c0 == c1
                        successes += count
            
            print(f"\nDistillation Success Rate: {(successes/total)*100:.2f}%")
        else:
            print("\nThe job is still in the IBM queue. Real hardware has a wait time based on other users in the cloud.")
            
    except Exception as e:
        print(f"Error checking job: {e}")

if __name__ == "__main__":
    check_job()
