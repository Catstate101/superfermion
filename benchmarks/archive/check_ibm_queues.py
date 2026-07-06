
import superfermion as sf
from superfermion.runtime.providers.ibm import IBMProvider

TOKEN = "l6ujymqMxhG7N6mO1-Vq4H0uh7Fkc-RyxA79kf74_jKX"

def check_queues():
    print("Fetching IBM Backend Status...")
    provider = IBMProvider(token=TOKEN)
    if not provider._service:
        print("Failed to initialize service.")
        return

    backends = provider._service.backends()
    print(f"{'Backend Name':<20} | {'Status':<15} | {'Pending Jobs':<12} | {'Qubits':<8}")
    print("-" * 65)
    
    available_backends = []
    for b in backends:
        status = b.status()
        pending = status.pending_jobs
        operational = b.status().operational
        if operational:
            print(f"{b.name:<20} | {'Operational':<15} | {pending:<12} | {b.num_qubits:<8}")
            available_backends.append((b.name, pending))
        else:
            print(f"{b.name:<20} | {'Offline':<15} | {'N/A':<12} | {b.num_qubits:<8}")
    
    if available_backends:
        best_backend = min(available_backends, key=lambda x: x[1])
        print(f"\nRecommended Backend (Shortest Queue): {best_backend[0]} ({best_backend[1]} pending jobs)")
        return best_backend[0]

if __name__ == "__main__":
    check_queues()
