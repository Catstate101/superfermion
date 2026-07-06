
from qiskit_ibm_runtime import QiskitRuntimeService
TOKEN = "l6ujymqMxhG7N6mO1-Vq4H0uh7Fkc-RyxA79kf74_jKX"

def list_recent_jobs():
    service = QiskitRuntimeService(channel="ibm_quantum_platform", token=TOKEN)
    print("Recent Jobs:")
    for job in service.jobs(limit=5):
        print(f"ID: {job.job_id()} | Backend: {job.backend().name} | Status: {job.status()}")

if __name__ == "__main__":
    list_recent_jobs()
