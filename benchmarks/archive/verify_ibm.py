
import superfermion as sf
from superfermion.runtime.providers.ibm import IBMProvider

TOKEN = "l6ujymqMxhG7N6mO1-Vq4H0uh7Fkc-RyxA79kf74_jKX"

def verify():
    print("Connecting to ibm_quantum_platform...")
    try:
        provider = IBMProvider(token=TOKEN)
        print("Connected successfully!")
        
        # Check available backends
        if provider._service:
            print("Available backends:")
            for b in provider._service.backends():
                print(f" - {b.name}")
        
    except Exception as e:
        print(f"Connection failed: {e}")

if __name__ == "__main__":
    verify()
