
import os
import sys

def check_gpu():
    print(f"Python Version: {sys.version}")
    
    # 1. JAX
    try:
        import jax
        print(f"JAX Version: {jax.__version__}")
        print(f"JAX Default Backend: {jax.default_backend()}")
        print(f"JAX Devices: {jax.devices()}")
    except Exception as e:
        print(f"JAX GPU Error: {e}")
        
    # 2. TensorFlow
    try:
        import tensorflow as tf
        print(f"TensorFlow Version: {tf.__version__}")
        gpu_devices = tf.config.list_physical_devices('GPU')
        print(f"TensorFlow GPU Devices: {gpu_devices}")
    except Exception as e:
        print(f"TensorFlow GPU Error: {e}")
        
    # 3. Qiskit/Aer
    try:
        from qiskit_aer import Aer
        backend = Aer.get_backend('statevector_simulator')
        print(f"Qiskit Aer GPU Available: {backend.configuration().device if hasattr(backend.configuration(), 'device') else 'Unknown'}")
        # Try to set GPU
        try:
             backend.set_options(device='GPU')
             print("Qiskit Aer GPU set successfully.")
        except Exception as e:
             print(f"Qiskit Aer GPU failed: {e}")
    except Exception as e:
        print(f"Qiskit Aer GPU Error: {e}")

if __name__ == "__main__":
    check_gpu()
