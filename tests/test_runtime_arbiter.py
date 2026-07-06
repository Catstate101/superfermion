"""
Unit test for Resource Arbiter and Cloud Routing.
"""

import superfermion as sf
from superfermion.runtime.arbiter import ResourceArbiter

def test_arbiter_routing():
    arbiter = ResourceArbiter()
    
    # 2 qubits should go to simulator or local
    backend_local = arbiter.route(n_qubits=2)
    print(f"Backend for 2 qubits: {backend_local}")
    assert "simulator" in backend_local or "statevector" in backend_local or "jax" in backend_local
    
    # 36 qubits falls in the "industrial gap" (SV_MAX < n <= QPU_MIN)
    # and should route to an MPS simulator (jax_mps on GPU/TPU, mps on CPU).
    backend_midsize = arbiter.route(n_qubits=36)
    assert backend_midsize in ("mps", "jax_mps")

    # 200 qubits exceeds classical simulation limits, must go to cloud hardware.
    backend_cloud = arbiter.route(n_qubits=200)
    assert backend_cloud == "ibm"

def test_arbiter_route_alias():
    arbiter = ResourceArbiter()
    # Test the alias I added earlier
    backend = arbiter.route(n_qubits=2)
    assert backend is not None

if __name__ == "__main__":
    test_arbiter_routing()
    test_arbiter_route_alias()
    print("Runtime Arbiter Tests: PASS")
