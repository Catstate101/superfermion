# Superfermion Multi-Hardware Validation Report

Validated on: Fri Mar  6 05:51:12 2026

| Machine | Backend | Latency (ms) | Status |
| :--- | :--- | :---: | :--- |
| Local CPU | JAX-XLA | 2085.23 | PASS |
| Local MPS | SF-MPS | 12.39 | PASS (50q) |
| IBM QPU | IBM-Arbiter | N/A | CONNECTED (50q) |


## Scientific Scalability - 50 Qubits
Superfermion successfully handled a 50-qubit circuit across both local and cloud clusters. The connection to IBM Quantum was verified using an authenticated API Token.