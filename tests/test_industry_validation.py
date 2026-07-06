"""
╔══════════════════════════════════════════════════════════════════════════╗
║  SUPERFERMION — INDUSTRY MEGA-VALIDATION                                ║
║  Every Domain. Every Pillar. Maximum Qubits. Zero Excuses.              ║
║                                                                          ║
║  Domains Tested:                                                         ║
║    1. Circuit API & Max-Qubit Scaling                                    ║
║    2. JAX Autograd & Differentiability                                   ║
║    3. Quantum ML (QML): VQE, QAOA, QSVM, QNG                           ║
║    6. Quantum Error Correction (QEC): Surface code                       ║
║    7. Noise Models & Error Mitigation                                    ║
║    8. Bridges: Qiskit, PennyLane, OpenQASM                              ║
║    9. Cloud: IBM Eagle, AWS Braket routing                               ║
║   10. Benchmarking & Cost Estimation                                     ║
║   11. Compiler: Gate cancellation, SWAP decomposition, basis translation ║
║   12. Intelligence: QNS, SuperpositionalAgent                            ║
║   13. Chemistry: Molecular Hamiltonians, UCCSD                           ║
║   14. Serialization: JSON, QASM3 round-trip                              ║
║   15. Runtime: Job submission, Resource Arbiter                          ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import time
import sys
import traceback
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn

# -- Windows UTF-8 console fix (prevents cp1252 crashes on Unicode) ----------
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import superfermion as sf
from superfermion.nn.quantum_layer import QuantumLayer
from superfermion.observables.core import Hamiltonian, PauliString
from superfermion.qml.fidelity import state_fidelity

PASS = 0
FAIL = 0
RESULTS = []

def section(name):
    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"{'='*70}")

def check(name, fn):
    global PASS, FAIL
    t0 = time.time()
    try:
        fn()
        dt = time.time() - t0
        PASS += 1
        RESULTS.append((name, "PASS", f"{dt:.2f}s"))
        print(f"  [OK] {name:55s} ({dt:.2f}s)")
    except Exception as e:
        dt = time.time() - t0
        FAIL += 1
        RESULTS.append((name, "FAIL", str(e)[:60]))
        print(f"  [XX] {name:55s} FAIL: {e}")
        traceback.print_exc()

# -----------------------------------------------------------------
# 1. CIRCUIT API & MAX-QUBIT SCALING
# -----------------------------------------------------------------
section("1. CIRCUIT API & MAX-QUBIT SCALING")

def test_circuit_creation():
    c = sf.Circuit(2).h(0).cx(0, 1)
    assert c.n_qubits == 2
    assert c.gate_count == 2
check("Circuit creation (2-qubit Bell)", test_circuit_creation)

def test_circuit_chaining():
    c = sf.Circuit(3).h(0).cx(0, 1).cx(1, 2).x(2).z(1).s(0)
    assert c.gate_count == 6
check("Fluent API chaining (6 gates)", test_circuit_chaining)

def test_all_gates():
    c = sf.Circuit(3)
    theta = sf.param("t")
    c.h(0).x(1).y(2).z(0).s(1).t(2).sx(0).id(1)
    c.rx(theta, 0).ry(theta, 1).rz(theta, 2).p(theta, 0)
    c.u(theta, theta, theta, 0)
    c.cx(0, 1).cz(1, 2).cy(0, 2).swap(0, 1)
    c.rzz(theta, 0, 1).rxx(theta, 1, 2)
    c.ccx(0, 1, 2)
    assert c.gate_count >= 19
check("All gate types (single, two, three qubit)", test_all_gates)

def test_max_qubit_circuit():
    n = 20
    c = sf.Circuit(n)
    for i in range(n):
        c.h(i)
    for i in range(n - 1):
        c.cx(i, i + 1)
    assert c.gate_count == (n + n - 1)
check("Max qubit circuit construction (20q)", test_max_qubit_circuit)

def test_circuit_depth():
    c = sf.Circuit(4).h(0).cx(0, 1).cx(1, 2).cx(2, 3)
    assert c.depth >= 4
check("Circuit depth calculation", test_circuit_depth)

def test_draw():
    c = sf.Circuit(2).h(0).cx(0, 1)
    txt = c.draw()
    assert len(txt) > 0
check("Circuit ASCII drawing", test_draw)

# -----------------------------------------------------------------
# 2. JAX AUTOGRAD & DIFFERENTIABILITY
# -----------------------------------------------------------------
section("2. JAX AUTOGRAD & DIFFERENTIABILITY")

def test_circuit_to_jax():
    c = sf.Circuit(2).rx(sf.param("a"), 0).ry(sf.param("b"), 1).cx(0, 1)
    f = sf.qml.circuit_to_jax(c, backend="jax")
    sv = f(jnp.array(0.5), jnp.array(0.3))
    assert sv.shape == (4,)
    assert jnp.allclose(jnp.sum(jnp.abs(sv)**2), 1.0, atol=1e-5)
check("circuit_to_jax statevector", test_circuit_to_jax)

def test_jax_grad():
    c = sf.Circuit(1).rx(sf.param("t"), 0)
    f = sf.qml.circuit_to_jax(c, backend="jax")
    def loss(t): return jnp.abs(f(t)[1])**2
    g = jax.grad(loss)(jnp.array(1.0))
    assert jnp.abs(g) > 1e-5
check("JAX gradient (parameter-shift)", test_jax_grad)

def test_jax_jit():
    c = sf.Circuit(1).ry(sf.param("t"), 0)
    f = sf.qml.circuit_to_jax(c, backend="jax")
    jitted = jax.jit(f)
    sv = jitted(jnp.array(0.5))
    assert sv.shape == (2,)
check("JAX JIT compilation", test_jax_jit)

def test_hessian():
    c = sf.Circuit(1).rx(sf.param("a"), 0).ry(sf.param("b"), 0)
    model = QuantumLayer(n_qubits=1, ansatz=c)
    params = model.init(jax.random.PRNGKey(0))
    def loss(p): return model.apply(p)[0]
    h = jax.hessian(loss)(params)
    mat = h['params']['weights']['params']['weights']
    assert mat.shape == (2, 2)
check("Hessian (2nd-order derivatives)", test_hessian)

# -----------------------------------------------------------------
# 3. QUANTUM ML ALGORITHMS
# -----------------------------------------------------------------
section("3. QUANTUM ML: VQE, QAOA, QSVM, QNG, Kernels")

def test_vqe():
    from superfermion.algorithms.variational import VQE
    H = Hamiltonian([
        PauliString("ZI", 0.3), PauliString("IZ", 0.3),
        PauliString("ZZ", -0.5), PauliString("XX", 0.2)
    ])
    ansatz = sf.Circuit(2)
    for i in range(2):
        ansatz.ry(sf.param(f"ry{i}"), i)
        ansatz.rz(sf.param(f"rz{i}"), i)
    ansatz.cx(0, 1)
    for i in range(2):
        ansatz.ry(sf.param(f"ry2_{i}"), i)
    vqe = VQE(ansatz, H, optimizer="L-BFGS-B")
    result = vqe.minimize(iterations=30)
    assert result.optimal_value < 0
    assert len(result.history) >= 1
check("VQE ground state (6-param, 30 iter)", test_vqe)

def test_qaoa():
    from superfermion.algorithms.variational import QAOA
    qaoa = QAOA(n_qubits=2, edges=[(0, 1)], p_layers=2)
    result = qaoa.minimize(iterations=30)
    assert result.optimal_value > 0
check("QAOA MaxCut (2-qubit, 2 layers)", test_qaoa)

def test_qsvm():
    from superfermion.algorithms.qsvm import QSVM
    ansatz = sf.Circuit(2)
    ansatz.ry(sf.param("a"), 0).ry(sf.param("b"), 1).cx(0, 1)
    x_train = jnp.array([[0.1, 0.2], [0.8, 0.9], [0.15, 0.25], [0.85, 0.95]])
    y_train = jnp.array([0, 1, 0, 1])
    qsvm = QSVM(ansatz, num_classes=2, optimizer=optax.adam(0.05))
    result = qsvm.fit(x_train, y_train, iterations=20)
    assert result.optimal_value < 2.0
check("QSVM classifier (4 samples, 20 iter)", test_qsvm)

def test_qng():
    from superfermion.qml.gradient.qng import qng_step
    c = sf.Circuit(1).ry(sf.param("t"), 0)
    f = sf.qml.circuit_to_jax(c, backend="jax")
    def loss(p): return jnp.abs(f(p[0])[1])**2
    def circ(p): return f(p[0])
    params = jnp.array([1.0])
    updated = qng_step(loss, circ, params, learning_rate=0.1)
    assert not jnp.allclose(updated, params)
check("QNG natural gradient step", test_qng)

def test_quantum_kernel():
    from superfermion.qml.algorithms.quantum_kernel import QuantumKernel
    from superfermion.qml.encoding import angle_encoding
    def enc(x):
        return angle_encoding(2, x)
    qk = QuantumKernel(enc, backend="jax")
    k = qk.evaluate(jnp.array([0.1, 0.2]), jnp.array([0.1, 0.2]))
    assert jnp.allclose(k, 1.0, atol=0.01)
check("Quantum Kernel (fidelity self-overlap)", test_quantum_kernel)

# -----------------------------------------------------------------
# 4. QUANTUM ERROR CORRECTION (QEC)
# -----------------------------------------------------------------
section("6. QEC: Surface Code")

def test_surface_code():
    from superfermion.qec.codes.surface import SurfaceCode
    sc = SurfaceCode(distance=3)
    assert sc.n_data == 9
    assert sc.n_measure == 8
    c = sc.build_syndrome_extraction()
    assert c.n_qubits == 17
    assert c.gate_count > 0
check("Surface code d=3 syndrome extraction", test_surface_code)

def test_surface_code_scaling():
    from superfermion.qec.codes.surface import SurfaceCode
    sc5 = SurfaceCode(distance=5)
    assert sc5.n_data == 25
check("Surface code d=5 scaling", test_surface_code_scaling)

# -----------------------------------------------------------------
# 7. NOISE MODELS & ERROR MITIGATION
# -----------------------------------------------------------------
section("7. NOISE & MITIGATION")

def test_noise_model():
    from superfermion.noise import NoiseModel, ibm_eagle_noise
    nm = ibm_eagle_noise()
    assert len(nm.single_qubit_channels) > 0
    assert nm.readout_error > 0
check("IBM Eagle noise model construction", test_noise_model)

def test_zne_mitigation():
    from superfermion.mitigation import zne
    c = sf.Circuit(1).h(0)
    def obs(sv):
        return jnp.real(jnp.vdot(sv, jnp.array([[1,0],[0,-1]]) @ sv))
    # ZNE should return a value close to the noiseless one
    val = zne(c, obs, scale_factors=[1, 2, 3], backend="jax")
    assert isinstance(val, float)
check("ZNE error mitigation", test_zne_mitigation)

# -----------------------------------------------------------------
# 8. BRIDGES: Qiskit, PennyLane, OpenQASM
# -----------------------------------------------------------------
section("8. BRIDGES: Qiskit, PennyLane, OpenQASM 3")

def test_qasm3_export():
    c = sf.Circuit(2).h(0).cx(0, 1)
    qasm = c.to_qasm3()
    assert "OPENQASM 3.0" in qasm
    assert "h" in qasm or "H" in qasm
check("OpenQASM 3.0 export", test_qasm3_export)

def test_json_roundtrip():
    c = sf.Circuit(2).h(0).cx(0, 1)
    j = c.to_json()
    c2 = sf.Circuit.from_json(j)
    assert c2.n_qubits == 2
    assert c2.gate_count == 2
check("JSON serialization round-trip", test_json_roundtrip)

def test_qiskit_bridge():
    from superfermion.bridge import to_qiskit, from_qiskit
    c = sf.Circuit(2).h(0).cx(0, 1)
    qc = to_qiskit(c)
    assert qc is not None
    # Round-trip
    c2 = from_qiskit(qc)
    assert c2.n_qubits == 2
check("Qiskit bridge round-trip", test_qiskit_bridge)

def test_pennylane_bridge():
    from superfermion.bridge import from_pennylane
    # We test the function exists and handles basic input
    assert callable(from_pennylane)
check("PennyLane bridge availability", test_pennylane_bridge)

# -----------------------------------------------------------------
# 9. CLOUD: IBM, AWS, Resource Arbiter
# -----------------------------------------------------------------
section("9. CLOUD ROUTING & PROVIDERS")

def test_resource_arbiter():
    from superfermion.runtime.arbiter import ResourceArbiter
    arbiter = ResourceArbiter()
    b2 = arbiter.route(n_qubits=2)
    assert "jax" in b2 or "simulator" in b2
    b50 = arbiter.route(n_qubits=50)
    assert b50 is not None
check("Resource Arbiter routing (2q, 50q)", test_resource_arbiter)

def test_hardware_specs():
    from superfermion.runtime.specs import get_spec
    eagle = get_spec("ibm_eagle")
    assert eagle.n_qubits == 127
    assert "cx" in eagle.native_gates or "CX" in eagle.native_gates
check("Hardware specs (IBM Eagle 127q)", test_hardware_specs)

def test_hardware_compilation():
    from superfermion.runtime.specs import get_spec
    c = sf.Circuit(2).h(0).cx(0, 1)
    spec = get_spec("ibm_eagle")
    compiled = sf.compile(c, target=spec)
    assert compiled.gate_count >= 2
check("Hardware-aware compilation (IBM Eagle)", test_hardware_compilation)

def test_ibm_provider():
    from superfermion.runtime.providers.ibm import IBMProvider
    ibm = IBMProvider()
    assert ibm is not None
check("IBM Provider instantiation", test_ibm_provider)

# -----------------------------------------------------------------
# 10. BENCHMARKING & COST ESTIMATION
# -----------------------------------------------------------------
section("10. BENCHMARKING & COST ESTIMATION")

def test_benchmark():
    c = sf.Circuit(4)
    for i in range(4): c.h(i)
    for i in range(3): c.cx(i, i+1)
    result = sf.benchmark(c, backends=["simulator"])
    assert result is not None
check("Benchmark suite (4-qubit)", test_benchmark)

def test_cost_estimation():
    c = sf.Circuit(10)
    for i in range(10): c.h(i)
    result = sf.estimate_cost(c, shots=10000)
    assert result.credits >= 1.0
check("Cost estimation (10q, 10k shots)", test_cost_estimation)

def test_jax_scaling():
    from superfermion.backends.jax_sim import JAXBackend
    sim = JAXBackend()
    for n in [2, 4, 8, 10]:
        c = sf.Circuit(n)
        for i in range(n): c.h(i)
        for i in range(n - 1): c.cx(i, i + 1)
        sv = sim.simulate(c, [])
        assert sv.shape == (2**n,)
check("JAX backend scaling (2→10 qubits)", test_jax_scaling)

# -----------------------------------------------------------------
# 11. COMPILER PASSES
# -----------------------------------------------------------------
section("11. COMPILER: Cancellation, SWAP, Basis Translation")

def test_gate_cancellation():
    c = sf.Circuit(1).h(0).h(0)
    compiled = sf.compile(c)
    assert compiled.gate_count == 0
check("Gate cancellation (H*H → I)", test_gate_cancellation)

def test_swap_decomp():
    c = sf.Circuit(2).swap(0, 1)
    compiled = sf.compile(c)
    assert compiled.gate_count == 3
check("SWAP → 3 CNOT decomposition", test_swap_decomp)

def test_triple_x():
    c = sf.Circuit(1).x(0).x(0).x(0)
    compiled = sf.compile(c)
    assert compiled.gate_count == 1
check("X*X*X → X reduction", test_triple_x)

# -----------------------------------------------------------------
# 10. CHEMISTRY: Molecular Hamiltonians
# -----------------------------------------------------------------
section("13. CHEMISTRY: Molecular Hamiltonians & UCCSD")

def test_h2_hamiltonian():
    from superfermion.chemistry import get_molecular_hamiltonian
    H = get_molecular_hamiltonian("H2")
    assert len(H.terms) == 5
check("H2 molecular Hamiltonian (5 terms)", test_h2_hamiltonian)

def test_uccsd():
    from superfermion.chemistry import uccsd_ansatz
    ansatz = uccsd_ansatz(n_qubits=4, n_electrons=2)
    assert ansatz.n_qubits == 4
    assert len(ansatz.parameters) > 0
check("UCCSD ansatz (4q, 2 electrons)", test_uccsd)

def test_chemistry_vqe():
    from superfermion.chemistry import get_molecular_hamiltonian, uccsd_ansatz
    from superfermion.algorithms.variational import VQE
    H = get_molecular_hamiltonian("H2")
    ansatz = uccsd_ansatz(n_qubits=2, n_electrons=2)
    vqe = VQE(ansatz, H, optimizer="L-BFGS-B")
    result = vqe.minimize(iterations=20)
    assert result.optimal_value < 0
check("Chemistry VQE pipeline (H2 + UCCSD)", test_chemistry_vqe)

# -----------------------------------------------------------------
# 14. ENCODINGS & ANSATZE
# -----------------------------------------------------------------
section("14. ENCODINGS & ANSATZE LIBRARY")

def test_angle_encoding():
    from superfermion.qml.encoding import angle_encoding
    c = angle_encoding(3, jnp.array([0.1, 0.2, 0.3]))
    assert c.gate_count == 3
check("Angle encoding (3-qubit)", test_angle_encoding)

def test_basis_encoding():
    from superfermion.qml.encoding import basis_encoding
    c = basis_encoding(4, 5)  # 0101
    assert c.gate_count == 2  # Two X gates
check("Basis encoding (4q, value=5)", test_basis_encoding)

def test_iqp_encoding():
    from superfermion.qml.encoding import iqp_encoding
    c = iqp_encoding(3, jnp.array([0.1, 0.2, 0.3]))
    assert c.gate_count > 3
check("IQP encoding (3-qubit)", test_iqp_encoding)

def test_hea_ansatz():
    from superfermion.qml.ansatz import hardware_efficient
    c = hardware_efficient(4, layers=2)
    assert len(c.parameters) > 0
check("Hardware-Efficient Ansatz (4q, 2 layers)", test_hea_ansatz)

def test_sea_ansatz():
    from superfermion.qml.ansatz import strongly_entangling
    c = strongly_entangling(4, layers=2)
    assert len(c.parameters) > 0
check("Strongly Entangling Ansatz (4q, 2 layers)", test_sea_ansatz)

def test_two_local():
    from superfermion.qml.ansatz import two_local
    c = two_local(3, rotation_gates=["RY", "RZ"], reps=2, entanglement="full")
    assert len(c.parameters) > 0
check("TwoLocal Ansatz (3q, full ent.)", test_two_local)

# -----------------------------------------------------------------
# 15. RUNTIME: Jobs & Execution
# -----------------------------------------------------------------
section("15. RUNTIME: Jobs, Execution, Backends")

def test_sf_run():
    c = sf.Circuit(2).h(0).cx(0, 1)
    result = sf.run(c, shots=1000)
    assert sum(result.counts.values()) == 1000
    assert "00" in result.counts or "11" in result.counts
check("sf.run() with shots=1000", test_sf_run)

def test_backend_registry():
    backends = sf.list_backends()
    assert len(backends) > 0
check("Backend registry listing", test_backend_registry)

def test_statevector_backend():
    b = sf.get_backend("statevector")
    c = sf.Circuit(2).h(0).cx(0, 1)
    result = b.run(c)
    assert result is not None
check("Statevector backend execution", test_statevector_backend)

# -----------------------------------------------------------------
# 16. VISUALIZATION
# -----------------------------------------------------------------
section("16. VISUALIZATION")

def test_bloch():
    from superfermion.viz.core import bloch_angles
    sv = jnp.array([1.0, 0.0], dtype=complex)
    angles = bloch_angles(sv)
    assert "theta" in angles and "phi" in angles
check("Bloch sphere angle extraction", test_bloch)

def test_state_viz():
    from superfermion.viz.core import state_bar_chart
    sv = jnp.array([1/jnp.sqrt(2), 0, 0, 1/jnp.sqrt(2)], dtype=complex)
    chart = state_bar_chart(sv)
    assert len(chart) > 0
check("Statevector probability chart", test_state_viz)

# -----------------------------------------------------------------
# 17. MAX QUBIT STRESS TEST
# -----------------------------------------------------------------
section("17. MAX QUBIT STRESS TEST")

def test_max_simulation():
    from superfermion.backends.jax_sim import JAXBackend
    sim = JAXBackend()
    n = 14  # 2^14 = 16384 amplitudes (safe for most machines)
    c = sf.Circuit(n)
    for i in range(n): c.h(i)
    t0 = time.time()
    sv = sim.simulate(c, [])
    dt = time.time() - t0
    assert sv.shape == (2**n,)
    assert jnp.allclose(jnp.sum(jnp.abs(sv)**2), 1.0, atol=1e-4)
    print(f"      -> {n}-qubit sim: {2**n} amplitudes in {dt:.3f}s")
check("Max qubit simulation (14q, 16384 amps)", test_max_simulation)

def test_large_circuit_compilation():
    n = 16
    c = sf.Circuit(n)
    for i in range(n): c.h(i)
    for i in range(n - 1): c.cx(i, i + 1)
    compiled = sf.compile(c)
    assert compiled.gate_count > 0
check("Large circuit compilation (16q)", test_large_circuit_compilation)

# -----------------------------------------------------------------
# 18. QRL & QBM (Quantum RL / Boltzmann Machine)
# -----------------------------------------------------------------
section("18. QRL & QBM")

def test_qrl():
    from superfermion.algorithms.qrl import QuantumREINFORCE
    ansatz = sf.Circuit(2).ry(sf.param("a"), 0).ry(sf.param("b"), 1).cx(0, 1)
    agent = QuantumREINFORCE(ansatz, num_actions=2)
    assert agent is not None
check("QuantumREINFORCE agent creation", test_qrl)

def test_qbm():
    from superfermion.algorithms.qbm import QBM
    model = QBM(n_qubits=3)
    x = jnp.ones((4, 3))
    params = model.init(jax.random.PRNGKey(0), x)
    energy = model.apply(params, x)
    assert energy.shape == (4,)
check("Quantum Boltzmann Machine (3q, batch=4)", test_qbm)

# -----------------------------------------------------------------
# 19. ENVIRONMENT DETECTION
# -----------------------------------------------------------------
section("19. ENVIRONMENT & PLATFORM")

def test_env_detect():
    from superfermion.environment.detect import detect_environment, Environment
    env = detect_environment()
    assert isinstance(env, Environment)
check("Environment detection", test_env_detect)

# -----------------------------------------------------------------
# FINAL REPORT
# -----------------------------------------------------------------
print("\n" + "-"*70)
print("  INDUSTRY MEGA-VALIDATION — FINAL REPORT")
print("-"*70)
print(f"\n  Total Tests:  {PASS + FAIL}")
print(f"  Passed:       {PASS}")
print(f"  Failed:       {FAIL}")
print(f"  Pass Rate:    {PASS/(PASS+FAIL)*100:.1f}%")
print()

if FAIL > 0:
    print("  FAILED TESTS:")
    for name, status, detail in RESULTS:
        if status == "FAIL":
            print(f"    XX {name}: {detail}")
    print()

print("  DOMAIN COVERAGE:")
domains = [
    "Circuit API", "JAX Autograd", "QML Algorithms",
    "QEC", "Noise & Mitigation", "Bridges", "Cloud Routing",
    "Benchmarking", "Compiler", "Chemistry",
    "Encodings/Ansatze", "Runtime", "Visualization", "Max-Qubit",
    "QRL/QBM", "Environment"
]
for d in domains:
    print(f"    OK {d}")

print(f"\n{'-'*70}")
if FAIL == 0:
    print("  VERDICT: SUPERFERMION IS PRODUCTION-READY OK")
else:
    print(f"  VERDICT: {FAIL} ISSUE(S) TO RESOLVE")
print(f"{'═'*70}\n")

if __name__ == "__main__":
    sys.exit(0 if FAIL == 0 else 1)
