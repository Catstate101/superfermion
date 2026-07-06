"""
Superfermion Notebook Validation - Runs all notebook cells as a script.
"""
import superfermion as sf
import jax
import jax.numpy as jnp
print(f'Superfermion {sf.__version__}')
print(f'JAX {jax.__version__} ({jax.devices()[0].device_kind})')

# 1. Circuit API
print('\n--- 1. Circuit API ---')
c = sf.Circuit(2).h(0).cx(0, 1)
print(c.draw())
result = sf.run(c, shots=1000)
print(f'Counts: {result.counts}')

c_swap = sf.Circuit(2).swap(0, 1)
compiled = sf.compile(c_swap)
print(f'SWAP ({c_swap.gate_count}) -> {compiled.gate_count} CNOTs')

qasm = c.to_qasm3()
print(f'QASM3 lines: {len(qasm.splitlines())}')
j = c.to_json()
c2 = sf.Circuit.from_json(j)
print(f'JSON round-trip: {c2.gate_count} gates OK')

# 2. JAX Autograd
print('\n--- 2. JAX Autograd ---')
c = sf.Circuit(1).rx(sf.param('theta'), 0)
f = sf.qml.circuit_to_jax(c, backend='jax')
def loss(t): return jnp.abs(f(t)[1])**2
g = jax.grad(loss)(jnp.array(1.0))
print(f'Gradient at theta=1.0: {g:.6f}')

from superfermion.nn.quantum_layer import QuantumLayer
c = sf.Circuit(1).rx(sf.param('a'), 0).ry(sf.param('b'), 0)
model = QuantumLayer(n_qubits=1, ansatz=c)
params = model.init(jax.random.PRNGKey(0))
def loss2(p): return model.apply(p)[0]
h = jax.hessian(loss2)(params)
print(f'Hessian shape: {h["params"]["weights"]["params"]["weights"].shape}')

# 3. VQE
print('\n--- 3. VQE ---')
from superfermion.observables.core import Hamiltonian, PauliString
from superfermion.algorithms.variational import VQE
H = Hamiltonian([PauliString('ZI',0.3), PauliString('IZ',0.3), PauliString('ZZ',-0.5), PauliString('XX',0.2)])
ansatz = sf.Circuit(2)
for i in range(2):
    ansatz.ry(sf.param(f'ry{i}'), i)
    ansatz.rz(sf.param(f'rz{i}'), i)
ansatz.cx(0, 1)
for i in range(2):
    ansatz.ry(sf.param(f'ry2_{i}'), i)
vqe = VQE(ansatz, H, optimizer="L-BFGS-B")
result = vqe.minimize(iterations=30)
print(f'VQE energy: {result.optimal_value:.6f}')

# 4. Chemistry
print('\n--- 4. Chemistry ---')
from superfermion.chemistry import get_molecular_hamiltonian, uccsd_ansatz
H2 = get_molecular_hamiltonian('H2')
print(f'H2 terms: {len(H2.terms)}')
ansatz = uccsd_ansatz(n_qubits=2, n_electrons=2)
vqe = VQE(ansatz, H2, optimizer="L-BFGS-B")
result = vqe.minimize(iterations=20)
print(f'H2 energy: {result.optimal_value:.4f} Ha')

# 5. QEC
print('\n--- 5. QEC ---')
from superfermion.qec.codes.surface import SurfaceCode
sc = SurfaceCode(distance=3)
c = sc.build_syndrome_extraction()
print(f'{sc}: {c.n_qubits}q, {c.gate_count} gates')

# 6. Cloud/Security
print('\n--- 6. Cloud/Security ---')
from superfermion.runtime.arbiter import ResourceArbiter
from superfermion.runtime.specs import list_devices, get_spec
from superfermion.serve.auth import VAULT, check_qubit_limit
print(f'Devices: {list_devices()}')
arb = ResourceArbiter()
print(f'Route 5q: {arb.route(n_qubits=5)}')
check_qubit_limit(10, 'free')
print('Free tier 10q: OK')

# 7. Viz
print('\n--- 8. Visualization ---')
from superfermion.viz.core import bloch_angles, state_bar_chart
from superfermion.backends.jax_sim import JAXBackend
sv = jnp.array([1/jnp.sqrt(2), 1/jnp.sqrt(2)], dtype=complex)
angles = bloch_angles(sv)
print(f'|+> theta={angles["theta"]:.4f}, phi={angles["phi"]:.4f}')
bell = sf.Circuit(2).h(0).cx(0, 1)
sv_bell = JAXBackend().simulate(bell, [])
print(state_bar_chart(sv_bell))

print('\n' + '='*50)
print('  NOTEBOOK VALIDATION COMPLETE - ALL CELLS PASS')
print('='*50)
