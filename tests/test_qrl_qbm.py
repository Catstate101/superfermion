"""
Test script for QRL (Quantum RL) and QBM.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import superfermion as sf
from superfermion.algorithms.qrl import QuantumREINFORCE
from superfermion.algorithms.qbm import QBM


def test_qrl_reinforce():
    print("Testing QuantumREINFORCE agent update...")
    
    # 1. Setup
    n_qubits = 2
    c = sf.Circuit(n_qubits)
    c.rx(sf.param("theta0"), 0)
    
    agent = QuantumREINFORCE(c, num_actions=2)
    key = jax.random.PRNGKey(0)
    
    # 2. Dummy Trajectory
    state = jnp.array([0.1, 0.2])
    params = agent.model.init(key, state[None, :])
    opt_state = agent.optimizer.init(params)
    
    trajectories = [
        {'state': state, 'action': 0, 'reward': 1.0},
        {'state': state, 'action': 1, 'reward': -1.0}
    ]
    
    # 3. Update
    print("  Performing policy update...")
    new_params, _ = agent.update(params, opt_state, trajectories)
    
    # Verify weights changed
    diff = jnp.mean(jnp.abs(new_params['params']['weights'] - params['params']['weights']))
    print(f"  Parameter Shift: {diff:.8f}")
    assert diff > 0
    print("[PASS] QuantumRL update verified.")


def test_qbm_energy():
    print("\nTesting Quantum Boltzmann Machine energy calculation...")
    
    n_qubits = 3
    model = QBM(n_qubits=n_qubits)
    key = jax.random.PRNGKey(1)
    
    # Dummy data (batch=4, bits=3)
    x = jax.random.randint(key, (4, 3), 0, 2).astype(jnp.float32)
    params = model.init(key, x)
    
    # Compute energy
    energies = model.apply(params, x)
    print(f"  Input states:\n{x}")
    print(f"  Energies: {energies}")
    
    # Test partition function
    z = model.get_partition_function(params)
    print(f"  Partition Function Z: {z:.4f}")
    assert z > 0
    
    print("[PASS] QBM energy/partition verified.")


if __name__ == "__main__":
    try:
        test_qrl_reinforce()
        test_qbm_energy()
        print("\nSession 23: QRL + QBM officially ready.")
    except Exception as e:
        print(f"\nTests failed: {e}")
        import traceback
        traceback.print_exc()
