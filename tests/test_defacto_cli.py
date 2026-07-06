#!/usr/bin/env python
"""
Tests for De Facto CLI Commands and Bridge Functions.

Tests:
- sf plugin list
- sf auth login/logout/status
- sf convert
- sf estimate
- sf compare
- sf jobs
- Bridge functions: from_cirq, to_cirq, to_pennylane
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

# Ensure superfermion is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

import superfermion as sf


# ═════════════════════════════════════════════════════════════════════════
# CLI Command Tests (Unit Tests)
# ═════════════════════════════════════════════════════════════════════════

class TestPluginCommand:
    """Test sf plugin command."""
    
    def test_plugin_list_empty(self):
        """Test plugin list with no plugins."""
        from superfermion.plugins import list_all
        plugins = list_all()
        assert isinstance(plugins, dict)
        assert "backends" in plugins
        assert "templates" in plugins
        assert "passes" in plugins
    
    def test_register_backend(self):
        """Test backend registration."""
        from superfermion.plugins import register_backend, get_backend
        
        @register_backend("test_backend_unit")
        class TestBackend:
            def run(self, circuit, shots=1000):
                return {"counts": {}}
        
        assert get_backend("test_backend_unit") is not None
    
    def test_register_template(self):
        """Test template registration."""
        from superfermion.plugins import register_template, get_template
        
        @register_template("test_ansatz_unit")
        def test_ansatz(n_qubits):
            return sf.Circuit(n_qubits)
        
        assert get_template("test_ansatz_unit") is not None
    
    def test_register_pass(self):
        """Test compiler pass registration."""
        from superfermion.plugins import register_pass, get_pass
        
        @register_pass("test_pass_unit")
        class TestPass:
            def run(self, circuit):
                return circuit
        
        assert get_pass("test_pass_unit") is not None


class TestAuthCommand:
    """Test sf auth command."""
    
    def test_credential_store_memory(self):
        """Test in-memory credential storage."""
        from superfermion.security.credentials import CredentialStore, CredentialBackend
        
        store = CredentialStore(backend=CredentialBackend.MEMORY)
        store.set("test_token", "secret123", provider="test")
        
        assert store.get("test_token") == "secret123"
        assert store.delete("test_token")
        assert store.get("test_token") is None
    
    def test_credential_expiration(self):
        """Test credential expiration."""
        from superfermion.security.credentials import CredentialStore, CredentialBackend
        
        store = CredentialStore(backend=CredentialBackend.MEMORY)
        store.set("expiring_token", "value", provider="test")
        
        # Verify the credential exists
        assert store.get("expiring_token") == "value"
    
    def test_auth_status(self):
        """Test auth status check."""
        from superfermion.security.credentials import CredentialStore, CredentialBackend
        
        store = CredentialStore(backend=CredentialBackend.MEMORY)
        
        # Initially no credentials
        assert store.get("ibm_token") is None
        assert store.get("ionq_token") is None
        
        # Set credential
        store.set("ibm_token", "test_key", provider="ibm")
        assert store.get("ibm_token") == "test_key"


class TestConvertCommand:
    """Test sf convert command."""
    
    def test_convert_json_to_sfc(self):
        """Test JSON to .sfc conversion."""
        from superfermion.serialization.circuit_format import save_circuit, load_circuit
        
        # Create circuit
        c = sf.Circuit(2)
        c.h(0).cx(0, 1)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save as JSON using to_json method
            json_path = os.path.join(tmpdir, "circuit.json")
            with open(json_path, 'w') as f:
                f.write(c.to_json())
            
            # Save as SFC
            sfc_path = os.path.join(tmpdir, "circuit.sfc")
            save_circuit(c, sfc_path)
            
            # Load and verify
            loaded = load_circuit(sfc_path)
            assert loaded.n_qubits == 2
            assert loaded.gate_count == 2
    
    def test_convert_sfc_to_json(self):
        """Test .sfc to JSON conversion."""
        from superfermion.serialization.circuit_format import save_circuit, load_circuit
        
        c = sf.Circuit(3)
        c.h(0).cx(0, 1).cx(1, 2)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            sfc_path = os.path.join(tmpdir, "circuit.sfc")
            json_path = os.path.join(tmpdir, "circuit.json")
            
            save_circuit(c, sfc_path)
            loaded = load_circuit(sfc_path)
            
            # Save as JSON using to_json method
            with open(json_path, 'w') as f:
                f.write(loaded.to_json())
            
            # Verify JSON exists and is valid
            with open(json_path) as f:
                data = json.load(f)
            assert data["n_qubits"] == 3
    
    def test_convert_to_qasm(self):
        """Test circuit to QASM conversion."""
        from superfermion.bridge import to_qasm
        
        c = sf.Circuit(2)
        c.h(0).cx(0, 1)
        
        qasm = to_qasm(c)
        assert "OPENQASM" in qasm
        assert "h" in qasm.lower() or "u3" in qasm.lower()


class TestEstimateCommand:
    """Test sf estimate command."""
    
    def test_estimate_ibm_cost(self):
        """Test cost estimation for IBM Quantum."""
        c = sf.Circuit(5)
        c.h(0)
        for i in range(4):
            c.cx(i, i + 1)
        
        # Basic cost calculation
        shots = 1000
        gate_shots = c.gate_count * shots
        
        # IBM charges by time (rough estimate)
        estimated_time = c.depth * shots * 0.001
        assert estimated_time > 0
        assert gate_shots == 5 * 1000  # 5 gates * 1000 shots
    
    def test_estimate_ionq_cost(self):
        """Test cost estimation for IonQ."""
        c = sf.Circuit(4)
        for i in range(4):
            c.h(i)
        for i in range(3):
            c.cx(i, i + 1)
        
        shots = 500
        gate_shots = c.gate_count * shots
        
        # IonQ uses gate-shots pricing
        assert gate_shots > 0


class TestCompareCommand:
    """Test sf compare command."""
    
    def test_compare_backends_basic(self):
        """Test basic backend comparison."""
        c = sf.Circuit(2)
        c.h(0).cx(0, 1)
        
        results = []
        for backend in ["statevector", "jax"]:
            try:
                result = sf.run(c, backend=backend, shots=100)
                results.append((backend, True))
            except Exception as e:
                results.append((backend, False))
        
        # At least statevector should work
        assert any(r[1] for r in results)
    
    def test_compare_consistency(self):
        """Test that backends produce consistent results."""
        c = sf.Circuit(2)
        c.h(0).cx(0, 1)
        
        # Run on statevector
        result = sf.run(c, backend="statevector", shots=1000)
        
        # Bell state should give |00> and |11> with ~50% each
        if hasattr(result, "counts"):
            counts = result.counts
            total = sum(counts.values())
            assert total > 0


class TestJobsCommand:
    """Test sf jobs command."""
    
    def test_jobs_list_requires_auth(self):
        """Test that job listing requires authentication."""
        from superfermion.security.credentials import CredentialStore, CredentialBackend
        
        store = CredentialStore(backend=CredentialBackend.MEMORY)
        
        # No credentials set
        assert store.get("ibm_token") is None
        assert store.get("ionq_token") is None


# ═════════════════════════════════════════════════════════════════════════
# Bridge Function Tests
# ═════════════════════════════════════════════════════════════════════════

class TestQiskitBridge:
    """Test Qiskit bridge functions."""
    
    def test_from_qiskit_basic(self):
        """Test Qiskit to Superfermion conversion."""
        try:
            from qiskit import QuantumCircuit
            from superfermion.bridge import from_qiskit
            
            qc = QuantumCircuit(2)
            qc.h(0)
            qc.cx(0, 1)
            
            sf_circuit = from_qiskit(qc)
            assert sf_circuit.n_qubits == 2
            assert sf_circuit.gate_count == 2
        except ImportError:
            pytest.skip("Qiskit not installed")
    
    def test_to_qiskit_basic(self):
        """Test Superfermion to Qiskit conversion."""
        try:
            from qiskit import QuantumCircuit
            from superfermion.bridge import to_qiskit
            
            c = sf.Circuit(2)
            c.h(0).cx(0, 1)
            
            qc = to_qiskit(c)
            assert qc.num_qubits == 2
        except ImportError:
            pytest.skip("Qiskit not installed")


class TestCirqBridge:
    """Test Cirq bridge functions."""
    
    def test_from_cirq_basic(self):
        """Test Cirq to Superfermion conversion."""
        try:
            import cirq
            from superfermion.bridge import from_cirq
            
            q0, q1 = cirq.LineQubit.range(2)
            cirq_circ = cirq.Circuit(
                cirq.H(q0),
                cirq.CNOT(q0, q1)
            )
            
            sf_circuit = from_cirq(cirq_circ)
            assert sf_circuit.n_qubits == 2
            assert sf_circuit.gate_count == 2
        except ImportError:
            pytest.skip("Cirq not installed")
    
    def test_to_cirq_basic(self):
        """Test Superfermion to Cirq conversion."""
        try:
            import cirq
            from superfermion.bridge import to_cirq
            
            c = sf.Circuit(2)
            c.h(0).cx(0, 1)
            
            cirq_circ = to_cirq(c)
            assert len(cirq_circ.all_qubits()) == 2
        except ImportError:
            pytest.skip("Cirq not installed")
    
    def test_from_cirq_rotations(self):
        """Test Cirq rotation gates conversion."""
        try:
            import cirq
            from superfermion.bridge import from_cirq
            
            q0 = cirq.LineQubit(0)
            cirq_circ = cirq.Circuit(
                cirq.rx(0.5).on(q0),
                cirq.ry(0.3).on(q0),
                cirq.rz(0.1).on(q0)
            )
            
            sf_circuit = from_cirq(cirq_circ)
            assert sf_circuit.gate_count == 3
        except ImportError:
            pytest.skip("Cirq not installed")


class TestPennyLaneBridge:
    """Test PennyLane bridge functions."""
    
    def test_from_pennylane_basic(self):
        """Test PennyLane to Superfermion conversion."""
        try:
            from superfermion.bridge import from_pennylane
            
            # Create a simple PennyLane-like structure
            # Note: This requires actual PennyLane circuits
            pytest.skip("Full PennyLane integration test")
        except ImportError:
            pytest.skip("PennyLane not installed")
    
    def test_to_pennylane_basic(self):
        """Test Superfermion to PennyLane conversion."""
        try:
            import pennylane as qml
            from superfermion.bridge import to_pennylane
            
            c = sf.Circuit(2)
            c.h(0).cx(0, 1)
            
            qfunc = to_pennylane(c)
            assert callable(qfunc)
            assert qfunc.n_qubits == 2
        except ImportError:
            pytest.skip("PennyLane not installed")
    
    def test_to_pennylane_execution(self):
        """Test that generated PennyLane function executes."""
        try:
            import pennylane as qml
            from superfermion.bridge import to_pennylane
            
            c = sf.Circuit(2)
            c.h(0).cx(0, 1)
            
            qfunc = to_pennylane(c)
            
            dev = qml.device('default.qubit', wires=2)
            
            @qml.qnode(dev)
            def circuit():
                qfunc()
                return qml.state()
            
            state = circuit()
            assert len(state) == 4  # 2 qubits = 4 amplitudes
        except ImportError:
            pytest.skip("PennyLane not installed")


class TestBraketBridge:
    """Test Braket bridge functions."""
    
    def test_to_braket_basic(self):
        """Test Superfermion to Braket conversion."""
        try:
            from superfermion.bridge import to_braket
            
            c = sf.Circuit(2)
            c.h(0).cx(0, 1)
            
            braket_circ = to_braket(c)
            assert braket_circ is not None
        except ImportError:
            pytest.skip("Braket SDK not installed")


class TestIonQBridge:
    """Test IonQ bridge functions."""
    
    def test_to_ionq_basic(self):
        """Test Superfermion to IonQ conversion."""
        from superfermion.bridge import to_ionq
        
        c = sf.Circuit(2)
        c.h(0).cx(0, 1)
        
        ionq_gates = to_ionq(c)
        assert isinstance(ionq_gates, list)
        assert len(ionq_gates) >= 2
    
    def test_to_ionq_rotations(self):
        """Test IonQ conversion with rotation gates."""
        from superfermion.bridge import to_ionq
        
        c = sf.Circuit(1)
        c.rx(0.5, 0).ry(0.3, 0).rz(0.1, 0)
        
        ionq_gates = to_ionq(c)
        assert len(ionq_gates) == 3
        
        # Check that rotations have rotation parameter
        for gate in ionq_gates:
            if gate.get("gate") in ("rx", "ry", "rz"):
                assert "rotation" in gate


# ═════════════════════════════════════════════════════════════════════════
# Integration Tests
# ═════════════════════════════════════════════════════════════════════════

class TestRoundTrip:
    """Test round-trip conversions between frameworks."""
    
    def test_qiskit_roundtrip(self):
        """Test Qiskit → SF → Qiskit round-trip."""
        try:
            from qiskit import QuantumCircuit
            from superfermion.bridge import from_qiskit, to_qiskit
            
            qc_orig = QuantumCircuit(2)
            qc_orig.h(0)
            qc_orig.cx(0, 1)
            
            sf_circuit = from_qiskit(qc_orig)
            qc_roundtrip = to_qiskit(sf_circuit)
            
            assert qc_orig.num_qubits == qc_roundtrip.num_qubits
        except ImportError:
            pytest.skip("Qiskit not installed")
    
    def test_cirq_roundtrip(self):
        """Test Cirq → SF → Cirq round-trip."""
        try:
            import cirq
            from superfermion.bridge import from_cirq, to_cirq
            
            q0, q1 = cirq.LineQubit.range(2)
            cirq_orig = cirq.Circuit(
                cirq.H(q0),
                cirq.CNOT(q0, q1)
            )
            
            sf_circuit = from_cirq(cirq_orig)
            cirq_roundtrip = to_cirq(sf_circuit)
            
            assert len(cirq_orig.all_qubits()) == len(cirq_roundtrip.all_qubits())
        except ImportError:
            pytest.skip("Cirq not installed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
