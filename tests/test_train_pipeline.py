"""
Test Training and Pipeline modules (Phase 4.1h).

Tests: VQE convergence, Pipeline workflow, optimizer swap, checkpoint save/restore.
"""

import pytest
import tempfile
import os
from pathlib import Path


class TestVQEConvergence:
    """Test VQE training convergence."""
    
    def test_vqe_h2_convergence(self):
        """Test VQE converges on H2 Hamiltonian."""
        import superfermion as sf
        from superfermion.algorithms.variational import VQE
        from superfermion.observables.core import Hamiltonian, PauliString
        
        # Simple H2-like Hamiltonian
        h = Hamiltonian([
            PauliString("II", coeffs=-1.0),
            PauliString("ZI", coeffs=0.4),
            PauliString("IZ", coeffs=0.4),
            PauliString("ZZ", coeffs=0.2),
            PauliString("XX", coeffs=0.2),
        ])
        
        # Parameterized ansatz
        c = sf.Circuit(2)
        c.ry(sf.param("t0"), 0)
        c.ry(sf.param("t1"), 1)
        c.cx(0, 1)
        
        vqe = VQE(c, h, backend="statevector")
        result = vqe.minimize(iterations=50)
        
        # Should converge to reasonable energy
        assert result.optimal_value < 0.5
        print(f"  VQE converged to {result.optimal_value:.4f}")
    
    def test_vqe_with_shots(self):
        """Test VQE with shot-based sampling."""
        import superfermion as sf
        from superfermion.algorithms.variational import VQE
        from superfermion.observables.core import Hamiltonian, PauliString
        
        h = Hamiltonian([PauliString("Z", coeffs=1.0)])
        c = sf.Circuit(1).ry(sf.param("t"), 0)
        
        vqe = VQE(c, h, backend="statevector", shots=1000)
        result = vqe.minimize(iterations=30)
        
        # Energy should be minimized
        assert isinstance(result.optimal_value, float)
    
    def test_vqe_optimizer_choice(self):
        """Test VQE with different optimizers."""
        import superfermion as sf
        from superfermion.algorithms.variational import VQE
        from superfermion.observables.core import Hamiltonian, PauliString
        
        h = Hamiltonian([PauliString("Z", coeffs=1.0)])
        c = sf.Circuit(1).ry(sf.param("t"), 0)
        
        for optimizer in ["L-BFGS-B", "COBYLA"]:
            vqe = VQE(c, h, backend="statevector", optimizer=optimizer)
            result = vqe.minimize(iterations=20)
            assert result.optimal_value is not None
            print(f"  {optimizer}: {result.optimal_value:.4f}")


class TestPipelineWorkflow:
    """Test Pipeline workflow with custom stages."""
    
    def test_pipeline_basic(self):
        """Test basic pipeline construction."""
        from superfermion.pipeline import Pipeline
        
        pipeline = Pipeline()
        assert pipeline is not None
    
    def test_pipeline_with_stages(self):
        """Test pipeline with multiple stages."""
        import superfermion as sf
        from superfermion.pipeline import Pipeline
        
        # Build a simple pipeline
        pipeline = Pipeline()
        
        # Add stages
        def compile_stage(circuit):
            return circuit
        
        def run_stage(circuit):
            return sf.run(circuit, backend="statevector")
        
        pipeline.add_stage("compile", compile_stage)
        pipeline.add_stage("run", run_stage)
        
        c = sf.Circuit(1).h(0)
        result = pipeline.run(c)
        
        assert result is not None
    
    def test_pipeline_checkpoint(self):
        """Test pipeline checkpoint save/restore."""
        import superfermion as sf
        from superfermion.pipeline import Pipeline
        
        pipeline = Pipeline(name="test_checkpoint")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.json"
            
            # Run and save checkpoint
            c = sf.Circuit(1).h(0)
            result = sf.run(c, backend="statevector")
            
            pipeline.save_checkpoint(str(checkpoint_path), {"result": "success"})
            assert checkpoint_path.exists()
            
            # Restore
            data = pipeline.load_checkpoint(str(checkpoint_path))
            assert data["result"] == "success"


class TestOptimizerSwap:
    """Test optimizer swap during training."""
    
    def test_optimizer_swap_vqe(self):
        """Test swapping optimizer mid-training."""
        import superfermion as sf
        from superfermion.algorithms.variational import VQE
        from superfermion.observables.core import Hamiltonian, PauliString
        
        h = Hamiltonian([PauliString("Z", coeffs=1.0)])
        c = sf.Circuit(1).ry(sf.param("t"), 0)
        
        # Train with L-BFGS-B first
        vqe1 = VQE(c, h, optimizer="L-BFGS-B")
        result1 = vqe1.minimize(iterations=20)
        
        # Continue with COBYLA
        vqe2 = VQE(c, h, optimizer="COBYLA")
        result2 = vqe2.minimize(
            iterations=20,
            initial_params=[result1.optimal_params.get("t", 0.0)]
        )
        
        assert result2.optimal_value is not None


class TestCheckpointRestore:
    """Test checkpoint save/restore for training."""
    
    def test_vqe_checkpoint(self):
        """Test VQE checkpoint save and restore."""
        import superfermion as sf
        import json
        from superfermion.algorithms.variational import VQE
        from superfermion.observables.core import Hamiltonian, PauliString
        
        h = Hamiltonian([PauliString("Z", coeffs=1.0)])
        c = sf.Circuit(1).ry(sf.param("t"), 0)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Train and save
            vqe = VQE(c, h)
            result = vqe.minimize(iterations=30)
            
            checkpoint = {
                "optimal_params": result.optimal_params,
                "optimal_value": result.optimal_value,
            }
            
            ckpt_path = Path(tmpdir) / "vqe_checkpoint.json"
            with open(ckpt_path, 'w') as f:
                json.dump(checkpoint, f)
            
            # Restore and verify
            with open(ckpt_path, 'r') as f:
                loaded = json.load(f)
            
            assert loaded["optimal_value"] == result.optimal_value


class TestTrainingHistory:
    """Test training history tracking."""
    
    def test_vqe_history(self):
        """Test VQE returns training history."""
        import superfermion as sf
        from superfermion.algorithms.variational import VQE
        from superfermion.observables.core import Hamiltonian, PauliString
        
        h = Hamiltonian([PauliString("Z", coeffs=1.0)])
        c = sf.Circuit(1).ry(sf.param("t"), 0)
        
        vqe = VQE(c, h)
        result = vqe.minimize(iterations=30, verbose=False)
        
        # History should track energy
        assert result.history is not None
        assert len(result.history) > 0
        
        # Energy should decrease or stay same
        for i in range(1, len(result.history)):
            assert result.history[i] <= result.history[i-1] + 0.1  # Allow small noise
    
    def test_training_metadata(self):
        """Test training returns metadata."""
        import superfermion as sf
        from superfermion.algorithms.variational import VQE
        from superfermion.observables.core import Hamiltonian, PauliString
        
        h = Hamiltonian([PauliString("Z", coeffs=1.0)])
        c = sf.Circuit(1).ry(sf.param("t"), 0)
        
        vqe = VQE(c, h)
        result = vqe.minimize(iterations=20)
        
        assert result.metadata is not None
        assert "optimizer" in result.metadata


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
