pub mod decompose;
pub mod passes;
pub mod rotation_merge;
pub mod twirl;

use sf_ir::{QuantumDAG};
use thiserror::Error;
use rand::Rng;

#[derive(Error, Debug)]
pub enum CompilerError {
    #[error("Target backend {0} does not support {1} bits, circuit has {2}")]
    QubitMismatch(String, usize, usize),
    #[error("Encountered unsupported gate {0} after decomposition")]
    UnsupportedGate(String),
    #[error("Routing failed: {0}")]
    RoutingError(String),
}

/// Target hardware backend specification.
#[derive(Clone, Debug)]
pub struct BackendSpec {
    pub name: String,
    pub native_gates: Vec<String>,
    pub connectivity: Vec<(usize, usize)>,
    pub n_qubits: usize,
    pub optimization_level: u8,
}

impl Default for BackendSpec {
    fn default() -> Self {
        Self {
            name: "generic".to_string(),
            native_gates: vec!["H".to_string(), "X".to_string(), "Y".to_string(), "Z".to_string(), "CNOT".to_string()],
            connectivity: vec![],
            n_qubits: 32,
            optimization_level: 1,
        }
    }
}

/// Compilation passes for circuit transformation.
pub trait Pass {
    fn name(&self) -> &str;
    fn run(&self, dag: &mut QuantumDAG) -> Result<(), CompilerError>;
}

/// Orchestrates the sequence of compilation stages.
pub struct PassManager {
    passes: Vec<Box<dyn Pass>>,
}

impl PassManager {
    pub fn new() -> Self {
        Self { passes: Vec::new() }
    }

    pub fn add_pass(&mut self, pass: Box<dyn Pass>) {
        self.passes.push(pass);
    }

    pub fn run(&self, dag: &mut QuantumDAG) -> Result<(), CompilerError> {
        for pass in &self.passes {
            log::info!("Running pass: {}", pass.name());
            pass.run(dag)?;
        }
        Ok(())
    }
}

/// High-level compiler interface.
pub struct Compiler {
    pub backend: BackendSpec,
}

impl Compiler {
    pub fn new(backend: BackendSpec) -> Self {
        Self { backend }
    }

    /// Compile a circuit for the target backend using an optimized pipeline.
    pub fn compile(&self, dag: &QuantumDAG) -> Result<QuantumDAG, CompilerError> {
        let mut result = dag.clone_dag();
        
        let mut manager = PassManager::new();
        
        // 1. Initial Gate Cancellation (Remove identity ops)
        manager.add_pass(Box::new(passes::GateCancellationPass::new()));
        
        // 2. High-level Decomposition (e.g. SWAP -> 3 CNOTs)
        manager.add_pass(Box::new(decompose::BasicDecomposePass::new()));
        
        // 3. Backend-specific target decomposition
        if self.backend.native_gates.contains(&"SX".to_string()) {
            manager.add_pass(Box::new(
                decompose::superconducting::SuperconductingDecomposePass::new()
            ));
        }
        
        // 4. Rotation merging (collapse consecutive Rz/Rx/Ry)
        manager.add_pass(Box::new(rotation_merge::RotationMergingPass::new()));
        
        manager.run(&mut result)?;

        // 5. Routing: map logical qubits to hardware topology
        if !self.backend.connectivity.is_empty() {
            let coupling = sf_router::CouplingMap::from_edges(
                self.backend.n_qubits,
                &self.backend.connectivity,
            );
            let router = sf_router::Router::new(coupling);
            let (routed_dag, _mapping) = router
                .route(&result)
                .map_err(|e| CompilerError::RoutingError(e.to_string()))?;
            result = routed_dag;

            // After routing, re-run optimization on the routed DAG
            let mut post_manager = PassManager::new();
            post_manager.add_pass(Box::new(passes::GateCancellationPass::new()));
            post_manager.add_pass(Box::new(rotation_merge::RotationMergingPass::new()));
            post_manager.run(&mut result)?;
        }

        // 6. Pauli twirling at optimization level >= 2
        if self.backend.optimization_level >= 2 {
            let mut rng = rand::thread_rng();
            let seed: u64 = rng.gen();
            let twirl_pass = twirl::PauliTwirlPass::new(seed);
            let mut twirl_manager = PassManager::new();
            twirl_manager.add_pass(Box::new(twirl_pass));
            twirl_manager.run(&mut result)?;
        }
        
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sf_ir::OpType;

    #[test]
    fn test_gate_cancellation() -> Result<(), CompilerError> {
        let mut dag = QuantumDAG::new(1, 0);
        dag.add_op(OpType::H, &[0]);
        dag.add_op(OpType::H, &[0]);
        assert_eq!(dag.gate_count(), 2);

        let compiler = Compiler::new(BackendSpec::default());
        let compiled = compiler.compile(&dag)?;
        
        // H * H should cancel out
        assert_eq!(compiled.gate_count(), 0);
        Ok(())
    }

    #[test]
    fn test_swap_decomposition() -> Result<(), CompilerError> {
        let mut dag = QuantumDAG::new(2, 0);
        dag.add_op(OpType::SWAP, &[0, 1]);
        assert_eq!(dag.gate_count(), 1);

        let compiler = Compiler::new(BackendSpec::default());
        let compiled = compiler.compile(&dag)?;
        
        // SWAP should become 3 CNOTs
        assert_eq!(compiled.gate_count(), 3);
        assert_eq!(compiled.count_ops_of_type("CNOT"), 3);
        Ok(())
    }

    #[test]
    fn test_mixed_optimization() -> Result<(), CompilerError> {
        let mut dag = QuantumDAG::new(2, 0);
        dag.add_op(OpType::H, &[0]);
        dag.add_op(OpType::H, &[0]); // Cancels
        dag.add_op(OpType::SWAP, &[0, 1]); // 3 CNOTs
        dag.add_op(OpType::X, &[1]);
        dag.add_op(OpType::X, &[1]); // Cancels

        let compiler = Compiler::new(BackendSpec::default());
        let compiled = compiler.compile(&dag)?;
        
        assert_eq!(compiled.gate_count(), 3);
        Ok(())
    }
}
