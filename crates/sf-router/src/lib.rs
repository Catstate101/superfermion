//! Superfermion Circuit Router
//!
//! Maps logical qubits to physical qubits and inserts SWAP gates
//! to satisfy hardware connectivity constraints.
//!
//! Implements:
//! - SABRE (SWAP-Based BidiREctional heuristic search)
//! - Hardware topology graphs
//! - Layout management (trivial, noise-aware)
//! - Token swapping for permutation routing

pub mod topology;
pub mod layout;
pub mod sabre;
pub mod token_swap;

use sf_ir::{QuantumDAG, QubitMapping};
use thiserror::Error;

pub use topology::{CouplingMap, HardwareTopology};
pub use layout::{LayoutStrategy, TrivialLayout, NoiseAwareLayout};
pub use sabre::SabreRouter;
pub use token_swap::token_swap_route;

#[derive(Error, Debug)]
pub enum RouterError {
    #[error("Cannot route: circuit has {0} qubits but device has only {1}")]
    InsufficientQubits(usize, usize),
    #[error("No valid SWAP found to make progress (stuck at gate on qubits {0:?})")]
    RoutingFailed(Vec<usize>),
    #[error("Token swap failed: {0}")]
    TokenSwapError(String),
}

/// High-level router interface.
pub struct Router {
    pub topology: CouplingMap,
}

impl Router {
    pub fn new(topology: CouplingMap) -> Self {
        Self { topology }
    }

    /// Route a circuit for the given hardware topology.
    ///
    /// 1. Choose initial layout
    /// 2. Run SABRE routing (forward + backward passes)
    /// 3. Return routed DAG + final qubit mapping
    pub fn route(&self, dag: &QuantumDAG) -> Result<(QuantumDAG, QubitMapping), RouterError> {
        if dag.n_qubits > self.topology.n_qubits() {
            return Err(RouterError::InsufficientQubits(
                dag.n_qubits,
                self.topology.n_qubits(),
            ));
        }

        let initial_layout = TrivialLayout::compute(dag.n_qubits);
        let router = SabreRouter::new(&self.topology);
        router.route(dag, &initial_layout)
    }

    /// Route with a specific initial layout.
    pub fn route_with_layout(
        &self,
        dag: &QuantumDAG,
        layout: &QubitMapping,
    ) -> Result<(QuantumDAG, QubitMapping), RouterError> {
        let router = SabreRouter::new(&self.topology);
        router.route(dag, layout)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sf_ir::OpType;

    #[test]
    fn test_no_swaps_needed_for_adjacent() {
        let topo = CouplingMap::linear(3);
        let router = Router::new(topo);

        let mut dag = QuantumDAG::new(2, 0);
        dag.add_op(OpType::H, &[0]);
        dag.add_op(OpType::CNOT, &[0, 1]); // adjacent on linear

        let (routed, _mapping) = router.route(&dag).unwrap();
        // Should not add any SWAPs for adjacent qubits
        assert_eq!(routed.count_ops_of_type("SWAP"), 0);
    }

    #[test]
    fn test_swap_inserted_for_non_adjacent() {
        let topo = CouplingMap::linear(3);
        let router = Router::new(topo);

        let mut dag = QuantumDAG::new(3, 0);
        dag.add_op(OpType::CNOT, &[0, 2]); // NOT adjacent on 0-1-2

        let (routed, _mapping) = router.route(&dag).unwrap();
        // Must insert at least one SWAP
        assert!(routed.gate_count() > 1);
    }

    #[test]
    fn test_insufficient_qubits() {
        let topo = CouplingMap::linear(2);
        let router = Router::new(topo);

        let dag = QuantumDAG::new(5, 0);
        assert!(router.route(&dag).is_err());
    }
}
