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
    /// Uses multi-trial bidirectional SABRE:
    /// 1. Runs N trials with different initial layouts (first is trivial)
    /// 2. Each trial does forward + backward passes
    /// 3. Returns the result with the fewest SWAPs
    pub fn route(&self, dag: &QuantumDAG) -> Result<(QuantumDAG, QubitMapping), RouterError> {
        if dag.n_qubits > self.topology.n_qubits() {
            return Err(RouterError::InsufficientQubits(
                dag.n_qubits,
                self.topology.n_qubits(),
            ));
        }

        let config = sabre::SabreConfig {
            n_trials: 5,
            seed: None,
            ..Default::default()
        };
        let router = SabreRouter::with_config(&self.topology, config);
        router.route_multi_trial(dag, dag.n_qubits)
    }

    /// Route with a specific initial layout (single bidirectional pass).
    pub fn route_with_layout(
        &self,
        dag: &QuantumDAG,
        layout: &QubitMapping,
    ) -> Result<(QuantumDAG, QubitMapping), RouterError> {
        let router = SabreRouter::new(&self.topology);
        router.route_bidirectional(dag, layout)
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
        // Single forward pass with identity layout guarantees SWAPs
        let topo = CouplingMap::linear(4);
        let sabre = SabreRouter::new(&topo);
        let layout = QubitMapping::identity(4);

        let mut dag = QuantumDAG::new(4, 0);
        dag.add_op(OpType::CNOT, &[0, 3]); // non-adjacent on 0-1-2-3

        let (routed, _mapping) = sabre.route(&dag, &layout).unwrap();
        // With identity layout on linear(4), CNOT(0,3) needs SWAPs
        assert!(routed.count_ops_of_type("SWAP") > 0,
            "Expected SWAPs, got {} gates, {} SWAPs",
            routed.gate_count(), routed.count_ops_of_type("SWAP"));
    }

    #[test]
    fn test_insufficient_qubits() {
        let topo = CouplingMap::linear(2);
        let router = Router::new(topo);

        let dag = QuantumDAG::new(5, 0);
        assert!(router.route(&dag).is_err());
    }

    #[test]
    fn test_coupling_map_from_edges() {
        let topo = CouplingMap::from_edges(4, &[(0, 1), (1, 2), (2, 3)]);

        assert_eq!(topo.n_qubits(), 4);
        assert_eq!(topo.n_edges(), 3);
        assert!(topo.is_connected(0, 1));
        assert!(topo.is_connected(1, 2));
        assert!(!topo.is_connected(0, 2));
        assert_eq!(topo.distance(0, 3), 3);
    }

    #[test]
    fn test_linear_topology_routing() {
        let topo = CouplingMap::from_edges(4, &[(0, 1), (1, 2), (2, 3)]);
        let router = Router::new(topo);

        let mut dag = QuantumDAG::new(4, 0);
        dag.add_op(OpType::H, &[0]);
        dag.add_op(OpType::CNOT, &[0, 3]);

        let (routed, mapping) = router.route(&dag).unwrap();

        assert_eq!(routed.n_qubits, 4);
        assert_eq!(mapping.n_qubits(), 4);
        assert!(routed.gate_count() >= dag.gate_count());
        assert_eq!(routed.count_ops_of_type("CNOT"), 1);
    }
}
