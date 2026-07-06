//! SABRE Router — SWAP-Based BidiREctional heuristic search.
//!
//! Reference: Li, Ding, Xie (2019), "Tackling the Qubit Mapping Problem
//! for NISQ-Era Quantum Devices."
//!
//! Algorithm:
//! 1. Build a "front layer" of gates whose dependencies are satisfied
//! 2. For gates in the front layer:
//!    - If both qubits are adjacent on hardware, execute the gate
//!    - Otherwise, evaluate all candidate SWAPs and pick the one
//!      that minimizes the total "nearest neighbor cost" (sum of
//!      distances for all front-layer gates)
//! 3. After a forward pass, reverse the circuit and do a backward pass
//!    (the backward pass's initial layout becomes the forward pass's layout)

use sf_ir::{QuantumDAG, OpType, QubitMapping, PhysicalQubit, QubitId};
use crate::topology::CouplingMap;
use crate::RouterError;

use std::collections::HashSet;

/// SABRE router implementation.
pub struct SabreRouter<'a> {
    coupling: &'a CouplingMap,
    /// Weight for the lookahead term in the heuristic cost
    decay_delta: f64,
    /// Extended set size (lookahead depth)
    extended_set_size: usize,
}

impl<'a> SabreRouter<'a> {
    pub fn new(coupling: &'a CouplingMap) -> Self {
        Self {
            coupling,
            decay_delta: 0.5,
            extended_set_size: 20,
        }
    }

    /// Route a circuit with a given initial layout.
    pub fn route(
        &self,
        dag: &QuantumDAG,
        initial_layout: &QubitMapping,
    ) -> Result<(QuantumDAG, QubitMapping), RouterError> {
        let mut mapping = initial_layout.clone();
        let mut routed_dag = QuantumDAG::new(dag.n_qubits, dag.n_cbits);
        routed_dag.metadata = dag.metadata.clone();

        // Get topological order of 2Q gates that need routing
        let topo = dag.topological_order();
        let mut pending: Vec<(OpType, Vec<QubitId>)> = Vec::new();

        // Collect all operations in topological order
        for &node_id in &topo {
            let op = &dag.graph()[node_id];
            pending.push((op.op_type.clone(), op.qubits.to_vec()));
        }

        let mut i = 0;
        let mut stuck_counter = 0;
        let max_iterations = pending.len() * pending.len() + 1000;

        while i < pending.len() && stuck_counter < max_iterations {
            let (ref op_type, ref logical_qubits) = pending[i];

            if op_type.n_qubits() <= 1 {
                // Single-qubit gate: just remap and add
                let physical = mapping.logical_to_physical(
                    sf_ir::LogicalQubit(logical_qubits[0])
                ).0;
                routed_dag.add_op(op_type.clone(), &[physical]);
                i += 1;
                stuck_counter = 0;
                continue;
            }

            if op_type.n_qubits() == 2 {
                let pq0 = mapping.logical_to_physical(
                    sf_ir::LogicalQubit(logical_qubits[0])
                ).0;
                let pq1 = mapping.logical_to_physical(
                    sf_ir::LogicalQubit(logical_qubits[1])
                ).0;

                if self.coupling.is_connected(pq0, pq1) {
                    // Gate can execute directly
                    routed_dag.add_op(op_type.clone(), &[pq0, pq1]);
                    i += 1;
                    stuck_counter = 0;
                    continue;
                }

                // Need a SWAP — find the best one
                let best_swap = self.find_best_swap(
                    &pending[i..],
                    &mapping,
                );

                if let Some((s0, s1)) = best_swap {
                    // Insert SWAP on physical qubits
                    routed_dag.add_op(OpType::SWAP, &[s0, s1]);
                    mapping.swap_physical(PhysicalQubit(s0), PhysicalQubit(s1));
                    stuck_counter += 1;
                    // Don't increment i — retry the same gate
                    continue;
                } else {
                    return Err(RouterError::RoutingFailed(logical_qubits.clone()));
                }
            }

            // 3-qubit gates: decompose first, then route the pieces
            // For now, just remap all qubits
            let physical_qubits: Vec<usize> = logical_qubits
                .iter()
                .map(|&lq| mapping.logical_to_physical(sf_ir::LogicalQubit(lq)).0)
                .collect();
            routed_dag.add_op(op_type.clone(), &physical_qubits);
            i += 1;
            stuck_counter = 0;
        }

        Ok((routed_dag, mapping))
    }

    /// Find the SWAP that minimizes the nearest-neighbor cost for pending gates.
    fn find_best_swap(
        &self,
        pending: &[(OpType, Vec<QubitId>)],
        mapping: &QubitMapping,
    ) -> Option<(usize, usize)> {
        // Collect the front layer: 2Q gates that need routing
        let front_gates: Vec<(usize, usize)> = pending
            .iter()
            .take(self.extended_set_size)
            .filter(|(op, _qubits)| op.n_qubits() == 2)
            .map(|(_, qubits)| {
                let p0 = mapping.logical_to_physical(sf_ir::LogicalQubit(qubits[0])).0;
                let p1 = mapping.logical_to_physical(sf_ir::LogicalQubit(qubits[1])).0;
                (p0, p1)
            })
            .collect();

        if front_gates.is_empty() {
            return None;
        }

        // Current total cost
        let current_cost = self.compute_cost(&front_gates);

        // Try all candidate SWAPs (neighbors of qubits involved in front gates)
        let mut involved_qubits: HashSet<usize> = HashSet::new();
        for &(p0, p1) in &front_gates {
            involved_qubits.insert(p0);
            involved_qubits.insert(p1);
        }

        let mut candidate_swaps: HashSet<(usize, usize)> = HashSet::new();
        for &q in &involved_qubits {
            for &neighbor in self.coupling.neighbors(q) {
                let swap = if q < neighbor { (q, neighbor) } else { (neighbor, q) };
                candidate_swaps.insert(swap);
            }
        }

        // Evaluate each candidate SWAP
        let mut best: Option<(usize, usize)> = None;
        let mut best_cost = current_cost;

        for &(s0, s1) in &candidate_swaps {
            // Simulate the SWAP
            let swapped_gates: Vec<(usize, usize)> = front_gates
                .iter()
                .map(|&(p0, p1)| {
                    let new_p0 = if p0 == s0 { s1 } else if p0 == s1 { s0 } else { p0 };
                    let new_p1 = if p1 == s0 { s1 } else if p1 == s1 { s0 } else { p1 };
                    (new_p0, new_p1)
                })
                .collect();

            let new_cost = self.compute_cost(&swapped_gates);

            if new_cost < best_cost {
                best_cost = new_cost;
                best = Some((s0, s1));
            }
        }

        best
    }

    /// Compute the nearest-neighbor cost: sum of distances for gate pairs.
    fn compute_cost(&self, gates: &[(usize, usize)]) -> f64 {
        let mut total = 0.0;
        for (i, &(p0, p1)) in gates.iter().enumerate() {
            let dist = self.coupling.distance(p0, p1) as f64;
            // Decay factor: front-layer gates are weighted more
            let weight = 1.0 / (1.0 + i as f64 * self.decay_delta);
            total += dist * weight;
        }
        total
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sabre_adjacent_gates() {
        let coupling = CouplingMap::linear(3);
        let router = SabreRouter::new(&coupling);
        let layout = QubitMapping::identity(3);

        let mut dag = QuantumDAG::new(3, 0);
        dag.add_op(OpType::H, &[0]);
        dag.add_op(OpType::CNOT, &[0, 1]);
        dag.add_op(OpType::CNOT, &[1, 2]);

        let (routed, _) = router.route(&dag, &layout).unwrap();
        // All gates are on adjacent qubits, no SWAPs needed
        assert_eq!(routed.count_ops_of_type("SWAP"), 0);
        assert_eq!(routed.gate_count(), 3); // H + 2 CNOT
    }

    #[test]
    fn test_sabre_inserts_swaps() {
        let coupling = CouplingMap::linear(4);
        let router = SabreRouter::new(&coupling);
        let layout = QubitMapping::identity(4);

        let mut dag = QuantumDAG::new(4, 0);
        dag.add_op(OpType::CNOT, &[0, 3]); // Distance 3 on linear chain

        let (routed, _) = router.route(&dag, &layout).unwrap();
        // Should have inserted SWAPs to bring 0 and 3 adjacent
        assert!(routed.count_ops_of_type("SWAP") > 0);
    }

    #[test]
    fn test_sabre_all_to_all_no_swaps() {
        let coupling = CouplingMap::all_to_all(5);
        let router = SabreRouter::new(&coupling);
        let layout = QubitMapping::identity(5);

        let mut dag = QuantumDAG::new(5, 0);
        dag.add_op(OpType::CNOT, &[0, 4]);
        dag.add_op(OpType::CNOT, &[1, 3]);
        dag.add_op(OpType::CNOT, &[2, 4]);

        let (routed, _) = router.route(&dag, &layout).unwrap();
        assert_eq!(routed.count_ops_of_type("SWAP"), 0);
    }
}
