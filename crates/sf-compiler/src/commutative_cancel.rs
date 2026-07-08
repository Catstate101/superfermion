//! Commutative Cancellation Pass.
//!
//! Walks the DAG and builds "commutation groups" per qubit wire.
//! Within each group, gates that commute are grouped together and
//! inverse pairs are cancelled, even if they are not directly adjacent.
//!
//! This is how Qiskit eliminates redundant CX gates from SWAP decomposition:
//! after routing inserts `CX CX CX` for each SWAP, the middle CX may commute
//! past a 1Q gate to cancel with an adjacent CX from a neighboring SWAP.

use sf_ir::{OpType, QuantumDAG, commutation};
use crate::{Pass, CompilerError};
use petgraph::visit::EdgeRef;
use petgraph::Direction;

pub struct CommutativeCancellationPass;

impl CommutativeCancellationPass {
    pub fn new() -> Self {
        Self
    }
}

impl Pass for CommutativeCancellationPass {
    fn name(&self) -> &str {
        "CommutativeCancellationPass"
    }

    fn run(&self, dag: &mut QuantumDAG) -> Result<(), CompilerError> {
        let mut changed = true;

        while changed {
            changed = false;

            let topo = dag.topological_order();
            let mut cancelled = std::collections::HashSet::new();

            for (idx, &node_id) in topo.iter().enumerate() {
                if cancelled.contains(&node_id) || !dag.graph().contains_node(node_id) {
                    continue;
                }

                let op1 = dag.graph()[node_id].op_type.clone();
                let qubits1: Vec<usize> = dag.graph()[node_id].qubits.to_vec();

                if op1.is_boundary() || op1 == OpType::Barrier || op1.is_measurement() {
                    continue;
                }

                // Look ahead for a matching inverse gate that can be reached
                // by commuting through intermediate gates.
                for &candidate_id in topo[idx + 1..].iter() {
                    if cancelled.contains(&candidate_id) || !dag.graph().contains_node(candidate_id) {
                        continue;
                    }

                    let op2 = dag.graph()[candidate_id].op_type.clone();
                    let qubits2: Vec<usize> = dag.graph()[candidate_id].qubits.to_vec();

                    // Skip if operating on different qubits or not sharing any
                    if qubits1.iter().all(|q| !qubits2.contains(q)) {
                        continue;
                    }

                    // Check if op1 and op2 are inverse pairs
                    if is_inverse_pair(&op1, &qubits1, &op2, &qubits2) {
                        // Check if op1 can commute through all gates between them on shared qubits
                        if can_commute_through(dag, &topo[idx + 1..], node_id, candidate_id, &op1, &qubits1, &cancelled) {
                            cancelled.insert(node_id);
                            cancelled.insert(candidate_id);
                            changed = true;
                            break;
                        }
                    }

                    // If this gate doesn't commute with op1 and shares qubits, stop looking
                    if !commutation::commutes(&op1, &qubits1, &op2, &qubits2) {
                        break;
                    }
                }
            }

            // Remove all cancelled nodes
            for node_id in &cancelled {
                if dag.graph().contains_node(*node_id) {
                    remove_node_and_rewire(dag, *node_id);
                }
            }
        }

        Ok(())
    }
}

/// Check if two operations form an inverse pair (cancel to identity).
fn is_inverse_pair(op1: &OpType, q1: &[usize], op2: &OpType, q2: &[usize]) -> bool {
    if q1 != q2 {
        return false;
    }
    match (op1, op2) {
        // 1q self-inverse
        (OpType::H, OpType::H) => true,
        (OpType::X, OpType::X) => true,
        (OpType::Y, OpType::Y) => true,
        (OpType::Z, OpType::Z) => true,
        (OpType::S, OpType::Sdg) | (OpType::Sdg, OpType::S) => true,
        (OpType::T, OpType::Tdg) | (OpType::Tdg, OpType::T) => true,
        (OpType::SX, OpType::SXdg) | (OpType::SXdg, OpType::SX) => true,
        // 2q self-inverse
        (OpType::CNOT, OpType::CNOT) => true,
        (OpType::CZ, OpType::CZ) => true,
        (OpType::SWAP, OpType::SWAP) => true,
        (OpType::CY, OpType::CY) => true,
        // Rotation inverses
        (OpType::Rz(p1), OpType::Rz(p2)) => {
            if let (Some(a), Some(b)) = (p1.try_evaluate(), p2.try_evaluate()) {
                ((a + b) % (2.0 * std::f64::consts::PI)).abs() < 1e-10
            } else {
                false
            }
        }
        (OpType::Rx(p1), OpType::Rx(p2)) => {
            if let (Some(a), Some(b)) = (p1.try_evaluate(), p2.try_evaluate()) {
                ((a + b) % (2.0 * std::f64::consts::PI)).abs() < 1e-10
            } else {
                false
            }
        }
        _ => false,
    }
}

/// Check if `op1` on `qubits1` can commute through all gates between
/// `start_node` (exclusive) and `end_node` (exclusive) in topological order.
fn can_commute_through(
    dag: &QuantumDAG,
    topo_slice: &[petgraph::prelude::NodeIndex],
    _start_node: petgraph::prelude::NodeIndex,
    end_node: petgraph::prelude::NodeIndex,
    op1: &OpType,
    qubits1: &[usize],
    cancelled: &std::collections::HashSet<petgraph::prelude::NodeIndex>,
) -> bool {
    for &between_id in topo_slice {
        if between_id == end_node {
            return true;
        }
        if cancelled.contains(&between_id) || !dag.graph().contains_node(between_id) {
            continue;
        }

        let between_op = &dag.graph()[between_id].op_type;
        let between_qubits: Vec<usize> = dag.graph()[between_id].qubits.to_vec();

        // If disjoint qubits, no need to commute
        if qubits1.iter().all(|q| !between_qubits.contains(q)) {
            continue;
        }

        if !commutation::commutes(op1, qubits1, between_op, &between_qubits) {
            return false;
        }
    }
    true
}

/// Remove a single node from the DAG and rewire predecessor→successor edges.
fn remove_node_and_rewire(dag: &mut QuantumDAG, node_id: petgraph::prelude::NodeIndex) {
    let qubits: Vec<usize> = dag.graph()[node_id].qubits.to_vec();

    for q in qubits {
        let pred = dag.graph()
            .edges_directed(node_id, Direction::Incoming)
            .find(|e| *e.weight() == sf_ir::WireType::Qubit(q))
            .map(|e| e.source());

        let succ = dag.graph()
            .edges_directed(node_id, Direction::Outgoing)
            .find(|e| *e.weight() == sf_ir::WireType::Qubit(q))
            .map(|e| e.target());

        if let (Some(p), Some(s)) = (pred, succ) {
            dag.graph_mut().add_edge(p, s, sf_ir::WireType::Qubit(q));
        }
    }

    dag.graph_mut().remove_node(node_id);
}

#[cfg(test)]
mod tests {
    use super::*;
    use sf_ir::ops::Parameter;

    #[test]
    fn test_cx_cancel_through_rz_on_control() {
        // CX(0,1) - Rz(0) - CX(0,1) → Rz(0) only
        // Rz on control commutes with CX, so the two CX cancel
        let mut dag = QuantumDAG::new(2, 0);
        dag.add_op(OpType::CNOT, &[0, 1]);
        dag.add_op(OpType::Rz(Parameter::Const(0.5)), &[0]);
        dag.add_op(OpType::CNOT, &[0, 1]);

        CommutativeCancellationPass::new().run(&mut dag).unwrap();
        assert_eq!(dag.count_ops_of_type("CNOT"), 0, "CX pair should cancel");
        assert_eq!(dag.count_ops_of_type("Rz"), 1, "Rz should remain");
    }

    #[test]
    fn test_cx_no_cancel_through_h_on_control() {
        // CX(0,1) - H(0) - CX(0,1) → no cancellation (H doesn't commute with CX on control)
        let mut dag = QuantumDAG::new(2, 0);
        dag.add_op(OpType::CNOT, &[0, 1]);
        dag.add_op(OpType::H, &[0]);
        dag.add_op(OpType::CNOT, &[0, 1]);

        CommutativeCancellationPass::new().run(&mut dag).unwrap();
        assert_eq!(dag.gate_count(), 3, "No cancellation expected");
    }

    #[test]
    fn test_cz_cancel_through_z_diagonal() {
        // CZ(0,1) - S(0) - T(1) - CZ(0,1) → S(0), T(1)
        let mut dag = QuantumDAG::new(2, 0);
        dag.add_op(OpType::CZ, &[0, 1]);
        dag.add_op(OpType::S, &[0]);
        dag.add_op(OpType::T, &[1]);
        dag.add_op(OpType::CZ, &[0, 1]);

        CommutativeCancellationPass::new().run(&mut dag).unwrap();
        assert_eq!(dag.count_ops_of_type("CZ"), 0, "CZ pair should cancel");
        assert_eq!(dag.gate_count(), 2, "S and T should remain");
    }

    #[test]
    fn test_no_cancel_non_commuting_intermediate() {
        // CX(0,1) - H(1) - CX(0,1) → no cancel (H on target doesn't commute with CX)
        let mut dag = QuantumDAG::new(2, 0);
        dag.add_op(OpType::CNOT, &[0, 1]);
        dag.add_op(OpType::H, &[1]);
        dag.add_op(OpType::CNOT, &[0, 1]);

        CommutativeCancellationPass::new().run(&mut dag).unwrap();
        assert_eq!(dag.gate_count(), 3);
    }

    #[test]
    fn test_cx_cancel_through_x_on_target() {
        // CX(0,1) - X(1) - CX(0,1) → X(1) only
        // X on target commutes with CX
        let mut dag = QuantumDAG::new(2, 0);
        dag.add_op(OpType::CNOT, &[0, 1]);
        dag.add_op(OpType::X, &[1]);
        dag.add_op(OpType::CNOT, &[0, 1]);

        CommutativeCancellationPass::new().run(&mut dag).unwrap();
        assert_eq!(dag.count_ops_of_type("CNOT"), 0, "CX pair should cancel");
        assert_eq!(dag.count_ops_of_type("X"), 1, "X should remain");
    }

    #[test]
    fn test_swap_decomposed_cancellation() {
        // Simulates what happens after SWAP decomposition:
        // CX(0,1) CX(1,0) CX(0,1) CX(0,1) CX(1,0) CX(0,1)
        // The last CX(0,1) of first SWAP and first CX(0,1) of second SWAP cancel
        let mut dag = QuantumDAG::new(2, 0);
        dag.add_op(OpType::CNOT, &[0, 1]);
        dag.add_op(OpType::CNOT, &[1, 0]);
        dag.add_op(OpType::CNOT, &[0, 1]);
        dag.add_op(OpType::CNOT, &[0, 1]); // cancels with above
        dag.add_op(OpType::CNOT, &[1, 0]);
        dag.add_op(OpType::CNOT, &[0, 1]);

        CommutativeCancellationPass::new().run(&mut dag).unwrap();
        assert!(dag.gate_count() < 6, "Should cancel at least one pair, got {}", dag.gate_count());
    }

    #[test]
    fn test_disjoint_qubit_gates_allow_commutation() {
        // CX(0,1) - H(2) - CX(0,1) → CX cancel, H remains
        let mut dag = QuantumDAG::new(3, 0);
        dag.add_op(OpType::CNOT, &[0, 1]);
        dag.add_op(OpType::H, &[2]);
        dag.add_op(OpType::CNOT, &[0, 1]);

        CommutativeCancellationPass::new().run(&mut dag).unwrap();
        assert_eq!(dag.count_ops_of_type("CNOT"), 0);
        assert_eq!(dag.count_ops_of_type("H"), 1);
    }
}
