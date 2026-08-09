use crate::{CompilerError, Pass};
use petgraph::visit::EdgeRef;
use petgraph::Direction;
use sf_ir::{OpType, QuantumDAG};

/// Cancels adjacent pairs of self-inverse gates on the same qubits.
///
/// Handles both 1-qubit (H-H, X-X, etc.) and 2-qubit (CX-CX, CZ-CZ, SWAP-SWAP)
/// inverse pairs. For 2-qubit gates, both gates must act on the identical qubit pair.
pub struct GateCancellationPass;

impl GateCancellationPass {
    pub fn new() -> Self {
        Self
    }

    fn can_cancel(&self, op1: &OpType, op2: &OpType, qubits1: &[usize], qubits2: &[usize]) -> bool {
        if qubits1 != qubits2 {
            return false;
        }

        match (op1, op2) {
            // 1-qubit self-inverse pairs
            (OpType::H, OpType::H) => true,
            (OpType::X, OpType::X) => true,
            (OpType::Y, OpType::Y) => true,
            (OpType::Z, OpType::Z) => true,
            (OpType::S, OpType::Sdg) => true,
            (OpType::Sdg, OpType::S) => true,
            (OpType::T, OpType::Tdg) => true,
            (OpType::Tdg, OpType::T) => true,
            (OpType::SX, OpType::SXdg) => true,
            (OpType::SXdg, OpType::SX) => true,
            // 2-qubit self-inverse pairs
            (OpType::CNOT, OpType::CNOT) => true,
            (OpType::CZ, OpType::CZ) => true,
            (OpType::SWAP, OpType::SWAP) => true,
            (OpType::CY, OpType::CY) => true,
            _ => false,
        }
    }

    /// Check if two nodes are directly connected on ALL their shared qubit wires
    /// (i.e., no other gate intervenes between them on any qubit).
    fn are_adjacent_on_all_qubits(
        &self,
        dag: &QuantumDAG,
        n1: petgraph::prelude::NodeIndex,
        n2: petgraph::prelude::NodeIndex,
    ) -> bool {
        let qubits = &dag.graph()[n1].qubits;
        for &q in qubits.iter() {
            let succ_on_q = dag
                .graph()
                .edges_directed(n1, Direction::Outgoing)
                .find(|e| *e.weight() == sf_ir::WireType::Qubit(q))
                .map(|e| e.target());
            if succ_on_q != Some(n2) {
                return false;
            }
        }
        true
    }
}

impl Pass for GateCancellationPass {
    fn name(&self) -> &str {
        "GateCancellationPass"
    }

    fn run(&self, dag: &mut QuantumDAG) -> Result<(), CompilerError> {
        let mut changed = true;

        while changed {
            changed = false;
            let topo = dag.topological_order();

            for &node_id in &topo {
                if !dag.graph().contains_node(node_id) {
                    continue;
                }

                let op1 = dag.graph()[node_id].op_type.clone();
                let qubits1: Vec<usize> = dag.graph()[node_id].qubits.to_vec();

                // For each outgoing qubit edge, look at the successor
                for edge in dag.graph().edges_directed(node_id, Direction::Outgoing) {
                    if let sf_ir::WireType::Qubit(_q) = edge.weight() {
                        let next_node = edge.target();

                        if dag.is_boundary_node(next_node) {
                            continue;
                        }

                        let op2 = &dag.graph()[next_node].op_type;
                        let qubits2: Vec<usize> = dag.graph()[next_node].qubits.to_vec();

                        if self.can_cancel(&op1, op2, &qubits1, &qubits2) {
                            // For multi-qubit gates, verify adjacency on ALL qubits
                            if qubits1.len() > 1
                                && !self.are_adjacent_on_all_qubits(dag, node_id, next_node)
                            {
                                continue;
                            }

                            self.remove_pair(dag, node_id, next_node);
                            changed = true;
                            break;
                        }
                    }
                }

                if changed {
                    break;
                }
            }
        }

        Ok(())
    }
}

impl GateCancellationPass {
    fn remove_pair(
        &self,
        dag: &mut QuantumDAG,
        n1: petgraph::prelude::NodeIndex,
        n2: petgraph::prelude::NodeIndex,
    ) {
        let qubits: Vec<usize> = dag.graph()[n1].qubits.to_vec();

        for q in qubits {
            let pred = dag
                .graph()
                .edges_directed(n1, Direction::Incoming)
                .find(|e| *e.weight() == sf_ir::WireType::Qubit(q))
                .unwrap()
                .source();

            let succ = dag
                .graph()
                .edges_directed(n2, Direction::Outgoing)
                .find(|e| *e.weight() == sf_ir::WireType::Qubit(q))
                .unwrap()
                .target();

            dag.graph_mut()
                .add_edge(pred, succ, sf_ir::WireType::Qubit(q));
        }

        dag.graph_mut().remove_node(n1);
        dag.graph_mut().remove_node(n2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hh_cancels() {
        let mut dag = QuantumDAG::new(1, 0);
        dag.add_op(OpType::H, &[0]);
        dag.add_op(OpType::H, &[0]);
        assert_eq!(dag.gate_count(), 2);

        GateCancellationPass::new().run(&mut dag).unwrap();
        assert_eq!(dag.gate_count(), 0);
    }

    #[test]
    fn test_xx_cancels() {
        let mut dag = QuantumDAG::new(1, 0);
        dag.add_op(OpType::X, &[0]);
        dag.add_op(OpType::X, &[0]);

        GateCancellationPass::new().run(&mut dag).unwrap();
        assert_eq!(dag.gate_count(), 0);
    }

    #[test]
    fn test_s_sdg_cancels() {
        let mut dag = QuantumDAG::new(1, 0);
        dag.add_op(OpType::S, &[0]);
        dag.add_op(OpType::Sdg, &[0]);

        GateCancellationPass::new().run(&mut dag).unwrap();
        assert_eq!(dag.gate_count(), 0);
    }

    #[test]
    fn test_cx_cx_cancels() {
        let mut dag = QuantumDAG::new(2, 0);
        dag.add_op(OpType::CNOT, &[0, 1]);
        dag.add_op(OpType::CNOT, &[0, 1]);

        GateCancellationPass::new().run(&mut dag).unwrap();
        assert_eq!(dag.gate_count(), 0, "CX·CX should cancel");
    }

    #[test]
    fn test_cz_cz_cancels() {
        let mut dag = QuantumDAG::new(2, 0);
        dag.add_op(OpType::CZ, &[0, 1]);
        dag.add_op(OpType::CZ, &[0, 1]);

        GateCancellationPass::new().run(&mut dag).unwrap();
        assert_eq!(dag.gate_count(), 0, "CZ·CZ should cancel");
    }

    #[test]
    fn test_swap_swap_cancels() {
        let mut dag = QuantumDAG::new(2, 0);
        dag.add_op(OpType::SWAP, &[0, 1]);
        dag.add_op(OpType::SWAP, &[0, 1]);

        GateCancellationPass::new().run(&mut dag).unwrap();
        assert_eq!(dag.gate_count(), 0, "SWAP·SWAP should cancel");
    }

    #[test]
    fn test_cx_different_qubits_no_cancel() {
        let mut dag = QuantumDAG::new(3, 0);
        dag.add_op(OpType::CNOT, &[0, 1]);
        dag.add_op(OpType::CNOT, &[0, 2]);

        GateCancellationPass::new().run(&mut dag).unwrap();
        assert_eq!(dag.gate_count(), 2, "Different qubits should not cancel");
    }

    #[test]
    fn test_cx_reversed_qubits_no_cancel() {
        let mut dag = QuantumDAG::new(2, 0);
        dag.add_op(OpType::CNOT, &[0, 1]);
        dag.add_op(OpType::CNOT, &[1, 0]);

        GateCancellationPass::new().run(&mut dag).unwrap();
        assert_eq!(dag.gate_count(), 2, "Reversed CX should not cancel");
    }

    #[test]
    fn test_cx_non_adjacent_no_cancel() {
        let mut dag = QuantumDAG::new(2, 0);
        dag.add_op(OpType::CNOT, &[0, 1]);
        dag.add_op(OpType::H, &[0]);
        dag.add_op(OpType::CNOT, &[0, 1]);

        GateCancellationPass::new().run(&mut dag).unwrap();
        assert_eq!(
            dag.gate_count(),
            3,
            "Intervening H should prevent cancellation"
        );
    }

    #[test]
    fn test_multiple_pairs_cancel() {
        let mut dag = QuantumDAG::new(2, 0);
        dag.add_op(OpType::CNOT, &[0, 1]);
        dag.add_op(OpType::CNOT, &[0, 1]);
        dag.add_op(OpType::H, &[0]);
        dag.add_op(OpType::CZ, &[0, 1]);
        dag.add_op(OpType::CZ, &[0, 1]);

        GateCancellationPass::new().run(&mut dag).unwrap();
        assert_eq!(dag.gate_count(), 1, "Both pairs should cancel, H remains");
    }
}
