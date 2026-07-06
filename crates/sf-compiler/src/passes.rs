use sf_ir::{OpType, QuantumDAG};
use crate::{Pass, CompilerError};
use petgraph::visit::EdgeRef;
use petgraph::Direction;

/// A simple pass that removes pairs of gates that cancel each other out.
/// Example: X followed by X on the same qubit is removed.
pub struct GateCancellationPass;

impl GateCancellationPass {
    pub fn new() -> Self {
        Self
    }

    fn can_cancel(&self, op1: &OpType, op2: &OpType) -> bool {
        match (op1, op2) {
            (OpType::H, OpType::H) => true,
            (OpType::X, OpType::X) => true,
            (OpType::Y, OpType::Y) => true,
            (OpType::Z, OpType::Z) => true,
            (OpType::S, OpType::Sdg) => true,
            (OpType::Sdg, OpType::S) => true,
            (OpType::T, OpType::Tdg) => true,
            (OpType::Tdg, OpType::T) => true,
            // We could also check for Rx(theta), Rx(-theta) etc.
            _ => false,
        }
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
                // If node was already removed in this iteration
                if !dag.graph().contains_node(node_id) {
                    continue;
                }

                let op1 = &dag.graph()[node_id].op_type;
                
                // For each outgoing qubit edge, look at the successor
                let mut qubits_to_cancel = Vec::new();
                for edge in dag.graph().edges_directed(node_id, Direction::Outgoing) {
                    if let sf_ir::WireType::Qubit(q) = edge.weight() {
                        let next_node = edge.target();
                        
                        // If it's a boundary node, we can't cancel
                        if dag.is_boundary_node(next_node) {
                            continue;
                        }

                        let op2 = &dag.graph()[next_node].op_type;
                        
                        // Both nodes must act on EXACTLY the same qubits for simple cancellation
                        if dag.graph()[node_id].qubits == dag.graph()[next_node].qubits {
                            if self.can_cancel(op1, op2) {
                                qubits_to_cancel.push((node_id, next_node, *q));
                            }
                        }
                    }
                }

                if !qubits_to_cancel.is_empty() {
                    // For simplicity in this first pass, we only handle 1-qubit gates or 
                    // multi-qubit gates where ALL qubits match perfectly and both are cancellable.
                    // This logic needs to be careful with commute/anti-commute.
                    let (n1, n2, _q) = qubits_to_cancel[0];
                    
                    // Rewire: pred of n1 -> succ of n2 for each qubit
                    let qubits = dag.graph()[n1].qubits.clone();
                    
                    for &q_idx in &qubits {
                        // Find predecessor of n1 for this qubit
                        let _pred_node = dag.graph()
                            .edges_directed(n1, Direction::Incoming)
                            .find(|e| *e.weight() == sf_ir::WireType::Qubit(q_idx))
                            .map(|e| e.source())
                            .unwrap();
                        
                        // Find successor of n2 for this qubit
                        let _succ_node = dag.graph()
                            .edges_directed(n2, Direction::Outgoing)
                            .find(|e| *e.weight() == sf_ir::WireType::Qubit(q_idx))
                            .map(|e| e.target())
                            .unwrap();
                        
                        // Remove edges
                        // Note: actual removal is tricky while iterating, 
                        // but StableDiGraph handles it if we are careful.
                        // For now, we collect nodes and remove after the loop or use node removal.
                    }
                    
                    // Actually remove nodes and rewire
                    // (Simplified implementation for now)
                    self.remove_pair(dag, n1, n2);
                    changed = true;
                    break; // Restart topo scan after graph change
                }
            }
        }
        
        Ok(())
    }
}

impl GateCancellationPass {
    fn remove_pair(&self, dag: &mut QuantumDAG, n1: petgraph::prelude::NodeIndex, n2: petgraph::prelude::NodeIndex) {
        let qubits = dag.graph()[n1].qubits.clone();
        
        for q in qubits {
            let pred = dag.graph().edges_directed(n1, Direction::Incoming)
                .find(|e| *e.weight() == sf_ir::WireType::Qubit(q))
                .unwrap().source();
            
            let succ = dag.graph().edges_directed(n2, Direction::Outgoing)
                .find(|e| *e.weight() == sf_ir::WireType::Qubit(q))
                .unwrap().target();
            
            dag.graph_mut().add_edge(pred, succ, sf_ir::WireType::Qubit(q));
        }
        
        dag.graph_mut().remove_node(n1);
        dag.graph_mut().remove_node(n2);
    }
}
