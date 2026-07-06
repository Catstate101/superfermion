//! Gate decomposition modules for different hardware targets.

pub mod superconducting;

use sf_ir::{QuantumDAG, OpType};
use crate::{Pass, CompilerError};

pub struct BasicDecomposePass;

impl BasicDecomposePass {
    pub fn new() -> Self {
        Self
    }
}

use petgraph::Direction;
use petgraph::visit::EdgeRef;

impl Pass for BasicDecomposePass {
    fn name(&self) -> &str {
        "BasicDecomposePass"
    }

    fn run(&self, dag: &mut QuantumDAG) -> Result<(), CompilerError> {
        let topo = dag.topological_order();
        let mut to_replace = Vec::new();
        
        for node_id in topo {
            let op = &dag.graph()[node_id];
            match op.op_type {
                OpType::SWAP => to_replace.push((node_id, OpType::SWAP)),
                OpType::CCX => to_replace.push((node_id, OpType::CCX)),
                _ => {}
            }
        }

        for (node_id, op_type) in to_replace {
            let op = dag.graph()[node_id].clone();
            
            match op_type {
                OpType::SWAP => {
                    let (q0, q1) = (op.qubits[0], op.qubits[1]);
                    // SWAP(q0, q1) = CX(q0, q1), CX(q1, q0), CX(q0, q1)
                    
                    self.replace_node_with_gates(dag, node_id, &[
                        (OpType::CNOT, vec![q0, q1]),
                        (OpType::CNOT, vec![q1, q0]),
                        (OpType::CNOT, vec![q0, q1]),
                    ]);
                },
                OpType::CCX => {
                    return Err(CompilerError::UnsupportedGate(
                        "CCX/Toffoli".to_string()
                    ));
                },
                _ => {}
            }
        }
        
        Ok(())
    }
}

impl BasicDecomposePass {
    fn replace_node_with_gates(&self, dag: &mut QuantumDAG, node_id: petgraph::prelude::NodeIndex, gates: &[(OpType, Vec<sf_ir::QubitId>)]) {
        let qubits = dag.graph()[node_id].qubits.clone();
        
        // Find input wires
        let mut preds = std::collections::HashMap::new();
        for &q in &qubits {
            let pred = dag.graph()
                .edges_directed(node_id, Direction::Incoming)
                .find(|e| *e.weight() == sf_ir::WireType::Qubit(q))
                .map(|e| e.source())
                .expect("Gate must have input for all qubits");
            preds.insert(q, pred);
        }
        
        // Find output wires
        let mut succs = std::collections::HashMap::new();
        for &q in &qubits {
            let succ = dag.graph()
                .edges_directed(node_id, Direction::Outgoing)
                .find(|e| *e.weight() == sf_ir::WireType::Qubit(q))
                .map(|e| e.target())
                .expect("Gate must have output for all qubits");
            succs.insert(q, succ);
        }
        
        // Remove the original node edges
        // (Removing the node itself is more robust later)
        
        // Insert new gates
        let mut current_preds = preds;
        for (new_op, qs) in gates {
            let new_node = dag.graph_mut().add_node(sf_ir::QuantumOp::new(new_op.clone(), qs));
            for &q in qs {
                let p = current_preds[&q];
                dag.graph_mut().add_edge(p, new_node, sf_ir::WireType::Qubit(q));
                current_preds.insert(q, new_node);
            }
        }
        
        // Wire the final stage to the original successors
        for &q in &qubits {
            let last_node = current_preds[&q];
            let target = succs[&q];
            dag.graph_mut().add_edge(last_node, target, sf_ir::WireType::Qubit(q));
        }
        
        // Final cleanup
        dag.graph_mut().remove_node(node_id);
    }
}
