//! Pauli Twirling Pass — Rust implementation.
//!
//! Wraps each two-qubit Clifford gate (CNOT, CZ) in a random Pauli sandwich
//! that preserves the logical circuit behavior while converting coherent errors
//! into stochastic Pauli errors.
//!
//! The twirl condition: P_after · G = G · P_before
//! For CNOT/CZ gates, there are 16 valid Pauli pairs (P_before, P_after).

use sf_ir::{OpType, QuantumDAG, QuantumOp, WireType};
use crate::{Pass, CompilerError};
use petgraph::Direction;
use petgraph::visit::EdgeRef;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand::Rng;
use std::cell::RefCell;

/// 14 valid CNOT twirl pairs: (P1_before, P2_before, P1_after, P2_after)
/// Each pair satisfies P_after · CNOT = CNOT · P_before.
/// Pauli encoding: 0=I, 1=X, 2=Z, 3=Y.
/// Verified numerically: for each (p1_before, p2_before), exactly one (p1_after, p2_after) satisfies.
const CNOT_TWIRL: &[(u8, u8, u8, u8)] = &[
    (0, 0, 0, 0), // I,I → I,I
    (0, 1, 0, 1), // I,X → I,X
    (0, 2, 2, 2), // I,Z → Z,Z
    (0, 3, 2, 3), // I,Y → Z,Y
    (1, 0, 1, 1), // X,I → X,X
    (1, 1, 1, 0), // X,X → X,I
    (1, 3, 3, 2), // X,Y → Y,Z
    (2, 0, 2, 0), // Z,I → Z,I
    (2, 1, 2, 1), // Z,X → Z,X
    (2, 2, 0, 2), // Z,Z → I,Z
    (2, 3, 0, 3), // Z,Y → I,Y
    (3, 0, 3, 1), // Y,I → Y,X
    (3, 1, 3, 0), // Y,X → Y,I
    (3, 2, 1, 3), // Y,Z → X,Y
];

/// 14 valid CZ twirl pairs: (P1_before, P2_before, P1_after, P2_after)
/// Each pair satisfies P_after · CZ = CZ · P_before.
/// Pauli encoding: 0=I, 1=X, 2=Z, 3=Y.
/// Verified numerically: for each (p1_before, p2_before), exactly one (p1_after, p2_after) satisfies.
const CZ_TWIRL: &[(u8, u8, u8, u8)] = &[
    (0, 0, 0, 0), // I,I → I,I
    (0, 1, 2, 1), // I,X → Z,X
    (0, 2, 0, 2), // I,Z → I,Z
    (0, 3, 2, 3), // I,Y → Z,Y
    (1, 0, 1, 2), // X,I → X,Z
    (1, 1, 3, 3), // X,X → Y,Y
    (1, 2, 1, 0), // X,Z → X,I
    (2, 0, 2, 0), // Z,I → Z,I
    (2, 1, 0, 1), // Z,X → I,X
    (2, 2, 2, 2), // Z,Z → Z,Z
    (2, 3, 0, 3), // Z,Y → I,Y
    (3, 0, 3, 2), // Y,I → Y,Z
    (3, 2, 3, 0), // Y,Z → Y,I
    (3, 3, 1, 1), // Y,Y → X,X
];

/// Map 0=I, 1=X, 2=Z, 3=Y to OpType.
fn pauli_op(idx: u8) -> OpType {
    match idx {
        0 => OpType::Id,
        1 => OpType::X,
        2 => OpType::Z,
        3 => OpType::Y,
        _ => OpType::Id,
    }
}

/// Pauli twirling pass for two-qubit Clifford gates.
pub struct PauliTwirlPass {
    rng: RefCell<StdRng>,
}

impl PauliTwirlPass {
    pub fn new(seed: u64) -> Self {
        Self {
            rng: RefCell::new(StdRng::seed_from_u64(seed)),
        }
    }
}

impl Pass for PauliTwirlPass {
    fn name(&self) -> &str {
        "PauliTwirlPass"
    }

    fn run(&self, dag: &mut QuantumDAG) -> Result<(), CompilerError> {
        let topo = dag.topological_order();

        // Collect nodes that are CNOT or CZ
        let mut to_twirl: Vec<(petgraph::prelude::NodeIndex, OpType, Vec<usize>)> = Vec::new();
        for &node_id in &topo {
            let op = &dag.graph()[node_id];
            match &op.op_type {
                OpType::CNOT | OpType::CZ => {
                    to_twirl.push((node_id, op.op_type.clone(), op.qubits.to_vec()));
                }
                _ => {}
            }
        }

        // For each CNOT/CZ, insert Pauli sandwich
        for (node_id, op_type, qubits) in to_twirl {
            let pairs = match &op_type {
                OpType::CNOT => CNOT_TWIRL,
                OpType::CZ => CZ_TWIRL,
                _ => continue,
            };

            let pair = pairs[self.rng.borrow_mut().gen_range(0..pairs.len())];
            let (p1_b, p2_b, p1_a, p2_a) = (pair.0, pair.1, pair.2, pair.3);

            let q0 = qubits[0];
            let q1 = qubits[1];

            // Build the sandwich: P1_b(q0), P2_b(q1), G(q0,q1), P1_a(q0), P2_a(q1)
            let sandwich: Vec<(OpType, Vec<usize>)> = {
                let mut ops = Vec::new();

                // Paulis before (skip Identity)
                if p1_b != 0 {
                    ops.push((pauli_op(p1_b), vec![q0]));
                }
                if p2_b != 0 {
                    ops.push((pauli_op(p2_b), vec![q1]));
                }

                // Original gate
                ops.push((op_type.clone(), vec![q0, q1]));

                // Paulis after (skip Identity)
                if p1_a != 0 {
                    ops.push((pauli_op(p1_a), vec![q0]));
                }
                if p2_a != 0 {
                    ops.push((pauli_op(p2_a), vec![q1]));
                }

                ops
            };

            // Replace the node with the sandwich
            replace_node_with_gates(dag, node_id, &sandwich);
        }

        Ok(())
    }
}

/// Replace a single node in the DAG with a sequence of gates.
/// Reuses the same technique as the decompose modules.
fn replace_node_with_gates(
    dag: &mut QuantumDAG,
    node_id: petgraph::prelude::NodeIndex,
    gates: &[(OpType, Vec<usize>)],
) {
    use std::collections::HashMap;

    let original_qubits = dag.graph()[node_id].qubits.clone();

    // Find predecessors for each qubit
    let mut preds = HashMap::new();
    for &q in &original_qubits {
        let pred = dag
            .graph()
            .edges_directed(node_id, Direction::Incoming)
            .find(|e| *e.weight() == WireType::Qubit(q))
            .map(|e| e.source());
        if let Some(p) = pred {
            preds.insert(q, p);
        }
    }

    // Find successors for each qubit
    let mut succs = HashMap::new();
    for &q in &original_qubits {
        let succ = dag
            .graph()
            .edges_directed(node_id, Direction::Outgoing)
            .find(|e| *e.weight() == WireType::Qubit(q))
            .map(|e| e.target());
        if let Some(s) = succ {
            succs.insert(q, s);
        }
    }

    // Insert new gates
    let mut current_preds = preds;
    for (new_op, qs) in gates {
        let new_node = dag
            .graph_mut()
            .add_node(QuantumOp::new(new_op.clone(), qs));
        for &q in qs {
            if let Some(&p) = current_preds.get(&q) {
                dag.graph_mut()
                    .add_edge(p, new_node, WireType::Qubit(q));
            }
            current_preds.insert(q, new_node);
        }
    }

    // Wire to successors
    for &q in &original_qubits {
        if let (Some(&last_node), Some(&target)) = (current_preds.get(&q), succs.get(&q)) {
            dag.graph_mut()
                .add_edge(last_node, target, WireType::Qubit(q));
        }
    }

    // Remove original node
    dag.graph_mut().remove_node(node_id);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pauli_twirl_cnot() {
        let mut dag = QuantumDAG::new(2, 0);
        dag.add_op(OpType::CNOT, &[0, 1]);

        let pass = PauliTwirlPass::new(42);
        pass.run(&mut dag).unwrap();

        // CNOT should still be present
        assert!(dag.count_ops_of_type("CNOT") >= 1);
    }

    #[test]
    fn test_pauli_twirl_preserves_count() {
        let mut dag = QuantumDAG::new(3, 0);
        dag.add_op(OpType::H, &[0]);
        dag.add_op(OpType::CNOT, &[0, 1]);
        dag.add_op(OpType::CZ, &[1, 2]);

        let pass = PauliTwirlPass::new(123);
        pass.run(&mut dag).unwrap();

        // H should remain, CNOT and CZ should remain (possibly with extra Paulis)
        assert!(dag.count_ops_of_type("H") >= 1);
        assert!(dag.count_ops_of_type("CNOT") >= 1);
        assert!(dag.count_ops_of_type("CZ") >= 1);
    }
}
