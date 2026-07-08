//! Superconducting hardware decomposition.
//! Target native gate set: {Rz, SX, X, CX} (IBM Eagle/Heron basis)
//!
//! Decomposes arbitrary single-qubit gates into Rz-SX sequences and
//! multi-qubit gates into CX-based circuits.

use sf_ir::{OpType, QuantumDAG, QuantumOp, WireType};
use crate::{Pass, CompilerError};
use petgraph::Direction;
use petgraph::visit::EdgeRef;
use std::f64::consts::PI;

/// Decomposes gates to IBM native basis: {Rz, SX, X, CX}
///
/// Key decompositions:
/// - H → Rz(π) · SX · Rz(π)  (Rz-SX-Rz form)
/// - S → Rz(π/2)
/// - T → Rz(π/4)
/// - Rx(θ) → Rz(π/2) · SX · Rz(θ-π) · SX · Rz(-π/2)  (general)
/// - Ry(θ) → Rz(π/2) · SX · Rz(θ-π) · SX · Rz(-3π/2)
/// - CZ → (H_target) · CX · (H_target) → decomposed H + CX
pub struct SuperconductingDecomposePass;

impl SuperconductingDecomposePass {
    pub fn new() -> Self {
        Self
    }

    fn is_native(op: &OpType) -> bool {
        matches!(
            op,
            OpType::Rz(_)
                | OpType::SX
                | OpType::X
                | OpType::CNOT
                | OpType::Measure
                | OpType::Barrier
                | OpType::Reset
                | OpType::Id
                | OpType::Input
                | OpType::Output
        )
    }
}

impl Pass for SuperconductingDecomposePass {
    fn name(&self) -> &str {
        "SuperconductingDecomposePass"
    }

    fn run(&self, dag: &mut QuantumDAG) -> Result<(), CompilerError> {
        use sf_ir::ops::Parameter;

        let topo = dag.topological_order();
        let mut to_replace: Vec<(petgraph::prelude::NodeIndex, OpType, Vec<usize>)> = Vec::new();

        for node_id in topo {
            let op = &dag.graph()[node_id];
            if !Self::is_native(&op.op_type) {
                to_replace.push((node_id, op.op_type.clone(), op.qubits.to_vec()));
            }
        }

        for (node_id, op_type, qubits) in to_replace {
            let decomposition = match &op_type {
                // H = Rz(π) · SX · Rz(π)
                OpType::H => {
                    let q = qubits[0];
                    vec![
                        (OpType::Rz(Parameter::Const(PI)), vec![q]),
                        (OpType::SX, vec![q]),
                        (OpType::Rz(Parameter::Const(PI)), vec![q]),
                    ]
                }
                // S = Rz(π/2)
                OpType::S => vec![(OpType::Rz(Parameter::Const(PI / 2.0)), vec![qubits[0]])],
                // Sdg = Rz(-π/2)
                OpType::Sdg => vec![(OpType::Rz(Parameter::Const(-PI / 2.0)), vec![qubits[0]])],
                // T = Rz(π/4)
                OpType::T => vec![(OpType::Rz(Parameter::Const(PI / 4.0)), vec![qubits[0]])],
                // Tdg = Rz(-π/4)
                OpType::Tdg => vec![(OpType::Rz(Parameter::Const(-PI / 4.0)), vec![qubits[0]])],
                // Y = X · Rz(π) (global phase ignored)
                OpType::Y => {
                    let q = qubits[0];
                    vec![
                        (OpType::Rz(Parameter::Const(PI)), vec![q]),
                        (OpType::X, vec![q]),
                    ]
                }
                // Z = Rz(π) (up to global phase)
                OpType::Z => vec![(OpType::Rz(Parameter::Const(PI)), vec![qubits[0]])],
                // SXdg = Rz(π) · SX · Rz(π)  (X · SX = SXdg up to phase)
                OpType::SXdg => {
                    let q = qubits[0];
                    vec![
                        (OpType::Rz(Parameter::Const(PI)), vec![q]),
                        (OpType::SX, vec![q]),
                        (OpType::Rz(Parameter::Const(PI)), vec![q]),
                        (OpType::Rz(Parameter::Const(PI)), vec![q]),
                    ]
                }
                // Rx(θ) → Rz(-π/2) · SX · Rz(π-θ) · SX · Rz(-π/2)
                OpType::Rx(param) => {
                    let q = qubits[0];
                    if let Some(theta) = param.try_evaluate() {
                        vec![
                            (OpType::Rz(Parameter::Const(-PI / 2.0)), vec![q]),
                            (OpType::SX, vec![q]),
                            (OpType::Rz(Parameter::Const(PI - theta)), vec![q]),
                            (OpType::SX, vec![q]),
                            (OpType::Rz(Parameter::Const(-PI / 2.0)), vec![q]),
                        ]
                    } else {
                        // Keep parametric — can't decompose symbolically here
                        vec![(op_type.clone(), qubits.clone())]
                    }
                }
                // Ry(θ) → SX · Rz(θ) · SX†  ≈ SX · Rz(θ) · Rz(π) · SX · Rz(π)
                OpType::Ry(param) => {
                    let q = qubits[0];
                    if let Some(theta) = param.try_evaluate() {
                        vec![
                            (OpType::Rz(Parameter::Const(PI / 2.0)), vec![q]),
                            (OpType::SX, vec![q]),
                            (OpType::Rz(Parameter::Const(theta - PI)), vec![q]),
                            (OpType::SX, vec![q]),
                            (OpType::Rz(Parameter::Const(-3.0 * PI / 2.0)), vec![q]),
                        ]
                    } else {
                        vec![(op_type.clone(), qubits.clone())]
                    }
                }
                // CZ → H(target) · CX · H(target) → decompose H into native
                OpType::CZ => {
                    let (ctrl, tgt) = (qubits[0], qubits[1]);
                    vec![
                        // H on target
                        (OpType::Rz(Parameter::Const(PI)), vec![tgt]),
                        (OpType::SX, vec![tgt]),
                        (OpType::Rz(Parameter::Const(PI)), vec![tgt]),
                        // CX
                        (OpType::CNOT, vec![ctrl, tgt]),
                        // H on target
                        (OpType::Rz(Parameter::Const(PI)), vec![tgt]),
                        (OpType::SX, vec![tgt]),
                        (OpType::Rz(Parameter::Const(PI)), vec![tgt]),
                    ]
                }
                // R1(φ) = Rz(φ) (up to global phase)
                OpType::R1(p) | OpType::P(p) => {
                    vec![(OpType::Rz(p.clone()), vec![qubits[0]])]
                }
                // ECR → native IBM ECR (already native on Heron, but not Eagle)
                // For Eagle basis: decompose into CX
                OpType::ECR => {
                    let (q0, q1) = (qubits[0], qubits[1]);
                    vec![
                        (OpType::SX, vec![q0]),
                        (OpType::CNOT, vec![q0, q1]),
                        (OpType::X, vec![q0]),
                    ]
                }
                // Fallback: keep as-is
                _ => vec![(op_type.clone(), qubits.clone())],
            };

            // Replace the node with the decomposition
            // Use the same technique as BasicDecomposePass
            replace_node_with_gates(dag, node_id, &decomposition);
        }

        Ok(())
    }
}

/// Replace a single node in the DAG with a sequence of gates.
pub fn replace_node_with_gates(
    dag: &mut QuantumDAG,
    node_id: petgraph::prelude::NodeIndex,
    gates: &[(OpType, Vec<usize>)],
) {
    let original_qubits = dag.graph()[node_id].qubits.clone();

    // Find predecessors
    let mut preds = std::collections::HashMap::new();
    for &q in &original_qubits {
        let pred = dag
            .graph()
            .edges_directed(node_id, Direction::Incoming)
            .find(|e| *e.weight() == WireType::Qubit(q))
            .map(|e| e.source())
            .expect("Node must have input for all qubits");
        preds.insert(q, pred);
    }

    // Find successors
    let mut succs = std::collections::HashMap::new();
    for &q in &original_qubits {
        let succ = dag
            .graph()
            .edges_directed(node_id, Direction::Outgoing)
            .find(|e| *e.weight() == WireType::Qubit(q))
            .map(|e| e.target())
            .expect("Node must have output for all qubits");
        succs.insert(q, succ);
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
        let last_node = current_preds[&q];
        let target = succs[&q];
        dag.graph_mut()
            .add_edge(last_node, target, WireType::Qubit(q));
    }

    // Remove original node
    dag.graph_mut().remove_node(node_id);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_h_decomposition() {
        let mut dag = QuantumDAG::new(1, 0);
        dag.add_op(OpType::H, &[0]);

        let pass = SuperconductingDecomposePass::new();
        pass.run(&mut dag).unwrap();

        // H → Rz · SX · Rz
        assert_eq!(dag.count_ops_of_type("Rz"), 2);
        assert_eq!(dag.count_ops_of_type("SX"), 1);
        assert_eq!(dag.count_ops_of_type("H"), 0);
    }

    #[test]
    fn test_s_decomposition() {
        let mut dag = QuantumDAG::new(1, 0);
        dag.add_op(OpType::S, &[0]);

        let pass = SuperconductingDecomposePass::new();
        pass.run(&mut dag).unwrap();

        assert_eq!(dag.count_ops_of_type("Rz"), 1);
        assert_eq!(dag.count_ops_of_type("S"), 0);
    }

    #[test]
    fn test_native_gates_untouched() {
        let mut dag = QuantumDAG::new(2, 0);
        dag.add_op(OpType::SX, &[0]);
        dag.add_op(OpType::CNOT, &[0, 1]);
        dag.add_op(OpType::X, &[1]);

        let pass = SuperconductingDecomposePass::new();
        pass.run(&mut dag).unwrap();

        // Already native, shouldn't change
        assert_eq!(dag.gate_count(), 3);
    }
}
