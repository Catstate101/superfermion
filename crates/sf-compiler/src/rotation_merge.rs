//! Rotation Merging Pass — Merge consecutive rotation gates on the same qubit.
//!
//! Rz(α) · Rz(β) → Rz(α+β)
//! Rx(α) · Rx(β) → Rx(α+β)
//! If merged angle is 0 (mod 2π), the gate is eliminated entirely.

use sf_ir::{OpType, QuantumDAG, WireType};
use crate::{Pass, CompilerError};
use petgraph::Direction;
use petgraph::visit::EdgeRef;
use std::f64::consts::PI;

pub struct RotationMergingPass;

impl RotationMergingPass {
    pub fn new() -> Self {
        Self
    }

    /// Check if two rotation ops are mergeable (same type, same qubit).
    fn can_merge(op1: &OpType, op2: &OpType) -> Option<OpType> {
        use sf_ir::ops::Parameter;

        match (op1, op2) {
            (OpType::Rz(p1), OpType::Rz(p2)) => {
                if let (Some(a), Some(b)) = (p1.try_evaluate(), p2.try_evaluate()) {
                    let sum = normalize_angle(a + b);
                    if sum.abs() < 1e-12 {
                        None // Cancels to identity
                    } else {
                        Some(OpType::Rz(Parameter::Const(sum)))
                    }
                } else {
                    // Can't merge symbolic
                    Some(op1.clone())
                }
            }
            (OpType::Rx(p1), OpType::Rx(p2)) => {
                if let (Some(a), Some(b)) = (p1.try_evaluate(), p2.try_evaluate()) {
                    let sum = normalize_angle(a + b);
                    if sum.abs() < 1e-12 {
                        None
                    } else {
                        Some(OpType::Rx(Parameter::Const(sum)))
                    }
                } else {
                    Some(op1.clone())
                }
            }
            (OpType::Ry(p1), OpType::Ry(p2)) => {
                if let (Some(a), Some(b)) = (p1.try_evaluate(), p2.try_evaluate()) {
                    let sum = normalize_angle(a + b);
                    if sum.abs() < 1e-12 {
                        None
                    } else {
                        Some(OpType::Ry(Parameter::Const(sum)))
                    }
                } else {
                    Some(op1.clone())
                }
            }
            _ => Some(op1.clone()), // Not mergeable
        }
    }
}

/// Normalize angle to [-π, π].
fn normalize_angle(theta: f64) -> f64 {
    let mut a = theta % (2.0 * PI);
    if a > PI {
        a -= 2.0 * PI;
    } else if a < -PI {
        a += 2.0 * PI;
    }
    a
}

impl Pass for RotationMergingPass {
    fn name(&self) -> &str {
        "RotationMergingPass"
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

                let op1 = &dag.graph()[node_id].op_type;
                if !matches!(op1, OpType::Rz(_) | OpType::Rx(_) | OpType::Ry(_)) {
                    continue;
                }

                let qubits = dag.graph()[node_id].qubits.clone();
                if qubits.len() != 1 {
                    continue;
                }
                let qubit = qubits[0];

                // Find the successor on the same qubit wire
                let next_node = dag
                    .graph()
                    .edges_directed(node_id, Direction::Outgoing)
                    .find(|e| *e.weight() == WireType::Qubit(qubit))
                    .map(|e| e.target());

                if let Some(next) = next_node {
                    if dag.is_boundary_node(next) {
                        continue;
                    }

                    let op2 = &dag.graph()[next].op_type;
                    let next_qubits = &dag.graph()[next].qubits;

                    // Must be same qubit, single-qubit gate
                    if next_qubits.len() != 1 || next_qubits[0] != qubit {
                        continue;
                    }

                    let op1_clone = dag.graph()[node_id].op_type.clone();
                    let op2_clone = op2.clone();

                    match Self::can_merge(&op1_clone, &op2_clone) {
                        Some(merged) if merged != op1_clone => {
                            // Replace op1 with merged, remove op2
                            // Rewire: pred(op1) → merged_node → succ(op2)
                            let pred = dag
                                .graph()
                                .edges_directed(node_id, Direction::Incoming)
                                .find(|e| *e.weight() == WireType::Qubit(qubit))
                                .map(|e| e.source())
                                .unwrap();
                            let succ = dag
                                .graph()
                                .edges_directed(next, Direction::Outgoing)
                                .find(|e| *e.weight() == WireType::Qubit(qubit))
                                .map(|e| e.target())
                                .unwrap();

                            let new_node = dag.graph_mut().add_node(
                                sf_ir::QuantumOp::new(merged, &[qubit]),
                            );
                            dag.graph_mut().add_edge(pred, new_node, WireType::Qubit(qubit));
                            dag.graph_mut().add_edge(new_node, succ, WireType::Qubit(qubit));

                            dag.graph_mut().remove_node(node_id);
                            dag.graph_mut().remove_node(next);
                            changed = true;
                            break;
                        }
                        None => {
                            // Cancels to identity — just remove both
                            let pred = dag
                                .graph()
                                .edges_directed(node_id, Direction::Incoming)
                                .find(|e| *e.weight() == WireType::Qubit(qubit))
                                .map(|e| e.source())
                                .unwrap();
                            let succ = dag
                                .graph()
                                .edges_directed(next, Direction::Outgoing)
                                .find(|e| *e.weight() == WireType::Qubit(qubit))
                                .map(|e| e.target())
                                .unwrap();

                            dag.graph_mut().add_edge(pred, succ, WireType::Qubit(qubit));
                            dag.graph_mut().remove_node(node_id);
                            dag.graph_mut().remove_node(next);
                            changed = true;
                            break;
                        }
                        _ => {}
                    }
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sf_ir::ops::Parameter;

    #[test]
    fn test_merge_rz() {
        let mut dag = QuantumDAG::new(1, 0);
        dag.add_op(OpType::Rz(Parameter::Const(0.5)), &[0]);
        dag.add_op(OpType::Rz(Parameter::Const(0.3)), &[0]);
        assert_eq!(dag.gate_count(), 2);

        let pass = RotationMergingPass::new();
        pass.run(&mut dag).unwrap();

        assert_eq!(dag.gate_count(), 1);
        let ops = dag.to_instructions();
        match &ops[0].op_type {
            OpType::Rz(Parameter::Const(v)) => assert!((v - 0.8).abs() < 1e-10),
            other => panic!("Expected Rz(0.8), got {:?}", other),
        }
    }

    #[test]
    fn test_cancel_rz_to_zero() {
        let mut dag = QuantumDAG::new(1, 0);
        dag.add_op(OpType::Rz(Parameter::Const(PI)), &[0]);
        dag.add_op(OpType::Rz(Parameter::Const(-PI)), &[0]);

        let pass = RotationMergingPass::new();
        pass.run(&mut dag).unwrap();

        assert_eq!(dag.gate_count(), 0);
    }

    #[test]
    fn test_no_merge_different_types() {
        let mut dag = QuantumDAG::new(1, 0);
        dag.add_op(OpType::Rz(Parameter::Const(0.5)), &[0]);
        dag.add_op(OpType::Rx(Parameter::Const(0.3)), &[0]);

        let pass = RotationMergingPass::new();
        pass.run(&mut dag).unwrap();

        assert_eq!(dag.gate_count(), 2); // Not merged
    }

    #[test]
    fn test_normalize_angle() {
        assert!((normalize_angle(0.0)).abs() < 1e-12);
        assert!((normalize_angle(2.0 * PI)).abs() < 1e-12);
        assert!((normalize_angle(-2.0 * PI)).abs() < 1e-12);
        assert!((normalize_angle(3.0 * PI) - PI).abs() < 1e-10);
    }
}
