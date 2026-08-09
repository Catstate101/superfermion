//! Target-driven basis translation pass.
//!
//! Decomposes any gate not in the target native basis set into an equivalent
//! sequence of native gates. This replaces the Python `BasisTranslationPass`
//! shim with a fast, in-place DAG transformation.

use crate::{CompilerError, Pass};
use sf_ir::ops::Parameter;
use sf_ir::{OpType, QuantumDAG};
use std::collections::HashSet;
use std::f64::consts::PI;

const PI2: f64 = PI / 2.0;

pub struct BasisTranslationPass {
    native: HashSet<String>,
}

impl BasisTranslationPass {
    pub fn new(native_gates: &[String]) -> Self {
        Self {
            native: native_gates.iter().map(|s| s.to_uppercase()).collect(),
        }
    }

    fn is_native(&self, op: &OpType) -> bool {
        if op.is_boundary() {
            return true;
        }
        match op {
            OpType::Measure | OpType::MeasureAll | OpType::Reset | OpType::Barrier | OpType::Id => {
                true
            }
            _ => self.native.contains(&op.name()),
        }
    }

    fn has(&self, name: &str) -> bool {
        self.native.contains(name)
    }

    fn h_decomp(&self, q: usize) -> Vec<(OpType, Vec<usize>)> {
        if self.has("H") {
            vec![(OpType::H, vec![q])]
        } else if self.has("SX") {
            vec![
                (OpType::Rz(Parameter::Const(PI2)), vec![q]),
                (OpType::SX, vec![q]),
                (OpType::Rz(Parameter::Const(PI2)), vec![q]),
            ]
        } else {
            vec![
                (OpType::Rz(Parameter::Const(PI2)), vec![q]),
                (OpType::Rx(Parameter::Const(PI2)), vec![q]),
                (OpType::Rz(Parameter::Const(PI2)), vec![q]),
            ]
        }
    }

    fn cx_decomp(&self, ctrl: usize, tgt: usize) -> Vec<(OpType, Vec<usize>)> {
        if self.has("CX") || self.has("CNOT") {
            vec![(OpType::CNOT, vec![ctrl, tgt])]
        } else if self.has("CZ") {
            let mut v = self.h_decomp(tgt);
            v.push((OpType::CZ, vec![ctrl, tgt]));
            v.extend(self.h_decomp(tgt));
            v
        } else if self.has("ECR") {
            vec![
                (OpType::Rz(Parameter::Const(PI2)), vec![ctrl]),
                (OpType::Rx(Parameter::Const(PI2)), vec![ctrl]),
                (OpType::ECR, vec![ctrl, tgt]),
                (OpType::Rx(Parameter::Const(PI2)), vec![tgt]),
            ]
        } else {
            vec![(OpType::CNOT, vec![ctrl, tgt])]
        }
    }

    fn decompose(&self, op: &OpType, qubits: &[usize]) -> Vec<(OpType, Vec<usize>)> {
        match op {
            OpType::X => {
                let q = qubits[0];
                if self.has("X") {
                    vec![(OpType::X, vec![q])]
                } else if self.has("SX") {
                    vec![(OpType::SX, vec![q]), (OpType::SX, vec![q])]
                } else {
                    vec![(OpType::Rx(Parameter::Const(PI)), vec![q])]
                }
            }
            OpType::Y => {
                let q = qubits[0];
                if self.has("SX") {
                    vec![
                        (OpType::Rz(Parameter::Const(PI)), vec![q]),
                        (OpType::SX, vec![q]),
                        (OpType::SX, vec![q]),
                    ]
                } else {
                    vec![
                        (OpType::Rx(Parameter::Const(PI)), vec![q]),
                        (OpType::Rz(Parameter::Const(PI)), vec![q]),
                    ]
                }
            }
            OpType::Z => vec![(OpType::Rz(Parameter::Const(PI)), vec![qubits[0]])],
            OpType::H => self.h_decomp(qubits[0]),
            OpType::CNOT => self.cx_decomp(qubits[0], qubits[1]),
            OpType::CZ => {
                if self.has("CZ") {
                    vec![(OpType::CZ, vec![qubits[0], qubits[1]])]
                } else {
                    let mut v = self.h_decomp(qubits[1]);
                    v.extend(self.cx_decomp(qubits[0], qubits[1]));
                    v.extend(self.h_decomp(qubits[1]));
                    v
                }
            }
            OpType::S => vec![(OpType::Rz(Parameter::Const(PI2)), vec![qubits[0]])],
            OpType::Sdg => vec![(OpType::Rz(Parameter::Const(-PI2)), vec![qubits[0]])],
            OpType::T => vec![(OpType::Rz(Parameter::Const(PI / 4.0)), vec![qubits[0]])],
            OpType::Tdg => vec![(OpType::Rz(Parameter::Const(-PI / 4.0)), vec![qubits[0]])],
            OpType::SX => {
                if self.has("SX") {
                    vec![(OpType::SX, vec![qubits[0]])]
                } else {
                    vec![(OpType::Rx(Parameter::Const(PI2)), vec![qubits[0]])]
                }
            }
            OpType::SXdg => {
                if self.has("SX") {
                    let q = qubits[0];
                    vec![
                        (OpType::Rz(Parameter::Const(PI)), vec![q]),
                        (OpType::SX, vec![q]),
                        (OpType::Rz(Parameter::Const(PI)), vec![q]),
                    ]
                } else {
                    vec![(OpType::Rx(Parameter::Const(-PI2)), vec![qubits[0]])]
                }
            }
            OpType::Rx(p) => {
                let q = qubits[0];
                if self.has("RX") {
                    vec![(op.clone(), vec![q])]
                } else if let Some(theta) = p.try_evaluate() {
                    if self.has("SX") {
                        vec![
                            (OpType::Rz(Parameter::Const(-PI2)), vec![q]),
                            (OpType::SX, vec![q]),
                            (OpType::Rz(Parameter::Const(PI - theta)), vec![q]),
                            (OpType::SX, vec![q]),
                            (OpType::Rz(Parameter::Const(-PI2)), vec![q]),
                        ]
                    } else {
                        vec![(op.clone(), vec![q])]
                    }
                } else {
                    vec![(op.clone(), vec![q])]
                }
            }
            OpType::Ry(p) => {
                let q = qubits[0];
                if self.has("RY") {
                    vec![(op.clone(), vec![q])]
                } else if let Some(theta) = p.try_evaluate() {
                    if self.has("SX") {
                        vec![
                            (OpType::Rz(Parameter::Const(PI2)), vec![q]),
                            (OpType::SX, vec![q]),
                            (OpType::Rz(Parameter::Const(theta - PI)), vec![q]),
                            (OpType::SX, vec![q]),
                            (OpType::Rz(Parameter::Const(-3.0 * PI2)), vec![q]),
                        ]
                    } else {
                        vec![
                            (OpType::Rz(Parameter::Const(-PI2)), vec![q]),
                            (OpType::Rx(Parameter::Const(theta)), vec![q]),
                            (OpType::Rz(Parameter::Const(PI2)), vec![q]),
                        ]
                    }
                } else {
                    vec![(op.clone(), vec![q])]
                }
            }
            OpType::Rz(p) => {
                let q = qubits[0];
                if self.has("RZ") {
                    vec![(op.clone(), vec![q])]
                } else if let Some(theta) = p.try_evaluate() {
                    if self.has("P") {
                        vec![(OpType::P(Parameter::Const(theta)), vec![q])]
                    } else {
                        vec![(op.clone(), vec![q])]
                    }
                } else {
                    vec![(op.clone(), vec![q])]
                }
            }
            OpType::P(p) | OpType::R1(p) => {
                let q = qubits[0];
                vec![(OpType::Rz(p.clone()), vec![q])]
            }
            OpType::U(t, p, l) => {
                let q = qubits[0];
                if let (Some(theta), Some(phi), Some(lam)) =
                    (t.try_evaluate(), p.try_evaluate(), l.try_evaluate())
                {
                    if self.has("SX") {
                        vec![
                            (OpType::Rz(Parameter::Const(lam)), vec![q]),
                            (OpType::SX, vec![q]),
                            (OpType::Rz(Parameter::Const(theta + PI)), vec![q]),
                            (OpType::SX, vec![q]),
                            (OpType::Rz(Parameter::Const(phi + PI)), vec![q]),
                        ]
                    } else {
                        vec![
                            (OpType::Rz(Parameter::Const(lam)), vec![q]),
                            (OpType::Rx(Parameter::Const(PI2)), vec![q]),
                            (OpType::Rz(Parameter::Const(theta)), vec![q]),
                            (OpType::Rx(Parameter::Const(-PI2)), vec![q]),
                            (OpType::Rz(Parameter::Const(phi)), vec![q]),
                        ]
                    }
                } else {
                    vec![(op.clone(), vec![q])]
                }
            }
            OpType::SWAP => {
                let (q0, q1) = (qubits[0], qubits[1]);
                let mut v = self.cx_decomp(q0, q1);
                v.extend(self.cx_decomp(q1, q0));
                v.extend(self.cx_decomp(q0, q1));
                v
            }
            OpType::CCX => {
                let (q0, q1, q2) = (qubits[0], qubits[1], qubits[2]);
                let mut v = self.h_decomp(q2);
                v.extend(self.cx_decomp(q1, q2));
                v.push((OpType::Rz(Parameter::Const(-PI / 4.0)), vec![q2]));
                v.extend(self.cx_decomp(q0, q2));
                v.push((OpType::Rz(Parameter::Const(PI / 4.0)), vec![q2]));
                v.extend(self.cx_decomp(q1, q2));
                v.push((OpType::Rz(Parameter::Const(-PI / 4.0)), vec![q2]));
                v.extend(self.cx_decomp(q0, q2));
                v.push((OpType::Rz(Parameter::Const(PI / 4.0)), vec![q1]));
                v.push((OpType::Rz(Parameter::Const(PI / 4.0)), vec![q2]));
                v.extend(self.h_decomp(q2));
                v.extend(self.cx_decomp(q0, q1));
                v.push((OpType::Rz(Parameter::Const(PI / 4.0)), vec![q0]));
                v.push((OpType::Rz(Parameter::Const(-PI / 4.0)), vec![q1]));
                v.extend(self.cx_decomp(q0, q1));
                v
            }
            OpType::CY => {
                let (ctrl, tgt) = (qubits[0], qubits[1]);
                let mut v = vec![(OpType::Rz(Parameter::Const(-PI2)), vec![tgt])];
                v.extend(self.cx_decomp(ctrl, tgt));
                v.push((OpType::Rz(Parameter::Const(PI2)), vec![tgt]));
                v
            }
            OpType::CP(p) => {
                if let Some(lam) = p.try_evaluate() {
                    let (q0, q1) = (qubits[0], qubits[1]);
                    let mut v = vec![(OpType::Rz(Parameter::Const(lam / 2.0)), vec![q0])];
                    v.extend(self.cx_decomp(q0, q1));
                    v.push((OpType::Rz(Parameter::Const(-lam / 2.0)), vec![q1]));
                    v.extend(self.cx_decomp(q0, q1));
                    v.push((OpType::Rz(Parameter::Const(lam / 2.0)), vec![q1]));
                    v
                } else {
                    vec![(op.clone(), qubits.to_vec())]
                }
            }
            OpType::ISWAP => {
                let (q0, q1) = (qubits[0], qubits[1]);
                let mut v = self.cx_decomp(q0, q1);
                v.extend(self.h_decomp(q0));
                v.extend(self.cx_decomp(q1, q0));
                v.push((OpType::Rz(Parameter::Const(PI2)), vec![q0]));
                v.extend(self.cx_decomp(q1, q0));
                v.push((OpType::Rz(Parameter::Const(-PI2)), vec![q0]));
                v.extend(self.h_decomp(q0));
                v.extend(self.cx_decomp(q0, q1));
                v
            }
            OpType::ECR => {
                if self.has("ECR") {
                    vec![(OpType::ECR, vec![qubits[0], qubits[1]])]
                } else {
                    let (q0, q1) = (qubits[0], qubits[1]);
                    let mut v = vec![(OpType::SX, vec![q0])];
                    v.extend(self.cx_decomp(q0, q1));
                    v.push((OpType::X, vec![q0]));
                    v
                }
            }
            OpType::Rzz(p) => {
                if let Some(theta) = p.try_evaluate() {
                    let (q0, q1) = (qubits[0], qubits[1]);
                    let mut v = self.cx_decomp(q0, q1);
                    v.push((OpType::Rz(Parameter::Const(theta)), vec![q1]));
                    v.extend(self.cx_decomp(q0, q1));
                    v
                } else {
                    vec![(op.clone(), qubits.to_vec())]
                }
            }
            OpType::Rxx(p) => {
                if let Some(theta) = p.try_evaluate() {
                    let (q0, q1) = (qubits[0], qubits[1]);
                    let mut v = self.h_decomp(q0);
                    v.extend(self.h_decomp(q1));
                    v.extend(self.cx_decomp(q0, q1));
                    v.push((OpType::Rz(Parameter::Const(theta)), vec![q1]));
                    v.extend(self.cx_decomp(q0, q1));
                    v.extend(self.h_decomp(q0));
                    v.extend(self.h_decomp(q1));
                    v
                } else {
                    vec![(op.clone(), qubits.to_vec())]
                }
            }
            OpType::Ryy(p) => {
                if let Some(theta) = p.try_evaluate() {
                    let (q0, q1) = (qubits[0], qubits[1]);
                    let mut v = vec![
                        (OpType::Rx(Parameter::Const(PI2)), vec![q0]),
                        (OpType::Rx(Parameter::Const(PI2)), vec![q1]),
                    ];
                    v.extend(self.cx_decomp(q0, q1));
                    v.push((OpType::Rz(Parameter::Const(theta)), vec![q1]));
                    v.extend(self.cx_decomp(q0, q1));
                    v.push((OpType::Rx(Parameter::Const(-PI2)), vec![q0]));
                    v.push((OpType::Rx(Parameter::Const(-PI2)), vec![q1]));
                    v
                } else {
                    vec![(op.clone(), qubits.to_vec())]
                }
            }
            _ => vec![(op.clone(), qubits.to_vec())],
        }
    }
}

impl Pass for BasisTranslationPass {
    fn name(&self) -> &str {
        "BasisTranslationPass"
    }

    fn run(&self, dag: &mut QuantumDAG) -> Result<(), CompilerError> {
        let topo = dag.topological_order();
        let mut to_replace: Vec<(petgraph::prelude::NodeIndex, OpType, Vec<usize>)> = Vec::new();

        for node_id in topo {
            let op = &dag.graph()[node_id];
            if !self.is_native(&op.op_type) {
                to_replace.push((node_id, op.op_type.clone(), op.qubits.to_vec()));
            }
        }

        for (node_id, op_type, qubits) in to_replace {
            let decomposition = self.decompose(&op_type, &qubits);
            super::superconducting::replace_node_with_gates(dag, node_id, &decomposition);
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cz_basis() -> BasisTranslationPass {
        BasisTranslationPass::new(&["RZ".into(), "SX".into(), "X".into(), "CZ".into()])
    }

    fn cx_basis() -> BasisTranslationPass {
        BasisTranslationPass::new(&["RZ".into(), "SX".into(), "X".into(), "CX".into()])
    }

    #[test]
    fn test_cx_to_cz_basis() {
        let mut dag = QuantumDAG::new(2, 0);
        dag.add_op(OpType::CNOT, &[0, 1]);

        let pass = cz_basis();
        pass.run(&mut dag).unwrap();

        assert_eq!(dag.count_ops_of_type("CNOT"), 0);
        assert_eq!(dag.count_ops_of_type("CZ"), 1);
        assert!(dag.gate_count() > 1); // H decomps around CZ
    }

    #[test]
    fn test_h_to_rz_sx() {
        let mut dag = QuantumDAG::new(1, 0);
        dag.add_op(OpType::H, &[0]);

        let pass = cz_basis();
        pass.run(&mut dag).unwrap();

        assert_eq!(dag.count_ops_of_type("H"), 0);
        assert_eq!(dag.count_ops_of_type("SX"), 1);
        assert_eq!(dag.count_ops_of_type("Rz"), 2);
    }

    #[test]
    fn test_native_gates_untouched() {
        let mut dag = QuantumDAG::new(2, 0);
        dag.add_op(OpType::Rz(Parameter::Const(0.5)), &[0]);
        dag.add_op(OpType::SX, &[0]);
        dag.add_op(OpType::CZ, &[0, 1]);

        let pass = cz_basis();
        pass.run(&mut dag).unwrap();

        assert_eq!(dag.gate_count(), 3);
    }

    #[test]
    fn test_rzz_decomposition() {
        let mut dag = QuantumDAG::new(2, 0);
        dag.add_op(OpType::Rzz(Parameter::Const(0.5)), &[0, 1]);

        let pass = cx_basis();
        pass.run(&mut dag).unwrap();

        assert_eq!(dag.count_ops_of_type("Rzz"), 0);
        assert_eq!(dag.count_ops_of_type("CNOT"), 2);
    }

    #[test]
    fn test_swap_decomposition() {
        let mut dag = QuantumDAG::new(2, 0);
        dag.add_op(OpType::SWAP, &[0, 1]);

        let pass = cz_basis();
        pass.run(&mut dag).unwrap();

        assert_eq!(dag.count_ops_of_type("SWAP"), 0);
        assert_eq!(dag.count_ops_of_type("CZ"), 3);
    }

    #[test]
    fn test_ccx_decomposition() {
        let mut dag = QuantumDAG::new(3, 0);
        dag.add_op(OpType::CCX, &[0, 1, 2]);

        let pass = cx_basis();
        pass.run(&mut dag).unwrap();

        assert_eq!(dag.count_ops_of_type("CCX"), 0);
        assert!(dag.count_ops_of_type("CNOT") >= 6);
    }
}
