//! Adjoint differentiation for parameterised quantum circuits.
//!
//! Computes the full gradient d<O>/d(theta) in O(M * 2^n) time
//! (one forward pass + one backward pass) regardless of the number
//! of parameters N.  This is a 2N-fold speedup over parameter-shift.
//!
//! This implementation uses the memory-efficient variant: instead of
//! caching all M intermediate states (~M * 2^n memory), it stores
//! only the final state and recomputes intermediates on the fly
//! during the backward pass.  Total cost: 2M gate applications
//! (same as the caching variant) but O(2^n) memory instead of
//! O(M * 2^n).

use crate::dag::{QuantumDAG, QuantumOp};
use crate::ops::{OpType, Parameter};
use crate::state::MethodError;
use num_complex::Complex64;

/// A Pauli term in an observable: coefficient * pauli_string.
/// Pauli encoding per qubit: 0=I, 1=X, 2=Y, 3=Z.
#[derive(Clone, Debug)]
pub struct PauliTerm {
    pub paulis: Vec<u8>,
    pub coef: Complex64,
}

/// Result of adjoint differentiation: gradient indexed by parameter name.
pub struct AdjointGradResult {
    pub param_names: Vec<String>,
    pub gradients: Vec<f64>,
}

fn generator_info(op: &OpType) -> Option<(&'static [u8], f64)> {
    match op {
        OpType::Rx(_) => Some((&[1], 0.5)),
        OpType::Ry(_) => Some((&[2], 0.5)),
        OpType::Rz(_) => Some((&[3], 0.5)),
        OpType::Rzz(_) => Some((&[3, 3], 0.5)),
        OpType::Rxx(_) => Some((&[1, 1], 0.5)),
        OpType::Ryy(_) => Some((&[2, 2], 0.5)),
        _ => None,
    }
}

fn get_param_name(op: &OpType) -> Option<String> {
    match op {
        OpType::Rx(Parameter::Variable { name, .. })
        | OpType::Ry(Parameter::Variable { name, .. })
        | OpType::Rz(Parameter::Variable { name, .. })
        | OpType::Rzz(Parameter::Variable { name, .. })
        | OpType::Rxx(Parameter::Variable { name, .. })
        | OpType::Ryy(Parameter::Variable { name, .. }) => Some(name.clone()),
        _ => None,
    }
}

// ─── In-place gate application (no allocation) ─────────────────────────

fn apply_1q_inplace(
    state: &mut [Complex64],
    u: [[Complex64; 2]; 2],
    target: usize,
    n_qubits: usize,
) {
    let half = 1usize << target;
    let block = half * 2;
    let dim = 1usize << n_qubits;
    let n_blocks = dim / block;
    let u00 = u[0][0];
    let u01 = u[0][1];
    let u10 = u[1][0];
    let u11 = u[1][1];

    for b in 0..n_blocks {
        let off = b * block;
        for i in 0..half {
            let lo = state[off + i];
            let hi = state[off + half + i];
            state[off + i] = u00 * lo + u01 * hi;
            state[off + half + i] = u10 * lo + u11 * hi;
        }
    }
}

fn apply_2q_inplace(
    state: &mut [Complex64],
    gate: [[Complex64; 4]; 4],
    q1: usize,
    q2: usize,
    n_qubits: usize,
) {
    let dim = 1usize << n_qubits;
    let mq1 = 1usize << q1;
    let mq2 = 1usize << q2;
    for k in 0..(dim >> 2) {
        // Map k to an index with bits q1 and q2 cleared
        let mut i00 = k;
        let (lo, hi) = if q1 < q2 { (q1, q2) } else { (q2, q1) };
        // Insert zeros at bit positions lo and hi
        i00 = ((i00 >> lo) << (lo + 1)) | (i00 & ((1 << lo) - 1));
        i00 = ((i00 >> hi) << (hi + 1)) | (i00 & ((1 << hi) - 1));

        let i01 = i00 | mq2;
        let i10 = i00 | mq1;
        let i11 = i00 | mq1 | mq2;

        let s = [state[i00], state[i01], state[i10], state[i11]];
        state[i00] = gate[0][0] * s[0] + gate[0][1] * s[1] + gate[0][2] * s[2] + gate[0][3] * s[3];
        state[i01] = gate[1][0] * s[0] + gate[1][1] * s[1] + gate[1][2] * s[2] + gate[1][3] * s[3];
        state[i10] = gate[2][0] * s[0] + gate[2][1] * s[1] + gate[2][2] * s[2] + gate[2][3] * s[3];
        state[i11] = gate[3][0] * s[0] + gate[3][1] * s[1] + gate[3][2] * s[2] + gate[3][3] * s[3];
    }
}

fn apply_pauli_1q_inplace(state: &mut [Complex64], pauli: u8, qubit: usize, n_qubits: usize) {
    let dim = 1 << n_qubits;
    let mask = 1usize << qubit;
    match pauli {
        1 => {
            for i in 0..dim {
                if (i & mask) == 0 {
                    state.swap(i, i | mask);
                }
            }
        }
        2 => {
            let neg_i = Complex64::new(0.0, -1.0);
            let pos_i = Complex64::new(0.0, 1.0);
            for i in 0..dim {
                if (i & mask) == 0 {
                    let j = i | mask;
                    let a = state[i];
                    let b = state[j];
                    state[i] = neg_i * b;
                    state[j] = pos_i * a;
                }
            }
        }
        3 => {
            for i in 0..dim {
                if (i & mask) != 0 {
                    state[i] = -state[i];
                }
            }
        }
        _ => {}
    }
}

// ─── Gate operation wrapper ─────────────────────────────────────────────

struct GateOp {
    op_type: OpType,
    qubits: Vec<usize>,
}

impl GateOp {
    fn unitary_2x2(&self) -> [[Complex64; 2]; 2] {
        let m = self.op_type.to_matrix();
        [[m[(0, 0)], m[(0, 1)]], [m[(1, 0)], m[(1, 1)]]]
    }

    fn unitary_4x4(&self) -> [[Complex64; 4]; 4] {
        let m = self.op_type.to_matrix();
        [
            [m[(0, 0)], m[(0, 1)], m[(0, 2)], m[(0, 3)]],
            [m[(1, 0)], m[(1, 1)], m[(1, 2)], m[(1, 3)]],
            [m[(2, 0)], m[(2, 1)], m[(2, 2)], m[(2, 3)]],
            [m[(3, 0)], m[(3, 1)], m[(3, 2)], m[(3, 3)]],
        ]
    }

    fn dagger_2x2(&self) -> [[Complex64; 2]; 2] {
        let u = self.unitary_2x2();
        [
            [u[0][0].conj(), u[1][0].conj()],
            [u[0][1].conj(), u[1][1].conj()],
        ]
    }

    fn dagger_4x4(&self) -> [[Complex64; 4]; 4] {
        let u = self.unitary_4x4();
        [
            [
                u[0][0].conj(),
                u[1][0].conj(),
                u[2][0].conj(),
                u[3][0].conj(),
            ],
            [
                u[0][1].conj(),
                u[1][1].conj(),
                u[2][1].conj(),
                u[3][1].conj(),
            ],
            [
                u[0][2].conj(),
                u[1][2].conj(),
                u[2][2].conj(),
                u[3][2].conj(),
            ],
            [
                u[0][3].conj(),
                u[1][3].conj(),
                u[2][3].conj(),
                u[3][3].conj(),
            ],
        ]
    }
}

/// Compute the adjoint gradient for a parameterised circuit.
///
/// Memory-efficient variant: O(2^n) memory, 2M gate applications.
/// No intermediate state caching — recomputes from |0> during backward pass.
///
/// Errors with `MethodError` when `param_values` omits any parameter that
/// appears in the circuit (including variables nested in expressions).
pub fn adjoint_grad(
    dag: &QuantumDAG,
    observable_terms: &[PauliTerm],
    param_values: &std::collections::HashMap<String, f64>,
) -> Result<AdjointGradResult, MethodError> {
    let n_qubits = dag.n_qubits;
    let dim = 1usize << n_qubits;

    let bound_dag = dag.bind(param_values);

    let order = bound_dag.topological_order();
    let ops: Vec<GateOp> = order
        .iter()
        .filter_map(|&node_id| {
            let op = &bound_dag.graph()[node_id];
            if op.op_type.is_boundary()
                || op.op_type == OpType::Barrier
                || op.op_type.is_measurement()
            {
                return None;
            }
            Some(GateOp {
                op_type: op.op_type.clone(),
                qubits: op.qubits.to_vec(),
            })
        })
        .collect();

    // Any variable still symbolic after bind() has no value in
    // `param_values`; evaluating it would panic in ops.rs. Fail with a
    // catchable error instead. variable_names() also sees variables nested
    // in parameter expressions, which dag.parameter_names() does not.
    let mut missing: Vec<String> = ops
        .iter()
        .flat_map(|g| g.op_type.parameters())
        .flat_map(|p| p.variable_names())
        .collect();
    if !missing.is_empty() {
        missing.sort();
        missing.dedup();
        return Err(MethodError(format!(
            "no value provided for parameter(s): {}. Pass all parameter \
             values via param_values= (the dag itself stays unbound)",
            missing.join(", ")
        )));
    }

    let orig_order = dag.topological_order();
    let orig_ops: Vec<&QuantumOp> = orig_order
        .iter()
        .filter_map(|&node_id| {
            let op = &dag.graph()[node_id];
            if op.op_type.is_boundary()
                || op.op_type == OpType::Barrier
                || op.op_type.is_measurement()
            {
                return None;
            }
            Some(op)
        })
        .collect();

    let param_names: Vec<String> = dag.parameter_names();
    let param_idx: std::collections::HashMap<&str, usize> = param_names
        .iter()
        .enumerate()
        .map(|(i, n)| (n.as_str(), i))
        .collect();
    let n_params = param_names.len();

    // Pre-compute all gate matrices (avoid recomputing during backward pass)
    let fwd_matrices_1q: Vec<Option<[[Complex64; 2]; 2]>> = ops
        .iter()
        .map(|g| {
            if g.qubits.len() == 1 {
                Some(g.unitary_2x2())
            } else {
                None
            }
        })
        .collect();
    let fwd_matrices_2q: Vec<Option<[[Complex64; 4]; 4]>> = ops
        .iter()
        .map(|g| {
            if g.qubits.len() == 2 {
                Some(g.unitary_4x4())
            } else {
                None
            }
        })
        .collect();
    let dag_matrices_1q: Vec<Option<[[Complex64; 2]; 2]>> = ops
        .iter()
        .map(|g| {
            if g.qubits.len() == 1 {
                Some(g.dagger_2x2())
            } else {
                None
            }
        })
        .collect();
    let dag_matrices_2q: Vec<Option<[[Complex64; 4]; 4]>> = ops
        .iter()
        .map(|g| {
            if g.qubits.len() == 2 {
                Some(g.dagger_4x4())
            } else {
                None
            }
        })
        .collect();

    // Forward pass: evolve |0> → |psi_final> (in-place, no caching)
    let mut psi = vec![Complex64::new(0.0, 0.0); dim];
    psi[0] = Complex64::new(1.0, 0.0);

    for (k, gate_op) in ops.iter().enumerate() {
        match gate_op.qubits.len() {
            1 => apply_1q_inplace(
                &mut psi,
                fwd_matrices_1q[k].unwrap(),
                gate_op.qubits[0],
                n_qubits,
            ),
            2 => apply_2q_inplace(
                &mut psi,
                fwd_matrices_2q[k].unwrap(),
                gate_op.qubits[0],
                gate_op.qubits[1],
                n_qubits,
            ),
            _ => {}
        }
    }

    // Build phi = O|psi_final>
    let mut phi = vec![Complex64::new(0.0, 0.0); dim];
    for term in observable_terms {
        let mut term_state = psi.clone();
        for (q, &p) in term.paulis.iter().enumerate() {
            if p != 0 {
                apply_pauli_1q_inplace(&mut term_state, p, q, n_qubits);
            }
        }
        for i in 0..dim {
            phi[i] += term.coef * term_state[i];
        }
    }

    // Backward pass
    // psi currently holds the final state.
    // For each gate k from M-1 down to 0:
    //   1. Un-apply gate k from psi: psi <- U_k^dag * psi  (gives psi_{k})
    //   2. Compute gradient contribution using psi_{k} and phi
    //   3. Un-apply gate k from phi: phi <- U_k^dag * phi
    let mut grad = vec![0.0f64; n_params];

    for k in (0..ops.len()).rev() {
        let gate_op = &ops[k];
        let orig_op = &orig_ops[k];

        // Un-apply gate k from psi to get psi_k (state before gate k)
        match gate_op.qubits.len() {
            1 => apply_1q_inplace(
                &mut psi,
                dag_matrices_1q[k].unwrap(),
                gate_op.qubits[0],
                n_qubits,
            ),
            2 => apply_2q_inplace(
                &mut psi,
                dag_matrices_2q[k].unwrap(),
                gate_op.qubits[0],
                gate_op.qubits[1],
                n_qubits,
            ),
            _ => {}
        }
        // psi now holds psi_k (state BEFORE gate k was applied)

        // Gradient contribution
        let gen_info = generator_info(&orig_op.original_op_type);
        let param_name = get_param_name(&orig_op.original_op_type);

        if let (Some(gen_info), Some(param_name)) = (gen_info, param_name) {
            if let Some(&idx) = param_idx.get(param_name.as_str()) {
                let (gen_paulis, alpha) = gen_info;

                // We need <phi | G_k | psi_after_k>.
                // psi currently is psi_k (before gate k).
                // psi_after_k = U_k * psi_k.
                // G_k | psi_after_k> = G_k U_k | psi_k>
                //
                // For efficiency: apply U_k to a copy of psi_k, then apply G_k.
                let mut g_psi = psi.clone();
                // Re-apply gate k to get psi_after_k
                match gate_op.qubits.len() {
                    1 => apply_1q_inplace(
                        &mut g_psi,
                        fwd_matrices_1q[k].unwrap(),
                        gate_op.qubits[0],
                        n_qubits,
                    ),
                    2 => apply_2q_inplace(
                        &mut g_psi,
                        fwd_matrices_2q[k].unwrap(),
                        gate_op.qubits[0],
                        gate_op.qubits[1],
                        n_qubits,
                    ),
                    _ => {}
                }
                // Apply generator
                for (q_local, &pauli_id) in gen_paulis.iter().enumerate() {
                    let qubit = gate_op.qubits[q_local];
                    apply_pauli_1q_inplace(&mut g_psi, pauli_id, qubit, n_qubits);
                }

                let ip: Complex64 = phi
                    .iter()
                    .zip(g_psi.iter())
                    .map(|(a, b)| a.conj() * b)
                    .sum();
                grad[idx] += 2.0 * alpha * ip.im;
            }
        }

        // Back-walk phi
        match gate_op.qubits.len() {
            1 => apply_1q_inplace(
                &mut phi,
                dag_matrices_1q[k].unwrap(),
                gate_op.qubits[0],
                n_qubits,
            ),
            2 => apply_2q_inplace(
                &mut phi,
                dag_matrices_2q[k].unwrap(),
                gate_op.qubits[0],
                gate_op.qubits[1],
                n_qubits,
            ),
            _ => {}
        }
    }

    Ok(AdjointGradResult {
        param_names,
        gradients: grad,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dag::QuantumDAG;
    use crate::ops::{OpType, Parameter};

    #[test]
    fn test_rx_gradient() {
        let theta = 0.7;
        let mut dag = QuantumDAG::new(1, 0);
        dag.add_op(
            OpType::Rx(Parameter::Variable {
                name: "theta".into(),
                id: 0,
            }),
            &[0],
        );
        let obs = vec![PauliTerm {
            paulis: vec![3],
            coef: Complex64::new(1.0, 0.0),
        }];
        let mut params = std::collections::HashMap::new();
        params.insert("theta".into(), theta);
        let result = adjoint_grad(&dag, &obs, &params).unwrap();
        let expected = -theta.sin();
        assert!(
            (result.gradients[0] - expected).abs() < 1e-10,
            "Got {}, expected {}",
            result.gradients[0],
            expected
        );
    }

    #[test]
    fn test_ry_gradient() {
        let theta = 1.2;
        let mut dag = QuantumDAG::new(1, 0);
        dag.add_op(
            OpType::Ry(Parameter::Variable {
                name: "theta".into(),
                id: 0,
            }),
            &[0],
        );
        let obs = vec![PauliTerm {
            paulis: vec![3],
            coef: Complex64::new(1.0, 0.0),
        }];
        let mut params = std::collections::HashMap::new();
        params.insert("theta".into(), theta);
        let result = adjoint_grad(&dag, &obs, &params).unwrap();
        let expected = -theta.sin();
        assert!((result.gradients[0] - expected).abs() < 1e-10);
    }

    #[test]
    fn test_rz_gradient_x_observable() {
        let theta = 0.6;
        let mut dag = QuantumDAG::new(1, 0);
        dag.add_op(OpType::H, &[0]);
        dag.add_op(
            OpType::Rz(Parameter::Variable {
                name: "theta".into(),
                id: 0,
            }),
            &[0],
        );
        let obs = vec![PauliTerm {
            paulis: vec![1],
            coef: Complex64::new(1.0, 0.0),
        }];
        let mut params = std::collections::HashMap::new();
        params.insert("theta".into(), theta);
        let result = adjoint_grad(&dag, &obs, &params).unwrap();
        let expected = -theta.sin();
        assert!((result.gradients[0] - expected).abs() < 1e-10);
    }

    #[test]
    fn test_multi_param_gradient() {
        let t0 = 0.5;
        let t1 = 1.0;
        let mut dag = QuantumDAG::new(1, 0);
        dag.add_op(
            OpType::Rx(Parameter::Variable {
                name: "t0".into(),
                id: 0,
            }),
            &[0],
        );
        dag.add_op(
            OpType::Ry(Parameter::Variable {
                name: "t1".into(),
                id: 1,
            }),
            &[0],
        );
        let obs = vec![PauliTerm {
            paulis: vec![3],
            coef: Complex64::new(1.0, 0.0),
        }];
        let mut params = std::collections::HashMap::new();
        params.insert("t0".into(), t0);
        params.insert("t1".into(), t1);
        let result = adjoint_grad(&dag, &obs, &params).unwrap();
        assert_eq!(result.param_names.len(), 2);
        assert!(result.gradients[0].is_finite());
        assert!(result.gradients[1].is_finite());
    }

    #[test]
    fn test_two_qubit_rzz_gradient() {
        let theta = 0.4;
        let mut dag = QuantumDAG::new(2, 0);
        dag.add_op(OpType::H, &[0]);
        dag.add_op(OpType::H, &[1]);
        dag.add_op(
            OpType::Rzz(Parameter::Variable {
                name: "theta".into(),
                id: 0,
            }),
            &[0, 1],
        );
        let obs = vec![PauliTerm {
            paulis: vec![1, 0],
            coef: Complex64::new(1.0, 0.0),
        }];
        let mut params = std::collections::HashMap::new();
        params.insert("theta".into(), theta);
        let result = adjoint_grad(&dag, &obs, &params).unwrap();
        let expected = -(theta).sin();
        assert!((result.gradients[0] - expected).abs() < 1e-8);
    }

    #[test]
    fn test_multi_observable_terms() {
        let theta = 0.6;
        let mut dag = QuantumDAG::new(1, 0);
        dag.add_op(
            OpType::Rx(Parameter::Variable {
                name: "theta".into(),
                id: 0,
            }),
            &[0],
        );
        let obs = vec![
            PauliTerm {
                paulis: vec![3],
                coef: Complex64::new(0.5, 0.0),
            },
            PauliTerm {
                paulis: vec![1],
                coef: Complex64::new(0.3, 0.0),
            },
        ];
        let mut params = std::collections::HashMap::new();
        params.insert("theta".into(), theta);
        let result = adjoint_grad(&dag, &obs, &params).unwrap();
        let expected = -0.5 * theta.sin();
        assert!((result.gradients[0] - expected).abs() < 1e-10);
    }

    #[test]
    fn test_same_param_multiple_gates() {
        let theta = 0.7;
        let mut dag = QuantumDAG::new(2, 0);
        dag.add_op(
            OpType::Rx(Parameter::Variable {
                name: "theta".into(),
                id: 0,
            }),
            &[0],
        );
        dag.add_op(
            OpType::Rx(Parameter::Variable {
                name: "theta".into(),
                id: 0,
            }),
            &[1],
        );
        let obs = vec![PauliTerm {
            paulis: vec![3, 0],
            coef: Complex64::new(1.0, 0.0),
        }];
        let mut params = std::collections::HashMap::new();
        params.insert("theta".into(), theta);
        let result = adjoint_grad(&dag, &obs, &params).unwrap();
        let expected = -theta.sin();
        assert!((result.gradients[0] - expected).abs() < 1e-10);
    }
}
