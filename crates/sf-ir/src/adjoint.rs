//! Adjoint differentiation for parameterised quantum circuits.
//!
//! Computes the full gradient d<O>/d(theta) in O(M * 2^n) time
//! (one forward pass + one backward pass) regardless of the number
//! of parameters N.  This is a 2N-fold speedup over parameter-shift.
//!
//! Algorithm (Jones & Gacon 2020):
//!   1. Forward: evolve |psi> = U(theta)|0>, caching intermediates.
//!   2. Build |phi> = O|psi> (apply observable as sum of Pauli terms).
//!   3. Walk gates in reverse:
//!      - d<O>/dtheta_k = 2*alpha * Im(<phi|G_k|psi_{k-1}>)
//!      - phi <- U_k^dag |phi>

use crate::dag::{QuantumDAG, QuantumOp};
use crate::ops::{OpType, Parameter};
use num_complex::Complex64;
use rayon::prelude::*;

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

/// Which Pauli generator a rotation gate has, and the prefactor alpha
/// such that U(theta) = exp(-i * alpha * theta * G).
fn generator_info(op: &OpType) -> Option<(&'static [u8], f64)> {
    match op {
        OpType::Rx(_) => Some((&[1], 0.5)),   // X generator
        OpType::Ry(_) => Some((&[2], 0.5)),   // Y generator
        OpType::Rz(_) => Some((&[3], 0.5)),   // Z generator
        OpType::Rzz(_) => Some((&[3, 3], 0.5)), // ZZ generator
        OpType::Rxx(_) => Some((&[1, 1], 0.5)), // XX generator
        OpType::Ryy(_) => Some((&[2, 2], 0.5)), // YY generator
        _ => None,
    }
}

/// Get the parameter name from an OpType if it has a variable parameter.
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

/// Apply a 1-qubit gate matrix to a statevector (in-place src→dst ping-pong style).
fn apply_1q_into(
    src: &[Complex64],
    dst: &mut [Complex64],
    u: [[Complex64; 2]; 2],
    target: usize,
    n_qubits: usize,
) {
    let dim = 1 << n_qubits;
    let half = 1usize << target;
    let block = half * 2;
    let n_blocks = dim / block;

    let u00 = u[0][0]; let u01 = u[0][1];
    let u10 = u[1][0]; let u11 = u[1][1];

    if n_blocks >= 4 {
        dst.par_chunks_mut(block).enumerate().for_each(|(b, dst_chunk)| {
            let off = b * block;
            let (lo_dst, hi_dst) = dst_chunk.split_at_mut(half);
            let lo_src = &src[off..off + half];
            let hi_src = &src[off + half..off + block];
            for i in 0..half {
                lo_dst[i] = u00 * lo_src[i] + u01 * hi_src[i];
                hi_dst[i] = u10 * lo_src[i] + u11 * hi_src[i];
            }
        });
    } else {
        for b in 0..n_blocks {
            let off = b * block;
            for i in 0..half {
                let lo = src[off + i];
                let hi = src[off + half + i];
                dst[off + i] = u00 * lo + u01 * hi;
                dst[off + half + i] = u10 * lo + u11 * hi;
            }
        }
    }
}

/// Apply a 2-qubit gate to a statevector (src→dst).
fn apply_2q_into(
    src: &[Complex64],
    dst: &mut [Complex64],
    gate: &[[Complex64; 4]; 4],
    q1: usize,
    q2: usize,
    n_qubits: usize,
) {
    let dim = 1 << n_qubits;
    let mq1 = 1usize << q1;
    let mq2 = 1usize << q2;
    let chunk_size = (dim / 16).max(1024);

    dst.par_chunks_mut(chunk_size).enumerate().for_each(|(c, dst_chunk)| {
        let off = c * chunk_size;
        for (k, target) in dst_chunk.iter_mut().enumerate() {
            let i = off + k;
            let bit1 = (i >> q1) & 1;
            let bit2 = (i >> q2) & 1;
            let i00 = i & !mq1 & !mq2;
            let i01 = i00 | mq2;
            let i10 = i00 | mq1;
            let i11 = i00 | mq1 | mq2;
            let row = bit1 * 2 + bit2;
            *target = gate[row][0] * src[i00]
                    + gate[row][1] * src[i01]
                    + gate[row][2] * src[i10]
                    + gate[row][3] * src[i11];
        }
    });
}

/// Apply a Pauli operator (single qubit) in-place on the statevector.
fn apply_pauli_1q_inplace(state: &mut [Complex64], pauli: u8, qubit: usize, n_qubits: usize) {
    let dim = 1 << n_qubits;
    let mask = 1usize << qubit;
    match pauli {
        1 => { // X: swap amplitudes
            for i in 0..dim {
                if (i & mask) == 0 {
                    let j = i | mask;
                    state.swap(i, j);
                }
            }
        }
        2 => { // Y: swap with phases
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
        3 => { // Z: phase flip
            for i in 0..dim {
                if (i & mask) != 0 {
                    state[i] = -state[i];
                }
            }
        }
        _ => {} // I: no-op
    }
}

/// Gate operation data needed for adjoint traversal (flattened from DAG).
struct GateOp {
    op_type: OpType,
    qubits: Vec<usize>,
}

impl GateOp {
    fn unitary_2x2(&self) -> [[Complex64; 2]; 2] {
        let m = self.op_type.to_matrix();
        [
            [m[(0, 0)], m[(0, 1)]],
            [m[(1, 0)], m[(1, 1)]],
        ]
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
            [u[0][0].conj(), u[1][0].conj(), u[2][0].conj(), u[3][0].conj()],
            [u[0][1].conj(), u[1][1].conj(), u[2][1].conj(), u[3][1].conj()],
            [u[0][2].conj(), u[1][2].conj(), u[2][2].conj(), u[3][2].conj()],
            [u[0][3].conj(), u[1][3].conj(), u[2][3].conj(), u[3][3].conj()],
        ]
    }

    /// Apply this gate forward: src → dst
    fn apply_forward(&self, src: &[Complex64], dst: &mut [Complex64], n_qubits: usize) {
        match self.qubits.len() {
            1 => apply_1q_into(src, dst, self.unitary_2x2(), self.qubits[0], n_qubits),
            2 => apply_2q_into(src, dst, &self.unitary_4x4(), self.qubits[0], self.qubits[1], n_qubits),
            _ => dst.copy_from_slice(src), // 3q gates not differentiable
        }
    }

    /// Apply U^dag: src → dst
    fn apply_dagger(&self, src: &[Complex64], dst: &mut [Complex64], n_qubits: usize) {
        match self.qubits.len() {
            1 => apply_1q_into(src, dst, self.dagger_2x2(), self.qubits[0], n_qubits),
            2 => apply_2q_into(src, dst, &self.dagger_4x4(), self.qubits[0], self.qubits[1], n_qubits),
            _ => dst.copy_from_slice(src),
        }
    }
}

/// Compute the adjoint gradient for a parameterised circuit.
///
/// `observable_terms`: list of (pauli_per_qubit, coefficient) for the observable.
/// `param_values`: map from parameter name to its current value (circuit must be bound).
///
/// Returns a map from parameter name to gradient value.
pub fn adjoint_grad(
    dag: &QuantumDAG,
    observable_terms: &[PauliTerm],
    param_values: &std::collections::HashMap<String, f64>,
) -> AdjointGradResult {
    let n_qubits = dag.n_qubits;
    let dim = 1usize << n_qubits;

    // Bind the DAG parameters for forward simulation
    let bound_dag = dag.bind(param_values);

    // Extract gate sequence in topological order
    let order = bound_dag.topological_order();
    let ops: Vec<GateOp> = order.iter().filter_map(|&node_id| {
        let op = &bound_dag.graph()[node_id];
        if op.op_type.is_boundary() || op.op_type == OpType::Barrier || op.op_type.is_measurement() {
            return None;
        }
        Some(GateOp {
            op_type: op.op_type.clone(),
            qubits: op.qubits.to_vec(),
        })
    }).collect();

    // Also extract the original (unbound) ops to identify which parameters
    let orig_order = dag.topological_order();
    let orig_ops: Vec<&QuantumOp> = orig_order.iter().filter_map(|&node_id| {
        let op = &dag.graph()[node_id];
        if op.op_type.is_boundary() || op.op_type == OpType::Barrier || op.op_type.is_measurement() {
            return None;
        }
        Some(op)
    }).collect();

    // Build parameter name→index map
    let param_names: Vec<String> = dag.parameter_names();
    let param_idx: std::collections::HashMap<&str, usize> = param_names.iter()
        .enumerate()
        .map(|(i, n)| (n.as_str(), i))
        .collect();
    let n_params = param_names.len();

    // Forward pass: evolve |0> through all gates, caching post-gate states
    let mut buf_a = vec![Complex64::new(0.0, 0.0); dim];
    let mut buf_b = vec![Complex64::new(0.0, 0.0); dim];
    buf_a[0] = Complex64::new(1.0, 0.0);
    let mut current_is_a = true;

    let mut intermediates: Vec<Vec<Complex64>> = Vec::with_capacity(ops.len() + 1);
    intermediates.push(buf_a.clone());

    for gate_op in &ops {
        if current_is_a {
            gate_op.apply_forward(&buf_a, &mut buf_b, n_qubits);
        } else {
            gate_op.apply_forward(&buf_b, &mut buf_a, n_qubits);
        }
        current_is_a = !current_is_a;
        let state_ref = if current_is_a { &buf_a } else { &buf_b };
        intermediates.push(state_ref.to_vec());
    }

    let psi_final = if current_is_a { &buf_a } else { &buf_b };

    // Build phi = O|psi_final> = sum_k coef_k * P_k |psi_final>
    let mut phi = vec![Complex64::new(0.0, 0.0); dim];
    for term in observable_terms {
        let mut term_state = psi_final.to_vec();
        for (q, &p) in term.paulis.iter().enumerate() {
            if p != 0 {
                apply_pauli_1q_inplace(&mut term_state, p, q, n_qubits);
            }
        }
        for i in 0..dim {
            phi[i] += term.coef * term_state[i];
        }
    }

    // Backward pass: walk gates in reverse
    let mut grad = vec![0.0f64; n_params];
    let mut phi_buf = vec![Complex64::new(0.0, 0.0); dim];

    for k in (0..ops.len()).rev() {
        let gate_op = &ops[k];
        let orig_op = &orig_ops[k];

        let gen_info = generator_info(&orig_op.original_op_type);
        let param_name = get_param_name(&orig_op.original_op_type);

        if let Some(gen_info) = gen_info {
            if let Some(param_name) = param_name {
                if let Some(&idx) = param_idx.get(param_name.as_str()) {
                    let (gen_paulis, alpha) = gen_info;
                    let psi_after = &intermediates[k + 1];

                    let mut g_psi = psi_after.clone();
                    for (q_local, &pauli_id) in gen_paulis.iter().enumerate() {
                        let qubit = gate_op.qubits[q_local];
                        apply_pauli_1q_inplace(&mut g_psi, pauli_id, qubit, n_qubits);
                    }

                    let ip: Complex64 = phi.iter().zip(g_psi.iter())
                        .map(|(a, b)| a.conj() * b)
                        .sum();
                    grad[idx] += 2.0 * alpha * ip.im;
                }
            }
        }

        // Back-walk phi: phi <- U_k^dag * phi
        gate_op.apply_dagger(&phi, &mut phi_buf, n_qubits);
        std::mem::swap(&mut phi, &mut phi_buf);
    }

    AdjointGradResult {
        param_names,
        gradients: grad,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dag::QuantumDAG;
    use crate::ops::{OpType, Parameter};

    #[test]
    fn test_rx_gradient() {
        // d/dtheta <0|Rx(theta)^dag Z Rx(theta)|0> = -sin(theta)
        let theta = 0.7;
        let mut dag = QuantumDAG::new(1, 0);
        dag.add_op(OpType::Rx(Parameter::Variable { name: "theta".into(), id: 0 }), &[0]);

        let obs = vec![PauliTerm {
            paulis: vec![3], // Z
            coef: Complex64::new(1.0, 0.0),
        }];

        let mut params = std::collections::HashMap::new();
        params.insert("theta".into(), theta);

        let result = adjoint_grad(&dag, &obs, &params);
        let expected = -theta.sin();
        assert!((result.gradients[0] - expected).abs() < 1e-10,
            "Got {}, expected {}", result.gradients[0], expected);
    }

    #[test]
    fn test_ry_gradient() {
        let theta = 1.2;
        let mut dag = QuantumDAG::new(1, 0);
        dag.add_op(OpType::Ry(Parameter::Variable { name: "theta".into(), id: 0 }), &[0]);

        let obs = vec![PauliTerm {
            paulis: vec![3], // Z
            coef: Complex64::new(1.0, 0.0),
        }];

        let mut params = std::collections::HashMap::new();
        params.insert("theta".into(), theta);

        let result = adjoint_grad(&dag, &obs, &params);
        // <0|Ry(t)^dag Z Ry(t)|0> = cos(t), d/dt = -sin(t)
        let expected = -theta.sin();
        assert!((result.gradients[0] - expected).abs() < 1e-10,
            "Got {}, expected {}", result.gradients[0], expected);
    }

    #[test]
    fn test_rz_gradient() {
        // <0|Rz(t)^dag Z Rz(t)|0> = 1 (Z is diagonal, Rz doesn't change it)
        // So gradient should be 0
        let theta = 0.9;
        let mut dag = QuantumDAG::new(1, 0);
        dag.add_op(OpType::Rz(Parameter::Variable { name: "theta".into(), id: 0 }), &[0]);

        let obs = vec![PauliTerm {
            paulis: vec![3], // Z
            coef: Complex64::new(1.0, 0.0),
        }];

        let mut params = std::collections::HashMap::new();
        params.insert("theta".into(), theta);

        let result = adjoint_grad(&dag, &obs, &params);
        assert!(result.gradients[0].abs() < 1e-10,
            "Rz gradient wrt Z should be 0, got {}", result.gradients[0]);
    }

    #[test]
    fn test_rz_gradient_x_observable() {
        // <0|Rz(t)^dag X Rz(t)|0> = cos(t) from Bloch sphere rotation
        // But |0> is eigenstate of Z, so Rz|0> = e^{-it/2}|0>, and
        // <0|X|0> = 0 regardless of phase. Gradient = 0.
        // Instead: start with H|0> = |+>, then Rz rotates in X-Y plane.
        let theta = 0.6;
        let mut dag = QuantumDAG::new(1, 0);
        dag.add_op(OpType::H, &[0]);
        dag.add_op(OpType::Rz(Parameter::Variable { name: "theta".into(), id: 0 }), &[0]);

        let obs = vec![PauliTerm {
            paulis: vec![1], // X
            coef: Complex64::new(1.0, 0.0),
        }];

        let mut params = std::collections::HashMap::new();
        params.insert("theta".into(), theta);

        let result = adjoint_grad(&dag, &obs, &params);
        // H|0> = |+>. Rz(t)|+> rotates in XY plane.
        // <+|Rz^dag X Rz|+> = cos(t), d/dt = -sin(t)
        let expected = -theta.sin();
        assert!((result.gradients[0] - expected).abs() < 1e-10,
            "Got {}, expected {}", result.gradients[0], expected);
    }

    #[test]
    fn test_multi_param_gradient() {
        let t0 = 0.5;
        let t1 = 1.0;
        let mut dag = QuantumDAG::new(1, 0);
        dag.add_op(OpType::Rx(Parameter::Variable { name: "t0".into(), id: 0 }), &[0]);
        dag.add_op(OpType::Ry(Parameter::Variable { name: "t1".into(), id: 1 }), &[0]);

        let obs = vec![PauliTerm {
            paulis: vec![3], // Z
            coef: Complex64::new(1.0, 0.0),
        }];

        let mut params = std::collections::HashMap::new();
        params.insert("t0".into(), t0);
        params.insert("t1".into(), t1);

        let result = adjoint_grad(&dag, &obs, &params);
        assert_eq!(result.param_names.len(), 2);
        assert_eq!(result.gradients.len(), 2);
        // Verify both gradients are finite
        assert!(result.gradients[0].is_finite());
        assert!(result.gradients[1].is_finite());
    }

    #[test]
    fn test_two_qubit_rzz_gradient() {
        // RZZ(t) on 2 qubits with X⊗I observable (doesn't commute with ZZ)
        // |++> → RZZ(t)|++>, observable X⊗I
        // <++|RZZ^dag (X⊗I) RZZ|++> = cos(t), d/dt = -sin(t)
        let theta = 0.4;
        let mut dag = QuantumDAG::new(2, 0);
        dag.add_op(OpType::H, &[0]);
        dag.add_op(OpType::H, &[1]);
        dag.add_op(OpType::Rzz(Parameter::Variable { name: "theta".into(), id: 0 }), &[0, 1]);

        let obs = vec![PauliTerm {
            paulis: vec![1, 0], // X on q0, I on q1
            coef: Complex64::new(1.0, 0.0),
        }];

        let mut params = std::collections::HashMap::new();
        params.insert("theta".into(), theta);

        let result = adjoint_grad(&dag, &obs, &params);
        let expected = -(theta).sin();
        assert!((result.gradients[0] - expected).abs() < 1e-8,
            "RZZ grad: got {}, expected {}", result.gradients[0], expected);
    }

    #[test]
    fn test_gradient_at_zero() {
        // At theta=0, Rx(0)=I so gradient of <Z> = d/dt cos(t)|_{t=0} = 0... no:
        // <0|Rx(t)^dag Z Rx(t)|0> = cos(t), d/dt|_{t=0} = -sin(0) = 0
        let mut dag = QuantumDAG::new(1, 0);
        dag.add_op(OpType::Rx(Parameter::Variable { name: "theta".into(), id: 0 }), &[0]);

        let obs = vec![PauliTerm {
            paulis: vec![3],
            coef: Complex64::new(1.0, 0.0),
        }];

        let mut params = std::collections::HashMap::new();
        params.insert("theta".into(), 0.0);

        let result = adjoint_grad(&dag, &obs, &params);
        assert!(result.gradients[0].abs() < 1e-10,
            "Gradient at theta=0 should be 0, got {}", result.gradients[0]);
    }

    #[test]
    fn test_gradient_at_pi() {
        // At theta=pi, d/dt cos(t)|_{t=pi} = -sin(pi) = 0
        let mut dag = QuantumDAG::new(1, 0);
        dag.add_op(OpType::Rx(Parameter::Variable { name: "theta".into(), id: 0 }), &[0]);

        let obs = vec![PauliTerm {
            paulis: vec![3],
            coef: Complex64::new(1.0, 0.0),
        }];

        let mut params = std::collections::HashMap::new();
        params.insert("theta".into(), std::f64::consts::PI);

        let result = adjoint_grad(&dag, &obs, &params);
        assert!(result.gradients[0].abs() < 1e-10,
            "Gradient at theta=pi should be 0, got {}", result.gradients[0]);
    }

    #[test]
    fn test_non_parametric_gates_ignored() {
        // Circuit with H (non-parametric) + Rx (parametric)
        // Gradient should only apply to Rx
        let theta = 0.8;
        let mut dag = QuantumDAG::new(1, 0);
        dag.add_op(OpType::H, &[0]);
        dag.add_op(OpType::Rx(Parameter::Variable { name: "theta".into(), id: 0 }), &[0]);
        dag.add_op(OpType::H, &[0]);

        let obs = vec![PauliTerm {
            paulis: vec![3],
            coef: Complex64::new(1.0, 0.0),
        }];

        let mut params = std::collections::HashMap::new();
        params.insert("theta".into(), theta);

        let result = adjoint_grad(&dag, &obs, &params);
        assert_eq!(result.param_names.len(), 1);
        assert!(result.gradients[0].is_finite());
    }

    #[test]
    fn test_multi_observable_terms() {
        // Observable = 0.5*Z + 0.3*X
        let theta = 0.6;
        let mut dag = QuantumDAG::new(1, 0);
        dag.add_op(OpType::Rx(Parameter::Variable { name: "theta".into(), id: 0 }), &[0]);

        let obs = vec![
            PauliTerm { paulis: vec![3], coef: Complex64::new(0.5, 0.0) },
            PauliTerm { paulis: vec![1], coef: Complex64::new(0.3, 0.0) },
        ];

        let mut params = std::collections::HashMap::new();
        params.insert("theta".into(), theta);

        let result = adjoint_grad(&dag, &obs, &params);
        // Gradient of 0.5*<Z> + 0.3*<X>:
        // <0|Rx^dag Z Rx|0> = cos(t), d = -0.5*sin(t)
        // <0|Rx^dag X Rx|0> = 0 (stays 0), d = 0
        let expected = -0.5 * theta.sin();
        assert!((result.gradients[0] - expected).abs() < 1e-10,
            "Got {}, expected {}", result.gradients[0], expected);
    }

    #[test]
    fn test_same_param_multiple_gates() {
        // Same parameter used in two gates: Rx(t) on q0, Rx(t) on q1
        let theta = 0.7;
        let mut dag = QuantumDAG::new(2, 0);
        dag.add_op(OpType::Rx(Parameter::Variable { name: "theta".into(), id: 0 }), &[0]);
        dag.add_op(OpType::Rx(Parameter::Variable { name: "theta".into(), id: 0 }), &[1]);

        let obs = vec![PauliTerm {
            paulis: vec![3, 0], // Z on q0, I on q1
            coef: Complex64::new(1.0, 0.0),
        }];

        let mut params = std::collections::HashMap::new();
        params.insert("theta".into(), theta);

        let result = adjoint_grad(&dag, &obs, &params);
        // Only the first Rx on q0 contributes to d<Z_0>/dt = -sin(t)
        // The second Rx on q1 doesn't affect Z_0
        let expected = -theta.sin();
        assert!((result.gradients[0] - expected).abs() < 1e-10,
            "Got {}, expected {}", result.gradients[0], expected);
    }
}
