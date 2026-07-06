use num_complex::Complex64;
use nalgebra::DMatrix;
use rayon::prelude::*;
pub struct DensityMatrixState {
    pub data: Vec<Complex64>,
    pub n_qubits: usize,
}

impl DensityMatrixState {
    pub fn new(n_qubits: usize) -> Self {
        let dim = 1 << (2 * n_qubits);
        let mut data = vec![Complex64::new(0.0, 0.0); dim];
        // Initial state |0...0><0...0| is index 0 in the vectorized representation
        data[0] = Complex64::new(1.0, 0.0);
        Self { data, n_qubits }
    }

    /// Apply a unitary gate U to the density matrix: rho -> U rho U†
    pub fn apply_unitary(&mut self, u: &DMatrix<Complex64>, qubits: &[usize]) {
        match qubits.len() {
            1 => {
                let q = qubits[0];
                let n = self.n_qubits;
                // rho -> (U \otimes I) rho (U† \otimes I)
                // In vectorized form: |rho>> -> (U \otimes U*) |rho>>
                // This means apply U to ket-index q and U* to bra-index n+q
                self.apply_1q_gate(u, q);
                let u_star = u.map(|c| c.conj());
                self.apply_1q_gate(&u_star, n + q);
            }
            2 => {
                let q0 = qubits[0];
                let q1 = qubits[1];
                let n = self.n_qubits;
                self.apply_2q_gate(u, q0, q1);
                let u_star = u.map(|c| c.conj());
                self.apply_2q_gate(&u_star, n + q0, n + q1);
            }
            _ => panic!("DensityMatrixState::apply_unitary only supports 1q and 2q gates"),
        }
    }

    /// Internal: apply a gate to the vectorized statevector representation
    fn apply_1q_gate(&mut self, u: &DMatrix<Complex64>, q: usize) {
        let n_total = 2 * self.n_qubits;
        let dim = 1 << n_total;
        let dist = 1 << q;
        
        let u00r = u[(0, 0)].re; let u00i = u[(0, 0)].im;
        let u01r = u[(0, 1)].re; let u01i = u[(0, 1)].im;
        let u10r = u[(1, 0)].re; let u10i = u[(1, 0)].im;
        let u11r = u[(1, 1)].re; let u11i = u[(1, 1)].im;

        let mut next_data = vec![Complex64::new(0.0, 0.0); dim];
        
        // Chunked parallel execution (same as statevector)
        let chunk_size = std::cmp::max(1, dist);
        let _n_chunks = dim / (2 * chunk_size);

        next_data.par_chunks_mut(2 * chunk_size).enumerate().for_each(|(c, chunk)| {
            let (lo, hi) = chunk.split_at_mut(chunk_size);
            let src_offset = c * 2 * chunk_size;
            let src_lo = &self.data[src_offset..src_offset + chunk_size];
            let src_hi = &self.data[src_offset + chunk_size..src_offset + 2 * chunk_size];
            
            crate::dag::apply_2x2_kernel_f64(
                src_lo, src_hi, lo, hi,
                u00r, u00i, u01r, u01i, u10r, u10i, u11r, u11i
            );
        });
        
        self.data = next_data;
    }

    fn apply_2q_gate(&mut self, u: &DMatrix<Complex64>, q0: usize, q1: usize) {
        // Reuse the logic from Statevector but adapted for DM size.
        // q0 = control (first qubit), q1 = target (second qubit)
        // Row ordering: row = bit(q0)*2 + bit(q1) = |q0,q1⟩
        // This matches the statevector convention (dag.rs line 819).
        let n_total = 2 * self.n_qubits;
        let dim = 1 << n_total;
        let mut next_data = vec![Complex64::new(0.0, 0.0); dim];

        next_data.par_iter_mut().enumerate().for_each(|(i, target)| {
            let bit0 = (i >> q0) & 1;  // control bit
            let bit1 = (i >> q1) & 1;  // target bit
            let i_base = i & !(1 << q0) & !(1 << q1);
            let row = bit0 * 2 + bit1;  // Matches statevector: control*2 + target
            let mut acc = Complex64::new(0.0, 0.0);
            for col in 0..4 {
                let b0 = (col >> 1) & 1;  // control bit for column
                let b1 = col & 1;         // target bit for column
                let idx = i_base | (b0 << q0) | (b1 << q1);
                acc += u[(row, col)] * self.data[idx];
            }
            *target = acc;
        });
        self.data = next_data;
    }

    pub fn apply_kraus(&mut self, kraus_set: &[DMatrix<Complex64>], qubit: usize) {
        let n_total = 2 * self.n_qubits;
        let dim = 1 << n_total;
        let mut total_next = vec![Complex64::new(0.0, 0.0); dim];

        for k in kraus_set {
            let mut current_k = self.data.clone();
            // Vectorized apply: rho -> K rho K†  =>  |rho>> -> (K \otimes K*) |rho>>
            self.apply_1q_gate_to_vec(&mut current_k, k, qubit);
            let k_star = k.map(|c| c.conj());
            self.apply_1q_gate_to_vec(&mut current_k, &k_star, self.n_qubits + qubit);
            
            for (i, val) in current_k.iter().enumerate() {
                total_next[i] += val;
            }
        }
        self.data = total_next;
    }

    fn apply_1q_gate_to_vec(&self, data: &mut Vec<Complex64>, u: &DMatrix<Complex64>, q: usize) {
        let n_total = 2 * self.n_qubits;
        let dim = 1 << n_total;
        let dist = 1 << q;
        
        let u00r = u[(0, 0)].re; let u00i = u[(0, 0)].im;
        let u01r = u[(0, 1)].re; let u01i = u[(0, 1)].im;
        let u10r = u[(1, 0)].re; let u10i = u[(1, 0)].im;
        let u11r = u[(1, 1)].re; let u11i = u[(1, 1)].im;

        let mut next_data = vec![Complex64::new(0.0, 0.0); dim];
        let chunk_size = std::cmp::max(1, dist);
        
        next_data.par_chunks_mut(2 * chunk_size).enumerate().for_each(|(c, chunk)| {
            let (lo, hi) = chunk.split_at_mut(chunk_size);
            let src_offset = c * 2 * chunk_size;
            let src_lo = &data[src_offset..src_offset + chunk_size];
            let src_hi = &data[src_offset + chunk_size..src_offset + 2 * chunk_size];
            crate::dag::apply_2x2_kernel_f64(
                src_lo, src_hi, lo, hi,
                u00r, u00i, u01r, u01i, u10r, u10i, u11r, u11i
            );
        });
        *data = next_data;
    }
}
