use std::any::Any;
use std::collections::HashMap;

use num_complex::Complex64;

use crate::adjoint::{adjoint_grad, PauliTerm};
use crate::dag::QuantumDAG;
use crate::dm::DensityMatrixState;
use crate::mps::MPSState;
use crate::ops::{OpType, Parameter};
use crate::stabilizer::StabilizerTableau;

#[derive(Debug, Clone)]
pub struct MethodError(pub String);

impl std::fmt::Display for MethodError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for MethodError {}

// ── Pauli algebra helpers (variance needs A² = (Σ c_i P_i)²) ──────────────
// Pauli codes: 0=I, 1=X, 2=Y, 3=Z (same encoding as Python _PAULI_ENCODE).
fn pauli_mul(a: u8, b: u8) -> (u8, Complex64) {
    let one = Complex64::new(1.0, 0.0);
    let i = Complex64::new(0.0, 1.0);
    if a == 0 {
        return (b, one); // I·P = P
    }
    if b == 0 {
        return (a, one); // P·I = P
    }
    if a == b {
        return (0, one); // P·P = I (+1 phase)
    }
    let third = 6 - a - b; // {1,2,3} sums to 6 → the remaining Pauli
    let cyclic = (a == 1 && b == 2) || (a == 2 && b == 3) || (a == 3 && b == 1);
    if cyclic {
        (third, i) // X·Y = iZ, Y·Z = iX, Z·X = iY
    } else {
        (third, -i) // anti-cyclic order picks up −i
    }
}

fn multiply_pauli_terms(a: &PauliTerm, b: &PauliTerm) -> PauliTerm {
    let n = a.paulis.len().max(b.paulis.len());
    let mut paulis = Vec::with_capacity(n);
    let mut phase = Complex64::new(1.0, 0.0);
    for q in 0..n {
        let (code, ph) = pauli_mul(
            a.paulis.get(q).copied().unwrap_or(0),
            b.paulis.get(q).copied().unwrap_or(0),
        );
        paulis.push(code);
        phase *= ph;
    }
    PauliTerm {
        paulis,
        coef: a.coef * b.coef * phase,
    }
}

/// Expand (Σ_i c_i P_i)² into a single weighted Pauli list (exact, O(m²)
/// terms, m = number of observable terms).
fn square_observable(observable: &[PauliTerm]) -> Vec<PauliTerm> {
    let mut acc: HashMap<Vec<u8>, Complex64> = HashMap::new();
    for t1 in observable {
        for t2 in observable {
            let prod = multiply_pauli_terms(t1, t2);
            *acc.entry(prod.paulis).or_insert(Complex64::new(0.0, 0.0)) += prod.coef;
        }
    }
    acc.into_iter()
        .map(|(paulis, coef)| PauliTerm { paulis, coef })
        .collect()
}

pub trait QuantumStateImpl: Send + Sync {
    fn expectation(&self, observable: &[PauliTerm]) -> Result<f64, MethodError>;
    /// Exact variance Var(O) = ⟨O²⟩ − ⟨O⟩² for a Pauli observable.
    ///
    /// ⟨O²⟩ is evaluated exactly by expanding O² into its O(m²) Pauli
    /// products (m = number of observable terms), so every state type gets
    /// a correct value through this default implementation - no per-state
    /// code required.
    fn variance(&self, observable: &[PauliTerm]) -> Result<f64, MethodError> {
        let mean = self.expectation(observable)?;
        let squared = square_observable(observable);
        let mean_sq = self.expectation(&squared)?;
        Ok(mean_sq - mean * mean)
    }
    fn sample(&self, shots: usize, seed: u64) -> Result<HashMap<String, usize>, MethodError>;
    fn grad(
        &self,
        observable: &[PauliTerm],
        dag: &QuantumDAG,
        param_values: &HashMap<String, f64>,
    ) -> Result<Vec<f64>, MethodError>;
    fn entropy(&self) -> Result<f64, MethodError>;
    fn purity(&self) -> Result<f64, MethodError>;
    fn fidelity(&self, other: &dyn QuantumStateImpl) -> Result<f64, MethodError>;
    fn qfim(
        &self,
        dag: &QuantumDAG,
        param_values: &HashMap<String, f64>,
    ) -> Result<Vec<Vec<f64>>, MethodError>;
    fn to_vec(&self) -> Result<Vec<Complex64>, MethodError>;
    fn probabilities(&self) -> Result<Vec<f64>, MethodError>;
    /// Is [`Self::to_vec`] a plain copy of data the state already holds?
    ///
    /// Callers that memoize a densification (the bindings' `State.numpy()`
    /// keeps one so repeated calls skip an O(n·2^n·D²) MPS contraction) can
    /// skip the memo for a state that answers `true`: the memo would only add
    /// a second full-size copy of the same bytes, and re-reading is exactly
    /// as expensive as reading the memo back.  Default `false`, which keeps
    /// the memoized path for MPS, density-matrix and stabilizer states.
    fn to_vec_is_a_copy(&self) -> bool {
        false
    }
    fn partial_trace(
        &self,
        keep_qubits: &[usize],
    ) -> Result<Box<dyn QuantumStateImpl>, MethodError>;
    /// Mutual information I(A;B) = S(A) + S(B) − S(AB) between two
    /// (disjoint) wire sets, computed from partial traces.
    ///
    /// Entropy is in natural-log units (same convention as `entropy()`,
    /// which uses `ln`), matching PennyLane's `qml.mutual_info`.
    fn mutual_info(&self, wires_a: &[usize], wires_b: &[usize]) -> Result<f64, MethodError> {
        let n = self.n_qubits();
        for &w in wires_a.iter().chain(wires_b.iter()) {
            if w >= n {
                return Err(MethodError(format!(
                    "mutual_info() qubit {w} out of range (n_qubits={n})"
                )));
            }
        }
        if wires_a.iter().any(|a| wires_b.contains(a)) {
            return Err(MethodError(
                "mutual_info() wire sets must be disjoint".into(),
            ));
        }
        let s_a = self.partial_trace(wires_a)?.entropy()?;
        let s_b = self.partial_trace(wires_b)?.entropy()?;
        let mut ab: Vec<usize> = wires_a
            .iter()
            .cloned()
            .chain(wires_b.iter().cloned())
            .collect();
        ab.sort_unstable();
        ab.dedup();
        let s_ab = self.partial_trace(&ab)?.entropy()?;
        Ok(s_a + s_b - s_ab)
    }
    fn n_qubits(&self) -> usize;
    fn method_name(&self) -> &str;
    fn device_name(&self) -> &str;
    /// Virtual-to-physical wire map of this handle's amplitude order.
    ///
    /// `perm[virtual_qubit] = physical_site`.  Every backend that stores its
    /// amplitudes in documented qubit order returns the identity.  The MPS
    /// handle returns its lazy-SWAP routing map, because its tensors stay on
    /// fixed *sites* while the circuit's wires move: `to_vec` and `sample`
    /// answer in site order, so a caller that compares a densified MPS against
    /// another engine's statevector must apply this map first.  Identity
    /// unless the circuit contains a non-adjacent two-qubit gate.
    fn perm(&self) -> Vec<usize> {
        (0..self.n_qubits()).collect()
    }
    fn as_any(&self) -> &dyn Any;
}

// ═══════════════════════════════════════════════════════════
// StatevectorState
// ═══════════════════════════════════════════════════════════

pub struct StatevectorState {
    pub data: Vec<Complex64>,
    pub num_qubits: usize,
    pub device: String,
}

impl StatevectorState {
    pub fn new(data: Vec<Complex64>, n_qubits: usize, device: &str) -> Self {
        Self {
            data,
            num_qubits: n_qubits,
            device: device.to_string(),
        }
    }

    fn compute_expval(&self, observable: &[PauliTerm]) -> Result<f64, MethodError> {
        let mut total = Complex64::new(0.0, 0.0);
        for term in observable {
            total += term.coef * pauli_expval_bitmask(&self.data, &term.paulis)?;
        }
        Ok(total.re)
    }

    /// Sum_q <Z_q>/n, using one popcount and no probability-array allocation.
    pub fn mean_z(&self) -> Result<f64, MethodError> {
        if self.num_qubits == 0 {
            return Err(MethodError("mean_z requires at least one qubit".into()));
        }
        let n = self.num_qubits as f64;
        Ok(self
            .data
            .iter()
            .enumerate()
            .map(|(i, z)| z.norm_sqr() * (n - 2.0 * i.count_ones() as f64) / n)
            .sum())
    }

    pub fn diagonal_expectation(&self, weights: &[f64]) -> Result<f64, MethodError> {
        if weights.len() != self.data.len() {
            return Err(MethodError(
                "diagonal length must equal the state dimension".into(),
            ));
        }
        Ok(self
            .data
            .iter()
            .zip(weights)
            .map(|(z, w)| z.norm_sqr() * w)
            .sum())
    }
}

impl QuantumStateImpl for StatevectorState {
    fn expectation(&self, observable: &[PauliTerm]) -> Result<f64, MethodError> {
        self.compute_expval(observable)
    }

    fn sample(&self, shots: usize, seed: u64) -> Result<HashMap<String, usize>, MethodError> {
        if shots == 0 {
            return Ok(HashMap::new());
        }
        let n = self.num_qubits;
        let dim = self.data.len();
        let probs: Vec<f64> = self
            .data
            .iter()
            .map(|c| c.re * c.re + c.im * c.im)
            .collect();

        use rand::Rng;
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let mut counts: HashMap<String, usize> = HashMap::new();

        let mut cumulative = vec![0.0f64; dim + 1];
        for i in 0..dim {
            cumulative[i + 1] = cumulative[i] + probs[i];
        }
        let total = cumulative[dim];

        for _ in 0..shots {
            let r: f64 = rng.gen::<f64>() * total;
            let idx = match cumulative
                .binary_search_by(|v| v.partial_cmp(&r).unwrap_or(std::cmp::Ordering::Less))
            {
                Ok(i) => i.min(dim - 1),
                Err(i) => (i - 1).min(dim - 1),
            };
            let bitstring: String = (0..n)
                .rev()
                .map(|q| if (idx >> q) & 1 == 1 { '1' } else { '0' })
                .collect();
            *counts.entry(bitstring).or_insert(0) += 1;
        }
        Ok(counts)
    }

    fn grad(
        &self,
        observable: &[PauliTerm],
        dag: &QuantumDAG,
        param_values: &HashMap<String, f64>,
    ) -> Result<Vec<f64>, MethodError> {
        let result = adjoint_grad(dag, observable, param_values)?;
        Ok(result.gradients)
    }

    fn entropy(&self) -> Result<f64, MethodError> {
        Ok(0.0) // pure state
    }

    fn purity(&self) -> Result<f64, MethodError> {
        Ok(1.0) // pure state
    }

    fn fidelity(&self, other: &dyn QuantumStateImpl) -> Result<f64, MethodError> {
        let other_sv = other
            .as_any()
            .downcast_ref::<StatevectorState>()
            .ok_or_else(|| {
                MethodError("fidelity() requires both states to be the same type".into())
            })?;
        if self.num_qubits != other_sv.num_qubits {
            return Err(MethodError(
                "fidelity() requires same number of qubits".into(),
            ));
        }
        let ip: Complex64 = self
            .data
            .iter()
            .zip(other_sv.data.iter())
            .map(|(a, b)| a.conj() * b)
            .sum();
        Ok(ip.re * ip.re + ip.im * ip.im) // |<ψ|φ>|²
    }

    fn qfim(
        &self,
        dag: &QuantumDAG,
        param_values: &HashMap<String, f64>,
    ) -> Result<Vec<Vec<f64>>, MethodError> {
        let param_names = dag.parameter_names();
        let n_params = param_names.len();
        if n_params == 0 {
            return Ok(vec![]);
        }

        let eps = 1e-5;
        let mut d_psi: Vec<Vec<Complex64>> = Vec::with_capacity(n_params);

        for name in &param_names {
            let orig_val = *param_values.get(name.as_str()).ok_or_else(|| {
                MethodError(format!("parameter '{}' not found in param_values", name))
            })?;

            let mut params_plus = param_values.clone();
            params_plus.insert(name.clone(), orig_val + eps);
            let dag_plus = dag.bind(&params_plus);
            let psi_plus = dag_plus.simulate();

            let mut params_minus = param_values.clone();
            params_minus.insert(name.clone(), orig_val - eps);
            let dag_minus = dag.bind(&params_minus);
            let psi_minus = dag_minus.simulate();

            let deriv: Vec<Complex64> = psi_plus
                .iter()
                .zip(psi_minus.iter())
                .map(|(p, m)| (*p - *m) / (2.0 * eps))
                .collect();
            d_psi.push(deriv);
        }

        // Fubini-Study metric: Q[i][j] = 4 Re( <dpsi_i|dpsi_j>
        //                            - <dpsi_i|psi><psi|dpsi_j> ).
        // conn[k] = <psi|dpsi_k> supplies the covariance (gauge) term that
        // removes global-phase motion of the state (SUP-18).
        let conn: Vec<Complex64> = d_psi
            .iter()
            .map(|d| {
                self.data
                    .iter()
                    .zip(d.iter())
                    .map(|(a, b)| a.conj() * b)
                    .sum()
            })
            .collect();

        let mut qfim = vec![vec![0.0f64; n_params]; n_params];
        for i in 0..n_params {
            for j in i..n_params {
                let overlap: Complex64 = d_psi[i]
                    .iter()
                    .zip(d_psi[j].iter())
                    .map(|(a, b)| a.conj() * b)
                    .sum();
                let val = 4.0 * (overlap - conn[i].conj() * conn[j]).re;
                qfim[i][j] = val;
                qfim[j][i] = val;
            }
        }
        Ok(qfim)
    }

    fn to_vec(&self) -> Result<Vec<Complex64>, MethodError> {
        Ok(self.data.clone())
    }

    // `self.data` *is* the dense vector, so a memoized copy of `to_vec` would
    // double a 2^n·16-byte array for no benefit -- re-reading costs the same
    // as reading the memo back.
    fn to_vec_is_a_copy(&self) -> bool {
        true
    }

    fn probabilities(&self) -> Result<Vec<f64>, MethodError> {
        Ok(self
            .data
            .iter()
            .map(|c| c.re * c.re + c.im * c.im)
            .collect())
    }

    fn partial_trace(
        &self,
        keep_qubits: &[usize],
    ) -> Result<Box<dyn QuantumStateImpl>, MethodError> {
        let n = self.num_qubits;
        let n_keep = keep_qubits.len();
        let dim_keep = 1usize << n_keep;

        let mut traced: Vec<usize> = (0..n).filter(|q| !keep_qubits.contains(q)).collect();
        traced.sort();
        let n_traced = traced.len();
        let dim_traced = 1usize << n_traced;

        let mut rho_matrix = vec![vec![Complex64::new(0.0, 0.0); dim_keep]; dim_keep];

        for i_keep in 0..dim_keep {
            for j_keep in 0..dim_keep {
                let mut val = Complex64::new(0.0, 0.0);
                for t in 0..dim_traced {
                    let mut idx_i = 0usize;
                    let mut idx_j = 0usize;
                    for (k, &q) in keep_qubits.iter().enumerate() {
                        if (i_keep >> k) & 1 == 1 {
                            idx_i |= 1 << q;
                        }
                        if (j_keep >> k) & 1 == 1 {
                            idx_j |= 1 << q;
                        }
                    }
                    for (k, &q) in traced.iter().enumerate() {
                        if (t >> k) & 1 == 1 {
                            idx_i |= 1 << q;
                            idx_j |= 1 << q;
                        }
                    }
                    val += self.data[idx_i] * self.data[idx_j].conj();
                }
                rho_matrix[i_keep][j_keep] = val;
            }
        }

        let mut dm = DensityMatrixState::new(n_keep);
        for ket in 0..dim_keep {
            for bra in 0..dim_keep {
                dm.set_at(ket, bra, rho_matrix[ket][bra]);
            }
        }

        Ok(Box::new(DensityMatrixStateWrapper {
            inner: dm,
            device: self.device.clone(),
        }))
    }

    fn n_qubits(&self) -> usize {
        self.num_qubits
    }
    fn method_name(&self) -> &str {
        "statevector"
    }
    fn device_name(&self) -> &str {
        &self.device
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

// ═══════════════════════════════════════════════════════════
// DensityMatrixStateWrapper
// ═══════════════════════════════════════════════════════════

pub struct DensityMatrixStateWrapper {
    pub inner: DensityMatrixState,
    pub device: String,
}

impl DensityMatrixStateWrapper {
    pub fn new(dm: DensityMatrixState, device: &str) -> Self {
        Self {
            inner: dm,
            device: device.to_string(),
        }
    }

    fn to_matrix(&self) -> Vec<Vec<Complex64>> {
        let n = self.inner.n_qubits;
        let dim = 1 << n;
        let mut rho = vec![vec![Complex64::new(0.0, 0.0); dim]; dim];
        for ket in 0..dim {
            for bra in 0..dim {
                rho[ket][bra] = self.inner.at(ket, bra);
            }
        }
        rho
    }
}

impl QuantumStateImpl for DensityMatrixStateWrapper {
    fn expectation(&self, observable: &[PauliTerm]) -> Result<f64, MethodError> {
        let n = self.inner.n_qubits;
        let dim = 1 << n;
        let rho = self.to_matrix();
        let mut total = Complex64::new(0.0, 0.0);

        for term in observable {
            for i in 0..dim {
                let mut phase = Complex64::new(1.0, 0.0);
                let mut target_idx = i;

                for (q, &pauli_op) in term.paulis.iter().enumerate() {
                    // Qubit q lives at bit position q (little-endian statevector).
                    let bit_pos = q;
                    match pauli_op {
                        1 => {
                            target_idx ^= 1 << bit_pos;
                        }
                        2 => {
                            target_idx ^= 1 << bit_pos;
                            if (i >> bit_pos) & 1 == 0 {
                                phase *= Complex64::new(0.0, -1.0);
                            } else {
                                phase *= Complex64::new(0.0, 1.0);
                            }
                        }
                        3 if (i >> bit_pos) & 1 == 1 => {
                            phase *= -1.0;
                        }
                        3 => {}
                        _ => {}
                    }
                }
                total += term.coef * phase * rho[target_idx][i];
            }
        }
        Ok(total.re)
    }

    fn sample(&self, shots: usize, seed: u64) -> Result<HashMap<String, usize>, MethodError> {
        if shots == 0 {
            return Ok(HashMap::new());
        }
        let n = self.inner.n_qubits;
        let dim = 1 << n;
        let probs: Vec<f64> = (0..dim).map(|i| self.inner.at(i, i).re).collect();

        use rand::Rng;
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let mut counts: HashMap<String, usize> = HashMap::new();

        let mut cumulative = vec![0.0f64; dim + 1];
        for i in 0..dim {
            cumulative[i + 1] = cumulative[i] + probs[i].max(0.0);
        }
        let total = cumulative[dim];

        for _ in 0..shots {
            let r: f64 = rng.gen::<f64>() * total;
            let idx = match cumulative
                .binary_search_by(|v| v.partial_cmp(&r).unwrap_or(std::cmp::Ordering::Less))
            {
                Ok(i) => i.min(dim - 1),
                Err(i) => (i - 1).min(dim - 1),
            };
            let bitstring: String = (0..n)
                .rev()
                .map(|q| if (idx >> q) & 1 == 1 { '1' } else { '0' })
                .collect();
            *counts.entry(bitstring).or_insert(0) += 1;
        }
        Ok(counts)
    }

    fn grad(
        &self,
        _observable: &[PauliTerm],
        _dag: &QuantumDAG,
        _param_values: &HashMap<String, f64>,
    ) -> Result<Vec<f64>, MethodError> {
        Err(MethodError(
            "grad() not supported for density_matrix method".into(),
        ))
    }

    fn entropy(&self) -> Result<f64, MethodError> {
        let n = self.inner.n_qubits;
        let dim = 1 << n;
        let mut rho = nalgebra::DMatrix::<Complex64>::zeros(dim, dim);
        for ket in 0..dim {
            for bra in 0..dim {
                rho[(ket, bra)] = self.inner.at(ket, bra);
            }
        }
        let eig = rho.symmetric_eigen();
        let mut s = 0.0;
        for &lam in eig.eigenvalues.iter() {
            if lam > 1e-15 {
                s -= lam * lam.ln();
            }
        }
        Ok(s)
    }

    fn purity(&self) -> Result<f64, MethodError> {
        let n = self.inner.n_qubits;
        let dim = 1 << n;
        let mut rho = nalgebra::DMatrix::<Complex64>::zeros(dim, dim);
        for ket in 0..dim {
            for bra in 0..dim {
                rho[(ket, bra)] = self.inner.at(ket, bra);
            }
        }
        Ok((&rho * &rho).trace().re)
    }

    fn fidelity(&self, other: &dyn QuantumStateImpl) -> Result<f64, MethodError> {
        let other_dm = other
            .as_any()
            .downcast_ref::<DensityMatrixStateWrapper>()
            .ok_or_else(|| {
                MethodError(
                    "fidelity() between density_matrix states requires both to be density_matrix"
                        .into(),
                )
            })?;
        let n = self.inner.n_qubits;
        if n != other_dm.inner.n_qubits {
            return Err(MethodError(
                "fidelity() requires same number of qubits".into(),
            ));
        }
        let dim = 1 << n;

        let mut rho = nalgebra::DMatrix::<Complex64>::zeros(dim, dim);
        let mut sigma = nalgebra::DMatrix::<Complex64>::zeros(dim, dim);
        for ket in 0..dim {
            for bra in 0..dim {
                rho[(ket, bra)] = self.inner.at(ket, bra);
                sigma[(ket, bra)] = other_dm.inner.at(ket, bra);
            }
        }

        Ok((&rho * &sigma).trace().re.abs())
    }

    fn qfim(
        &self,
        _dag: &QuantumDAG,
        _param_values: &HashMap<String, f64>,
    ) -> Result<Vec<Vec<f64>>, MethodError> {
        Err(MethodError(
            "qfim() not supported for density_matrix method".into(),
        ))
    }

    fn to_vec(&self) -> Result<Vec<Complex64>, MethodError> {
        let n = self.inner.n_qubits;
        let dim = 1 << n;
        let mut flat = Vec::with_capacity(dim * dim);
        for ket in 0..dim {
            for bra in 0..dim {
                flat.push(self.inner.at(ket, bra));
            }
        }
        Ok(flat)
    }

    fn probabilities(&self) -> Result<Vec<f64>, MethodError> {
        let n = self.inner.n_qubits;
        let dim = 1 << n;
        Ok((0..dim).map(|i| self.inner.at(i, i).re.max(0.0)).collect())
    }

    fn partial_trace(
        &self,
        keep_qubits: &[usize],
    ) -> Result<Box<dyn QuantumStateImpl>, MethodError> {
        let n = self.inner.n_qubits;
        let n_keep = keep_qubits.len();
        let dim_keep = 1usize << n_keep;
        let rho = self.to_matrix();

        let mut traced: Vec<usize> = (0..n).filter(|q| !keep_qubits.contains(q)).collect();
        traced.sort();
        let n_traced = traced.len();
        let dim_traced = 1usize << n_traced;

        let mut rho_reduced = vec![vec![Complex64::new(0.0, 0.0); dim_keep]; dim_keep];
        for i_keep in 0..dim_keep {
            for j_keep in 0..dim_keep {
                let mut val = Complex64::new(0.0, 0.0);
                for t in 0..dim_traced {
                    let mut idx_i = 0usize;
                    let mut idx_j = 0usize;
                    for (k, &q) in keep_qubits.iter().enumerate() {
                        if (i_keep >> k) & 1 == 1 {
                            idx_i |= 1 << q;
                        }
                        if (j_keep >> k) & 1 == 1 {
                            idx_j |= 1 << q;
                        }
                    }
                    for (k, &q) in traced.iter().enumerate() {
                        if (t >> k) & 1 == 1 {
                            idx_i |= 1 << q;
                            idx_j |= 1 << q;
                        }
                    }
                    val += rho[idx_i][idx_j];
                }
                rho_reduced[i_keep][j_keep] = val;
            }
        }

        let mut dm = DensityMatrixState::new(n_keep);
        for ket in 0..dim_keep {
            for bra in 0..dim_keep {
                dm.set_at(ket, bra, rho_reduced[ket][bra]);
            }
        }

        Ok(Box::new(DensityMatrixStateWrapper {
            inner: dm,
            device: self.device.clone(),
        }))
    }

    fn n_qubits(&self) -> usize {
        self.inner.n_qubits
    }
    fn method_name(&self) -> &str {
        "density_matrix"
    }
    fn device_name(&self) -> &str {
        &self.device
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

// ═══════════════════════════════════════════════════════════
// MPS State Wrapper
// ═══════════════════════════════════════════════════════════

pub struct MPSStateWrapper {
    pub inner: MPSState,
    pub device: String,
    /// Right-canonical copy built on the first `sample` call.  Evolution
    /// skips the eager O(n·D³) canonicalization sweep (expectation is a
    /// gauge-invariant contraction); sequential sampling needs canonical
    /// form, so it is built (and kept) here on demand instead.
    canon_cache: std::sync::Mutex<Option<MPSState>>,
}

impl MPSStateWrapper {
    pub fn new(mps: MPSState, device: &str) -> Self {
        Self {
            inner: mps,
            device: device.to_string(),
            canon_cache: std::sync::Mutex::new(None),
        }
    }
}

impl QuantumStateImpl for MPSStateWrapper {
    fn expectation(&self, observable: &[PauliTerm]) -> Result<f64, MethodError> {
        let mut total = 0.0;
        for term in observable {
            let pauli_u8: Vec<u8> = term.paulis.clone();
            // Wire order: the chain is in site order after lazy-SWAP routing.
            let z = self.inner.pauli_expval_wires(&pauli_u8);
            total += (term.coef * z).re;
        }
        Ok(total)
    }

    fn sample(&self, shots: usize, seed: u64) -> Result<HashMap<String, usize>, MethodError> {
        // Sampling walks the chain left-to-right assuming orthonormal right
        // tails (see MPSState::sample_one_with_rng), so the state must be
        // right-canonical.  States from simulate_to_state("mps") are no
        // longer canonicalized eagerly; build the canonical copy on first
        // draw and cache it, so only the first sample pays the sweep.
        // Sites ≥ rc_start are already isometric, so the sweep only needs
        // the dirty window 1..rc_start.
        if self.inner.rc_start == 1 {
            return Ok(self.inner.sample_wires(shots, seed));
        }
        let mut guard = self.canon_cache.lock().unwrap();
        let canon = guard.get_or_insert_with(|| {
            let mut c = self.inner.clone();
            let rc = c.rc_start;
            c.canonicalize_right_range(1, rc);
            c.rc_start = 1;
            c
        });
        Ok(canon.sample_wires(shots, seed))
    }

    fn grad(
        &self,
        observable: &[PauliTerm],
        dag: &QuantumDAG,
        param_values: &HashMap<String, f64>,
    ) -> Result<Vec<f64>, MethodError> {
        mps_param_shift_grad(&self.inner, observable, dag, param_values)
    }

    fn entropy(&self) -> Result<f64, MethodError> {
        // The handle is a pure state vector — bond truncation leaves it pure,
        // only sub-normalized — so the von Neumann entropy of the whole state
        // is zero, exactly as `statevector` answers.  For a subsystem use
        // `partial_trace(&wires)?.entropy()` (or `mutual_info`).
        Ok(0.0)
    }

    fn purity(&self) -> Result<f64, MethodError> {
        Ok(1.0) // pure state, as for statevector
    }

    fn fidelity(&self, other: &dyn QuantumStateImpl) -> Result<f64, MethodError> {
        mps_fidelity(&self.inner, other)
    }

    fn qfim(
        &self,
        _dag: &QuantumDAG,
        _param_values: &HashMap<String, f64>,
    ) -> Result<Vec<Vec<f64>>, MethodError> {
        // Unlike the readouts below, the QFIM is not a contraction of the
        // chain: it needs the state tangent ∂_i|ψ⟩, which this handle does not
        // expose.  The only faithful route is O(P²) state overlaps -- 2P MPS
        // evolutions plus one contraction per pair, i.e. ~2·10³ s at P = 430,
        // n = 22 -- so this refuses instead of pretending the cost is fine.
        Err(MethodError(
            "qfim() needs the state tangent, which the MPS handle does not \
             expose (a faithful implementation costs O(P²) chain contractions); \
             use method='statevector' for QFIM"
                .into(),
        ))
    }

    fn to_vec(&self) -> Result<Vec<Complex64>, MethodError> {
        Ok(self.inner.to_statevector())
    }

    fn probabilities(&self) -> Result<Vec<f64>, MethodError> {
        mps_probabilities(&self.inner)
    }

    fn partial_trace(
        &self,
        keep_qubits: &[usize],
    ) -> Result<Box<dyn QuantumStateImpl>, MethodError> {
        mps_partial_trace(&self.inner, keep_qubits, &self.device)
    }

    fn n_qubits(&self) -> usize {
        self.inner.n_qubits
    }
    fn method_name(&self) -> &str {
        "mps"
    }
    fn perm(&self) -> Vec<usize> {
        self.inner.perm.clone()
    }
    fn device_name(&self) -> &str {
        &self.device
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

// ═══════════════════════════════════════════════════════════
// MPS gradient: parameter-shift on the tensor-network forward
// ═══════════════════════════════════════════════════════════
//
// `grad()` on an MPS handle used to be a stub.  A reverse-mode adjoint needs
// the dense 2^n state that MPS exists to avoid, so the gradient is taken on
// the forward instead: 2 forwards per parameter, each a full tensor-network
// evolution that never densifies (`expectation` contracts the chain).
//
// The rule.  For a gate U(θ) = exp(−iθG) whose generator G has one non-zero
// eigenvalue spacing ω, ⟨O⟩(θ) is a sum of sinusoids exp(±iωθ), so
//
//     ∂⟨O⟩/∂θ = (ω/2)·[⟨O⟩(θ + π/2ω) − ⟨O⟩(θ − π/2ω)]
//
// is exact (sin(ω·s) = 1 at s = π/2ω).  Every single-angle gate in this IR —
// Rx, Ry, Rz, R1, P, Rxx, Ryy, Rzz, CP and all three slots of U — has ω = 1,
// i.e. the familiar shift π/2 with scale ½.  The controlled rotations (CRx,
// CRz, Cu slots 0-2) have generator spectrum {0, ±½}, so frequencies ½ *and*
// 1 are both present and NO single-shift rule is exact for them; the same is
// true of a parameter that enters as an expression (2·v, v+1) or that feeds
// two gates: two occurrences add a second harmonic exp(±2iωθ), which the same
// single shift cannot also cancel.  Those take a finite difference on the
// parameter VALUE instead, which is exact for any occurrence pattern.
//
// There are two of those, chosen by what the caller is comparing against:
//   * a smooth forward with no exact rule keeps the five-point rule (h = 1e-3,
//     O(h⁴), ~1e-13) because it is cheap there and the adjoint comparison is
//     held to 1e-8;
//   * a *truncating* run takes a three-point rule (h = 1e-4, O(h²), measured
//     4e-9 against a fine reference) because re-evolving is the entire cost
//     and the truncated functional is already ~1e-1 away from exact.  Same
//     evolution count as the two-term rule, so a binding bond cap no longer
//     doubles the price of a gradient.
//
// Validated against `sf.cat(mode="adjoint")`: max|dG| = 1.1e-11 at n=22
// (P=430) and 2.2e-16 at n=6 — see _sfpl_bench/_mpsgrad.py.

/// `(shift, scale)` of the exact two-term rule for `slot` of `op`, or `None`
/// when no single-shift rule is exact (see the section note above).
fn mps_shift_rule(op: &OpType, slot: usize) -> Option<(f64, f64)> {
    let omega1 = Some((std::f64::consts::FRAC_PI_2, 0.5));
    match op {
        OpType::Rx(_)
        | OpType::Ry(_)
        | OpType::Rz(_)
        | OpType::R1(_)
        | OpType::P(_)
        | OpType::Rxx(_)
        | OpType::Ryy(_)
        | OpType::Rzz(_)
        | OpType::CP(_)
            if slot == 0 =>
        {
            omega1
        }
        OpType::U(..) if slot < 3 => omega1,
        OpType::Cu(..) if slot == 3 => omega1,
        _ => None,
    }
}

/// The one exact rule shared by every occurrence of `name`, if the parameter
/// occurs *exactly once* and that gate has one; `None` means "use the
/// stencil" — a gate with no exact two-term rule, an expression argument, a
/// repeated parameter, or two occurrences with different shifts.
fn mps_exact_shift(name: &str, ops: &[&OpType]) -> Option<(f64, f64)> {
    let mut rule: Option<(f64, f64)> = None;
    let mut occurrences = 0usize;
    for op in ops {
        for (slot, param) in op.parameters().iter().enumerate() {
            if !param.variable_names().iter().any(|v| v == name) {
                continue;
            }
            occurrences += 1;
            // Only a bare variable moves the gate angle 1:1 with the value.
            let bare = matches!(
                param,
                Parameter::Variable { name: n, .. } if n == name
            );
            let r = if bare { mps_shift_rule(op, slot) } else { None };
            match (r, rule) {
                (Some(r), None) => rule = Some(r),
                (Some(r), Some(prev)) if prev == r => {}
                _ => return None,
            }
        }
    }
    // A parameter used by two gates makes ⟨O⟩ a sum of exp(±iωθ) *and*
    // exp(±2iωθ): a single shift no longer cancels both, so two occurrences
    // are stencil work even when their rules agree.
    if occurrences == 1 {
        rule
    } else {
        None
    }
}

/// Evolve an already-bound `dag` and contract the observable on the resulting
/// chain — the same two steps, in the same order, as
/// `MPSStateWrapper::expectation`, so the gradient differentiates exactly the
/// functional that `expectation()` reports (truncation included).
///
/// Returns `(⟨O⟩, max_discarded_weight)`; the second value is what decides
/// whether the two-term shift rule is allowed for this parameter.
fn mps_expectation(bound: &QuantumDAG, observable: &[PauliTerm], bond: usize) -> (f64, f64) {
    // Test-only tally of how many forward evaluations the last gradient paid.
    // The cost of this lane *is* the evolution count — one full sweep per
    // sample — so the two regressions worth guarding are "a binding cap
    // doubled the price" and "a gate with no exact rule silently took the
    // shift path".  Compiled out of release builds.
    #[cfg(test)]
    MPS_EVAL_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let st = bound.evolve_mps(bond);
    let value = observable
        .iter()
        .map(|term| (term.coef * st.pauli_expval_wires(&term.paulis)).re)
        .sum();
    (value, st.max_discarded_weight)
}

/// Threshold on `max_discarded_weight` below which the two-term rule is the
/// derivative of the reported functional to ~1e-12, far inside the gradient
/// tolerance.
///
/// It used to be `1e-15`, borrowed from [`MPSState::record_discard`]'s
/// "non-negligible" epsilon — and that was wrong, because an *untruncated*
/// chain still reports the round-off floor as a discard: measured 5.551e-16
/// on the 5-qubit ladder at bond 8, i.e. just under the threshold.  Whether a
/// parameter took the two-evolution rule or the four-evolution fallback was
/// then decided by floating-point dust, so an exact run cost 2P here and 4P
/// there and the measured gradient cost sat between the two.  A real cut is
/// 5.4e-02 on the same shape at bond 1, so the two regimes are 14 decades
/// apart and any threshold in between separates them by physics; 1e-12 is
/// 3.3 decades above the floor and 9 below the smallest cut measured.
const MPS_SHIFT_TRUNC_EPS: f64 = 1e-12;

// Test-only count of forward evaluations, incremented by `mps_expectation`.
//
// Atomic rather than thread-local: the gradient loop splits across parameters
// whenever rayon has threads, so a `Cell` would only ever see the calling
// thread's share and the cost regressions these tests guard (2P vs 4P) would
// read as a pass.
#[cfg(test)]
static MPS_EVAL_CALLS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Parameter-shift gradient of `⟨O⟩` on the MPS forward.  Returns one value
/// per `dag.parameter_names()` entry, in that order; `param_values` must bind
/// every parameter (same contract as the statevector adjoint).
fn mps_param_shift_grad(
    state: &MPSState,
    observable: &[PauliTerm],
    dag: &QuantumDAG,
    param_values: &HashMap<String, f64>,
) -> Result<Vec<f64>, MethodError> {
    let names = dag.parameter_names();
    if names.is_empty() {
        return Ok(Vec::new());
    }
    // Re-evolve at the cap the handle was built with, so every forward in the
    // loop is the same truncated functional the user already has.
    let bond = state.bond_dim;

    // Gate list built once: deciding whether the exact rule applies needs the
    // op type and slot of every occurrence of a parameter.  Read the
    // *symbolic* op, not `op_type`: `QuantumDAG::update_parameters` replaces
    // `op_type` with the bound gate, so a caller that had already rebound the
    // dag (a `cat` mps plan keeps one per handle) would show no variables here
    // and silently send every parameter down the 4-evolution stencil -- 2x the
    // cost, and a value that drifts from the rule by ~5e-13.
    let order = dag.topological_order();
    let ops: Vec<&OpType> = order
        .iter()
        .filter_map(|&node_id| {
            let op = &dag.graph()[node_id];
            let sym = &op.original_op_type;
            if sym.is_boundary() || *sym == OpType::Barrier || sym.is_measurement() {
                return None;
            }
            Some(sym)
        })
        .collect();

    // Caveat: `add_gate` only registers *bare variables* in `param_locations`,
    // so a gate whose angle is an expression still needs the full rebind.
    let needs_full_bind = ops
        .iter()
        .flat_map(|op| op.parameters())
        .any(|p| !p.is_variable() && p.try_evaluate().is_none());
    // Truncation is a property of the handle, not of the parameter, so this
    // gate is decided once instead of per parameter.
    let state_truncated = state.max_discarded_weight > MPS_SHIFT_TRUNC_EPS;

    // One parameter's gradient, run against a value map and a DAG the caller
    // owns.
    //
    // The loop over parameters used to share a single `bound` and a single
    // `values`, mutated in place and restored, which pinned it to one thread:
    // measured 626 ms at n=20, L=4, bond 64, P=160 — and flat from 1 to 8
    // threads.  Splitting *across* parameters is the right axis here, because
    // one evolution at that size is a chain of ~8 us SVD merges (far too fine
    // to split) while P is embarrassingly wide.  Measured after the split:
    // 910 -> 583 -> 444 -> 348 ms at 1/2/4/8 threads (2.6x).
    //
    // The shared-map form existed to avoid re-cloning the DAG per sample
    // (`update_parameters` touches only the nodes a parameter sits on and
    // always rebinds from `original_op_type`, so nothing compounds across
    // samples, and the per-evaluation op rebuild that `dag.bind` pays
    // disappears — 1.11x of the hand-written loop, measured by
    // _mpsgrad_overhead.py).  So the clone is *not* taken per parameter: the
    // serial arm below takes one and reuses it, which is the old cost exactly,
    // and only the parallel arm pays a clone per worker task.  (First cut
    // cloned unconditionally and cost 45% at one thread, 626 -> 910 ms, for
    // machinery the single-threaded path does not use.)
    let one = |name: &String,
               values: &mut HashMap<String, f64>,
               bound: &mut QuantumDAG|
     -> Result<f64, MethodError> {
        let theta = *param_values.get(name.as_str()).ok_or_else(|| {
            MethodError(format!("parameter '{}' not found in param_values", name))
        })?;
        // The map and the DAG are per-worker in the parallel arm: sharing the
        // in-place insert/restore this used to rely on would leave a shifted
        // value behind and silently corrupt every later parameter — the bug
        // that once made this lane read 4e-02 wrong.
        let mut evaluate = |v: &HashMap<String, f64>| {
            if needs_full_bind {
                *bound = dag.bind(v);
            } else {
                bound.update_parameters(v);
            }
            mps_expectation(bound, observable, bond)
        };

        // The two-term rule differentiates the *unitary* evolution.  A
        // truncated cut moves its discarded subspace with θ, so once weight is
        // dropped the rule answers for a functional the handle does not
        // report: measured 8.6e-02 wrong on a 5-qubit ladder at bond_dim=1,
        // while the stencil matched the forward's finite difference to
        // 4.5e-10.  So the rule is used only when nothing was discarded -- in
        // the state in hand and on both ± legs.
        //
        // The state's own flag is tested *before* the pair is evaluated: at a
        // binding cap the pair is known to be rejected, and paying two
        // evolutions to discover that was the whole difference between a 2P
        // and a 4P gradient.
        let mut truncated = state_truncated;
        let mut from_shift = None;
        if !truncated {
            if let Some((shift, scale)) = mps_exact_shift(name, &ops) {
                values.insert(name.clone(), theta + shift);
                let (fp, dp) = evaluate(values);
                values.insert(name.clone(), theta - shift);
                let (fm, dm) = evaluate(values);
                values.insert(name.clone(), theta);
                if dp <= MPS_SHIFT_TRUNC_EPS && dm <= MPS_SHIFT_TRUNC_EPS {
                    from_shift = Some(scale * (fp - fm));
                } else {
                    // Exact in hand, truncating off it: the pair is already
                    // paid for, so the fallback is the accurate one.
                    truncated = true;
                }
            }
        }

        let g = match from_shift {
            Some(g) => g,
            None if truncated => {
                // Truncating (see the note above the shift rule): two samples
                // instead of four, because every sample re-evolves the whole
                // circuit and the functional being differentiated is already
                // approximate.  Measured 4e-9 on n=12 at bond 2-4 against a
                // fine 5-point reference -- round-off is 1e-16/h = 1e-12, so
                // the O(h²) bias dominates and 1e-4 is the knee.
                const H3: f64 = 1e-4;
                values.insert(name.clone(), theta - H3);
                let fm = evaluate(values).0;
                values.insert(name.clone(), theta + H3);
                let fp = evaluate(values).0;
                values.insert(name.clone(), theta);
                (fp - fm) / (2.0 * H3)
            }
            None => {
                // Smooth but no exact rule: truncation O(h⁴), round-off
                // O(1e-16/h), so h = 1e-3 lands near 1e-13 — inside the 1e-8
                // gradient tolerance and only reached by the gates listed.
                const H: f64 = 1e-3;
                values.insert(name.clone(), theta - 2.0 * H);
                let fm2 = evaluate(values).0;
                values.insert(name.clone(), theta - H);
                let fm1 = evaluate(values).0;
                values.insert(name.clone(), theta + H);
                let fp1 = evaluate(values).0;
                values.insert(name.clone(), theta + 2.0 * H);
                let fp2 = evaluate(values).0;
                values.insert(name.clone(), theta);
                (8.0 * (fp1 - fm1) - (fp2 - fm2)) / (12.0 * H)
            }
        };
        Ok(g)
    };

    // One parameter or one thread: the serial loop is the same work with less
    // machinery — one clone for the whole sweep instead of one per parameter.
    // `par_iter` preserves `names` order, so the result layout is unchanged
    // either way.
    if names.len() > 1 && rayon::current_num_threads() > 1 {
        use rayon::prelude::*;
        names
            .par_iter()
            .map(|name| {
                let mut values = param_values.clone();
                let mut bound = dag.clone_dag();
                one(name, &mut values, &mut bound)
            })
            .collect()
    } else {
        let mut values = param_values.clone();
        let mut bound = dag.clone_dag();
        names
            .iter()
            .map(|name| one(name, &mut values, &mut bound))
            .collect()
    }
}

// ═══════════════════════════════════════════════════════════
// MPS readout: probabilities, reduced states, fidelity
// ═══════════════════════════════════════════════════════════
//
// `probabilities`, `partial_trace`, `fidelity`, `entropy` and `purity` used
// to answer "not supported for mps method".  None of them needs the dense 2^n
// vector: the probability table is the same left-to-right sweep
// `to_statevector` already runs, the reduced state is a transfer-matrix sweep
// over the chain, and the fidelity contracts two chains against each other.
// `qfim` still refuses -- see its message for why that one is different.

/// Site block `A_s` of a tensor shaped `(D_L·2, D_R)`: `A_s[l, r] = T[l + s·D_L, r]`.
fn mps_site_matrix(t: &nalgebra::DMatrix<Complex64>, s: usize) -> nalgebra::DMatrix<Complex64> {
    let (rows, d_r) = t.shape();
    let d_l = rows / 2;
    nalgebra::DMatrix::from_fn(d_l, d_r, |l, r| t[(l + s * d_l, r)])
}

/// `⟨a|b⟩` by contracting the two chains against each other: the boundary
/// matrix `E[l_a, l_b]` starts at 1 and updates as `E ← Σ_s A_sᴴ E B_s`, so the
/// cost is O(n·D³) and the memory O(D²) — no 2^n vector is ever built.
fn mps_overlap(a: &MPSState, b: &MPSState) -> Result<Complex64, MethodError> {
    if a.n_qubits != b.n_qubits {
        return Err(MethodError(format!(
            "MPS contraction requires the same number of qubits ({} vs {})",
            a.n_qubits, b.n_qubits
        )));
    }
    let mut e = nalgebra::DMatrix::<Complex64>::from_element(1, 1, Complex64::new(1.0, 0.0));
    for i in 0..a.n_qubits {
        let ta = &a.tensors[i];
        let tb = &b.tensors[i];
        let d_ra = ta.shape().1;
        let d_rb = tb.shape().1;
        if e.nrows() != ta.shape().0 / 2 || e.ncols() != tb.shape().0 / 2 {
            return Err(MethodError(format!("MPS bond mismatch at site {i}")));
        }
        let mut next = nalgebra::DMatrix::<Complex64>::zeros(d_ra, d_rb);
        for s in 0..2usize {
            next += mps_site_matrix(ta, s).adjoint() * &e * mps_site_matrix(tb, s);
        }
        e = next;
    }
    Ok(e[(0, 0)])
}

/// `⟨ψ|ψ⟩`.  Truncation can leave the raw chain sub-normalized by Π(1−ε), so
/// every readout below divides that factor back out: the physical state is the
/// normalized one (`sample()` normalizes too), and the raw deficit stays
/// visible in `to_vec()` and `truncation_report()`.
fn mps_norm_sq(state: &MPSState) -> Result<f64, MethodError> {
    Ok(mps_overlap(state, state)?.re.max(0.0))
}

/// Exact `|ψ(x)|²` table in wire (logical qubit) order (bit q ↔ qubit q, the
/// order `to_vec` reports), normalized so the entries sum to 1.  The table is
/// 2^n f64 no matter how it is produced, so this is the same sweep the forward
/// runs; `to_statevector` undoes any lazy-SWAP routing, so the table's bit
/// labels are qubit labels and match `probabilities()` on every other backend.
fn mps_probabilities(state: &MPSState) -> Result<Vec<f64>, MethodError> {
    let amps = state.to_statevector();
    let nrm = mps_norm_sq(state)?;
    let scale = if nrm > 0.0 { 1.0 / nrm } else { 1.0 };
    Ok(amps.iter().map(|c| c.norm_sqr() * scale).collect())
}

/// Reduced density matrix of `keep`, bit j of the result ↔ `keep[j]` — the
/// same convention the statevector method uses.
///
/// `keep` is in wire order: lazy-SWAP routing leaves the tensors in site
/// order, so each wire is mapped through `state.perm` first.  The sweep
/// carries `E[(l1, l2), a, b]`: the bra/ket bond pair of the sites folded in
/// so far plus the kept bits (ket `a`, bra `b`).  A traced site applies
/// `Σ_s A_s ⊗ A_s*` and leaves the kept index alone; a kept site applies the
/// four `A_s ⊗ A_t*` pieces and appends `s` / `t`.  Sites past the last kept
/// qubit are skipped when the chain's suffix is already isometric, because an
/// isometric tail contracts to the identity on the bond pair.
/// Cost is O(n·D³·4^k), memory O(D²·4^k); 4^k is the output size, and the
/// estimate is refused up front rather than run for minutes.
fn mps_partial_trace(
    state: &MPSState,
    keep_wires: &[usize],
    device: &str,
) -> Result<Box<dyn QuantumStateImpl>, MethodError> {
    let n = state.n_qubits;
    // Wire -> site; kept bits are appended in site order, and this `keep`
    // carries the same order into the closing index permutation.
    let mut keep: Vec<usize> = Vec::with_capacity(keep_wires.len());
    let mut is_kept = vec![false; n];
    for &q in keep_wires {
        if q >= n {
            return Err(MethodError(format!(
                "partial_trace() qubit {q} out of range (n_qubits={n})"
            )));
        }
        let site = state.perm[q];
        if is_kept[site] {
            return Err(MethodError(format!(
                "partial_trace() qubit {q} listed twice"
            )));
        }
        is_kept[site] = true;
        keep.push(site);
    }
    let k = keep.len();
    let last_kept = keep.iter().copied().max();
    let m_final = 1usize << k;
    let sites = last_kept.map(|l| l + 1).unwrap_or(0);
    let estimate = sites as f64 * 4.0 * (state.bond_dim as f64).powi(3) * (m_final as f64).powi(2);
    if estimate > 6.0e9 {
        return Err(MethodError(format!(
            "partial_trace() on the MPS handle is O(n·D³·4^k) and this request \
             ({k} kept qubits at bond {}) needs about {:.0} G operations; use \
             method='statevector' or fewer wires",
            state.bond_dim,
            estimate / 1e9
        )));
    }

    let mut d_l = 1usize;
    let mut m = 1usize; // 2^(kept qubits seen)
    let mut broke_early = false;
    let mut e: Vec<Complex64> = vec![Complex64::new(1.0, 0.0)]; // (D_L², m, m)
    for i in 0..n {
        // An isometric tail leaves the bond pair untouched, so once every
        // remaining site is kept-free and isometric the sweep is done.
        if i > last_kept.unwrap_or(usize::MAX) && i >= state.rc_start {
            broke_early = true;
            break;
        }
        let t = &state.tensors[i];
        let (rows, d_r) = t.shape();
        let d_li = rows / 2;
        if d_li != d_l {
            return Err(MethodError(format!(
                "partial_trace() bond mismatch at site {i}: {d_li} != {d_l}"
            )));
        }
        let kept = is_kept[i];
        let m_next = if kept { m * 2 } else { m };
        let m2 = m * m;
        let mn2 = m_next * m_next;
        let mut out = vec![Complex64::new(0.0, 0.0); d_r * d_r * mn2];
        let mut f = vec![Complex64::new(0.0, 0.0); d_r * d_li * m2];
        for s in 0..2usize {
            let a_s = mps_site_matrix(t, s);
            for u in 0..2usize {
                if !kept && u != s {
                    continue; // traced bits are summed, not paired
                }
                // Bra side: A_u, not A_s — the ket and bra physical bits are
                // independent indices of ρ, so they need their own site block.
                let a_u = mps_site_matrix(t, u);
                // F[(r1, l2), a, b] = Σ_l1 A_s[l1, r1] · E[(l1, l2), a, b]
                for r1 in 0..d_r {
                    for l2 in 0..d_li {
                        let base = (r1 * d_li + l2) * m2;
                        for ab in 0..m2 {
                            let mut sum = Complex64::new(0.0, 0.0);
                            for l1 in 0..d_li {
                                sum += a_s[(l1, r1)] * e[(l1 * d_li + l2) * m2 + ab];
                            }
                            f[base + ab] = sum;
                        }
                    }
                }
                // out[(r1, r2), a', b'] += Σ_l2 conj(A_u[l2, r2]) · F[(r1, l2), a, b]
                let a_off = if kept { s * m } else { 0 };
                let b_off = if kept { u * m } else { 0 };
                for r1 in 0..d_r {
                    for r2 in 0..d_r {
                        let out_base = (r1 * d_r + r2) * mn2 + a_off * m_next + b_off;
                        for a in 0..m {
                            for b in 0..m {
                                let mut sum = Complex64::new(0.0, 0.0);
                                for l2 in 0..d_li {
                                    sum +=
                                        a_u[(l2, r2)].conj() * f[(r1 * d_li + l2) * m2 + a * m + b];
                                }
                                out[out_base + a * m_next + b] += sum;
                            }
                        }
                    }
                }
            }
        }
        e = out;
        d_l = d_r;
        m = m_next;
    }
    if d_l != 1 {
        if !broke_early {
            return Err(MethodError(
                "partial_trace() requires the chain's right boundary bond to be 1".into(),
            ));
        }
        // The isometric tail contributes δ on the bond pair, so what is left to
        // do is the trace over that pair: Σ_l E[(l, l), a, b].
        let m2 = m * m;
        let mut collapsed = vec![Complex64::new(0.0, 0.0); m2];
        for a in 0..m {
            for b in 0..m {
                let mut sum = Complex64::new(0.0, 0.0);
                for l in 0..d_l {
                    sum += e[(l * d_l + l) * m2 + a * m + b];
                }
                collapsed[a * m + b] = sum;
            }
        }
        e = collapsed;
    }
    // Trace-normalize: the result is the physical (unit-trace) reduced state.
    let tr: f64 = (0..m).map(|a| e[a * m + a].re).sum();
    let inv = if tr.abs() > 0.0 { 1.0 / tr } else { 1.0 };
    // The sweep appends kept bits in *site* order; the API indexes them in the
    // caller's order (bit k ↔ keep[k]), so permute once at the end.
    let mut sorted: Vec<usize> = keep.to_vec();
    sorted.sort_unstable();
    let mut perm = vec![0usize; m];
    for (rank, &q) in keep.iter().enumerate() {
        let j = sorted.iter().position(|&x| x == q).unwrap_or(rank);
        for (idx, p) in perm.iter_mut().enumerate() {
            if (idx >> j) & 1 == 1 {
                *p |= 1 << rank;
            }
        }
    }
    let mut dm = DensityMatrixState::new(k);
    for ket in 0..m {
        for bra in 0..m {
            dm.set_at(perm[ket], perm[bra], e[ket * m + bra] * inv);
        }
    }
    Ok(Box::new(DensityMatrixStateWrapper {
        inner: dm,
        device: device.to_string(),
    }))
}

/// Site order -> wire order: `out[wire bits] = amps[site bits]`.
///
/// `perm[w]` is the site that holds wire `w`, so wire bit `w` is taken from
/// site bit `perm[w]`.  The identity routing is returned without copying.
fn mps_to_wire_order(amps: &[Complex64], perm: &[usize]) -> Vec<Complex64> {
    let n = perm.len();
    if n <= 1 || perm.iter().enumerate().all(|(w, &s)| w == s) {
        return amps.to_vec();
    }
    let mut out = vec![Complex64::new(0.0, 0.0); amps.len()];
    for (index, &amp) in amps.iter().enumerate() {
        let mut wire = 0usize;
        for (w, &site) in perm.iter().enumerate() {
            if (index >> site) & 1 == 1 {
                wire |= 1 << w;
            }
        }
        out[wire] = amp;
    }
    out
}

/// `|⟨ψ|φ⟩|²` — the convention every other state type uses.
///
/// MPS↔MPS is contracted exactly at O(n·D³) and needs both chains in the
/// same site layout (identical routing).  A dense statevector partner is
/// compared by densifying this chain and relabeling it into wire order,
/// which costs 16·2^n bytes, so it is only allowed while that stays small;
/// the alternative (sweeping the dense vector through the chain) costs
/// O(n·2^n·D²) and is far slower at these sizes.
fn mps_fidelity(state: &MPSState, other: &dyn QuantumStateImpl) -> Result<f64, MethodError> {
    if let Some(o) = other.as_any().downcast_ref::<MPSStateWrapper>() {
        if state.n_qubits != o.inner.n_qubits {
            return Err(MethodError(format!(
                "fidelity() requires same number of qubits ({} vs {})",
                state.n_qubits, o.inner.n_qubits
            )));
        }
        if state.perm != o.inner.perm {
            return Err(MethodError(
                "fidelity() between two mps handles needs the same site routing \
                 (both chains are contracted site by site); compare against a \
                 statevector instead"
                    .into(),
            ));
        }
        let ov = mps_overlap(state, &o.inner)?;
        let na = mps_norm_sq(state)?;
        let nb = mps_norm_sq(&o.inner)?;
        if na <= 0.0 || nb <= 0.0 {
            return Err(MethodError(
                "fidelity() of a zero-norm MPS is undefined".into(),
            ));
        }
        return Ok((ov.norm_sqr() / (na * nb)).clamp(0.0, 1.0));
    }
    if let Some(sv) = other.as_any().downcast_ref::<StatevectorState>() {
        if state.n_qubits != sv.num_qubits {
            return Err(MethodError(format!(
                "fidelity() requires same number of qubits ({} vs {})",
                state.n_qubits, sv.num_qubits
            )));
        }
        if state.n_qubits > 24 {
            return Err(MethodError(format!(
                "fidelity() between mps and statevector densifies the chain \
                 (2^{} amplitudes, {:.1} GiB); compare two mps handles instead",
                state.n_qubits,
                (1u64 << state.n_qubits) as f64 * 16.0 / (1u64 << 30) as f64
            )));
        }
        // Physical site order in, wire order out: `to_statevector` already
        // applies `perm`, so the raw map is the one to relabel here.
        let psi = mps_to_wire_order(&state.to_statevector_site_order(), &state.perm);
        let nrm = mps_norm_sq(state)?.sqrt();
        let nrm_other: f64 = sv.data.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
        if nrm <= 0.0 || nrm_other <= 0.0 {
            return Err(MethodError(
                "fidelity() of a zero-norm state is undefined".into(),
            ));
        }
        let ip: Complex64 = psi
            .iter()
            .zip(sv.data.iter())
            .map(|(a, b)| a.conj() * b)
            .sum();
        return Ok((ip.norm_sqr() / (nrm * nrm * nrm_other * nrm_other)).clamp(0.0, 1.0));
    }
    Err(MethodError(
        "fidelity() between mps and this state type is not supported; \
         compare two mps handles or a statevector"
            .into(),
    ))
}

// ═══════════════════════════════════════════════════════════
// Stabilizer Wrapper
// ═══════════════════════════════════════════════════════════

pub struct StabilizerStateWrapper {
    pub inner: StabilizerTableau,
    pub device: String,
}

impl StabilizerStateWrapper {
    pub fn new(tab: StabilizerTableau, device: &str) -> Self {
        Self {
            inner: tab,
            device: device.to_string(),
        }
    }
}

impl QuantumStateImpl for StabilizerStateWrapper {
    fn expectation(&self, observable: &[PauliTerm]) -> Result<f64, MethodError> {
        let mut total = 0.0;
        for term in observable {
            let n = self.inner.n;
            let mut px = vec![0u8; n];
            let mut pz = vec![0u8; n];
            for (q, &p) in term.paulis.iter().enumerate() {
                if q >= n {
                    break;
                }
                // Tableau qubit index q corresponds to qubit q.
                match p {
                    1 => {
                        px[q] = 1;
                    } // X
                    2 => {
                        px[q] = 1;
                        pz[q] = 1;
                    } // Y = iXZ
                    3 => {
                        pz[q] = 1;
                    } // Z
                    _ => {}
                }
            }
            let ev = self.inner.pauli_expval(&px, &pz);
            total += term.coef.re * ev;
        }
        Ok(total)
    }

    fn sample(&self, shots: usize, seed: u64) -> Result<HashMap<String, usize>, MethodError> {
        Ok(self.inner.sample(shots, Some(seed)))
    }

    fn grad(
        &self,
        _observable: &[PauliTerm],
        _dag: &QuantumDAG,
        _param_values: &HashMap<String, f64>,
    ) -> Result<Vec<f64>, MethodError> {
        Err(MethodError(
            "grad() not supported for stabilizer method".into(),
        ))
    }

    fn entropy(&self) -> Result<f64, MethodError> {
        Err(MethodError(
            "entropy() not supported for stabilizer method".into(),
        ))
    }

    fn purity(&self) -> Result<f64, MethodError> {
        Err(MethodError(
            "purity() not supported for stabilizer method".into(),
        ))
    }

    fn fidelity(&self, _other: &dyn QuantumStateImpl) -> Result<f64, MethodError> {
        Err(MethodError(
            "fidelity() not supported for stabilizer method".into(),
        ))
    }

    fn qfim(
        &self,
        _dag: &QuantumDAG,
        _param_values: &HashMap<String, f64>,
    ) -> Result<Vec<Vec<f64>>, MethodError> {
        Err(MethodError(
            "qfim() not supported for stabilizer method".into(),
        ))
    }

    fn to_vec(&self) -> Result<Vec<Complex64>, MethodError> {
        Err(MethodError(
            "to_vec() not supported for stabilizer method (exponential memory)".into(),
        ))
    }

    fn probabilities(&self) -> Result<Vec<f64>, MethodError> {
        Err(MethodError(
            "probabilities() not supported for stabilizer method".into(),
        ))
    }

    fn partial_trace(
        &self,
        _keep_qubits: &[usize],
    ) -> Result<Box<dyn QuantumStateImpl>, MethodError> {
        Err(MethodError(
            "partial_trace() not supported for stabilizer method".into(),
        ))
    }

    fn n_qubits(&self) -> usize {
        self.inner.n
    }
    fn method_name(&self) -> &str {
        "stabilizer"
    }
    fn device_name(&self) -> &str {
        &self.device
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bell_sv() -> StatevectorState {
        let s = 1.0 / 2.0f64.sqrt();
        StatevectorState::new(
            vec![
                Complex64::new(s, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(s, 0.0),
            ],
            2,
            "cpu",
        )
    }

    #[test]
    fn test_sv_expectation_zz() {
        let state = bell_sv();
        let obs = vec![PauliTerm {
            paulis: vec![3, 3],
            coef: Complex64::new(1.0, 0.0),
        }];
        let ev = state.expectation(&obs).unwrap();
        assert!(
            (ev - 1.0).abs() < 1e-10,
            "ZZ on Bell state should be 1.0, got {}",
            ev
        );
    }

    #[test]
    fn test_sv_sample() {
        let state = bell_sv();
        let counts = state.sample(10000, 42).unwrap();
        let total: usize = counts.values().sum();
        assert_eq!(total, 10000);
        for key in counts.keys() {
            assert!(key == "00" || key == "11", "unexpected bitstring: {}", key);
        }
    }

    #[test]
    fn test_sv_entropy_pure() {
        let state = bell_sv();
        assert!((state.entropy().unwrap()).abs() < 1e-10);
    }

    #[test]
    fn test_sv_purity_pure() {
        let state = bell_sv();
        assert!((state.purity().unwrap() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_sv_fidelity_self() {
        let state = bell_sv();
        let fid = state.fidelity(&state).unwrap();
        assert!(
            (fid - 1.0).abs() < 1e-10,
            "self-fidelity should be 1.0, got {}",
            fid
        );
    }

    #[test]
    fn test_sv_probabilities() {
        let state = bell_sv();
        let probs = state.probabilities().unwrap();
        assert!((probs[0] - 0.5).abs() < 1e-10);
        assert!(probs[1].abs() < 1e-10);
        assert!(probs[2].abs() < 1e-10);
        assert!((probs[3] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_sv_to_vec() {
        let state = bell_sv();
        let v = state.to_vec().unwrap();
        assert_eq!(v.len(), 4);
    }

    #[test]
    fn test_sv_partial_trace_bell() {
        let state = bell_sv();
        let reduced = state.partial_trace(&[0]).unwrap();
        assert_eq!(reduced.n_qubits(), 1);
        assert_eq!(reduced.method_name(), "density_matrix");
        let purity = reduced.purity().unwrap();
        assert!(
            (purity - 0.5).abs() < 1e-10,
            "partial trace of Bell state should give mixed state with purity 0.5, got {}",
            purity
        );
    }

    #[test]
    fn test_stabilizer_method_errors() {
        let tab = StabilizerTableau::new(2);
        let state = StabilizerStateWrapper::new(tab, "cpu");
        assert!(state
            .grad(&[], &QuantumDAG::new(2, 0), &HashMap::new())
            .is_err());
        assert!(state.entropy().is_err());
        assert!(state.purity().is_err());
        assert!(state.qfim(&QuantumDAG::new(2, 0), &HashMap::new()).is_err());
    }
}

/// Pauli expectation in the Rust core's q0=LSB convention. The masks are
/// compiled once per term, eliminating the per-amplitude Pauli-string walk.
pub fn pauli_expval_bitmask(data: &[Complex64], paulis: &[u8]) -> Result<Complex64, MethodError> {
    if data.is_empty()
        || !data.len().is_power_of_two()
        || paulis.len() > data.len().trailing_zeros() as usize
        || paulis.iter().any(|&p| p > 3)
    {
        return Err(MethodError(
            "invalid state dimension or Pauli string".into(),
        ));
    }
    let (mut flip, mut sign, mut ys) = (0usize, 0usize, 0u32);
    for (q, &p) in paulis.iter().enumerate() {
        if p == 1 || p == 2 {
            flip |= 1usize << q;
        }
        if p == 2 || p == 3 {
            sign |= 1usize << q;
        }
        if p == 2 {
            ys += 1;
        }
    }
    if flip == 0 {
        let value: f64 = data
            .iter()
            .enumerate()
            .map(|(i, z)| {
                if (i & sign).count_ones() & 1 == 0 {
                    z.norm_sqr()
                } else {
                    -z.norm_sqr()
                }
            })
            .sum();
        return Ok(Complex64::new(value, 0.0));
    }
    let phase = match ys % 4 {
        0 => Complex64::new(1.0, 0.0),
        1 => Complex64::new(0.0, -1.0),
        2 => Complex64::new(-1.0, 0.0),
        _ => Complex64::new(0.0, 1.0),
    };
    let mut value = Complex64::new(0.0, 0.0);
    for (i, z) in data.iter().enumerate() {
        let product = z.conj() * phase * data[i ^ flip];
        value += if (i & sign).count_ones() & 1 == 0 {
            product
        } else {
            -product
        };
    }
    Ok(value)
}

#[cfg(test)]
mod performance_readout_tests {
    use super::*;

    #[test]
    fn performance_pauli_masks_match_direct_walk() {
        let mut data: Vec<Complex64> = (0..16)
            .map(|i| Complex64::new((i as f64 + 0.7).sin(), (i as f64 * 1.3).cos()))
            .collect();
        let norm = data.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
        for z in &mut data {
            *z /= norm;
        }
        for code in 0..256usize {
            let paulis: Vec<u8> = (0..4).map(|q| ((code >> (2 * q)) & 3) as u8).collect();
            let mut expected = Complex64::new(0.0, 0.0);
            for i in 0..16 {
                let mut j = i;
                let mut phase = Complex64::new(1.0, 0.0);
                for (q, p) in paulis.iter().enumerate() {
                    match p {
                        1 => j ^= 1 << q,
                        2 => {
                            j ^= 1 << q;
                            phase *= if (i >> q) & 1 == 0 {
                                Complex64::new(0.0, -1.0)
                            } else {
                                Complex64::new(0.0, 1.0)
                            };
                        }
                        3 if (i >> q) & 1 == 1 => phase = -phase,
                        _ => {}
                    }
                }
                expected += data[i].conj() * phase * data[j];
            }
            let actual = pauli_expval_bitmask(&data, &paulis).unwrap();
            assert!((actual - expected).norm() < 1e-12);
        }
        let state = StatevectorState::new(data, 4, "cpu");
        let weights: Vec<f64> = (0..16usize)
            .map(|i| (4.0 - 2.0 * i.count_ones() as f64) / 4.0)
            .collect();
        assert!(
            (state.mean_z().unwrap() - state.diagonal_expectation(&weights).unwrap()).abs() < 1e-12
        );
        assert!(state.diagonal_expectation(&[0.0]).is_err());
    }
}

// ═══════════════════════════════════════════
// MPS gradient tests
// ═══════════════════════════════════════════

#[cfg(test)]
mod mps_grad_tests {
    use super::*;

    /// Pauli-term observable `Z` on qubit 0 of an `n`-qubit register.
    fn z0(n: usize) -> Vec<PauliTerm> {
        let mut paulis = vec![0u8; n];
        paulis[0] = 3;
        vec![PauliTerm {
            paulis,
            coef: Complex64::new(1.0, 0.0),
        }]
    }

    fn var(name: &str, id: usize) -> Parameter {
        Parameter::Variable {
            name: name.into(),
            id,
        }
    }

    fn params_for(names: &[String]) -> HashMap<String, f64> {
        names
            .iter()
            .enumerate()
            .map(|(i, n)| (n.clone(), 0.3 + 0.17 * i as f64))
            .collect()
    }

    /// The MPS gradient against the exact statevector adjoint, on the gate
    /// family the shift rule is exact for.  Bond 8 exceeds 2^min(k, n-k) for
    /// n = 3, so the MPS is exact and the only error can come from the rule.
    #[test]
    fn test_mps_grad_matches_adjoint_exact_rule() {
        let _serial = eval_lock();
        let n = 3;
        let mut dag = QuantumDAG::new(n, 0);
        let mut names = Vec::new();
        let mut id = 0;
        for q in 0..n {
            let name = format!("a{q}");
            dag.add_op(OpType::Ry(var(&name, id)), &[q]);
            names.push(name);
            id += 1;
        }
        for q in 0..n - 1 {
            let name = format!("b{q}");
            dag.add_op(OpType::Ryy(var(&name, id)), &[q, q + 1]);
            names.push(name);
            id += 1;
        }
        let params = params_for(&names);
        let obs = z0(n);

        let state = MPSStateWrapper::new(dag.bind(&params).evolve_mps(8), "cpu");
        let got = state.grad(&obs, &dag, &params).unwrap();
        let want = adjoint_grad(&dag, &obs, &params).unwrap().gradients;

        assert_eq!(got.len(), want.len());
        for (g, w) in got.iter().zip(want.iter()) {
            assert!((g - w).abs() < 1e-10, "mps {g} vs adjoint {w}");
        }
    }

    /// A dag that a caller already rebound (a `cat` mps plan keeps one per
    /// handle) must still be read as symbolic: `update_parameters` overwrites
    /// `op_type` with the bound gate, so a rule lookup on `op_type` finds no
    /// variable and silently falls back to the stencil, moving the answer by
    /// ~1e-13 and doubling the evolution count.
    #[test]
    fn test_mps_grad_rule_survives_a_rebound_dag() {
        let _serial = eval_lock();
        let n = 3;
        let mut dag = QuantumDAG::new(n, 0);
        let mut names = Vec::new();
        let mut id = 0;
        for q in 0..n {
            let name = format!("a{q}");
            dag.add_op(OpType::Ry(var(&name, id)), &[q]);
            names.push(name);
            id += 1;
        }
        for q in 0..n - 1 {
            let name = format!("b{q}");
            dag.add_op(OpType::Ryy(var(&name, id)), &[q, q + 1]);
            names.push(name);
            id += 1;
        }
        let params = params_for(&names);
        let obs = z0(n);

        let symbolic = MPSStateWrapper::new(dag.bind(&params).evolve_mps(8), "cpu")
            .grad(&obs, &dag, &params)
            .unwrap();

        let mut rebound = dag.clone_dag();
        rebound.update_parameters(&params);
        let bound = MPSStateWrapper::new(rebound.clone_dag().evolve_mps(8), "cpu")
            .grad(&obs, &rebound, &params)
            .unwrap();

        for (a, b) in symbolic.iter().zip(bound.iter()) {
            assert!(
                (a - b).abs() < 1e-15,
                "rebinding changed the rule: {a} vs {b}"
            );
        }
    }

    /// Controlled rotations have generator spectrum {0, ±½}, so the shipped
    /// path is the five-point stencil; it still has to agree with the adjoint.
    #[test]
    fn test_mps_grad_controlled_rotation_stencil() {
        let _serial = eval_lock();
        let n = 2;
        let mut dag = QuantumDAG::new(n, 0);
        dag.add_op(OpType::Ry(var("a", 0)), &[0]);
        dag.add_op(OpType::Ry(var("b", 1)), &[1]);
        dag.add_op(OpType::CRz(var("c", 2)), &[0, 1]);
        let names = vec!["a".to_string(), "b".into(), "c".into()];
        let params = params_for(&names);
        let obs = z0(n);

        let state = MPSStateWrapper::new(dag.bind(&params).evolve_mps(8), "cpu");
        let got = state.grad(&obs, &dag, &params).unwrap();
        let want = adjoint_grad(&dag, &obs, &params).unwrap().gradients;

        for (g, w) in got.iter().zip(want.iter()) {
            assert!((g - w).abs() < 1e-8, "mps {g} vs adjoint {w}");
        }
    }

    /// A parameter feeding two gates adds a second harmonic (exp(±2iωθ)); the
    /// single-shift rule cannot cancel it, so this must not silently use it.
    #[test]
    fn test_mps_grad_repeated_parameter() {
        let _serial = eval_lock();
        let n = 2;
        let mut dag = QuantumDAG::new(n, 0);
        dag.add_op(OpType::Ry(var("a", 0)), &[0]);
        dag.add_op(OpType::Ry(var("a", 0)), &[1]);
        let names = vec!["a".to_string()];
        let params = params_for(&names);
        let obs = z0(n);

        let state = MPSStateWrapper::new(dag.bind(&params).evolve_mps(8), "cpu");
        let got = state.grad(&obs, &dag, &params).unwrap();
        let want = adjoint_grad(&dag, &obs, &params).unwrap().gradients;

        for (g, w) in got.iter().zip(want.iter()) {
            assert!((g - w).abs() < 1e-8, "mps {g} vs adjoint {w}");
        }
    }

    /// A truncated cut is not unitary: the discarded subspace moves with θ, so
    /// `⟨ψ(θ±π/2)|O|ψ(θ±π/2)⟩` stopped being a two-point sample of the
    /// derivative of the truncated functional.  Measured on a 5-qubit ladder
    /// at bond 1 the rule was 8.6e-02 off while the stencil matched a finite
    /// difference of the shipped forward to 4.5e-10.  The gradient must track
    /// the forward it returns with.
    #[test]
    fn test_mps_grad_follows_the_truncated_forward() {
        let _serial = eval_lock();
        // Two Ry/CNOT/Rz layers on 5 qubits.  At bond 1 the chain discards
        // half its weight and the two rules below genuinely disagree (a GHZ
        // plus local rotations does not: there the discarded subspace is a
        // fixed basis, so shift and stencil coincide to 4e-14 and the test
        // would pass for the wrong reason).
        let n = 5;
        let mut dag = QuantumDAG::new(n, 0);
        let mut names: Vec<String> = Vec::new();
        let mut id = 0usize;
        for k in 0..2 {
            for q in 0..n {
                let name = format!("t{k}_{q}");
                dag.add_op(OpType::Ry(var(&name, id)), &[q]);
                names.push(name);
                id += 1;
            }
            for q in 0..n - 1 {
                dag.add_op(OpType::CNOT, &[q, q + 1]);
            }
            for q in 0..n {
                let name = format!("u{k}_{q}");
                dag.add_op(OpType::Rz(var(&name, id)), &[q]);
                names.push(name);
                id += 1;
            }
        }
        let params = params_for(&names);
        let obs = z0(n);

        let state = MPSStateWrapper::new(dag.bind(&params).evolve_mps(1), "cpu");
        assert!(
            state.inner.max_discarded_weight > 1e-3,
            "this shape has to truncate for the test to mean anything"
        );
        let got = state.grad(&obs, &dag, &params).unwrap();

        // Central difference of the same truncated functional.
        const H: f64 = 1e-6;
        let mut fd = Vec::new();
        for name in &names {
            let mut vp = params.clone();
            let mut vm = params.clone();
            vp.insert(name.clone(), params[name] + H);
            vm.insert(name.clone(), params[name] - H);
            let fp = mps_expectation(&dag.bind(&vp), &obs, 1).0;
            let fm = mps_expectation(&dag.bind(&vm), &obs, 1).0;
            fd.push((fp - fm) / (2.0 * H));
        }

        // The two-term rule on the same truncated chain, computed by hand, so
        // the test fails if the guard ever stops rejecting it.
        let mut shift = Vec::new();
        for name in &names {
            let mut vp = params.clone();
            let mut vm = params.clone();
            vp.insert(name.clone(), params[name] + std::f64::consts::FRAC_PI_2);
            vm.insert(name.clone(), params[name] - std::f64::consts::FRAC_PI_2);
            let fp = mps_expectation(&dag.bind(&vp), &obs, 1).0;
            let fm = mps_expectation(&dag.bind(&vm), &obs, 1).0;
            shift.push(0.5 * (fp - fm));
        }

        for i in 0..names.len() {
            assert!(
                (got[i] - fd[i]).abs() < 1e-6,
                "{}: gradient {} vs finite difference {}",
                names[i],
                got[i],
                fd[i]
            );
        }
        let worst = (0..names.len())
            .map(|i| (got[i] - shift[i]).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            worst > 1e-3,
            "the truncated chain still took the two-term rule ({worst:e} from it)"
        );
    }

    /// A circuit with no symbolic parameters answers with an empty vector --
    /// the same shape the statevector lane returns.
    #[test]
    fn test_mps_grad_no_parameters() {
        let mut dag = QuantumDAG::new(1, 0);
        dag.add_op(OpType::H, &[0]);
        let state = MPSStateWrapper::new(dag.evolve_mps(4), "cpu");
        let got = state.grad(&z0(1), &dag, &HashMap::new()).unwrap();
        assert!(got.is_empty());
    }

    fn eval_calls() -> usize {
        MPS_EVAL_CALLS.load(std::sync::atomic::Ordering::Relaxed)
    }

    fn reset_calls() {
        MPS_EVAL_CALLS.store(0, std::sync::atomic::Ordering::Relaxed);
    }

    /// Serialises the tests that read [`MPS_EVAL_CALLS`].
    ///
    /// The counter has to be process-global: a rayon-parallel gradient
    /// increments it from worker threads, so the thread-local `Cell` it
    /// replaced could not see them.  The price is that two tests running side
    /// by side both add to it, and a tally read between `reset_calls` and the
    /// assertion then picks up the neighbour's evolutions.  Holding this lock
    /// for the whole measure window is what keeps those reads exact; every
    /// test here that reaches `mps_expectation` takes it, and the module runs
    /// in 0.12 s, so serialising it costs nothing.
    static MPS_EVAL_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn eval_lock() -> std::sync::MutexGuard<'static, ()> {
        MPS_EVAL_LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// The two-layer Ry/CNOT/Rz ladder, one parameter per rotation.
    fn ladder(n: usize, layers: usize) -> (QuantumDAG, Vec<String>) {
        let mut dag = QuantumDAG::new(n, 0);
        let mut names: Vec<String> = Vec::new();
        let mut id = 0usize;
        for k in 0..layers {
            for q in 0..n {
                let name = format!("t{k}_{q}");
                dag.add_op(OpType::Ry(var(&name, id)), &[q]);
                names.push(name);
                id += 1;
            }
            for q in 0..n - 1 {
                dag.add_op(OpType::CNOT, &[q, q + 1]);
            }
            for q in 0..n {
                let name = format!("u{k}_{q}");
                dag.add_op(OpType::Rz(var(&name, id)), &[q]);
                names.push(name);
                id += 1;
            }
        }
        (dag, names)
    }

    /// A binding cap must not double the price.  The two-term rule was already
    /// two evolutions per parameter; the truncated fallback used to cost four,
    /// which is the one shape where this lane lost to Qiskit's fixed
    /// two-evolution rule.  Measured 4e-9 against a fine reference on n=12 at
    /// bond 2-4 before this was switched.
    #[test]
    fn test_mps_grad_truncated_pays_two_evolutions_per_parameter() {
        let _serial = eval_lock();
        let (dag, names) = ladder(5, 2);
        let params = params_for(&names);
        let obs = z0(5);
        let state = MPSStateWrapper::new(dag.bind(&params).evolve_mps(1), "cpu");
        assert!(
            state.inner.max_discarded_weight > 1e-3,
            "this shape has to truncate for the test to mean anything"
        );

        reset_calls();
        let got = state.grad(&obs, &dag, &params).unwrap();
        assert_eq!(
            eval_calls(),
            2 * names.len(),
            "a truncated gradient must cost 2 evolutions per parameter"
        );

        // ... and still be a derivative of the truncated forward: central
        // difference at h = 1e-6, whose own round-off is O(1e-10).
        for (i, name) in names.iter().enumerate() {
            const H: f64 = 1e-6;
            let mut vp = params.clone();
            let mut vm = params.clone();
            vp.insert(name.clone(), params[name] + H);
            vm.insert(name.clone(), params[name] - H);
            let fp = mps_expectation(&dag.bind(&vp), &obs, 1).0;
            let fm = mps_expectation(&dag.bind(&vm), &obs, 1).0;
            let fd = (fp - fm) / (2.0 * H);
            assert!(
                (got[i] - fd).abs() < 1e-6,
                "{}: gradient {} vs finite difference {}",
                name,
                got[i],
                fd
            );
        }
    }

    /// At a bond large enough to be exact the two-term rule runs, so the cost
    /// is the same 2P -- the cap is the only thing that can change it.
    #[test]
    fn test_mps_grad_exact_pays_two_evolutions_per_parameter() {
        let _serial = eval_lock();
        let (dag, names) = ladder(5, 2);
        let params = params_for(&names);
        let state = MPSStateWrapper::new(dag.bind(&params).evolve_mps(8), "cpu");
        assert!(state.inner.max_discarded_weight <= MPS_SHIFT_TRUNC_EPS);

        reset_calls();
        let _ = state.grad(&z0(5), &dag, &params).unwrap();
        assert_eq!(eval_calls(), 2 * names.len());
    }

    /// Controlled rotations have no exact two-term rule and do not truncate,
    /// so they keep the five-point rule: the halved cost is only for a
    /// truncating run, where two more samples buy nothing.
    #[test]
    fn test_mps_grad_without_a_rule_keeps_four_evolutions() {
        let _serial = eval_lock();
        let mut dag = QuantumDAG::new(2, 0);
        dag.add_op(OpType::Ry(var("a", 0)), &[0]);
        dag.add_op(OpType::Ry(var("b", 1)), &[1]);
        dag.add_op(OpType::CRz(var("c", 2)), &[0, 1]);
        let names = vec!["a".to_string(), "b".into(), "c".into()];
        let params = params_for(&names);
        let state = MPSStateWrapper::new(dag.bind(&params).evolve_mps(8), "cpu");

        reset_calls();
        let _ = state.grad(&z0(2), &dag, &params).unwrap();
        // a and b take the rule (2 each); only c has no rule (4).
        assert_eq!(eval_calls(), 2 * (names.len() - 1) + 4);
    }
}

// ═══════════════════════════════════════════
// MPS readout tests
// ═══════════════════════════════════════════

#[cfg(test)]
mod mps_readout_tests {
    use super::*;

    /// Ry layer, CNOT ladder, Rz layer: entangled but exactly representable
    /// at bond 16 for n = 4, so the MPS and the statevector must agree to
    /// round-off.
    fn ladder(n: usize) -> (QuantumDAG, HashMap<String, f64>) {
        let mut dag = QuantumDAG::new(n, 0);
        let mut params = HashMap::new();
        let mut id = 0;
        for q in 0..n {
            let name = format!("a{q}");
            dag.add_op(
                OpType::Ry(Parameter::Variable {
                    name: name.clone(),
                    id,
                }),
                &[q],
            );
            params.insert(name, 0.3 + 0.17 * q as f64);
            id += 1;
        }
        for q in 0..n.saturating_sub(1) {
            dag.add_op(OpType::CNOT, &[q, q + 1]);
        }
        for q in 0..n {
            let name = format!("b{q}");
            dag.add_op(
                OpType::Rz(Parameter::Variable {
                    name: name.clone(),
                    id,
                }),
                &[q],
            );
            params.insert(name, 0.9 - 0.11 * q as f64);
            id += 1;
        }
        (dag, params)
    }

    /// `ρ[ket, bra]` of a `partial_trace` result, with the type checked once.
    fn dm_at(state: &dyn QuantumStateImpl, ket: usize, bra: usize) -> Complex64 {
        state
            .as_any()
            .downcast_ref::<DensityMatrixStateWrapper>()
            .expect("partial_trace must return a density matrix")
            .inner
            .at(ket, bra)
    }

    #[test]
    fn test_mps_probabilities_match_statevector() {
        let n = 4;
        let (dag, params) = ladder(n);
        let bound = dag.bind(&params);
        let sv = StatevectorState::new(bound.simulate(), n, "cpu");
        let mps = MPSStateWrapper::new(bound.evolve_mps(16), "cpu");

        let p_mps = mps.probabilities().unwrap();
        let p_sv = sv.probabilities().unwrap();
        assert_eq!(p_mps.len(), 1 << n);
        for (a, b) in p_mps.iter().zip(p_sv.iter()) {
            assert!((a - b).abs() < 1e-12, "probabilities {a} vs {b}");
        }
        let total: f64 = p_mps.iter().sum();
        assert!((total - 1.0).abs() < 1e-12, "table sums to {total}");
    }

    #[test]
    fn test_mps_entropy_purity_whole_state() {
        let (dag, params) = ladder(3);
        let mps = MPSStateWrapper::new(dag.bind(&params).evolve_mps(8), "cpu");
        assert!((mps.purity().unwrap() - 1.0).abs() < 1e-14);
        assert!(mps.entropy().unwrap().abs() < 1e-14);
    }

    #[test]
    fn test_mps_partial_trace_matches_statevector() {
        let n = 4;
        let (dag, params) = ladder(n);
        let bound = dag.bind(&params);
        let sv = StatevectorState::new(bound.simulate(), n, "cpu");
        let mps = MPSStateWrapper::new(bound.evolve_mps(16), "cpu");

        // One wire, and two wires given out of order: bit j of the result is
        // keep[j], so the out-of-order case also checks the bit mapping.
        for keep in [vec![0usize], vec![2], vec![2, 0]] {
            let a = mps.partial_trace(&keep).unwrap();
            let b = sv.partial_trace(&keep).unwrap();
            let dim = 1usize << keep.len();
            for ket in 0..dim {
                for bra in 0..dim {
                    let (x, y) = (dm_at(a.as_ref(), ket, bra), dm_at(b.as_ref(), ket, bra));
                    assert!(
                        (x - y).norm() < 1e-12,
                        "keep {keep:?} rho[{ket},{bra}] {x} vs {y}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_mps_partial_trace_entropy_and_mutual_info() {
        let mut dag = QuantumDAG::new(2, 0);
        dag.add_op(OpType::H, &[0]);
        dag.add_op(OpType::CNOT, &[0, 1]);
        let mps = MPSStateWrapper::new(dag.evolve_mps(4), "cpu");

        let s = mps.partial_trace(&[0]).unwrap().entropy().unwrap();
        assert!((s - std::f64::consts::LN_2).abs() < 1e-12, "S(A) = {s}");
        let mi = mps.mutual_info(&[0], &[1]).unwrap();
        assert!(
            (mi - 2.0 * std::f64::consts::LN_2).abs() < 1e-10,
            "I(A;B) = {mi}"
        );
    }

    #[test]
    fn test_mps_fidelity_against_itself_and_statevector() {
        let n = 3;
        let (dag, params) = ladder(n);
        let bound = dag.bind(&params);
        let a = MPSStateWrapper::new(bound.evolve_mps(8), "cpu");
        let b = MPSStateWrapper::new(bound.evolve_mps(8), "cpu");
        assert!((a.fidelity(&b).unwrap() - 1.0).abs() < 1e-12);

        let sv = StatevectorState::new(bound.simulate(), n, "cpu");
        assert!((a.fidelity(&sv).unwrap() - 1.0).abs() < 1e-12);

        // A different parameter point must land strictly below 1.
        let mut other: HashMap<String, f64> = params.clone();
        for (i, v) in other.values_mut().enumerate() {
            *v += 0.1 * (i as f64 + 1.0);
        }
        let c = MPSStateWrapper::new(dag.bind(&other).evolve_mps(8), "cpu");
        let f = a.fidelity(&c).unwrap();
        assert!((0.0..0.999).contains(&f), "fidelity {f}");
    }

    /// Truncation leaves the chain sub-normalized; every readout still reports
    /// the normalized physical state.
    #[test]
    fn test_mps_readout_normalizes_truncated_chain() {
        let n = 4;
        let (dag, params) = ladder(n);
        let mps = MPSStateWrapper::new(dag.bind(&params).evolve_mps(1), "cpu");
        let total: f64 = mps.probabilities().unwrap().iter().sum();
        assert!((total - 1.0).abs() < 1e-12, "table sums to {total}");
        let tr: Complex64 = {
            let rho = mps.partial_trace(&[0, 1]).unwrap();
            let dm = rho
                .as_any()
                .downcast_ref::<DensityMatrixStateWrapper>()
                .unwrap();
            (0..4).map(|i| dm.inner.at(i, i)).sum()
        };
        assert!((tr.re - 1.0).abs() < 1e-12, "trace = {tr}");
    }
}
