use std::any::Any;
use std::collections::HashMap;

use num_complex::Complex64;

use crate::adjoint::{adjoint_grad, PauliTerm};
use crate::dag::QuantumDAG;
use crate::dm::DensityMatrixState;
use crate::mps::MPSState;
use crate::stabilizer::StabilizerTableau;

#[derive(Debug, Clone)]
pub struct MethodError(pub String);

impl std::fmt::Display for MethodError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for MethodError {}

pub trait QuantumStateImpl: Send + Sync {
    fn expectation(&self, observable: &[PauliTerm]) -> Result<f64, MethodError>;
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
    fn partial_trace(&self, keep_qubits: &[usize]) -> Result<Box<dyn QuantumStateImpl>, MethodError>;
    fn n_qubits(&self) -> usize;
    fn method_name(&self) -> &str;
    fn device_name(&self) -> &str;
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

    fn compute_expval(&self, observable: &[PauliTerm]) -> f64 {
        let n = self.num_qubits;
        let dim = self.data.len();
        let mut total = Complex64::new(0.0, 0.0);

        for term in observable {
            let mut expval = Complex64::new(0.0, 0.0);
            for i in 0..dim {
                let mut phase = Complex64::new(1.0, 0.0);
                let mut target_idx = i;

                for (q, &pauli_op) in term.paulis.iter().enumerate() {
                    let bit_pos = if q < n { n - 1 - q } else { q };
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
                        3 => {
                            if (i >> bit_pos) & 1 == 1 {
                                phase *= -1.0;
                            }
                        }
                        _ => {}
                    }
                }
                expval += self.data[i].conj() * phase * self.data[target_idx];
            }
            total += term.coef * expval;
        }
        total.re
    }
}

impl QuantumStateImpl for StatevectorState {
    fn expectation(&self, observable: &[PauliTerm]) -> Result<f64, MethodError> {
        Ok(self.compute_expval(observable))
    }

    fn sample(&self, shots: usize, seed: u64) -> Result<HashMap<String, usize>, MethodError> {
        if shots == 0 {
            return Ok(HashMap::new());
        }
        let n = self.num_qubits;
        let dim = self.data.len();
        let probs: Vec<f64> = self.data.iter().map(|c| c.re * c.re + c.im * c.im).collect();

        use rand::SeedableRng;
        use rand::Rng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let mut counts: HashMap<String, usize> = HashMap::new();

        let mut cumulative = vec![0.0f64; dim + 1];
        for i in 0..dim {
            cumulative[i + 1] = cumulative[i] + probs[i];
        }
        let total = cumulative[dim];

        for _ in 0..shots {
            let r: f64 = rng.gen::<f64>() * total;
            let idx = match cumulative.binary_search_by(|v| v.partial_cmp(&r).unwrap_or(std::cmp::Ordering::Less)) {
                Ok(i) => i.min(dim - 1),
                Err(i) => (i - 1).min(dim - 1),
            };
            let bitstring: String = (0..n).rev().map(|q| if (idx >> q) & 1 == 1 { '1' } else { '0' }).collect();
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
        let result = adjoint_grad(dag, observable, param_values);
        Ok(result.gradients)
    }

    fn entropy(&self) -> Result<f64, MethodError> {
        Ok(0.0) // pure state
    }

    fn purity(&self) -> Result<f64, MethodError> {
        Ok(1.0) // pure state
    }

    fn fidelity(&self, other: &dyn QuantumStateImpl) -> Result<f64, MethodError> {
        let other_sv = other.as_any().downcast_ref::<StatevectorState>()
            .ok_or_else(|| MethodError("fidelity() requires both states to be the same type".into()))?;
        if self.num_qubits != other_sv.num_qubits {
            return Err(MethodError("fidelity() requires same number of qubits".into()));
        }
        let ip: Complex64 = self.data.iter().zip(other_sv.data.iter())
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
            let orig_val = *param_values.get(name.as_str())
                .ok_or_else(|| MethodError(format!("parameter '{}' not found in param_values", name)))?;

            let mut params_plus = param_values.clone();
            params_plus.insert(name.clone(), orig_val + eps);
            let dag_plus = dag.bind(&params_plus);
            let psi_plus = dag_plus.simulate();

            let mut params_minus = param_values.clone();
            params_minus.insert(name.clone(), orig_val - eps);
            let dag_minus = dag.bind(&params_minus);
            let psi_minus = dag_minus.simulate();

            let deriv: Vec<Complex64> = psi_plus.iter().zip(psi_minus.iter())
                .map(|(p, m)| (*p - *m) / (2.0 * eps))
                .collect();
            d_psi.push(deriv);
        }

        let mut qfim = vec![vec![0.0f64; n_params]; n_params];
        for i in 0..n_params {
            for j in i..n_params {
                let overlap: Complex64 = d_psi[i].iter().zip(d_psi[j].iter())
                    .map(|(a, b)| a.conj() * b)
                    .sum();
                let val = 4.0 * overlap.re;
                qfim[i][j] = val;
                qfim[j][i] = val;
            }
        }
        Ok(qfim)
    }

    fn to_vec(&self) -> Result<Vec<Complex64>, MethodError> {
        Ok(self.data.clone())
    }

    fn probabilities(&self) -> Result<Vec<f64>, MethodError> {
        Ok(self.data.iter().map(|c| c.re * c.re + c.im * c.im).collect())
    }

    fn partial_trace(&self, keep_qubits: &[usize]) -> Result<Box<dyn QuantumStateImpl>, MethodError> {
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
                        if (i_keep >> k) & 1 == 1 { idx_i |= 1 << q; }
                        if (j_keep >> k) & 1 == 1 { idx_j |= 1 << q; }
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
                dm.data[ket | (bra << n_keep)] = rho_matrix[ket][bra];
            }
        }

        Ok(Box::new(DensityMatrixStateWrapper {
            inner: dm,
            device: self.device.clone(),
        }))
    }

    fn n_qubits(&self) -> usize { self.num_qubits }
    fn method_name(&self) -> &str { "statevector" }
    fn device_name(&self) -> &str { &self.device }
    fn as_any(&self) -> &dyn Any { self }
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
        Self { inner: dm, device: device.to_string() }
    }

    fn to_matrix(&self) -> Vec<Vec<Complex64>> {
        let n = self.inner.n_qubits;
        let dim = 1 << n;
        let mut rho = vec![vec![Complex64::new(0.0, 0.0); dim]; dim];
        for ket in 0..dim {
            for bra in 0..dim {
                rho[ket][bra] = self.inner.data[ket | (bra << n)];
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
                    let bit_pos = if q < n { n - 1 - q } else { q };
                    match pauli_op {
                        1 => { target_idx ^= 1 << bit_pos; }
                        2 => {
                            target_idx ^= 1 << bit_pos;
                            if (i >> bit_pos) & 1 == 0 {
                                phase *= Complex64::new(0.0, -1.0);
                            } else {
                                phase *= Complex64::new(0.0, 1.0);
                            }
                        }
                        3 => {
                            if (i >> bit_pos) & 1 == 1 { phase *= -1.0; }
                        }
                        _ => {}
                    }
                }
                total += term.coef * phase * rho[target_idx][i];
            }
        }
        Ok(total.re)
    }

    fn sample(&self, shots: usize, seed: u64) -> Result<HashMap<String, usize>, MethodError> {
        if shots == 0 { return Ok(HashMap::new()); }
        let n = self.inner.n_qubits;
        let dim = 1 << n;
        let probs: Vec<f64> = (0..dim).map(|i| self.inner.data[i | (i << n)].re).collect();

        use rand::SeedableRng;
        use rand::Rng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let mut counts: HashMap<String, usize> = HashMap::new();

        let mut cumulative = vec![0.0f64; dim + 1];
        for i in 0..dim {
            cumulative[i + 1] = cumulative[i] + probs[i].max(0.0);
        }
        let total = cumulative[dim];

        for _ in 0..shots {
            let r: f64 = rng.gen::<f64>() * total;
            let idx = match cumulative.binary_search_by(|v| v.partial_cmp(&r).unwrap_or(std::cmp::Ordering::Less)) {
                Ok(i) => i.min(dim - 1),
                Err(i) => (i - 1).min(dim - 1),
            };
            let bitstring: String = (0..n).rev().map(|q| if (idx >> q) & 1 == 1 { '1' } else { '0' }).collect();
            *counts.entry(bitstring).or_insert(0) += 1;
        }
        Ok(counts)
    }

    fn grad(&self, _observable: &[PauliTerm], _dag: &QuantumDAG, _param_values: &HashMap<String, f64>) -> Result<Vec<f64>, MethodError> {
        Err(MethodError("grad() not supported for density_matrix method".into()))
    }

    fn entropy(&self) -> Result<f64, MethodError> {
        let n = self.inner.n_qubits;
        let dim = 1 << n;
        let mut rho = nalgebra::DMatrix::<Complex64>::zeros(dim, dim);
        for ket in 0..dim {
            for bra in 0..dim {
                rho[(ket, bra)] = self.inner.data[ket | (bra << n)];
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
                rho[(ket, bra)] = self.inner.data[ket | (bra << n)];
            }
        }
        Ok((&rho * &rho).trace().re)
    }

    fn fidelity(&self, other: &dyn QuantumStateImpl) -> Result<f64, MethodError> {
        let other_dm = other.as_any().downcast_ref::<DensityMatrixStateWrapper>()
            .ok_or_else(|| MethodError("fidelity() between density_matrix states requires both to be density_matrix".into()))?;
        let n = self.inner.n_qubits;
        if n != other_dm.inner.n_qubits {
            return Err(MethodError("fidelity() requires same number of qubits".into()));
        }
        let dim = 1 << n;

        let mut rho = nalgebra::DMatrix::<Complex64>::zeros(dim, dim);
        let mut sigma = nalgebra::DMatrix::<Complex64>::zeros(dim, dim);
        for ket in 0..dim {
            for bra in 0..dim {
                rho[(ket, bra)] = self.inner.data[ket | (bra << n)];
                sigma[(ket, bra)] = other_dm.inner.data[ket | (bra << n)];
            }
        }

        Ok((&rho * &sigma).trace().re.abs())
    }

    fn qfim(&self, _dag: &QuantumDAG, _param_values: &HashMap<String, f64>) -> Result<Vec<Vec<f64>>, MethodError> {
        Err(MethodError("qfim() not supported for density_matrix method".into()))
    }

    fn to_vec(&self) -> Result<Vec<Complex64>, MethodError> {
        let n = self.inner.n_qubits;
        let dim = 1 << n;
        let mut flat = Vec::with_capacity(dim * dim);
        for ket in 0..dim {
            for bra in 0..dim {
                flat.push(self.inner.data[ket | (bra << n)]);
            }
        }
        Ok(flat)
    }

    fn probabilities(&self) -> Result<Vec<f64>, MethodError> {
        let n = self.inner.n_qubits;
        let dim = 1 << n;
        Ok((0..dim).map(|i| self.inner.data[i | (i << n)].re.max(0.0)).collect())
    }

    fn partial_trace(&self, keep_qubits: &[usize]) -> Result<Box<dyn QuantumStateImpl>, MethodError> {
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
                        if (i_keep >> k) & 1 == 1 { idx_i |= 1 << q; }
                        if (j_keep >> k) & 1 == 1 { idx_j |= 1 << q; }
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
                dm.data[ket | (bra << n_keep)] = rho_reduced[ket][bra];
            }
        }

        Ok(Box::new(DensityMatrixStateWrapper {
            inner: dm,
            device: self.device.clone(),
        }))
    }

    fn n_qubits(&self) -> usize { self.inner.n_qubits }
    fn method_name(&self) -> &str { "density_matrix" }
    fn device_name(&self) -> &str { &self.device }
    fn as_any(&self) -> &dyn Any { self }
}

// ═══════════════════════════════════════════════════════════
// MPS State Wrapper
// ═══════════════════════════════════════════════════════════

pub struct MPSStateWrapper {
    pub inner: MPSState,
    pub device: String,
}

impl MPSStateWrapper {
    pub fn new(mps: MPSState, device: &str) -> Self {
        Self { inner: mps, device: device.to_string() }
    }
}

impl QuantumStateImpl for MPSStateWrapper {
    fn expectation(&self, observable: &[PauliTerm]) -> Result<f64, MethodError> {
        let mut total = 0.0;
        for term in observable {
            let pauli_u8: Vec<u8> = term.paulis.clone();
            let z = self.inner.pauli_expval(&pauli_u8);
            total += (term.coef * z).re;
        }
        Ok(total)
    }

    fn sample(&self, shots: usize, seed: u64) -> Result<HashMap<String, usize>, MethodError> {
        Ok(self.inner.sample(shots, seed))
    }

    fn grad(&self, _observable: &[PauliTerm], _dag: &QuantumDAG, _param_values: &HashMap<String, f64>) -> Result<Vec<f64>, MethodError> {
        Err(MethodError("grad() not supported for mps method".into()))
    }

    fn entropy(&self) -> Result<f64, MethodError> {
        Err(MethodError("entropy() not supported for mps method".into()))
    }

    fn purity(&self) -> Result<f64, MethodError> {
        Err(MethodError("purity() not supported for mps method".into()))
    }

    fn fidelity(&self, _other: &dyn QuantumStateImpl) -> Result<f64, MethodError> {
        Err(MethodError("fidelity() not supported for mps method".into()))
    }

    fn qfim(&self, _dag: &QuantumDAG, _param_values: &HashMap<String, f64>) -> Result<Vec<Vec<f64>>, MethodError> {
        Err(MethodError("qfim() not supported for mps method".into()))
    }

    fn to_vec(&self) -> Result<Vec<Complex64>, MethodError> {
        Ok(self.inner.to_statevector())
    }

    fn probabilities(&self) -> Result<Vec<f64>, MethodError> {
        Err(MethodError("probabilities() not supported for mps method".into()))
    }

    fn partial_trace(&self, _keep_qubits: &[usize]) -> Result<Box<dyn QuantumStateImpl>, MethodError> {
        Err(MethodError("partial_trace() not supported for mps method".into()))
    }

    fn n_qubits(&self) -> usize { self.inner.n_qubits }
    fn method_name(&self) -> &str { "mps" }
    fn device_name(&self) -> &str { &self.device }
    fn as_any(&self) -> &dyn Any { self }
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
        Self { inner: tab, device: device.to_string() }
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
                if q >= n { break; }
                let bit_pos = n - 1 - q;
                match p {
                    1 => { px[bit_pos] = 1; }       // X
                    2 => { px[bit_pos] = 1; pz[bit_pos] = 1; } // Y = iXZ
                    3 => { pz[bit_pos] = 1; }       // Z
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

    fn grad(&self, _observable: &[PauliTerm], _dag: &QuantumDAG, _param_values: &HashMap<String, f64>) -> Result<Vec<f64>, MethodError> {
        Err(MethodError("grad() not supported for stabilizer method".into()))
    }

    fn entropy(&self) -> Result<f64, MethodError> {
        Err(MethodError("entropy() not supported for stabilizer method".into()))
    }

    fn purity(&self) -> Result<f64, MethodError> {
        Err(MethodError("purity() not supported for stabilizer method".into()))
    }

    fn fidelity(&self, _other: &dyn QuantumStateImpl) -> Result<f64, MethodError> {
        Err(MethodError("fidelity() not supported for stabilizer method".into()))
    }

    fn qfim(&self, _dag: &QuantumDAG, _param_values: &HashMap<String, f64>) -> Result<Vec<Vec<f64>>, MethodError> {
        Err(MethodError("qfim() not supported for stabilizer method".into()))
    }

    fn to_vec(&self) -> Result<Vec<Complex64>, MethodError> {
        Err(MethodError("to_vec() not supported for stabilizer method (exponential memory)".into()))
    }

    fn probabilities(&self) -> Result<Vec<f64>, MethodError> {
        Err(MethodError("probabilities() not supported for stabilizer method".into()))
    }

    fn partial_trace(&self, _keep_qubits: &[usize]) -> Result<Box<dyn QuantumStateImpl>, MethodError> {
        Err(MethodError("partial_trace() not supported for stabilizer method".into()))
    }

    fn n_qubits(&self) -> usize { self.inner.n }
    fn method_name(&self) -> &str { "stabilizer" }
    fn device_name(&self) -> &str { &self.device }
    fn as_any(&self) -> &dyn Any { self }
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
        assert!((ev - 1.0).abs() < 1e-10, "ZZ on Bell state should be 1.0, got {}", ev);
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
        assert!((fid - 1.0).abs() < 1e-10, "self-fidelity should be 1.0, got {}", fid);
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
        assert!((purity - 0.5).abs() < 1e-10, "partial trace of Bell state should give mixed state with purity 0.5, got {}", purity);
    }

    #[test]
    fn test_stabilizer_method_errors() {
        let tab = StabilizerTableau::new(2);
        let state = StabilizerStateWrapper::new(tab, "cpu");
        assert!(state.grad(&[], &QuantumDAG::new(2, 0), &HashMap::new()).is_err());
        assert!(state.entropy().is_err());
        assert!(state.purity().is_err());
        assert!(state.qfim(&QuantumDAG::new(2, 0), &HashMap::new()).is_err());
    }
}
