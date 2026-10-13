//! sf-cat adjoint engine (Lever 3): a compile-once reverse-mode gradient
//! handle for parameterised small-qubit circuits.
//!
//! Numerical semantics are a faithful port of `sf_ir::adjoint::adjoint_grad`
//! (same gate-matrix formulas, same pair-pass application order, same
//! generator dots, same gradient accumulation), so gradients agree with the
//! house engine to floating-point tolerance — pinned by a parity test against
//! `sf_ir::adjoint::adjoint_grad` below.  What the handle removes is sf-ir's
//! per-call fixed cost: the topological walk, `OpType` matching and
//! `bind_params`, the `HashMap` parameter lookup, the per-gate `to_matrix()`
//! heap allocations, and the per-call buffer take/put churn.  All of that is
//! resolved once in [`AdjProgram`] construction; a per-call [`grad`] is one
//! matrix-fill pass plus one forward and one backward sweep over flat arrays
//! with resident pages.
//!
//! Scope (explicit): RY/RZ gates with variable parameters and CNOT gates —
//! the benchmark ansatz — with arbitrary [`PauliTerm`] observables.  Any
//! other gate is rejected at compile time instead of being silently applied.
//!
//! [`grad`]: AdjProgram::grad

use std::collections::HashMap;

use num_complex::Complex64;
use sf_ir::adjoint::PauliTerm;
use sf_ir::dag::QuantumDAG;
use sf_ir::ops::{OpType, Parameter};
use sf_ir::simd::pair_pass;

/// Upper bound on the parameter-state cache (`n_param_gates * 2^n` complex).
/// The handle targets small-n workloads; above 64 Mi complex (1 GiB) the
/// caller should stay on `sf_ir::adjoint` (which streams over the cap).
const CACHE_ELEM_CAP: usize = 1 << 26;

/// One flat gate of the compiled program.  `slot` indexes the parameter
/// vector (the DAG's `parameter_names()` order for [`AdjProgram::from_dag`]).
#[derive(Clone, Copy, Debug)]
enum AdjGate {
    Ry { q: u32, slot: u32 },
    Rz { q: u32, slot: u32 },
    Cx { ctl: u32, tgt: u32 },
}

impl AdjGate {
    #[inline]
    fn is_param(&self) -> bool {
        matches!(self, AdjGate::Ry { .. } | AdjGate::Rz { .. })
    }
}

/// Compile-once adjoint program: flat gate list + preallocated, page-resident
/// state buffers.  Built once per circuit; `grad` / `expectation` may be
/// called any number of times with fresh parameter values.
pub struct AdjProgram {
    n: usize,
    dim: usize,
    p: usize,
    gates: Vec<AdjGate>,
    n_param_gates: usize,
    /// Forward unitary per 1q gate (garbage on `Cx` entries), filled per call.
    fwd: Vec<[[Complex64; 2]; 2]>,
    /// Dagger of `fwd` (back-walk), filled per call.
    dg: Vec<[[Complex64; 2]; 2]>,
    /// Forward state; starts at |0…0⟩ each call.
    psi: Vec<Complex64>,
    /// Post-gate state per parameterised gate, in gate order.
    cache: Vec<Complex64>,
    /// phi = O|psi_final> plus the term scratch (observable build).
    phi: Vec<Complex64>,
    term_state: Vec<Complex64>,
    grad: Vec<f64>,
}

impl AdjProgram {
    /// The benchmark ansatz: per layer, RY then RZ on every qubit (slots
    /// assigned in that order), then a CNOT ladder `(i, i+1)`.  `P = 2*n*layers`.
    pub fn from_ladder(n: usize, layers: usize) -> Self {
        let mut gates = Vec::with_capacity(layers * (3 * n));
        let mut slot = 0u32;
        for _ in 0..layers {
            for i in 0..n {
                gates.push(AdjGate::Ry { q: i as u32, slot });
                slot += 1;
                gates.push(AdjGate::Rz { q: i as u32, slot });
                slot += 1;
            }
            for i in 0..n.saturating_sub(1) {
                gates.push(AdjGate::Cx {
                    ctl: i as u32,
                    tgt: i as u32 + 1,
                });
            }
        }
        Self::new(n, gates, slot as usize)
    }

    /// Lower a `QuantumDAG` of RY/RZ (variable parameters) + CNOT gates.
    /// Parameter slots follow the DAG's `parameter_names()` order, so the
    /// gradient vector is aligned with `sf_ir::adjoint::adjoint_grad`.
    pub fn from_dag(dag: &QuantumDAG) -> Result<Self, String> {
        let names = dag.parameter_names();
        let idx: HashMap<&str, u32> = names
            .iter()
            .enumerate()
            .map(|(i, s)| (s.as_str(), i as u32))
            .collect();
        let mut gates: Vec<AdjGate> = Vec::new();
        for &node in dag.topological_order().iter() {
            let op = &dag.graph()[node];
            if op.op_type.is_boundary()
                || op.op_type == OpType::Barrier
                || op.op_type.is_measurement()
            {
                continue;
            }
            match &op.op_type {
                OpType::Ry(param) => gates.push(AdjGate::Ry {
                    q: one_qubit(&op.qubits, "Ry")?,
                    slot: var_slot(param, &idx)?,
                }),
                OpType::Rz(param) => gates.push(AdjGate::Rz {
                    q: one_qubit(&op.qubits, "Rz")?,
                    slot: var_slot(param, &idx)?,
                }),
                OpType::CNOT => {
                    if op.qubits.len() != 2 {
                        return Err("cat adj: CNOT needs two qubits".to_string());
                    }
                    gates.push(AdjGate::Cx {
                        ctl: op.qubits[0] as u32,
                        tgt: op.qubits[1] as u32,
                    });
                }
                other => {
                    return Err(format!(
                        "cat adj: unsupported gate {} (only RY/RZ with variable \
                         parameters and CNOT are supported)",
                        other.name()
                    ))
                }
            }
        }
        let p = names.len();
        let n = dag.n_qubits;
        if n >= 63 {
            return Err("cat adj: n must be in 1..=62".to_string());
        }
        let n_param_gates = gates.iter().filter(|g| g.is_param()).count();
        if n_param_gates * (1usize << n) > CACHE_ELEM_CAP {
            return Err(format!(
                "cat adj: parameter-state cache would exceed the {} Mi-complex \
                 handle cap; use sf_ir::adjoint for this size",
                CACHE_ELEM_CAP >> 20
            ));
        }
        Ok(Self::new(n, gates, p))
    }

    fn new(n: usize, gates: Vec<AdjGate>, p: usize) -> Self {
        assert!((1..63).contains(&n), "cat adj: n must be in 1..=62");
        let dim = 1usize << n;
        for g in &gates {
            match *g {
                AdjGate::Ry { q, slot } | AdjGate::Rz { q, slot } => {
                    assert!((q as usize) < n, "cat adj: qubit out of range");
                    assert!((slot as usize) < p, "cat adj: slot out of range");
                }
                AdjGate::Cx { ctl, tgt } => {
                    assert!(
                        (ctl as usize) < n && (tgt as usize) < n && ctl != tgt,
                        "cat adj: bad CNOT pair"
                    );
                }
            }
        }
        let n_param_gates = gates.iter().filter(|g| g.is_param()).count();
        assert!(
            n_param_gates * dim <= CACHE_ELEM_CAP,
            "cat adj: parameter-state cache exceeds the handle cap"
        );
        let zero = Complex64::new(0.0, 0.0);
        let n_gates = gates.len();
        Self {
            n,
            dim,
            p,
            gates,
            n_param_gates,
            fwd: vec![[[zero; 2]; 2]; n_gates],
            dg: vec![[[zero; 2]; 2]; n_gates],
            psi: vec![zero; dim],
            cache: vec![zero; n_param_gates * dim],
            phi: vec![zero; dim],
            term_state: vec![zero; dim],
            grad: vec![0.0; p],
        }
    }

    pub fn n_qubits(&self) -> usize {
        self.n
    }

    /// Number of parameters (gradient length).
    pub fn num_params(&self) -> usize {
        self.p
    }

    /// Fill `fwd`/`dg` for every parameterised gate from `theta`.
    /// Matrix formulas are verbatim sf-ir `OpType::to_matrix` (Ry/Rz).
    fn fill_matrices(&mut self, theta: &[f64]) {
        let zero = Complex64::new(0.0, 0.0);
        for k in 0..self.gates.len() {
            let m = match self.gates[k] {
                AdjGate::Ry { slot, .. } => {
                    let t = theta[slot as usize];
                    let c = Complex64::new((t / 2.0).cos(), 0.0);
                    let s = Complex64::new((t / 2.0).sin(), 0.0);
                    [[c, -s], [s, c]]
                }
                AdjGate::Rz { slot, .. } => {
                    let t = theta[slot as usize];
                    let em = Complex64::from_polar(1.0, -t / 2.0);
                    let ep = Complex64::from_polar(1.0, t / 2.0);
                    [[em, zero], [zero, ep]]
                }
                AdjGate::Cx { .. } => continue,
            };
            self.fwd[k] = m;
            self.dg[k] = [
                [m[0][0].conj(), m[1][0].conj()],
                [m[0][1].conj(), m[1][1].conj()],
            ];
        }
    }

    /// Forward pass: |0…0⟩ through every gate, caching the post-state of
    /// each parameterised gate (sf-ir's cached variant).
    fn forward(&mut self) {
        for x in self.psi.iter_mut() {
            *x = Complex64::new(0.0, 0.0);
        }
        self.psi[0] = Complex64::new(1.0, 0.0);
        let dim = self.dim;
        let mut slot = 0usize;
        for k in 0..self.gates.len() {
            match self.gates[k] {
                AdjGate::Ry { q, .. } | AdjGate::Rz { q, .. } => {
                    let m = self.fwd[k];
                    pair_pass(&mut self.psi, 1usize << q, m);
                }
                AdjGate::Cx { ctl, tgt } => {
                    cx_apply(&mut self.psi, dim, ctl as usize, tgt as usize);
                }
            }
            if self.gates[k].is_param() {
                self.cache[slot * dim..(slot + 1) * dim].copy_from_slice(&self.psi);
                slot += 1;
            }
        }
    }

    /// phi = Σ_terms coef · P|psi_final>, porting sf-ir's observable builder
    /// (fused X/Z ≤ 2 non-identity factors, else copy + Pauli applications).
    fn build_phi(&mut self, terms: &[PauliTerm]) {
        for x in self.phi.iter_mut() {
            *x = Complex64::new(0.0, 0.0);
        }
        let dim = self.dim;
        for term in terms {
            let mut xmask = 0usize;
            let mut zmask = 0usize;
            let mut nz = 0usize;
            let mut fusable = true;
            for (q, &p) in term.paulis.iter().enumerate() {
                match p {
                    0 => {}
                    1 => {
                        xmask |= 1usize << q;
                        nz += 1;
                    }
                    3 => {
                        zmask |= 1usize << q;
                        nz += 1;
                    }
                    _ => fusable = false,
                }
            }
            if nz > 2 {
                fusable = false;
            }
            if fusable {
                let coef = term.coef;
                for i in 0..dim {
                    let neg = zmask != 0 && ((i & zmask).count_ones() & 1) == 1;
                    let c = if neg { -coef } else { coef };
                    self.phi[i] += c * self.psi[i ^ xmask];
                }
            } else {
                self.term_state.copy_from_slice(&self.psi);
                for (q, &p) in term.paulis.iter().enumerate() {
                    if p != 0 {
                        apply_pauli_1q(&mut self.term_state, p, q);
                    }
                }
                for i in 0..dim {
                    self.phi[i] += term.coef * self.term_state[i];
                }
            }
        }
    }

    /// Backward sweep: gradient per parameter gate (generator dot against the
    /// cached post-state), then un-apply the gate on phi.
    fn backward(&mut self) {
        for g in self.grad.iter_mut() {
            *g = 0.0;
        }
        let dim = self.dim;
        let mut slot = self.n_param_gates;
        for k in (0..self.gates.len()).rev() {
            match self.gates[k] {
                AdjGate::Ry { q, slot: s } => {
                    slot -= 1;
                    let ip = dot_one_pauli(
                        2,
                        &self.phi,
                        &self.cache[slot * dim..(slot + 1) * dim],
                        1usize << q,
                        dim,
                    );
                    // grad += 2·alpha·Im⟨φ|G|ψ_after⟩ with alpha = 0.5
                    // (sf-ir's generator_info factor for Rx/Ry/Rz), i.e.
                    // grad += Im⟨φ|Y|ψ_after⟩ here.
                    self.grad[s as usize] += 2.0 * 0.5 * ip.im;
                    let m = self.dg[k];
                    pair_pass(&mut self.phi, 1usize << q, m);
                }
                AdjGate::Rz { q, slot: s } => {
                    slot -= 1;
                    let ip = dot_one_pauli(
                        3,
                        &self.phi,
                        &self.cache[slot * dim..(slot + 1) * dim],
                        1usize << q,
                        dim,
                    );
                    self.grad[s as usize] += 2.0 * 0.5 * ip.im;
                    let m = self.dg[k];
                    pair_pass(&mut self.phi, 1usize << q, m);
                }
                AdjGate::Cx { ctl, tgt } => {
                    cx_apply(&mut self.phi, dim, ctl as usize, tgt as usize);
                }
            }
        }
    }

    /// Full gradient of ⟨O⟩ for `terms` at `theta`, in parameter-slot order.
    pub fn grad(&mut self, theta: &[f64], terms: &[PauliTerm]) -> Result<&[f64], String> {
        if theta.len() != self.p {
            return Err(format!(
                "cat adj: theta has {} values, expected {}",
                theta.len(),
                self.p
            ));
        }
        self.fill_matrices(theta);
        self.forward();
        self.build_phi(terms);
        self.backward();
        Ok(&self.grad)
    }

    /// Forward expectation ⟨ψ|O|ψ⟩ for `terms` at `theta` (chain-rule
    /// arbiter for finite-difference checks; not part of the gradient path).
    pub fn expectation(&mut self, theta: &[f64], terms: &[PauliTerm]) -> Result<f64, String> {
        if theta.len() != self.p {
            return Err(format!(
                "cat adj: theta has {} values, expected {}",
                theta.len(),
                self.p
            ));
        }
        self.fill_matrices(theta);
        self.forward();
        self.build_phi(terms);
        let mut acc = Complex64::new(0.0, 0.0);
        for i in 0..self.dim {
            acc += self.psi[i].conj() * self.phi[i];
        }
        Ok(acc.re)
    }
}

fn one_qubit(qubits: &[usize], name: &str) -> Result<u32, String> {
    if qubits.len() != 1 {
        return Err(format!("cat adj: {name} needs exactly one qubit"));
    }
    Ok(qubits[0] as u32)
}

fn var_slot(param: &Parameter, idx: &HashMap<&str, u32>) -> Result<u32, String> {
    match param {
        Parameter::Variable { name, .. } => idx
            .get(name.as_str())
            .copied()
            .ok_or_else(|| format!("cat adj: parameter '{name}' is not in dag.parameter_names()")),
        _ => Err("cat adj: only variable parameters are supported \
                  (found a constant or expression)"
            .to_string()),
    }
}

/// CNOT on the flat state: swap pairs (bit ctl set, bit tgt clear).
/// Exact index permutation (no arithmetic), identical to sf-ir's CNOT apply.
#[inline]
fn cx_apply(s: &mut [Complex64], dim: usize, ctl: usize, tgt: usize) {
    let am = 1usize << ctl;
    let bm = 1usize << tgt;
    for i in 0..dim {
        if (i & am) != 0 && (i & bm) == 0 {
            s.swap(i, i ^ bm);
        }
    }
}

/// Single-qubit Pauli application, port of sf-ir's `apply_pauli_1q_inplace`
/// (X permutes, Y rotates by ±i, Z flips the sign — all exact factors).
fn apply_pauli_1q(s: &mut [Complex64], pauli: u8, qubit: usize) {
    let dim = s.len();
    let mask = 1usize << qubit;
    match pauli {
        1 => {
            // X: swap contiguous runs of 2^qubit (pair form of swap_runs).
            let run = mask;
            let block = run << 1;
            for b in 0..dim / block {
                let base = b * block;
                for k in 0..run {
                    s.swap(base + k, base + k + run);
                }
            }
        }
        2 => {
            let neg_i = Complex64::new(0.0, -1.0);
            let pos_i = Complex64::new(0.0, 1.0);
            for i in 0..dim {
                if (i & mask) == 0 {
                    let j = i | mask;
                    let a = s[i];
                    let b = s[j];
                    s[i] = neg_i * b;
                    s[j] = pos_i * a;
                }
            }
        }
        3 => {
            for i in 0..dim {
                if (i & mask) != 0 {
                    s[i] = -s[i];
                }
            }
        }
        _ => {}
    }
}

/// ⟨φ|G|src⟩ for one single-qubit generator pauli (1=X, 2=Y, 3=Z) with qubit
/// mask `m` — verbatim port of sf-ir's scalar `dot_one_pauli` loop structure
/// and accumulation order (`lo` is always 0 / `hi` is the full dimension here).
#[inline]
fn dot_one_pauli(p: u8, phi: &[Complex64], src: &[Complex64], m: usize, dim: usize) -> Complex64 {
    let block = m << 1;
    let mut ip = Complex64::new(0.0, 0.0);
    let mut base = 0;
    while base < dim {
        match p {
            1 => {
                // X: summand_i = conj(phi[i]) · src[i ^ m].
                for (a, b) in phi[base..base + m].iter().zip(&src[base + m..base + block]) {
                    ip += a.conj() * b;
                }
                for (a, b) in phi[base + m..base + block].iter().zip(&src[base..base + m]) {
                    ip += a.conj() * b;
                }
            }
            3 => {
                // Z: upper run negated (subtracting is exact and equals
                // adding -src).
                for (a, b) in phi[base..base + m].iter().zip(&src[base..base + m]) {
                    ip += a.conj() * b;
                }
                for (a, b) in phi[base + m..base + block]
                    .iter()
                    .zip(&src[base + m..base + block])
                {
                    ip -= a.conj() * b;
                }
            }
            2 => {
                // Y: v = -i·src[i|m] / +i·src[i&!m] — an exact swap+sign.
                for (a, b) in phi[base..base + m].iter().zip(&src[base + m..base + block]) {
                    ip += a.conj() * Complex64::new(b.im, -b.re);
                }
                for (a, b) in phi[base + m..base + block].iter().zip(&src[base..base + m]) {
                    ip += a.conj() * Complex64::new(-b.im, b.re);
                }
            }
            _ => {}
        }
        base += block;
    }
    ip
}

#[cfg(test)]
mod tests {
    use super::*;
    use sf_ir::ops::Parameter;

    fn var(name: &str) -> Parameter {
        Parameter::Variable {
            name: name.into(),
            id: 0,
        }
    }

    /// Build the ladder DAG with parameter names t0..tP-1 (worker order).
    fn ladder_dag(n: usize, layers: usize) -> (QuantumDAG, usize) {
        let mut dag = QuantumDAG::new(n, 0);
        let mut slot = 0usize;
        for _ in 0..layers {
            for i in 0..n {
                dag.add_op(OpType::Ry(var(&format!("t{slot}"))), &[i]);
                slot += 1;
                dag.add_op(OpType::Rz(var(&format!("t{slot}"))), &[i]);
                slot += 1;
            }
            for i in 0..n.saturating_sub(1) {
                dag.add_op(OpType::CNOT, &[i, i + 1]);
            }
        }
        (dag, slot)
    }

    fn z_terms(n: usize) -> Vec<PauliTerm> {
        vec![PauliTerm {
            paulis: vec![3u8; n],
            coef: Complex64::new(1.0, 0.0),
        }]
    }

    fn theta(p: usize) -> Vec<f64> {
        (0..p).map(|i| 0.15 + 0.05 * ((i % 7) as f64)).collect()
    }

    #[test]
    fn cat_adj_matches_sf_ir_house_engine() {
        let (n, layers) = (6, 2);
        let (dag, p) = ladder_dag(n, layers);
        assert_eq!(p, 2 * n * layers);
        let terms = z_terms(n);
        let th = theta(p);

        let values: HashMap<String, f64> = (0..p).map(|i| (format!("t{i}"), th[i])).collect();
        let want = sf_ir::adjoint::adjoint_grad(&dag, &terms, &values).unwrap();
        assert_eq!(want.gradients.len(), p);

        let mut prog = AdjProgram::from_dag(&dag).unwrap();
        let got = prog.grad(&th, &terms).unwrap().to_vec();
        let mut max_diff = 0.0f64;
        for (a, b) in got.iter().zip(want.gradients.iter()) {
            max_diff = max_diff.max((a - b).abs());
        }
        assert!(
            max_diff < 1e-11,
            "cat vs sf-ir adjoint mismatch: max_diff={max_diff:e}"
        );

        // from_ladder produces the identical program (same layout, same slots).
        let mut lad = AdjProgram::from_ladder(n, layers);
        let got2 = lad.grad(&th, &terms).unwrap().to_vec();
        assert_eq!(got, got2, "from_ladder and from_dag must agree bit-for-bit");

        // Handle reuse with a fresh parameter vector stays aligned.
        let th2: Vec<f64> = th.iter().map(|v| v * 0.7 + 0.2).collect();
        let values2: HashMap<String, f64> = (0..p).map(|i| (format!("t{i}"), th2[i])).collect();
        let want2 = sf_ir::adjoint::adjoint_grad(&dag, &terms, &values2).unwrap();
        let got3 = prog.grad(&th2, &terms).unwrap().to_vec();
        for (a, b) in got3.iter().zip(want2.gradients.iter()) {
            assert!((a - b).abs() < 1e-11, "handle reuse mismatch");
        }
    }

    #[test]
    fn cat_adj_matches_central_fd() {
        let (n, layers) = (3, 2);
        let (dag, p) = ladder_dag(n, layers);
        let terms = z_terms(n);
        let th = theta(p);
        let mut prog = AdjProgram::from_dag(&dag).unwrap();
        let g = prog.grad(&th, &terms).unwrap().to_vec();

        let eps = 1e-5;
        let mut fd = Vec::with_capacity(p);
        for i in 0..p {
            let mut tp = th.clone();
            tp[i] += eps;
            let mut tm = th.clone();
            tm[i] -= eps;
            let ep = prog.expectation(&tp, &terms).unwrap();
            let em = prog.expectation(&tm, &terms).unwrap();
            fd.push((ep - em) / (2.0 * eps));
        }
        let mut max_err = 0.0f64;
        for (a, b) in g.iter().zip(fd.iter()) {
            max_err = max_err.max((a - b).abs());
        }
        assert!(
            max_err < 1e-7,
            "central-FD arbiter failed: max_err={max_err:e}"
        );
    }

    #[test]
    fn cat_adj_rejects_unsupported_and_bad_theta() {
        let mut dag = QuantumDAG::new(2, 0);
        dag.add_op(OpType::H, &[0]);
        dag.add_op(OpType::Ry(var("x")), &[1]);
        let err = AdjProgram::from_dag(&dag).err().unwrap();
        assert!(err.contains("unsupported gate"), "got: {err}");

        let mut ok = QuantumDAG::new(2, 0);
        ok.add_op(OpType::Ry(var("x")), &[0]);
        ok.add_op(OpType::CNOT, &[0, 1]);
        let mut prog = AdjProgram::from_dag(&ok).unwrap();
        let terms = z_terms(2);
        assert!(prog.grad(&[0.5, 0.5], &terms).is_err());
        assert!(prog.grad(&[0.5], &terms).is_ok());
        assert_eq!(prog.num_params(), 1);
        assert_eq!(prog.n_qubits(), 2);

        // Constant parameters are rejected explicitly (scoped handle).
        let mut cst = QuantumDAG::new(1, 0);
        cst.add_op(OpType::Ry(Parameter::Const(0.4)), &[0]);
        assert!(AdjProgram::from_dag(&cst).is_err());
    }
}
