use nalgebra::DMatrix;
use num_complex::Complex64;

/// Bit-reversal of the low `n` bits of `x` (`rev_n` in the bindings).
#[inline(always)]
pub(crate) fn rev_bits(x: usize, n: usize) -> usize {
    if n == 0 {
        return 0;
    }
    x.reverse_bits() >> (usize::BITS as usize - n)
}

/// Storage layout of a density-matrix buffer.
///
/// `Engine` is the simulator's canonical interleaved layout
/// `data[ket | (bra << n)]`: every constructor and every evolution kernel in
/// this module produces/consumes it.
///
/// `Public` is the row-major *public* layout handed to Python as
/// `metadata["density_matrix"]` (qubit-0-first / big-endian,
/// `public[r][c] = rho_BE(r, c) = rho_LE(rev(r), rev(c))`).  The buffer is
/// byte-for-byte the array the caller receives, so the state and the Python
/// array share ONE 4^n buffer instead of keeping two alive.  Read access goes
/// through `at`, which translates back to engine coordinates; the evolution
/// kernels are engine-only and assert that invariant.
pub enum DmLayout {
    Engine,
    Public,
}

pub struct DensityMatrixState {
    /// Raw 4^n buffer.  Private on purpose: the layout decides how it is
    /// indexed (see `at`), so nothing outside this module may index it
    /// directly — a missed accessor becomes a compile error, not a silently
    /// wrong number.
    data: Vec<Complex64>,
    pub n_qubits: usize,
    layout: DmLayout,
}

impl DensityMatrixState {
    pub fn new(n_qubits: usize) -> Self {
        let dim = 1 << (2 * n_qubits);
        let mut data = vec![Complex64::new(0.0, 0.0); dim];
        // Initial state |0...0><0...0| is index 0 in the vectorized representation
        data[0] = Complex64::new(1.0, 0.0);
        Self {
            data,
            n_qubits,
            layout: DmLayout::Engine,
        }
    }

    /// Adopt an existing vectorized buffer (`data[ket | (bra << n)]` layout,
    /// see `new`) instead of allocating and zero-filling a fresh 4^n vector.
    ///
    /// Callers that already own a correctly sized, fully written buffer (e.g.
    /// the evolve-then-return bindings) use this to avoid a throwaway 4^n
    /// allocation + memset — at n=13 that is a 1 GiB buffer allocated and
    /// discarded per call, which shows up directly in peak RSS.
    pub fn from_data(data: Vec<Complex64>, n_qubits: usize) -> Self {
        debug_assert_eq!(data.len(), 1usize << (2 * n_qubits));
        Self {
            data,
            n_qubits,
            layout: DmLayout::Engine,
        }
    }

    /// Adopt a buffer already permuted into the PUBLIC row-major layout
    /// (`public[r][c] = rho_LE(rev(r), rev(c))`, the array handed to Python as
    /// `metadata["density_matrix"]`).  Read accessors translate the index back,
    /// so no second 4^n buffer has to be materialised.
    pub fn from_data_public(data: Vec<Complex64>, n_qubits: usize) -> Self {
        debug_assert_eq!(data.len(), 1usize << (2 * n_qubits));
        Self {
            data,
            n_qubits,
            layout: DmLayout::Public,
        }
    }

    /// Give the raw buffer back (bindings that return the flat vec).
    pub fn into_data(self) -> Vec<Complex64> {
        self.data
    }

    #[inline(always)]
    pub fn layout(&self) -> &DmLayout {
        &self.layout
    }

    /// Engine density-matrix entry (ket, bra) — layout-aware.
    ///
    /// Engine: `data[ket | (bra << n)]` (the historical index).  Public: the
    /// buffer holds the big-endian matrix `public[r][c] = rho_LE(rev(r),
    /// rev(c))`, so `rho_LE(ket, bra)` sits at `data[rev(ket) * dim +
    /// rev(bra)]`.
    #[inline(always)]
    pub fn at(&self, ket: usize, bra: usize) -> Complex64 {
        match self.layout {
            DmLayout::Engine => self.data[ket | (bra << self.n_qubits)],
            DmLayout::Public => {
                let n = self.n_qubits;
                let dim = 1usize << n;
                self.data[rev_bits(ket, n) * dim + rev_bits(bra, n)]
            }
        }
    }

    /// Engine-layout write of entry (ket, bra).  Public-layout states are
    /// read-only handles (no binding evolves them); a write to one would
    /// silently desynchronise the Python view that shares the buffer, so it is
    /// rejected loudly instead.
    pub fn set_at(&mut self, ket: usize, bra: usize, v: Complex64) {
        self.assert_engine("set_at");
        self.data[ket | (bra << self.n_qubits)] = v;
    }

    /// Pointer/length of the raw buffer (bindings hand it to numpy as a view).
    pub fn data_ptr(&self) -> *const Complex64 {
        self.data.as_ptr()
    }

    #[inline(always)]
    fn assert_engine(&self, what: &str) {
        assert!(
            matches!(self.layout, DmLayout::Engine),
            "DensityMatrixState::{what} requires the engine layout (this state was returned in the public, view-shared layout and must not be evolved)"
        );
    }

    /// Apply a unitary gate U to the density matrix: rho -> U rho U†
    pub fn apply_unitary(&mut self, u: &DMatrix<Complex64>, qubits: &[usize]) {
        self.assert_engine("apply_unitary");
        match qubits.len() {
            1 => {
                let q = qubits[0];
                let m = [[u[(0, 0)], u[(0, 1)]], [u[(1, 0)], u[(1, 1)]]];
                // rho -> (U \otimes I) rho (U† \otimes I): one fused pass over
                // the closed (ket q, bra n+q) 4-cycles — the ket and bra
                // transforms commute, so both apply in the same sweep.
                crate::simd::dm_1q_fused(&mut self.data, q, m);
            }
            2 => {
                let q0 = qubits[0];
                let q1 = qubits[1];
                let mut g = [[Complex64::new(0.0, 0.0); 4]; 4];
                for r in 0..4 {
                    for c in 0..4 {
                        g[r][c] = u[(r, c)];
                    }
                }
                // One fused pass over the closed (ket q0/q1, bra n+q0/n+q1)
                // 16-blocks: ket 4×4 then bra conj 4×4, register-resident.
                crate::simd::dm_2q_fused(&mut self.data, q0, q1, &g);
            }
            _ => panic!("DensityMatrixState::apply_unitary only supports 1q and 2q gates"),
        }
    }

    /// Internal: apply a gate to the vectorized statevector representation
    /// (in-place, single fused pass — no allocation).
    #[allow(dead_code)] // internal helper retained from the pre-fusion path
    fn apply_1q_gate(&mut self, u: &DMatrix<Complex64>, q: usize) {
        let m = [[u[(0, 0)], u[(0, 1)]], [u[(1, 0)], u[(1, 1)]]];
        crate::simd::dm_1q_fused(&mut self.data, q, m);
    }

    /// Internal: 2q gate on the vectorized state (in-place, single fused
    /// pass — no allocation).
    #[allow(dead_code)] // internal helper retained from the pre-fusion path
    fn apply_2q_gate(&mut self, u: &DMatrix<Complex64>, q0: usize, q1: usize) {
        let mut g = [[Complex64::new(0.0, 0.0); 4]; 4];
        for r in 0..4 {
            for c in 0..4 {
                g[r][c] = u[(r, c)];
            }
        }
        crate::simd::dm_2q_fused(&mut self.data, q0, q1, &g);
    }

    /// Pre-summed 1-qubit channel superoperator M = Σ_k K_k ⊗ K_k* on the
    /// closed (ket q, bra n+q) blocks (row = 2·ket + bra). Shared by
    /// `apply_kraus` and the batched pair application so both build the
    /// 4×4 exactly the same way.
    pub fn kraus_superop_1q(kraus_set: &[DMatrix<Complex64>]) -> [[Complex64; 4]; 4] {
        let mut m = [[Complex64::new(0.0, 0.0); 4]; 4];
        for k in kraus_set {
            for r in 0..2 {
                for rp in 0..2 {
                    let krrp = k[(r, rp)];
                    for c in 0..2 {
                        for cp in 0..2 {
                            m[r * 2 + c][rp * 2 + cp] += krrp * k[(c, cp)].conj();
                        }
                    }
                }
            }
        }
        m
    }

    /// Apply a 1-qubit Kraus channel ρ → Σ_k K_k ρ K_k† in ONE fused pass:
    /// the channel superoperator M = Σ_k K_k ⊗ K_k* (a 4×4 matrix on the
    /// closed (ket q, bra n+q) blocks) is pre-summed, then applied per
    /// block. No per-Kraus clones or intermediate buffers.
    pub fn apply_kraus(&mut self, kraus_set: &[DMatrix<Complex64>], qubit: usize) {
        self.assert_engine("apply_kraus");
        let m = Self::kraus_superop_1q(kraus_set);
        crate::simd::dm_super_1q(&mut self.data, qubit, &m);
    }

    /// Two 1-qubit channel superoperators on disjoint qubits applied in a
    /// single sweep over the joint 16-blocks (half the memory traffic of
    /// two sequential `apply_kraus` calls; the arithmetic order per element
    /// matches them exactly).
    pub fn apply_1q_superop_pair(
        &mut self,
        qa: usize,
        m_a: &[[Complex64; 4]; 4],
        qb: usize,
        m_b: &[[Complex64; 4]; 4],
    ) {
        self.assert_engine("apply_1q_superop_pair");
        crate::simd::dm_super_1q_pair(&mut self.data, qa, m_a, qb, m_b);
    }

    /// Apply a pre-summed 1-qubit channel superoperator in one sweep
    /// (same kernel `apply_kraus` uses; lets callers that already built
    /// the 4×4 reuse it without recomputing).
    pub fn apply_1q_superop(&mut self, q: usize, m: &[[Complex64; 4]; 4]) {
        self.assert_engine("apply_1q_superop");
        crate::simd::dm_super_1q(&mut self.data, q, m);
    }

    /// Pre-summed 2-qubit channel superoperator M = Σ_c K_c ⊗ K_c* on the
    /// closed (ket q0q1, bra n+q0/q1) 16-blocks (row = 4·ket + bra, ket
    /// index = 2·bit(q0) + bit(q1) — the same packing `dm_super_2q` and the
    /// fused gate+noise 2q path use).
    pub fn kraus_superop_2q(kraus_set: &[DMatrix<Complex64>]) -> [[Complex64; 16]; 16] {
        let mut m = [[Complex64::new(0.0, 0.0); 16]; 16];
        for k in kraus_set {
            for ket in 0..4 {
                for ketp in 0..4 {
                    let kk = k[(ket, ketp)];
                    for bra in 0..4 {
                        for brap in 0..4 {
                            m[4 * ket + bra][4 * ketp + brap] += kk * k[(bra, brap)].conj();
                        }
                    }
                }
            }
        }
        m
    }

    /// Apply a 2-qubit Kraus channel ρ → Σ_c K_c ρ K_c† in ONE fused sweep
    /// over the joint 16-blocks (the same kernel the fused gate+noise 2q
    /// path uses).
    pub fn apply_kraus_2q(&mut self, kraus_set: &[DMatrix<Complex64>], q0: usize, q1: usize) {
        self.assert_engine("apply_kraus_2q");
        let m = Self::kraus_superop_2q(kraus_set);
        crate::simd::dm_super_2q(&mut self.data, q0, q1, &m);
    }

    /// Internal: in-place single-axis transform on the vectorized state
    /// (now a thin wrapper over the fused kernel — no allocation).
    #[allow(dead_code)] // internal helper retained from the pre-fusion path
    fn apply_1q_gate_to_vec(&mut self, u: &DMatrix<Complex64>, q: usize) {
        self.apply_1q_gate(u, q);
    }

    /// Apply a unitary U (1q or 2q) followed immediately by the 1q Kraus
    /// channels on the touching noise qubits, ALL in one fused
    /// superoperator sweep. The combined Kraus set is
    /// K' = (K_b ⊗ K_a) · U, which is exact:
    /// Σ_kl (K_b,l K_a,k) U ρ U† (K_a,k† K_b,l†)  =  gate-then-noise order.
    /// One memory pass instead of 1 + (#touching channels) passes.
    pub fn apply_gate_noise_fused(
        &mut self,
        u: &DMatrix<Complex64>,
        qubits: &[usize],
        noise: &[(usize, &[DMatrix<Complex64>])],
    ) {
        self.assert_engine("apply_gate_noise_fused");
        match qubits.len() {
            1 => {
                // One touching channel on the gate qubit (caller guarantees).
                debug_assert_eq!(noise.len(), 1);
                let q = qubits[0];
                let fused: Vec<DMatrix<Complex64>> = noise[0].1.iter().map(|k| k * u).collect();
                self.apply_kraus(&fused, q);
            }
            2 => {
                let (q0, q1) = (qubits[0], qubits[1]);
                // Split the touching channels by qubit (at most one channel
                // per qubit comes from the noise model).
                let mut on_q0: Option<&[DMatrix<Complex64>]> = None;
                let mut on_q1: Option<&[DMatrix<Complex64>]> = None;
                for (nq, ks) in noise {
                    if *nq == q0 {
                        on_q0 = Some(ks);
                    } else if *nq == q1 {
                        on_q1 = Some(ks);
                    }
                }
                // Lift a 2×2 Kraus op onto the 4-dim ket space.
                // Row/col = 2·bit(q0) + bit(q1):
                //   lift_q0 = K ⊗ I₂   (acts on the q0 bit)
                //   lift_q1 = I₂ ⊗ K   (acts on the q1 bit)
                fn lift_q0(k: &DMatrix<Complex64>) -> DMatrix<Complex64> {
                    let mut l = DMatrix::<Complex64>::zeros(4, 4);
                    for r in 0..2 {
                        for rp in 0..2 {
                            for s in 0..2 {
                                l[(2 * r + s, 2 * rp + s)] = k[(r, rp)];
                            }
                        }
                    }
                    l
                }
                fn lift_q1(k: &DMatrix<Complex64>) -> DMatrix<Complex64> {
                    let mut l = DMatrix::<Complex64>::zeros(4, 4);
                    for r in 0..2 {
                        for rp in 0..2 {
                            for s in 0..2 {
                                l[(2 * s + r, 2 * s + rp)] = k[(r, rp)];
                            }
                        }
                    }
                    l
                }
                // Combined Kraus set for gate-then-noise: K' = K_b · K_a · U
                // over all channel pairs (empty side = identity).
                let mut combos: Vec<DMatrix<Complex64>> = Vec::with_capacity(16);
                match (on_q0, on_q1) {
                    (Some(a), Some(b)) => {
                        for ka in a {
                            for kb in b {
                                combos.push(lift_q1(kb) * lift_q0(ka) * u);
                            }
                        }
                    }
                    (Some(a), None) => {
                        for ka in a {
                            combos.push(lift_q0(ka) * u);
                        }
                    }
                    (None, Some(b)) => {
                        for kb in b {
                            combos.push(lift_q1(kb) * u);
                        }
                    }
                    (None, None) => {
                        self.apply_unitary(u, qubits);
                        return;
                    }
                }
                // M = Σ_c (K'_c ⊗ K'_c*) — the 16×16 block superoperator
                // (row = 4·ket + bra, col = 4·ket' + bra').
                let mut m = [[Complex64::new(0.0, 0.0); 16]; 16];
                for kp in &combos {
                    for ket in 0..4 {
                        for bra in 0..4 {
                            let row = 4 * ket + bra;
                            for ketp in 0..4 {
                                for brap in 0..4 {
                                    let col = 4 * ketp + brap;
                                    m[row][col] += kp[(ket, ketp)] * kp[(bra, brap)].conj();
                                }
                            }
                        }
                    }
                }
                crate::simd::dm_super_2q(&mut self.data, q0, q1, &m);
            }
            _ => {
                // 3q+ gates: preserve the historical unfused path.
                self.apply_unitary(u, qubits);
                for (nq, ks) in noise {
                    if qubits.contains(nq) {
                        self.apply_kraus(ks, *nq);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::OpType;
    use approx::assert_relative_eq;
    use num_complex::Complex64;

    fn dm_trace(state: &DensityMatrixState) -> Complex64 {
        let n = state.n_qubits;
        let dim = 1 << n;
        (0..dim)
            .map(|i| state.data[i | (i << n)])
            .fold(Complex64::new(0.0, 0.0), |acc, v| acc + v)
    }

    fn dm_purity(state: &DensityMatrixState) -> f64 {
        let n = state.n_qubits;
        let dim = 1 << n;
        let mut rho = DMatrix::<Complex64>::zeros(dim, dim);
        for ket in 0..dim {
            for bra in 0..dim {
                rho[(ket, bra)] = state.data[ket | (bra << n)];
            }
        }
        (&rho * &rho).trace().re
    }

    fn depolarizing_kraus(p: f64) -> Vec<DMatrix<Complex64>> {
        let s0 = (1.0 - 3.0 * p / 4.0).sqrt();
        let sp = (p / 4.0).sqrt();
        let i = Complex64::i();

        let mut k0 = DMatrix::zeros(2, 2);
        k0[(0, 0)] = Complex64::new(s0, 0.0);
        k0[(1, 1)] = Complex64::new(s0, 0.0);

        let mut k1 = DMatrix::zeros(2, 2);
        k1[(0, 1)] = Complex64::new(sp, 0.0);
        k1[(1, 0)] = Complex64::new(sp, 0.0);

        let mut k2 = DMatrix::zeros(2, 2);
        k2[(0, 1)] = -i * sp;
        k2[(1, 0)] = i * sp;

        let mut k3 = DMatrix::zeros(2, 2);
        k3[(0, 0)] = Complex64::new(sp, 0.0);
        k3[(1, 1)] = Complex64::new(-sp, 0.0);

        vec![k0, k1, k2, k3]
    }

    #[test]
    fn test_depolarizing_kraus_preserves_trace() {
        let mut state = DensityMatrixState::new(2);
        let h = OpType::H.to_matrix();
        let cnot = OpType::CNOT.to_matrix();
        state.apply_unitary(&h, &[0]);
        state.apply_unitary(&cnot, &[0, 1]);

        state.apply_kraus(&depolarizing_kraus(0.1), 0);

        let tr = dm_trace(&state);
        assert_relative_eq!(tr.re, 1.0, epsilon = 1e-8);
        assert!(tr.im.abs() < 1e-8);
    }

    #[test]
    fn test_pure_state_purity_one() {
        let state = DensityMatrixState::new(1);
        assert_relative_eq!(dm_purity(&state), 1.0, epsilon = 1e-10);
        assert_relative_eq!(dm_trace(&state).re, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_trace_one_after_gate_application() {
        let mut state = DensityMatrixState::new(2);
        let h = OpType::H.to_matrix();
        let cnot = OpType::CNOT.to_matrix();

        state.apply_unitary(&h, &[0]);
        state.apply_unitary(&cnot, &[0, 1]);

        let tr = dm_trace(&state);
        assert_relative_eq!(tr.re, 1.0, epsilon = 1e-10);
        assert!(tr.im.abs() < 1e-10);
        assert_relative_eq!(dm_purity(&state), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_fused_gate_noise_matches_sequential_1q() {
        let mut fused = DensityMatrixState::new(3);
        let mut seq = DensityMatrixState::new(3);
        let h = OpType::H.to_matrix();
        let rx = OpType::Rx(crate::ops::Parameter::Const(0.3)).to_matrix();
        let dep = depolarizing_kraus(0.05);

        // Shared warm-up so both states start from the same rho.
        for st in [&mut fused, &mut seq] {
            st.apply_unitary(&h, &[0]);
            st.apply_unitary(&OpType::CNOT.to_matrix(), &[0, 1]);
        }

        // Sequential: gate then channel.
        seq.apply_unitary(&rx, &[1]);
        seq.apply_kraus(&dep, 1);
        // Fused: one sweep.
        fused.apply_gate_noise_fused(&rx, &[1], &[(1, dep.as_slice())]);

        for i in 0..(1usize << 6) {
            assert_relative_eq!(fused.data[i].re, seq.data[i].re, epsilon = 1e-12);
            assert_relative_eq!(fused.data[i].im, seq.data[i].im, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_fused_gate_noise_matches_sequential_2q() {
        let mut fused = DensityMatrixState::new(3);
        let mut seq = DensityMatrixState::new(3);
        let h = OpType::H.to_matrix();
        let cnot = OpType::CNOT.to_matrix();
        let dep_a = depolarizing_kraus(0.05);
        let dep_b = depolarizing_kraus(0.07);

        for st in [&mut fused, &mut seq] {
            st.apply_unitary(&h, &[0]);
            st.apply_unitary(
                &OpType::Rx(crate::ops::Parameter::Const(0.4)).to_matrix(),
                &[1],
            );
        }

        // Sequential: cnot then channels on both touched qubits.
        seq.apply_unitary(&cnot, &[1, 2]);
        seq.apply_kraus(&dep_a, 1);
        seq.apply_kraus(&dep_b, 2);
        // Fused: one sweep.
        fused.apply_gate_noise_fused(
            &cnot,
            &[1, 2],
            &[(1, dep_a.as_slice()), (2, dep_b.as_slice())],
        );

        for i in 0..(1usize << 6) {
            assert_relative_eq!(fused.data[i].re, seq.data[i].re, epsilon = 1e-12);
            assert_relative_eq!(fused.data[i].im, seq.data[i].im, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_fused_2q_single_channel_matches_sequential() {
        let mut fused = DensityMatrixState::new(3);
        let mut seq = DensityMatrixState::new(3);
        let h = OpType::H.to_matrix();
        let cnot = OpType::CNOT.to_matrix();
        let dep = depolarizing_kraus(0.03);

        for st in [&mut fused, &mut seq] {
            st.apply_unitary(&h, &[0]);
        }

        seq.apply_unitary(&cnot, &[0, 2]);
        seq.apply_kraus(&dep, 2);
        fused.apply_gate_noise_fused(&cnot, &[0, 2], &[(2, dep.as_slice())]);

        for i in 0..(1usize << 6) {
            assert_relative_eq!(fused.data[i].re, seq.data[i].re, epsilon = 1e-12);
            assert_relative_eq!(fused.data[i].im, seq.data[i].im, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_kraus_superop_2q_single_unitary_matches_apply_unitary() {
        let mut via_kraus = DensityMatrixState::new(2);
        let mut via_gate = DensityMatrixState::new(2);
        let h = OpType::H.to_matrix();
        let cnot = OpType::CNOT.to_matrix();
        for st in [&mut via_kraus, &mut via_gate] {
            st.apply_unitary(&h, &[0]);
            st.apply_unitary(&cnot, &[0, 1]);
        }
        // {U} as a (trivial) Kraus set must equal the unitary itself.
        let u = DMatrix::<Complex64>::from_fn(4, 4, |r, c| cnot[(r, c)]);
        via_gate.apply_unitary(&u, &[0, 1]);
        via_kraus.apply_kraus_2q(std::slice::from_ref(&u), 0, 1);
        for i in 0..(1usize << 4) {
            assert_relative_eq!(via_kraus.data[i].re, via_gate.data[i].re, epsilon = 1e-12);
            assert_relative_eq!(via_kraus.data[i].im, via_gate.data[i].im, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_kraus_2q_matches_weighted_unitary_mixture_and_preserves_trace() {
        // Bit-packed 2q Pauli embedding: row = 2·bit(q0) + bit(q1).
        fn p2q(a: &[[Complex64; 2]; 2], b: &[[Complex64; 2]; 2]) -> DMatrix<Complex64> {
            let mut m = DMatrix::<Complex64>::zeros(4, 4);
            for a1 in 0..2 {
                for a2 in 0..2 {
                    for b1 in 0..2 {
                        for b2 in 0..2 {
                            m[(2 * a1 + a2, 2 * b1 + b2)] = a[a1][b1] * b[a2][b2];
                        }
                    }
                }
            }
            m
        }
        let z = Complex64::new(0.0, 0.0);
        let o = Complex64::new(1.0, 0.0);
        let i2 = [[o, z], [z, o]];
        let x2 = [[z, o], [o, z]];
        let i4 = p2q(&i2, &i2);
        let xx = p2q(&x2, &x2);

        let w: f64 = 0.5;
        let ks = vec![
            i4.clone() * Complex64::new(w.sqrt(), 0.0),
            xx.clone() * Complex64::new(w.sqrt(), 0.0),
        ];

        let mut channel = DensityMatrixState::new(2);
        let h = OpType::H.to_matrix();
        let cnot = OpType::CNOT.to_matrix();
        channel.apply_unitary(&h, &[0]);
        channel.apply_unitary(&cnot, &[0, 1]);
        channel.apply_kraus_2q(&ks, 0, 1);

        // Reference mixture: w·ρ + w·(XX) ρ (XX).
        let mut base = DensityMatrixState::new(2);
        base.apply_unitary(&h, &[0]);
        base.apply_unitary(&cnot, &[0, 1]);
        let base_data = base.data.clone();
        base.apply_unitary(&xx, &[0, 1]);
        for i in 0..(1usize << 4) {
            let want = base_data[i] * w + base.data[i] * w;
            assert_relative_eq!(channel.data[i].re, want.re, epsilon = 1e-12);
            assert_relative_eq!(channel.data[i].im, want.im, epsilon = 1e-12);
        }
        let tr = dm_trace(&channel);
        assert_relative_eq!(tr.re, 1.0, epsilon = 1e-10);
        assert!(tr.im.abs() < 1e-10);
    }
}
