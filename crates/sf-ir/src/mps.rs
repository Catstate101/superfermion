use crate::dag::QuantumDAG;
use crate::ops::OpType;
use nalgebra::{DMatrix, DVector};
use num_complex::Complex64;

// faer is used for both the heavy matmul AND QR factorisation in
// apply_2q_gate.  At bond dimension D=64 nalgebra runs ~1 GFLOPS (no
// SIMD) while faer reaches ~8 GFLOPS via cache-blocked AVX-2 microkernels.
// We pay an O(D^2) conversion cost which is amortised over the O(D^3)
// matmul + QR.
use faer::linalg::matmul::matmul as faer_matmul;
use faer::linalg::solvers::Qr as FaerQr;
use faer::linalg::solvers::ThinSvd;
use faer::Mat as FaerMat;
use faer::Parallelism;

/// Fast nalgebra→faer conversion exploiting shared column-major layout.
fn na_to_faer(m: &DMatrix<Complex64>) -> FaerMat<Complex64> {
    let (rows, cols) = m.shape();
    FaerMat::from_fn(rows, cols, |i, j| m[(i, j)])
}

/// Fast faer→nalgebra conversion exploiting shared column-major layout.
fn faer_to_na(f: &FaerMat<Complex64>) -> DMatrix<Complex64> {
    let rows = f.nrows();
    let cols = f.ncols();
    let mut m = DMatrix::<Complex64>::zeros(rows, cols);
    for j in 0..cols {
        for i in 0..rows {
            m[(i, j)] = f.read(i, j);
        }
    }
    m
}

/// Re-pack a 4x4 two-qubit gate from (q1_high=A, q2_low=B) bit order to
/// (q1_low=A, q2_high=B) by swapping bit 0 with bit 1 in both row and
/// column indices.  Used when the MPS caller passes (q1, q2) with q1 > q2
/// — the gate's matrix layout puts q1's bit at position 1 (the high bit
/// of the 2-bit index), but the internal contraction uses idx1<idx2 with
/// idx1's bit at position 1.
fn swap_2q_gate_bits(g: &DMatrix<Complex64>) -> DMatrix<Complex64> {
    // Permutation: bits {0,1,2,3} -> {0,2,1,3} (swap of bits in 2-bit index).
    let perm = [0usize, 2, 1, 3];
    let mut out = DMatrix::zeros(4, 4);
    for r in 0..4 {
        for c in 0..4 {
            out[(r, c)] = g[(perm[r], perm[c])];
        }
    }
    out
}

/// 4x4 SWAP gate used for long-range routing in the MPS.
fn swap_matrix() -> DMatrix<Complex64> {
    let mut m = DMatrix::zeros(4, 4);
    let one = Complex64::new(1.0, 0.0);
    m[(0, 0)] = one;
    m[(1, 2)] = one;
    m[(2, 1)] = one;
    m[(3, 3)] = one;
    m
}

/// Historical v1 acceptance threshold, retained for reference: v1 kept the
/// QR position cut when it discarded ≤ this fraction of the squared weight.
/// It was superseded because ~zero QR loss does not imply the cut is safe —
/// the row-norm rank test keeps bond dimensions whose singular values are far
/// below τ (row norms are not bounded by σ), which inflated bonds until the
/// cap discarded real weight (brick12@26/cap 64: Σε = 6.7e-2).  The default
/// policy now τ-prunes every merge on the optimal SVD subspace and no longer
/// consults this constant (see [`mps_truncation_svd_enabled`]).
#[allow(dead_code)]
const MPS_QR_ACCEPT_EPS: f64 = 1e-14;

/// Bond-truncation policy for 2-qubit gate merges, read once from the
/// `SF_MPS_TRUNCATION` env var:
///
/// * unset / `"svd"` (default) — truncate EVERY merge on the optimal rank-k
///   SVD subspace (keep the k largest singular values, k = min(bond_dim,
///   #σ > 1e-12·‖M‖_F)).  Pruning sub-τ directions at every step keeps bonds
///   at their τ-numerical rank, so the cap only ever cuts directions of
///   weight ≤ ~1e-24·‖M‖²_F (brick12@26 cap 64: err 4.2e-6 → 3.6e-14,
///   Σε 6.7e-2 → 1.1e-16).
/// * `"qr"` / `"legacy"` — always keep the historical position cut
///   (unpivoted QR rows 0..k, i.e. span of the first k merged columns).
fn mps_truncation_svd_enabled() -> bool {
    static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| {
        !matches!(
            std::env::var("SF_MPS_TRUNCATION")
                .unwrap_or_default()
                .trim()
                .to_ascii_lowercase()
                .as_str(),
            "qr" | "legacy"
        )
    })
}

/// Single-qubit Pauli enum used for expectation-value calls from Python.
/// Stored as one byte per site so Python can pass a raw &[u8].
///   0 = I, 1 = X, 2 = Y, 3 = Z
///
/// Matrix Product State (MPS) core for high-efficiency simulation.
pub struct MPSState {
    pub n_qubits: usize,
    pub bond_dim: usize,
    /// Tensors: Vec of DMatrix (left_bond * physical_qubit, right_bond)
    /// Shaped as (D_L * 2, D_R)
    pub tensors: Vec<DMatrix<Complex64>>,
    /// Lazy-SWAP routing: perm[virtual_qubit] = physical_site
    pub perm: Vec<usize>,
    /// Inverse map: perm_inv[physical_site] = virtual_qubit
    pub perm_inv: Vec<usize>,
    /// Accumulated discarded-weight fraction Σε over all 2q-gate truncation
    /// steps (0.0 = exact evolution).  Per step ε is either the QR
    /// position-cut estimate (‖R_discarded‖²_F / ‖R‖²_F) on the historical
    /// "qr"/"legacy" path, or the exact Σσ²_discarded / Σσ² of the optimal
    /// SVD truncation under the default τ-prune policy
    /// (see [`mps_truncation_svd_enabled`]).
    pub discarded_weight: f64,
    /// Largest single-step discarded fraction observed.
    pub max_discarded_weight: f64,
    /// Number of 2q-gate steps that discarded non-negligible weight (ε > 1e-15).
    pub truncation_events: usize,
}

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

impl MPSState {
    pub fn new(n_qubits: usize, bond_dim: usize) -> Self {
        let mut tensors = Vec::with_capacity(n_qubits);
        // Initial state |0...0>: each tensor is (1*2, 1) with [1, 0]^T
        for _ in 0..n_qubits {
            let mut t = DMatrix::zeros(2, 1);
            t[(0, 0)] = Complex64::new(1.0, 0.0);
            tensors.push(t);
        }
        Self {
            n_qubits,
            bond_dim,
            tensors,
            perm: (0..n_qubits).collect(),
            perm_inv: (0..n_qubits).collect(),
            discarded_weight: 0.0,
            max_discarded_weight: 0.0,
            truncation_events: 0,
        }
    }

    pub fn apply_1q_gate(&mut self, target: usize, gate: &nalgebra::DMatrix<Complex64>) {
        Self::apply_1q_gate_static(&mut self.tensors[target], gate);
    }

    /// Left-canonicalize the MPS via left-to-right QR sweep.
    ///
    /// After applying 2-qubit gates via `apply_2q_gate_static` (which does
    /// right-canonical QR), the MPS is in mixed canonical form.  This sweep
    /// converts it to left-canonical form.
    ///
    /// Cost: O(n * D^3) — one QR per site.
    pub fn canonicalize_left(&mut self) {
        for i in 0..self.n_qubits.saturating_sub(1) {
            let t = &self.tensors[i];
            let (rows, cols) = t.shape(); // rows = D_L * 2, cols = D_R

            let qr = t.clone_owned().qr();
            let q = qr.q();
            let r = qr.r();

            let rank = std::cmp::min(rows, cols);
            let k = std::cmp::min(self.bond_dim, rank);

            let new_t = q.columns(0, k).into_owned();
            self.tensors[i] = new_t;

            let r_top = r.rows(0, k).into_owned();
            let next = &self.tensors[i + 1];
            let d_l_next = next.shape().0 / 2;
            let d_r_next = next.shape().1;

            let next_0 = next.rows(0, d_l_next).into_owned();
            let next_1 = next.rows(d_l_next, d_l_next).into_owned();

            let new_0 = &r_top * &next_0;
            let new_1 = &r_top * &next_1;

            let mut next_new = DMatrix::zeros(k * 2, d_r_next);
            next_new.rows_mut(0, k).copy_from(&new_0);
            next_new.rows_mut(k, k).copy_from(&new_1);
            self.tensors[i + 1] = next_new;
        }
    }

    /// Right-canonicalize the MPS via right-to-left LQ sweep.
    ///
    /// Produces right-canonical form (Σ_s B^s B^s† = I at each site),
    /// which is required for correct left-to-right sampling in [`sample`].
    ///
    /// Cost: O(n * D^3) — one LQ per site.
    pub fn canonicalize_right(&mut self) {
        for i in (1..self.n_qubits).rev() {
            let t = &self.tensors[i]; // shape (D_L * 2, D_R)
            let d_l = t.shape().0 / 2;
            let d_r = t.shape().1;

            // Reshape from packed (D_L*2, D_R) to (D_L, 2*D_R) where the
            // two physical blocks sit side-by-side in columns.
            let mut m = DMatrix::<Complex64>::zeros(d_l, 2 * d_r);
            for s in 0..2usize {
                for l in 0..d_l {
                    for r in 0..d_r {
                        m[(l, s * d_r + r)] = t[(l + s * d_l, r)];
                    }
                }
            }

            // LQ decomposition via QR of the adjoint (conjugate transpose):
            //   M^H = Q_t R_t  ⟹  M = R_t^H Q_t^H  ≡  L Q_lq
            let mt = m.adjoint(); // (2*D_R, D_L)
            let qr = mt.qr();
            let q_t = qr.q(); // (2*D_R, min(2*D_R, D_L))
            let r_t = qr.r(); // (min(2*D_R, D_L), D_L)

            let k = std::cmp::min(self.bond_dim, std::cmp::min(2 * d_r, d_l));

            // Q_lq = first-k-cols(Q_t)^T → shape (k, 2*D_R)
            // Reshape back to packed (2*k, D_R).
            let mut new_t = DMatrix::<Complex64>::zeros(2 * k, d_r);
            for s in 0..2usize {
                for j in 0..k {
                    for r in 0..d_r {
                        // Q_lq[j, s*d_r + r] = Q_t[(s*d_r + r), j]^*
                        // (conjugate-transpose)
                        new_t[(s * k + j, r)] = q_t[(s * d_r + r, j)].conj();
                    }
                }
            }
            self.tensors[i] = new_t;

            // L = first-k-rows(R_t)^T → shape (D_L, k)
            //   = conjugate-transpose of R_t's first k rows
            // But R_t has shape (min(2*D_R, D_L), D_L), first k rows = (k, D_L)
            // L = R_t[0:k, :]^T  but we need conjugate transpose for LQ.
            // Actually: M = L Q, where L = R_t^†[:, 0:k] = R_t[0:k, :]^T.conj()
            // But R is upper triangular from real QR, and our data is complex.
            // Correct: L = (R_t[0:k, :])^H  shape (D_L, k)
            let mut l_matrix = DMatrix::<Complex64>::zeros(d_l, k);
            for ii in 0..d_l {
                for jj in 0..k {
                    l_matrix[(ii, jj)] = r_t[(jj, ii)].conj();
                }
            }

            // Absorb L into tensor[i-1]'s right bond.
            // tensor[i-1] has shape (D_{prev}*2, D_L), result: (D_{prev}*2, k)
            let prev = &self.tensors[i - 1];
            let d_l_prev = prev.shape().0 / 2;

            let prev_0 = prev.rows(0, d_l_prev).into_owned(); // (D_prev, D_L)
            let prev_1 = prev.rows(d_l_prev, d_l_prev).into_owned(); // (D_prev, D_L)

            let new_0 = &prev_0 * &l_matrix; // (D_prev, k)
            let new_1 = &prev_1 * &l_matrix; // (D_prev, k)

            let mut new_prev = DMatrix::<Complex64>::zeros(d_l_prev * 2, k);
            new_prev.rows_mut(0, d_l_prev).copy_from(&new_0);
            new_prev.rows_mut(d_l_prev, d_l_prev).copy_from(&new_1);
            self.tensors[i - 1] = new_prev;
        }
    }

    pub fn apply_1q_gate_static(
        t: &mut nalgebra::DMatrix<Complex64>,
        gate: &nalgebra::DMatrix<Complex64>,
    ) {
        let (rows, cols) = t.shape(); // rows = D_L * 2
        let d_l = rows / 2;

        let mut new_t = nalgebra::DMatrix::zeros(rows, cols);

        // Cache gate elements
        let g00 = gate[(0, 0)];
        let g01 = gate[(0, 1)];
        let g10 = gate[(1, 0)];
        let g11 = gate[(1, 1)];

        for c in 0..cols {
            for l in 0..d_l {
                let v0 = t[(l, c)];
                let v1 = t[(l + d_l, c)];
                new_t[(l, c)] = g00 * v0 + g01 * v1;
                new_t[(l + d_l, c)] = g10 * v0 + g11 * v1;
            }
        }

        *t = new_t;
    }

    /// Apply a SWAP on adjacent physical sites (phys_lo, phys_lo+1) and update
    /// the virtual-to-physical permutation map (lazy: qubit stays at new site).
    pub fn apply_lazy_swap_phys(&mut self, phys_lo: usize) {
        let swap_gate = swap_matrix();
        self.apply_2q_gate(phys_lo, phys_lo + 1, &swap_gate);
        let vi = self.perm_inv[phys_lo];
        let vj = self.perm_inv[phys_lo + 1];
        self.perm[vi] = phys_lo + 1;
        self.perm[vj] = phys_lo;
        self.perm_inv[phys_lo] = vj;
        self.perm_inv[phys_lo + 1] = vi;
    }

    /// Apply a 2-qubit gate on virtual qubits (v1, v2) using lazy-SWAP routing:
    /// moves the higher physical qubit leftward until adjacent, then applies
    /// the gate. The qubit stays at its new physical site (no swap-back).
    pub fn apply_gate_lazy_routed(&mut self, v1: usize, v2: usize, gate: &DMatrix<Complex64>) {
        let p1 = self.perm[v1];
        let p2 = self.perm[v2];
        if p1 == p2 {
            return;
        }
        if (p1 as isize - p2 as isize).abs() == 1 {
            self.apply_2q_gate(p1, p2, gate);
            return;
        }
        let plo = p1.min(p2);
        // Move the higher-positioned virtual qubit leftward to plo+1
        let phi_virt = if p1 > p2 { v1 } else { v2 };
        while self.perm[phi_virt] > plo + 1 {
            let cur = self.perm[phi_virt];
            self.apply_lazy_swap_phys(cur - 1);
        }
        let new_p1 = self.perm[v1];
        let new_p2 = self.perm[v2];
        self.apply_2q_gate(new_p1, new_p2, gate);
    }

    /// Apply a 2-qubit gate to qubits q1, q2 (any positions). For non-adjacent
    /// qubits, route via SWAP chain so the gate ends up acting on adjacent
    /// sites, then SWAP back.
    pub fn apply_2q_gate_routed(&mut self, q1: usize, q2: usize, gate: &DMatrix<Complex64>) {
        let dist = (q1 as isize - q2 as isize).unsigned_abs();
        if dist == 0 {
            return;
        }
        if dist == 1 {
            self.apply_2q_gate(q1, q2, gate);
            return;
        }
        let (lo, hi) = if q1 < q2 { (q1, q2) } else { (q2, q1) };
        let swap_gate = swap_matrix();
        // SWAP qubit `lo` up to `hi - 1` (so it becomes adjacent to hi)
        for k in lo..(hi - 1) {
            self.apply_2q_gate(k, k + 1, &swap_gate);
        }
        // Now apply the original gate on (hi-1, hi). If original was (q1, q2)
        // with q1 > q2 we already SWAP'd-routed; preserve original control/
        // target order from the caller's perspective.
        if q1 < q2 {
            self.apply_2q_gate(hi - 1, hi, gate);
        } else {
            self.apply_2q_gate(hi, hi - 1, gate);
        }
        // SWAP back to restore the original qubit ordering of the MPS sites.
        for k in (lo..(hi - 1)).rev() {
            self.apply_2q_gate(k, k + 1, &swap_gate);
        }
    }
    pub fn apply_2q_gate(&mut self, q1: usize, q2: usize, gate: &nalgebra::DMatrix<Complex64>) {
        // This is safe because we check q1 != q2 and only access those two.
        // For borrow checker we have to do some dancing if they are adjacent.
        let eps = if q1 < q2 {
            let (left, right) = self.tensors.split_at_mut(q2);
            Self::apply_2q_gate_static(&mut left[q1], &mut right[0], q1, q2, gate, self.bond_dim)
        } else {
            let (left, right) = self.tensors.split_at_mut(q1);
            Self::apply_2q_gate_static(&mut left[q2], &mut right[0], q1, q2, gate, self.bond_dim)
        };
        self.record_discard(eps);
    }

    /// Accumulate truncation telemetry from one 2q-gate QR step.
    ///
    /// `eps` is the fraction of QR weight discarded at that step (0.0 =
    /// exact).  Phantom-bond drops carry ~machine-zero weight and add to
    /// `discarded_weight` but are not counted as events.
    pub fn record_discard(&mut self, eps: f64) {
        if eps > 0.0 {
            self.discarded_weight += eps;
            if eps > 1e-15 {
                self.truncation_events += 1;
            }
            if eps > self.max_discarded_weight {
                self.max_discarded_weight = eps;
            }
        }
    }

    /// Conservative truncation metric: 1 − Σε, clamped to [0, 1].
    ///
    /// ``discarded_weight`` accumulates the per-step *relative* discarded
    /// fractions ε of the squared state norm, so by the union bound the
    /// surviving norm² is ≥ 1 − Σε for a normalized input.  In truncation
    /// benchmarks (brickwork sweeps vs exact statevectors) this value also
    /// stayed below the measured physical fidelity of the normalized MPS
    /// (F = |⟨ψ_exact|ψ̃⟩|² / ‖ψ̃‖²); that is empirical, not a certified
    /// bound — compounding across truncating steps is not controlled.
    /// Use ``discarded_weight`` as the primary truncation metric.
    pub fn fidelity_lower_bound(&self) -> f64 {
        (1.0 - self.discarded_weight).clamp(0.0, 1.0)
    }

    /// Apply a 2-qubit gate to adjacent sites via QR factorization.
    ///
    /// Returns the discarded-weight fraction ε ∈ [0, 1) of this step: the
    /// squared Frobenius weight of R rows dropped by the `bond_dim` cap (plus
    /// phantom rows below the 1e-12 threshold), relative to the full
    /// rank-range weight.  ε = 0.0 means the step was exact.
    pub fn apply_2q_gate_static(
        t1: &mut nalgebra::DMatrix<Complex64>,
        t2: &mut nalgebra::DMatrix<Complex64>,
        q1: usize,
        q2: usize,
        gate: &nalgebra::DMatrix<Complex64>,
        bond_dim: usize,
    ) -> f64 {
        if (q1 as isize - q2 as isize).abs() != 1 {
            return 0.0;
        }
        let (idx1, idx2, gate_eff) = if q1 < q2 {
            (q1, q2, gate.clone())
        } else {
            (q2, q1, swap_2q_gate_bits(gate))
        };
        let _ = (idx1, idx2); // idx1 < idx2 ensured

        let d_l1 = t1.shape().0 / 2;
        let d_m = t1.shape().1;
        let d_r2 = t2.shape().1;

        let t1_a = t1.rows(0, d_l1).into_owned();
        let t1_b = t1.rows(d_l1, d_l1).into_owned();
        let t2_a = t2.rows(0, d_m).into_owned();
        let t2_b = t2.rows(d_m, d_m).into_owned();

        let use_faer_matmul = d_l1 >= 16 && d_m >= 16 && d_r2 >= 16;
        let (t00, t01, t10, t11) = if use_faer_matmul {
            // Fast nalgebra→faer conversion: both are column-major, so we
            // copy contiguous column slices instead of element-by-element.
            let t1a_f = na_to_faer(&t1_a);
            let t1b_f = na_to_faer(&t1_b);
            let t2a_f = na_to_faer(&t2_a);
            let t2b_f = na_to_faer(&t2_b);

            let alpha = Complex64::new(1.0, 0.0);
            // Use Rayon parallelism for large bond dimensions (D >= 64) where
            // the O(D^3) matmul cost justifies thread-pool overhead.
            let par = if d_m >= 64 {
                Parallelism::Rayon(0)
            } else {
                Parallelism::None
            };
            let mut t00_f: FaerMat<Complex64> = FaerMat::zeros(d_l1, d_r2);
            let mut t01_f: FaerMat<Complex64> = FaerMat::zeros(d_l1, d_r2);
            let mut t10_f: FaerMat<Complex64> = FaerMat::zeros(d_l1, d_r2);
            let mut t11_f: FaerMat<Complex64> = FaerMat::zeros(d_l1, d_r2);

            faer_matmul(
                t00_f.as_mut(),
                t1a_f.as_ref(),
                t2a_f.as_ref(),
                None,
                alpha,
                par,
            );
            faer_matmul(
                t01_f.as_mut(),
                t1a_f.as_ref(),
                t2b_f.as_ref(),
                None,
                alpha,
                par,
            );
            faer_matmul(
                t10_f.as_mut(),
                t1b_f.as_ref(),
                t2a_f.as_ref(),
                None,
                alpha,
                par,
            );
            faer_matmul(
                t11_f.as_mut(),
                t1b_f.as_ref(),
                t2b_f.as_ref(),
                None,
                alpha,
                par,
            );

            (
                faer_to_na(&t00_f),
                faer_to_na(&t01_f),
                faer_to_na(&t10_f),
                faer_to_na(&t11_f),
            )
        } else {
            (&t1_a * &t2_a, &t1_a * &t2_b, &t1_b * &t2_a, &t1_b * &t2_b)
        };

        let mut m_matrix = nalgebra::DMatrix::zeros(d_l1 * 2, 2 * d_r2);
        for s_out in 0..4usize {
            let g0 = gate_eff[(s_out, 0)];
            let g1 = gate_eff[(s_out, 1)];
            let g2 = gate_eff[(s_out, 2)];
            let g3 = gate_eff[(s_out, 3)];
            let block = &t00 * g0 + &t01 * g1 + &t10 * g2 + &t11 * g3;
            let row_off = (s_out >> 1) * d_l1;
            let col_off = (s_out & 1) * d_r2;
            for l in 0..d_l1 {
                for r in 0..d_r2 {
                    m_matrix[(row_off + l, col_off + r)] = block[(l, r)];
                }
            }
        }

        let nrows = d_l1 * 2;
        let ncols = 2 * d_r2;
        let use_faer_qr = nrows >= 16 && ncols >= 16;

        // ── Factorisation: QR via faer (BLAS-quality SIMD) ──
        // QR factorisation is numerically exact for unitary matrices and preserves
        // norm to machine precision. We use faer for large matrices (AVX-2 optimized)
        // and fall back to nalgebra for small matrices.
        let (q_out, r_matrix, eps): (
            nalgebra::DMatrix<Complex64>,
            nalgebra::DMatrix<Complex64>,
            f64,
        ) = if use_faer_qr {
            let m_faer: FaerMat<Complex64> =
                FaerMat::from_fn(nrows, ncols, |i, j| m_matrix[(i, j)]);

            // ── Optimal τ-pruned truncation (additive, default-on) ──
            // Unpivoted Householder QR keeps span(M[:, 0:k]) — a position-
            // based subspace — and its row-norm rank test keeps "garbage"
            // dimensions whose singular values are far below τ = 1e-12·‖M‖_F
            // (row norms are not bounded by σ).  Those dims inflate the bond
            // until the cap cuts real weight (measured on brick12@26/cap 64:
            // Σε = 6.7e-2, observable error 4.2e-6).  The default policy
            // truncates EVERY merge on the optimal τ-pruned SVD subspace
            // (k = min(bond_dim, #σ > τ)), which keeps bonds at their
            // τ-numerical rank at every step; SF_MPS_TRUNCATION=qr|legacy
            // restores the historical position cut.
            if mps_truncation_svd_enabled() {
                let (t1_svd, t2_svd, eps_svd) =
                    Self::svd_truncate_merged(&m_faer, d_l1, d_r2, bond_dim);
                *t1 = t1_svd;
                *t2 = t2_svd;
                return eps_svd;
            }

            let qr = FaerQr::new(m_faer.as_ref());

            // Extract R first: the truncation-decision weights below need only
            // R, and the optimal-SVD override can then skip the O(n³)
            // `compute_q` entirely on the steps it re-truncates.
            let r_full = qr.compute_r();

            // Rank-revealing via R row-norms.
            //
            // For phantom-bond removal (e.g. BV product states), we need to
            // drop trailing rows of R whose entire row-norm is ~machine eps.
            // Using R[i,i] alone is unreliable since unpivoted Householder QR
            // does not sort the diagonal. Instead use the FULL row norm
            // sqrt(sum_j |R[i,j]|^2) (only j>=i are nonzero since R is upper
            // triangular). A row with all entries below ~1e-12 carries no
            // signal and can be safely dropped.
            //
            // We also use the Frobenius norm of R as a global scale reference,
            // so the threshold scales with the matrix magnitude (handles both
            // normalized and unnormalized MPS states).
            let k_max = std::cmp::min(bond_dim, std::cmp::min(nrows, ncols));
            // Compute ||R||_F over the kept range k_max (threshold scale,
            // unchanged) plus the full rank-range weight used as the
            // truncation denominator.
            let mut frob_sq = 0.0_f64;
            let mut frob_sq_full = 0.0_f64;
            for i in 0..k_max {
                for j in i..ncols {
                    let nsq = r_full.read(i, j).norm_sqr();
                    frob_sq += nsq;
                    frob_sq_full += nsq;
                }
            }
            let rank_max = std::cmp::min(nrows, ncols);
            let r_lim = std::cmp::min(rank_max, r_full.nrows());
            for i in k_max..r_lim {
                for j in i..ncols {
                    frob_sq_full += r_full.read(i, j).norm_sqr();
                }
            }
            let frob = frob_sq.sqrt().max(1e-300_f64);
            // Threshold per row: drop row if ||R[i, i:]||_2 < eps * ||R||_F.
            // eps = 1e-12 is well above QR roundoff (~1e-14 for double) but
            // below any singular value we'd want to keep (~1e-10 in practice).
            let row_eps_sq = (1e-12_f64 * frob).powi(2);
            // Find k_final = (last row index with significant norm) + 1.
            let mut k_final = 1usize;
            for i in 0..k_max {
                let mut row_norm_sq = 0.0_f64;
                for j in i..ncols {
                    row_norm_sq += r_full.read(i, j).norm_sqr();
                }
                if row_norm_sq > row_eps_sq {
                    k_final = i + 1;
                }
            }
            // Kept weight Σ_{i<k_final} ||R[i, i:]||².
            let mut kept_sq = 0.0_f64;
            for i in 0..k_final {
                for j in i..ncols {
                    kept_sq += r_full.read(i, j).norm_sqr();
                }
            }
            // Discarded weight = rows beyond k_max (bond_dim truncation)
            // plus phantom rows dropped below the threshold.
            let discarded_sq = (frob_sq_full - kept_sq).max(0.0);
            let eps = discarded_sq / frob_sq_full.max(1e-300_f64);

            let q_full = qr.compute_q();

            let mut q_dst = nalgebra::DMatrix::<Complex64>::zeros(nrows, k_final);
            let mut r_dst = nalgebra::DMatrix::<Complex64>::zeros(k_final, ncols);

            for j in 0..k_final {
                for i in 0..nrows {
                    q_dst[(i, j)] = q_full.read(i, j);
                }
            }
            for j in 0..ncols {
                for i in 0..k_final {
                    r_dst[(i, j)] = r_full.read(i, j);
                }
            }
            (q_dst, r_dst, eps)
        } else {
            // Same τ-prune policy for small tensors — the truncation policy
            // must not depend on which backend factorizes the merged tensor.
            if mps_truncation_svd_enabled() {
                let m_faer: FaerMat<Complex64> =
                    FaerMat::from_fn(nrows, ncols, |i, j| m_matrix[(i, j)]);
                let (t1_svd, t2_svd, eps_svd) =
                    Self::svd_truncate_merged(&m_faer, d_l1, d_r2, bond_dim);
                *t1 = t1_svd;
                *t2 = t2_svd;
                return eps_svd;
            }
            let qr = m_matrix.qr();
            let q_full = qr.q();
            let r_full = qr.r();

            // Rank-revealing via R row-norms (see faer-path comment above).
            let k_max = std::cmp::min(bond_dim, std::cmp::min(nrows, ncols));
            let mut frob_sq = 0.0_f64;
            let mut frob_sq_full = 0.0_f64;
            for i in 0..k_max {
                for j in i..ncols {
                    let nsq = r_full[(i, j)].norm_sqr();
                    frob_sq += nsq;
                    frob_sq_full += nsq;
                }
            }
            let rank_max = std::cmp::min(nrows, ncols);
            for i in k_max..rank_max {
                for j in i..ncols {
                    frob_sq_full += r_full[(i, j)].norm_sqr();
                }
            }
            let frob = frob_sq.sqrt().max(1e-300_f64);
            let row_eps_sq = (1e-12_f64 * frob).powi(2);
            let mut k_final = 1usize;
            for i in 0..k_max {
                let mut row_norm_sq = 0.0_f64;
                for j in i..ncols {
                    row_norm_sq += r_full[(i, j)].norm_sqr();
                }
                if row_norm_sq > row_eps_sq {
                    k_final = i + 1;
                }
            }
            // Kept weight Σ_{i<k_final} ||R[i, i:]||².
            let mut kept_sq = 0.0_f64;
            for i in 0..k_final {
                for j in i..ncols {
                    kept_sq += r_full[(i, j)].norm_sqr();
                }
            }
            // Discarded weight = rows beyond k_max (bond_dim truncation)
            // plus phantom rows dropped below the threshold.
            let discarded_sq = (frob_sq_full - kept_sq).max(0.0);
            let eps = discarded_sq / frob_sq_full.max(1e-300_f64);

            let q_dst: nalgebra::DMatrix<Complex64> = q_full.columns(0, k_final).into_owned();
            let r_dst: nalgebra::DMatrix<Complex64> = r_full.rows(0, k_final).into_owned();
            (q_dst, r_dst, eps)
        };

        *t1 = q_out;
        let k_final = r_matrix.shape().0;

        let mut new_t2 = nalgebra::DMatrix::zeros(k_final * 2, d_r2);
        for i in 0..k_final {
            for q_out in 0..2 {
                for r in 0..d_r2 {
                    new_t2[(i + q_out * k_final, r)] = r_matrix[(i, r + q_out * d_r2)];
                }
            }
        }
        *t2 = new_t2;
        eps
    }

    /// Optimal rank-k truncation of the merged two-site tensor via thin SVD.
    ///
    /// `m` is the (2·D_L, 2·D_R) merged tensor in the same packing used by
    /// [`Self::apply_2q_gate_static`].  On return `t1` is (2·D_L, k) holding
    /// the left factor (orthonormal to machine precision) and `t2` is the
    /// packed (2·k, D_R) tensor holding the right factor, where
    /// k = min(bond_dim, #σ > 1e-12·‖M‖_F).  Also returns the
    /// discarded-weight fraction ε, computed from directly *measured*
    /// weights (never from a backend's reported σ) so callers report
    /// honest truncation instead of a position-cut estimate.
    ///
    /// The left factor is built from the measured action of `m` on the
    /// right-singular columns of `v()`: column j is M·v_c / ‖M·v_c‖ and the
    /// matching `t2` row is ‖M·v_c‖·conj(v_c).  The pair therefore
    /// reproduces M·(V_k V_k^H) exactly, for any v set — self-consistent by
    /// construction.  `svd.u()` is deliberately never consulted: faer
    /// 0.20.2's complex SVD returns a `u()` inconsistent with
    /// (`s_diagonal()`, `v()`) whenever the spectrum contains exactly
    /// degenerate or exactly-zero singular values (minimal repro: 16×16
    /// complex, σ = [√2 ×8, 0 ×8] ⇒ ‖M − U Σ V^H‖_F/‖M‖_F ≈ 6.4e-1 while
    /// u/v are individually orthonormal; see examples/svd_probe4.rs and
    /// svd_probe5.rs, surfaced by the SF_MPS_SVD_DEBUG self-check on the
    /// 26q brick12@cap64 run).  In that same case the v columns and the
    /// values ‖M·v_c‖ are consistent to machine precision.
    ///
    /// faer does not guarantee a descending σ order, so the kept indices are
    /// selected by explicit magnitude sort.  The reported σ themselves are
    /// unreliable on degenerate spectra (phantom zeros, and a σ-honest but
    /// σ↔v-mismatched variant seen on routed Clifford merges), so the pair
    /// built from the σ-selected columns is verified by its measured
    /// retained mass: if a τ stop leaves more than 1e-12·‖M‖²_F outside the
    /// selected span — impossible for an honest stop — the kept columns are
    /// re-selected by the measured ‖M·v_c‖ instead; the factors and ε always
    /// carry only measured values.
    fn svd_truncate_merged(
        m: &FaerMat<Complex64>,
        d_l1: usize,
        d_r2: usize,
        bond_dim: usize,
    ) -> (
        nalgebra::DMatrix<Complex64>,
        nalgebra::DMatrix<Complex64>,
        f64,
    ) {
        let nrows = 2 * d_l1;
        let ncols = 2 * d_r2;
        let r = std::cmp::min(nrows, ncols);

        let svd = ThinSvd::new(m.as_ref());
        let s = svd.s_diagonal();

        // ‖M‖_F² measured directly from the tensor entries (column-major
        // sweep) rather than summed from the reported σ: keeps both the
        // phantom threshold and the reported ε honest even if a backend
        // ever returns σ inconsistent with the matrix itself.
        let mut total = 0.0_f64;
        for c in 0..ncols {
            for i in 0..nrows {
                total += m.read(i, c).norm_sqr();
            }
        }
        let total = total.max(1e-300_f64);
        let tau = 1e-12_f64 * total.sqrt();

        // Select the kept singular triplets: largest first, stop at the bond
        // cap or at the first value below the phantom threshold.
        let mut idx: Vec<usize> = (0..r).collect();
        idx.sort_by(|&a, &b| {
            let sa = s.read(a).norm();
            let sb = s.read(b).norm();
            sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut k = 0usize;
        let mut stopped_by_tau = false;
        for &i in &idx {
            let si = s.read(i).norm();
            if k >= bond_dim {
                break;
            }
            if si <= tau {
                stopped_by_tau = true;
                break;
            }
            k += 1;
        }
        if k == 0 {
            // σ_max ≥ ‖M‖_F/√r > τ always holds, but stay defensive: never
            // return a zero-bond tensor pair.
            k = 1;
        }
        let v = svd.v();

        let mut sel: Vec<usize> = idx[..k].to_vec();

        // ── Factor construction: v-side only (see doc comment) ──
        // w_c = M·v_c by direct contraction; the left column is w_c/‖w_c‖
        // and the right row is ‖w_c‖·conj(v_c), so the pair equals
        // M·(V_k V_k^H) — no backend u() factor involved.  Returns
        // (t1, t2, kept) where kept = Σ_{sel}‖M·v_c‖², the measured
        // retained mass.
        let build = |sel: &[usize]| {
            let k = sel.len();
            let mut t1 = nalgebra::DMatrix::<Complex64>::zeros(nrows, k);
            let mut t2 = nalgebra::DMatrix::<Complex64>::zeros(2 * k, d_r2);
            let mut kept = 0.0_f64;
            for (j, &ci) in sel.iter().enumerate() {
                let mut w = vec![Complex64::new(0.0, 0.0); nrows];
                for c in 0..ncols {
                    let vc = v.read(c, ci);
                    for i in 0..nrows {
                        w[i] += m.read(i, c) * vc;
                    }
                }
                let nrm_sq: f64 = w.iter().map(|x| x.norm_sqr()).sum();
                kept += nrm_sq;
                if nrm_sq > 0.0 {
                    let snorm = nrm_sq.sqrt();
                    let inv = 1.0 / snorm;
                    for i in 0..nrows {
                        t1[(i, j)] = w[i] * inv;
                    }
                    let sj = Complex64::new(snorm, 0.0);
                    for q_out in 0..2usize {
                        for rr in 0..d_r2 {
                            t2[(q_out * k + j, rr)] =
                                sj * v.read(q_out * d_r2 + rr, ci).conj();
                        }
                    }
                }
                // ‖M·v_c‖ = 0 (degenerate phantom column): both factors
                // stay zero — the column contributes nothing and the
                // reconstruction remains exact.
            }
            (t1, t2, kept)
        };

        // ── Degenerate-spectrum guard (measured-loss trigger) ──
        // faer 0.20.2's reported σ are unreliable on (near-)degenerate
        // spectra: u() is inconsistent with (s_diagonal(), v()) whenever the
        // spectrum contains exactly degenerate or exactly-zero singular
        // values (minimal repro: 16×16 complex, σ = [√2 ×8, 0 ×8] ⇒
        // ‖M − U Σ V^H‖_F/‖M‖_F ≈ 6.4e-1 with u/v individually orthonormal;
        // see examples/svd_probe4.rs and svd_probe5.rs).  A second, subtler
        // pathology was observed on routed Clifford merges at bond 64
        // (SF_MPS_SVD_DEBUG seq=195 of the n=16 seed=0 stream: M = 32×64,
        // σ_max = √2, σ_cut = 0, ε = 1.03e-1): the reported σ² DO sum to
        // the measured ‖M‖²_F and the sort/stop is σ-honest, yet the σ↔v
        // pairing is wrong — the selected columns' measured action ‖M·v_c‖
        // carries only a fraction of the mass, so the τ stop undercounts
        // the kept rank.  An honest τ stop can lose at most O(r·τ²) =
        // O(r·1e-24)·‖M‖²_F, so the pass-1 measurement decides: if more
        // than 1e-12·‖M‖²_F is missing, re-select the kept columns by the
        // measured ‖M·v_c‖ (the quantity the factors and ε already use),
        // capped by bond_dim and τ on the measured values, and rebuild.
        let (mut t1, mut t2, mut kept) = build(&sel);
        let mut guarded = false;
        if stopped_by_tau && k < r && (total - kept) > 1e-12_f64 * total {
            guarded = true;
            // W = M·V (all r right-singular columns) in one matmul.
            let par = if r >= 64 {
                Parallelism::Rayon(0)
            } else {
                Parallelism::None
            };
            let mut w = FaerMat::<Complex64>::zeros(nrows, r);
            faer_matmul(
                w.as_mut(),
                m.as_ref(),
                v.as_ref(),
                None,
                Complex64::new(1.0, 0.0),
                par,
            );
            let mut meas: Vec<(usize, f64)> = (0..r)
                .map(|c| {
                    let mut nrm_sq = 0.0_f64;
                    for i in 0..nrows {
                        nrm_sq += w.read(i, c).norm_sqr();
                    }
                    (c, nrm_sq)
                })
                .collect();
            meas.sort_by(|a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            let mut sel2: Vec<usize> = Vec::with_capacity(std::cmp::min(bond_dim, r));
            for &(c, nrm_sq) in &meas {
                if sel2.len() >= bond_dim {
                    break;
                }
                if nrm_sq.sqrt() <= tau {
                    break;
                }
                sel2.push(c);
            }
            if sel2.is_empty() {
                sel2.push(meas[0].0);
            }
            if sel2 != sel {
                let (t1b, t2b, keptb) = build(&sel2);
                sel = sel2;
                t1 = t1b;
                t2 = t2b;
                kept = keptb;
            }
        }
        let k = sel.len();
        let eps = ((total - kept) / total).clamp(0.0, 1.0);

        // ── Debug-only self-check (env SF_MPS_SVD_DEBUG=1) ──
        // Additive instrumentation: for every call, validates that the packed
        // truncation reproduces the merged tensor M under the same packing the
        // QR tail uses (t2 row = q_out·k + j), and that the retained
        // left-factor columns are orthonormal.  Reports σ_max, the σ at the
        // cut, k, ε and both errors on stderr.  The default path never
        // consults this.
        if std::env::var_os("SF_MPS_SVD_DEBUG").is_some() {
            use std::sync::atomic::{AtomicUsize, Ordering};
            static SEQ: AtomicUsize = AtomicUsize::new(0);
            let seq = SEQ.fetch_add(1, Ordering::Relaxed);
            let mut num = 0.0_f64;
            let mut den = 0.0_f64;
            for i in 0..nrows {
                for c in 0..ncols {
                    let q_out = c / d_r2.max(1);
                    let rr = c % d_r2.max(1);
                    let mut acc = Complex64::new(0.0, 0.0);
                    for j in 0..k {
                        acc += t1[(i, j)] * t2[(q_out * k + j, rr)];
                    }
                    num += (acc - m.read(i, c)).norm_sqr();
                    den += m.read(i, c).norm_sqr();
                }
            }
            let mut gram_err = 0.0_f64;
            for a in 0..k {
                for b in 0..k {
                    let mut ip = Complex64::new(0.0, 0.0);
                    for i in 0..nrows {
                        ip += t1[(i, a)].conj() * t1[(i, b)];
                    }
                    let want = if a == b {
                        Complex64::new(1.0, 0.0)
                    } else {
                        Complex64::new(0.0, 0.0)
                    };
                    gram_err += (ip - want).norm_sqr();
                }
            }
            let s_cut = if k < idx.len() {
                s.read(idx[k]).norm()
            } else {
                0.0
            };
            eprintln!(
                "[svd-dbg] seq={seq} M={nrows}x{ncols} k={k} smax={:.3e} scut={:.3e} \
                 eps={:.3e} recon={:.3e} ugram={:.3e} guard={guarded}",
                s.read(idx[0]).norm(),
                s_cut,
                eps,
                (num / den.max(1e-300)).sqrt(),
                gram_err.sqrt()
            );

            // Extended accounting for the reported-σ vs measured-action
            // mismatch: rep = Σ all reported σ², kept = measured retained
            // mass, n_gt_tau = #columns with measured ‖M·v_c‖ > τ, and
            // vgram = ‖V^H·V − I‖_F verifies faer's v columns are
            // orthonormal.  Lossy merges (ε > 1e-6) are additionally dumped
            // (M, V, σ, measured, sel) as _svd_dbg_*_{seq}.* for offline
            // analysis.
            let rep_total: f64 = (0..r).map(|j| s.read(j).norm_sqr()).sum();
            let mut meas_sq: Vec<f64> = Vec::with_capacity(r);
            let mut n_gt_tau = 0usize;
            let mut max_meas = 0.0_f64;
            for ci in 0..r {
                let mut nrm_sq = 0.0_f64;
                for i in 0..nrows {
                    let mut acc = Complex64::new(0.0, 0.0);
                    for c in 0..ncols {
                        acc += m.read(i, c) * v.read(c, ci);
                    }
                    nrm_sq += acc.norm_sqr();
                }
                if nrm_sq.sqrt() > tau {
                    n_gt_tau += 1;
                }
                if nrm_sq > max_meas {
                    max_meas = nrm_sq;
                }
                meas_sq.push(nrm_sq);
            }
            let mut vgram = 0.0_f64;
            for a in 0..r {
                for b in 0..r {
                    let mut ip = Complex64::new(0.0, 0.0);
                    for c in 0..ncols {
                        ip += v.read(c, a).conj() * v.read(c, b);
                    }
                    let want = if a == b {
                        Complex64::new(1.0, 0.0)
                    } else {
                        Complex64::new(0.0, 0.0)
                    };
                    vgram += (ip - want).norm_sqr();
                }
            }
            eprintln!(
                "[svd-dbg2] seq={seq} total={total:.6e} rep={rep_total:.6e} \
                 kept={kept:.6e} n_gt_tau={n_gt_tau}/{r} max_meas={max_meas:.6e} \
                 vgram={:.3e}",
                vgram.sqrt()
            );
            if eps > 1e-6 {
                let mut mbin: Vec<u8> = Vec::with_capacity(nrows * ncols * 16);
                for c in 0..ncols {
                    for i in 0..nrows {
                        let z = m.read(i, c);
                        mbin.extend_from_slice(&z.re.to_le_bytes());
                        mbin.extend_from_slice(&z.im.to_le_bytes());
                    }
                }
                let mut vbin: Vec<u8> = Vec::with_capacity(ncols * r * 16);
                for c in 0..ncols {
                    for j in 0..r {
                        let z = v.read(c, j);
                        vbin.extend_from_slice(&z.re.to_le_bytes());
                        vbin.extend_from_slice(&z.im.to_le_bytes());
                    }
                }
                let sigma_rep: Vec<f64> = (0..r).map(|j| s.read(j).norm()).collect();
                let meta = format!(
                    "nrows={nrows} ncols={ncols} r={r} k={k} bond_dim={bond_dim} \
                     eps={eps:.6e} total={total:.6e} rep_total={rep_total:.6e} \
                     kept={kept:.6e} guard={guarded} stopped_by_tau={stopped_by_tau}\n\
                     sigma_reported={sigma_rep:?}\nmeasured_sq={meas_sq:?}\nsel={sel:?}\n"
                );
                let _ = std::fs::write(format!("_svd_dbg_meta_{seq}.txt"), meta);
                let _ = std::fs::write(format!("_svd_dbg_M_{seq}.bin"), &mbin);
                let _ = std::fs::write(format!("_svd_dbg_V_{seq}.bin"), &vbin);
            }
        }

        (t1, t2, eps)
    }

    pub fn to_statevector(&self) -> Vec<Complex64> {
        // Convert MPS to full statevector
        let dim = 1 << self.n_qubits;
        let mut state = vec![Complex64::new(0.0, 0.0); dim];
        state[0] = Complex64::new(1.0, 0.0);

        // Final statevector reconstruction by contracting tensors correctly
        // This is O(2^N * D^2), only for testing small circuits!
        for i in 0..dim {
            let mut left_vec = nalgebra::DVector::from_vec(vec![Complex64::new(1.0, 0.0)]);
            for idx in 0..self.n_qubits {
                let t = &self.tensors[idx];
                let (d_l_rows, d_r) = t.shape();
                let d_l = d_l_rows / 2;
                let bit = (i >> idx) & 1;

                let mut next_vec = nalgebra::DVector::zeros(d_r);
                for r in 0..d_r {
                    let mut sum = Complex64::new(0.0, 0.0);
                    for l in 0..d_l {
                        sum += left_vec[l] * t[(l + bit * d_l, r)];
                    }
                    next_vec[r] = sum;
                }
                left_vec = next_vec;
            }
            state[i] = left_vec[0];
        }

        state
    }

    /// Direct bit-by-bit sampling from MPS: O(N * D^2 * shots)
    pub fn sample(&self, shots: usize, _seed: u64) -> std::collections::HashMap<String, usize> {
        let mut counts = std::collections::HashMap::new();
        if self.n_qubits == 0 {
            return counts;
        }

        // Final measurement simulation (linear bit-by-bit)
        // For performance, we pre-calculate bond-bond contractions
        let _master_rng = StdRng::seed_from_u64(_seed);

        for shot_idx in 0..shots {
            let mut bitstring = String::with_capacity(self.n_qubits);
            let mut left_vec = DVector::from_vec(vec![Complex64::new(1.0, 0.0)]); // D_L1 = 1

            // Unique seed for this shot to ensure diversity
            let mut rng = StdRng::seed_from_u64(_seed.wrapping_add(shot_idx as u64));

            for i in 0..self.n_qubits {
                let t = &self.tensors[i];
                let d_l = t.shape().0 / 2;
                let d_r = t.shape().1;

                // Compute unnormalized P(0) and P(1) for this qubit
                let mut prob0 = 0.0_f64;
                for r in 0..d_r {
                    let mut sum = Complex64::new(0.0, 0.0);
                    for l in 0..d_l {
                        sum += left_vec[l] * t[(l, r)];
                    }
                    prob0 += sum.norm_sqr();
                }
                let mut prob1 = 0.0_f64;
                for r in 0..d_r {
                    let mut sum = Complex64::new(0.0, 0.0);
                    for l in 0..d_l {
                        sum += left_vec[l] * t[(l + d_l, r)];
                    }
                    prob1 += sum.norm_sqr();
                }

                // Normalize: P(0) = p0 / (p0 + p1)
                let total = prob0 + prob1;
                let p0_normalized = if total > 1e-30 { prob0 / total } else { 0.5 };

                let random_val: f64 = rng.gen();
                let b = if random_val < p0_normalized { '0' } else { '1' };
                bitstring.push(b);

                // Update left_vec (projection)
                let p_idx = if b == '0' { 0 } else { 1 };
                let mut next_vec = DVector::zeros(d_r);
                for r in 0..d_r {
                    let mut sum = Complex64::new(0.0, 0.0);
                    for l in 0..d_l {
                        sum += left_vec[l] * t[(l + p_idx * d_l, r)];
                    }
                    next_vec[r] = sum;
                }
                // Normalize to avoid underflow
                let norm = next_vec.norm();
                if norm > 1e-12_f64 {
                    let inv_norm = 1.0 / norm;
                    for elem in next_vec.iter_mut() {
                        *elem *= inv_norm;
                    }
                }
                left_vec = next_vec;
            }
            // Emit bitstrings q0-last (qubit 0 = rightmost char), matching the
            // statevector backend's sample() convention.
            let key: String = bitstring.chars().rev().collect();
            *counts.entry(key).or_insert(0) += 1;
        }
        counts
    }
}

impl MPSState {
    /// Boundary contraction <psi|P|psi> where P is a tensor product of
    /// single-site Paulis (encoded as one byte per site, 0=I, 1=X, 2=Y, 3=Z).
    /// Cost: O(n * chi^2) — beats the Python implementation by 10-100x
    /// because the inner per-site contraction is in Rust without Python
    /// per-site overhead.
    pub fn pauli_expval(&self, pauli: &[u8]) -> Complex64 {
        assert!(
            pauli.len() <= self.n_qubits,
            "pauli string length {} exceeds n_qubits {}",
            pauli.len(),
            self.n_qubits
        );
        // Prefix semantics (statevector-compatible): a shorter Pauli string
        // acts on the first len(pauli) qubits; the remaining sites are I.
        let mut pauli_full = vec![0u8; self.n_qubits];
        pauli_full[..pauli.len()].copy_from_slice(pauli);
        let pauli = pauli_full.as_slice();

        // Carry a complex matrix L of shape (D_top, D_bot) where:
        //   D_top = bond at top of bra
        //   D_bot = bond at top of ket
        // Initial: 1x1 identity (empty product)
        let mut left: DMatrix<Complex64> = DMatrix::from_element(1, 1, Complex64::new(1.0, 0.0));

        for i in 0..self.n_qubits {
            let t = &self.tensors[i]; // shape (d_l*2, d_r)
            let (rows, _d_r) = t.shape();
            let d_l = rows / 2;
            // Split t into top (physical=0) and bottom (physical=1) halves.
            // top: rows 0..d_l, bottom: rows d_l..2*d_l
            let top = t.rows(0, d_l).clone_owned();
            let bot = t.rows(d_l, d_l).clone_owned();

            // Apply Pauli at this site:
            // Pauli I:  out_top = top   ; out_bot = bot
            // Pauli X:  out_top = bot   ; out_bot = top         (swap top/bot)
            // Pauli Y:  out_top = -i*bot; out_bot = i*top
            // Pauli Z:  out_top = top   ; out_bot = -bot
            let (a, b) = match pauli[i] {
                0 => (top.clone(), bot.clone()), // I
                1 => (bot.clone(), top.clone()), // X
                2 => (
                    bot.clone() * Complex64::new(0.0, -1.0),
                    top.clone() * Complex64::new(0.0, 1.0),
                ), // Y
                3 => (top.clone(), -bot.clone()), // Z
                _ => (top.clone(), bot.clone()),
            };

            // Compute new left matrix:
            // L_new[r, r'] = sum_{l, l', s} conj(top_or_bot_bra[l,r]) * L[l,l'] * (P*ket)[l',s,r']
            // where the s sum is implicit in our split (we already apply P).
            //
            // Equivalent: form combined ket' = stack(a, b) of shape (2*d_l, d_r)
            //  (so ket'[l + s*d_l, r] = (P*ket)[l, s, r])
            // bra: stack(top.conj(), bot.conj()) of shape (2*d_l, d_r)
            // Contract over (s, l_bra=l_ket, l'_bra=l'_ket): we sum the two
            // physical s sectors.
            //
            // We use:  L_new = bra_top.adjoint() * L * a + bra_bot.adjoint() * L * b
            let bra_top_adj = top.adjoint(); // (d_r, d_l)
            let bra_bot_adj = bot.adjoint(); // (d_r, d_l)
            let l_a = &left * &a; // (D_top, d_r)  — wait, dims off
                                  // Actually we need: L is (d_l_bra_prev, d_l_ket_prev). After site i,
                                  // new L should be (d_r_bra, d_r_ket).
                                  // Standard formula:
                                  //   L_new[r_bra, r_ket] = sum_{s, l_bra, l_ket}
                                  //       conj(bra_t_s[l_bra, r_bra]) * L[l_bra, l_ket] * a_or_b_s[l_ket, r_ket]
                                  // where bra_t_s is the bra tensor at this site for physical s, and
                                  // a_or_b_s is the ket tensor with the Pauli applied (here `a` is s=0
                                  // contribution, `b` is s=1 contribution).
                                  //
                                  // We can compute as: L_new = bra_top.adjoint() * L * a + bra_bot.adjoint() * L * b
                                  // bra_top: (d_l, d_r), so bra_top.adjoint(): (d_r, d_l)
                                  // L: (d_l_bra, d_l_ket)  --> after first site it's (1,1), generally (d_l, d_l)
                                  // a: (d_l_ket, d_r_ket) -- shape (d_l, d_r) since we sliced from t
                                  // (d_r, d_l) * (d_l, d_l) * (d_l, d_r) = (d_r, d_r). Correct.
            let _ = l_a; // silence warning; we'll redo below cleanly
            let term0 = &bra_top_adj * &left * &a;
            let term1 = &bra_bot_adj * &left * &b;
            left = term0 + term1;
        }

        // Final left should be 1x1 — that's the expectation value.
        left[(0, 0)]
    }
}

impl QuantumDAG {
    /// Evolve the circuit into an MPS state.
    pub fn evolve_mps(&self, bond_dim: usize) -> MPSState {
        let mut state = MPSState::new(self.n_qubits, bond_dim);
        self._evolve_into(&mut state);
        state
    }

    /// Run MPS evolution for `circuit`, then compute <psi|P|psi> where P is a
    /// tensor product of single-site Paulis (`pauli` is one byte per site).
    /// Bond dimension capped at `bond_dim`.
    pub fn simulate_mps_pauli_expval(&self, bond_dim: usize, pauli: &[u8]) -> Complex64 {
        let mut state = MPSState::new(self.n_qubits, bond_dim);
        self._evolve_into(&mut state);
        state.pauli_expval(pauli)
    }

    /// Batched Pauli expectation: evolve the MPS ONCE, then contract every
    /// Pauli string in `paulis` against the same evolved state.  For
    /// Hamiltonians with N Pauli terms this is N× faster than calling
    /// `simulate_mps_pauli_expval` in a loop, because the (expensive)
    /// gate-sweep evolution happens just once.
    ///
    /// Each `pauli` is one byte per site (0=I, 1=X, 2=Y, 3=Z).  Returns a
    /// vector of `Complex64` parallel to `paulis`.
    pub fn simulate_mps_pauli_expval_batch(
        &self,
        bond_dim: usize,
        paulis: &[Vec<u8>],
    ) -> Vec<Complex64> {
        let mut state = MPSState::new(self.n_qubits, bond_dim);
        self._evolve_into(&mut state);
        paulis.iter().map(|p| state.pauli_expval(p)).collect()
    }

    /// Sequential lazy-SWAP evolution: processes gates in topological order,
    /// routing non-adjacent 2q gates by moving the far qubit leftward without
    /// swapping back. O(n^2) adjacent gate ops total for QFT vs O(n^3) for
    /// SWAP-back routing.
    pub fn _evolve_into_lazy(&self, state: &mut MPSState) {
        let order = self.topological_order();
        for &node_id in &order {
            let op = &self.graph()[node_id];
            if op.op_type.is_boundary()
                || op.op_type == OpType::Barrier
                || op.op_type.is_measurement()
            {
                continue;
            }
            let gate_u = op.op_type.to_matrix();
            match op.qubits.len() {
                1 => {
                    let phys = state.perm[op.qubits[0]];
                    state.apply_1q_gate(phys, &gate_u);
                }
                2 => {
                    state.apply_gate_lazy_routed(op.qubits[0], op.qubits[1], &gate_u);
                }
                _ => {}
            }
        }
    }

    /// Evolve the MPS state in-place by walking gates in topological order.
    /// For circuits with long-range 2q gates (e.g. QFT), dispatches to the
    /// lazy-SWAP sequential path. Otherwise parallelizes qubit-disjoint gates.
    pub fn _evolve_into(&self, state: &mut MPSState) {
        // Detect any non-adjacent 2q gate — if present, use lazy-SWAP routing
        let has_long_range = self.graph().node_indices().any(|n| {
            let op = &self.graph()[n];
            !op.op_type.is_boundary()
                && op.qubits.len() == 2
                && (op.qubits[0] as isize - op.qubits[1] as isize).abs() > 1
        });
        if has_long_range {
            self._evolve_into_lazy(state);
            return;
        }

        use rayon::prelude::*;
        let layers = self.parallel_layers();
        let bond_dim = state.bond_dim;

        for layer in layers {
            // Group gates by whether they are routed (expensive, potentially
            // touching many qubits) or local/adjacent (fast, easy to parallelize).
            // For Heisenberg/QFT almost all gates are adjacent or near-adjacent.

            // To safely use Rayon on the tensors Vec, we use unsafe raw pointers
            // because we know that gates in a DAG layer are qubit-disjoint.
            let tensors_ptr = state.tensors.as_mut_ptr() as usize;

            // Collect the per-gate discarded fraction, then accumulate into
            // `state` outside the parallel region.
            let layer_discards: Vec<f64> = layer
                .par_iter()
                .map(|&node_id| -> f64 {
                    let op = &self.graph()[node_id];
                    if op.op_type.is_boundary()
                        || op.op_type == OpType::Barrier
                        || op.op_type.is_measurement()
                    {
                        return 0.0;
                    }
                    let gate_u = op.op_type.to_matrix();

                    unsafe {
                        let ptr = tensors_ptr as *mut nalgebra::DMatrix<Complex64>;
                        match op.qubits.len() {
                            1 => {
                                let t = &mut *ptr.add(op.qubits[0]);
                                MPSState::apply_1q_gate_static(t, &gate_u);
                                0.0
                            }
                            2 => {
                                let q1 = op.qubits[0];
                                let q2 = op.qubits[1];
                                let dist = (q1 as isize - q2 as isize).abs();
                                if dist == 1 {
                                    let (low_q, high_q) = if q1 < q2 { (q1, q2) } else { (q2, q1) };
                                    let t1 = &mut *ptr.add(low_q);
                                    let t2 = &mut *ptr.add(high_q);
                                    // Safety: low_q and high_q are disjoint from other gates in this layer.
                                    MPSState::apply_2q_gate_static(
                                        t1, t2, q1, q2, &gate_u, bond_dim,
                                    )
                                } else {
                                    // For non-adjacent routed gates, parallelization is trickier
                                    // because they touch a range of qubits.
                                    // For now, we fallback to a sequential path for routed gates
                                    // or just accept the risk if the DAG says they are disjoint.
                                    // Actually, a routed gate acts on many qubits, so it should
                                    // have edges to all of them in the DAG, meaning no other
                                    // gate touching those qubits can be in this layer.
                                    // So even routed gates are safe!
                                    // But `apply_2q_gate_routed` is not static.
                                    // We'll just skip parallelizing routed gates for now to be safe,
                                    // or implement a static version.
                                    0.0
                                }
                            }
                            _ => 0.0,
                        }
                    }
                })
                .collect();
            for eps in layer_discards {
                state.record_discard(eps);
            }

            // Handle routed gates sequentially for now (rare in optimized Heisenberg/QFT)
            for &node_id in &layer {
                let op = &self.graph()[node_id];
                if op.qubits.len() == 2 && (op.qubits[0] as isize - op.qubits[1] as isize).abs() > 1
                {
                    let gate_u = op.op_type.to_matrix();
                    state.apply_2q_gate_routed(op.qubits[0], op.qubits[1], &gate_u);
                }
            }
        }
    }

    pub fn simulate_mps(&self, bond_dim: usize) -> Vec<Complex64> {
        let mut state = MPSState::new(self.n_qubits, bond_dim);
        let order = self.topological_order();

        for &node_id in &order {
            let op = &self.graph()[node_id];
            if op.op_type.is_boundary() || op.op_type == OpType::Barrier {
                continue;
            }

            let gate_u = op.op_type.to_matrix();
            if op.qubits.len() == 1 {
                state.apply_1q_gate(op.qubits[0], &gate_u);
            } else if op.qubits.len() == 2 {
                // Use routed 2q gate so non-adjacent qubits work too.
                state.apply_2q_gate_routed(op.qubits[0], op.qubits[1], &gate_u);
            }
        }
        state.to_statevector()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::OpType;
    use approx::assert_relative_eq;

    fn mps_norm_sq(state: &MPSState) -> f64 {
        state.to_statevector().iter().map(|c| c.norm_sqr()).sum()
    }

    #[test]
    fn test_bell_state_norm() {
        let mut mps = MPSState::new(2, 64);
        let h = OpType::H.to_matrix();
        let cnot = OpType::CNOT.to_matrix();

        mps.apply_1q_gate(0, &h);
        mps.apply_2q_gate(0, 1, &cnot);

        assert_relative_eq!(mps_norm_sq(&mps), 1.0, epsilon = 1e-8);
    }

    #[test]
    fn test_ghz_state_norm() {
        let mut mps = MPSState::new(4, 64);
        let h = OpType::H.to_matrix();
        let cnot = OpType::CNOT.to_matrix();

        mps.apply_1q_gate(0, &h);
        for i in 0..3 {
            mps.apply_2q_gate(i, i + 1, &cnot);
        }

        assert_relative_eq!(mps_norm_sq(&mps), 1.0, epsilon = 1e-8);
    }

    #[test]
    fn test_bell_state_zz_expval() {
        let mut mps = MPSState::new(2, 64);
        let h = OpType::H.to_matrix();
        let cnot = OpType::CNOT.to_matrix();

        mps.apply_1q_gate(0, &h);
        mps.apply_2q_gate(0, 1, &cnot);

        // Pauli bytes: 0=I, 1=X, 2=Y, 3=Z
        let zz = mps.pauli_expval(&[3, 3]);
        assert_relative_eq!(zz.re, 1.0, epsilon = 1e-8);
        assert!(zz.im.abs() < 1e-8);
    }

    #[test]
    fn test_mps_initial_state_is_zero() {
        let mps = MPSState::new(2, 64);
        let sv = mps.to_statevector();
        assert_relative_eq!(sv[0].re, 1.0, epsilon = 1e-10);
        assert!(sv[1].norm() < 1e-10);
        assert!(sv[2].norm() < 1e-10);
        assert!(sv[3].norm() < 1e-10);
    }

    #[test]
    fn test_truncation_report_exact_small() {
        // Adequate bond dimension: every step is exact, nothing discarded.
        let mut mps = MPSState::new(3, 64);
        let h = OpType::H.to_matrix();
        let cnot = OpType::CNOT.to_matrix();
        mps.apply_1q_gate(0, &h);
        mps.apply_2q_gate(0, 1, &cnot);
        mps.apply_2q_gate(1, 2, &cnot);

        assert_eq!(mps.truncation_events, 0);
        assert_relative_eq!(mps.discarded_weight, 0.0, epsilon = 1e-18);
        assert_relative_eq!(mps.fidelity_lower_bound(), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn test_truncation_report_detects_bond_cap() {
        // A Bell pair needs bond dimension 2; forcing bond_dim=1 must
        // discard weight — and report it instead of failing silently.
        let mut mps = MPSState::new(2, 1);
        let h = OpType::H.to_matrix();
        let cnot = OpType::CNOT.to_matrix();
        mps.apply_1q_gate(0, &h);
        mps.apply_2q_gate(0, 1, &cnot);

        assert!(mps.truncation_events >= 1);
        assert!(mps.discarded_weight > 0.0);
        assert!(mps.discarded_weight < 1.0);
        assert!(mps.max_discarded_weight >= mps.discarded_weight / mps.truncation_events as f64);
        assert_relative_eq!(
            mps.fidelity_lower_bound(),
            (1.0 - mps.discarded_weight).clamp(0.0, 1.0),
            epsilon = 1e-15
        );
        // The surviving norm matches the reported bound (≈ 1 − discarded).
        assert_relative_eq!(
            mps_norm_sq(&mps),
            mps.fidelity_lower_bound(),
            epsilon = 1e-9
        );
    }

    #[test]
    fn test_truncation_report_exact_small_after_override() {
        // Even with the optimal-SVD override active (default policy), a step
        // that discards nothing must stay bit-identical to the QR path: the
        // override only fires when the position cut would lose > 1e-14.
        let mut mps = MPSState::new(4, 64);
        let h = OpType::H.to_matrix();
        let cnot = OpType::CNOT.to_matrix();
        mps.apply_1q_gate(0, &h);
        for i in 0..3 {
            mps.apply_2q_gate(i, i + 1, &cnot);
        }
        assert_eq!(mps.truncation_events, 0);
        assert_relative_eq!(mps.discarded_weight, 0.0, epsilon = 1e-18);
        assert_relative_eq!(mps_norm_sq(&mps), 1.0, epsilon = 1e-9);
    }

    #[test]
    fn test_svd_truncate_keeps_top_subspace() {
        // Anti-diagonal merged tensor: the σ_j direction lives in column
        // ncols-1-j, so the historical position cut (span of the FIRST k
        // columns) keeps only the smallest singular directions while the
        // optimal top-k SVD cut keeps the largest.  The numpy replica
        // (_mps_trunc_diag.py, brick12@26) identifies exactly this failure
        // mode as the 6.7e-2 of discarded weight under the QR policy.
        let sigmas = [1.0_f64, 0.5, 0.25, 0.125, 1e-3, 1e-6, 1e-12, 1e-15];
        let n = sigmas.len();
        let d = n / 2;
        let m = FaerMat::<Complex64>::from_fn(n, n, |i, j| {
            if i + j == n - 1 {
                Complex64::new(sigmas[i], 0.0)
            } else {
                Complex64::new(0.0, 0.0)
            }
        });
        let total: f64 = sigmas.iter().map(|s| s * s).sum();

        // Position cut at k = 4 keeps span(M[:, 0:4]) = span(e_4..e_7),
        // i.e. only σ_4..σ_7's weight (≈1e-6 of 1.328) — catastrophic.
        let kept_pos: f64 = sigmas[4..].iter().map(|s| s * s).sum();
        let eps_pos = 1.0 - kept_pos / total;
        assert!(eps_pos > 0.99, "position cut should be catastrophic here");

        // Optimal cut: keep the four largest singular directions.
        let (t1, t2, eps_svd) = MPSState::svd_truncate_merged(&m, d, d, 4);
        let kept_svd: f64 = sigmas[..4].iter().map(|s| s * s).sum();
        let eps_expected = 1.0 - kept_svd / total;
        assert!(
            (eps_svd - eps_expected).abs() < 1e-12,
            "eps_svd={eps_svd} expected≈{eps_expected}"
        );
        assert!(
            eps_svd * 1e5 < eps_pos,
            "SVD cut must beat the position cut by >1e5x"
        );

        // Shapes: t1 is (2·d_site, k) and isometric; t2 packs Σ_k·V_k^H.
        assert_eq!(t1.shape(), (2 * d, 4));
        assert_eq!(t2.shape(), (2 * 4, d));
        let gram = t1.adjoint() * &t1;
        for i in 0..4 {
            for j in 0..4 {
                let want = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (gram[(i, j)] - Complex64::new(want, 0.0)).norm() < 1e-12,
                    "t1 not isometric at ({i},{j}): {}",
                    gram[(i, j)]
                );
            }
        }

        // Contracting t1/t2 reproduces M minus the dropped σ_4..σ_7
        // (relative Frobenius error ≈ √(σ_4²+…)/‖M‖_F ≈ 8.7e-4).
        let k = 4usize;
        let mut num = 0.0_f64;
        for s1 in 0..2 {
            for s2 in 0..2 {
                for r1 in 0..d {
                    for r2 in 0..d {
                        let mut acc = Complex64::new(0.0, 0.0);
                        for i in 0..k {
                            acc += t1[(s1 * d + r1, i)] * t2[(s2 * k + i, r2)];
                        }
                        num += (acc - m.read(s1 * d + r1, s2 * d + r2)).norm_sqr();
                    }
                }
            }
        }
        let rel = (num / total).sqrt();
        assert!(rel < 2e-3, "relative reconstruction error too large: {rel}");
    }

    #[test]
    fn test_svd_policy_keeps_dominant_direction() {
        if !mps_truncation_svd_enabled() {
            return; // process explicitly opted into the legacy position cut
        }
        // A unitary gate whose action on |00> prepares M = [[0, α], [β, 0]]
        // with α = √(1−β²), β = 1e-3 (its first column, embedded as the
        // 2x2 output block).  At bond cap 1 the position cut keeps
        // span(M[:, 0]) (weight β² = 1e-6) while the optimal cut keeps the
        // σ = α direction (weight α² = 1−1e-6): the override must take the
        // latter, i.e. report dw ≈ 1e-6 and a near-unit surviving norm.
        let beta = 1e-3_f64;
        let alpha = (1.0 - beta * beta).sqrt();
        let mut gate = nalgebra::DMatrix::<Complex64>::zeros(4, 4);
        gate[(1, 0)] = Complex64::new(alpha, 0.0);
        gate[(2, 0)] = Complex64::new(beta, 0.0);
        gate[(1, 2)] = Complex64::new(beta, 0.0);
        gate[(2, 2)] = Complex64::new(-alpha, 0.0);
        gate[(0, 3)] = Complex64::new(1.0, 0.0);
        gate[(3, 1)] = Complex64::new(1.0, 0.0);

        let mut mps = MPSState::new(2, 1);
        mps.apply_2q_gate(0, 1, &gate);

        // Legacy position cut would report dw ≈ 1 − 1e-6 and leave a
        // 1e-6-norm survivor; the optimal cut keeps the dominant weight.
        assert!(
            mps.discarded_weight < 1e-5,
            "override did not fire: dw={}",
            mps.discarded_weight
        );
        assert!(
            mps.discarded_weight > 1e-7,
            "expected a small but nonzero discard: dw={}",
            mps.discarded_weight
        );
        assert_relative_eq!(
            mps_norm_sq(&mps),
            1.0 - mps.discarded_weight,
            epsilon = 1e-9
        );
    }
}
