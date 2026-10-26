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

/// True when the 4×4 gate preserves the left physical bit, i.e. g[o, i] = 0
/// whenever the left output bit of o differs from the left input bit of i.
/// This is exactly the condition under which the merged two-site tensor
/// factors as M = blockdiag(t1_0, t1_1)·B with a 2·min(d_l1, d_m)-entry
/// middle (see [`MPSState::blockdiag_exact_split`]).  Every diagonal gate
/// satisfies it (rzz, crz, cz, cp, cu1, crzz, cphase, ...); cx, swap, ryy,
/// rxx do not.
fn gate_left_block_diagonal(g: &DMatrix<Complex64>) -> bool {
    if g.shape() != (4, 4) {
        return false;
    }
    let zero = Complex64::new(0.0, 0.0);
    for o in 0..4usize {
        for i in 0..4usize {
            if (o >> 1) != (i >> 1) && g[(o, i)] != zero {
                return false;
            }
        }
    }
    true
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
/// * unset / `"svd"` (default) — the exact per-merge SVD path everywhere.
///   Left-bit-preserving gates whose structural rank bound
///   2·min(d_l1, d_m, d_r2) fits the cap take the exact block-diagonal
///   split ([`MPSState::blockdiag_exact_split`]; the brick12 layer-11
///   128×128 merges, ε ≈ 0); every other merge takes the exact thin SVD
///   ([`MPSState::svd_truncate_merged`]), keeping the k largest singular
///   values with k = min(bond_dim, #σ > 1e-12·‖M‖_F) and pruning sub-τ
///   directions at every step (brick12@26 cap 64: err 4.2e-6 → 3.6e-14,
///   Σε 6.7e-2 → 1.1e-16).
/// * `"qr"` / `"legacy"` — the historical position cut (unpivoted QR rows
///   0..k, i.e. span of the first k merged columns).
///
/// The earlier hybrid default routed r = min(nrows, ncols) ≤ bond_dim
/// merges around the SVD into the plain QR split; that was reverted after
/// measurement: the unpivoted Householder row-norm cut perturbs the
/// near-degenerate Schmidt structure brickwork merges produce, cascading
/// into a dirty bond ladder {34, 42, 52, 64} with 180 guard fires and
/// ε ≈ 5.2e-4 at 2.18 s on brick12@n=200 — while the all-SVD path
/// measured 1.04 s, Σε ≈ 6.7e-14 and a clean 2^k ladder.
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

/// Whether a merge of shape (nrows × ncols) at `bond_dim` must take the
/// exact-SVD path.  True for every merge under the default policy — the
/// SVD path already subsumes the exactly-representable cases (the exact
/// block-diagonal split and the τ-pruned thin SVD) and measured faster
/// than the unpivoted QR split at these shapes; false only under the
/// historical position cut (SF_MPS_TRUNCATION=qr|legacy).  The shape
/// arguments are kept for call-site readability and future policy hooks.
fn mps_svd_path_needed(_nrows: usize, _ncols: usize, _bond_dim: usize) -> bool {
    mps_truncation_svd_enabled()
}

/// Single-qubit Pauli enum used for expectation-value calls from Python.
/// Stored as one byte per site so Python can pass a raw &[u8].
///   0 = I, 1 = X, 2 = Y, 3 = Z
///
/// Matrix Product State (MPS) core for high-efficiency simulation.
#[derive(Clone)]
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
    /// Smallest site index such that every site ≥ rc_start has an
    /// isometric tail (Σ_s B^s B^s† = I over the (D_L, 2·D_R) reshape).
    /// Maintained conservatively by the gate/measure primitives so that
    /// [`Self::measure_site`] repairs only the dirty suffix window instead
    /// of right-canonicalizing the whole chain.
    pub rc_start: usize,
    /// Cached left environment for left-to-right mid-circuit measurement
    /// sequences: contraction of sites `0..env_upto` with every physical
    /// index traced out (see [`Self::measure_site`]).
    left_env: Vec<Complex64>,
    /// Number of sites folded into [`Self::left_env`].
    env_upto: usize,
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
use rand::SeedableRng;

// Canonicalization phase accumulators (debug-only, see canonicalize_right).
static __CANON_QR_US: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static __CANON_BLD_US: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static __CANON_MUL_US: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

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
            rc_start: 1,
            left_env: vec![Complex64::new(1.0, 0.0)],
            env_upto: 0,
            discarded_weight: 0.0,
            max_discarded_weight: 0.0,
            truncation_events: 0,
        }
    }

    pub fn apply_1q_gate(&mut self, target: usize, gate: &nalgebra::DMatrix<Complex64>) {
        if target < self.env_upto {
            self.invalidate_meas_env();
        }
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
        // The left sweep rewrites every tensor and guarantees no
        // right-isometric suffix, so the dirty bound becomes the empty suffix.
        self.rc_start = self.n_qubits;
        self.invalidate_meas_env();
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
    ///
    /// Deliberately not run eagerly after evolution: expectation values
    /// contract gauge-invariantly, so the sampling paths canonicalize
    /// lazily instead (see `MPSStateWrapper::sample`; `evolve_mps` never
    /// canonicalized at all).
    pub fn canonicalize_right(&mut self) {
        let __dbg = std::env::var_os("SF_MPS_SVD_DEBUG").is_some();
        let __t0 = std::time::Instant::now();
        self.canonicalize_right_range(1, self.n_qubits);
        self.rc_start = 1;
        if __dbg {
            use std::sync::atomic::Ordering::Relaxed;
            eprintln!(
                "[loop-dbg] canon n={} {}us qr={}us bld={}us mul={}us",
                self.n_qubits,
                __t0.elapsed().as_micros(),
                __CANON_QR_US.swap(0, Relaxed),
                __CANON_BLD_US.swap(0, Relaxed),
                __CANON_MUL_US.swap(0, Relaxed)
            );
        }
    }

    /// Right-canonicalize the window `from..to` (clamped to `1..n`),
    /// absorbing each LQ residue into the left neighbor.  Sites ≥ `to` must
    /// already be right-isometric (see [`Self::rc_start`]); afterwards sites
    /// `from..to` are isometric too.  Used by [`Self::measure_site`] to
    /// repair only the suffix invalidated by earlier gates/measurements.
    pub fn canonicalize_right_range(&mut self, from: usize, to: usize) {
        use std::sync::atomic::Ordering::Relaxed;
        let from = from.max(1);
        let to = to.min(self.n_qubits);
        if from >= to {
            return;
        }
        // The sweep rewrites sites from-1..to; the cached measurement
        // environment is only valid below the first touched site.
        if from <= self.env_upto {
            self.invalidate_meas_env();
        }
        for i in (from..to).rev() {
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
            let __t_ph = std::time::Instant::now();
            let mt = m.adjoint(); // (2*D_R, D_L)
            let qr = mt.qr();
            let q_t = qr.q(); // (2*D_R, min(2*D_R, D_L))
            let r_t = qr.r(); // (min(2*D_R, D_L), D_L)

            let k = std::cmp::min(self.bond_dim, std::cmp::min(2 * d_r, d_l));
            __CANON_QR_US.fetch_add(__t_ph.elapsed().as_micros() as u64, Relaxed);

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
            __CANON_BLD_US.fetch_add(__t_ph.elapsed().as_micros() as u64, Relaxed);

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
            __CANON_MUL_US.fetch_add(__t_ph.elapsed().as_micros() as u64, Relaxed);
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
        if q1.min(q2) < self.env_upto {
            self.invalidate_meas_env();
        }
        // Both rewritten sites lose their right-isometry claim: the left site
        // becomes left-isometric (norm weights moved) and the right site
        // carries the new weights.  Sites above hi are untouched and stay
        // right-isometric, so the dirty bound moves to hi + 1.
        let hi = q1.max(q2);
        if hi >= self.rc_start {
            self.rc_start = hi + 1;
        }
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

        // Wall-clock origin for the per-merge phase profile printed by the
        // SF_MPS_SVD_DEBUG blocks in both split paths.  A few Instant::now()
        // calls cost ~100ns per merge, negligible against the O(D^3) work
        // they account for.  SF_MPS_SVD_DEBUG=1 alone gives the cheap phase
        // table; SF_MPS_SVD_DEBUG_FULL=1 adds the heavy reference rebuilds.
        let __t_start = std::time::Instant::now();

        let d_l1 = t1.shape().0 / 2;
        let d_m = t1.shape().1;
        let d_r2 = t2.shape().1;

        let t1_a = t1.rows(0, d_l1).into_owned();
        let t1_b = t1.rows(d_l1, d_l1).into_owned();
        let t2_a = t2.rows(0, d_m).into_owned();
        let t2_b = t2.rows(d_m, d_m).into_owned();

        // ── Exact block-diagonal split (hybrid fast path) ──
        // A gate that preserves the left physical bit (all diagonal gates:
        // rzz, crz, cz, cp, cu1, crzz, ...) makes the merged matrix factor
        // exactly as M = blockdiag(t1_0, t1_1)·B over a middle of only
        // 2·min(d_l1, d_m) entries — rank(M) ≤ 2·min(d_l1, d_m, d_r2) — so
        // when that structural bound fits the bond cap the merge needs no
        // truncation at all and the r > bond_dim thin SVD (the brick12
        // layer-11 128×128 merges: faer ThinSvd ≈ 15.4 ms each) is replaced
        // by two (d_l1×d_m) QRs plus one small matmul.  Anything this
        // branch cannot certify falls through to the SVD path below.
        {
            let nrows = 2 * d_l1;
            let ncols = 2 * d_r2;
            let rank_bound = 2 * std::cmp::min(d_l1, d_m).min(d_r2);
            if mps_svd_path_needed(nrows, ncols, bond_dim)
                && nrows >= 16
                && ncols >= 16
                && rank_bound <= bond_dim
                && gate_left_block_diagonal(&gate_eff)
            {
                if let Some((t1_bd, t2_bd, eps_bd)) = Self::blockdiag_exact_split(
                    &t1_a, &t1_b, &t2_a, &t2_b, d_l1, d_m, d_r2, &gate_eff, bond_dim, __t_start,
                ) {
                    *t1 = t1_bd;
                    *t2 = t2_bd;
                    return eps_bd;
                }
            }
        }

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
            // the O(D^3) matmul cost justifies thread-pool overhead.  When the
            // pool has a single thread (the benchmark protocol pins
            // RAYON_NUM_THREADS=1) the dispatch is pure overhead, so skip it.
            let par = if d_m >= 64 && rayon::current_num_threads() > 1 {
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
        let zero = Complex64::new(0.0, 0.0);
        for s_out in 0..4usize {
            let g0 = gate_eff[(s_out, 0)];
            let g1 = gate_eff[(s_out, 1)];
            let g2 = gate_eff[(s_out, 2)];
            let g3 = gate_eff[(s_out, 3)];
            // Fused accumulation straight into `m_matrix` in the historical
            // (((t00·g0 + t01·g1) + t10·g2) + t11·g3) element order — the
            // same association as the old `&t00 * g0 + ...` expression, but
            // without 7 temporaries x 4 s_out (28 allocations per merge).
            // Exactly-zero (and -0.0) scalars skip their term: adding
            // 0·t is identity for finite entries.
            let row_off = (s_out >> 1) * d_l1;
            let col_off = (s_out & 1) * d_r2;
            for l in 0..d_l1 {
                for r in 0..d_r2 {
                    let mut v = t00[(l, r)] * g0;
                    if g1 != zero {
                        v += t01[(l, r)] * g1;
                    }
                    if g2 != zero {
                        v += t10[(l, r)] * g2;
                    }
                    if g3 != zero {
                        v += t11[(l, r)] * g3;
                    }
                    m_matrix[(row_off + l, col_off + r)] = v;
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

            // ── Truncation policy (default: exact SVD everywhere) ──
            // Every merge takes the exact SVD path by default — see the
            // policy doc on mps_truncation_svd_enabled.  The plain QR
            // split below is only reachable through the historical
            // position cut selected by SF_MPS_TRUNCATION=qr|legacy.
            if mps_svd_path_needed(nrows, ncols, bond_dim) {
                let (t1_svd, t2_svd, eps_svd) =
                    Self::svd_truncate_merged(&m_faer, d_l1, d_r2, bond_dim, __t_start);
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
            // Same truncation policy for small tensors — the policy must
            // not depend on which backend factorizes the merged tensor.
            if mps_svd_path_needed(nrows, ncols, bond_dim) {
                let m_faer: FaerMat<Complex64> =
                    FaerMat::from_fn(nrows, ncols, |i, j| m_matrix[(i, j)]);
                let (t1_svd, t2_svd, eps_svd) =
                    Self::svd_truncate_merged(&m_faer, d_l1, d_r2, bond_dim, __t_start);
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

    /// Exact split of the merged matrix for left-bit-preserving gates.
    ///
    /// With the merge identity of [`Self::apply_2q_gate_static`]
    ///
    ///   M[(o1,l),(o2,r)] = Σ_{i1,i2} g[2·o1+o2, 2·i1+i2]·t1_{i1}[l,·]·t2_{i2}[·,r]
    ///
    /// a gate with g[o,i] = 0 unless (o>>1) == (i>>1) collapses the sum to
    /// i1 = o1 and gives the exact factorisation
    ///
    ///   M = blockdiag(t1_0, t1_1) · B,
    ///   B[(o1,m),(o2,r)] = Σ_{i2} g[2·o1+o2, 2·o1+i2]·t2_{i2}[m,r],
    ///
    /// whose middle index carries only 2·d_m entries, so
    /// rank(M) ≤ 2·min(d_l1, d_m, d_r2).  The caller only takes this path
    /// when that bound fits the bond cap — the merge is then exactly
    /// representable and no singular values need to be computed: thin-QR
    /// each t1_p (Q_p·R_p), form C = blockdiag(R_0,R_1)·B, and drop the
    /// trailing rows of C below 1e-12·‖C‖_F (floating-point phantoms), with
    /// ε measured from the dropped row weights exactly like the SVD path.
    /// The pair (blockdiag(Q_0,Q_1)[:, :k], C[:k, :]) reproduces M to
    /// machine precision.
    ///
    /// Bond-cap caveat, added after the 0.1.17 prefix-selection defect:
    /// "exactly representable" is a statement about the rank, not about the
    /// number of rows.  C has 2·r_q rows, and the pair actually kept is
    /// (blockdiag(Q_0,Q_1)[:, :k], C[:k, :]) with k = min(last row above
    /// 1e-12·‖C‖_F, bond_dim) — a prefix.  Whenever 2·r_q > bond_dim that
    /// prefix is all of C_top and at most the first bond_dim − r_q rows of
    /// C_bot, so a gate that concentrates the state in the lower block loses
    /// the live rows while rank(M) still fits the cap.  Measured at
    /// M = 32×16, bond_dim = 16: rows 0–15 empty, live rows in 16–31, and
    /// ε = 0.9871 was reported as ordinary truncation.  The norm-ordered
    /// selection below therefore ranks rows by measured energy, and returns
    /// None (handing the merge to the SVD path) when ε still exceeds 1e-10.
    /// Do not reintroduce an unconditional assumption that the prefix is
    /// sufficient.
    #[allow(clippy::too_many_arguments)]
    fn blockdiag_exact_split(
        t1_a: &DMatrix<Complex64>,
        t1_b: &DMatrix<Complex64>,
        t2_a: &DMatrix<Complex64>,
        t2_b: &DMatrix<Complex64>,
        d_l1: usize,
        d_m: usize,
        d_r2: usize,
        gate: &DMatrix<Complex64>,
        bond_dim: usize,
        t_start: std::time::Instant,
    ) -> Option<(DMatrix<Complex64>, DMatrix<Complex64>, f64)> {
        let zero = Complex64::new(0.0, 0.0);
        let r_q = std::cmp::min(d_l1, d_m);
        if r_q == 0 || d_r2 == 0 {
            return None;
        }

        let __b_t0 = std::time::Instant::now();

        // Stage 1: thin QR of each left block: t1_p = Q_p · R_p.
        let t1a_f = na_to_faer(t1_a);
        let t1b_f = na_to_faer(t1_b);
        let __b_ta = std::time::Instant::now();
        let qr_a = FaerQr::new(t1a_f.as_ref());
        let qr_b = FaerQr::new(t1b_f.as_ref());
        let __b_tb = std::time::Instant::now();
        let q_a = qr_a.compute_thin_q();
        let q_b = qr_b.compute_thin_q();
        let __b_tc = std::time::Instant::now();
        let r_a = qr_a.compute_thin_r();
        let r_b = qr_b.compute_thin_r();
        let __b_t1 = std::time::Instant::now();

        // Stage 2: B built directly from the t2 blocks with the gate's
        // 2×2 left-preserving blocks — per output block o1 (d_m × 2·d_r2
        // each) so stage 3 can skip the structural zeros of the padded
        // blockdiag form.
        let build_b = |o1: usize| -> FaerMat<Complex64> {
            FaerMat::from_fn(d_m, 2 * d_r2, |mi, j| {
                let o2 = j / d_r2;
                let ri = j % d_r2;
                let g0 = gate[(2 * o1 + o2, 2 * o1)];
                let g1 = gate[(2 * o1 + o2, 2 * o1 + 1)];
                let mut v = if g0 != zero {
                    g0 * t2_a[(mi, ri)]
                } else {
                    zero
                };
                if g1 != zero {
                    v += g1 * t2_b[(mi, ri)];
                }
                v
            })
        };
        let b_top = build_b(0);
        let b_bot = build_b(1);
        let __b_t2 = std::time::Instant::now();

        // Stage 3: C = blockdiag(R_0, R_1) · B (2·r_q × 2·d_r2), split as
        // C_top = R_0·B_top and C_bot = R_1·B_bot — the padded blockdiag
        // product would multiply a zero off-diagonal block for nothing, so
        // the two half-size matmuls do half the arithmetic.
        let par = if d_m >= 64 && rayon::current_num_threads() > 1 {
            Parallelism::Rayon(0)
        } else {
            Parallelism::None
        };
        // compute_thin_r already returns exactly the top r_q × d_m triangle
        // (rows below the diagonal zeroed) the two matmuls need.
        let r_a_thin = r_a;
        let r_b_thin = r_b;
        let mut c_top = FaerMat::<Complex64>::zeros(r_q, 2 * d_r2);
        let mut c_bot = FaerMat::<Complex64>::zeros(r_q, 2 * d_r2);
        faer_matmul(
            c_top.as_mut(),
            r_a_thin.as_ref(),
            b_top.as_ref(),
            None,
            Complex64::new(1.0, 0.0),
            par,
        );
        faer_matmul(
            c_bot.as_mut(),
            r_b_thin.as_ref(),
            b_bot.as_ref(),
            None,
            Complex64::new(1.0, 0.0),
            par,
        );
        let __b_t3 = std::time::Instant::now();

        // Stage 4: phantom-row rank reveal + measured ε (same discipline
        // as the plain QR split: a row whose whole norm is below
        // 1e-12·‖C‖_F carries no signal).
        let mut row_sq = vec![0.0_f64; 2 * r_q];
        let mut frob_sq_full = 0.0_f64;
        for i in 0..r_q {
            let mut acc = 0.0_f64;
            for j in 0..2 * d_r2 {
                acc += c_top.read(i, j).norm_sqr();
            }
            row_sq[i] = acc;
            frob_sq_full += acc;
        }
        for i in 0..r_q {
            let mut acc = 0.0_f64;
            for j in 0..2 * d_r2 {
                acc += c_bot.read(i, j).norm_sqr();
            }
            row_sq[r_q + i] = acc;
            frob_sq_full += acc;
        }
        let __b_t4 = std::time::Instant::now();
        let frob_sq_full = frob_sq_full.max(1e-300_f64);
        let row_eps_sq = (1e-12_f64 * frob_sq_full.sqrt()).powi(2);
        let mut k_final = 1usize;
        for i in 0..2 * r_q {
            if row_sq[i] > row_eps_sq {
                k_final = i + 1;
            }
        }
        let k_final = std::cmp::min(k_final, bond_dim);
        let kept_sq: f64 = row_sq[..k_final].iter().sum();
        let eps = ((frob_sq_full - kept_sq) / frob_sq_full).clamp(0.0, 1.0);

        // ── Additive repair: norm-ordered row selection ──
        // The prefix selection above is exact only while the live rows of C
        // form a prefix.  A left-bit-preserving gate can concentrate the
        // state in the lower block (o1 = 1), and rows [0, k) are then nearly
        // empty while the cap drops the rows that carry the mass — the
        // brick12@16 collapse (M = 32x16, eps = 9.871e-1 at k = 16).  The
        // caller admits this merge only when rank_bound <= bond_dim, i.e.
        // when it is exactly representable, so any k independent rows
        // reproduce M.  Rank the rows by measured norm and keep that set
        // when it retains strictly more mass than the prefix.
        let mut order: Vec<usize> = (0..2 * r_q).collect();
        order.sort_by(|&a, &b| {
            row_sq[b]
                .partial_cmp(&row_sq[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut k_alt = 0usize;
        for &i in &order {
            if k_alt >= bond_dim || row_sq[i] <= row_eps_sq {
                break;
            }
            k_alt += 1;
        }
        if k_alt == 0 {
            k_alt = 1;
        }
        let kept_alt: f64 = order[..k_alt].iter().map(|&i| row_sq[i]).sum();

        // Stage 5: assemble in the caller's packing — t1 rows (o1,l),
        // t2 rows (o2,j) with column order matching C's rows.
        let q_a_na = faer_to_na(&q_a);
        let q_b_na = faer_to_na(&q_b);
        let mut t1_new = DMatrix::<Complex64>::zeros(2 * d_l1, k_final);
        let ka = std::cmp::min(k_final, r_q);
        if ka > 0 {
            t1_new
                .view_mut((0, 0), (d_l1, ka))
                .copy_from(&q_a_na.view((0, 0), (d_l1, ka)));
        }
        if k_final > r_q {
            let kb = k_final - r_q;
            t1_new
                .view_mut((d_l1, r_q), (d_l1, kb))
                .copy_from(&q_b_na.view((0, 0), (d_l1, kb)));
        }
        let mut t2_new = DMatrix::<Complex64>::zeros(2 * k_final, d_r2);
        for j in 0..k_final {
            let (c_h, jj) = if j < r_q {
                (&c_top, j)
            } else {
                (&c_bot, j - r_q)
            };
            for o2 in 0..2usize {
                for rr in 0..d_r2 {
                    t2_new[(o2 * k_final + j, rr)] = c_h.read(jj, o2 * d_r2 + rr);
                }
            }
        }
        let __b_t5 = std::time::Instant::now();

        // ── Debug-only self-check (env SF_MPS_SVD_DEBUG=1) ──
        // Rebuilds the reference merge from the 4 gate-weighted block
        // products and reports the reconstruction error against
        // (blockdiag(Q_0,Q_1)[:, :k], C[:k, :]) plus the left-factor Gram
        // defect, in the same [svd-dbg] format the SVD path prints.
        // The heavy reference rebuild only runs with SF_MPS_SVD_DEBUG_FULL=1.
        if std::env::var_os("SF_MPS_SVD_DEBUG").is_some() {
            use std::sync::atomic::{AtomicUsize, Ordering};
            static SEQ_BD: AtomicUsize = AtomicUsize::new(0);
            let seq = 2_000_000 + SEQ_BD.fetch_add(1, Ordering::Relaxed);
            let full_dbg = std::env::var_os("SF_MPS_SVD_DEBUG_FULL").is_some();
            let __b_tg = t_start.elapsed();
            let mut num = 0.0_f64;
            let mut den = 0.0_f64;
            let mut gram_err = 0.0_f64;
            if full_dbg {
                let p00 = t1_a * t2_a;
                let p01 = t1_a * t2_b;
                let p10 = t1_b * t2_a;
                let p11 = t1_b * t2_b;
                for o1 in 0..2usize {
                    for o2 in 0..2usize {
                        for l in 0..d_l1 {
                            for r in 0..d_r2 {
                                let mut mv = zero;
                                for i1 in 0..2usize {
                                    for i2 in 0..2usize {
                                        let gi = gate[(2 * o1 + o2, 2 * i1 + i2)];
                                        if gi != zero {
                                            let p = match (i1, i2) {
                                                (0, 0) => &p00,
                                                (0, 1) => &p01,
                                                (1, 0) => &p10,
                                                _ => &p11,
                                            };
                                            mv += gi * p[(l, r)];
                                        }
                                    }
                                }
                                let mut rv = zero;
                                for j in 0..k_final {
                                    rv +=
                                        t1_new[(o1 * d_l1 + l, j)] * t2_new[(o2 * k_final + j, r)];
                                }
                                num += (rv - mv).norm_sqr();
                                den += mv.norm_sqr();
                            }
                        }
                    }
                }
                for a in 0..k_final {
                    for b in 0..k_final {
                        let mut ip = zero;
                        for i in 0..2 * d_l1 {
                            ip += t1_new[(i, a)].conj() * t1_new[(i, b)];
                        }
                        let want = if a == b {
                            Complex64::new(1.0, 0.0)
                        } else {
                            zero
                        };
                        gram_err += (ip - want).norm_sqr();
                    }
                }
            }
            let smax = row_sq.iter().cloned().fold(0.0_f64, f64::max).sqrt();
            let scut = if k_final < row_sq.len() {
                row_sq[k_final].sqrt()
            } else {
                0.0
            };
            eprintln!(
                "[svd-dbg] seq={seq} M={}x{} k={k_final} smax={smax:.3e} scut={scut:.3e} \
                 eps={eps:.3e} recon={:.3e} ugram={:.3e} guard=false \
                 t_pre={}us t_svd={}us t_bld={}us t_rep={}us t_qr={}us t_tot={}us \
                 t_gate={}us t_cv={}us t_qrf={}us t_qq={}us t_qr2={}us",
                2 * d_l1,
                2 * d_r2,
                (num / den.max(1e-300)).sqrt(),
                gram_err.sqrt(),
                (__b_t1 - __b_t0).as_micros(),
                (__b_t2 - __b_t1).as_micros(),
                (__b_t3 - __b_t2).as_micros(),
                (__b_t5 - __b_t4).as_micros(),
                (__b_t4 - __b_t3).as_micros(),
                (__b_t5 - __b_t0).as_micros(),
                __b_tg.as_micros(),
                (__b_ta - __b_t0).as_micros(),
                (__b_tb - __b_ta).as_micros(),
                (__b_tc - __b_tb).as_micros(),
                (__b_t1 - __b_tc).as_micros(),
            );
        }

        // ── Additive repair: apply the norm-ordered selection ──
        // Runs after the debug block so the self-check above always reports
        // the prefix pair; the returned pair is the better of the two.
        if kept_alt > kept_sq * (1.0 + 1e-12) {
            let mut t1_alt = DMatrix::<Complex64>::zeros(2 * d_l1, k_alt);
            let mut t2_alt = DMatrix::<Complex64>::zeros(2 * k_alt, d_r2);
            for (j, &src) in order[..k_alt].iter().enumerate() {
                let (c_h, jj, q_h) = if src < r_q {
                    (&c_top, src, &q_a_na)
                } else {
                    (&c_bot, src - r_q, &q_b_na)
                };
                let col_off = if src < r_q { 0 } else { d_l1 };
                for i in 0..d_l1 {
                    t1_alt[(col_off + i, j)] = q_h[(i, jj)];
                }
                for o2 in 0..2usize {
                    for rr in 0..d_r2 {
                        t2_alt[(o2 * k_alt + j, rr)] = c_h.read(jj, o2 * d_r2 + rr);
                    }
                }
            }
            t1_new = t1_alt;
            t2_new = t2_alt;
            if std::env::var_os("SF_MPS_SVD_DEBUG").is_some() {
                eprintln!(
                    "[bd-repair] M={}x{} bond={bond_dim} prefix_k={k_final} \
                     kept_prefix={kept_sq:.6e} alt_k={k_alt} kept_alt={kept_alt:.6e}",
                    2 * d_l1,
                    2 * d_r2,
                );
            }
        }
        // Report the measured deficit of the pair actually returned.
        let eps = if kept_alt > kept_sq {
            ((frob_sq_full - kept_alt) / frob_sq_full).clamp(0.0, 1.0)
        } else {
            eps
        };
        // This branch is entered only for exactly representable merges, so a
        // material deficit means the row choice is still deficient; hand the
        // merge to the SVD path rather than return a silently lossy pair.
        if eps > 1e-10 {
            return None;
        }

        Some((t1_new, t2_new, eps))
    }

    /// Randomized-sketch truncation (retired from the live policy: on
    /// brick12 it measured slower AND less accurate than the exact paths —
    /// every sketch merge paid the full thin-SVD-sized pre-pass plus its
    /// own O(r²p) work, and admitting ε up to 1e-3·total traded accuracy
    /// for no speed; kept as a documented experiment, no caller invokes
    /// it anymore).
    ///
    /// For merges whose r = min(nrows, ncols) exceeds the bond cap
    /// (r ≤ bond_dim merges take the exact QR split in the caller and
    /// never reach this function).
    ///
    /// A rank-p sketch (p = min(r, bond_dim + 16), one power iteration) of
    /// the right-singular space is built from R = M^H·(M·Ω); QR(R) supplies
    /// the orthonormal basis V̂, and W = M·V̂ holds the measured action of
    /// every sketch column.  Selection and factors are then built the same
    /// self-consistent way as the exact path — kept columns chosen by the
    /// *measured* ‖M·v_c‖ (cap + τ on measured values, never on reported
    /// σ) — and the pair Qw·(Rw·V̂_k^H) reproduces M·(V̂ V̂^H) exactly for
    /// the sketch basis, with Qw from QR(W) so t1 is exactly isometric.
    /// ε is the measured (total − kept)/total, honest even if the sketch
    /// subspace is suboptimal for a flat spectrum.
    ///
    /// Returns None (⇒ exact SVD fallback) when the selection is empty
    /// (fully phantom merge) or the merge actually loses more than
    /// 1e-3·total under the sketch — the regime where subspace optimality
    /// matters and the exact SVD must adjudicate.
    #[allow(dead_code)]
    fn sketch_truncate_merged(
        m: &FaerMat<Complex64>,
        nrows: usize,
        ncols: usize,
        r: usize,
        d_r2: usize,
        bond_dim: usize,
        total: f64,
        tau: f64,
        t_start: std::time::Instant,
    ) -> Option<(
        nalgebra::DMatrix<Complex64>,
        nalgebra::DMatrix<Complex64>,
        f64,
    )> {
        let p = std::cmp::min(r, bond_dim + 16);
        if p < 2 {
            return None;
        }
        let __t0 = std::time::Instant::now();
        let par = if r >= 64 {
            Parallelism::Rayon(0)
        } else {
            Parallelism::None
        };
        let one = Complex64::new(1.0, 0.0);
        // Per-call deterministic pseudo-random sketch matrix (LCG — the
        // sketch only needs generic entries; per-call seeding keeps merges
        // independent while staying reproducible for debugging).
        static SKETCH_SEQ: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let sketch_seq = SKETCH_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let mut seed = (sketch_seq as u64)
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add(0xD1B5_4A32_D192_ED03)
            | 1;
        let mut next = move || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 11) as f64) / ((1u64 << 53) as f64) - 0.5
        };
        let omega = FaerMat::<Complex64>::from_fn(ncols, p, |_, _| Complex64::new(next(), next()));
        // Y = M·Ω, then one power iteration on M^H·M: R = M^H·Y.
        let mut y1 = FaerMat::<Complex64>::zeros(nrows, p);
        faer_matmul(y1.as_mut(), m.as_ref(), omega.as_ref(), None, one, par);
        let mh = FaerMat::<Complex64>::from_fn(ncols, nrows, |i, j| m.read(j, i).conj());
        let mut rsk = FaerMat::<Complex64>::zeros(ncols, p);
        faer_matmul(rsk.as_mut(), mh.as_ref(), y1.as_ref(), None, one, par);
        // V̂ = orthonormal basis of the sketched right space.
        let qr_r = FaerQr::new(rsk.as_ref());
        let q_v = qr_r.compute_q();
        let vk = FaerMat::<Complex64>::from_fn(ncols, p, |i, j| q_v.read(i, j));
        let __t1 = std::time::Instant::now();
        // Measured action of every sketch column: W = M·V̂.
        let mut w = FaerMat::<Complex64>::zeros(nrows, p);
        faer_matmul(w.as_mut(), m.as_ref(), vk.as_ref(), None, one, par);
        let mut meas: Vec<(usize, f64)> = (0..p)
            .map(|c| {
                let mut nrm_sq = 0.0_f64;
                for i in 0..nrows {
                    nrm_sq += w.read(i, c).norm_sqr();
                }
                (c, nrm_sq)
            })
            .collect();
        meas.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let mut sel: Vec<usize> = Vec::with_capacity(std::cmp::min(bond_dim, p));
        for &(c, nrm_sq) in &meas {
            if sel.len() >= bond_dim {
                break;
            }
            if nrm_sq.sqrt() <= tau {
                break;
            }
            sel.push(c);
        }
        if sel.is_empty() {
            return None;
        }
        let k = sel.len();
        let mut kept = 0.0_f64;
        for &c in &sel {
            kept += meas
                .iter()
                .find(|&&(cc, _)| cc == c)
                .map(|&(_, v)| v)
                .unwrap_or(0.0);
        }
        // Sketch subspace misses material mass ⇒ let the exact SVD decide.
        if (total - kept) > 1e-3_f64 * total {
            return None;
        }
        let __t2 = std::time::Instant::now();
        // Pair-sum = Σ_c w_c·v̂_c^H.  QR(W_sel) = Qw·Rw ⇒
        // pair-sum = Qw·(Rw·V̂_k^H): t1 = Qw exactly isometric, t2 absorbs
        // the triangular factor through the selected sketch columns.
        let wk = FaerMat::<Complex64>::from_fn(nrows, k, |i, j| w.read(i, sel[j]));
        let qr_w = FaerQr::new(wk.as_ref());
        let q_w = qr_w.compute_q();
        let r_w = qr_w.compute_r();
        let mut t1 = nalgebra::DMatrix::<Complex64>::zeros(nrows, k);
        for j in 0..k {
            for i in 0..nrows {
                t1[(i, j)] = q_w.read(i, j);
            }
        }
        let mut t2 = nalgebra::DMatrix::<Complex64>::zeros(2 * k, d_r2);
        for q_out in 0..2usize {
            for rr in 0..d_r2 {
                let vcol = q_out * d_r2 + rr;
                for j in 0..k {
                    let mut acc = Complex64::new(0.0, 0.0);
                    for c in 0..k {
                        acc += r_w.read(j, c) * vk.read(vcol, sel[c]).conj();
                    }
                    t2[(q_out * k + j, rr)] = acc;
                }
            }
        }
        let __t3 = std::time::Instant::now();
        let eps = ((total - kept) / total).clamp(0.0, 1.0);
        if std::env::var_os("SF_MPS_SVD_DEBUG").is_some() {
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
            let smax = meas.first().map(|&(_, v)| v.sqrt()).unwrap_or(0.0);
            eprintln!(
                "[svd-dbg] seq={} M={nrows}x{ncols} k={k} smax={smax:.3e} scut={tau:.3e} \
                 eps={eps:.3e} recon={:.3e} ugram={:.3e} guard=false \
                 t_pre={}us t_svd={}us t_bld={}us t_rep=0us t_qr={}us t_tot={}us",
                1_000_000 + sketch_seq,
                (num / den.max(1e-300)).sqrt(),
                gram_err.sqrt(),
                (__t0 - t_start).as_micros(),
                (__t1 - __t0).as_micros(),
                (__t2 - __t1).as_micros(),
                (__t3 - __t2).as_micros(),
                t_start.elapsed().as_micros(),
            );
        }
        Some((t1, t2, eps))
    }

    /// Optimal rank-k truncation of the merged two-site tensor via thin SVD.
    ///
    /// Under the default policy this runs for every merge, after the exact
    /// block-diagonal split has been tried for eligible left-bit-preserving
    /// gates (see [`mps_svd_path_needed`]); only the historical position
    /// cut (SF_MPS_TRUNCATION=qr|legacy) skips it.
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
        t_start: std::time::Instant,
    ) -> (
        nalgebra::DMatrix<Complex64>,
        nalgebra::DMatrix<Complex64>,
        f64,
    ) {
        let nrows = 2 * d_l1;
        let ncols = 2 * d_r2;
        let r = std::cmp::min(nrows, ncols);

        let __t_svd0 = std::time::Instant::now();
        let svd = ThinSvd::new(m.as_ref());
        let __t_svd1 = std::time::Instant::now();
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
                            t2[(q_out * k + j, rr)] = sj * v.read(q_out * d_r2 + rr, ci).conj();
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
        //
        // The same corruption also strikes bond-cap stops: brick12@1000,
        // bond 64 (SF_MPS_SVD_DEBUG seq=5897) truncated at k = 64 with
        // stopped_by_tau = false, so this guard never ran while faer had
        // reported σ_72 = 0 for a column whose measured ‖M·v_72‖² = 0.3202
        // (offline numpy ground truth: rank(M) = 64 exactly, true top-64
        // ε ≈ 2e-32 — the state lost NOTHING; the entire reported
        // ε = 5.004e-3 was mispaired selection).  The trigger therefore no
        // longer requires the τ stop: any stop reason fires whenever
        // k < r and the measured deficit is material.
        let (mut t1, mut t2, mut kept) = build(&sel);
        let __t_b1 = std::time::Instant::now();
        let mut guarded = false;
        let __t_g0 = std::time::Instant::now();
        if k < r && (total - kept) > 1e-12_f64 * total {
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
            meas.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
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
        let __t_g1 = std::time::Instant::now();
        // Restore the left-isometric gauge after a measured re-selection:
        // the repaired columns are M·v_c/‖M·v_c‖ for a V basis that is only
        // orthonormal, not the exact right-singular basis (that is the very
        // defect the re-selection works around), so the columns need not be
        // mutually orthogonal — ugram ≈ 1.37 was measured on seq=5897 even
        // after the re-selection.  QR-fold t1 = Q·R with R absorbed into t2
        // is an exact change of bond basis (t1·B = Q·(R·B)), so the state
        // and ε are untouched while Σ t1†t1 = I is restored.  On the clean
        // path (σ-honest pairing) R ≈ diag(phases) and this is a no-op gauge.
        if guarded {
            let kk = t1.ncols();
            let qr = t1.clone_owned().qr();
            let q = qr.q();
            let r = qr.r();
            let r_top = r.rows(0, kk).into_owned();
            let mut t2_new = DMatrix::<Complex64>::zeros(2 * kk, d_r2);
            for q_out in 0..2usize {
                let blk = t2.rows(q_out * kk, kk).into_owned();
                let nb = &r_top * blk;
                t2_new.rows_mut(q_out * kk, kk).copy_from(&nb);
            }
            t1 = q.columns(0, kk).into_owned();
            t2 = t2_new;
        }
        let __t_qr1 = std::time::Instant::now();
        let __t_tot = t_start.elapsed();
        let k = sel.len();
        let eps = ((total - kept) / total).clamp(0.0, 1.0);

        // ── Debug-only self-check (env SF_MPS_SVD_DEBUG=1) ──
        // Additive instrumentation: for every call, validates that the packed
        // truncation reproduces the merged tensor M under the same packing the
        // QR tail uses (t2 row = q_out·k + j), and that the retained
        // left-factor columns are orthonormal.  Reports σ_max, the σ at the
        // cut, k, ε and both errors on stderr.  The default path never
        // consults this.  The heavy reference rebuilds only run with
        // SF_MPS_SVD_DEBUG_FULL=1.
        if std::env::var_os("SF_MPS_SVD_DEBUG").is_some() {
            use std::sync::atomic::{AtomicUsize, Ordering};
            static SEQ: AtomicUsize = AtomicUsize::new(0);
            let seq = SEQ.fetch_add(1, Ordering::Relaxed);
            let full_dbg = std::env::var_os("SF_MPS_SVD_DEBUG_FULL").is_some();
            let mut num = 0.0_f64;
            let mut den = 0.0_f64;
            let mut gram_err = 0.0_f64;
            if full_dbg {
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
            }
            let s_cut = if k < idx.len() {
                s.read(idx[k]).norm()
            } else {
                0.0
            };
            eprintln!(
                "[svd-dbg] seq={seq} M={nrows}x{ncols} k={k} smax={:.3e} scut={:.3e} \
                 eps={:.3e} recon={:.3e} ugram={:.3e} guard={guarded} \
                 t_pre={}us t_svd={}us t_bld={}us t_rep={}us t_qr={}us t_tot={}us",
                s.read(idx[0]).norm(),
                s_cut,
                eps,
                (num / den.max(1e-300)).sqrt(),
                gram_err.sqrt(),
                (__t_svd0 - t_start).as_micros(),
                (__t_svd1 - __t_svd0).as_micros(),
                (__t_b1 - __t_svd1).as_micros(),
                (__t_g1 - __t_g0).as_micros(),
                (__t_qr1 - __t_g1).as_micros(),
                __t_tot.as_micros(),
            );

            if full_dbg {
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
        }

        (t1, t2, eps)
    }

    pub fn to_statevector(&self) -> Vec<Complex64> {
        // Sparse-prefix fast path first (area-law states like GHZ keep O(1)
        // live rows: exact zeros stay exact, so the zero test below never
        // misfires; dense states bail out early to the GEMM sweep).
        if let Some(v) = self.to_statevector_sparse() {
            return v;
        }
        self.to_statevector_gemm()
    }

    /// Sparse-prefix densify: track only nonzero prefix rows.
    ///
    /// Returns `None` (caller falls back to the GEMM sweep) as soon as the
    /// live set exceeds 1024 rows, bounding wasted work far below dense cost.
    /// Zero test is exact (`== 0.0`): truncated/canonical MPS tensors keep
    /// structural zeros exact, so no false sparsity is possible.
    fn to_statevector_sparse(&self) -> Option<Vec<Complex64>> {
        const LIMIT: usize = 1024;
        let n = self.n_qubits;
        if n == 0 {
            return Some(vec![Complex64::new(1.0, 0.0)]);
        }
        // (prefix index, bond vector)
        let mut active: Vec<(usize, Vec<Complex64>)> = vec![(0, vec![Complex64::new(1.0, 0.0)])];
        for idx in 0..n {
            let t = &self.tensors[idx];
            let (rows, dr) = t.shape();
            let dl = rows / 2;
            let mut next: Vec<(usize, Vec<Complex64>)> = Vec::new();
            for (i, vec) in &active {
                if vec.len() != dl {
                    return None;
                }
                for s in 0..2usize {
                    let mut child = vec![Complex64::new(0.0, 0.0); dr];
                    for r in 0..dr {
                        let mut sum = Complex64::new(0.0, 0.0);
                        for l in 0..dl {
                            sum += vec[l] * t[(l + s * dl, r)];
                        }
                        child[r] = sum;
                    }
                    // Child index: qubit idx lives at BIT idx (little-endian),
                    // so the longer prefix keeps its bits: i + (s << idx).
                    // (NOT i*2+s — that would append at bit 0 and only works
                    // for bit-reversal-symmetric states like GHZ.)
                    if child.iter().any(|&z| z != Complex64::new(0.0, 0.0)) {
                        next.push((i + (s << idx), child));
                        if next.len() > LIMIT {
                            return None;
                        }
                    }
                }
            }
            active = next;
        }
        let dim = 1usize << n;
        let mut state = vec![Complex64::new(0.0, 0.0); dim];
        for (i, v) in &active {
            // Right boundary bond must be 1 (pure state); otherwise the
            // scalar read is undefined — fall back to the GEMM sweep.
            if v.len() != 1 {
                return None;
            }
            state[*i] = v[0];
        }
        Some(state)
    }

    /// GEMM-sweep densify for general bonds.
    fn to_statevector_gemm(&self) -> Vec<Complex64> {
        // State after site idx is a (2^idx, D) matrix M with M[i, l] the
        // amplitude of prefix-bitstring i ending in left-bond l. Site step:
        //   M'[i + s*2^idx, r] = sum_l M[i, l] * T[(l + s*D_L), r]
        // i.e. two GEMMs per site (s = 0, 1) assembled with block copies.
        // Total O(2^n * D^2) with BLAS efficiency and O(n) allocations —
        // the previous per-amplitude scalar loop spent most of its time
        // allocating (2 allocs per amplitude per site). Qubit convention
        // unchanged: qubit q lives at bit q of the output index
        // (little-endian). NOTE: faer-parallel (direct-write, Rayon pool)
        // and zero-alloc scalar variants were tried and measured slower
        // here (dispatch overhead dominates these tall-skinny shapes);
        // sequential nalgebra GEMM wins. Closing the rest of the gap to
        // threaded-BLAS frameworks needs OpenBLAS linkage (future work).
        let n = self.n_qubits;
        let mut m = nalgebra::DMatrix::<Complex64>::from_element(1, 1, Complex64::new(1.0, 0.0));
        for idx in 0..n {
            let t = &self.tensors[idx];
            let (rows, d_r) = t.shape();
            let d_l = rows / 2;
            let p = m.nrows();
            debug_assert_eq!(m.ncols(), d_l);
            let mut m_next = nalgebra::DMatrix::<Complex64>::zeros(p * 2, d_r);
            for s in 0..2usize {
                // T_s = rows [s*d_l, (s+1)*d_l) of T, shape (d_l, d_r).
                // Block copy (one call per site per s) keeps assembly at
                // O(n) view operations instead of O(2^n) per-row copies.
                let t_s =
                    nalgebra::DMatrix::<Complex64>::from_fn(d_l, d_r, |l, r| t[(l + s * d_l, r)]);
                let prod = &m * &t_s;
                m_next
                    .index_mut((s * p..(s + 1) * p, 0..d_r))
                    .copy_from(&prod);
            }
            m = m_next;
        }
        debug_assert_eq!(m.nrows(), 1usize << n);
        // Right boundary bond is 1; take column 0.
        m.column(0).iter().copied().collect()
    }

    /// Direct bit-by-bit sampling from MPS: O(N * D^2 * shots)
    pub fn sample(&self, shots: usize, _seed: u64) -> std::collections::HashMap<String, usize> {
        let mut counts = std::collections::HashMap::new();
        if self.n_qubits == 0 {
            return counts;
        }

        for shot_idx in 0..shots {
            // Unique seed for this shot to ensure diversity
            let mut rng = StdRng::seed_from_u64(_seed.wrapping_add(shot_idx as u64));
            let key = self.sample_one_with_rng(&mut rng);
            *counts.entry(key).or_insert(0) += 1;
        }
        counts
    }

    /// Single-shot sample with a caller-supplied RNG: linear bit-by-bit
    /// conditional sampling, emitting the raw q0-last bitstring over
    /// physical tensor order.  Shared by `sample` (one seeded RNG per shot,
    /// bit-for-bit the historical draws) and by the dynamic-MCM trajectory
    /// loop, which threads one continuous RNG through the whole replay.
    pub fn sample_one_with_rng(&self, rng: &mut StdRng) -> String {
        use rand::Rng;
        let n = self.n_qubits;
        if n == 0 {
            return String::new();
        }

        let mut left_vec = DVector::from_vec(vec![Complex64::new(1.0, 0.0)]);
        let mut next_vec = DVector::zeros(1);
        let mut key_chars = vec!['0'; n];

        for i in 0..n {
            let t = &self.tensors[i];
            let d_l = t.shape().0 / 2;
            let d_r = t.shape().1;

            // Unnormalized P(0) and P(1) in one pass over the (r, l) pairs,
            // same sum order as the historical two-loop form.
            let mut prob0 = 0.0_f64;
            let mut prob1 = 0.0_f64;
            for r in 0..d_r {
                let mut sum0 = Complex64::new(0.0, 0.0);
                let mut sum1 = Complex64::new(0.0, 0.0);
                for l in 0..d_l {
                    sum0 += left_vec[l] * t[(l, r)];
                    sum1 += left_vec[l] * t[(l + d_l, r)];
                }
                prob0 += sum0.norm_sqr();
                prob1 += sum1.norm_sqr();
            }

            let total = prob0 + prob1;
            let p0_normalized = if total > 1e-30 { prob0 / total } else { 0.5 };

            let random_val: f64 = rng.gen();
            let b = if random_val < p0_normalized { '0' } else { '1' };
            key_chars[n - 1 - i] = b;

            // Left-vector projection onto the drawn outcome.
            let p_idx = if b == '0' { 0 } else { 1 };
            next_vec.resize_vertically_mut(d_r, Complex64::new(0.0, 0.0));
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
            std::mem::swap(&mut left_vec, &mut next_vec);
        }
        // Emit bitstrings q0-last (qubit 0 = rightmost char), matching the
        // statevector backend's sample() convention.
        key_chars.iter().collect()
    }

    /// Projectively measure tensor site `site` in the Z basis and collapse.
    ///
    /// The suffix above `site` is first right-canonicalized (only the dirty
    /// window `site+1..rc_start` is swept), so the right tails entering
    /// `site` are orthonormal and the outcome weights are exactly
    /// p_p = ‖Σ_l v_l · T_s[p·D_L + l, :]‖², with v the fully contracted left
    /// boundary (no conditioning on earlier sites).  The sampled outcome
    /// zeroes the other physical block and rescales the survivor by
    /// √((p0+p1)/p_p), conserving the state norm — the MPS analogue of the
    /// statevector `collapse_measure`.  Returns the measured bit.
    pub fn measure_site(&mut self, site: usize, rng: &mut StdRng) -> u8 {
        use rand::Rng;
        // Delta-canonicalization: only sites above `site` need to be
        // right-isometric for the outcome weights below.  `rc_start` bounds
        // the dirty suffix, so sweep just the window `(site+1, rc_start)`
        // instead of the whole chain; after the collapse the measured site
        // itself is non-isometric, so the new dirty bound is site + 1.
        if self.rc_start > site + 1 {
            let rc = self.rc_start;
            self.canonicalize_right_range(site + 1, rc);
        }
        self.rc_start = site + 1;

        // Left environment: contract sites 0..site-1, summed over every
        // physical outcome (marginalized — earlier sites are not observed).
        // The contraction is cached across measurements, so a left-to-right
        // measurement sequence costs one site-contraction per site in total.
        self.meas_env_extend_to(site);

        // Outcome weights and the draw (same convention as collapse_measure:
        // draw < p0_normalized ⇒ bit 0).
        let v = self.left_env.as_slice();
        let t = &mut self.tensors[site];
        let d_l = t.shape().0 / 2;
        let d_r = t.shape().1;
        let mut p = [0.0_f64, 0.0_f64];
        for (bit, pk) in p.iter_mut().enumerate() {
            for r in 0..d_r {
                let mut acc = Complex64::new(0.0, 0.0);
                for l in 0..d_l {
                    acc += v[l] * t[(bit * d_l + l, r)];
                }
                *pk += acc.norm_sqr();
            }
        }
        let total = p[0] + p[1];
        let p0_normalized = if total > 1e-30 { p[0] / total } else { 0.5 };
        let random_val: f64 = rng.gen();
        let bit = if random_val < p0_normalized {
            0usize
        } else {
            1usize
        };

        // Collapse: zero the discarded block, rescale the survivor so the
        // projected branch keeps the pre-measurement norm.
        let pb = p[bit];
        let scale = if pb > 1e-300 {
            (total / pb).sqrt()
        } else {
            1.0
        };
        let keep = bit * d_l;
        let drop = (1 - bit) * d_l;
        for r in 0..d_r {
            for l in 0..d_l {
                t[(keep + l, r)] *= scale;
                t[(drop + l, r)] = Complex64::new(0.0, 0.0);
            }
        }
        self.env_upto = site;
        bit as u8
    }

    /// Trace out one site's physical index while folding it into a left
    /// environment vector: `env'[r] = Σ_l env[l]·(T[l,r] + T[l+D_L,r])`.
    fn contract_site_into(env: &mut Vec<Complex64>, t: &DMatrix<Complex64>) {
        let d_l = t.shape().0 / 2;
        let d_r = t.shape().1;
        let mut nv = vec![Complex64::new(0.0, 0.0); d_r];
        for r in 0..d_r {
            let mut acc = Complex64::new(0.0, 0.0);
            for l in 0..d_l {
                acc += env[l] * (t[(l, r)] + t[(l + d_l, r)]);
            }
            nv[r] = acc;
        }
        *env = nv;
    }

    /// Extend the cached measurement left environment so it covers sites
    /// `0..site`.  A backwards request restarts from the boundary; otherwise
    /// only the not-yet-folded sites are contracted.
    fn meas_env_extend_to(&mut self, site: usize) {
        if self.env_upto > site {
            self.invalidate_meas_env();
        }
        for j in self.env_upto..site {
            if self.left_env.len() != self.tensors[j].shape().0 / 2 {
                // Defensive: rebuild the prefix from the left boundary.
                self.left_env.clear();
                self.left_env.push(Complex64::new(1.0, 0.0));
                for jj in 0..j {
                    Self::contract_site_into(&mut self.left_env, &self.tensors[jj]);
                }
                self.env_upto = j;
            }
            Self::contract_site_into(&mut self.left_env, &self.tensors[j]);
            self.env_upto = j + 1;
        }
    }

    /// Drop the cached measurement left environment: a rewrite at any site
    /// below the frontier makes the cached contraction stale.
    fn invalidate_meas_env(&mut self) {
        self.env_upto = 0;
        self.left_env.clear();
        self.left_env.push(Complex64::new(1.0, 0.0));
    }

    /// Mid-circuit reset on tensor site `site`: measure-and-discard, then
    /// prepare |0> (X-flip when the sampled bit was 1).  Mirrors the
    /// statevector `reset_qubit` semantics.  Returns the discarded bit.
    pub fn reset_site(&mut self, site: usize, rng: &mut StdRng) -> u8 {
        let bit = self.measure_site(site, rng);
        if bit == 1 {
            let x = OpType::X.to_matrix();
            self.apply_1q_gate(site, &x);
        }
        bit
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
        let __dbg = std::env::var_os("SF_MPS_SVD_DEBUG").is_some();
        let __t0 = std::time::Instant::now();
        let mut state = MPSState::new(self.n_qubits, bond_dim);
        let __t1 = std::time::Instant::now();
        self._evolve_into(&mut state);
        if __dbg {
            eprintln!(
                "[loop-dbg] evolve_mps n={} new={}us sweep={}us",
                self.n_qubits,
                (__t1 - __t0).as_micros(),
                __t1.elapsed().as_micros()
            );
        }
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
        let __dbg = std::env::var_os("SF_MPS_SVD_DEBUG").is_some();
        let __tl0 = std::time::Instant::now();
        let layers = self.parallel_layers();
        let __tl1 = std::time::Instant::now();
        let bond_dim = state.bond_dim;
        let mut __lstat: Vec<(usize, u128)> = Vec::new();

        for layer in layers {
            let __lt = std::time::Instant::now();
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
            // The parallel path applies adjacent 2q gates through the raw
            // static helper (bypassing `apply_2q_gate`), so mirror its
            // rc_start bookkeeping: both rewritten sites lose their
            // right-isometry claim.  Routed (non-adjacent) gates below go
            // through `apply_2q_gate_routed`, which updates rc_start itself.
            for &node_id in &layer {
                let op = &self.graph()[node_id];
                if op.op_type.is_boundary()
                    || op.op_type == OpType::Barrier
                    || op.op_type.is_measurement()
                {
                    continue;
                }
                if op.qubits.len() == 2 {
                    let dist = (op.qubits[0] as isize - op.qubits[1] as isize).abs();
                    if dist == 1 {
                        let lo = op.qubits[0].min(op.qubits[1]);
                        if lo < state.env_upto {
                            state.invalidate_meas_env();
                        }
                        let hi = op.qubits[0].max(op.qubits[1]);
                        if hi >= state.rc_start {
                            state.rc_start = hi + 1;
                        }
                    }
                }
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
            __lstat.push((layer.len(), __lt.elapsed().as_micros()));
        }

        if __dbg {
            let __tot: u128 = __lstat.iter().map(|s| s.1).sum();
            let mut __idx: Vec<usize> = (0..__lstat.len()).collect();
            __idx.sort_by_key(|&i| std::cmp::Reverse(__lstat[i].1));
            let __top = __idx
                .iter()
                .take(4)
                .map(|&i| format!("L{}[g={} {}us]", i, __lstat[i].0, __lstat[i].1))
                .collect::<Vec<_>>()
                .join(" ");
            eprintln!(
                "[loop-dbg] _evolve_into layers={} plan={}us loop={}us par_sum={}us top: {}",
                __lstat.len(),
                (__tl1 - __tl0).as_micros(),
                __tl0.elapsed().as_micros(),
                __tot,
                __top
            );
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
    fn test_blockdiag_repair_keeps_live_rows_below_the_prefix() {
        // The block-diagonal split keeps a prefix of C's rows; when the
        // populated block is the lower one (o1 = 1) and the cap equals the
        // rank bound, that prefix is empty.  The repair must re-select the
        // live rows instead of reporting a near-total discard.
        let d_l1 = 4usize;
        let d_m = 4usize;
        let d_r2 = 2usize;
        let bond_dim = 4usize;
        let one = Complex64::new(1.0, 0.0);
        let mut t1 = nalgebra::DMatrix::zeros(2 * d_l1, d_m);
        for i in 0..d_m {
            t1[(d_l1 + i, i)] = one;
        }
        let t2 = nalgebra::DMatrix::from_element(2 * d_m, d_r2, one);
        let t1_a = t1.rows(0, d_l1).into_owned();
        let t1_b = t1.rows(d_l1, d_l1).into_owned();
        let t2_a = t2.rows(0, d_m).into_owned();
        let t2_b = t2.rows(d_m, d_m).into_owned();
        let gate = OpType::CNOT.to_matrix();
        let out = MPSState::blockdiag_exact_split(
            &t1_a,
            &t1_b,
            &t2_a,
            &t2_b,
            d_l1,
            d_m,
            d_r2,
            &gate,
            bond_dim,
            std::time::Instant::now(),
        );
        let (q, _r, eps) = out.expect("exactly representable merge must not fall back");
        assert_eq!(q.ncols(), bond_dim);
        assert!(eps < 1e-12, "row selection lost mass: eps={eps}");
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
    fn test_sparse_densify_little_endian_layout() {
        // |001> (X on qubit 0): the amplitude must land at index 1, not 4.
        // Regression test for the MSB-append (i*2+s) indexing bug in the
        // sparse-prefix path, which bit-reversal-symmetric states like GHZ
        // could not catch (both orders coincide there).
        let mut mps = MPSState::new(3, 64);
        let x = OpType::X.to_matrix();
        mps.apply_1q_gate(0, &x);
        let sv = mps.to_statevector();
        assert_eq!(sv.len(), 8);
        for (i, amp) in sv.iter().enumerate() {
            let want = if i == 1 { 1.0 } else { 0.0 };
            assert!(
                (amp.re - want).abs() < 1e-12 && amp.im.abs() < 1e-12,
                "i={i} amp={amp:?}"
            );
        }
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
        let (t1, t2, eps_svd) =
            MPSState::svd_truncate_merged(&m, d, d, 4, std::time::Instant::now());
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
