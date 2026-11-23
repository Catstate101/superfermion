//! Is the per-gate cost at a binding cap the merge, or the policy lookup?
//!
//! §42.6 attributed the whole 2.57x gauge cost to `apply_layer_mixed_canonical`
//! and proposed a per-layer canonicalization rewrite.  Before paying for that
//! rewrite, this separates the two things a gate actually does:
//!
//!   1. `mps_gauge_enabled()` calls `std::env::var` on every adjacent 2q gate.
//!   2. `apply_2q_gate_static` factorizes the merged pair.  At bond 1 that is
//!      a 2x2 complex matrix through faer's `ThinSvd`, which allocates a
//!      workspace and a `Vec` of indices per call.
//!
//! Timing a merge at d = 1, 2, 4, ... tells them apart: overhead that does
//! not move with the matrix size is not arithmetic.
//!
//! It also prototypes the replacement: an exact one-sided complex Jacobi SVD
//! for tiny merges, and checks it against faer on the same matrices (sigma,
//! reconstruction, kept rank and epsilon) before any of it goes into mps.rs.
//!
//! Run: `cargo run --release -p sf-ir --example mps_merge_probe`

use faer::linalg::solvers::ThinSvd;
use faer::Mat as FaerMat;
use nalgebra::DMatrix;
use num_complex::Complex64;
use sf_ir::mps::MPSState;
use std::hint::black_box;
use std::time::Instant;

fn rnd(i: usize, j: usize, s: f64) -> Complex64 {
    let a = ((i * 37 + j * 91) % 101) as f64 / 101.0 - 0.5;
    let b = ((i * 13 + j * 57) % 89) as f64 / 89.0 - 0.5;
    Complex64::new(a * s, b * s)
}

fn fresh(d: usize) -> (DMatrix<Complex64>, DMatrix<Complex64>) {
    // t1: (2d x d), t2: (2d x d) — exactly the shape a d-bond merge sees.
    let t1 = DMatrix::from_fn(2 * d, d, |i, j| rnd(i, j, 0.31));
    let t2 = DMatrix::from_fn(2 * d, d, |i, j| rnd(i, j, 0.29));
    (t1, t2)
}

/// `(t1, t2, kept mass, sigma, tail)` — the packed SVD result both reference
/// implementations return; `tail` is each one's own convergence datum (`bool`
/// for the `DMatrix` version, the sweep count for the flat one).
type MergeOut<T> = Option<(DMatrix<Complex64>, DMatrix<Complex64>, f64, Vec<f64>, T)>;

/// Exact one-sided complex Jacobi SVD of a small merged matrix, in the same
/// (t1, t2, eps) packing the faer path returns.
///
/// One-sided Jacobi orthogonalizes the columns of `b = m·V` by unitary 2x2
/// rotations in the (p,q) plane until every pair is orthogonal; then
/// sigma_j = ||b_j||, u_j = b_j/sigma_j, and the construction is *by
/// definition* t1·t2 = M·(V_k V_k^H) — the same functional the faer path
/// builds from its reported (sigma, v) pair, without depending on that pair
/// being consistent (the faer defect the degenerate-spectrum guard works
/// around).  Returns selection, kept mass and eps under the caller's policy.
///
/// Kept as the readable reference for the algorithm: this `DMatrix` version
/// measured *slower* than faer from d=2 upward (21.5 us vs 14.3 us at 4x4)
/// because an index recomputes `i + j*nrows` and bounds-checks per element,
/// so the numbers in this file come from [`small_merge_flat`] instead.
#[allow(dead_code)]
fn small_merge_svd(
    m: &DMatrix<Complex64>,
    d_l1: usize,
    d_r2: usize,
    bond_dim: usize,
    tol: f64,
    max_sweeps: usize,
) -> MergeOut<bool> {
    let nrows = 2 * d_l1;
    let ncols = 2 * d_r2;
    let r = nrows.min(ncols);
    let mut b = m.clone();
    let mut v = DMatrix::<Complex64>::identity(ncols, ncols);
    let mut converged = false;
    for _sweep in 0..max_sweeps {
        let mut rotated = 0usize;
        for p in 0..ncols {
            for q in (p + 1)..ncols {
                let mut alpha = 0.0_f64;
                let mut beta = 0.0_f64;
                let mut gamma = Complex64::new(0.0, 0.0);
                for i in 0..nrows {
                    alpha += b[(i, p)].norm_sqr();
                    beta += b[(i, q)].norm_sqr();
                    gamma += b[(i, p)].conj() * b[(i, q)];
                }
                if alpha == 0.0 || beta == 0.0 {
                    continue;
                }
                if gamma.norm() <= tol * (alpha * beta).sqrt() {
                    continue;
                }
                rotated += 1;
                // G = [[alpha, gamma], [conj(gamma), beta]] is Hermitian;
                // D = diag(1, e^{-i phi}) makes it real symmetric, and the
                // real Jacobi angle zeroes the off-diagonal of R^H D^H G D R.
                let phi = gamma.arg();
                let g = gamma.norm();
                let theta = if alpha == beta {
                    std::f64::consts::FRAC_PI_4
                } else {
                    let tau = (alpha - beta) / (2.0 * g);
                    let t = if tau >= 0.0 {
                        1.0 / (tau + (1.0 + tau * tau).sqrt())
                    } else {
                        -1.0 / (-tau + (1.0 + tau * tau).sqrt())
                    };
                    t.atan()
                };
                let (c, s) = (theta.cos(), theta.sin());
                let phase = Complex64::from_polar(1.0, -phi);
                let sp = s * phase;
                let cp = c * phase;
                // U2 = D·R = [[c, -s], [s·e^{-i phi}, c·e^{-i phi}]]
                for i in 0..nrows {
                    let bp = b[(i, p)];
                    let bq = b[(i, q)];
                    b[(i, p)] = bp * c + bq * sp;
                    b[(i, q)] = bp * (-s) + bq * cp;
                }
                for i in 0..ncols {
                    let vp = v[(i, p)];
                    let vq = v[(i, q)];
                    v[(i, p)] = vp * c + vq * sp;
                    v[(i, q)] = vp * (-s) + vq * cp;
                }
            }
        }
        if rotated == 0 {
            converged = true;
            break;
        }
    }
    if !converged {
        return None;
    }

    let sigma: Vec<f64> = (0..ncols)
        .map(|j| (0..nrows).map(|i| b[(i, j)].norm_sqr()).sum::<f64>().sqrt())
        .collect();
    let total: f64 = sigma.iter().map(|s| s * s).sum::<f64>().max(1e-300);
    let tau = 1e-12 * total.sqrt();
    let mut idx: Vec<usize> = (0..r).collect();
    idx.sort_by(|&a, &c| {
        sigma[c]
            .partial_cmp(&sigma[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut k = 0usize;
    for &i in &idx {
        if k >= bond_dim || sigma[i] <= tau {
            break;
        }
        k += 1;
    }
    let k = k.max(1);
    let sel = idx[..k].to_vec();

    let mut t1 = DMatrix::<Complex64>::zeros(nrows, k);
    let mut t2 = DMatrix::<Complex64>::zeros(2 * k, d_r2);
    let mut kept = 0.0_f64;
    for (j, &ci) in sel.iter().enumerate() {
        kept += sigma[ci] * sigma[ci];
        if sigma[ci] > 0.0 {
            let inv = 1.0 / sigma[ci];
            for i in 0..nrows {
                t1[(i, j)] = b[(i, ci)] * inv;
            }
            let sj = Complex64::new(sigma[ci], 0.0);
            for q_out in 0..2usize {
                for rr in 0..d_r2 {
                    t2[(q_out * k + j, rr)] = sj * v[(q_out * d_r2 + rr, ci)].conj();
                }
            }
        }
    }
    let eps = ((total - kept) / total).clamp(0.0, 1.0);
    Some((t1, t2, eps, sigma, converged))
}

/// Column-major exact SVD of a tiny merged matrix on flat buffers.
///
/// One-sided complex Jacobi orthogonalizes the columns of `b = m·V` by unitary
/// 2x2 rotations in the (p,q) plane until every pair is orthogonal; then
/// sigma_j = ||b_j||, u_j = b_j/sigma_j, and the construction is *by definition*
/// t1·t2 = M·(V_k V_k^H) — the same functional the faer path builds from its
/// reported (sigma, v) pair, without depending on that pair being consistent
/// (the faer defect the degenerate-spectrum guard works around).
///
/// `DMatrix` bounds-checks every element and recomputes `i + j*nrows` on each
/// access, which at these shapes costs more than the arithmetic, so the
/// working set lives in two `Vec`s reached unchecked.
fn small_merge_flat(
    m: &DMatrix<Complex64>,
    d_l1: usize,
    d_r2: usize,
    bond_dim: usize,
    tol: f64,
    max_sweeps: usize,
) -> MergeOut<usize> {
    let nrows = 2 * d_l1;
    let ncols = 2 * d_r2;
    let r = nrows.min(ncols);
    let mut b = vec![Complex64::new(0.0, 0.0); nrows * ncols];
    for j in 0..ncols {
        for i in 0..nrows {
            b[i + j * nrows] = m[(i, j)];
        }
    }
    let mut v = vec![Complex64::new(0.0, 0.0); ncols * ncols];
    for j in 0..ncols {
        v[j + j * ncols] = Complex64::new(1.0, 0.0);
    }
    let one = Complex64::new(1.0, 0.0);
    let mut sweeps = 0usize;
    let mut converged = false;
    while sweeps < max_sweeps {
        sweeps += 1;
        let mut rotated = 0usize;
        for p in 0..ncols {
            for q in (p + 1)..ncols {
                let mut alpha = 0.0_f64;
                let mut beta = 0.0_f64;
                let mut gamma = Complex64::new(0.0, 0.0);
                unsafe {
                    let bp = b.as_ptr().add(p * nrows);
                    let bq = b.as_ptr().add(q * nrows);
                    for i in 0..nrows {
                        let x = *bp.add(i);
                        let y = *bq.add(i);
                        alpha += (x * x.conj()).re;
                        beta += (y * y.conj()).re;
                        gamma += x.conj() * y;
                    }
                }
                if alpha == 0.0 || beta == 0.0 {
                    continue;
                }
                if gamma.norm() <= tol * (alpha * beta).sqrt() {
                    continue;
                }
                rotated += 1;
                // zero trig: the rotation angle comes from one sqrt, and the
                // phase e^{-i phi} is conj(gamma)/|gamma|.
                let g = gamma.norm();
                let (c, s) = if alpha == beta {
                    let h = std::f64::consts::FRAC_1_SQRT_2;
                    (h, h)
                } else {
                    let tau = (alpha - beta) / (2.0 * g);
                    let t = if tau >= 0.0 {
                        1.0 / (tau + (1.0 + tau * tau).sqrt())
                    } else {
                        -1.0 / (-tau + (1.0 + tau * tau).sqrt())
                    };
                    let c = 1.0 / (1.0 + t * t).sqrt();
                    (c, t * c)
                };
                let phase = gamma.conj() * (1.0 / g);
                let sp = s * phase;
                let cp = c * phase;
                unsafe {
                    let bp = b.as_mut_ptr().add(p * nrows);
                    let bq = b.as_mut_ptr().add(q * nrows);
                    for i in 0..nrows {
                        let x = *bp.add(i);
                        let y = *bq.add(i);
                        *bp.add(i) = x * c + y * sp;
                        *bq.add(i) = x * (-s) + y * cp;
                    }
                    let vp = v.as_mut_ptr().add(p * ncols);
                    let vq = v.as_mut_ptr().add(q * ncols);
                    for i in 0..ncols {
                        let x = *vp.add(i);
                        let y = *vq.add(i);
                        *vp.add(i) = x * c + y * sp;
                        *vq.add(i) = x * (-s) + y * cp;
                    }
                }
                let _ = one;
            }
        }
        if rotated == 0 {
            converged = true;
            break;
        }
    }
    if !converged {
        return None;
    }

    let mut sigma = vec![0.0_f64; ncols];
    for (j, s) in sigma.iter_mut().enumerate() {
        let mut nrm = 0.0_f64;
        unsafe {
            let bp = b.as_ptr().add(j * nrows);
            for i in 0..nrows {
                let x = *bp.add(i);
                nrm += (x * x.conj()).re;
            }
        }
        *s = nrm.sqrt();
    }
    let total: f64 = sigma.iter().map(|s| s * s).sum::<f64>().max(1e-300);
    let tau = 1e-12 * total.sqrt();
    let mut idx: Vec<usize> = (0..r).collect();
    idx.sort_by(|&a, &c| {
        sigma[c]
            .partial_cmp(&sigma[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut k = 0usize;
    for &i in &idx {
        if k >= bond_dim || sigma[i] <= tau {
            break;
        }
        k += 1;
    }
    let k = k.max(1);
    let mut t1 = DMatrix::<Complex64>::zeros(nrows, k);
    let mut t2 = DMatrix::<Complex64>::zeros(2 * k, d_r2);
    let mut kept = 0.0_f64;
    for (j, &ci) in idx[..k].iter().enumerate() {
        kept += sigma[ci] * sigma[ci];
        if sigma[ci] > 0.0 {
            let inv = 1.0 / sigma[ci];
            unsafe {
                let bp = b.as_ptr().add(ci * nrows);
                for i in 0..nrows {
                    t1[(i, j)] = *bp.add(i) * inv;
                }
            }
            let sj = Complex64::new(sigma[ci], 0.0);
            for q_out in 0..2usize {
                for rr in 0..d_r2 {
                    t2[(q_out * k + j, rr)] = sj * v[(q_out * d_r2 + rr) + ci * ncols].conj();
                }
            }
        }
    }
    let eps = ((total - kept) / total).clamp(0.0, 1.0);
    Some((t1, t2, eps, sigma, sweeps))
}

fn recon_err(
    m: &DMatrix<Complex64>,
    t1: &DMatrix<Complex64>,
    t2: &DMatrix<Complex64>,
    d_r2: usize,
) -> f64 {
    let (nrows, ncols) = m.shape();
    let k = t1.ncols();
    let mut num = 0.0_f64;
    let mut den = 0.0_f64;
    for i in 0..nrows {
        for c in 0..ncols {
            let (q_out, rr) = (c / d_r2.max(1), c % d_r2.max(1));
            let mut acc = Complex64::new(0.0, 0.0);
            for j in 0..k {
                acc += t1[(i, j)] * t2[(q_out * k + j, rr)];
            }
            num += (acc - m[(i, c)]).norm_sqr();
            den += m[(i, c)].norm_sqr();
        }
    }
    (num / den.max(1e-300)).sqrt()
}

/// The gradient benchmark's circuit shape: `layers` of Ry-all / CX-even /
/// CX-odd / Rz-all, one parameter per rotation.  n=16, L=3 gives 141 gates of
/// which 45 are two-qubit — the mix the cap-cell gap is measured on.
fn circuit_dag(n: usize, layers: usize) -> sf_ir::QuantumDAG {
    let mut dag = sf_ir::QuantumDAG::new(n, 0);
    let mut k = 0.0_f64;
    for _ in 0..layers {
        for q in 0..n {
            k += 0.017;
            dag.add_op(sf_ir::OpType::Ry(sf_ir::Parameter::Const(k)), &[q]);
        }
        for q in (0..n.saturating_sub(1)).step_by(2) {
            dag.add_op(sf_ir::OpType::CNOT, &[q, q + 1]);
        }
        for q in (1..n.saturating_sub(1)).step_by(2) {
            dag.add_op(sf_ir::OpType::CNOT, &[q, q + 1]);
        }
        for q in 0..n {
            k += 0.013;
            dag.add_op(sf_ir::OpType::Rz(sf_ir::Parameter::Const(k)), &[q]);
        }
    }
    dag
}

fn main() {
    let debug = std::env::var_os("SF_MPS_SVD_DEBUG").is_some();
    if debug {
        println!("(SF_MPS_SVD_DEBUG is set: mps.rs will print per-merge lines)");
    }

    // ── 1. policy lookup ──
    for key in ["SF_MPS_GAUGE", "SF_MPS_SVD_DEBUG_NOT_SET"] {
        let n = 200_000;
        let t0 = Instant::now();
        let mut hits = 0usize;
        for _ in 0..n {
            if std::env::var(key).is_ok() {
                hits += 1;
            }
        }
        let per = t0.elapsed().as_secs_f64() / n as f64;
        println!(
            "env::var({key:<30}) {:>8.0} ns/call  hits={hits}",
            per * 1e9
        );
    }

    // ── 2. merge at fixed shapes ──
    println!();
    println!("{:>5} {:>10} {:>10}", "d", "us/merge", "cells");
    for d in [1usize, 2, 4, 8, 16, 32] {
        let (t1, t2) = fresh(d);
        let gate = DMatrix::from_row_slice(
            4,
            4,
            &[
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
            ],
        );
        let reps = if d <= 8 { 2000 } else { 200 };
        for _ in 0..20 {
            let (mut a, mut b) = (t1.clone(), t2.clone());
            black_box(MPSState::apply_2q_gate_static(
                &mut a, &mut b, 0, 1, &gate, d,
            ));
        }
        let t0 = Instant::now();
        for _ in 0..reps {
            let (mut a, mut b) = (t1.clone(), t2.clone());
            black_box(MPSState::apply_2q_gate_static(
                &mut a, &mut b, 0, 1, &gate, d,
            ));
        }
        let per = t0.elapsed().as_secs_f64() / reps as f64;
        println!("{:>5} {:>10.2} {:>10}", d, per * 1e6, (2 * d) * (2 * d));
    }

    // ── 3. raw ThinSvd on the same shapes ──
    println!();
    println!("{:>5} {:>14}", "d", "us/ThinSvd");
    for d in [1usize, 2, 4, 8, 16] {
        let m = FaerMat::<Complex64>::from_fn(2 * d, 2 * d, |i, j| rnd(i, j, 1.0));
        let reps = if d <= 8 { 2000 } else { 200 };
        for _ in 0..20 {
            black_box(ThinSvd::new(m.as_ref()));
        }
        let t0 = Instant::now();
        for _ in 0..reps {
            black_box(ThinSvd::new(m.as_ref()));
        }
        let per = t0.elapsed().as_secs_f64() / reps as f64;
        println!("{:>5} {:>14.2}", d, per * 1e6);
    }

    // ── 4. candidate: exact Jacobi SVD, correctness then timing ──
    println!();
    println!(
        "{:>5} {:>11} {:>11} {:>10} {:>8} {:>9} {:>9} {:>8} {:>9}",
        "d", "recon_faer", "recon_jac", "sig_rel", "dk", "deps", "sweeps", "us/faer", "us/jac"
    );
    for d in [1usize, 2, 4, 8, 16, 32] {
        let (t1, t2) = fresh(d);
        let (nrows, ncols) = (2 * d, 2 * d);
        // M[(o1,l),(o2,r)] = sum g[2o1+o2, 2i1+i2] t1_{i1}[l,m] t2_{i2}[m,r]
        let gate = DMatrix::from_row_slice(
            4,
            4,
            &[
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
            ],
        );
        let mut m = DMatrix::<Complex64>::zeros(nrows, ncols);
        for o1 in 0..2usize {
            for o2 in 0..2usize {
                for l in 0..d {
                    for rr in 0..d {
                        let mut acc = Complex64::new(0.0, 0.0);
                        for i1 in 0..2usize {
                            for i2 in 0..2usize {
                                let mut mid = Complex64::new(0.0, 0.0);
                                for mm in 0..d {
                                    mid += t1[(i1 * d + l, mm)] * t2[(i2 * d + mm, rr)];
                                }
                                acc += gate[(2 * o1 + o2, 2 * i1 + i2)] * mid;
                            }
                        }
                        m[(o1 * d + l, o2 * d + rr)] = acc;
                    }
                }
            }
        }
        let (mut b1, mut b2) = (t1.clone(), t2.clone());
        let eps_f = MPSState::apply_2q_gate_static(&mut b1, &mut b2, 0, 1, &gate, d);
        let recon_f = recon_err(&m, &b1, &b2, d);
        let k_f = b1.ncols();

        let (j1, j2, eps_j, sigma_j, sweeps) =
            small_merge_flat(&m, d, d, d, 1e-15, 40).expect("jacobi converged");
        let recon_j = recon_err(&m, &j1, &j2, d);

        let fm = FaerMat::<Complex64>::from_fn(nrows, ncols, |i, j| m[(i, j)]);
        let svd = ThinSvd::new(fm.as_ref());
        let mut sf_sorted: Vec<f64> = (0..nrows.min(ncols))
            .map(|j| svd.s_diagonal().read(j).norm())
            .collect();
        sf_sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
        let smax = sf_sorted.first().copied().unwrap_or(1.0).max(1e-300);
        let mut sig_j_sorted = sigma_j.clone();
        sig_j_sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
        let sig_rel = sf_sorted
            .iter()
            .zip(sig_j_sorted.iter())
            .map(|(a, c)| ((a - c) / smax).abs())
            .fold(0.0_f64, f64::max);

        let reps = if d <= 8 { 2000 } else { 200 };
        for _ in 0..20 {
            small_merge_flat(&m, d, d, d, 1e-15, 40);
        }
        let t0 = Instant::now();
        for _ in 0..reps {
            let (mut c1, mut c2) = (t1.clone(), t2.clone());
            black_box(MPSState::apply_2q_gate_static(
                &mut c1, &mut c2, 0, 1, &gate, d,
            ));
        }
        let per_f = t0.elapsed().as_secs_f64() / reps as f64;
        let t0 = Instant::now();
        for _ in 0..reps {
            black_box(small_merge_flat(&m, d, d, d, 1e-15, 40));
        }
        let per_j = t0.elapsed().as_secs_f64() / reps as f64;
        println!(
            "{:>5} {:>11.3e} {:>11.3e} {:>10.2e} {:>8} {:>9.1e} {:>9} {:>8.2} {:>9.2}",
            d,
            recon_f,
            recon_j,
            sig_rel,
            (k_f as i64 - j1.ncols() as i64),
            (eps_f - eps_j).abs(),
            sweeps,
            per_f * 1e6,
            per_j * 1e6
        );
    }

    // ── 5. whole-evolution cost, and where it goes ──
    // Needs SF_MPS_PROF=1 for the phase columns; without it only the totals
    // are available and the phase line is skipped.
    println!("\n== evolve_mps, n=16 L=3 (141 gates, 45 two-qubit)");
    println!(
        "\n{:>5} {:>13} {:>10}  phase split (ms/evolution)",
        "chi", "ms/evol", "us/gate"
    );
    let dag = circuit_dag(16, 3);
    for chi in [1usize, 2, 4, 8, 64] {
        let reps = if chi <= 4 { 40 } else { 20 };
        for _ in 0..3 {
            black_box(dag.evolve_mps(chi));
        }
        MPSState::mps_prof_dump("warmup");
        let t0 = Instant::now();
        for _ in 0..reps {
            black_box(dag.evolve_mps(chi));
        }
        let el = t0.elapsed().as_secs_f64() / reps as f64;
        println!("{:>5} {:>10.3} ms {:>8.3}", chi, el * 1e3, el * 1e6 / 141.0);
        MPSState::mps_prof_dump(&format!("chi={chi}"));
    }

    // ── 6. gate-matrix construction ──
    // The evolve loop calls `OpType::to_matrix()` once per gate per evolution,
    // so at 141 gates x 192 evolutions any per-call cost here is multiplied by
    // 27k.  Measured because `other` in the phase split is dominated by it.
    // Bound parameters only: an unbound `Parameter::Variable` panics in
    // `to_matrix` by design, so it is not a cost this loop ever pays.
    println!("\n== OpType::to_matrix(), per call");
    let ops: [(&str, sf_ir::OpType); 4] = [
        (
            "Ry(Const)",
            sf_ir::OpType::Ry(sf_ir::Parameter::Const(0.37)),
        ),
        (
            "Rz(Const)",
            sf_ir::OpType::Rz(sf_ir::Parameter::Const(0.37)),
        ),
        ("H", sf_ir::OpType::H),
        ("CNOT", sf_ir::OpType::CNOT),
    ];
    for (name, op) in ops {
        let reps = 200_000usize;
        for _ in 0..1000 {
            black_box(op.to_matrix());
        }
        let t0 = Instant::now();
        for _ in 0..reps {
            black_box(op.to_matrix());
        }
        let fast_ns = t0.elapsed().as_secs_f64() * 1e9 / reps as f64;
        // The path `to_matrix` took before `to_matrix_small`: build the nested
        // `Vec<Vec<_>>`, then `from_iterator` (column-major) then `transpose`.
        // Reconstructed here from the public `to_unitary_matrix` so the A/B is
        // both arms in **one process**, not two runs that a busy host can
        // separate — the reason §44.3's per-call numbers are re-measurable.
        let old = || {
            let u = op.to_unitary_matrix().unwrap();
            let (r, c) = (u.len(), u[0].len());
            nalgebra::DMatrix::from_iterator(r, c, u.into_iter().flatten()).transpose()
        };
        for _ in 0..1000 {
            black_box(old());
        }
        let t1 = Instant::now();
        for _ in 0..reps {
            black_box(old());
        }
        let old_ns = t1.elapsed().as_secs_f64() * 1e9 / reps as f64;
        println!(
            "{:>13} {:8.1} ns/call  (vs {:8.1} via the old Vec<Vec>+transpose path, {:.2}x)",
            name,
            fast_ns,
            old_ns,
            old_ns / fast_ns
        );
    }

    // ── 7. residue isolation: 1q-only vs 2q-only circuits ──
    // Section 5's `other` is whatever the four spans do not cover.  Running the
    // same n=16 circuit as a rotations-only and a CX-only variant attributes
    // that residue without span arithmetic: each variant exercises exactly one
    // path, so its own total minus its own spans is that path's loop overhead.
    println!("\n== residue isolation at chi=1, n=16");
    let mut rot_only = sf_ir::QuantumDAG::new(16, 0);
    let mut k = 0.0_f64;
    for _ in 0..6 {
        for q in 0..16 {
            k += 0.017;
            rot_only.add_op(sf_ir::OpType::Ry(sf_ir::Parameter::Const(k)), &[q]);
            k += 0.013;
            rot_only.add_op(sf_ir::OpType::Rz(sf_ir::Parameter::Const(k)), &[q]);
        }
    }
    let mut cx_only = sf_ir::QuantumDAG::new(16, 0);
    for _ in 0..3 {
        for q in (0..15).step_by(2) {
            cx_only.add_op(sf_ir::OpType::CNOT, &[q, q + 1]);
        }
        for q in (1..15).step_by(2) {
            cx_only.add_op(sf_ir::OpType::CNOT, &[q, q + 1]);
        }
    }
    println!(
        "\n{:<12} {:>6} {:>13} {:>10}",
        "circuit", "gates", "ms/evol", "us/gate"
    );
    for (name, d, ngates) in [
        ("192x Ry/Rz", &rot_only, 192usize),
        ("48x CNOT", &cx_only, 48usize),
    ] {
        let reps = 40usize;
        for _ in 0..3 {
            black_box(d.evolve_mps(1));
        }
        MPSState::mps_prof_dump("warmup");
        let t0 = Instant::now();
        for _ in 0..reps {
            black_box(d.evolve_mps(1));
        }
        let el = t0.elapsed().as_secs_f64() / reps as f64;
        println!(
            "{:<12} {:>6} {:>10.3} ms {:>8.3}",
            name,
            ngates,
            el * 1e3,
            el * 1e6 / ngates as f64
        );
        MPSState::mps_prof_dump(name);
    }
}
