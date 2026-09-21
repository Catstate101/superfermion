//! Minimal reproduction of the `ThinSvd` inconsistency found by the engine's
//! SF_MPS_SVD_DEBUG self-check (26q brick12@cap64, seq=20):
//!
//!   M = 16x16, spectrum [√2 ×8, exactly 0 ×8]  →
//!   u() is orthonormal (gram ≈ 2.6e-15) but
//!   ||M − U_k Σ_k V_k^H||_F / ||M||_F ≈ 1.8e-2.
//!
//! This probe builds exactly-degenerate rank-deficient complex matrices with
//! known factors and cross-checks, per singular index c:
//!   - s_diag[c]  vs  ||M^H u_c||  vs  ||M v_c||   (the three must agree)
//!   - recon via (u, s_diag, v) under the verified c1 convention
//!   - orthonormality of u/v
//! for: ThinSvd (current engine path), full Svd, and nalgebra SVD if it
//! supports complex scalars.
//!
//! Run: cargo run --release -p sf-ir --example svd_probe4

use faer::linalg::solvers::{Svd, ThinSvd};
use faer::Mat as FaerMat;
use num_complex::Complex64;

fn c(re: f64, im: f64) -> Complex64 {
    Complex64::new(re, im)
}

/// Deterministic pseudo-random complex isometric columns (n x r), Gram-Schmidt.
fn iso_cols(n: usize, r: usize, seed: u64) -> Vec<Vec<Complex64>> {
    let mut cols: Vec<Vec<Complex64>> = Vec::new();
    for j in 0..r {
        let mut v: Vec<Complex64> = (0..n)
            .map(|i| {
                let a = ((i * 37 + j * 101 + seed as usize * 7 + 11) % 97) as f64 / 97.0 - 0.5;
                let b = ((i * 53 + j * 71 + seed as usize * 13 + 29) % 89) as f64 / 89.0 - 0.5;
                c(a, b)
            })
            .collect();
        for q in &cols {
            let ip: Complex64 = (0..n).map(|i| q[i].conj() * v[i]).sum();
            for i in 0..n {
                v[i] -= ip * q[i];
            }
        }
        let nrm: f64 = v.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt();
        for x in v.iter_mut() {
            *x /= nrm;
        }
        cols.push(v);
    }
    cols
}

/// Deterministic pseudo-random real isometric columns (n x r).
fn real_iso(n: usize, r: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut cols: Vec<Vec<f64>> = Vec::new();
    for j in 0..r {
        let mut v: Vec<f64> = (0..n)
            .map(|i| ((i * 37 + j * 101 + seed as usize * 7 + 11) % 97) as f64 / 97.0 - 0.5)
            .collect();
        for q in &cols {
            let ip: f64 = (0..n).map(|i| q[i] * v[i]).sum();
            for i in 0..n {
                v[i] -= ip * q[i];
            }
        }
        let nrm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        for x in v.iter_mut() {
            *x /= nrm;
        }
        cols.push(v);
    }
    cols
}

/// M = U diag(sigma) V^H with U (n x r), V (m x r) isometric.
fn build(n: usize, m: usize, sigma: &[f64], seed: u64) -> FaerMat<Complex64> {
    let r = sigma.len();
    let u = iso_cols(n, r, seed);
    let v = iso_cols(m, r, seed + 1000);
    FaerMat::from_fn(n, m, |i, j| {
        let mut acc = c(0.0, 0.0);
        for k in 0..r {
            acc += u[k][i] * sigma[k] * v[k][j].conj();
        }
        acc
    })
}

fn frob_sq(m: &FaerMat<Complex64>) -> f64 {
    let mut s = 0.0;
    for i in 0..m.nrows() {
        for j in 0..m.ncols() {
            s += m.read(i, j).norm_sqr();
        }
    }
    s
}

/// ||M||_F of (M^H u_c) style quantities: v^H M^H u.
fn thin_checks(
    label: &str,
    m: &FaerMat<Complex64>,
    u: &FaerMat<Complex64>,
    v: &FaerMat<Complex64>,
    s: &[f64],
) {
    let nrows = m.nrows();
    let ncols = m.ncols();
    let r = s.len();
    // recon under c1: M[i,j] = sum_k s_k u[i,k] conj(v[j,k])
    let mut num = 0.0;
    for i in 0..nrows {
        for j in 0..ncols {
            let mut acc = c(0.0, 0.0);
            for k in 0..r {
                acc += s[k] * u.read(i, k) * v.read(j, k).conj();
            }
            num += (acc - m.read(i, j)).norm_sqr();
        }
    }
    println!("  {label}: recon c1 = {:.3e}", (num / frob_sq(m)).sqrt());
    // Per-index consistency: s_c vs ||M^H u_c|| vs ||M v_c||.  For an exact
    // triplet these are all equal to sigma_c regardless of degeneracies.
    let mut worst = 0.0_f64;
    let mut worst_idx = 0usize;
    for k in 0..r {
        let mut a = 0.0_f64;
        for j in 0..ncols {
            let mut acc = c(0.0, 0.0);
            for i in 0..nrows {
                acc += m.read(i, j).conj() * u.read(i, k);
            }
            a += acc.norm_sqr();
        }
        let mut b = 0.0_f64;
        for i in 0..nrows {
            let mut acc = c(0.0, 0.0);
            for j in 0..ncols {
                acc += m.read(i, j) * v.read(j, k);
            }
            b += acc.norm_sqr();
        }
        let rel = (a.sqrt() - s[k]).abs().max((b.sqrt() - s[k]).abs()) / s[0].max(1e-300);
        if rel > worst {
            worst = rel;
            worst_idx = k;
        }
    }
    println!(
        "  {label}: worst |s_c - norms|/s_max = {:.3e} (col {worst_idx})",
        worst
    );
}

fn main() {
    // Case A: the engine's failing structure — rank 8, σ = √2 exactly, tail 0.
    let sq2 = 2.0_f64.sqrt();
    let sig_a: Vec<f64> = std::iter::repeat(sq2)
        .take(8)
        .chain(std::iter::repeat(0.0).take(8))
        .collect();
    let m_a = build(16, 16, &sig_a, 3);
    println!(
        "case A: 16x16, sigma=[{sq2:.4} x8, 0 x8]  ||M||_F^2={:.6}",
        frob_sq(&m_a)
    );

    let svd = ThinSvd::new(m_a.as_ref());
    println!(
        "  ThinSvd shapes: u={}x{} v={}x{} s_len={}",
        svd.u().nrows(),
        svd.u().ncols(),
        svd.v().nrows(),
        svd.v().ncols(),
        svd.s_diagonal().nrows()
    );
    let mut s_t = Vec::new();
    for k in 0..svd.s_diagonal().nrows() {
        s_t.push(svd.s_diagonal().read(k).norm());
    }
    println!(
        "  ThinSvd s = {:?}",
        s_t.iter().map(|x| format!("{:.4}", x)).collect::<Vec<_>>()
    );
    let u_t = svd.u().to_owned();
    let v_t = svd.v().to_owned();
    // recon using the engine helper's exact access pattern
    let mut num = 0.0;
    for i in 0..16 {
        for j in 0..16 {
            let mut acc = c(0.0, 0.0);
            for k in 0..8 {
                acc += c(s_t[k], 0.0) * u_t.read(i, k) * v_t.read(j, k).conj();
            }
            num += (acc - m_a.read(i, j)).norm_sqr();
        }
    }
    println!(
        "  ThinSvd recon (k=8, engine pattern) = {:.3e}",
        (num / frob_sq(&m_a)).sqrt()
    );
    let mut gram = 0.0;
    for a in 0..8 {
        for b in 0..8 {
            let ip: Complex64 = (0..16)
                .map(|i| u_t.read(i, a).conj() * u_t.read(i, b))
                .sum();
            let want = if a == b { 1.0 } else { 0.0 };
            gram += (ip.norm() - want).abs().powi(2);
        }
    }
    println!("  ThinSvd u-gram dev (k=8) = {:.3e}", gram.sqrt());
    let mut gramv = 0.0;
    for a in 0..8 {
        for b in 0..8 {
            let ip: Complex64 = (0..16)
                .map(|j| v_t.read(j, a).conj() * v_t.read(j, b))
                .sum();
            let want = if a == b { 1.0 } else { 0.0 };
            gramv += (ip.norm() - want).abs().powi(2);
        }
    }
    println!("  ThinSvd v-gram dev (k=8) = {:.3e}", gramv.sqrt());
    thin_checks("ThinSvd", &m_a, &u_t, &v_t, &s_t[..8]);

    let full = Svd::new(m_a.as_ref());
    println!(
        "  full Svd shapes: u={}x{} v={}x{} s_len={}",
        full.u().nrows(),
        full.u().ncols(),
        full.v().nrows(),
        full.v().ncols(),
        full.s_diagonal().nrows()
    );
    let mut s_f = Vec::new();
    for k in 0..full.s_diagonal().nrows() {
        s_f.push(full.s_diagonal().read(k).norm());
    }
    println!(
        "  full Svd s = {:?}",
        s_f.iter().map(|x| format!("{:.4}", x)).collect::<Vec<_>>()
    );
    let u_f = full.u().to_owned();
    let v_f = full.v().to_owned();
    let mut numf = 0.0;
    for i in 0..16 {
        for j in 0..16 {
            let mut acc = c(0.0, 0.0);
            for k in 0..8 {
                acc += c(s_f[k], 0.0) * u_f.read(i, k) * v_f.read(j, k).conj();
            }
            numf += (acc - m_a.read(i, j)).norm_sqr();
        }
    }
    println!(
        "  full Svd recon (k=8) = {:.3e}",
        (numf / frob_sq(&m_a)).sqrt()
    );

    // Case B: same structure but σ perturbed to break exact degeneracy.
    let mut sig_b = sig_a.clone();
    for (k, s) in sig_b.iter_mut().enumerate().take(8) {
        *s = sq2 * (1.0 + 1e-13 * (k as f64 + 1.0));
    }
    let m_b = build(16, 16, &sig_b, 3);
    let svd_b = ThinSvd::new(m_b.as_ref());
    let mut s_b = Vec::new();
    for k in 0..svd_b.s_diagonal().nrows() {
        s_b.push(svd_b.s_diagonal().read(k).norm());
    }
    let u_b = svd_b.u().to_owned();
    let v_b = svd_b.v().to_owned();
    let mut numb = 0.0;
    for i in 0..16 {
        for j in 0..16 {
            let mut acc = c(0.0, 0.0);
            for k in 0..8 {
                acc += c(s_b[k], 0.0) * u_b.read(i, k) * v_b.read(j, k).conj();
            }
            numb += (acc - m_b.read(i, j)).norm_sqr();
        }
    }
    println!(
        "case B: near-degenerate sigma (1+1e-13 k) ThinSvd recon = {:.3e}",
        (numb / frob_sq(&m_b)).sqrt()
    );

    // Case C: generic distinct σ (control).
    let sig_c: Vec<f64> = (0..8).map(|k| 1.0 / (1.0 + k as f64)).collect();
    let m_c = build(16, 16, &sig_c, 5);
    let svd_c = ThinSvd::new(m_c.as_ref());
    let mut s_c = Vec::new();
    for k in 0..svd_c.s_diagonal().nrows() {
        s_c.push(svd_c.s_diagonal().read(k).norm());
    }
    let u_c = svd_c.u().to_owned();
    let v_c = svd_c.v().to_owned();
    let mut numc = 0.0;
    for i in 0..16 {
        for j in 0..16 {
            let mut acc = c(0.0, 0.0);
            for k in 0..8 {
                acc += c(s_c[k], 0.0) * u_c.read(i, k) * v_c.read(j, k).conj();
            }
            numc += (acc - m_c.read(i, j)).norm_sqr();
        }
    }
    println!(
        "case C: distinct sigma ThinSvd recon = {:.3e}",
        (numc / frob_sq(&m_c)).sqrt()
    );

    // Case D: real-valued degenerate matrix (structure without complex phases).
    let u_real = real_iso(16, 8, 7);
    let v_real = real_iso(16, 8, 17);
    let m_d = FaerMat::from_fn(16, 16, |i, j| {
        let mut acc = 0.0;
        for k in 0..8 {
            acc += u_real[k][i] * sig_a[k] * v_real[k][j];
        }
        c(acc, 0.0)
    });
    let svd_d = ThinSvd::new(m_d.as_ref());
    let mut s_d = Vec::new();
    for k in 0..svd_d.s_diagonal().nrows() {
        s_d.push(svd_d.s_diagonal().read(k).norm());
    }
    let u_d = svd_d.u().to_owned();
    let v_d = svd_d.v().to_owned();
    let mut numd = 0.0;
    for i in 0..16 {
        for j in 0..16 {
            let mut acc = c(0.0, 0.0);
            for k in 0..8 {
                acc += c(s_d[k], 0.0) * u_d.read(i, k) * v_d.read(j, k).conj();
            }
            numd += (acc - m_d.read(i, j)).norm_sqr();
        }
    }
    println!(
        "case D: real degenerate ThinSvd recon = {:.3e}",
        (numd / frob_sq(&m_d)).sqrt()
    );

    // Case E: shape variants of case A (wide/tall merges).
    for (n, mm) in [(16usize, 8usize), (8, 16), (32, 16), (16, 32)] {
        let r = (n.min(mm)) / 2;
        let sig_e: Vec<f64> = std::iter::repeat(sq2)
            .take(r)
            .chain(std::iter::repeat(0.0).take(n.min(mm) - r))
            .collect();
        let m_e = build(n, mm, &sig_e, 11);
        let svd_e = ThinSvd::new(m_e.as_ref());
        let mut s_e = Vec::new();
        for k in 0..svd_e.s_diagonal().nrows() {
            s_e.push(svd_e.s_diagonal().read(k).norm());
        }
        let u_e = svd_e.u().to_owned();
        let v_e = svd_e.v().to_owned();
        let mut num_e = 0.0;
        for i in 0..n {
            for j in 0..mm {
                let mut acc = c(0.0, 0.0);
                for k in 0..r {
                    acc += c(s_e[k], 0.0) * u_e.read(i, k) * v_e.read(j, k).conj();
                }
                num_e += (acc - m_e.read(i, j)).norm_sqr();
            }
        }
        println!(
            "case E: {n}x{mm} rank {r} degenerate ThinSvd recon = {:.3e}",
            (num_e / frob_sq(&m_e)).sqrt()
        );
    }
}
