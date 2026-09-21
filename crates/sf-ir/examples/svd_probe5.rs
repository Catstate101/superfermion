//! Follow-up to svd_probe4: for the faer degenerate-complex-SVD failure case,
//! test candidate repairs:
//!   (a) u-side:  M' = U_k (U_k^H M)             — self-consistent by construction
//!   (b) v-side:  M' = (M V_k) V_k^H
//!   (c) nalgebra's complex SVD (0.33.2) as an alternative backend
//!
//! Also prints the per-column consistency ||M^H u_c|| and ||M v_c|| vs σ_c to
//! see which factor (if either) is trustworthy in the degenerate case.
//!
//! Run: cargo run --release -p sf-ir --example svd_probe5

use faer::linalg::solvers::ThinSvd;
use faer::Mat as FaerMat;
use nalgebra::DMatrix;
use num_complex::Complex64;

fn c(re: f64, im: f64) -> Complex64 {
    Complex64::new(re, im)
}

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

fn main() {
    let n = 16usize;
    let sq2 = 2.0_f64.sqrt();
    let sig: Vec<f64> = std::iter::repeat(sq2)
        .take(8)
        .chain(std::iter::repeat(0.0).take(8))
        .collect();
    let m = build(n, n, &sig, 3);
    println!(
        "case A: {n}x{n}, sigma=[{sq2:.4} x8, 0 x8], ||M||_F^2={:.6}",
        frob_sq(&m)
    );

    let svd = ThinSvd::new(m.as_ref());
    let u = svd.u().to_owned();
    let v = svd.v().to_owned();
    let mut s = Vec::new();
    for k in 0..svd.s_diagonal().nrows() {
        s.push(svd.s_diagonal().read(k).norm());
    }

    // Per-column consistency: ||M^H u_c|| and ||M v_c|| vs sigma_c
    println!("col | faer_s |  ||M^H u_c|| |  ||M v_c||");
    for k in 0..n {
        let mut a = 0.0_f64;
        for j in 0..n {
            let mut acc = c(0.0, 0.0);
            for i in 0..n {
                acc += m.read(i, j).conj() * u.read(i, k);
            }
            a += acc.norm_sqr();
        }
        let mut b = 0.0_f64;
        for i in 0..n {
            let mut acc = c(0.0, 0.0);
            for j in 0..n {
                acc += m.read(i, j) * v.read(j, k);
            }
            b += acc.norm_sqr();
        }
        println!(
            "{k:3} | {:6.4} | {:11.5} | {:10.5}",
            s[k],
            a.sqrt(),
            b.sqrt()
        );
    }

    let k = 8usize;
    // (a) u-side repair: B = U_k^H M (k x n), M' = U_k B
    let mut b_mat = vec![vec![c(0.0, 0.0); n]; k];
    for (ci, row) in b_mat.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            let mut acc = c(0.0, 0.0);
            for i in 0..n {
                acc += u.read(i, ci).conj() * m.read(i, j);
            }
            *cell = acc;
        }
    }
    let mut num_u = 0.0;
    for i in 0..n {
        for j in 0..n {
            let mut acc = c(0.0, 0.0);
            for (ci, row) in b_mat.iter().enumerate().take(k) {
                acc += u.read(i, ci) * row[j];
            }
            num_u += (acc - m.read(i, j)).norm_sqr();
        }
    }
    println!(
        "(a) u-repair recon (k={k}) = {:.3e}",
        (num_u / frob_sq(&m)).sqrt()
    );
    let b_norm_sq: f64 = b_mat.iter().flatten().map(|x| x.norm_sqr()).sum();
    println!(
        "    eps via u-repair = {:.3e}",
        1.0 - b_norm_sq / frob_sq(&m)
    );

    // (b) v-side repair: C = M V_k (n x k), M' = C V_k^H
    let mut c_mat = vec![vec![c(0.0, 0.0); k]; n];
    for (i, row) in c_mat.iter_mut().enumerate() {
        for (ci, cell) in row.iter_mut().enumerate() {
            let mut acc = c(0.0, 0.0);
            for j in 0..n {
                acc += m.read(i, j) * v.read(j, ci);
            }
            *cell = acc;
        }
    }
    let mut num_v = 0.0;
    for i in 0..n {
        for j in 0..n {
            let mut acc = c(0.0, 0.0);
            for ci in 0..k {
                acc += c_mat[i][ci] * v.read(j, ci).conj();
            }
            num_v += (acc - m.read(i, j)).norm_sqr();
        }
    }
    println!(
        "(b) v-repair recon (k={k}) = {:.3e}",
        (num_v / frob_sq(&m)).sqrt()
    );

    // (c) nalgebra complex SVD
    let m_na = DMatrix::<Complex64>::from_fn(n, n, |i, j| m.read(i, j));
    let svd_na = m_na.clone().svd(true, true);
    let s_na = svd_na.singular_values.clone();
    println!(
        "(c) nalgebra s = {:?}",
        (0..s_na.len())
            .map(|i| format!("{:.4}", s_na[i]))
            .collect::<Vec<_>>()
    );
    if let (Some(u_na), Some(vt_na)) = (svd_na.u.as_ref(), svd_na.v_t.as_ref()) {
        // convention check: M = U S V^T (v_t as returned)
        let mut e1 = 0.0;
        let mut e2 = 0.0;
        for i in 0..n {
            for j in 0..n {
                let mut a1 = c(0.0, 0.0);
                let mut a2 = c(0.0, 0.0);
                for kk in 0..k {
                    let sk = c(s_na[kk], 0.0);
                    a1 += sk * u_na[(i, kk)] * vt_na[(kk, j)];
                    a2 += sk * u_na[(i, kk)] * vt_na[(j, kk)].conj();
                }
                e1 += (a1 - m.read(i, j)).norm_sqr();
                e2 += (a2 - m.read(i, j)).norm_sqr();
            }
        }
        println!(
            "(c) nalgebra recon  c1 (U S Vt)   = {:.3e}",
            (e1 / frob_sq(&m)).sqrt()
        );
        println!(
            "(c) nalgebra recon  c2 (U S Vt^H) = {:.3e}",
            (e2 / frob_sq(&m)).sqrt()
        );
        println!(
            "(c) nalgebra shapes: u={}x{} vt={}x{}",
            u_na.nrows(),
            u_na.ncols(),
            vt_na.nrows(),
            vt_na.ncols()
        );
    }
}
