//! Decisive probe for faer `ThinSvd` API conventions — distinct singular
//! values, asymmetric complex factors.  The earlier mps.rs unit tests used a
//! real anti-diagonal matrix with ALL σ = 1 and a symmetric V, so they cannot
//! detect: (a) singular values absorbed into u(), (b) a transposed v(),
//! (c) a conjugation-convention mismatch.
//!
//! Run: cargo run --release -p sf-ir --example svd_probe2

use faer::linalg::solvers::ThinSvd;
use faer::Mat as FaerMat;
use num_complex::Complex64;

fn c(re: f64, im: f64) -> Complex64 {
    Complex64::new(re, im)
}

/// Deterministic "random" complex unitary-ish factor via Gram-Schmidt.
fn rand_unitary(n: usize, seed: u64) -> Vec<Vec<Complex64>> {
    let mut cols: Vec<Vec<Complex64>> = Vec::new();
    for j in 0..n {
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

fn main() {
    let n = 4usize;
    let sigmas = [1.0_f64, 0.5, 0.1, 0.01];
    let u = rand_unitary(n, 1);
    let v = rand_unitary(n, 2);
    // M = U * diag(sigma) * V^H
    let m = FaerMat::from_fn(n, n, |i, j| {
        let mut acc = c(0.0, 0.0);
        for k in 0..n {
            acc += u[k][i] * sigmas[k] * v[k][j].conj();
        }
        acc
    });

    let svd = ThinSvd::new(m.as_ref());
    println!(
        "u shape: {}x{}   v shape: {}x{}   s len: {}",
        svd.u().nrows(),
        svd.u().ncols(),
        svd.v().nrows(),
        svd.v().ncols(),
        svd.s_diagonal().nrows()
    );
    let mut s_out = Vec::new();
    for k in 0..n {
        s_out.push(svd.s_diagonal().read(k).norm());
    }
    println!(
        "s = {:?}",
        s_out
            .iter()
            .map(|x| format!("{:.4}", x))
            .collect::<Vec<_>>()
    );

    let uu = svd.u();
    let vv = svd.v();
    let ss = svd.s_diagonal();

    // Conventions to test:
    // c1: v() = V  ->  M = U S V^H:  sum_j s_j u_j conj(v_j)
    // c2: v() = V^H -> M = U S v():  sum_j s_j u_j v_j    (v row j read as column)
    // c3: v() = conj(V) -> M = U S (conj(v))^H: sum_j s_j u_j v_j^T
    let mut err1 = 0.0_f64;
    let mut err2 = 0.0_f64;
    let mut err3 = 0.0_f64;
    let mut mnorm = 0.0_f64;
    for i in 0..n {
        for j in 0..n {
            let mij = m.read(i, j);
            mnorm += mij.norm_sqr();
            let mut a1 = c(0.0, 0.0);
            let mut a2 = c(0.0, 0.0);
            let mut a3 = c(0.0, 0.0);
            for k in 0..n {
                let sk = c(ss.read(k).norm(), 0.0);
                a1 += sk * uu.read(i, k) * vv.read(j, k).conj(); // V[row=j, col=k]
                a2 += sk * uu.read(i, k) * vv.read(k, j); // V^H[row=k, col=j]
                a3 += sk * uu.read(i, k) * vv.read(k, j).conj(); // conj(V)[k, j]
            }
            err1 += (a1 - mij).norm_sqr();
            err2 += (a2 - mij).norm_sqr();
            err3 += (a3 - mij).norm_sqr();
        }
    }
    println!(
        "recon rel err  c1 (v()=V, V^H absorbed): {:.3e}",
        (err1 / mnorm).sqrt()
    );
    println!(
        "recon rel err  c2 (v()=V^H)           : {:.3e}",
        (err2 / mnorm).sqrt()
    );
    println!(
        "recon rel err  c3 (v()=conj(V))       : {:.3e}",
        (err3 / mnorm).sqrt()
    );

    // Is u() isometric, and does it absorb sigma?  Compare u_j against true U_j.
    let mut gram_err = 0.0_f64;
    for k1 in 0..n {
        for k2 in 0..n {
            let ip: Complex64 = (0..n).map(|i| uu.read(i, k1).conj() * uu.read(i, k2)).sum();
            let target = if k1 == k2 { c(1.0, 0.0) } else { c(0.0, 0.0) };
            gram_err += (ip - target).norm_sqr();
        }
    }
    println!("u() gram deviation from I: {:.3e}", gram_err.sqrt());

    // Which column of u() pairs with which sigma?  Match each u_j to the true
    // U columns (overlap magnitude) — exposes U/sigma order mismatches.
    for k in 0..n {
        let mut best = (0usize, 0.0_f64);
        for t in 0..n {
            let ov: Complex64 = (0..n).map(|i| uu.read(i, k).conj() * u[t][i]).sum();
            if ov.norm() > best.1 {
                best = (t, ov.norm());
            }
        }
        println!(
            "u col {k} (s={:.4}) best-matches true U col {} (|ov|={:.4})",
            ss.read(k).norm(),
            best.0,
            best.1
        );
    }

    // Wide matrix shape check (nrows < ncols)
    let w = FaerMat::from_fn(2, 8, |i, j| c((i + j) as f64, (i * j) as f64));
    let svd_w = ThinSvd::new(w.as_ref());
    println!(
        "\nwide 2x8: u={}x{} v={}x{} s_len={}",
        svd_w.u().nrows(),
        svd_w.u().ncols(),
        svd_w.v().nrows(),
        svd_w.v().ncols(),
        svd_w.s_diagonal().nrows()
    );
    // Tall matrix shape check
    let t = FaerMat::from_fn(16, 4, |i, j| c((i % 3 + j) as f64, (i + j * j) as f64));
    let svd_t = ThinSvd::new(t.as_ref());
    println!(
        "tall 16x4: u={}x{} v={}x{} s_len={}",
        svd_t.u().nrows(),
        svd_t.u().ncols(),
        svd_t.v().nrows(),
        svd_t.v().ncols(),
        svd_t.s_diagonal().nrows()
    );
}
