//! Micro-probe: cost of faer `Qr` vs `ThinSvd` at MPS-relevant merged-tensor
//! sizes, mirroring exactly how `apply_2q_gate_static` uses them.
//!
//! Run with:
//!   cargo run --release -p sf-ir --example svd_probe
//!
//! Sizes are (nrows, ncols) pairs actually seen in the brick12@26q cell at
//! bond cap 64: early merges are (68,68), mid merges (84,84), the widest
//! binding step (110,84), and cap-width merges (128,128).

use faer::linalg::solvers::Qr as FaerQr;
use faer::linalg::solvers::ThinSvd;
use faer::Mat as FaerMat;
use num_complex::Complex64;
use std::hint::black_box;
use std::time::Instant;

fn make_mat(nrows: usize, ncols: usize) -> FaerMat<Complex64> {
    FaerMat::from_fn(nrows, ncols, |i, j| {
        let x = ((i * 7919 + j * 104729 + 13) % 1000) as f64;
        let y = ((i * 104729 + j * 7919 + 37) % 1000) as f64;
        Complex64::new(x / 1000.0 - 0.5, y / 1000.0 - 0.5)
    })
}

fn main() {
    let sizes: [(usize, usize); 6] = [(4, 2), (16, 16), (68, 68), (84, 84), (110, 84), (128, 128)];

    println!(
        "{:>10}  {:>10}  {:>10}  {:>12}  {:>12}  {:>8}",
        "shape", "qr(R) us", "qr(Q) us", "svd(s+uv) us", "svd full us", "svd/qr"
    );
    for &(nrows, ncols) in &sizes {
        let m = make_mat(nrows, ncols);
        let r = std::cmp::min(nrows, ncols);
        let reps = if nrows * ncols >= 4096 { 400 } else { 4000 };
        let reps_qr = reps * 3;

        // QR without Q (the override-decision path).
        for _ in 0..20 {
            let qr = FaerQr::new(m.as_ref());
            black_box(qr.compute_r().nrows());
        }
        let t0 = Instant::now();
        for _ in 0..reps_qr {
            let qr = FaerQr::new(m.as_ref());
            black_box(qr.compute_r().nrows());
        }
        let t_qr_r = t0.elapsed().as_secs_f64() / reps_qr as f64;

        // QR with Q (the accept path at binding steps).
        for _ in 0..20 {
            let qr = FaerQr::new(m.as_ref());
            black_box(qr.compute_q().nrows());
        }
        let t0 = Instant::now();
        for _ in 0..reps_qr {
            let qr = FaerQr::new(m.as_ref());
            black_box(qr.compute_q().nrows());
        }
        let t_qr_q = t0.elapsed().as_secs_f64() / reps_qr as f64;

        // ThinSvd, singular values only.
        for _ in 0..20 {
            let svd = ThinSvd::new(m.as_ref());
            black_box(svd.s_diagonal().read(0).norm());
        }
        let t0 = Instant::now();
        for _ in 0..reps {
            let svd = ThinSvd::new(m.as_ref());
            black_box(svd.s_diagonal().read(0).norm());
        }
        let t_svd_s = t0.elapsed().as_secs_f64() / reps as f64;

        // ThinSvd, full usage: s + u + v, reading all elements (mirrors
        // svd_truncate_merged's U_k / V_k materialisation loops).
        for _ in 0..10 {
            let svd = ThinSvd::new(m.as_ref());
            let u = svd.u();
            let v = svd.v();
            let mut acc = 0.0_f64;
            for j in 0..r {
                for i in 0..nrows {
                    acc += u.read(i, j).norm_sqr();
                }
                for i in 0..ncols {
                    acc += v.read(i, j).norm_sqr();
                }
            }
            black_box(acc);
        }
        let t0 = Instant::now();
        for _ in 0..reps {
            let svd = ThinSvd::new(m.as_ref());
            let u = svd.u();
            let v = svd.v();
            let mut acc = 0.0_f64;
            for j in 0..r {
                for i in 0..nrows {
                    acc += u.read(i, j).norm_sqr();
                }
                for i in 0..ncols {
                    acc += v.read(i, j).norm_sqr();
                }
            }
            black_box(acc);
        }
        let t_svd_full = t0.elapsed().as_secs_f64() / reps as f64;

        println!(
            "{:>10}  {:>10.2}  {:>10.2}  {:>12.2}  {:>12.2}  {:>8.2}",
            format!("{}x{}", nrows, ncols),
            t_qr_r * 1e6,
            t_qr_q * 1e6,
            t_svd_s * 1e6,
            t_svd_full * 1e6,
            t_svd_full / t_qr_r.max(1e-12),
        );
    }
}
