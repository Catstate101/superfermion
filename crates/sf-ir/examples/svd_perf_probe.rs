//! Micro-benchmark of the MPS merge hot path primitives (session-scoped probe).
//!
//! Times, at the saturated brick12 merge shape (128x128 complex):
//!   1. faer ThinSvd::new           (the per-merge SVD)
//!   2. the naive per-column build  (the v-side factor construction loop)
//!   3. faer matmul W = M * V       (all r columns, Rayon)
//!   4. faer matmul W_k = M * V_k   (only the kept k columns)
//! so the 3x-vs-aer brick12 gap can be attributed to a specific primitive.
use faer::linalg::matmul::matmul as faer_matmul;
use faer::linalg::solvers::ThinSvd;
use faer::Mat as FaerMat;
use faer::Parallelism;
use num_complex::Complex64;
use std::hint::black_box;
use std::time::Instant;

fn main() {
    let nrows = 128usize;
    let ncols = 128usize;
    let k = 64usize;
    let reps = 30usize;

    // Deterministic pseudo-random complex matrix (LCG, no rand API risk).
    let mut seed = 0x9E3779B97F4A7C15u64;
    let mut next = move || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((seed >> 11) as f64) / ((1u64 << 53) as f64) - 0.5
    };
    let m = FaerMat::<Complex64>::from_fn(nrows, ncols, |_, _| Complex64::new(next(), next()));

    // ── 1. ThinSvd ──
    for _ in 0..5 {
        let svd = ThinSvd::new(m.as_ref());
        black_box(svd.s_diagonal().read(0));
    }
    let t0 = Instant::now();
    for _ in 0..reps {
        let svd = ThinSvd::new(m.as_ref());
        black_box(svd.s_diagonal().read(0));
    }
    let svd_t = t0.elapsed().as_secs_f64() / reps as f64;

    let svd = ThinSvd::new(m.as_ref());
    let v = svd.v();

    // ── 2. naive build loop, k=64 selected columns (exact mps.rs loop) ──
    let sel: Vec<usize> = (0..k).collect();
    for _ in 0..3 {
        for &ci in &sel {
            let mut w = vec![Complex64::new(0.0, 0.0); nrows];
            for c in 0..ncols {
                let vc = v.read(c, ci);
                for i in 0..nrows {
                    w[i] += m.read(i, c) * vc;
                }
            }
            black_box(&w);
        }
    }
    let t0 = Instant::now();
    for _ in 0..reps {
        for &ci in &sel {
            let mut w = vec![Complex64::new(0.0, 0.0); nrows];
            for c in 0..ncols {
                let vc = v.read(c, ci);
                for i in 0..nrows {
                    w[i] += m.read(i, c) * vc;
                }
            }
            black_box(&w);
        }
    }
    let build_t = t0.elapsed().as_secs_f64() / reps as f64;

    // ── 3. faer GEMM W = M * V, all r columns, Rayon ──
    let r = nrows.min(ncols);
    for _ in 0..3 {
        let mut w = FaerMat::<Complex64>::zeros(nrows, r);
        faer_matmul(
            w.as_mut(),
            m.as_ref(),
            v.as_ref(),
            None,
            Complex64::new(1.0, 0.0),
            Parallelism::Rayon(0),
        );
        black_box(w.read(0, 0));
    }
    let t0 = Instant::now();
    for _ in 0..reps {
        let mut w = FaerMat::<Complex64>::zeros(nrows, r);
        faer_matmul(
            w.as_mut(),
            m.as_ref(),
            v.as_ref(),
            None,
            Complex64::new(1.0, 0.0),
            Parallelism::Rayon(0),
        );
        black_box(w.read(0, 0));
    }
    let gemm_full_t = t0.elapsed().as_secs_f64() / reps as f64;

    // ── 4. faer GEMM W_k = M * V_k (k copied columns) ──
    let vk = FaerMat::<Complex64>::from_fn(ncols, k, |i, j| v.read(i, j));
    for _ in 0..3 {
        let mut w = FaerMat::<Complex64>::zeros(nrows, k);
        faer_matmul(
            w.as_mut(),
            m.as_ref(),
            vk.as_ref(),
            None,
            Complex64::new(1.0, 0.0),
            Parallelism::Rayon(0),
        );
        black_box(w.read(0, 0));
    }
    let t0 = Instant::now();
    for _ in 0..reps {
        let mut w = FaerMat::<Complex64>::zeros(nrows, k);
        faer_matmul(
            w.as_mut(),
            m.as_ref(),
            vk.as_ref(),
            None,
            Complex64::new(1.0, 0.0),
            Parallelism::Rayon(0),
        );
        black_box(w.read(0, 0));
    }
    let gemm_k_t = t0.elapsed().as_secs_f64() / reps as f64;

    println!("shape {nrows}x{ncols} complex, k={k}, reps={reps}");
    println!("  ThinSvd::new          {svd_t:8.4} ms");
    println!("  naive build (k cols)  {build_t:8.4} ms");
    println!("  GEMM M*V (r={r} cols)  {gemm_full_t:8.4} ms");
    println!("  GEMM M*V_k ({k} cols)   {gemm_k_t:8.4} ms");
    println!(
        "  old-path per merge = svd + build = {:.4} ms; gemm path = svd + gemm_full = {:.4} ms",
        (svd_t + build_t) * 1e3,
        (svd_t + gemm_full_t) * 1e3
    );
}
