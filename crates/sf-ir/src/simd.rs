//! AVX2/FMA micro-kernels for the statevector apply loop (runtime dispatch).
//!
//! The portable wheel targets x86-64-v2 (SSE4.2) so shipped binaries run
//! everywhere; this module adds explicitly-vectorized kernels that are
//! selected at runtime when the CPU reports AVX2 + FMA. Every kernel has a
//! scalar fallback that is bit-equivalent to the previous implementation, so
//! non-x86 hosts (and pre-AVX2 x86) keep the old behavior.
//!
//! Layout: state amplitudes are `Complex64` (interleaved re/im f64), so one
//! `__m256d` register holds two complex numbers. Two complex-multiply forms
//! are implemented:
//!
//! ```text
//!   cmul4(v, w)      vr = movedup(v)       // [re, re, re, re] (per complex)
//!   (legacy)         vi = permute(v, 0xF)  // [im, im, im, im]
//!                    ws = permute(w, 0x5)  // [wi, wr, wi, wr]
//!                    res = fmaddsub(vr, w, vi * ws)     // 3 shuffles per mul
//!
//!   cmul_nat(v, mr, mi)  vs = permute(v, 0x5)  // [im, re, im, re]
//!                        res = fmaddsub(v, mr, vs * mi) // 1 shuffle per mul
//! ```
//!
//! `cmul_nat` takes the pre-broadcast constant pair (`mr` = [re x4],
//! `mi` = [im x4]) and keeps the data in its natural interleaved form, so a
//! 2x2 pair transform needs one shuffle per data register instead of three
//! per product — the port-5 shuffle pressure, not the FMA ports, bounds the
//! pair kernels at these sizes. Even output lanes are rounding-identical to
//! `cmul4`; odd lanes commute the two FMA operands (<=1 ulp, same class as
//! the SIMD-vs-scalar agreement asserted in the tests below).
//!
//! Kernels here are serial and operate on caller-provided slices; the
//! parallel (Rayon) structure stays in `dag.rs`, which splits the state into
//! kernel-aligned chunks and calls into this module per chunk.

use num_complex::Complex64;
use rayon::prelude::*;

/// Runtime AVX2+FMA detection (cached).
#[inline]
pub fn avx2_ok() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        use std::sync::atomic::{AtomicU8, Ordering};
        static STATE: AtomicU8 = AtomicU8::new(0);
        match STATE.load(Ordering::Relaxed) {
            1 => true,
            2 => false,
            _ => {
                let ok = std::arch::is_x86_feature_detected!("avx2")
                    && std::arch::is_x86_feature_detected!("fma");
                STATE.store(if ok { 1 } else { 2 }, Ordering::Relaxed);
                ok
            }
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Environment escape hatch: `SF_NO_SIMD=1` forces the scalar fallbacks
/// (used for A/B attribution runs; no effect on results beyond rounding).
/// Shared with the f32 lane kernels in `crate::f32lane`.
#[inline]
pub(crate) fn simd_enabled() -> bool {
    use std::sync::atomic::{AtomicU8, Ordering};
    static STATE: AtomicU8 = AtomicU8::new(0);
    match STATE.load(Ordering::Relaxed) {
        1 => true,
        2 => false,
        _ => {
            let off = std::env::var("SF_NO_SIMD")
                .map(|v| v == "1")
                .unwrap_or(false);
            let ok = !off && avx2_ok();
            STATE.store(if ok { 1 } else { 2 }, Ordering::Relaxed);
            ok
        }
    }
}

/// `SF_PAIR_V2=1` selects the natural-layout complex multiply inside the
/// pair kernels (latched once per process). Default is the historical
/// double-shuffle form: the `pair_v2_microbench` A/B shows the two within
/// noise of each other on hosts with two vector-shuffle ports (this
/// campaign's box — large strides are DRAM-bound and L1-resident chunks
/// are latency/ILP-bound, so the shuffle count is not the limiter), and
/// the historical form is the validated production path.
#[inline]
fn pair_v2_enabled() -> bool {
    use std::sync::atomic::{AtomicU8, Ordering};
    static STATE: AtomicU8 = AtomicU8::new(0);
    match STATE.load(Ordering::Relaxed) {
        1 => true,
        2 => false,
        _ => {
            let on = std::env::var("SF_PAIR_V2")
                .map(|v| v == "1")
                .unwrap_or(false);
            STATE.store(if on { 1 } else { 2 }, Ordering::Relaxed);
            on
        }
    }
}

/// Per-kernel SIMD size floors (amplitudes), latched once per process.
///
/// The floors exist because below them the AVX2 dispatch + cold-kernel cost
/// can outsized the saving (measured on the phase-gate micro at 4096:
/// scalar 1.51 ms vs SIMD 3.85 ms). The previous single floor of 8192 was
/// taken from that micro and applied to every gated kernel; the end-to-end
/// statevector ladder tells a different story for whole-slice sweeps — at
/// dim 4096 the scalar path costs ~0.91 us/amp vs ~0.07 us/amp at dim 8192
/// on the identical workload, i.e. the 8192 floor left n=12 entirely on the
/// slow path. The floor for the whole-slice gates is therefore 4096; the
/// chunked kernels stay ungated. Overridable for A/B attribution runs:
/// `SF_PAIR_MIN=<n>` (pair_pass), `SF_SCALE_MIN=<n>` (pattern_scale_const /
/// pattern_scale_periodic).
const PAIR_SIMD_MIN_DEFAULT: usize = 4096;
const SCALE_SIMD_MIN_DEFAULT: usize = 4096;

/// Size floor of the whole-slice `pair_pass` SIMD kernels (see
/// `PAIR_SIMD_MIN_DEFAULT`); `SF_PAIR_MIN` overrides.
#[inline]
fn pair_simd_min() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("SF_PAIR_MIN")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(PAIR_SIMD_MIN_DEFAULT)
    })
}

/// Size floor of the whole-slice `pattern_scale_*` SIMD kernels (see
/// `SCALE_SIMD_MIN_DEFAULT`); `SF_SCALE_MIN` overrides.
#[inline]
fn scale_simd_min() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("SF_SCALE_MIN")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(SCALE_SIMD_MIN_DEFAULT)
    })
}

// ──────────────────────────────────────────────────────────────────────
// x86_64 AVX2+FMA building blocks
// ──────────────────────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
mod x86 {
    use super::Complex64;
    use std::arch::x86_64::*;

    /// Complex multiply for two interleaved complex numbers per register.
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn cmul4(v: __m256d, w: __m256d) -> __m256d {
        let vr = _mm256_movedup_pd(v);
        let vi = _mm256_permute_pd(v, 0b1111);
        let ws = _mm256_permute_pd(w, 0b0101);
        _mm256_fmaddsub_pd(vr, w, _mm256_mul_pd(vi, ws))
    }

    /// Broadcast a complex scalar into both lanes: [re, im, re, im].
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn splat_c(z: Complex64) -> __m256d {
        _mm256_set_pd(z.im, z.re, z.im, z.re)
    }

    /// Broadcast the real part of a complex scalar: [re, re, re, re].
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn splat_re(z: Complex64) -> __m256d {
        _mm256_set1_pd(z.re)
    }

    /// Broadcast the imaginary part of a complex scalar: [im, im, im, im].
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn splat_im(z: Complex64) -> __m256d {
        _mm256_set1_pd(z.im)
    }

    /// Natural-layout complex multiply (see the module docs): `mr`/`mi` are
    /// the pre-broadcast constant pair, `v` stays `[re, im, re, im]`. One
    /// shuffle, one mul and one fmaddsub per two complex outputs.
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn cmul_nat(v: __m256d, mr: __m256d, mi: __m256d) -> __m256d {
        let vs = _mm256_permute_pd(v, 0b0101);
        _mm256_fmaddsub_pd(v, mr, _mm256_mul_pd(vs, mi))
    }

    /// Load 2 complex (unaligned).
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn load2(p: *const Complex64) -> __m256d {
        _mm256_loadu_pd(p as *const f64)
    }

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn store2(p: *mut Complex64, v: __m256d) {
        _mm256_storeu_pd(p as *mut f64, v);
    }

    /// Non-temporal store of 2 complex.  `p` must be 32-byte aligned (the
    /// instruction faults otherwise) and the caller must `_mm_sfence()` before
    /// anything reads the line: the store bypasses the cache, which is the
    /// point — it skips the read-for-ownership a normal store to a cold line
    /// would pay.
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn store2_nt(p: *mut Complex64, v: __m256d) {
        _mm256_stream_pd(p as *mut f64, v);
    }
}

// ──────────────────────────────────────────────────────────────────────
// Contiguous complex inner product Σ conj(a[i])·b[i] (adjoint generator
// dot). The scalar form is the elementwise reference; the AVX2 kernel
// reuses the production `cmul4` multiply and conjugates `a` with a sign
// xor, so each summand is bit-identical to `a[i].conj() * b[i]` and only
// the horizontal-sum grouping differs (~1 ulp — the same regrouping the
// rayon chunked path in `gen_dot` already accepts). Runs shorter than the
// floor (small generator qubit index) stay on the scalar path: at small
// stride the per-run hsum dispatch outweighs the vector body.
// ──────────────────────────────────────────────────────────────────────

const DOT_SIMD_MIN_DEFAULT: usize = 512;

/// Amplitude floor below which `dot_conj` uses the scalar path; `SF_DOT_MIN`
/// overrides for A/B attribution runs (latched once per process).
#[inline]
pub(crate) fn dot_min() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("SF_DOT_MIN")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(DOT_SIMD_MIN_DEFAULT)
    })
}

/// Σ_{i} conj(a[i]) · b[i] over two equal-length contiguous slices.
#[inline]
pub fn dot_conj(a: &[Complex64], b: &[Complex64]) -> Complex64 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "x86_64")]
    {
        let n = a.len();
        if simd_enabled() && n >= dot_min() {
            // SAFETY: `a` and `b` are equal-length, so `n` complex reads stay
            // in bounds for both; the kernel reads only [0, n) of each.
            unsafe {
                return dot_conj_avx2(a.as_ptr(), b.as_ptr(), n);
            }
        }
    }
    let mut ip = Complex64::new(0.0, 0.0);
    for (x, y) in a.iter().zip(b) {
        ip += x.conj() * *y;
    }
    ip
}

/// AVX2 body of `dot_conj`: 2 complex per register, `cmul4(conj(a), b)`,
/// accumulate, one horizontal sum, scalar tail for an odd count.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_conj_avx2(pa: *const Complex64, pb: *const Complex64, n: usize) -> Complex64 {
    use std::arch::x86_64::*;
    use x86::{cmul4, load2};
    // Negate the imaginary lanes of `[re, im, re, im]` (xor of the sign bit).
    let conj_sign = _mm256_setr_pd(0.0, -0.0, 0.0, -0.0);
    let mut acc = _mm256_setzero_pd();
    let mut k = 0usize;
    while k + 2 <= n {
        let a = load2(pa.add(k));
        let b = load2(pb.add(k));
        let ac = _mm256_xor_pd(a, conj_sign);
        acc = _mm256_add_pd(acc, cmul4(ac, b));
        k += 2;
    }
    let lo = _mm256_extractf128_pd(acc, 0);
    let hi = _mm256_extractf128_pd(acc, 1);
    let mut out = [0.0f64; 2];
    _mm_storeu_pd(out.as_mut_ptr(), _mm_add_pd(lo, hi));
    let mut ip = Complex64::new(out[0], out[1]);
    while k < n {
        ip += (*pa.add(k)).conj() * (*pb.add(k));
        k += 1;
    }
    ip
}

// ──────────────────────────────────────────────────────────────────────
// 2×2 pair transform: out_lo = m00*lo + m01*hi ; out_hi = m10*lo + m11*hi
// Applied to disjoint pairs (i, i+stride) for every block of 2*stride.
// The slice must start at a block boundary (caller guarantees alignment).
// ──────────────────────────────────────────────────────────────────────

#[inline]
pub fn pair_pass(s: &mut [Complex64], stride: usize, m: [[Complex64; 2]; 2]) {
    #[cfg(target_arch = "x86_64")]
    {
        // Whole-slice sweeps use the vector kernels from the per-kernel
        // floor up (`pair_simd_min`); below it the dispatch + cold-kernel
        // cost can outsized the saving.
        if simd_enabled() && s.len() >= pair_simd_min() && stride >= 2 {
            unsafe {
                if pair_v2_enabled() {
                    pair_pass_runs_avx2_v2(s.as_mut_ptr(), s.len(), stride, m);
                } else {
                    pair_pass_runs_avx2(s.as_mut_ptr(), s.len(), stride, m);
                }
            }
            return;
        }
        if simd_enabled() && s.len() >= pair_simd_min() && stride == 1 {
            unsafe {
                if pair_v2_enabled() {
                    pair_pass_stride1_avx2_v2(s.as_mut_ptr(), s.len(), m);
                } else {
                    pair_pass_stride1_avx2(s.as_mut_ptr(), s.len(), m);
                }
            }
            return;
        }
    }
    pair_pass_scalar(s, stride, m);
}

/// `pair_pass` without the small-size fallback: callers that split the
/// state into many chunks (rayon) amortize the dispatch cost across the
/// whole chunk set, so each chunk goes straight to the vector kernel when
/// SIMD is enabled.  Chunk length must be a multiple of 2*stride.
#[inline]
pub fn pair_pass_chunk(s: &mut [Complex64], stride: usize, m: [[Complex64; 2]; 2]) {
    #[cfg(target_arch = "x86_64")]
    {
        if simd_enabled() {
            unsafe {
                if pair_v2_enabled() {
                    if stride >= 2 {
                        pair_pass_runs_avx2_v2(s.as_mut_ptr(), s.len(), stride, m);
                    } else {
                        pair_pass_stride1_avx2_v2(s.as_mut_ptr(), s.len(), m);
                    }
                } else if stride >= 2 {
                    pair_pass_runs_avx2(s.as_mut_ptr(), s.len(), stride, m);
                } else {
                    pair_pass_stride1_avx2(s.as_mut_ptr(), s.len(), m);
                }
            }
            return;
        }
    }
    pair_pass_scalar(s, stride, m);
}

/// `dst = U·src`: the out-of-place twin of `pair_pass`.
///
/// The adjoint's cached forward sweep needs each parameter gate's post-state
/// in its cache slot; producing it there directly removes a whole-state copy
/// per gate (measured at 46-59% of that sweep, see `adjoint::adj_pp_on`).
/// Same formula as `pair_pass`, with the natural-layout arithmetic of the
/// `_v2` kernels; `dst` and `src` must not overlap, must have equal length,
/// and that length must be a multiple of `2·stride`.
#[inline]
pub fn pair_pass_out(
    dst: &mut [Complex64],
    src: &[Complex64],
    stride: usize,
    m: [[Complex64; 2]; 2],
) {
    debug_assert_eq!(dst.len(), src.len());
    debug_assert_eq!(dst.len() % (stride * 2), 0);
    #[cfg(target_arch = "x86_64")]
    {
        if simd_enabled() && dst.len() >= pair_simd_min() {
            unsafe {
                if stride >= 2 {
                    pair_pass_out_runs_avx2(dst.as_mut_ptr(), src.as_ptr(), dst.len(), stride, m);
                } else {
                    pair_pass_out_stride1_avx2(dst.as_mut_ptr(), src.as_ptr(), dst.len(), m);
                }
            }
            return;
        }
    }
    pair_pass_out_scalar(dst, src, stride, m);
}

/// Scalar fallback for `pair_pass_out` — the same expression order as
/// `pair_pass_scalar`, one pair at a time.
fn pair_pass_out_scalar(
    dst: &mut [Complex64],
    src: &[Complex64],
    stride: usize,
    m: [[Complex64; 2]; 2],
) {
    let block = stride * 2;
    for (d, s) in dst.chunks_exact_mut(block).zip(src.chunks_exact(block)) {
        let (dlo, dhi) = d.split_at_mut(stride);
        let (slo, shi) = s.split_at(stride);
        for (((dl, dh), &lo), &hi) in dlo.iter_mut().zip(dhi.iter_mut()).zip(slo).zip(shi) {
            *dl = m[0][0] * lo + m[0][1] * hi;
            *dh = m[1][0] * lo + m[1][1] * hi;
        }
    }
}

/// Natural-layout out-of-place twin of `pair_pass_runs_avx2_v2`: identical
/// arithmetic, loads from `s` and stores to `d`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn pair_pass_out_runs_avx2(
    d: *mut Complex64,
    s: *const Complex64,
    len: usize,
    stride: usize,
    m: [[Complex64; 2]; 2],
) {
    use std::arch::x86_64::*;
    use x86::*;
    let m00r = splat_re(m[0][0]);
    let m00i = splat_im(m[0][0]);
    let m01r = splat_re(m[0][1]);
    let m01i = splat_im(m[0][1]);
    let m10r = splat_re(m[1][0]);
    let m10i = splat_im(m[1][0]);
    let m11r = splat_re(m[1][1]);
    let m11i = splat_im(m[1][1]);
    let block = stride * 2;
    let n_groups = len / block;
    for g in 0..n_groups {
        let db = d.add(g * block);
        let sb = s.add(g * block);
        let mut k = 0usize;
        while k + 4 <= stride {
            let a0 = load2(sb.add(k));
            let a1 = load2(sb.add(k + 2));
            let b0 = load2(sb.add(k + stride));
            let b1 = load2(sb.add(k + stride + 2));
            let oa0 = _mm256_add_pd(cmul_nat(a0, m00r, m00i), cmul_nat(b0, m01r, m01i));
            let oa1 = _mm256_add_pd(cmul_nat(a1, m00r, m00i), cmul_nat(b1, m01r, m01i));
            let ob0 = _mm256_add_pd(cmul_nat(a0, m10r, m10i), cmul_nat(b0, m11r, m11i));
            let ob1 = _mm256_add_pd(cmul_nat(a1, m10r, m10i), cmul_nat(b1, m11r, m11i));
            store2(db.add(k), oa0);
            store2(db.add(k + 2), oa1);
            store2(db.add(k + stride), ob0);
            store2(db.add(k + stride + 2), ob1);
            k += 4;
        }
        while k + 2 <= stride {
            let a = load2(sb.add(k));
            let b = load2(sb.add(k + stride));
            let oa = _mm256_add_pd(cmul_nat(a, m00r, m00i), cmul_nat(b, m01r, m01i));
            let ob = _mm256_add_pd(cmul_nat(a, m10r, m10i), cmul_nat(b, m11r, m11i));
            store2(db.add(k), oa);
            store2(db.add(k + stride), ob);
            k += 2;
        }
    }
}

/// Natural-layout out-of-place twin of `pair_pass_stride1_avx2_v2`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn pair_pass_out_stride1_avx2(
    d: *mut Complex64,
    s: *const Complex64,
    len: usize,
    m: [[Complex64; 2]; 2],
) {
    use std::arch::x86_64::*;
    use x86::*;
    let m00r = splat_re(m[0][0]);
    let m00i = splat_im(m[0][0]);
    let m01r = splat_re(m[0][1]);
    let m01i = splat_im(m[0][1]);
    let m10r = splat_re(m[1][0]);
    let m10i = splat_im(m[1][0]);
    let m11r = splat_re(m[1][1]);
    let m11i = splat_im(m[1][1]);
    let mut k = 0usize;
    while k + 4 <= len {
        let va = load2(s.add(k));
        let vb = load2(s.add(k + 2));
        let lo = _mm256_permute2f128_pd(va, vb, 0x20); // [c0, c2]
        let hi = _mm256_permute2f128_pd(va, vb, 0x31); // [c1, c3]
        let olo = _mm256_add_pd(cmul_nat(lo, m00r, m00i), cmul_nat(hi, m01r, m01i));
        let ohi = _mm256_add_pd(cmul_nat(lo, m10r, m10i), cmul_nat(hi, m11r, m11i));
        store2(d.add(k), _mm256_permute2f128_pd(olo, ohi, 0x20));
        store2(d.add(k + 2), _mm256_permute2f128_pd(olo, ohi, 0x31));
        k += 4;
    }
    while k + 2 <= len {
        let a = *s.add(k);
        let b = *s.add(k + 1);
        *d.add(k) = m[0][0] * a + m[0][1] * b;
        *d.add(k + 1) = m[1][0] * a + m[1][1] * b;
        k += 2;
    }
}

/// `dst = U·src` with non-temporal stores: `pair_pass_out` for a destination
/// that is written once and not read again until it has long left the cache
/// (in the adjoint: a parameter gate's cache slot).  Returns `false` — having
/// touched neither slice — when the destination is not 32-byte aligned, is
/// below the SIMD floor, or SIMD is off; the caller must then use
/// `pair_pass_out`.
///
/// Why this exists: a normal store to a cold line first reads that line for
/// ownership, which is what made the ping-pong apply cost as much as the
/// apply-plus-copy it replaced (see `adjoint::adj_pp_on`).  `vmovntpd` writes
/// the line without that read, at the price of an `sfence` before a reader.
#[inline]
pub fn pair_pass_out_nt(
    dst: &mut [Complex64],
    src: &[Complex64],
    stride: usize,
    m: [[Complex64; 2]; 2],
) -> bool {
    debug_assert_eq!(dst.len(), src.len());
    debug_assert_eq!(dst.len() % (stride * 2), 0);
    #[cfg(target_arch = "x86_64")]
    {
        if simd_enabled() && dst.len() >= pair_simd_min() && (dst.as_ptr() as usize) & 31 == 0 {
            unsafe {
                if stride >= 2 {
                    pair_pass_out_runs_nt_avx2(
                        dst.as_mut_ptr(),
                        src.as_ptr(),
                        dst.len(),
                        stride,
                        m,
                    );
                } else {
                    pair_pass_out_stride1_nt_avx2(dst.as_mut_ptr(), src.as_ptr(), dst.len(), m);
                }
                std::arch::x86_64::_mm_sfence();
            }
            return true;
        }
    }
    false
}

/// `dst = CNOT(q1 -> q2)·src` with non-temporal stores — same contract and
/// fallback rule as [`pair_pass_out_nt`]; `cnot_out_vec` with `nt = true`.
pub fn cnot_out_nt(dst: &mut [Complex64], src: &[Complex64], q1: usize, q2: usize) -> bool {
    cnot_out_vec(dst, src, q1, q2, true)
}

/// `dst = CNOT(q1 -> q2)·src`, vectorised: one 32-byte copy per two
/// amplitudes, or a partner-pair blend when the control is qubit 0.
///
/// Returns `false` — having touched neither slice — when the platform cannot
/// take the path: SIMD off, destination below `pair_simd_min`, target on qubit
/// 0 (`m2 == 1` would reverse each pair), or `nt` with a destination that is
/// not 32-byte aligned.  `nt` implies an `sfence` before returning.
pub fn cnot_out_vec(
    dst: &mut [Complex64],
    src: &[Complex64],
    q1: usize,
    q2: usize,
    nt: bool,
) -> bool {
    let m1 = 1usize << q1;
    let m2 = 1usize << q2;
    #[cfg(target_arch = "x86_64")]
    {
        if simd_enabled()
            && dst.len() >= pair_simd_min()
            && m2 >= 2
            && dst.len() == src.len()
            && (!nt || (dst.as_ptr() as usize) & 31 == 0)
        {
            unsafe {
                if m1 >= 2 {
                    if nt {
                        cnot_pair_core::<true>(dst.as_mut_ptr(), src.as_ptr(), dst.len(), m1, m2);
                    } else {
                        cnot_pair_core::<false>(dst.as_mut_ptr(), src.as_ptr(), dst.len(), m1, m2);
                    }
                } else if nt {
                    cnot_ctrl0_core::<true>(dst.as_mut_ptr(), src.as_ptr(), dst.len(), m2);
                } else {
                    cnot_ctrl0_core::<false>(dst.as_mut_ptr(), src.as_ptr(), dst.len(), m2);
                }
                if nt {
                    std::arch::x86_64::_mm_sfence();
                }
            }
            return true;
        }
    }
    let _ = (m1, m2, nt);
    false
}

/// Streaming twin of `pair_pass_out_runs_avx2`: identical arithmetic and
/// block structure, `store2_nt` in place of `store2`.  Every store address is
/// a multiple of `stride >= 2` complex from the block base, i.e. a multiple of
/// 32 bytes, so an aligned base keeps every store aligned.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn pair_pass_out_runs_nt_avx2(
    d: *mut Complex64,
    s: *const Complex64,
    len: usize,
    stride: usize,
    m: [[Complex64; 2]; 2],
) {
    use std::arch::x86_64::*;
    use x86::*;
    let m00r = splat_re(m[0][0]);
    let m00i = splat_im(m[0][0]);
    let m01r = splat_re(m[0][1]);
    let m01i = splat_im(m[0][1]);
    let m10r = splat_re(m[1][0]);
    let m10i = splat_im(m[1][0]);
    let m11r = splat_re(m[1][1]);
    let m11i = splat_im(m[1][1]);
    let block = stride * 2;
    let n_groups = len / block;
    for g in 0..n_groups {
        let db = d.add(g * block);
        let sb = s.add(g * block);
        let mut k = 0usize;
        while k + 4 <= stride {
            let a0 = load2(sb.add(k));
            let a1 = load2(sb.add(k + 2));
            let b0 = load2(sb.add(k + stride));
            let b1 = load2(sb.add(k + stride + 2));
            let oa0 = _mm256_add_pd(cmul_nat(a0, m00r, m00i), cmul_nat(b0, m01r, m01i));
            let oa1 = _mm256_add_pd(cmul_nat(a1, m00r, m00i), cmul_nat(b1, m01r, m01i));
            let ob0 = _mm256_add_pd(cmul_nat(a0, m10r, m10i), cmul_nat(b0, m11r, m11i));
            let ob1 = _mm256_add_pd(cmul_nat(a1, m10r, m10i), cmul_nat(b1, m11r, m11i));
            store2_nt(db.add(k), oa0);
            store2_nt(db.add(k + 2), oa1);
            store2_nt(db.add(k + stride), ob0);
            store2_nt(db.add(k + stride + 2), ob1);
            k += 4;
        }
        while k + 2 <= stride {
            let a = load2(sb.add(k));
            let b = load2(sb.add(k + stride));
            let oa = _mm256_add_pd(cmul_nat(a, m00r, m00i), cmul_nat(b, m01r, m01i));
            let ob = _mm256_add_pd(cmul_nat(a, m10r, m10i), cmul_nat(b, m11r, m11i));
            store2_nt(db.add(k), oa);
            store2_nt(db.add(k + stride), ob);
            k += 2;
        }
    }
}

/// Streaming twin of `pair_pass_out_stride1_avx2`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn pair_pass_out_stride1_nt_avx2(
    d: *mut Complex64,
    s: *const Complex64,
    len: usize,
    m: [[Complex64; 2]; 2],
) {
    use std::arch::x86_64::*;
    use x86::*;
    let m00r = splat_re(m[0][0]);
    let m00i = splat_im(m[0][0]);
    let m01r = splat_re(m[0][1]);
    let m01i = splat_im(m[0][1]);
    let m10r = splat_re(m[1][0]);
    let m10i = splat_im(m[1][0]);
    let m11r = splat_re(m[1][1]);
    let m11i = splat_im(m[1][1]);
    let mut k = 0usize;
    while k + 4 <= len {
        let va = load2(s.add(k));
        let vb = load2(s.add(k + 2));
        let lo = _mm256_permute2f128_pd(va, vb, 0x20); // [c0, c2]
        let hi = _mm256_permute2f128_pd(va, vb, 0x31); // [c1, c3]
        let olo = _mm256_add_pd(cmul_nat(lo, m00r, m00i), cmul_nat(hi, m01r, m01i));
        let ohi = _mm256_add_pd(cmul_nat(lo, m10r, m10i), cmul_nat(hi, m11r, m11i));
        store2_nt(d.add(k), _mm256_permute2f128_pd(olo, ohi, 0x20));
        store2_nt(d.add(k + 2), _mm256_permute2f128_pd(olo, ohi, 0x31));
        k += 4;
    }
    while k + 2 <= len {
        let a = *s.add(k);
        let b = *s.add(k + 1);
        *d.add(k) = m[0][0] * a + m[0][1] * b;
        *d.add(k + 1) = m[1][0] * a + m[1][1] * b;
        k += 2;
    }
}

/// Streaming gather for the CNOT permutation: one aligned 32-byte store per
/// two amplitudes, source pair taken from `i ^ m2` when bit `q1` is set.
/// `NT` selects `vmovntpd` over the ordinary store; the arithmetic and the
/// addresses are identical either way, which is what makes the streaming
/// arm a one-variable comparison.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn cnot_pair_core<const NT: bool>(
    d: *mut Complex64,
    s: *const Complex64,
    len: usize,
    m1: usize,
    m2: usize,
) {
    use x86::*;
    let mut i = 0usize;
    while i + 2 <= len {
        let j = if (i & m1) != 0 { i ^ m2 } else { i };
        let v = load2(s.add(j));
        if NT {
            store2_nt(d.add(i), v);
        } else {
            store2(d.add(i), v);
        }
        i += 2;
    }
}

/// CNOT with the control on qubit 0 (`m1 == 1`): the two amplitudes of a pair
/// fall on opposite sides of the branch, so each pair combines its own even
/// element with its partner pair's odd element, and the partner the other way
/// round.  `m2 >= 2` keeps the partner 2-complex aligned.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn cnot_ctrl0_core<const NT: bool>(
    d: *mut Complex64,
    s: *const Complex64,
    len: usize,
    m2: usize,
) {
    use std::arch::x86_64::*;
    use x86::*;
    let mut i = 0usize;
    while i + 2 <= len {
        let j = i ^ m2;
        if i < j && j + 2 <= len {
            let own = load2(s.add(i));
            let part = load2(s.add(j));
            // doubles 2..4 (the odd complex) come from the partner
            let lo = _mm256_blend_pd(own, part, 0b1100);
            let hi = _mm256_blend_pd(part, own, 0b1100);
            if NT {
                store2_nt(d.add(i), lo);
                store2_nt(d.add(j), hi);
            } else {
                store2(d.add(i), lo);
                store2(d.add(j), hi);
            }
        }
        i += 2;
    }
}

/// Scalar fallback — same formula/order as the historical implementation.
fn pair_pass_scalar(s: &mut [Complex64], stride: usize, m: [[Complex64; 2]; 2]) {
    let block = stride * 2;
    for g in 0..(s.len() / block) {
        let base = g * block;
        for k in 0..stride {
            let lo = s[base + k];
            let hi = s[base + k + stride];
            s[base + k] = m[0][0] * lo + m[0][1] * hi;
            s[base + k + stride] = m[1][0] * lo + m[1][1] * hi;
        }
    }
}

/// Pairs at distance `stride >= 2` complex: contiguous runs of `stride`
/// complex, so both sides are vector loads.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn pair_pass_runs_avx2(
    p: *mut Complex64,
    len: usize,
    stride: usize,
    m: [[Complex64; 2]; 2],
) {
    use std::arch::x86_64::*;
    use x86::*;
    let m00 = splat_c(m[0][0]);
    let m01 = splat_c(m[0][1]);
    let m10 = splat_c(m[1][0]);
    let m11 = splat_c(m[1][1]);
    let block = stride * 2;
    let n_groups = len / block;
    for g in 0..n_groups {
        let base = p.add(g * block);
        let mut k = 0usize;
        while k + 2 <= stride {
            let a = load2(base.add(k));
            let b = load2(base.add(k + stride));
            let oa = _mm256_add_pd(cmul4(a, m00), cmul4(b, m01));
            let ob = _mm256_add_pd(cmul4(a, m10), cmul4(b, m11));
            store2(base.add(k), oa);
            store2(base.add(k + stride), ob);
            k += 2;
        }
    }
}

/// Pairs at distance 1: process 4 complex (2 pairs) per iteration with an
/// in-register deinterleave — `[c0 c1] [c2 c3]` → lo `[c0 c2]`, hi `[c1 c3]`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn pair_pass_stride1_avx2(p: *mut Complex64, len: usize, m: [[Complex64; 2]; 2]) {
    use std::arch::x86_64::*;
    use x86::*;
    let m00 = splat_c(m[0][0]);
    let m01 = splat_c(m[0][1]);
    let m10 = splat_c(m[1][0]);
    let m11 = splat_c(m[1][1]);
    let mut k = 0usize;
    while k + 4 <= len {
        let va = load2(p.add(k));
        let vb = load2(p.add(k + 2));
        let lo = _mm256_permute2f128_pd(va, vb, 0x20); // [c0, c2]
        let hi = _mm256_permute2f128_pd(va, vb, 0x31); // [c1, c3]
        let olo = _mm256_add_pd(cmul4(lo, m00), cmul4(hi, m01));
        let ohi = _mm256_add_pd(cmul4(lo, m10), cmul4(hi, m11));
        store2(p.add(k), _mm256_permute2f128_pd(olo, ohi, 0x20));
        store2(p.add(k + 2), _mm256_permute2f128_pd(olo, ohi, 0x31));
        k += 4;
    }
    while k + 2 <= len {
        let a = *p.add(k);
        let b = *p.add(k + 1);
        *p.add(k) = m[0][0] * a + m[0][1] * b;
        *p.add(k + 1) = m[1][0] * a + m[1][1] * b;
        k += 2;
    }
}

/// Natural-layout twin of `pair_pass_runs_avx2` (see the module docs): the
/// constants are broadcast once per call and each data register is shuffled
/// once per stage instead of three times per product. Same per-element
/// formula; even lanes are rounding-identical to `cmul4`, odd lanes commute
/// the FMA operands (<=1 ulp).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn pair_pass_runs_avx2_v2(
    p: *mut Complex64,
    len: usize,
    stride: usize,
    m: [[Complex64; 2]; 2],
) {
    use std::arch::x86_64::*;
    use x86::*;
    let m00r = splat_re(m[0][0]);
    let m00i = splat_im(m[0][0]);
    let m01r = splat_re(m[0][1]);
    let m01i = splat_im(m[0][1]);
    let m10r = splat_re(m[1][0]);
    let m10i = splat_im(m[1][0]);
    let m11r = splat_re(m[1][1]);
    let m11i = splat_im(m[1][1]);
    let block = stride * 2;
    let n_groups = len / block;
    for g in 0..n_groups {
        let base = p.add(g * block);
        let mut k = 0usize;
        // 4-complex unroll: the fused block buffers are L1-resident, so the
        // per-iteration overhead (loads, adds, loop bookkeeping) matters as
        // much as the FMA work.
        while k + 4 <= stride {
            let a0 = load2(base.add(k));
            let a1 = load2(base.add(k + 2));
            let b0 = load2(base.add(k + stride));
            let b1 = load2(base.add(k + stride + 2));
            let oa0 = _mm256_add_pd(cmul_nat(a0, m00r, m00i), cmul_nat(b0, m01r, m01i));
            let oa1 = _mm256_add_pd(cmul_nat(a1, m00r, m00i), cmul_nat(b1, m01r, m01i));
            let ob0 = _mm256_add_pd(cmul_nat(a0, m10r, m10i), cmul_nat(b0, m11r, m11i));
            let ob1 = _mm256_add_pd(cmul_nat(a1, m10r, m10i), cmul_nat(b1, m11r, m11i));
            store2(base.add(k), oa0);
            store2(base.add(k + 2), oa1);
            store2(base.add(k + stride), ob0);
            store2(base.add(k + stride + 2), ob1);
            k += 4;
        }
        while k + 2 <= stride {
            let a = load2(base.add(k));
            let b = load2(base.add(k + stride));
            let oa = _mm256_add_pd(cmul_nat(a, m00r, m00i), cmul_nat(b, m01r, m01i));
            let ob = _mm256_add_pd(cmul_nat(a, m10r, m10i), cmul_nat(b, m11r, m11i));
            store2(base.add(k), oa);
            store2(base.add(k + stride), ob);
            k += 2;
        }
    }
}
/// Natural-layout twin of `pair_pass_stride1_avx2`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn pair_pass_stride1_avx2_v2(p: *mut Complex64, len: usize, m: [[Complex64; 2]; 2]) {
    use std::arch::x86_64::*;
    use x86::*;
    let m00r = splat_re(m[0][0]);
    let m00i = splat_im(m[0][0]);
    let m01r = splat_re(m[0][1]);
    let m01i = splat_im(m[0][1]);
    let m10r = splat_re(m[1][0]);
    let m10i = splat_im(m[1][0]);
    let m11r = splat_re(m[1][1]);
    let m11i = splat_im(m[1][1]);
    let mut k = 0usize;
    while k + 4 <= len {
        let va = load2(p.add(k));
        let vb = load2(p.add(k + 2));
        let lo = _mm256_permute2f128_pd(va, vb, 0x20); // [c0, c2]
        let hi = _mm256_permute2f128_pd(va, vb, 0x31); // [c1, c3]
        let olo = _mm256_add_pd(cmul_nat(lo, m00r, m00i), cmul_nat(hi, m01r, m01i));
        let ohi = _mm256_add_pd(cmul_nat(lo, m10r, m10i), cmul_nat(hi, m11r, m11i));
        store2(p.add(k), _mm256_permute2f128_pd(olo, ohi, 0x20));
        store2(p.add(k + 2), _mm256_permute2f128_pd(olo, ohi, 0x31));
        k += 4;
    }
    while k + 2 <= len {
        let a = *p.add(k);
        let b = *p.add(k + 1);
        *p.add(k) = m[0][0] * a + m[0][1] * b;
        *p.add(k + 1) = m[1][0] * a + m[1][1] * b;
        k += 2;
    }
}

// ──────────────────────────────────────────────────────────────────────
// Diagonal passes: state[i] *= coeff(i). The coefficient is piecewise
// constant with a power-of-two period. When the varying bit positions are
// low (< 9), a period table (≤ 512 entries plus two wrap-around slack
// entries for full vector windows) is multiplied elementwise; when a bit is
// high, the pass walks aligned runs of 2^q where the coefficient is
// constant (broadcast multiply). `off` is the absolute index of `s[0]`, so
// slices may start at any alignment without changing the pattern phase.
// ──────────────────────────────────────────────────────────────────────

#[inline]
fn bit(i: usize, q: usize) -> usize {
    (i >> q) & 1
}

/// Build a period-`p` coefficient table with two wrap-around slack entries
/// so the vector kernel can always load a full 2-complex window.
fn make_table(p: usize, f: impl Fn(usize) -> Complex64) -> Vec<Complex64> {
    let mut t: Vec<Complex64> = (0..p).map(&f).collect();
    t.push(t[0]);
    t.push(t[1]);
    t
}

/// state[i] *= (bit q of i ? d1 : d0).
pub fn diag1_pass(s: &mut [Complex64], off: usize, q: usize, d0: Complex64, d1: Complex64) {
    let n = s.len();
    if q >= 9 {
        // The bit is constant on aligned runs of 2^q: broadcast per run.
        let bs = 1usize << q;
        let mut i = 0usize;
        while i < n {
            let a = off + i;
            let c = if bit(a, q) == 1 { d1 } else { d0 };
            let take = (bs - (a % bs)).min(n - i);
            pattern_scale_const(&mut s[i..i + take], c);
            i += take;
        }
    } else {
        let period = 1usize << (q + 1);
        let table = make_table(period, |j| if bit(j, q) == 1 { d1 } else { d0 });
        pattern_scale_periodic(s, off, &table);
    }
}

/// state[i] *= d[(bit(q1, i) << 1) | bit(q2, i)].
pub fn diag2_pass(s: &mut [Complex64], off: usize, q1: usize, q2: usize, d: &[Complex64; 4]) {
    let n = s.len();
    let (p1, p2) = if q1 < q2 { (q1, q2) } else { (q2, q1) };
    if p1 >= 9 {
        // Both bits are constant on aligned runs of 2^p1: broadcast per run.
        let bs = 1usize << p1;
        let mut i = 0usize;
        while i < n {
            let a = off + i;
            let c = d[(bit(a, q1) << 1) | bit(a, q2)];
            let take = (bs - (a % bs)).min(n - i);
            pattern_scale_const(&mut s[i..i + take], c);
            i += take;
        }
    } else if p2 >= 9 {
        // The high bit is constant on aligned runs of 2^p2; the low bit
        // keeps its period-2^(p1+1) table inside each run. Index bits follow
        // the original (q1, q2) order, so the varying bit goes into the slot
        // that `p1` occupies.
        let bs = 1usize << p2;
        let period = 1usize << (p1 + 1);
        let mk = |hi: usize| {
            make_table(period, |j| {
                let lo = bit(j, p1);
                let b1 = if q1 == p1 { lo } else { hi };
                let b2 = if q2 == p1 { lo } else { hi };
                d[(b1 << 1) | b2]
            })
        };
        let lo0 = mk(0);
        let lo1 = mk(1);
        let mut i = 0usize;
        while i < n {
            let a = off + i;
            let take = (bs - (a % bs)).min(n - i);
            let table = if bit(a, p2) == 1 { &lo1 } else { &lo0 };
            pattern_scale_periodic(&mut s[i..i + take], a, table);
            i += take;
        }
    } else {
        // Both bits are low: one period table covers the whole slice.
        let period = 1usize << (p2 + 1);
        let table = make_table(period, |j| d[(bit(j, q1) << 1) | bit(j, q2)]);
        pattern_scale_periodic(s, off, &table);
    }
}

/// Diagonal 3q multiply: `s[i] *= d[(bit(q0) << 2) | (bit(q1) << 1) | bit(q2)]`.
///
/// Same three regimes as `diag2_pass`, extended by one bit: broadcast when even
/// the lowest qubit sits at/above the 2^9 vector window, otherwise a period
/// table no longer than 2^10 with any higher qubit bit supplied by the run
/// walk (it is constant across such a run).
pub fn diag3_pass(s: &mut [Complex64], off: usize, q: [usize; 3], d: &[Complex64; 8]) {
    let n = s.len();
    let mut p = q;
    p.sort_unstable();
    let idx = |a: usize| (bit(a, q[0]) << 2) | (bit(a, q[1]) << 1) | bit(a, q[2]);
    if p[0] >= 9 {
        // Every bit is constant on aligned runs of 2^p0: broadcast per run.
        let bs = 1usize << p[0];
        let mut i = 0usize;
        while i < n {
            let a = off + i;
            let c = d[idx(a)];
            let take = (bs - (a % bs)).min(n - i);
            pattern_scale_const(&mut s[i..i + take], c);
            i += take;
        }
    } else if p[1] >= 9 || p[2] >= 9 {
        // One bit is run-constant; the remaining two fit a short period table.
        let (split, period) = if p[1] >= 9 {
            (p[1], 1usize << (p[0] + 1))
        } else {
            (p[2], 1usize << (p[1] + 1))
        };
        let bs = 1usize << split;
        let mask = period - 1;
        let mut i = 0usize;
        while i < n {
            let a = off + i;
            let take = (bs - (a % bs)).min(n - i);
            // `pattern_scale_periodic` indexes the table by the ABSOLUTE index
            // mod period, so every entry has to carry the bits this run holds
            // constant: only the qubit bits below `split` may vary with `j`.
            let hi = a & !mask;
            let table = make_table(period, |j| d[idx(hi | j)]);
            pattern_scale_periodic(&mut s[i..i + take], a, &table);
            i += take;
        }
    } else {
        // All three bits are low: one period table covers the whole slice.
        let period = 1usize << (p[2] + 1);
        let table = make_table(period, |j| d[idx(j)]);
        pattern_scale_periodic(s, off, &table);
    }
}

/// Multiply by a constant coefficient (vectorized broadcast).
fn pattern_scale_const(s: &mut [Complex64], c: Complex64) {
    if c == Complex64::new(1.0, 0.0) {
        return;
    }
    #[cfg(target_arch = "x86_64")]
    {
        if simd_enabled() && s.len() >= scale_simd_min() {
            unsafe { scale_const_avx2(s.as_mut_ptr(), s.len(), c) };
            return;
        }
    }
    for a in s.iter_mut() {
        *a = c * *a;
    }
}

/// Multiply by a periodic coefficient table (period = `table.len() - 2`;
/// the two trailing entries are wrap-around slack); `off` is the absolute
/// index of `s[0]`, so the phase carries across arbitrarily aligned slices.
fn pattern_scale_periodic(s: &mut [Complex64], off: usize, table: &[Complex64]) {
    let period = table.len() - 2;
    let ph0 = off & (period - 1);
    #[cfg(target_arch = "x86_64")]
    {
        if simd_enabled() && s.len() >= scale_simd_min() {
            unsafe {
                mul_by_table_phase_avx2(s.as_mut_ptr(), table.as_ptr(), s.len(), ph0, period)
            };
            return;
        }
    }
    for (i, a) in s.iter_mut().enumerate() {
        *a = table[(ph0 + i) & (period - 1)] * *a;
    }
}

/// `pattern_scale_periodic` without the small-size fallback: callers that
/// run the multiply once per group buffer (folded diagonal stages, ≤ 256
/// entries per call) amortize the dispatch cost across many groups, so each
/// buffer goes straight to the vector kernel when SIMD is enabled.
#[inline]
pub fn pattern_scale_chunk(s: &mut [Complex64], off: usize, table: &[Complex64]) {
    let period = table.len() - 2;
    let ph0 = off & (period - 1);
    #[cfg(target_arch = "x86_64")]
    {
        if simd_enabled() {
            unsafe {
                mul_by_table_phase_avx2(s.as_mut_ptr(), table.as_ptr(), s.len(), ph0, period)
            };
            return;
        }
    }
    for (i, a) in s.iter_mut().enumerate() {
        *a = table[(ph0 + i) & (period - 1)] * *a;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn scale_const_avx2(p: *mut Complex64, len: usize, c: Complex64) {
    use x86::*;
    let cv = splat_c(c);
    let mut i = 0usize;
    while i + 2 <= len {
        let v = load2(p.add(i));
        store2(p.add(i), cmul4(v, cv));
        i += 2;
    }
    while i < len {
        *p.add(i) = c * *p.add(i);
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn mul_by_table_phase_avx2(
    p: *mut Complex64,
    t: *const Complex64,
    len: usize,
    ph0: usize,
    period: usize,
) {
    use x86::*;
    let mask = period - 1;
    let mut i = 0usize;
    while i + 2 <= len {
        let v = load2(p.add(i));
        let w = load2(t.add((ph0 + i) & mask));
        store2(p.add(i), cmul4(v, w));
        i += 2;
    }
    while i < len {
        *p.add(i) = *t.add((ph0 + i) & mask) * *p.add(i);
        i += 1;
    }
}

// ──────────────────────────────────────────────────────────────────────
// Density-matrix kernels: ρ → UρU† fused into a single pass.
//
// The vectorized DM lives in the (ket, bra) index space
// (`i = ket | (bra << n)`), so a k-qubit unitary is two transforms on
// disjoint bit axes: U on the ket bits (q…), U* on the bra bits (n+q…).
// They commute, so one sweep can load each closed block once, apply both
// transforms in registers, and store once — half the memory traffic of two
// sequential passes, with per-element arithmetic order identical to the
// historical two-pass form (bit-compatible).
// ──────────────────────────────────────────────────────────────────────

/// Raw pointer handle for Rayon closures (same pattern as `dag.rs`).
struct DmPtr(*mut Complex64);
unsafe impl Send for DmPtr {}
unsafe impl Sync for DmPtr {}

#[inline(always)]
unsafe fn dm_get(p: &DmPtr, i: usize) -> Complex64 {
    *p.0.add(i)
}

#[inline(always)]
unsafe fn dm_set(p: &DmPtr, i: usize, v: Complex64) {
    *p.0.add(i) = v;
}

/// Insert a 0 bit at each ascending bit position in `positions`
/// (positions are in the original coordinate system).
#[inline(always)]
fn deposit_zeros(mut g: usize, positions: &[usize]) -> usize {
    for &p in positions {
        let mask = (1usize << p) - 1;
        g = (g & mask) | ((g & !mask) << 1);
    }
    g
}

/// 1-qubit unitary fused: ρ → (U⊗I)ρ(U†⊗I) in one pass over the closed
/// 4-cycles `{b, b|2^q, b|2^(n+q), b|2^q|2^(n+q)}` (ket bit q, bra bit n+q).
pub fn dm_1q_fused(s: &mut [Complex64], q: usize, u: [[Complex64; 2]; 2]) {
    let len = s.len();
    debug_assert!(len.is_power_of_two() && len >= 4);
    let n = len.trailing_zeros() as usize / 2;
    let nq = n + q;
    let mq = 1usize << q;
    let mnq = 1usize << nq;
    let u00 = u[0][0];
    let u01 = u[0][1];
    let u10 = u[1][0];
    let u11 = u[1][1];
    let c00 = u00.conj();
    let c01 = u01.conj();
    let c10 = u10.conj();
    let c11 = u11.conj();
    let positions = [q, nq];
    let n_groups = len >> 2;

    if len >= 1 << 14 && rayon::current_num_threads() > 1 {
        let sp = DmPtr(s.as_mut_ptr());
        (0..n_groups).into_par_iter().for_each(|g| unsafe {
            let base = deposit_zeros(g, &positions);
            let i00 = base;
            let i01 = base | mq;
            let i10 = base | mnq;
            let i11 = base | mq | mnq;
            let b00 = dm_get(&sp, i00);
            let b01 = dm_get(&sp, i01);
            let b10 = dm_get(&sp, i10);
            let b11 = dm_get(&sp, i11);
            // Ket transform first, then bra — the same arithmetic order
            // as two sequential passes.
            let t00 = u00 * b00 + u01 * b01;
            let t01 = u10 * b00 + u11 * b01;
            let t10 = u00 * b10 + u01 * b11;
            let t11 = u10 * b10 + u11 * b11;
            dm_set(&sp, i00, c00 * t00 + c01 * t10);
            dm_set(&sp, i10, c10 * t00 + c11 * t10);
            dm_set(&sp, i01, c00 * t01 + c01 * t11);
            dm_set(&sp, i11, c10 * t01 + c11 * t11);
        });
    } else {
        for g in 0..n_groups {
            let base = deposit_zeros(g, &positions);
            let i00 = base;
            let i01 = base | mq;
            let i10 = base | mnq;
            let i11 = base | mq | mnq;
            let b00 = s[i00];
            let b01 = s[i01];
            let b10 = s[i10];
            let b11 = s[i11];
            let t00 = u00 * b00 + u01 * b01;
            let t01 = u10 * b00 + u11 * b01;
            let t10 = u00 * b10 + u01 * b11;
            let t11 = u10 * b10 + u11 * b11;
            s[i00] = c00 * t00 + c01 * t10;
            s[i10] = c10 * t00 + c11 * t10;
            s[i01] = c00 * t01 + c01 * t11;
            s[i11] = c10 * t01 + c11 * t11;
        }
    }
}

/// 2-qubit unitary fused: one pass over the closed 16-blocks
/// (ket bits q0/q1 × bra bits n+q0/n+q1). Row/col convention identical
/// to `dag.rs::inplace_2q_general`: row = 2·bit(q0) + bit(q1).
pub fn dm_2q_fused(s: &mut [Complex64], q0: usize, q1: usize, g: &[[Complex64; 4]; 4]) {
    // AVX2 fast path: the separable ket(4x4)/bra(4x4) block transform, same
    // gather->register-math->scatter shape as `dm_super_2q_avx2`. The scalar
    // loops below stay as the non-x86 / `SF_NO_SIMD=1` reference and are
    // bit-identical to the AVX2 form (each output accumulates the same
    // products in the same ascending order).
    #[cfg(target_arch = "x86_64")]
    {
        if simd_enabled() {
            unsafe { dm_2q_fused_avx2(s, q0, q1, g) };
            return;
        }
    }
    let len = s.len();
    debug_assert!(len.is_power_of_two() && len >= 16);
    let n = len.trailing_zeros() as usize / 2;
    let mut c = [[Complex64::new(0.0, 0.0); 4]; 4];
    for r in 0..4 {
        for cc in 0..4 {
            c[r][cc] = g[r][cc].conj();
        }
    }
    let lo = q0.min(q1);
    let hi = q0.max(q1);
    let positions = [lo, hi, n + lo, n + hi];
    let mq0 = 1usize << q0;
    let mq1 = 1usize << q1;
    let mb0 = 1usize << (n + q0);
    let mb1 = 1usize << (n + q1);
    // Block index k ↔ bits (q0, q1) = (k>>1, k&1); b likewise on the bra bits.
    let ket_off = [0usize, mq1, mq0, mq0 | mq1];
    let bra_off = [0usize, mb1, mb0, mb0 | mb1];
    let n_groups = len >> 4;

    if len >= 1 << 14 && rayon::current_num_threads() > 1 {
        let sp = DmPtr(s.as_mut_ptr());
        (0..n_groups).into_par_iter().for_each(|gi| unsafe {
            let base = deposit_zeros(gi, &positions);
            let mut v = [[Complex64::new(0.0, 0.0); 4]; 4];
            for k in 0..4 {
                for b in 0..4 {
                    v[k][b] = dm_get(&sp, base | ket_off[k] | bra_off[b]);
                }
            }
            // Ket transform (rows/cols by 2·bit(q0)+bit(q1)).
            let mut t = [[Complex64::new(0.0, 0.0); 4]; 4];
            for k in 0..4 {
                for b in 0..4 {
                    t[k][b] = g[k][0] * v[0][b]
                        + g[k][1] * v[1][b]
                        + g[k][2] * v[2][b]
                        + g[k][3] * v[3][b];
                }
            }
            // Bra transform with conj(g).
            for k in 0..4 {
                for b in 0..4 {
                    dm_set(
                        &sp,
                        base | ket_off[k] | bra_off[b],
                        c[b][0] * t[k][0]
                            + c[b][1] * t[k][1]
                            + c[b][2] * t[k][2]
                            + c[b][3] * t[k][3],
                    );
                }
            }
        });
    } else {
        for gi in 0..n_groups {
            let base = deposit_zeros(gi, &positions);
            let mut v = [[Complex64::new(0.0, 0.0); 4]; 4];
            for k in 0..4 {
                for b in 0..4 {
                    v[k][b] = s[base | ket_off[k] | bra_off[b]];
                }
            }
            let mut t = [[Complex64::new(0.0, 0.0); 4]; 4];
            for k in 0..4 {
                for b in 0..4 {
                    t[k][b] = g[k][0] * v[0][b]
                        + g[k][1] * v[1][b]
                        + g[k][2] * v[2][b]
                        + g[k][3] * v[3][b];
                }
            }
            for k in 0..4 {
                for b in 0..4 {
                    s[base | ket_off[k] | bra_off[b]] = c[b][0] * t[k][0]
                        + c[b][1] * t[k][1]
                        + c[b][2] * t[k][2]
                        + c[b][3] * t[k][3];
                }
            }
        }
    }
}

/// AVX2 dispatch for the noiseless fused 2-qubit unitary: split the closed
/// 16-blocks across Rayon; each chunk runs the serial `#[target_feature]`
/// kernel (same split as `dm_super_2q_avx2`). Only called when
/// `simd_enabled()`.
#[cfg(target_arch = "x86_64")]
unsafe fn dm_2q_fused_avx2(s: &mut [Complex64], q0: usize, q1: usize, g: &[[Complex64; 4]; 4]) {
    let len = s.len();
    debug_assert!(len.is_power_of_two() && len >= 16);
    let n = len.trailing_zeros() as usize / 2;
    let lo = q0.min(q1);
    let hi = q0.max(q1);
    let positions = [lo, hi, n + lo, n + hi];
    let mq0 = 1usize << q0;
    let mq1 = 1usize << q1;
    let mb0 = 1usize << (n + q0);
    let mb1 = 1usize << (n + q1);
    let ket_off = [0usize, mq1, mq0, mq0 | mq1];
    let bra_off = [0usize, mb1, mb0, mb0 | mb1];
    let n_groups = len >> 4;
    // Conj(g) indexed [j][b] so the bra contraction's per-lane operand
    // (varying b at fixed j) loads a contiguous (b0,b1) pair.
    let mut ct = [[Complex64::new(0.0, 0.0); 4]; 4];
    for b in 0..4 {
        for j in 0..4 {
            ct[j][b] = g[b][j].conj();
        }
    }
    let sp = DmPtr(s.as_mut_ptr());
    if len >= 1 << 14 && rayon::current_num_threads() > 1 {
        let nchunks = rayon::current_num_threads() * 8;
        let per = n_groups.div_ceil(nchunks);
        (0..nchunks).into_par_iter().for_each(|c| {
            let gi0 = c * per;
            let gi1 = ((c + 1) * per).min(n_groups);
            if gi0 < gi1 {
                unsafe {
                    dm_2q_fused_avx2_chunk(&sp, g, &ct, &positions, &ket_off, &bra_off, gi0, gi1)
                };
            }
        });
    } else {
        dm_2q_fused_avx2_chunk(&sp, g, &ct, &positions, &ket_off, &bra_off, 0, n_groups);
    }
}

/// Serial AVX2 kernel: per closed 16-block gather the 16 amplitudes so each
/// ket row is a contiguous b-run, then run the separable ket transform
/// (t[k][:] = sum_j g[k][j]·v[j][:]) and bra transform
/// (out[k][b] = sum_j conj(g)[b][j]·t[k][j]) as row-pair accumulators
/// (`cmul4` = two complex FMAs), storing back. Each accumulator contracts
/// over j in ascending order, so every output is bit-identical to the scalar
/// row dot.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dm_2q_fused_avx2_chunk(
    sp: &DmPtr,
    g: &[[Complex64; 4]; 4],
    ct: &[[Complex64; 4]; 4],
    positions: &[usize; 4],
    ket_off: &[usize; 4],
    bra_off: &[usize; 4],
    gi0: usize,
    gi1: usize,
) {
    use std::arch::x86_64::*;
    use x86::{cmul4, load2, splat_c, store2};
    // Flat pointer into the contiguous [ [Complex64;4];4 ] conj(g) transpose.
    let ctp = ct.as_ptr() as *const Complex64;
    for gi in gi0..gi1 {
        let base = deposit_zeros(gi, positions);
        let mut v = [Complex64::new(0.0, 0.0); 16];
        for k in 0..4 {
            for b in 0..4 {
                v[4 * k + b] = dm_get(sp, base | ket_off[k] | bra_off[b]);
            }
        }
        // Ket stage: t[k][0..4] = sum_j g[k][j]·v[4j+0..4], two b-pair
        // accumulators (b0,b1) / (b2,b3) per row, j ascending.
        let mut t = [Complex64::new(0.0, 0.0); 16];
        for k in 0..4 {
            let mut alo = _mm256_setzero_pd();
            let mut ahi = _mm256_setzero_pd();
            for j in 0..4 {
                let gj = splat_c(g[k][j]);
                alo = _mm256_add_pd(alo, cmul4(gj, load2(v.as_ptr().add(4 * j))));
                ahi = _mm256_add_pd(ahi, cmul4(gj, load2(v.as_ptr().add(4 * j + 2))));
            }
            store2(t.as_mut_ptr().add(4 * k), alo);
            store2(t.as_mut_ptr().add(4 * k + 2), ahi);
        }
        // Bra stage: out[k][0..4] = sum_j ct[j][:]·t[4k+j], t[4k+j] broadcast
        // across the two b-lanes, j ascending. Result lands back in v.
        for k in 0..4 {
            let mut alo = _mm256_setzero_pd();
            let mut ahi = _mm256_setzero_pd();
            for j in 0..4 {
                let tj = splat_c(t[4 * k + j]);
                alo = _mm256_add_pd(alo, cmul4(tj, load2(ctp.add(4 * j))));
                ahi = _mm256_add_pd(ahi, cmul4(tj, load2(ctp.add(4 * j + 2))));
            }
            store2(v.as_mut_ptr().add(4 * k), alo);
            store2(v.as_mut_ptr().add(4 * k + 2), ahi);
        }
        for k in 0..4 {
            for b in 0..4 {
                dm_set(sp, base | ket_off[k] | bra_off[b], v[4 * k + b]);
            }
        }
    }
}

/// 1-qubit Kraus channel fused: out = Σ_k (K_k ⊗ K_k*)ρ via the pre-summed
/// 4×4 superoperator `m` (row = 2·ket + bra, matching the storage order
/// i = ket | (bra << n): position 1 = (ket0,bra1) = base|mnq,
/// position 2 = (ket1,bra0) = base|mq). Same closed-block layout and
/// ket-major convention as `dm_1q_fused`.
pub fn dm_super_1q(s: &mut [Complex64], q: usize, m: &[[Complex64; 4]; 4]) {
    let len = s.len();
    debug_assert!(len.is_power_of_two() && len >= 4);
    let n = len.trailing_zeros() as usize / 2;
    let nq = n + q;
    let mq = 1usize << q;
    let mnq = 1usize << nq;
    let positions = [q, nq];
    let n_groups = len >> 2;

    if len >= 1 << 14 && rayon::current_num_threads() > 1 {
        let sp = DmPtr(s.as_mut_ptr());
        (0..n_groups).into_par_iter().for_each(|g| unsafe {
            let base = deposit_zeros(g, &positions);
            let i00 = base;
            let i01 = base | mq;
            let i10 = base | mnq;
            let i11 = base | mq | mnq;
            let b00 = dm_get(&sp, i00);
            let b01 = dm_get(&sp, i10);
            let b10 = dm_get(&sp, i01);
            let b11 = dm_get(&sp, i11);
            dm_set(
                &sp,
                i00,
                m[0][0] * b00 + m[0][1] * b01 + m[0][2] * b10 + m[0][3] * b11,
            );
            dm_set(
                &sp,
                i10,
                m[1][0] * b00 + m[1][1] * b01 + m[1][2] * b10 + m[1][3] * b11,
            );
            dm_set(
                &sp,
                i01,
                m[2][0] * b00 + m[2][1] * b01 + m[2][2] * b10 + m[2][3] * b11,
            );
            dm_set(
                &sp,
                i11,
                m[3][0] * b00 + m[3][1] * b01 + m[3][2] * b10 + m[3][3] * b11,
            );
        });
    } else {
        for g in 0..n_groups {
            let base = deposit_zeros(g, &positions);
            let i00 = base;
            let i01 = base | mq;
            let i10 = base | mnq;
            let i11 = base | mq | mnq;
            let b00 = s[i00];
            let b01 = s[i10];
            let b10 = s[i01];
            let b11 = s[i11];
            s[i00] = m[0][0] * b00 + m[0][1] * b01 + m[0][2] * b10 + m[0][3] * b11;
            s[i10] = m[1][0] * b00 + m[1][1] * b01 + m[1][2] * b10 + m[1][3] * b11;
            s[i01] = m[2][0] * b00 + m[2][1] * b01 + m[2][2] * b10 + m[2][3] * b11;
            s[i11] = m[3][0] * b00 + m[3][1] * b01 + m[3][2] * b10 + m[3][3] * b11;
        }
    }
}

/// 2-qubit Kraus-channel superoperator fused: one pass over the closed
/// 16-blocks applying the pre-summed 16×16 superoperator `m`
/// (row = 4·ket + bra, ket/bra block bits (q0, q1) = (idx >> 1, idx & 1)).
/// Same closed-block layout as `dm_2q_fused`; used by the gate+noise
/// fusion so that U followed by touching 1q channels costs ONE sweep.
/// On AVX2+FMA hosts the 16×16 matvec runs as `cmul4` row-pairs over a
/// transposed copy of `m` (two complex lanes per FMA, eight independent
/// accumulators), which measured ~2.2× the scalar throughput — at the
/// same level as a plain memcpy of the buffer (software prefetching the
/// gathers *lost* ~0.6 ms/sweep and was removed on the same measurement).
pub fn dm_super_2q(s: &mut [Complex64], q0: usize, q1: usize, m: &[[Complex64; 16]; 16]) {
    #[cfg(target_arch = "x86_64")]
    {
        if simd_enabled() {
            unsafe { dm_super_2q_avx2(s, q0, q1, m) };
            return;
        }
    }
    let len = s.len();
    debug_assert!(len.is_power_of_two() && len >= 16);
    let n = len.trailing_zeros() as usize / 2;
    let lo = q0.min(q1);
    let hi = q0.max(q1);
    let positions = [lo, hi, n + lo, n + hi];
    let mq0 = 1usize << q0;
    let mq1 = 1usize << q1;
    let mb0 = 1usize << (n + q0);
    let mb1 = 1usize << (n + q1);
    let ket_off = [0usize, mq1, mq0, mq0 | mq1];
    let bra_off = [0usize, mb1, mb0, mb0 | mb1];
    let n_groups = len >> 4;

    if len >= 1 << 14 && rayon::current_num_threads() > 1 {
        let sp = DmPtr(s.as_mut_ptr());
        (0..n_groups).into_par_iter().for_each(|gi| unsafe {
            let base = deposit_zeros(gi, &positions);
            let mut v = [Complex64::new(0.0, 0.0); 16];
            for k in 0..4 {
                for b in 0..4 {
                    v[4 * k + b] = dm_get(&sp, base | ket_off[k] | bra_off[b]);
                }
            }
            for k in 0..4 {
                for b in 0..4 {
                    let row = 4 * k + b;
                    let mut acc = Complex64::new(0.0, 0.0);
                    for col in 0..16 {
                        acc += m[row][col] * v[col];
                    }
                    dm_set(&sp, base | ket_off[k] | bra_off[b], acc);
                }
            }
        });
    } else {
        for gi in 0..n_groups {
            let base = deposit_zeros(gi, &positions);
            let mut v = [Complex64::new(0.0, 0.0); 16];
            for k in 0..4 {
                for b in 0..4 {
                    v[4 * k + b] = s[base | ket_off[k] | bra_off[b]];
                }
            }
            for k in 0..4 {
                for b in 0..4 {
                    let row = 4 * k + b;
                    let mut acc = Complex64::new(0.0, 0.0);
                    for col in 0..16 {
                        acc += m[row][col] * v[col];
                    }
                    s[base | ket_off[k] | bra_off[b]] = acc;
                }
            }
        }
    }
}

/// AVX2 dispatch for `dm_super_2q`: builds the transposed superoperator
/// (so each row-pair loads two complex with one unaligned 32-byte load),
/// then splits the closed blocks across Rayon; every chunk runs the serial
/// `#[target_feature]` kernel (same serial-kernel + Rayon-chunking split as
/// the statevector kernels). Only called when `simd_enabled()`.
#[cfg(target_arch = "x86_64")]
unsafe fn dm_super_2q_avx2(s: &mut [Complex64], q0: usize, q1: usize, m: &[[Complex64; 16]; 16]) {
    let len = s.len();
    debug_assert!(len.is_power_of_two() && len >= 16);
    let n = len.trailing_zeros() as usize / 2;
    let lo = q0.min(q1);
    let hi = q0.max(q1);
    let positions = [lo, hi, n + lo, n + hi];
    let mq0 = 1usize << q0;
    let mq1 = 1usize << q1;
    let mb0 = 1usize << (n + q0);
    let mb1 = 1usize << (n + q1);
    let ket_off = [0usize, mq1, mq0, mq0 | mq1];
    let bra_off = [0usize, mb1, mb0, mb0 | mb1];
    let n_groups = len >> 4;
    let mut m_t = [[Complex64::new(0.0, 0.0); 16]; 16];
    for r in 0..16 {
        for c in 0..16 {
            m_t[c][r] = m[r][c];
        }
    }
    let sp = DmPtr(s.as_mut_ptr());
    if len >= 1 << 14 && rayon::current_num_threads() > 1 {
        let nchunks = rayon::current_num_threads() * 8;
        let per = n_groups.div_ceil(nchunks);
        (0..nchunks).into_par_iter().for_each(|c| {
            let gi0 = c * per;
            let gi1 = ((c + 1) * per).min(n_groups);
            if gi0 < gi1 {
                unsafe {
                    dm_super_2q_avx2_chunk(&sp, &m_t, &positions, &ket_off, &bra_off, gi0, gi1)
                };
            }
        });
    } else {
        dm_super_2q_avx2_chunk(&sp, &m_t, &positions, &ket_off, &bra_off, 0, n_groups);
    }
}

/// Serial AVX2 kernel: per closed 16-block, gather 16 amplitudes, apply
/// the transposed 16×16 superoperator as eight row-pair accumulators
/// (`cmul4` = two complex FMAs per instruction), store back. The row-pair
/// accumulators give eight independent dependency chains, which is what
/// removes the latency stall of the scalar row-by-row form.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dm_super_2q_avx2_chunk(
    sp: &DmPtr,
    m_t: &[[Complex64; 16]; 16],
    positions: &[usize; 4],
    ket_off: &[usize; 4],
    bra_off: &[usize; 4],
    gi0: usize,
    gi1: usize,
) {
    use std::arch::x86_64::*;
    for gi in gi0..gi1 {
        let base = deposit_zeros(gi, positions);
        let mut v = [Complex64::new(0.0, 0.0); 16];
        for k in 0..4 {
            for b in 0..4 {
                v[4 * k + b] = dm_get(sp, base | ket_off[k] | bra_off[b]);
            }
        }
        // acc[p] holds rows (2p, 2p+1) as one 2-complex vector.
        let mut acc = [_mm256_setzero_pd(); 8];
        for (col, vc) in v.iter().enumerate() {
            let vv = _mm256_set_pd(vc.im, vc.re, vc.im, vc.re);
            let mcol = m_t[col].as_ptr();
            for (p, a) in acc.iter_mut().enumerate() {
                let mv = _mm256_loadu_pd(mcol.add(2 * p) as *const f64);
                let vr = _mm256_movedup_pd(mv);
                let vi = _mm256_permute_pd(mv, 0b1111);
                let vs = _mm256_permute_pd(vv, 0b0101);
                let prod = _mm256_fmaddsub_pd(vr, vv, _mm256_mul_pd(vi, vs));
                *a = _mm256_add_pd(*a, prod);
            }
        }
        for (p, a) in acc.iter().enumerate() {
            _mm256_storeu_pd(v.as_mut_ptr().add(2 * p) as *mut f64, *a);
        }
        for k in 0..4 {
            for b in 0..4 {
                dm_set(sp, base | ket_off[k] | bra_off[b], v[4 * k + b]);
            }
        }
    }
}

/// Two 1-qubit channel superoperators in ONE sweep: each closed 16-block
/// of the joint (qa, qb) qubit pair is loaded once and both 4×4 superops
/// are applied in registers before the single store, halving the memory
/// traffic of two sequential `dm_super_1q` sweeps. The channels act on
/// disjoint bit axes, so the per-element arithmetic order is identical to
/// the sequential form (verified bit-exact) and the result is unchanged.
pub fn dm_super_1q_pair(
    s: &mut [Complex64],
    qa: usize,
    m_a: &[[Complex64; 4]; 4],
    qb: usize,
    m_b: &[[Complex64; 4]; 4],
) {
    let len = s.len();
    debug_assert!(len.is_power_of_two() && len >= 16);
    debug_assert_ne!(qa, qb);
    let n = len.trailing_zeros() as usize / 2;
    let lo = qa.min(qb);
    let hi = qa.max(qb);
    let positions = [lo, hi, n + lo, n + hi];
    // Block index layout (k = 2·bit(lo) + bit(hi)), same as `dm_2q_fused`.
    let k_lo = 1usize << lo;
    let k_hi = 1usize << hi;
    let b_lo = 1usize << (n + lo);
    let b_hi = 1usize << (n + hi);
    let ket_off = [0usize, k_hi, k_lo, k_lo | k_hi];
    let bra_off = [0usize, b_hi, b_lo, b_lo | b_hi];
    let n_groups = len >> 4;
    // Which axis (0 = lo bit, 1 = hi bit) each qubit occupies.
    let (axis_a, axis_b) = if qa == lo {
        (0usize, 1usize)
    } else {
        (1usize, 0usize)
    };

    if len >= 1 << 14 && rayon::current_num_threads() > 1 {
        let sp = DmPtr(s.as_mut_ptr());
        (0..n_groups).into_par_iter().for_each(|gi| unsafe {
            let base = deposit_zeros(gi, &positions);
            let mut v = [Complex64::new(0.0, 0.0); 16];
            for k in 0..4 {
                for b in 0..4 {
                    v[4 * k + b] = dm_get(&sp, base | ket_off[k] | bra_off[b]);
                }
            }
            apply_1q_axis(&mut v, axis_a, m_a);
            apply_1q_axis(&mut v, axis_b, m_b);
            for k in 0..4 {
                for b in 0..4 {
                    dm_set(&sp, base | ket_off[k] | bra_off[b], v[4 * k + b]);
                }
            }
        });
    } else {
        for gi in 0..n_groups {
            let base = deposit_zeros(gi, &positions);
            let mut v = [Complex64::new(0.0, 0.0); 16];
            for k in 0..4 {
                for b in 0..4 {
                    v[4 * k + b] = s[base | ket_off[k] | bra_off[b]];
                }
            }
            apply_1q_axis(&mut v, axis_a, m_a);
            apply_1q_axis(&mut v, axis_b, m_b);
            for k in 0..4 {
                for b in 0..4 {
                    s[base | ket_off[k] | bra_off[b]] = v[4 * k + b];
                }
            }
        }
    }
}

/// Apply the 4×4 superop `m` to the sub-vector varying qubit `axis`
/// (0 = lo bit, 1 = hi bit) of the 16-block `v`, for each fixed
/// combination of the sibling qubit's coordinates. Expression order
/// mirrors `dm_super_1q` exactly so the fused result is bit-identical to
/// the two-sweep path.
#[inline]
fn apply_1q_axis(v: &mut [Complex64; 16], axis: usize, m: &[[Complex64; 4]; 4]) {
    for other in 0..4 {
        // Sibling coordinates: 0b(k2, b2).
        let (k2, b2) = if axis == 0 {
            (other >> 1, other & 1)
        } else {
            (other & 1, other >> 1)
        };
        // Flat index in the 16-block for (ka, ba) given the sibling.
        let pos = |ka: usize, ba: usize| -> usize {
            let k = if axis == 0 { 2 * ka + k2 } else { 2 * k2 + ka };
            let b = if axis == 0 { 2 * ba + b2 } else { 2 * b2 + ba };
            4 * k + b
        };
        let i00 = pos(0, 0);
        let i01 = pos(1, 0); // ket bit set, bra clear
        let i10 = pos(0, 1); // ket clear, bra set
        let i11 = pos(1, 1);
        let b00 = v[i00];
        let b01 = v[i10];
        let b10 = v[i01];
        let b11 = v[i11];
        v[i00] = m[0][0] * b00 + m[0][1] * b01 + m[0][2] * b10 + m[0][3] * b11;
        v[i10] = m[1][0] * b00 + m[1][1] * b01 + m[1][2] * b10 + m[1][3] * b11;
        v[i01] = m[2][0] * b00 + m[2][1] * b01 + m[2][2] * b10 + m[2][3] * b11;
        v[i11] = m[3][0] * b00 + m[3][1] * b01 + m[3][2] * b10 + m[3][3] * b11;
    }
}

// ──────────────────────────────────────────────────────────────────────
// Contiguous run swap (CNOT / X blocks): swap `run` complex starting at
// `a` with `run` complex starting at `b`.
// ──────────────────────────────────────────────────────────────────────

/// # Safety
/// `p.add(a)` and `p.add(b)` must each be valid for `run` consecutive
/// `Complex64` elements in the same allocation, and the two runs must not
/// overlap.
#[inline]
pub unsafe fn swap_runs(p: *mut Complex64, a: usize, b: usize, run: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        if simd_enabled() && run >= 2 {
            unsafe { swap_runs_avx2(p, a, b, run) };
            return;
        }
    }
    unsafe {
        std::ptr::swap_nonoverlapping(p.add(a), p.add(b), run);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn swap_runs_avx2(p: *mut Complex64, a: usize, b: usize, run: usize) {
    use x86::*;
    let mut k = 0usize;
    while k + 2 <= run {
        let x = load2(p.add(a + k));
        let y = load2(p.add(b + k));
        store2(p.add(a + k), y);
        store2(p.add(b + k), x);
        k += 2;
    }
    if k < run {
        let x = *p.add(a + k);
        *p.add(a + k) = *p.add(b + k);
        *p.add(b + k) = x;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64 as C;

    fn lcg(n: usize, seed: u64) -> Vec<C> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let re = ((s >> 11) as f64 / (1u64 << 53) as f64) - 0.5;
                let im = ((s >> 33) as f64 / (1u64 << 31) as f64) - 0.5;
                C::new(re, im)
            })
            .collect()
    }

    /// Physical gate for the restructured (two streaming pair passes) 1q dm
    /// kernel: for a pure state `|psi>`, the fused `rho -> (U tensor I) rho
    /// (U^dagger tensor I)` must reproduce `|psi'> <psi'|` where `|psi'>` is the
    /// statevector after applying `U` to qubit `q`. Independent of the kernel's
    /// own arithmetic, so it validates the ket(U)/bra(conj U) strides, not just
    /// self-consistency. Layout: `data[ket | (bra << n)] = rho[ket, bra]`.
    #[test]
    fn dm_1q_fused_matches_outer_of_evolved_state() {
        for n in [1usize, 2, 3, 4] {
            let dim = 1usize << n;
            let psi = lcg(dim, 3 + n as u64);
            // Evolve psi with U on qubit q via an explicit statevector pair pass.
            for q in 0..n {
                let u = [
                    [C::new(0.6, 0.1), C::new(-0.3, 0.5)],
                    [C::new(0.4, -0.2), C::new(0.5, 0.3)],
                ];
                let mut psi_p = psi.clone();
                let s = 1usize << q;
                for base in (0..dim).step_by(s * 2) {
                    for k in 0..s {
                        let lo = base + k;
                        let hi = lo + s;
                        let (a, b) = (psi_p[lo], psi_p[hi]);
                        psi_p[lo] = u[0][0] * a + u[0][1] * b;
                        psi_p[hi] = u[1][0] * a + u[1][1] * b;
                    }
                }
                // rho = |psi><psi| in the (ket | bra<<n) layout, then evolve.
                let mut rho = vec![C::new(0.0, 0.0); dim * dim];
                for ket in 0..dim {
                    for bra in 0..dim {
                        rho[ket | (bra << n)] = psi[ket] * psi[bra].conj();
                    }
                }
                dm_1q_fused(&mut rho, q, u);
                let mut worst = 0.0f64;
                for ket in 0..dim {
                    for bra in 0..dim {
                        let want = psi_p[ket] * psi_p[bra].conj();
                        let got = rho[ket | (bra << n)];
                        worst = worst.max((got - want).norm());
                    }
                }
                assert!(
                    worst < 1e-12,
                    "dm_1q_fused rho-vs-outer n={n} q={q}: {worst}"
                );
            }
        }
    }

    /// `diag3_pass` against a naive per-entry loop, over the three regimes the
    /// kernel distinguishes: all three bits low (one period table), one bit at
    /// or above the 2^9 vector window (run-constant bit + short table), and
    /// every bit high (pure broadcast). `off != 0` covers the parallel chunks.
    #[test]
    fn diag3_pass_matches_naive() {
        let n = 12usize;
        let dim = 1usize << n;
        let d: [C; 8] = [
            C::new(1.0, 0.0),
            C::new(0.6, -0.8),
            C::new(-0.28, 0.96),
            C::new(0.0, 1.0),
            C::new(0.8, 0.6),
            C::new(-0.6, 0.8),
            C::new(0.0, -1.0),
            C::new(0.96, 0.28),
        ];
        for &q in &[
            [0usize, 1, 2],
            [2, 0, 5],
            [7, 3, 11],
            [0, 9, 10],
            [9, 10, 11],
            [4, 8, 2],
        ] {
            let base = lcg(dim, 77 + q[0] as u64);
            let mut want = base.clone();
            for a in 0..dim {
                let i = (bit(a, q[0]) << 2) | (bit(a, q[1]) << 1) | bit(a, q[2]);
                want[a] *= d[i];
            }
            // Whole slice, and the same slice as two halves with a nonzero
            // offset (the parallel-chunk path).
            let mut got = base.clone();
            diag3_pass(&mut got, 0, q, &d);
            for (a, (x, y)) in want.iter().zip(got.iter()).enumerate() {
                assert!((x - y).norm() < 1e-12, "q={q:?} whole a={a}: {x} vs {y}");
            }
            let half = dim / 2;
            let mut parts = base.clone();
            let (lo, hi) = parts.split_at_mut(half);
            diag3_pass(lo, 0, q, &d);
            diag3_pass(hi, half, q, &d);
            for (a, (x, y)) in want.iter().zip(parts.iter()).enumerate() {
                assert!((x - y).norm() < 1e-12, "q={q:?} chunked a={a}: {x} vs {y}");
            }
            // Odd chunks: 3 slices of unequal length.
            let mut parts = base.clone();
            let c1 = dim / 3;
            let c2 = 2 * dim / 3;
            let (s0, rest) = parts.split_at_mut(c1);
            let (s1, s2) = rest.split_at_mut(c2 - c1);
            diag3_pass(s0, 0, q, &d);
            diag3_pass(s1, c1, q, &d);
            diag3_pass(s2, c2, q, &d);
            for (a, (x, y)) in want.iter().zip(parts.iter()).enumerate() {
                assert!((x - y).norm() < 1e-12, "q={q:?} thirds a={a}: {x} vs {y}");
            }
        }
    }

    #[test]
    fn pair_pass_matches_scalar() {
        for stride in [1usize, 2, 4, 8] {
            let m = [
                [C::new(0.3, 0.7), C::new(-0.5, 0.1)],
                [C::new(0.2, -0.9), C::new(0.4, 0.6)],
            ];
            let mut a = lcg(128, stride as u64 + 1);
            let mut b = a.clone();
            pair_pass(&mut a, stride, m);
            pair_pass_scalar(&mut b, stride, m);
            for (x, y) in a.iter().zip(b.iter()) {
                assert!((x - y).norm() < 1e-12, "stride {stride}: {x} vs {y}");
            }
        }
    }

    /// `pair_pass_out` must reproduce the in-place kernel pointwise, for every
    /// stride in the state and at sizes on both sides of the SIMD floor.  The
    /// destination starts as NaN so a chunk the kernel skips (a stride that is
    /// not a block multiple, an index outside its own chunk) shows up as a
    /// value that is still NaN rather than as a plausible-looking number.
    #[test]
    fn pair_pass_out_matches_in_place() {
        let m = [
            [C::new(0.3, 0.7), C::new(-0.5, 0.1)],
            [C::new(0.2, -0.9), C::new(0.4, 0.6)],
        ];
        let nan = C::new(f64::NAN, f64::NAN);
        for n in [10usize, 13, 15] {
            let len = 1usize << n;
            let src = lcg(len, 99 + n as u64);
            for q in 0..n {
                let stride = 1usize << q;
                let mut want = src.clone();
                pair_pass(&mut want, stride, m);
                let mut got = vec![nan; len];
                pair_pass_out(&mut got, &src, stride, m);
                for (x, y) in got.iter().zip(want.iter()) {
                    assert!(!x.re.is_nan(), "n={n} q={q}: destination skipped");
                    assert!(
                        (x - y).norm() < 1e-12,
                        "n={n} q={q} stride={stride}: {x} vs {y}"
                    );
                }
            }
        }
    }

    /// The streaming kernels must reproduce the in-place result pointwise, and
    /// must refuse a destination that is not 32-byte aligned *without writing
    /// it*.  The destination is taken as the tail of an over-allocated buffer
    /// so its base is aligned on purpose — the alignment the cache offset in
    /// `adjoint` arranges for.  Sizes are above `pair_simd_min` because the
    /// streaming path has no scalar fallback of its own.
    #[test]
    fn pair_pass_out_nt_matches_in_place() {
        if !avx2_ok() {
            return;
        }
        let m = [
            [C::new(0.3, 0.7), C::new(-0.5, 0.1)],
            [C::new(0.2, -0.9), C::new(0.4, 0.6)],
        ];
        let nan = C::new(f64::NAN, f64::NAN);
        for n in [12usize, 13, 15] {
            let len = 1usize << n;
            let src = lcg(len, 31 + n as u64);
            let mut buf = vec![nan; len + 2];
            let off = if (buf.as_ptr() as usize) & 31 == 0 {
                0
            } else {
                1
            };
            for q in 0..n {
                let stride = 1usize << q;
                let mut want = src.clone();
                pair_pass(&mut want, stride, m);
                for x in buf[off..off + len].iter_mut() {
                    *x = nan;
                }
                let (_, tail) = buf.split_at_mut(off);
                assert!(
                    pair_pass_out_nt(&mut tail[..len], &src, stride, m),
                    "n={n} q={q}: streaming path refused an aligned destination"
                );
                for (x, y) in buf[off..off + len].iter().zip(want.iter()) {
                    assert!(!x.re.is_nan(), "n={n} q={q}: destination skipped");
                    assert!((x - y).norm() < 1e-12, "n={n} q={q}: {x} vs {y}");
                }
            }
            // A misaligned base is refused before a single element is written.
            let bad = if off == 0 { 1 } else { 0 };
            for x in buf[bad..bad + len].iter_mut() {
                *x = nan;
            }
            assert!(!pair_pass_out_nt(&mut buf[bad..bad + len], &src, 2, m));
            assert!(buf[bad..bad + len].iter().all(|x| x.re.is_nan()));
        }
    }

    /// Vectorised CNOT dst/src: the same permutation as the scalar gather for
    /// every (control, target) pair of a 6-qubit register — including a control
    /// on qubit 0, where a pair's two amplitudes take different branches — in
    /// both store modes, and a refusal (not a wrong answer) for the
    /// pair-reversing `m2 == 1` case.
    #[test]
    fn cnot_out_vec_matches_scalar() {
        if !avx2_ok() {
            return;
        }
        let n = 12usize;
        let len = 1usize << n;
        let src = lcg(len, 5);
        let nan = C::new(f64::NAN, f64::NAN);
        let mut buf = vec![nan; len + 2];
        let off = if (buf.as_ptr() as usize) & 31 == 0 {
            0
        } else {
            1
        };
        for &nt in &[false, true] {
            for q1 in 0..6usize {
                for q2 in 0..6usize {
                    if q1 == q2 {
                        continue;
                    }
                    let m1 = 1usize << q1;
                    let m2 = 1usize << q2;
                    for x in buf[off..off + len].iter_mut() {
                        *x = nan;
                    }
                    let (_, tail) = buf.split_at_mut(off);
                    if !cnot_out_vec(&mut tail[..len], &src, q1, q2, nt) {
                        assert!(
                            m2 < 2,
                            "nt={nt} q1={q1} q2={q2}: refused an aligned destination"
                        );
                        assert!(buf[off..off + len].iter().all(|x| x.re.is_nan()));
                        continue;
                    }
                    for (i, x) in buf[off..off + len].iter().enumerate() {
                        let j = if (i & m1) != 0 { i ^ m2 } else { i };
                        assert!((x - src[j]).norm() < 1e-15, "nt={nt} q1={q1} q2={q2} i={i}");
                    }
                }
            }
        }
    }

    /// A/B the legacy (`cmul4`) and natural-layout (`cmul_nat`) runs
    /// kernels: pointwise agreement plus interleaved min-of-5 timing on a
    /// cache-busting buffer. Run with `--nocapture` for the numbers.
    #[test]
    fn pair_v2_microbench() {
        use std::time::Instant;
        if !avx2_ok() {
            return;
        }
        let n = 1usize << 22;
        let m = [
            [C::new(0.3, 0.7), C::new(-0.5, 0.1)],
            [C::new(0.2, -0.9), C::new(0.4, 0.6)],
        ];
        let base = lcg(n, 4242);
        for stride in [2usize, 8, 32] {
            let mut a = base.clone();
            let mut b = base.clone();
            let mut t_leg = f64::MAX;
            let mut t_new = f64::MAX;
            for _ in 0..12 {
                let t0 = Instant::now();
                unsafe { pair_pass_runs_avx2(b.as_mut_ptr(), n, stride, m) };
                t_leg = t_leg.min(t0.elapsed().as_secs_f64());
                let t0 = Instant::now();
                unsafe { pair_pass_runs_avx2_v2(a.as_mut_ptr(), n, stride, m) };
                t_new = t_new.min(t0.elapsed().as_secs_f64());
            }
            let mut worst = 0.0f64;
            for (x, y) in a.iter().zip(b.iter()) {
                worst = worst.max((x - y).norm() / x.norm().max(1e-3));
            }
            println!(
                "stride {stride:2}: legacy {:7.2} ms  v2 {:7.2} ms  ({:.2}x)  max_rel_diff {worst:.2e}",
                t_leg * 1e3,
                t_new * 1e3,
                t_leg / t_new
            );
            assert!(worst < 1e-12, "v2 deviates from legacy: {worst}");
        }
        // 64-element chunks (the fused per-group call shape), norm-preserving
        // matrix so repeated application does not overflow.
        let c = C::new(0.6 / 2f64.sqrt(), 0.8 / 2f64.sqrt());
        let s = C::new(0.8 / 2f64.sqrt(), -0.6 / 2f64.sqrt());
        let mu = [[c, -s], [s, c]];
        let base64 = lcg(64, 7);
        let mut a = base64.clone();
        let mut b = base64.clone();
        // Batched, interleaved rounds: a single 20k-rep block is ~10 ms on
        // this host, small enough that scheduler noise dominated a one-shot
        // timing; min-of-10 over alternating blocks is stable.
        const REPS: usize = 20_000;
        let mut t_leg = f64::MAX;
        let mut t_new = f64::MAX;
        for _ in 0..10 {
            let t0 = Instant::now();
            for _ in 0..REPS {
                unsafe { pair_pass_runs_avx2(b.as_mut_ptr(), 64, 4, mu) };
            }
            t_leg = t_leg.min(t0.elapsed().as_secs_f64());
            let t0 = Instant::now();
            for _ in 0..REPS {
                unsafe { pair_pass_runs_avx2_v2(a.as_mut_ptr(), 64, 4, mu) };
            }
            t_new = t_new.min(t0.elapsed().as_secs_f64());
        }
        let mut worst = 0.0f64;
        for (x, y) in a.iter().zip(b.iter()) {
            worst = worst.max((x - y).norm());
        }
        println!(
            "64-el chunk x{REPS}: legacy {:7.2} ms  v2 {:7.2} ms  ({:.2}x)  max_abs_diff {worst:.2e}",
            t_leg * 1e3,
            t_new * 1e3,
            t_leg / t_new
        );
        assert!(worst < 1e-12);
    }

    #[test]
    fn diag1_matches_scalar() {
        let d0 = C::new(0.2, 0.9);
        let d1 = C::new(-0.7, 0.3);
        // Low bits use the period table; q >= 9 uses the constant-run walk;
        // non-zero offsets exercise the phase carry (unaligned slices).
        for q in [0usize, 1, 2, 5, 9, 12] {
            for off in [0usize, 3, 512] {
                let mut a = lcg(1600, (q * 4 + off) as u64 + 7);
                let mut b = a.clone();
                diag1_pass(&mut a[off..off + 1024], off, q, d0, d1);
                for (i, x) in b[off..off + 1024].iter_mut().enumerate() {
                    let c = if ((off + i) >> q) & 1 == 1 { d1 } else { d0 };
                    *x = c * *x;
                }
                for (x, y) in a.iter().zip(b.iter()) {
                    assert!((x - y).norm() < 1e-12, "q={q} off={off}");
                }
            }
        }
    }

    #[test]
    fn diag2_matches_scalar() {
        let d = [
            C::new(0.3, 0.4),
            C::new(-0.2, 0.8),
            C::new(0.9, -0.1),
            C::new(0.5, 0.5),
        ];
        // Mixed high/low bits cover all three branches of the walk logic.
        for (q1, q2) in [
            (0usize, 1usize),
            (1, 4),
            (3, 8),
            (2, 3),
            (5, 2),
            (10, 2),
            (1, 10),
            (11, 12),
        ] {
            for off in [0usize, 5, 512] {
                let mut a = lcg(1600, (q1 * 8 + q2 + off) as u64 + 3);
                let mut b = a.clone();
                diag2_pass(&mut a[off..off + 1024], off, q1, q2, &d);
                for (i, x) in b[off..off + 1024].iter_mut().enumerate() {
                    let j = off + i;
                    let c = d[(((j >> q1) & 1) << 1) | ((j >> q2) & 1)];
                    *x = c * *x;
                }
                for (x, y) in a.iter().zip(b.iter()) {
                    assert!((x - y).norm() < 1e-12, "q1={q1} q2={q2} off={off}");
                }
            }
        }
    }

    #[test]
    fn swap_runs_roundtrip() {
        let mut a = lcg(64, 11);
        let orig = a.clone();
        unsafe { swap_runs(a.as_mut_ptr(), 4, 36, 8) };
        unsafe { swap_runs(a.as_mut_ptr(), 4, 36, 8) };
        for (x, y) in a.iter().zip(orig.iter()) {
            assert!((x - y).norm() < 1e-15);
        }
        unsafe { swap_runs(a.as_mut_ptr(), 0, 8, 4) };
        for k in 0..4 {
            assert!((a[k] - orig[k + 8]).norm() < 1e-15);
            assert!((a[k + 8] - orig[k]).norm() < 1e-15);
        }
    }
}
