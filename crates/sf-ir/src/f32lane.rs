//! Opt-in f32 (complex64) statevector lane.
//!
//! The lane consumes the SAME instruction plan as the f64 simulator
//! (`QuantumDAG::build_sim_plan`) — gate fusion, CX·D·CX rewriting, diagonal
//! phase sweeps and permutation merges are all decided once on the f64
//! matrices — and runs f32 twins of every apply kernel over a `Complex32`
//! state (half the bytes per amplitude of the f64 lane). Because every plan
//! *decision* is made on the f64 matrices, both lanes execute an identical
//! instruction stream; only the arithmetic precision differs.
//!
//! Accuracy class: f32 arithmetic accumulates ~1e-6..1e-5 relative error
//! over typical circuits (the same class as qsim's default f32 backend).
//! The lane is strictly opt-in (`SF_F32=1` read at the binding boundary, or
//! a direct `QuantumDAG::simulate_f32` call); the f64 path stays the
//! default everywhere.
//!
//! Kernels mirror `crate::simd` with AVX2+FMA vectorization where the
//! interleaved-complex layout allows (four complexes per `__m256`), the
//! scalar fallbacks are formula-identical, and `SF_NO_SIMD=1` forces the
//! scalar paths.

use crate::dag::{
    is_diag_1q, is_diag_2q, perm_gather_config, perm_gather_dispatch, phase_eval, Diag2Stage,
    PermPlan, PhasePlan, QuantumDAG, QuantumOp, SimInst,
};
use crate::ops::OpType;
use num_complex::Complex32;
use rayon::prelude::*;

/// `SF_F32=1` latch (read once per process, like the SIMD escape hatch).
pub fn lane_enabled() -> bool {
    use std::sync::atomic::{AtomicU8, Ordering};
    static STATE: AtomicU8 = AtomicU8::new(0);
    match STATE.load(Ordering::Relaxed) {
        1 => true,
        2 => false,
        _ => {
            let on = std::env::var("SF_F32").map(|v| v == "1").unwrap_or(false);
            STATE.store(if on { 1 } else { 2 }, Ordering::Relaxed);
            on
        }
    }
}

/// Parallel dispatch threshold — same empirical calibration as the f64
/// simulator (Rayon overhead exceeds the benefit below this amplitude count).
const PARALLEL_THRESHOLD: usize = 1 << 16;

#[inline(always)]
fn c32(z: num_complex::Complex64) -> Complex32 {
    Complex32::new(z.re as f32, z.im as f32)
}

#[inline(always)]
fn bit(i: usize, q: usize) -> usize {
    (i >> q) & 1
}

/// Raw state pointer usable from Rayon closures (same pattern as the f64
/// lane's `SendPtr`; disjointness is guaranteed by the kernel structure).
struct SendPtr32(*mut Complex32);
unsafe impl Send for SendPtr32 {}
unsafe impl Sync for SendPtr32 {}

impl SendPtr32 {
    #[inline(always)]
    unsafe fn get(&self, idx: usize) -> Complex32 {
        *self.0.add(idx)
    }
    #[inline(always)]
    unsafe fn set(&self, idx: usize, val: Complex32) {
        *self.0.add(idx) = val;
    }
    /// Raw base pointer (kernel/prefetch use; address only).
    #[inline(always)]
    fn raw(&self) -> *mut Complex32 {
        self.0
    }
    #[inline(always)]
    unsafe fn swap(&self, a: usize, b: usize) {
        let tmp = *self.0.add(a);
        *self.0.add(a) = *self.0.add(b);
        *self.0.add(b) = tmp;
    }
    /// Swap two disjoint contiguous runs of `count` elements (runs must not
    /// overlap; callers guarantee this by construction).
    #[inline(always)]
    unsafe fn swap_run(&self, a: usize, b: usize, count: usize) {
        swap_runs(self.0, a, b, count);
    }
}

// ──────────────────────────────────────────────────────────────────────
// x86_64 AVX2+FMA building blocks (f32)
// ──────────────────────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
mod x86 {
    use super::Complex32;
    use std::arch::x86_64::*;

    /// Componentwise complex multiply of two interleaved-complex vectors
    /// (each `__m256` = 4 complexes); one operand is normally a broadcast
    /// scalar. Same moveldup/movehdup/fmaddsub pattern as the f64 kernel.
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn cmul8(v: __m256, w: __m256) -> __m256 {
        let vr = _mm256_moveldup_ps(v);
        let vi = _mm256_movehdup_ps(v);
        let ws = _mm256_permute_ps(w, 0b10110001);
        _mm256_fmaddsub_ps(vr, w, _mm256_mul_ps(vi, ws))
    }

    /// 128-bit variant: 2 complexes.
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn cmul4(v: __m128, w: __m128) -> __m128 {
        let vr = _mm_moveldup_ps(v);
        let vi = _mm_movehdup_ps(v);
        let ws = _mm_shuffle_ps(w, w, 0b10110001);
        _mm_fmaddsub_ps(vr, w, _mm_mul_ps(vi, ws))
    }

    /// Natural-layout complex multiply (4 complexes per `__m256`): the
    /// constants arrive pre-split into re/im broadcasts, so each data
    /// register swaps in-register once (`permute_ps`) instead of
    /// moveldup+movehdup+shuffle per product. Same per-element formula as
    /// `cmul8`; even lanes are bit-identical, odd lanes commute the FMA
    /// operands (<= 1 ulp f32). `mr`/`mi` may also carry per-complex
    /// constants (built with `setr_ps`) — the stride-1 kernel uses that.
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn cmul8_nat(v: __m256, mr: __m256, mi: __m256) -> __m256 {
        let vs = _mm256_permute_ps(v, 0b10110001);
        _mm256_fmaddsub_ps(v, mr, _mm256_mul_ps(vs, mi))
    }

    /// 128-bit natural variant (2 complexes).
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn cmul4_nat(v: __m128, mr: __m128, mi: __m128) -> __m128 {
        let vs = _mm_shuffle_ps(v, v, 0b10110001);
        _mm_fmaddsub_ps(v, mr, _mm_mul_ps(vs, mi))
    }

    /// Broadcast a complex scalar: [re, im, re, im, re, im, re, im].
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn splat_c(z: Complex32) -> __m256 {
        _mm256_setr_ps(z.re, z.im, z.re, z.im, z.re, z.im, z.re, z.im)
    }

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn splat_c128(z: Complex32) -> __m128 {
        _mm_setr_ps(z.re, z.im, z.re, z.im)
    }

    /// Load 4 complex (unaligned).
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn load4(p: *const Complex32) -> __m256 {
        _mm256_loadu_ps(p as *const f32)
    }

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn store4(p: *mut Complex32, v: __m256) {
        _mm256_storeu_ps(p as *mut f32, v)
    }

    /// Load 2 complex.
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn load2(p: *const Complex32) -> __m128 {
        _mm_loadu_ps(p as *const f32)
    }

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn store2(p: *mut Complex32, v: __m128) {
        _mm_storeu_ps(p as *mut f32, v)
    }
}

// ──────────────────────────────────────────────────────────────────────
// 2×2 pair transform: out_lo = m00·lo + m01·hi ; out_hi = m10·lo + m11·hi
// for every pair (i, i+stride) in each block of 2·stride amplitudes.
// Semantics are identical to `crate::simd::pair_pass`.
// ──────────────────────────────────────────────────────────────────────

/// `SF_PAIR_V2_F32=1` selects the natural-layout complex multiply inside
/// the f32 pair kernels (latched once per process; independent of the f64
/// `SF_PAIR_V2` latch so the two lanes can carry separate verdicts).
/// Default is the historical double-shuffle form until the
/// `pair_v2_microbench_f32` verdict says otherwise.
#[inline]
pub(crate) fn pair_v2_f32_enabled() -> bool {
    use std::sync::atomic::{AtomicU8, Ordering};
    static STATE: AtomicU8 = AtomicU8::new(0);
    match STATE.load(Ordering::Relaxed) {
        1 => true,
        2 => false,
        _ => {
            let on = std::env::var("SF_PAIR_V2_F32")
                .map(|v| v == "1")
                .unwrap_or(false);
            STATE.store(if on { 1 } else { 2 }, Ordering::Relaxed);
            on
        }
    }
}

#[inline]
pub fn pair_pass(s: &mut [Complex32], stride: usize, m: [[Complex32; 2]; 2]) {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::simd::simd_enabled() && stride >= 2 {
            unsafe {
                if pair_v2_f32_enabled() {
                    pair_runs_avx2_v2(s.as_mut_ptr(), s.len(), stride, m);
                } else {
                    pair_runs_avx2(s.as_mut_ptr(), s.len(), stride, m);
                }
            }
            return;
        }
        if crate::simd::simd_enabled() && stride == 1 {
            unsafe {
                if pair_v2_f32_enabled() {
                    pair_stride1_avx2_v2(s.as_mut_ptr(), s.len(), m);
                } else {
                    pair_stride1_avx2(s.as_mut_ptr(), s.len(), m);
                }
            }
            return;
        }
    }
    pair_scalar(s, stride, m);
}

/// Scalar fallback — same formula/order as the f64 scalar kernel.
fn pair_scalar(s: &mut [Complex32], stride: usize, m: [[Complex32; 2]; 2]) {
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

/// Pairs at distance `stride >= 2`: both sides are contiguous runs, so four
/// pairs are 4-complex (32 B) vector loads. 2-wide and scalar tails keep
/// every stride (including odd, non-power-of-two) fully covered.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn pair_runs_avx2(p: *mut Complex32, len: usize, stride: usize, m: [[Complex32; 2]; 2]) {
    use std::arch::x86_64::*;
    use x86::*;
    let m00 = splat_c(m[0][0]);
    let m01 = splat_c(m[0][1]);
    let m10 = splat_c(m[1][0]);
    let m11 = splat_c(m[1][1]);
    let m00h = splat_c128(m[0][0]);
    let m01h = splat_c128(m[0][1]);
    let m10h = splat_c128(m[1][0]);
    let m11h = splat_c128(m[1][1]);
    let block = stride * 2;
    let n_groups = len / block;
    for g in 0..n_groups {
        let base = p.add(g * block);
        let mut k = 0usize;
        while k + 4 <= stride {
            let a = load4(base.add(k));
            let b = load4(base.add(k + stride));
            let oa = _mm256_add_ps(cmul8(a, m00), cmul8(b, m01));
            let ob = _mm256_add_ps(cmul8(a, m10), cmul8(b, m11));
            store4(base.add(k), oa);
            store4(base.add(k + stride), ob);
            k += 4;
        }
        while k + 2 <= stride {
            let a = load2(base.add(k));
            let b = load2(base.add(k + stride));
            let oa = _mm_add_ps(cmul4(a, m00h), cmul4(b, m01h));
            let ob = _mm_add_ps(cmul4(a, m10h), cmul4(b, m11h));
            store2(base.add(k), oa);
            store2(base.add(k + stride), ob);
            k += 2;
        }
        while k < stride {
            let a = *base.add(k);
            let b = *base.add(k + stride);
            *base.add(k) = m[0][0] * a + m[0][1] * b;
            *base.add(k + stride) = m[1][0] * a + m[1][1] * b;
            k += 1;
        }
    }
}

/// Pairs at distance 1: one `__m256` holds two adjacent pairs
/// `[a0 b0 a1 b1]`; a two-instruction in-register deinterleave splits the
/// even/odd complexes into duplicate-pair form, transforms both, and
/// re-interleaves the outputs into `[A0 B0 A1 B1]`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn pair_stride1_avx2(p: *mut Complex32, len: usize, m: [[Complex32; 2]; 2]) {
    use std::arch::x86_64::*;
    use x86::*;
    let m00 = splat_c(m[0][0]);
    let m01 = splat_c(m[0][1]);
    let m10 = splat_c(m[1][0]);
    let m11 = splat_c(m[1][1]);
    let mut k = 0usize;
    while k + 4 <= len {
        let v = load4(p.add(k)); // [a0 b0 a1 b1]
        let lo = _mm256_permute_ps(v, 0x44); // [a0 a0 a1 a1] (dup form)
        let hi = _mm256_permute_ps(v, 0xEE); // [b0 b0 b1 b1] (dup form)
        let olo = _mm256_add_ps(cmul8(lo, m00), cmul8(hi, m01));
        let ohi = _mm256_add_ps(cmul8(lo, m10), cmul8(hi, m11));
        store4(p.add(k), _mm256_shuffle_ps(olo, ohi, 0x44)); // [A0 B0 A1 B1]
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

/// Natural-layout twin of `pair_runs_avx2` (selected by `SF_PAIR_V2_F32`):
/// the four constants are split into re/im broadcasts once per call and each
/// data register swaps in-register once per product instead of
/// moveldup+movehdup+shuffle. Same per-element formula; results agree with
/// the legacy kernels to float32 rounding (<= 1 ulp).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn pair_runs_avx2_v2(p: *mut Complex32, len: usize, stride: usize, m: [[Complex32; 2]; 2]) {
    use std::arch::x86_64::*;
    use x86::*;
    let m00r = _mm256_set1_ps(m[0][0].re);
    let m00i = _mm256_set1_ps(m[0][0].im);
    let m01r = _mm256_set1_ps(m[0][1].re);
    let m01i = _mm256_set1_ps(m[0][1].im);
    let m10r = _mm256_set1_ps(m[1][0].re);
    let m10i = _mm256_set1_ps(m[1][0].im);
    let m11r = _mm256_set1_ps(m[1][1].re);
    let m11i = _mm256_set1_ps(m[1][1].im);
    let m00h_r = _mm_set1_ps(m[0][0].re);
    let m00h_i = _mm_set1_ps(m[0][0].im);
    let m01h_r = _mm_set1_ps(m[0][1].re);
    let m01h_i = _mm_set1_ps(m[0][1].im);
    let m10h_r = _mm_set1_ps(m[1][0].re);
    let m10h_i = _mm_set1_ps(m[1][0].im);
    let m11h_r = _mm_set1_ps(m[1][1].re);
    let m11h_i = _mm_set1_ps(m[1][1].im);
    let block = stride * 2;
    let n_groups = len / block;
    for g in 0..n_groups {
        let base = p.add(g * block);
        let mut k = 0usize;
        while k + 4 <= stride {
            let a = load4(base.add(k));
            let b = load4(base.add(k + stride));
            let oa = _mm256_add_ps(cmul8_nat(a, m00r, m00i), cmul8_nat(b, m01r, m01i));
            let ob = _mm256_add_ps(cmul8_nat(a, m10r, m10i), cmul8_nat(b, m11r, m11i));
            store4(base.add(k), oa);
            store4(base.add(k + stride), ob);
            k += 4;
        }
        while k + 2 <= stride {
            let a = load2(base.add(k));
            let b = load2(base.add(k + stride));
            let oa = _mm_add_ps(cmul4_nat(a, m00h_r, m00h_i), cmul4_nat(b, m01h_r, m01h_i));
            let ob = _mm_add_ps(cmul4_nat(a, m10h_r, m10h_i), cmul4_nat(b, m11h_r, m11h_i));
            store2(base.add(k), oa);
            store2(base.add(k + stride), ob);
            k += 2;
        }
        while k < stride {
            let a = *base.add(k);
            let b = *base.add(k + stride);
            *base.add(k) = m[0][0] * a + m[0][1] * b;
            *base.add(k + stride) = m[1][0] * a + m[1][1] * b;
            k += 1;
        }
    }
}

/// Natural-layout twin of `pair_stride1_avx2`: the per-complex constants
/// alternate `(m00, m11)` between the register value and `(m01, m10)`
/// between its 64-bit-swapped twin, so each iteration is two natural
/// multiplies plus one add — no deinterleave/reinterleave shuffles at all.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn pair_stride1_avx2_v2(p: *mut Complex32, len: usize, m: [[Complex32; 2]; 2]) {
    use std::arch::x86_64::*;
    use x86::*;
    // v = [a0 b0 a1 b1] (two adjacent pairs); w = [b0 a0 b1 a1].
    let pr = _mm256_setr_ps(
        m[0][0].re, m[0][0].re, m[1][1].re, m[1][1].re, m[0][0].re, m[0][0].re, m[1][1].re,
        m[1][1].re,
    );
    let pi = _mm256_setr_ps(
        m[0][0].im, m[0][0].im, m[1][1].im, m[1][1].im, m[0][0].im, m[0][0].im, m[1][1].im,
        m[1][1].im,
    );
    let qr = _mm256_setr_ps(
        m[0][1].re, m[0][1].re, m[1][0].re, m[1][0].re, m[0][1].re, m[0][1].re, m[1][0].re,
        m[1][0].re,
    );
    let qi = _mm256_setr_ps(
        m[0][1].im, m[0][1].im, m[1][0].im, m[1][0].im, m[0][1].im, m[0][1].im, m[1][0].im,
        m[1][0].im,
    );
    let mut k = 0usize;
    while k + 4 <= len {
        let v = load4(p.add(k)); // [a0 b0 a1 b1]
        let w = _mm256_permute_ps(v, 0x4E); // [b0 a0 b1 a1]
        store4(
            p.add(k),
            _mm256_add_ps(cmul8_nat(v, pr, pi), cmul8_nat(w, qr, qi)),
        );
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
// Diagonal passes: same piecewise-constant coefficient logic as the f64
// kernels (period table for low bits, constant-run walk for high bits,
// `off` = absolute index of `s[0]` so unaligned slices keep the phase).
// ──────────────────────────────────────────────────────────────────────

/// Build a period-`p` coefficient table with three wrap-around slack entries
/// so the 4-complex vector window starting at phase `p-1` (the last valid
/// phase) reads `[t[p-1], t[0], t[1], t[2]]` contiguously.
fn make_table(p: usize, f: impl Fn(usize) -> Complex32) -> Vec<Complex32> {
    let mut t: Vec<Complex32> = (0..p).map(&f).collect();
    t.push(t[0]);
    t.push(t[1]);
    t.push(t[2 % p]);
    t
}

/// state[i] *= (bit q of i ? d1 : d0).
pub fn diag1_pass(s: &mut [Complex32], off: usize, q: usize, d0: Complex32, d1: Complex32) {
    let n = s.len();
    if q >= 9 {
        let bs = 1usize << q;
        let mut i = 0usize;
        while i < n {
            let a = off + i;
            let c = if bit(a, q) == 1 { d1 } else { d0 };
            let take = (bs - (a % bs)).min(n - i);
            scale_const(&mut s[i..i + take], c);
            i += take;
        }
    } else {
        let period = 1usize << (q + 1);
        let table = make_table(period, |j| if bit(j, q) == 1 { d1 } else { d0 });
        scale_periodic(s, off, &table);
    }
}

/// state[i] *= d[(bit(q1, i) << 1) | bit(q2, i)].
pub fn diag2_pass(s: &mut [Complex32], off: usize, q1: usize, q2: usize, d: &[Complex32; 4]) {
    let n = s.len();
    let (p1, p2) = if q1 < q2 { (q1, q2) } else { (q2, q1) };
    if p1 >= 9 {
        let bs = 1usize << p1;
        let mut i = 0usize;
        while i < n {
            let a = off + i;
            let c = d[(bit(a, q1) << 1) | bit(a, q2)];
            let take = (bs - (a % bs)).min(n - i);
            scale_const(&mut s[i..i + take], c);
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
            scale_periodic(&mut s[i..i + take], a, table);
            i += take;
        }
    } else {
        let period = 1usize << (p2 + 1);
        let table = make_table(period, |j| d[(bit(j, q1) << 1) | bit(j, q2)]);
        scale_periodic(s, off, &table);
    }
}

/// Multiply by a constant coefficient (vectorized broadcast).
fn scale_const(s: &mut [Complex32], c: Complex32) {
    if c == Complex32::new(1.0, 0.0) {
        return;
    }
    #[cfg(target_arch = "x86_64")]
    {
        if crate::simd::simd_enabled() {
            unsafe { scale_const_avx2(s.as_mut_ptr(), s.len(), c) };
            return;
        }
    }
    for a in s.iter_mut() {
        *a = c * *a;
    }
}

/// Multiply by a periodic coefficient table (period = `table.len() - 3`;
/// the three trailing entries are wrap-around slack); `off` is the absolute
/// index of `s[0]`, so the phase carries across arbitrarily aligned slices.
fn scale_periodic(s: &mut [Complex32], off: usize, table: &[Complex32]) {
    let period = table.len() - 3;
    let ph0 = off & (period - 1);
    #[cfg(target_arch = "x86_64")]
    {
        if crate::simd::simd_enabled() {
            unsafe { scale_table_avx2(s.as_mut_ptr(), table.as_ptr(), s.len(), ph0, period) };
            return;
        }
    }
    for (i, a) in s.iter_mut().enumerate() {
        *a = table[(ph0 + i) & (period - 1)] * *a;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn scale_const_avx2(p: *mut Complex32, len: usize, c: Complex32) {
    use x86::*;
    let cv = splat_c(c);
    let mut i = 0usize;
    while i + 4 <= len {
        let v = load4(p.add(i));
        store4(p.add(i), cmul8(v, cv));
        i += 4;
    }
    while i < len {
        *p.add(i) = c * *p.add(i);
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn scale_table_avx2(
    p: *mut Complex32,
    t: *const Complex32,
    len: usize,
    ph0: usize,
    period: usize,
) {
    use x86::*;
    let mask = period - 1;
    let mut i = 0usize;
    while i + 4 <= len {
        let v = load4(p.add(i));
        let w = load4(t.add((ph0 + i) & mask));
        store4(p.add(i), cmul8(v, w));
        i += 4;
    }
    while i < len {
        *p.add(i) = *t.add((ph0 + i) & mask) * *p.add(i);
        i += 1;
    }
}

// ──────────────────────────────────────────────────────────────────────
// Contiguous run swap (X / CNOT blocks).
// ──────────────────────────────────────────────────────────────────────

/// Swap `run` complex starting at `a` with `run` complex starting at `b`.
///
/// # Safety
/// `p.add(a)` and `p.add(b)` must each be valid for `run` consecutive
/// `Complex32` elements in the same allocation, and the two runs must not
/// overlap.
#[inline]
pub unsafe fn swap_runs(p: *mut Complex32, a: usize, b: usize, run: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::simd::simd_enabled() && run >= 2 {
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
unsafe fn swap_runs_avx2(p: *mut Complex32, a: usize, b: usize, run: usize) {
    use x86::*;
    let mut k = 0usize;
    while k + 4 <= run {
        let x = load4(p.add(a + k));
        let y = load4(p.add(b + k));
        store4(p.add(a + k), y);
        store4(p.add(b + k), x);
        k += 4;
    }
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

// ──────────────────────────────────────────────────────────────────────
// Gate application kernels (f32 twins of the dag.rs kernels)
// ──────────────────────────────────────────────────────────────────────

/// Swap amplitude pairs at stride `stride` in-place (X gate).
fn inplace_swap_pairs(state: &mut [Complex32], stride: usize, use_par: bool) {
    let dim = state.len();
    let block = stride * 2;
    let n_groups = dim / block;

    if use_par {
        let sp = SendPtr32(state.as_mut_ptr());
        (0..n_groups).into_par_iter().for_each(|g| {
            let base = g * block;
            unsafe {
                sp.swap_run(base, base + stride, stride);
            }
        });
    } else {
        let ptr = state.as_mut_ptr();
        for g in 0..n_groups {
            let base = g * block;
            unsafe {
                swap_runs(ptr, base, base + stride, stride);
            }
        }
    }
}

/// CNOT in-place: flip the target bit wherever the control bit is 1
/// (block-structured run swaps, same layout as the f64 kernel).
fn inplace_cnot(state: &mut [Complex32], ctrl: usize, tgt: usize, use_par: bool) {
    let dim = state.len();
    let (lo, hi) = if ctrl < tgt { (ctrl, tgt) } else { (tgt, ctrl) };
    let run = 1usize << lo;
    let hi_bit = 1usize << hi;
    let block = hi_bit << 1;
    let n_blocks = dim / block;
    let n_runs = hi_bit >> (lo + 1);
    let (a0, delta) = if ctrl < tgt {
        (run, hi_bit)
    } else {
        (hi_bit, run)
    };

    if use_par {
        const TILE: usize = 4096;
        let tile = run.min(TILE);
        let n_tiles = run / tile;
        let per_block = n_runs * n_tiles;
        let sp = SendPtr32(state.as_mut_ptr());
        (0..n_blocks * per_block).into_par_iter().for_each(|t| {
            let base = (t / per_block) * block;
            let rem = t % per_block;
            let a = base + a0 + ((rem / n_tiles) << 1) * run + (rem % n_tiles) * tile;
            let b = a + delta;
            unsafe {
                sp.swap_run(a, b, tile);
            }
        });
    } else {
        let ptr = state.as_mut_ptr();
        for g in 0..n_blocks {
            let base = g * block;
            for r in 0..n_runs {
                let a = base + a0 + (r << 1) * run;
                let b = a + delta;
                unsafe {
                    swap_runs(ptr, a, b, run);
                }
            }
        }
    }
}

/// CCX (Toffoli) in-place: swap state[i] and state[i|mt] where both
/// controls set and target=0.
fn inplace_ccx(state: &mut [Complex32], mc1: usize, mc2: usize, mt: usize, use_par: bool) {
    let dim = state.len();
    if use_par {
        let sp = SendPtr32(state.as_mut_ptr());
        (0..dim).into_par_iter().for_each(|i| {
            if (i & mc1) != 0 && (i & mc2) != 0 && (i & mt) == 0 {
                unsafe {
                    sp.swap(i, i | mt);
                }
            }
        });
    } else {
        for i in 0..dim {
            if (i & mc1) != 0 && (i & mc2) != 0 && (i & mt) == 0 {
                state.swap(i, i | mt);
            }
        }
    }
}

/// CSWAP in-place: swap target bits when control is set.
fn inplace_cswap(state: &mut [Complex32], mc: usize, mt1: usize, mt2: usize, use_par: bool) {
    let dim = state.len();
    if use_par {
        let sp = SendPtr32(state.as_mut_ptr());
        (0..dim).into_par_iter().for_each(|i| {
            if (i & mc) != 0 && (i & mt1) != 0 && (i & mt2) == 0 {
                unsafe {
                    sp.swap(i, (i ^ mt1) | mt2);
                }
            }
        });
    } else {
        for i in 0..dim {
            if (i & mc) != 0 && (i & mt1) != 0 && (i & mt2) == 0 {
                state.swap(i, (i ^ mt1) | mt2);
            }
        }
    }
}

/// General 1q gate in-place via the SIMD pair kernel.
fn inplace_1q_general(
    state: &mut [Complex32],
    stride: usize,
    m: [[Complex32; 2]; 2],
    use_par: bool,
) {
    let block = stride * 2;
    if use_par {
        // Chunks aligned to the pair-block size so SIMD blocks never
        // straddle a chunk boundary.
        let chunk = (state.len() / 16).max(1024).div_ceil(block) * block;
        state
            .par_chunks_mut(chunk)
            .for_each(|s| pair_pass(s, stride, m));
    } else {
        pair_pass(state, stride, m);
    }
}

/// General 2q gate in-place: transform groups of 4 amplitudes.
fn inplace_2q_general(
    state: &mut [Complex32],
    q1: usize,
    q2: usize,
    g: &[[Complex32; 4]; 4],
    use_par: bool,
) {
    let dim = state.len();
    let (lo_q, hi_q) = if q1 < q2 { (q1, q2) } else { (q2, q1) };
    let m_lo = 1usize << lo_q;
    let m_hi = 1usize << hi_q;
    let mq1 = 1usize << q1;
    let mq2 = 1usize << q2;
    let n_groups = dim >> 2;

    if use_par {
        let sp = SendPtr32(state.as_mut_ptr());
        (0..n_groups).into_par_iter().for_each(|gi| {
            let base = deposit_bits_2(gi, m_lo, m_hi);
            let i00 = base;
            let i01 = base | mq2;
            let i10 = base | mq1;
            let i11 = base | mq1 | mq2;
            unsafe {
                let v00 = sp.get(i00);
                let v01 = sp.get(i01);
                let v10 = sp.get(i10);
                let v11 = sp.get(i11);
                sp.set(
                    i00,
                    g[0][0] * v00 + g[0][1] * v01 + g[0][2] * v10 + g[0][3] * v11,
                );
                sp.set(
                    i01,
                    g[1][0] * v00 + g[1][1] * v01 + g[1][2] * v10 + g[1][3] * v11,
                );
                sp.set(
                    i10,
                    g[2][0] * v00 + g[2][1] * v01 + g[2][2] * v10 + g[2][3] * v11,
                );
                sp.set(
                    i11,
                    g[3][0] * v00 + g[3][1] * v01 + g[3][2] * v10 + g[3][3] * v11,
                );
            }
        });
    } else {
        for gi in 0..n_groups {
            let base = deposit_bits_2(gi, m_lo, m_hi);
            let i00 = base;
            let i01 = base | mq2;
            let i10 = base | mq1;
            let i11 = base | mq1 | mq2;
            let v00 = state[i00];
            let v01 = state[i01];
            let v10 = state[i10];
            let v11 = state[i11];
            state[i00] = g[0][0] * v00 + g[0][1] * v01 + g[0][2] * v10 + g[0][3] * v11;
            state[i01] = g[1][0] * v00 + g[1][1] * v01 + g[1][2] * v10 + g[1][3] * v11;
            state[i10] = g[2][0] * v00 + g[2][1] * v01 + g[2][2] * v10 + g[2][3] * v11;
            state[i11] = g[3][0] * v00 + g[3][1] * v01 + g[3][2] * v10 + g[3][3] * v11;
        }
    }
}

/// Apply one planned gate op (mirror of the f64 `apply_gate_inplace`
/// dispatch; all the classification decisions are made on the f64 matrix).
fn apply_gate(
    state: &mut [Complex32],
    op: &QuantumOp,
    gate_u: &nalgebra::DMatrix<num_complex::Complex64>,
    use_par: bool,
) {
    let dim = state.len();

    if op.qubits.len() == 1 {
        let t = op.qubits[0];
        let u00 = c32(gate_u[(0, 0)]);
        let u01 = c32(gate_u[(0, 1)]);
        let u10 = c32(gate_u[(1, 0)]);
        let u11 = c32(gate_u[(1, 1)]);
        let stride = 1usize << t;

        if is_diag_1q(gate_u) {
            apply_diag1(state, t, u00, u11, use_par);
            return;
        }

        let is_x = gate_u[(0, 0)].norm() < 1e-14
            && gate_u[(1, 1)].norm() < 1e-14
            && (gate_u[(0, 1)] - num_complex::Complex64::new(1.0, 0.0)).norm() < 1e-14
            && (gate_u[(1, 0)] - num_complex::Complex64::new(1.0, 0.0)).norm() < 1e-14;
        if is_x {
            inplace_swap_pairs(state, stride, use_par);
            return;
        }

        inplace_1q_general(state, stride, [[u00, u01], [u10, u11]], use_par);
    } else if op.qubits.len() == 2 {
        let q1 = op.qubits[0];
        let q2 = op.qubits[1];

        if op.op_type == OpType::CNOT {
            inplace_cnot(state, q1, q2, use_par);
            return;
        }

        if is_diag_2q(gate_u) {
            let d = [
                c32(gate_u[(0, 0)]),
                c32(gate_u[(1, 1)]),
                c32(gate_u[(2, 2)]),
                c32(gate_u[(3, 3)]),
            ];
            apply_diag2(state, q1, q2, &d, use_par);
            return;
        }

        let mut g = [[Complex32::new(0.0, 0.0); 4]; 4];
        for (r, row) in g.iter_mut().enumerate() {
            for (c, e) in row.iter_mut().enumerate() {
                *e = c32(gate_u[(r, c)]);
            }
        }
        inplace_2q_general(state, q1, q2, &g, use_par);
    } else if op.qubits.len() == 3 {
        match op.op_type {
            OpType::CCX => {
                let mc1 = 1usize << op.qubits[0];
                let mc2 = 1usize << op.qubits[1];
                let mt = 1usize << op.qubits[2];
                inplace_ccx(state, mc1, mc2, mt, use_par);
            }
            OpType::CSWAP => {
                let mc = 1usize << op.qubits[0];
                let mt1 = 1usize << op.qubits[1];
                let mt2 = 1usize << op.qubits[2];
                inplace_cswap(state, mc, mt1, mt2, use_par);
            }
            _ => {
                // General 3q: iterate over independent groups of 8 (scalar;
                // 3q unitaries are rare and the group math is trivial).
                let q1 = op.qubits[0];
                let q2 = op.qubits[1];
                let q3 = op.qubits[2];
                let qs = sorted_3(q1, q2, q3);
                let m0 = 1usize << qs[0];
                let m1 = 1usize << qs[1];
                let m2 = 1usize << qs[2];
                let n_groups = dim >> 3;
                for g in 0..n_groups {
                    let base = deposit_bits_3(g, m0, m1, m2);
                    let mut vals = [Complex32::new(0.0, 0.0); 8];
                    for col in 0..8usize {
                        let b1 = (col >> 2) & 1;
                        let b2 = (col >> 1) & 1;
                        let b3 = col & 1;
                        vals[col] = state[base | (b1 << q1) | (b2 << q2) | (b3 << q3)];
                    }
                    for row in 0..8usize {
                        let b1 = (row >> 2) & 1;
                        let b2 = (row >> 1) & 1;
                        let b3 = row & 1;
                        let idx = base | (b1 << q1) | (b2 << q2) | (b3 << q3);
                        let mut acc = Complex32::new(0.0, 0.0);
                        for col in 0..8usize {
                            acc += c32(gate_u[(row, col)]) * vals[col];
                        }
                        state[idx] = acc;
                    }
                }
            }
        }
    }
}

/// Diagonal 1q wrapper with the same parallel chunking as the f64 lane.
fn apply_diag1(state: &mut [Complex32], q: usize, d0: Complex32, d1: Complex32, use_par: bool) {
    if use_par {
        let chunk = (state.len() / 16).max(1024);
        state
            .par_chunks_mut(chunk)
            .enumerate()
            .for_each(|(c, s)| diag1_pass(s, c * chunk, q, d0, d1));
    } else {
        diag1_pass(state, 0, q, d0, d1);
    }
}

/// Diagonal 2q wrapper.
fn apply_diag2(state: &mut [Complex32], q1: usize, q2: usize, d: &[Complex32; 4], use_par: bool) {
    if use_par {
        let chunk = (state.len() / 16).max(1024);
        state
            .par_chunks_mut(chunk)
            .enumerate()
            .for_each(|(c, s)| diag2_pass(s, c * chunk, q1, q2, d));
    } else {
        diag2_pass(state, 0, q1, q2, d);
    }
}

/// Map a group index g to a state index with the two qubit bits cleared.
/// `m_lo` and `m_hi` are the masks for the lower and higher qubit.
#[inline(always)]
fn deposit_bits_2(g: usize, m_lo: usize, m_hi: usize) -> usize {
    let p0 = m_lo.trailing_zeros() as usize;
    let p1 = m_hi.trailing_zeros() as usize;
    let mask0 = (1usize << p0) - 1;
    let x = (g & mask0) | ((g & !mask0) << 1);
    let mask1 = (1usize << p1) - 1;
    (x & mask1) | ((x & !mask1) << 1)
}

/// Sort 3 qubit indices for iteration.
fn sorted_3(a: usize, b: usize, c: usize) -> [usize; 3] {
    let mut arr = [a, b, c];
    arr.sort_unstable();
    arr
}

/// Map group index to base index with 3 qubit bits cleared.
fn deposit_bits_3(g: usize, m0: usize, m1: usize, m2: usize) -> usize {
    let p0 = m0.trailing_zeros() as usize;
    let p1 = m1.trailing_zeros() as usize;
    let p2 = m2.trailing_zeros() as usize;
    let mask0 = (1usize << p0) - 1;
    let x = (g & mask0) | ((g & !mask0) << 1);
    let mask1 = (1usize << p1) - 1;
    let y = (x & mask1) | ((x & !mask1) << 1);
    let mask2 = (1usize << p2) - 1;
    (y & mask2) | ((y & !mask2) << 1)
}

/// Scatter the low `masks.len()` bits of `s` onto the bits of `masks`.
#[inline(always)]
fn spread_bits(s: usize, masks: &[usize]) -> usize {
    let mut out = 0usize;
    for (b, &m) in masks.iter().enumerate() {
        if (s >> b) & 1 == 1 {
            out |= m;
        }
    }
    out
}

/// Insert zero bits at the (ascending) positions of `masks` in `g`.
#[inline(always)]
fn deposit_bits_masked(g: usize, masks: &[usize]) -> usize {
    let mut x = g;
    for &m in masks {
        let p = m.trailing_zeros() as usize;
        let low = (1usize << p) - 1;
        x = (x & low) | ((x & !low) << 1);
    }
    x
}

/// Apply a fused block of k 1q gates on distinct qubits (tensor-product
/// structure) in ONE sweep — f32 twin of `apply_fused_qubits_inplace`,
/// including the `pre_diag`/`post_diag` folded diagonal stages.
fn apply_fused(
    state: &mut [Complex32],
    qs: &[usize],
    u: &[num_complex::Complex64],
    pre_diag: &[Diag2Stage],
    post_diag: &[Diag2Stage],
    use_par: bool,
) {
    use std::mem::MaybeUninit;
    let k = qs.len();
    let sub = 1usize << k;
    let dim = state.len();
    let n_groups = dim >> k;
    let fac_pre = build_stage_factors32(sub, pre_diag);
    let fac_post = build_stage_factors32(sub, post_diag);
    // Streaming twin: with no folded diagonal stages a fused block is pure
    // reordering plus `k` pair passes over a scratch copy — and the streaming
    // form is bit-identical (same kernels, same stage order; gather/scatter
    // are plain copies) while moving half the traffic per pass. In the f32
    // lane that makes streaming FASTER than the fused sweep (the fused kernel
    // is instruction/latency-bound): blockmap2 measured 1380.9 ms streaming
    // vs 1782.5 ms fused on the 24q vqc, so skip the gather/scatter whenever
    // nothing needs folding.
    if fac_pre.is_empty() && fac_post.is_empty() {
        let mats: Vec<[[Complex32; 2]; 2]> = (0..k)
            .map(|t| {
                [
                    [c32(u[4 * t]), c32(u[4 * t + 1])],
                    [c32(u[4 * t + 2]), c32(u[4 * t + 3])],
                ]
            })
            .collect();
        // Stage t runs on the machine-order stride of qubit qs[t] (the fused
        // path sees buffer bit t instead because its gather reorders them).
        if use_par {
            // Chunks must hold whole pair blocks so no pair straddles a
            // boundary: 2 * max(1 << qs).
            let block = 2usize << qs.iter().copied().max().unwrap_or(0);
            let chunk = ((dim / 16).max(block) / block).max(1) * block;
            state.par_chunks_mut(chunk).for_each(|s| {
                for t in 0..k {
                    pair_pass(s, 1usize << qs[t], mats[t]);
                }
            });
        } else {
            for t in 0..k {
                pair_pass(state, 1usize << qs[t], mats[t]);
            }
        }
        return;
    }

    let masks: Vec<usize> = qs.iter().map(|&q| 1usize << q).collect();
    let offs: Vec<usize> = (0..sub).map(|s| spread_bits(s, &masks)).collect();

    let body = |sp: &SendPtr32, g: usize| {
        let base = deposit_bits_masked(g, &masks);
        // SAFETY: [MaybeUninit<T>; N] has the same layout as MaybeUninit<[T; N]>.
        // 1024 = 2^FUSE_K_MAX (dag.rs) — the f64 twin uses the same capacity.
        let mut val: [MaybeUninit<Complex32>; 1024] =
            unsafe { MaybeUninit::uninit().assume_init() };
        for s in 0..sub {
            val[s].write(unsafe { sp.get(base | offs[s]) });
        }
        // SAFETY: the gather above initialized all `sub` slots.
        let buf: &mut [Complex32] =
            unsafe { std::slice::from_raw_parts_mut(val.as_mut_ptr() as *mut Complex32, sub) };
        if !fac_pre.is_empty() {
            for s in 0..sub {
                buf[s] *= fac_pre[s];
            }
        }
        // Apply gate t to every pair (s, s | 1<<t) — tensor structure, no
        // dense 2^k matvec. k vector passes over the resident buffer.
        for t in 0..k {
            let m = [
                [c32(u[4 * t]), c32(u[4 * t + 1])],
                [c32(u[4 * t + 2]), c32(u[4 * t + 3])],
            ];
            pair_pass(buf, 1usize << t, m);
        }
        if !fac_post.is_empty() {
            for s in 0..sub {
                buf[s] *= fac_post[s];
            }
        }
        for s in 0..sub {
            unsafe {
                sp.set(base | offs[s], buf[s]);
            }
        }
    };

    let sp = SendPtr32(state.as_mut_ptr());
    if use_par {
        (0..n_groups).into_par_iter().for_each(|g| body(&sp, g));
    } else {
        for g in 0..n_groups {
            body(&sp, g);
        }
    }
}

/// Combined per-entry factor table for folded diagonal stages (f32 twin of
/// `build_stage_factors`): entry `s` gets the product of every stage's
/// `d[(b_i << 1) | b_j]`, each entry cast to complex64 once.
fn build_stage_factors32(sub: usize, stages: &[Diag2Stage]) -> Vec<Complex32> {
    if stages.is_empty() {
        return Vec::new();
    }
    let mut fac = vec![Complex32::new(1.0, 0.0); sub];
    for st in stages {
        let mi = 1usize << (st.i as usize);
        let mj = 1usize << (st.j as usize);
        for (s, f) in fac.iter_mut().enumerate() {
            let idx = (((s & mi) != 0) as usize) << 1 | (((s & mj) != 0) as usize);
            *f *= c32(st.d[idx]);
        }
    }
    fac
}

/// Apply a whole diagonal run in ONE Gray-code phase sweep (f32 twin of
/// `apply_phase_run`). The phase product is re-seeded from a direct f64
/// evaluation every 16384 steps — a quarter of the f64 lane's interval, so
/// the f32 random-walk drift stays ≤ ~1e-5 per segment.
fn apply_phase(state: &mut [Complex32], n: usize, plan: &PhasePlan, use_par: bool) {
    let dim = state.len();
    debug_assert_eq!(dim, 1usize << n);

    let process = |s: &mut [Complex32], off: usize| {
        let len = s.len();
        let (sin0, cos0) = phase_eval(plan, off).sin_cos();
        let mut ph = Complex32::new(cos0 as f32, sin0 as f32);
        // Compensation residuals of the complex product (f32).
        let mut c_err: f32 = 0.0;
        let mut c_ierr: f32 = 0.0;
        s[0] *= ph;
        for k in 1..len {
            let idx = off + (k ^ (k >> 1));
            let q = (k as u64).trailing_zeros() as usize;
            let bq = (idx >> q) & 1;
            let mut inner = plan.lin[q];
            let row = &plan.quad[q * plan.n..q * plan.n + plan.n];
            for p in 0..plan.n {
                if p != q {
                    inner += row[p] * (((idx >> p) & 1) as f64);
                }
            }
            let delta = if bq == 1 { inner } else { -inner };
            let (sin_d, cos_d) = (delta as f32).sin_cos();
            // Compensated complex product (error-free transform): each
            // multiply's rounding residual is folded into the next, so the
            // phase-angle random walk is driven only by sin_cos rounding.
            let a = ph.re + c_err;
            let b = ph.im + c_ierr;
            let p1 = a * cos_d;
            let e1 = a.mul_add(cos_d, -p1);
            let q1 = b * sin_d;
            let e2 = b.mul_add(sin_d, -q1);
            let s1 = p1 - q1;
            let e3 = (p1 - s1) - q1;
            let p2 = a * sin_d;
            let e4 = a.mul_add(sin_d, -p2);
            let q2 = b * cos_d;
            let e5 = b.mul_add(cos_d, -q2);
            let s2 = p2 + q2;
            let e6 = (p2 - s2) + q2;
            ph = Complex32::new(s1, s2);
            c_err = (e1 - e2) + e3;
            c_ierr = (e4 + e5) + e6;
            // Same reduced reset interval as the f64 lane: exact phase
            // re-anchoring every 256 steps keeps drift ~16x lower.
            if k & 0x00FF == 0 {
                let (sin_r, cos_r) = phase_eval(plan, idx).sin_cos();
                ph = Complex32::new(cos_r as f32, sin_r as f32);
                c_err = 0.0;
                c_ierr = 0.0;
            }
            s[k ^ (k >> 1)] *= ph;
        }
    };

    if use_par {
        let chunk = (dim / 16).max(1024);
        state.par_chunks_mut(chunk).enumerate().for_each(|(c, s)| {
            process(s, c * chunk);
        });
    } else {
        process(state, 0);
    }
}

/// Goal index of amplitude `i` under the composed permutation.
#[inline(always)]
fn perm_dest(cols: &[usize], i: usize) -> usize {
    let mut d = 0usize;
    let mut x = i;
    while x != 0 {
        let q = x.trailing_zeros() as usize;
        d ^= cols[q];
        x &= x - 1;
    }
    d
}

/// Historical scatter sweep: `out[dest(i)] = state[i]` with a per-element
/// bit scan. Kept as the `SF_PERM_SWEEP=0` escape hatch (mirrors the f64
/// lane).
fn apply_perm_scatter(state: &[Complex32], out: &mut [Complex32], cols: &[usize], use_par: bool) {
    let dim = state.len();
    if use_par {
        let src = SendPtr32(state.as_ptr() as *mut Complex32);
        let dst = SendPtr32(out.as_mut_ptr());
        (0..dim).into_par_iter().for_each(|i| unsafe {
            let d = perm_dest(cols, i);
            dst.set(d, src.get(i));
        });
    } else {
        for i in 0..dim {
            out[perm_dest(cols, i)] = state[i];
        }
    }
}

/// Out-of-place permutation sweep: `out[dest(i)] = state[i]`.
///
/// Gather form (mirrors the f64 lane): sequential destination writes with
/// prefetched scattered reads driven by the shared byte-chunk tables;
/// `SF_PERM_SWEEP=0` restores the historical scatter loop. Parallel workers
/// split the destination space into cache-line-aligned contiguous ranges
/// (disjoint writes, shared read side).
fn apply_perm(state: &[Complex32], out: &mut [Complex32], plan: &PermPlan, use_par: bool) {
    let dim = state.len();
    let cols = &plan.cols;
    let (pf, _) = match perm_gather_config(plan) {
        Some(cfg) => cfg,
        None => return apply_perm_scatter(state, out, cols, use_par),
    };
    if use_par {
        let lanes = rayon::current_num_threads().max(1);
        let chunk = ((dim / (lanes * 8)).max(1 << 13) + 7) & !7usize;
        let n_ranges = dim.div_ceil(chunk);
        let src = SendPtr32(state.as_ptr() as *mut Complex32);
        let dst = SendPtr32(out.as_mut_ptr());
        (0..n_ranges).into_par_iter().for_each(|c| {
            let start = c * chunk;
            let end = (start + chunk).min(dim);
            unsafe { perm_gather_dispatch(src.raw(), dst.raw(), &plan.inv_tables, start, end, pf) };
        });
    } else {
        unsafe {
            perm_gather_dispatch(
                state.as_ptr(),
                out.as_mut_ptr(),
                &plan.inv_tables,
                0,
                dim,
                pf,
            )
        };
    }
}

// Per-thread scratch pool for the out-of-place permutation sweeps, so
// repeated simulations at the same size never re-zero the destination
// (same pooling strategy as the f64 lane).
std::thread_local! {
    #[allow(clippy::missing_const_for_thread_local)] // init already const; clippy 1.93 false positive
    static F32_PERM_SCRATCH: std::cell::RefCell<Vec<Complex32>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// Run the f32 (complex64) statevector lane on `dag`. The instruction plan
/// is built in f64 exactly like `QuantumDAG::simulate`; only the applied
/// arithmetic is f32.
pub fn simulate_f32(dag: &QuantumDAG) -> Vec<Complex32> {
    let n = dag.n_qubits;
    let (fused, insts) = dag.build_sim_plan();

    let dim = 1usize << n;
    let mut state = vec![Complex32::new(0.0, 0.0); dim];
    state[0] = Complex32::new(1.0, 0.0);

    // One parallelism decision per call (same policy as the f64 lane).
    let use_par = dim >= PARALLEL_THRESHOLD && rayon::current_num_threads() > 1;

    let mut scratch: Vec<Complex32> = Vec::new();

    for inst in &insts {
        match inst {
            SimInst::Gate { idx } => apply_gate(&mut state, fused[*idx].0, &fused[*idx].1, use_par),
            SimInst::Diag1 { q, d0, d1 } => {
                apply_diag1(&mut state, *q, c32(*d0), c32(*d1), use_par)
            }
            SimInst::Diag2 { q1, q2, d } => apply_diag2(
                &mut state,
                *q1,
                *q2,
                &[c32(d[0]), c32(d[1]), c32(d[2]), c32(d[3])],
                use_par,
            ),
            SimInst::PhaseRun { plan } => apply_phase(&mut state, n, plan, use_par),
            SimInst::Fused {
                qs,
                u,
                pre_diag,
                post_diag,
            } => apply_fused(&mut state, qs, u, pre_diag, post_diag, use_par),
            SimInst::PermuteRun { plan } => {
                if scratch.len() != dim {
                    let mut s = F32_PERM_SCRATCH.with(|c| std::mem::take(&mut *c.borrow_mut()));
                    if s.len() < dim {
                        s = vec![Complex32::new(0.0, 0.0); dim];
                    } else if s.len() > dim {
                        // Clamp a larger buffer from a previous simulation:
                        // the swap below must leave the state exactly 2^n
                        // elements long.
                        s.truncate(dim);
                    }
                    scratch = s;
                }
                apply_perm(&state, &mut scratch, plan, use_par);
                std::mem::swap(&mut state, &mut scratch);
            }
        }
    }

    // Return the (largest) scratch buffer to the thread pool.
    F32_PERM_SCRATCH.with(|c| {
        let mut slot = c.borrow_mut();
        if slot.len() < scratch.len() {
            *slot = std::mem::take(&mut scratch);
        }
    });

    state
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(n: usize, seed: u64) -> Vec<Complex32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let re = ((s >> 11) as f64 / (1u64 << 53) as f64) as f32 - 0.5;
                let im = ((s >> 33) as f64 / (1u64 << 31) as f64) as f32 - 0.5;
                Complex32::new(re, im)
            })
            .collect()
    }

    #[test]
    fn pair_matches_scalar() {
        for stride in [1usize, 2, 3, 4, 8, 11] {
            let m = [
                [Complex32::new(0.3, 0.7), Complex32::new(-0.5, 0.1)],
                [Complex32::new(0.2, -0.9), Complex32::new(0.4, 0.6)],
            ];
            let mut a = lcg(176, stride as u64 + 1);
            let mut b = a.clone();
            pair_pass(&mut a, stride, m);
            pair_scalar(&mut b, stride, m);
            for (x, y) in a.iter().zip(b.iter()) {
                assert!((x - y).norm() < 1e-5, "stride {stride}: {x} vs {y}");
            }
        }
    }

    #[test]
    fn diag1_matches_formula() {
        let d0 = Complex32::new(0.2, 0.9);
        let d1 = Complex32::new(-0.7, 0.3);
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
                    assert!((x - y).norm() < 1e-5, "q={q} off={off}");
                }
            }
        }
    }

    #[test]
    fn diag2_matches_formula() {
        let d = [
            Complex32::new(0.3, 0.4),
            Complex32::new(-0.2, 0.8),
            Complex32::new(0.9, -0.1),
            Complex32::new(0.5, 0.5),
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
                    assert!((x - y).norm() < 1e-5, "q1={q1} q2={q2} off={off}");
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
            assert!((x - y).norm() < 1e-7);
        }
        unsafe { swap_runs(a.as_mut_ptr(), 0, 8, 4) };
        for k in 0..4 {
            assert!((a[k] - orig[k + 8]).norm() < 1e-7);
            assert!((a[k + 8] - orig[k]).norm() < 1e-7);
        }
    }

    /// The f32 lane must track the f64 lane across every kernel family:
    /// fused 1q blocks, CX·D·CX rewrite, CNOT perm merge, Gray-code phase
    /// sweep, general 2q, 3q, X/diag shortcuts.
    #[test]
    fn f32_lane_matches_f64_lane() {
        use crate::ops::Parameter;
        let n = 6usize;
        let mut dag = QuantumDAG::new(n, 0);
        for q in 0..n {
            dag.add_op(OpType::H, &[q]);
            dag.add_op(OpType::Rx(Parameter::Const(0.31)), &[q]);
            dag.add_op(OpType::Ry(Parameter::Const(0.22)), &[q]);
            dag.add_op(OpType::Rz(Parameter::Const(0.17)), &[q]);
        }
        // CNOT chain → permutation merge sweep
        for q in 0..(n - 1) {
            dag.add_op(OpType::CNOT, &[q, q + 1]);
        }
        // CX·D·CX → diagonal-2q rewrite
        dag.add_op(OpType::CNOT, &[0, 2]);
        dag.add_op(OpType::T, &[2]);
        dag.add_op(OpType::CNOT, &[0, 2]);
        // Long diagonal run → Gray-code phase sweep
        for k in 0..40 {
            let q = k % n;
            dag.add_op(OpType::Rz(Parameter::Const(0.05 + 0.001 * k as f64)), &[q]);
        }
        // General 2q, SWAP, 3q and diagonal-2q tail paths
        dag.add_op(OpType::Rxx(Parameter::Const(0.4)), &[1, 3]);
        dag.add_op(OpType::SWAP, &[0, 4]);
        dag.add_op(OpType::CCX, &[0, 1, 2]);
        dag.add_op(OpType::CZ, &[2, 5]);

        let f64v = dag.simulate();
        let f32v = dag.simulate_f32();
        assert_eq!(f64v.len(), f32v.len());
        let diff = f64v
            .iter()
            .zip(f32v.iter())
            .map(|(x, y)| (*x - num_complex::Complex64::new(y.re as f64, y.im as f64)).norm())
            .fold(0.0f64, f64::max);
        assert!(diff < 1e-4, "f32 lane deviation {diff}");
    }

    /// The pooled scratch buffer must never change the state length when a
    /// larger simulation ran before a smaller one in the same process
    /// (both lanes share the pooling strategy).
    #[test]
    fn scratch_reuse_across_sizes() {
        let mut big = QuantumDAG::new(12, 0);
        big.add_op(OpType::H, &[0]);
        for q in 0..11 {
            big.add_op(OpType::CNOT, &[q, q + 1]);
        }
        assert_eq!(big.simulate().len(), 1 << 12);
        assert_eq!(big.simulate_f32().len(), 1 << 12);

        let mut small = QuantumDAG::new(6, 0);
        small.add_op(OpType::H, &[0]);
        for q in 0..5 {
            small.add_op(OpType::CNOT, &[q, q + 1]);
        }
        let r2 = small.simulate();
        let r2f = small.simulate_f32();
        assert_eq!(
            r2.len(),
            1 << 6,
            "f64 state length corrupted by pooled scratch"
        );
        assert_eq!(
            r2f.len(),
            1 << 6,
            "f32 state length corrupted by pooled scratch"
        );
        // The small circuit is a 6-qubit GHZ state: (|0..0> + |1..1>) / √2.
        let inv = 1.0 / 2f64.sqrt();
        assert!((r2[0].re - inv).abs() < 1e-9);
        assert!((r2[(1 << 6) - 1].re - inv).abs() < 1e-9);
        assert!((r2f[0].re as f64 - inv).abs() < 1e-4);
        assert!((r2f[(1 << 6) - 1].re as f64 - inv).abs() < 1e-4);
    }

    /// Regression: a fused 1q block of k=8 needs a 2^8 = 256-slots group
    /// buffer; the fixed 64-slot stack buffer of `apply_fused` only covered
    /// the old k≤6 fuse cap and panicked on the 20q VQC shape.
    #[test]
    fn fused_k8_matches_naive_reference() {
        let n = 12usize;
        let dim = 1usize << n;
        let k = 8usize;
        let qs: Vec<usize> = (3..3 + k).collect();
        let masks: Vec<usize> = qs.iter().map(|&q| 1usize << q).collect();
        let sub = 1usize << k;
        let mut u = Vec::with_capacity(4 * k);
        for t in 0..k {
            let a = 0.13 + 0.017 * t as f64;
            u.push(num_complex::Complex64::new(a.cos(), 0.0));
            u.push(num_complex::Complex64::new(-a.sin(), 0.31));
            u.push(num_complex::Complex64::new(a.sin(), -0.07));
            u.push(num_complex::Complex64::new(a.cos(), 0.11));
        }
        let mut got = lcg(dim, 99);
        let mut want = got.clone();

        apply_fused(&mut got, &qs, &u, &[], &[], false);

        let offs: Vec<usize> = (0..sub).map(|s| spread_bits(s, &masks)).collect();
        let mut buf = vec![Complex32::new(0.0, 0.0); sub];
        for g in 0..(dim >> k) {
            let base = deposit_bits_masked(g, &masks);
            for s in 0..sub {
                buf[s] = want[base | offs[s]];
            }
            for t in 0..k {
                let m = [[u[4 * t], u[4 * t + 1]], [u[4 * t + 2], u[4 * t + 3]]];
                let step = 1usize << t;
                let mut s = 0;
                while s < sub {
                    for x in s..s + step {
                        let a = buf[x];
                        let b = buf[x + step];
                        buf[x] = c32(m[0][0]) * a + c32(m[0][1]) * b;
                        buf[x + step] = c32(m[1][0]) * a + c32(m[1][1]) * b;
                    }
                    s += 2 * step;
                }
            }
            for s in 0..sub {
                want[base | offs[s]] = buf[s];
            }
        }
        let maxdiff = got
            .iter()
            .zip(want.iter())
            .map(|(x, y)| (*x - *y).norm())
            .fold(0.0f32, f32::max);
        assert!(maxdiff < 1e-4, "k=8 fused deviation {maxdiff}");
    }

    /// End-to-end twin of the kernel regression above: consecutive 1q gates
    /// on 8 distinct qubits make the planner emit k=8 fused blocks at n<23
    /// (fuse cap 8) — the exact shape that used to panic the f32 lane.
    #[test]
    fn f32_lane_handles_fuse_cap_block() {
        use crate::ops::Parameter;
        let n = 10usize;
        let mut dag = QuantumDAG::new(n, 0);
        for q in 0..8 {
            dag.add_op(OpType::H, &[q]);
        }
        for q in 0..8 {
            dag.add_op(OpType::Ry(Parameter::Const(0.19 + 0.02 * q as f64)), &[q]);
        }
        for q in 0..n - 1 {
            dag.add_op(OpType::CNOT, &[q, q + 1]);
        }
        let f64v = dag.simulate();
        let f32v = dag.simulate_f32();
        assert_eq!(f64v.len(), f32v.len());
        let diff = f64v
            .iter()
            .zip(f32v.iter())
            .map(|(x, y)| (*x - num_complex::Complex64::new(y.re as f64, y.im as f64)).norm())
            .fold(0.0f64, f64::max);
        assert!(diff < 1e-3, "f32 lane deviation {diff}");
    }

    /// `SF_PAIR_V2_F32` twin kernels must match the legacy kernels to float32
    /// rounding across the strides the production call sites use (same
    /// formula; the natural form commutes FMA operands on odd lanes, <=1 ulp).
    #[test]
    fn pair_v2_matches_legacy() {
        if !crate::simd::simd_enabled() {
            return;
        }
        for stride in [1usize, 2, 3, 4, 8, 11] {
            let m = [
                [Complex32::new(0.3, 0.7), Complex32::new(-0.5, 0.1)],
                [Complex32::new(0.2, -0.9), Complex32::new(0.4, 0.6)],
            ];
            let mut a = lcg(260, stride as u64 + 1);
            let mut b = a.clone();
            unsafe {
                if stride >= 2 {
                    pair_runs_avx2(b.as_mut_ptr(), b.len(), stride, m);
                    pair_runs_avx2_v2(a.as_mut_ptr(), a.len(), stride, m);
                } else {
                    pair_stride1_avx2(b.as_mut_ptr(), b.len(), m);
                    pair_stride1_avx2_v2(a.as_mut_ptr(), a.len(), m);
                }
            }
            for (x, y) in a.iter().zip(b.iter()) {
                let d = (*x - *y).norm();
                assert!(d < 2e-6, "stride {stride}: {x} vs {y}");
            }
        }
    }

    /// A/B the legacy and natural-layout f32 pair kernels: cache-busting
    /// buffer plus the fused per-group chunk shape. Run with `--nocapture`
    /// for the numbers.
    #[test]
    fn pair_v2_microbench_f32() {
        use std::time::Instant;
        if !crate::simd::simd_enabled() {
            return;
        }
        let n = 1usize << 22;
        let m = [
            [Complex32::new(0.3, 0.7), Complex32::new(-0.5, 0.1)],
            [Complex32::new(0.2, -0.9), Complex32::new(0.4, 0.6)],
        ];
        let base = lcg(n, 4242);
        for stride in [1usize, 2, 8, 32] {
            let mut a = base.clone();
            let mut b = base.clone();
            let (mut t_leg, mut t_new) = (f32::MAX, f32::MAX);
            for _ in 0..12 {
                let t0 = Instant::now();
                unsafe {
                    if stride >= 2 {
                        pair_runs_avx2(b.as_mut_ptr(), n, stride, m);
                    } else {
                        pair_stride1_avx2(b.as_mut_ptr(), n, m);
                    }
                }
                t_leg = t_leg.min(t0.elapsed().as_secs_f32());
                let t0 = Instant::now();
                unsafe {
                    if stride >= 2 {
                        pair_runs_avx2_v2(a.as_mut_ptr(), n, stride, m);
                    } else {
                        pair_stride1_avx2_v2(a.as_mut_ptr(), n, m);
                    }
                }
                t_new = t_new.min(t0.elapsed().as_secs_f32());
            }
            let mut worst = 0.0f32;
            for (x, y) in a.iter().zip(b.iter()) {
                worst = worst.max((*x - *y).norm() / x.norm().max(1e-3));
            }
            println!(
                "f32 stride {stride:2}: legacy {:7.2} ms  v2 {:7.2} ms  ({:.2}x)  max_rel_diff {worst:.2e}",
                t_leg * 1e3,
                t_new * 1e3,
                t_leg / t_new
            );
            assert!(worst < 2e-6, "v2 deviates from legacy: {worst}");
        }
        // 64-complex chunks (the fused per-group call shape), norm-preserving
        // matrix so repeated application does not overflow.
        let c = Complex32::new(0.6 / 2f32.sqrt(), 0.8 / 2f32.sqrt());
        let s = Complex32::new(0.8 / 2f32.sqrt(), -0.6 / 2f32.sqrt());
        let mu = [[c, -s], [s, c]];
        let base64 = lcg(64, 7);
        let mut a = base64.clone();
        let mut b = base64.clone();
        const REPS: usize = 40_000;
        let (mut t_leg, mut t_new) = (f32::MAX, f32::MAX);
        for _ in 0..10 {
            let t0 = Instant::now();
            for _ in 0..REPS {
                unsafe { pair_runs_avx2(b.as_mut_ptr(), 64, 4, mu) };
            }
            t_leg = t_leg.min(t0.elapsed().as_secs_f32());
            let t0 = Instant::now();
            for _ in 0..REPS {
                unsafe { pair_runs_avx2_v2(a.as_mut_ptr(), 64, 4, mu) };
            }
            t_new = t_new.min(t0.elapsed().as_secs_f32());
        }
        let mut worst = 0.0f32;
        for (x, y) in a.iter().zip(b.iter()) {
            worst = worst.max((*x - *y).norm());
        }
        println!(
            "f32 64-el chunk x{REPS}: legacy {:7.2} ms  v2 {:7.2} ms  ({:.2}x)  max_abs_diff {worst:.2e}",
            t_leg * 1e3,
            t_new * 1e3,
            t_leg / t_new
        );
        assert!(worst < 1e-4);
    }
}
