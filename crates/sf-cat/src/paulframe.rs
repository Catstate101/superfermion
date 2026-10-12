//! Pauli-frame bulk sampler — Lever 1 of the "beat them all in cat" program.
//!
//! Self-contained, additive sampler that lives entirely inside `sf-cat`
//! (sf-ir is not modified).  It re-implements the Aaronson–Gottesman
//! measurement plan with two deliberate changes over
//! `sf_ir::stabilizer::StabilizerTableau::sample`:
//!
//! 1. **Sparse variable lists.**  A determined outcome stores the *indices* of
//!    the drawn bits it depends on (`Vec<u32>`), not a dense `Vec<u64>` mask.
//!    Evaluation is one XOR per referenced variable, and no words have to be
//!    iterated for sparse locality-lightcone outcomes (GHZ: weight 1; the
//!    depth-8 brick: bounded by the light cone).
//! 2. **Packed-key counting.**  Shots are transposed into `n/64`-word integer
//!    keys and counted directly; the LE string is materialized once per
//!    *distinct* key instead of once per shot.  There is no per-shot `String`
//!    build and no per-shot char push.
//!
//! The tableau cell has **no 4096-qubit cap** (the product-path guard in
//! `superfermion/devices/rust_device.py` is Python-side; the sf-ir tableau
//! asserts `n <= 4096` internally, this cell does not).
//!
//! Correctness is pinned by tests against a naive independent statevector,
//! against `sf_ir::stabilizer::StabilizerTableau`, and by deterministic /
//! GHZ / mutation-cache cases (see `tests` below).

use std::collections::HashMap;
use std::hash::{BuildHasher, Hasher};
use std::sync::OnceLock;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

/// One precomputed measurement of the sampling plan.
#[derive(Debug, Clone)]
pub enum PEntry {
    /// Fresh random bit; `var` indexes the per-shot drawn-bit vector.
    Random { var: u32 },
    /// Parity of previously drawn bits (`vars`) XOR the constant `c & 1`.
    Determined { vars: Vec<u32>, c: u8 },
}

/// Minimal FxHash for packed `u64`-word keys (same rationale as sf-ir's
/// private counter hasher: no adversarial input, cheap word mixing).
#[derive(Default)]
struct FxHasher {
    hash: u64,
}

const FX_SEED: u64 = 0x51_7c_c1_b7_27_22_0a_95;

impl Hasher for FxHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.hash
    }
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.hash = (self.hash.rotate_left(5) ^ b as u64).wrapping_mul(FX_SEED);
        }
    }
    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.hash = (self.hash.rotate_left(5) ^ i).wrapping_mul(FX_SEED);
    }
}

#[derive(Clone, Default)]
struct FxBuildHasher;

impl BuildHasher for FxBuildHasher {
    type Hasher = FxHasher;
    #[inline]
    fn build_hasher(&self) -> FxHasher {
        FxHasher::default()
    }
}

/// Own stabilizer cell: rows `[0, n)` are destabilizers (init `X_q`), rows
/// `[n, 2n)` are stabilizers (init `Z_q`), all rows flattened to `words`
/// machine words.  Gate kernels mirror sf-ir's row-local updates bit for bit.
pub struct FrameTableau {
    n: usize,
    words: usize,
    x: Vec<u64>, // 2n * words
    z: Vec<u64>, // 2n * words
    r: Vec<u8>,  // 2n phase bits
    plan: OnceLock<Vec<PEntry>>,
}

impl Clone for FrameTableau {
    fn clone(&self) -> Self {
        Self {
            n: self.n,
            words: self.words,
            x: self.x.clone(),
            z: self.z.clone(),
            r: self.r.clone(),
            plan: OnceLock::new(),
        }
    }
}

#[inline(always)]
fn bit_pos(q: usize) -> (usize, u32, u64) {
    (q / 64, (q % 64) as u32, 1u64 << (q % 64))
}

/// Phase of the Pauli product `(x1,z1) * (x2,z2)` in units of `i` (mod 4).
/// Word-level popcount port of sf-ir's `phase_of_product`.
#[inline(always)]
fn phase_of_product(x1: &[u64], z1: &[u64], x2: &[u64], z2: &[u64], n: usize, words: usize) -> u8 {
    let mut phase: u32 = 0;
    for w in 0..words {
        let (xw1, zw1, xw2, zw2) = (x1[w], z1[w], x2[w], z2[w]);
        // A zero word in either row contributes no phase terms.
        if (xw1 | zw1) == 0 || (xw2 | zw2) == 0 {
            continue;
        }
        let c1 = (!xw1 & zw1 & xw2 & zw2) | (xw1 & !zw1 & !xw2 & zw2) | (xw1 & zw1 & xw2 & !zw2);
        let c3 = (!xw1 & zw1 & xw2 & !zw2) | (xw1 & !zw1 & xw2 & zw2) | (xw1 & zw1 & !xw2 & zw2);
        let mask = if w == words - 1 && !n.is_multiple_of(64) {
            (1u64 << (n % 64)) - 1
        } else {
            u64::MAX
        };
        phase = (phase + (c1 & mask).count_ones() + 3 * (c3 & mask).count_ones()) & 3;
    }
    phase as u8
}

#[inline]
fn row(arr: &[u64], i: usize, words: usize) -> &[u64] {
    &arr[i * words..(i + 1) * words]
}

/// Flat typed Clifford op for the fused row-major build.  CZ/CY/SX/SXDG are
/// expanded into these primitives at flatten time (mirrors sf-ir's `COp`).
#[derive(Clone, Copy)]
enum COp {
    H(u32),
    S(u32),
    Sdg(u32),
    X(u32),
    Y(u32),
    Z(u32),
    Cnot(u32, u32),
    Swap(u32, u32),
}

const PAR_MIN_DEFAULT: usize = 1 << 18;
const SAMPLE_PAR_MIN: usize = 1 << 20;

static PAR_MIN: OnceLock<usize> = OnceLock::new();

/// Parallel threshold for the bulk sampler (env-overridable once, for
/// calibration: `SF_CAT_SAMPLE_PAR_MIN`).  Work proxy = `shots * n`.
#[inline]
fn sample_par_min() -> usize {
    static SAMPLE_PAR: OnceLock<usize> = OnceLock::new();
    *SAMPLE_PAR.get_or_init(|| {
        std::env::var("SF_CAT_SAMPLE_PAR_MIN")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(SAMPLE_PAR_MIN)
    })
}

/// Plan-build phase trace switch (`SF_CAT_PLAN_TRACE=1`, cached once).
///
/// When set, every `build_plan` call prints one stderr line with the
/// per-phase millisecond breakdown (clone / init / column scans / phase /
/// xor / mask / deterministic branch).  Off by default; the clocks are only
/// read when the switch is on, so the hot loop is unchanged.
#[inline]
fn plan_trace_enabled() -> bool {
    static TRACE: OnceLock<bool> = OnceLock::new();
    *TRACE.get_or_init(|| std::env::var("SF_CAT_PLAN_TRACE").ok().as_deref() == Some("1"))
}

/// Accumulated per-phase timings for one `build_plan` call (trace mode).
#[derive(Default)]
struct PlanTrace {
    clone_ms: f64,
    init_ms: f64,
    gather_ms: f64,
    scan_ms: f64,
    det_ms: f64,
    phase_ms: f64,
    xor_ms: f64,
    mask_ms: f64,
    phase_calls: u64,
    hits: u64,
    rand_q: u64,
    det_q: u64,
    sub_alloc_ms: f64,
    sub_scan_ms: f64,
    sub_push_ms: f64,
}

#[inline(always)]
fn bitpos(q: u32) -> (usize, u32, u64) {
    let b = q & 63;
    ((q >> 6) as usize, b, 1u64 << b)
}

/// Apply the whole op list, in order, to ONE tableau row (fused row-major
/// build).  Semantics are identical to the per-gate methods.
#[inline]
fn apply_row(cops: &[COp], xr: &mut [u64], zr: &mut [u64], r: &mut u8) {
    for &op in cops {
        match op {
            COp::H(q) => {
                let (w, b, m) = bitpos(q);
                let xq = (xr[w] >> b) & 1;
                let zq = (zr[w] >> b) & 1;
                *r ^= (xq & zq) as u8;
                xr[w] = (xr[w] & !m) | (zq << b);
                zr[w] = (zr[w] & !m) | (xq << b);
            }
            COp::S(q) => {
                let (w, b, _) = bitpos(q);
                let xq = (xr[w] >> b) & 1;
                let zq = (zr[w] >> b) & 1;
                *r ^= (xq & zq) as u8;
                zr[w] ^= xq << b;
            }
            COp::Sdg(q) => {
                let (w, b, _) = bitpos(q);
                for _ in 0..3 {
                    let xq = (xr[w] >> b) & 1;
                    let zq = (zr[w] >> b) & 1;
                    *r ^= (xq & zq) as u8;
                    zr[w] ^= xq << b;
                }
            }
            COp::X(q) => {
                let (w, b, _) = bitpos(q);
                *r ^= ((zr[w] >> b) & 1) as u8;
            }
            COp::Z(q) => {
                let (w, b, _) = bitpos(q);
                *r ^= ((xr[w] >> b) & 1) as u8;
            }
            COp::Y(q) => {
                let (w, b, _) = bitpos(q);
                let xq = (xr[w] >> b) & 1;
                let zq = (zr[w] >> b) & 1;
                *r ^= (xq ^ zq) as u8;
            }
            COp::Cnot(a, b) => {
                let (wa, ba, _) = bitpos(a);
                let (wb, bb, _) = bitpos(b);
                let xa = (xr[wa] >> ba) & 1;
                let xb = (xr[wb] >> bb) & 1;
                let za = (zr[wa] >> ba) & 1;
                let zb = (zr[wb] >> bb) & 1;
                *r ^= (xa & zb & (xb ^ za ^ 1)) as u8;
                xr[wb] ^= xa << bb;
                zr[wa] ^= zb << ba;
            }
            COp::Swap(a, b) => {
                let (wa, ba, ma) = bitpos(a);
                let (wb, bb, mb) = bitpos(b);
                let xa = (xr[wa] >> ba) & 1;
                let xb = (xr[wb] >> bb) & 1;
                let za = (zr[wa] >> ba) & 1;
                let zb = (zr[wb] >> bb) & 1;
                xr[wa] = (xr[wa] & !ma) | (xb << ba);
                xr[wb] = (xr[wb] & !mb) | (xa << bb);
                zr[wa] = (zr[wa] & !ma) | (zb << ba);
                zr[wb] = (zr[wb] & !mb) | (za << bb);
            }
        }
    }
}

fn mask_to_vars(mask: &[u64]) -> Vec<u32> {
    let mut out = Vec::new();
    for (k, &m0) in mask.iter().enumerate() {
        let mut m = m0;
        while m != 0 {
            let j = m.trailing_zeros() as usize;
            m &= m - 1;
            out.push((64 * k + j) as u32);
        }
    }
    out
}

/// sf-ir's key convention: LE bitstring, leftmost char = highest qubit.
fn key_to_string(key: &[u64], n: usize) -> String {
    #[cfg(target_arch = "x86_64")]
    if n >= 64 && std::arch::is_x86_feature_detected!("bmi2") {
        // Safety: BMI2 availability is checked above.
        return unsafe { key_to_string_bmi2(key, n) };
    }
    key_to_string_scalar(key, n)
}

/// Portable set-bit walk (also the fallback when BMI2 is unavailable).
fn key_to_string_scalar(key: &[u64], n: usize) -> String {
    let mut buf = vec![b'0'; n];
    for (k, &word) in key.iter().enumerate() {
        let mut m = word;
        while m != 0 {
            let j = m.trailing_zeros() as usize;
            m &= m - 1;
            let q = 64 * k + j;
            buf[n - 1 - q] = b'1';
        }
    }
    String::from_utf8(buf).unwrap()
}

/// BMI2 fast path: each 8-bit chunk of a key word is expanded branchlessly
/// into 8 ASCII bytes with one `pdep` (bits -> one per byte), one add
/// (0x00/0x01 -> '0'/'1') and one `bswap` (string order is MSB-qubit-first).
/// Requires BMI2 (Haswell+); callers must check the feature.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "bmi2")]
unsafe fn key_to_string_bmi2(key: &[u64], n: usize) -> String {
    const ONES: u64 = 0x0101_0101_0101_0101;
    const ASCII: u64 = 0x3030_3030_3030_3030;
    let mut buf = vec![b'0'; n];
    let full_words = n / 64;
    for (k, &word) in key.iter().take(full_words).enumerate() {
        for j in 0..8usize {
            let b8 = (word >> (8 * j)) & 0xFF;
            // `_pdep_u64` is an unsafe fn; this fn is already unsafe.
            let bytes = core::arch::x86_64::_pdep_u64(b8, ONES).wrapping_add(ASCII);
            let i0 = n - 64 * k - 8 * j - 8;
            buf[i0..i0 + 8].copy_from_slice(&bytes.swap_bytes().to_le_bytes());
        }
    }
    let tail = n % 64;
    if tail != 0 {
        let mut m = key[full_words] & ((1u64 << tail) - 1);
        while m != 0 {
            let j = m.trailing_zeros() as usize;
            m &= m - 1;
            buf[n - 1 - (64 * full_words + j)] = b'1';
        }
    }
    String::from_utf8(buf).unwrap()
}

impl FrameTableau {
    pub fn new(n: usize) -> Self {
        assert!(n >= 1, "n must be >= 1");
        let words = n.div_ceil(64);
        let mut x = vec![0u64; 2 * n * words];
        let mut z = vec![0u64; 2 * n * words];
        for q in 0..n {
            let (w, _, m) = bit_pos(q);
            x[q * words + w] = m; // destabilizer q = X_q
            z[(n + q) * words + w] = m; // stabilizer q = Z_q
        }
        Self {
            n,
            words,
            x,
            z,
            r: vec![0u8; 2 * n],
            plan: OnceLock::new(),
        }
    }

    pub fn n(&self) -> usize {
        self.n
    }

    /// Machine words per row (and per packed shot key).
    pub fn words(&self) -> usize {
        self.words
    }

    #[inline]
    fn invalidate_plan(&mut self) {
        self.plan = OnceLock::new();
    }

    // ── Gate kernels (row-local, mirror sf-ir semantics) ──

    pub fn h(&mut self, q: usize) {
        self.invalidate_plan();
        let (w, b, m) = bit_pos(q);
        let words = self.words;
        for i in 0..2 * self.n {
            let o = i * words + w;
            let xq = (self.x[o] >> b) & 1;
            let zq = (self.z[o] >> b) & 1;
            self.r[i] ^= (xq & zq) as u8;
            self.x[o] = (self.x[o] & !m) | (zq << b);
            self.z[o] = (self.z[o] & !m) | (xq << b);
        }
    }

    pub fn s(&mut self, q: usize) {
        self.invalidate_plan();
        let (w, b, _) = bit_pos(q);
        let words = self.words;
        for i in 0..2 * self.n {
            let o = i * words + w;
            let xq = (self.x[o] >> b) & 1;
            let zq = (self.z[o] >> b) & 1;
            self.r[i] ^= (xq & zq) as u8;
            self.z[o] ^= xq << b;
        }
    }

    pub fn sdg(&mut self, q: usize) {
        self.s(q);
        self.s(q);
        self.s(q);
    }

    pub fn sx(&mut self, q: usize) {
        self.h(q);
        self.s(q);
        self.h(q);
    }

    pub fn sxdg(&mut self, q: usize) {
        self.h(q);
        self.sdg(q);
        self.h(q);
    }

    pub fn x_gate(&mut self, q: usize) {
        self.invalidate_plan();
        let (w, b, _) = bit_pos(q);
        let words = self.words;
        for i in 0..2 * self.n {
            self.r[i] ^= ((self.z[i * words + w] >> b) & 1) as u8;
        }
    }

    pub fn z_gate(&mut self, q: usize) {
        self.invalidate_plan();
        let (w, b, _) = bit_pos(q);
        let words = self.words;
        for i in 0..2 * self.n {
            self.r[i] ^= ((self.x[i * words + w] >> b) & 1) as u8;
        }
    }

    pub fn y_gate(&mut self, q: usize) {
        self.invalidate_plan();
        let (w, b, _) = bit_pos(q);
        let words = self.words;
        for i in 0..2 * self.n {
            let o = i * words + w;
            let xq = (self.x[o] >> b) & 1;
            let zq = (self.z[o] >> b) & 1;
            self.r[i] ^= (xq ^ zq) as u8;
        }
    }

    /// CNOT(a, b): control `a`, target `b`.
    pub fn cnot(&mut self, a: usize, b: usize) {
        self.invalidate_plan();
        let (wa, ba, _) = bit_pos(a);
        let (wb, bb, _) = bit_pos(b);
        let words = self.words;
        for i in 0..2 * self.n {
            let oa = i * words + wa;
            let ob = i * words + wb;
            let xa = (self.x[oa] >> ba) & 1;
            let xb = (self.x[ob] >> bb) & 1;
            let za = (self.z[oa] >> ba) & 1;
            let zb = (self.z[ob] >> bb) & 1;
            self.r[i] ^= (xa & zb & (xb ^ za ^ 1)) as u8;
            self.x[ob] ^= xa << bb;
            self.z[oa] ^= zb << ba;
        }
    }

    pub fn cz(&mut self, a: usize, b: usize) {
        self.h(b);
        self.cnot(a, b);
        self.h(b);
    }

    pub fn cy(&mut self, c: usize, t: usize) {
        self.sdg(t);
        self.cnot(c, t);
        self.s(t);
    }

    pub fn swap(&mut self, a: usize, b: usize) {
        self.invalidate_plan();
        let (wa, ba, ma) = bit_pos(a);
        let (wb, bb, mb) = bit_pos(b);
        let words = self.words;
        for i in 0..2 * self.n {
            let oa = i * words + wa;
            let ob = i * words + wb;
            let xa = (self.x[oa] >> ba) & 1;
            let xb = (self.x[ob] >> bb) & 1;
            let za = (self.z[oa] >> ba) & 1;
            let zb = (self.z[ob] >> bb) & 1;
            self.x[oa] = (self.x[oa] & !ma) | (xb << ba);
            self.x[ob] = (self.x[ob] & !mb) | (xa << bb);
            self.z[oa] = (self.z[oa] & !ma) | (zb << ba);
            self.z[ob] = (self.z[ob] & !mb) | (za << bb);
        }
    }

    /// Dispatch one named gate (sf-ir name vocabulary; MEASURE/BARRIER/RESET/
    /// ID are no-ops).  Case-insensitive.
    pub fn apply_gate(&mut self, name: &str, qubits: &[usize]) -> Result<(), String> {
        match name.to_uppercase().as_str() {
            "H" => self.h(qubits[0]),
            "S" => self.s(qubits[0]),
            "SDG" => self.sdg(qubits[0]),
            "SX" => self.sx(qubits[0]),
            "SXDG" => self.sxdg(qubits[0]),
            "X" => self.x_gate(qubits[0]),
            "Y" => self.y_gate(qubits[0]),
            "Z" => self.z_gate(qubits[0]),
            "CX" | "CNOT" => self.cnot(qubits[0], qubits[1]),
            "CZ" => self.cz(qubits[0], qubits[1]),
            "CY" => self.cy(qubits[0], qubits[1]),
            "SWAP" => self.swap(qubits[0], qubits[1]),
            "ID" | "BARRIER" | "MEASURE" | "RESET" => {}
            other => return Err(format!("Unsupported gate: {}", other)),
        }
        Ok(())
    }

    /// Build and evolve from a `(name, qubits)` list (same format sf-ir's
    /// `StabilizerTableau::from_gate_list` accepts).  Uses the fused
    /// row-major builder: each row consumes the whole op sequence once while
    /// its words stay cache-hot, rows split across rayon for large work.
    pub fn from_gate_list(n: usize, gates: &[(String, Vec<usize>)]) -> Result<Self, String> {
        let mut cops: Vec<COp> = Vec::with_capacity(gates.len());
        for (name, qubits) in gates {
            Self::flatten_gate(name, qubits, &mut cops)?;
        }
        Ok(Self::from_cops(n, &cops))
    }

    /// Flatten one named gate into typed ops (same vocabulary and
    /// decompositions as `apply_gate`).
    fn flatten_gate(name: &str, qubits: &[usize], out: &mut Vec<COp>) -> Result<(), String> {
        let q0 = qubits[0] as u32;
        match name.to_uppercase().as_str() {
            "H" => out.push(COp::H(q0)),
            "S" => out.push(COp::S(q0)),
            "SDG" => out.push(COp::Sdg(q0)),
            "SX" => out.extend([COp::H(q0), COp::S(q0), COp::H(q0)]),
            "SXDG" => out.extend([COp::H(q0), COp::Sdg(q0), COp::H(q0)]),
            "X" => out.push(COp::X(q0)),
            "Y" => out.push(COp::Y(q0)),
            "Z" => out.push(COp::Z(q0)),
            "CX" | "CNOT" => out.push(COp::Cnot(q0, qubits[1] as u32)),
            "CZ" => {
                let q1 = qubits[1] as u32;
                out.extend([COp::H(q1), COp::Cnot(q0, q1), COp::H(q1)]);
            }
            "CY" => {
                let q1 = qubits[1] as u32;
                out.extend([COp::Sdg(q1), COp::Cnot(q0, q1), COp::S(q1)]);
            }
            "SWAP" => out.push(COp::Swap(q0, qubits[1] as u32)),
            "ID" | "BARRIER" | "MEASURE" | "RESET" => {}
            other => return Err(format!("Unsupported gate: {}", other)),
        }
        Ok(())
    }

    fn from_cops(n: usize, cops: &[COp]) -> Self {
        let mut tab = Self::new(n);
        tab.apply_cops(cops);
        tab
    }

    /// Apply the flat op list with the fused row-major loop (sequential for
    /// small work, rayon row-chunks above `PAR_MIN_DEFAULT` — env override
    /// `SF_CAT_STAB_PAR_MIN`).
    fn apply_cops(&mut self, cops: &[COp]) {
        self.invalidate_plan();
        if cops.is_empty() {
            return;
        }
        let words = self.words;
        let rows = 2 * self.n;
        let work = rows.saturating_mul(cops.len());
        let par_min = *PAR_MIN.get_or_init(|| {
            std::env::var("SF_CAT_STAB_PAR_MIN")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(PAR_MIN_DEFAULT)
        });
        if work < par_min || rayon::current_num_threads() <= 1 {
            for i in 0..rows {
                let o = i * words;
                apply_row(
                    cops,
                    &mut self.x[o..o + words],
                    &mut self.z[o..o + words],
                    &mut self.r[i],
                );
            }
        } else {
            let threads = rayon::current_num_threads().max(1);
            let blk = rows.div_ceil(threads.saturating_mul(8)).max(32);
            self.x
                .par_chunks_mut(blk * words)
                .zip(self.z.par_chunks_mut(blk * words))
                .zip(self.r.par_chunks_mut(blk))
                .for_each(|((xs, zs), rs)| {
                    for i in 0..rs.len() {
                        let o = i * words;
                        apply_row(
                            cops,
                            &mut xs[o..o + words],
                            &mut zs[o..o + words],
                            &mut rs[i],
                        );
                    }
                });
        }
    }

    // ── Plan build (AG pass with GF(2) phase forms) ──

    /// Build the measurement plan for measuring every qubit in `Z` basis in
    /// order `q = 0..n`.  Structure mirrors sf-ir's `build_sample_plan` step
    /// for step, but emits sparse variable lists for determined outcomes.
    fn build_plan(&self) -> Vec<PEntry> {
        let n = self.n;
        let words = self.words;
        let mwords = n.div_ceil(64).max(1);
        let trace_on = plan_trace_enabled();
        let mut tr = PlanTrace::default();
        let t_all = std::time::Instant::now();
        let mut tab = self.clone();
        if trace_on {
            tr.clone_ms = t_all.elapsed().as_secs_f64() * 1e3;
        }
        let t_init = std::time::Instant::now();
        let mut r_mask: Vec<Vec<u64>> = vec![vec![0u64; mwords]; 2 * n];
        // One-sided flag: false => r_mask[i] is exactly zero (skip its XORs).
        let mut r_mask_nz: Vec<bool> = vec![false; 2 * n];
        let mut r_const: Vec<u8> = self.r.iter().map(|&v| v & 1).collect();
        let mut var_count: u32 = 0;
        let mut plan: Vec<PEntry> = Vec::with_capacity(n);
        // Column index: `colidx[w][c]` is the bitmask over rows (bit r =
        // row r) of x's column q = 64*w + c, built lazily per word from the
        // current tableau and invalidated en masse after every measure that
        // restructures rows.  Column queries then iterate set bits instead of
        // scanning one cache line per row (the measured plan bottleneck:
        // trace `scan_ms`/`det_ms` at n >= 1024).  Purely advisory: the
        // row-major tableau stays the single source of truth.
        let mut colidx: Vec<Option<Vec<Vec<u64>>>> = vec![None; words];
        if trace_on {
            tr.init_ms = t_init.elapsed().as_secs_f64() * 1e3;
        }

        for q in 0..n {
            let (w, _, qmask) = bit_pos(q);
            let t_q = trace_on.then(std::time::Instant::now);
            if colidx[w].is_none() {
                let t_g = trace_on.then(std::time::Instant::now);
                let nw = (2 * n).div_ceil(64);
                let mut cols = vec![vec![0u64; nw]; 64];
                for r in 0..2 * n {
                    let mut b = tab.x[r * words + w];
                    while b != 0 {
                        let c = b.trailing_zeros() as usize;
                        b &= b - 1;
                        cols[c][r >> 6] |= 1u64 << (r & 63);
                    }
                }
                colidx[w] = Some(cols);
                if let Some(t) = t_g {
                    tr.gather_ms += t.elapsed().as_secs_f64() * 1e3;
                }
            }

            // Random branch: first row p >= n with x[p][q] = 1 (bit index
            // q&63 of the column index for word w).
            let p_rand: Option<usize> = {
                let bs = &colidx[w].as_ref().unwrap()[q & 63];
                let mut found = None;
                for (k, &word) in bs.iter().enumerate() {
                    if k < n / 64 {
                        continue;
                    }
                    let mut m = word;
                    if k == n / 64 && !n.is_multiple_of(64) {
                        m &= !((1u64 << (n % 64)) - 1);
                    }
                    if m != 0 {
                        found = Some(k * 64 + m.trailing_zeros() as usize);
                        break;
                    }
                }
                found
            };

            if let Some(p) = p_rand {
                let mut hit_ms = 0f64;
                let rp_mask = r_mask[p].clone();
                let rp_nz = r_mask_nz[p];
                let rp_const = r_const[p];
                let xp = row(&tab.x, p, words).to_vec();
                let zp = row(&tab.z, p, words).to_vec();

                let bs_hits: Vec<u64> = std::mem::take(&mut colidx[w].as_mut().unwrap()[q & 63]);
                for (k, &word) in bs_hits.iter().enumerate() {
                    let mut m = word;
                    while m != 0 {
                        let i = k * 64 + m.trailing_zeros() as usize;
                        m &= m - 1;
                        if i == p {
                            continue;
                        }
                        let tp = trace_on.then(std::time::Instant::now);
                        let popp = phase_of_product(
                            row(&tab.x, i, words),
                            row(&tab.z, i, words),
                            &xp,
                            &zp,
                            n,
                            words,
                        );
                        if let Some(t) = tp {
                            let e = t.elapsed().as_secs_f64() * 1e3;
                            tr.phase_ms += e;
                            hit_ms += e;
                            tr.phase_calls += 1;
                        }
                        let tm = trace_on.then(std::time::Instant::now);
                        if rp_nz {
                            for k in 0..mwords {
                                r_mask[i][k] ^= rp_mask[k];
                            }
                            r_mask_nz[i] = true;
                        }
                        r_const[i] ^= rp_const ^ ((popp >> 1) & 1);
                        if let Some(t) = tm {
                            let e = t.elapsed().as_secs_f64() * 1e3;
                            tr.mask_ms += e;
                            hit_ms += e;
                        }
                        let tx = trace_on.then(std::time::Instant::now);
                        for ww in 0..words {
                            let (xw, zw) = (xp[ww], zp[ww]);
                            if (xw | zw) == 0 {
                                continue;
                            }
                            tab.x[i * words + ww] ^= xw;
                            tab.z[i * words + ww] ^= zw;
                        }
                        if let Some(t) = tx {
                            let e = t.elapsed().as_secs_f64() * 1e3;
                            tr.xor_ms += e;
                            hit_ms += e;
                            tr.hits += 1;
                        }
                    }
                }

                // Copy old row p into the destabilizer slot p-n, then set the
                // stabilizer slot p to Z_q with the fresh variable as phase.
                for ww in 0..words {
                    tab.x[(p - n) * words + ww] = tab.x[p * words + ww];
                    tab.z[(p - n) * words + ww] = tab.z[p * words + ww];
                    tab.x[p * words + ww] = 0;
                    tab.z[p * words + ww] = 0;
                }
                r_mask.swap(p - n, p);
                r_mask_nz.swap(p - n, p);
                r_const.swap(p - n, p);
                tab.r.swap(p - n, p);
                tab.z[p * words + w] = qmask;

                let var = var_count;
                var_count += 1;
                plan.push(PEntry::Random { var });
                r_mask[p].fill(0);
                r_mask[p][(var as usize) / 64] = 1u64 << ((var as usize) % 64);
                r_mask_nz[p] = true;
                r_const[p] = 0;
                // The measure restructured rows; every word's column index is
                // stale from here (lazily rebuilt on next use).
                for c in colidx.iter_mut() {
                    *c = None;
                }
                tr.rand_q += 1;
                if let Some(t) = t_q {
                    let e = t.elapsed().as_secs_f64() * 1e3 - hit_ms;
                    tr.scan_ms += e;
                }
            } else {
                // Deterministic branch: accumulate destabilizer rows with
                // x[i][q] = 1 exactly like the AG deterministic measure.
                let mut hit_ms = 0f64;
                let t_alloc = trace_on.then(std::time::Instant::now);
                let mut sx = vec![0u64; words];
                let mut sz = vec![0u64; words];
                let mut s_mask = vec![0u64; mwords];
                let mut s_const: u8 = 0;
                let mut s_bit0: u8 = 0;
                if let Some(t) = t_alloc {
                    tr.sub_alloc_ms += t.elapsed().as_secs_f64() * 1e3;
                }
                let t_scan = trace_on.then(std::time::Instant::now);
                let det_bs = &colidx[w].as_ref().unwrap()[q & 63];
                let k_last = n / 64;
                let tail_mask = if !n.is_multiple_of(64) {
                    (1u64 << (n % 64)) - 1
                } else {
                    u64::MAX
                };
                for (k, &word) in det_bs.iter().take(k_last + 1).enumerate() {
                    let mut m = word;
                    if k == k_last && n.is_multiple_of(64) {
                        break;
                    }
                    if k == k_last {
                        m &= tail_mask;
                    }
                    while m != 0 {
                        let i = k * 64 + m.trailing_zeros() as usize;
                        m &= m - 1;
                        let si = i + n;
                        let tp = trace_on.then(std::time::Instant::now);
                        let popp = phase_of_product(
                            &sx,
                            &sz,
                            row(&tab.x, si, words),
                            row(&tab.z, si, words),
                            n,
                            words,
                        );
                        if let Some(t) = tp {
                            let e = t.elapsed().as_secs_f64() * 1e3;
                            tr.phase_ms += e;
                            hit_ms += e;
                            tr.phase_calls += 1;
                        }
                        let tm = trace_on.then(std::time::Instant::now);
                        let (p0, popp1) = (popp & 1, (popp >> 1) & 1);
                        let carry = s_bit0 & p0;
                        if r_mask_nz[si] {
                            for k in 0..mwords {
                                s_mask[k] ^= r_mask[si][k];
                            }
                        }
                        s_const ^= r_const[si] ^ popp1 ^ carry;
                        s_bit0 = (s_bit0 + p0) & 1;
                        if let Some(t) = tm {
                            let e = t.elapsed().as_secs_f64() * 1e3;
                            tr.mask_ms += e;
                            hit_ms += e;
                        }
                        let tx = trace_on.then(std::time::Instant::now);
                        for ww in 0..words {
                            let (xw, zw) = (tab.x[si * words + ww], tab.z[si * words + ww]);
                            if (xw | zw) == 0 {
                                continue;
                            }
                            sx[ww] ^= xw;
                            sz[ww] ^= zw;
                        }
                        if let Some(t) = tx {
                            let e = t.elapsed().as_secs_f64() * 1e3;
                            tr.xor_ms += e;
                            hit_ms += e;
                            tr.hits += 1;
                        }
                    }
                }
                if let Some(t) = t_scan {
                    tr.sub_scan_ms += t.elapsed().as_secs_f64() * 1e3;
                }
                let t_push = trace_on.then(std::time::Instant::now);
                plan.push(PEntry::Determined {
                    vars: mask_to_vars(&s_mask),
                    c: s_const & 1,
                });
                if let Some(t) = t_push {
                    tr.sub_push_ms += t.elapsed().as_secs_f64() * 1e3;
                }
                tr.det_q += 1;
                if let Some(t) = t_q {
                    let e = t.elapsed().as_secs_f64() * 1e3 - hit_ms;
                    tr.det_ms += e;
                }
            }
        }
        if trace_on {
            eprintln!(
                "SF_CAT_PLAN_TRACE n={} rand_q={} det_q={} clone_ms={:.1} init_ms={:.1} \
                 gather_ms={:.1} scan_ms={:.1} det_ms={:.1} phase_ms={:.1} phase_calls={} \
                 xor_ms={:.1} mask_ms={:.1} hits={} total_ms={:.1} \
                 sub_alloc_ms={:.1} sub_scan_ms={:.1} sub_push_ms={:.1}",
                n,
                tr.rand_q,
                tr.det_q,
                tr.clone_ms,
                tr.init_ms,
                tr.gather_ms,
                tr.scan_ms,
                tr.det_ms,
                tr.phase_ms,
                tr.phase_calls,
                tr.xor_ms,
                tr.mask_ms,
                tr.hits,
                t_all.elapsed().as_secs_f64() * 1e3,
                tr.sub_alloc_ms,
                tr.sub_scan_ms,
                tr.sub_push_ms
            );
        }
        plan
    }

    /// Number of fresh random variables in the (cached) plan — the number of
    /// genuinely random measurement outcomes.
    pub fn random_var_count(&self) -> usize {
        let plan = self.plan.get_or_init(|| self.build_plan());
        plan.iter()
            .filter(|e| matches!(e, PEntry::Random { .. }))
            .count()
    }

    // ── Bulk sampling ──

    /// Sample `shots` bitstrings, shot i packed into `words` little-endian
    /// words at `[i*words, (i+1)*words)`: bit `q` = outcome of qubit `q`.
    pub fn sample_packed(&self, shots: usize, seed: Option<u64>) -> Vec<u64> {
        if shots == 0 {
            return Vec::new();
        }
        let words = self.words;
        let n = self.n;
        let plan = self.plan.get_or_init(|| self.build_plan());
        let var_count = plan
            .iter()
            .map(|e| match e {
                PEntry::Random { var } => *var as usize + 1,
                PEntry::Determined { .. } => 0,
            })
            .max()
            .unwrap_or(0);
        let base_seed = seed.unwrap_or_else(rand::random);
        const BLOCK: usize = 64;
        let nblocks = shots.div_ceil(BLOCK);

        let gen_block = |packed: &mut [u64], b: usize| {
            let cnt = (shots - b * BLOCK).min(BLOCK);
            let mut rng = StdRng::seed_from_u64(base_seed.wrapping_add(b as u64));
            let mut vars = vec![0u64; var_count];
            rng.fill(&mut vars[..]);
            // Evaluate the plan into one u64 outcome lane per qubit.
            let mut out = vec![0u64; n];
            for (q, e) in plan.iter().enumerate() {
                out[q] = match e {
                    PEntry::Random { var } => vars[*var as usize],
                    PEntry::Determined { vars: vv, c } => {
                        let mut lane: u64 = if *c & 1 != 0 { u64::MAX } else { 0 };
                        for &v in vv {
                            lane ^= vars[v as usize];
                        }
                        lane
                    }
                };
            }
            // Fast path: all qubit lanes identical (e.g. GHZ).  Bit q of a
            // shot key then equals one SINGLE lane bit for every q < n, so
            // the packed key is that lane broadcast over the n-bit mask.
            if out.iter().all(|&l| l == out[0]) {
                let lane = out[0];
                let tail = if n.is_multiple_of(64) {
                    u64::MAX
                } else {
                    (1u64 << (n % 64)) - 1
                };
                let mut masks = vec![u64::MAX; words];
                if let Some(last) = masks.last_mut() {
                    *last = tail;
                }
                for i in 0..cnt {
                    if (lane >> i) & 1 != 0 {
                        packed[i * words..(i + 1) * words].copy_from_slice(&masks);
                    }
                }
            } else if !crate::kernels::transpose(&out, packed, words, cnt) {
                // Transpose the block (64 lanes x n bits) into packed keys.
                // Branchless per bit: dense random lanes mispredict a
                // `if bit` form ~50% of the time.
                // When the additive C++ lane is built and selected
                // (`SF_CAT_KERNEL`), it already filled `packed` above; this
                // loop remains the always-available Rust fallback.
                for (q, lane) in out.iter().enumerate() {
                    let sh = q % 64;
                    let wq = q / 64;
                    for i in 0..cnt {
                        packed[i * words + wq] |= ((*lane >> i) & 1) << sh;
                    }
                }
            }
        };

        let mut packed = vec![0u64; shots * words];
        // Fork/join only pays off past a work threshold; below it the block
        // loop runs sequentially in the calling thread.
        let workers = rayon::current_num_threads();
        if workers <= 1 || shots.saturating_mul(n) < sample_par_min() {
            for b in 0..nblocks {
                let cnt = (shots - b * BLOCK).min(BLOCK);
                let s = b * BLOCK * words;
                gen_block(&mut packed[s..s + cnt * words], b);
            }
        } else {
            packed
                .par_chunks_mut(BLOCK * words)
                .enumerate()
                .for_each(|(b, chunk)| gen_block(chunk, b));
        }
        packed
    }

    /// Sample and return the counts map with sf-ir's key convention
    /// (LE string, leftmost char = highest qubit).  Strings are materialized
    /// once per *distinct* key.
    pub fn sample(&self, shots: usize, seed: Option<u64>) -> HashMap<String, usize> {
        if shots == 0 {
            return HashMap::new();
        }
        let words = self.words;
        let n = self.n;
        let packed = self.sample_packed(shots, seed);
        let workers = rayon::current_num_threads();

        if words == 1 {
            // n <= 64: the whole key is one u64; count without any Vec, and
            // materialize strings straight from the integer map.
            let mut wcounts: HashMap<u64, usize, FxBuildHasher> =
                HashMap::with_hasher(FxBuildHasher);
            for &w in packed.iter() {
                *wcounts.entry(w).or_insert(0) += 1;
            }
            let entries: Vec<(u64, usize)> = wcounts.into_iter().collect();
            let map_key = |(w, c): (u64, usize)| -> (String, usize) { (key_to_string(&[w], n), c) };
            if workers <= 1 || n.saturating_mul(entries.len()) < sample_par_min() {
                entries.into_iter().map(map_key).collect()
            } else {
                entries.into_par_iter().map(map_key).collect()
            }
        } else {
            // `get_mut` first so repeat keys cost no allocation.
            let mut counts: HashMap<Vec<u64>, usize, FxBuildHasher> =
                HashMap::with_hasher(FxBuildHasher);
            for i in 0..shots {
                let key = &packed[i * words..(i + 1) * words];
                match counts.get_mut(key) {
                    Some(c) => *c += 1,
                    None => {
                        counts.insert(key.to_vec(), 1);
                    }
                }
            }
            // Materialize key strings once per distinct key (parallel for bulk).
            let entries: Vec<(Vec<u64>, usize)> = counts.into_iter().collect();
            let map_key =
                |(key, c): (Vec<u64>, usize)| -> (String, usize) { (key_to_string(&key, n), c) };
            if workers <= 1 || n.saturating_mul(entries.len()) < sample_par_min() {
                entries.into_iter().map(map_key).collect()
            } else {
                entries.into_par_iter().map(map_key).collect()
            }
        }
    }

    /// Exact joint-outcome probability of one packed key (test support):
    /// only used to sanity-check deterministic plans.
    #[doc(hidden)]
    pub fn is_deterministic_plan(&self) -> bool {
        let plan = self.plan.get_or_init(|| self.build_plan());
        plan.iter().all(|e| matches!(e, PEntry::Determined { .. }))
    }
}

// ═══════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;

    fn ops(list: &[(&str, &[usize])]) -> Vec<(String, Vec<usize>)> {
        list.iter()
            .map(|(n, q)| (n.to_string(), q.to_vec()))
            .collect()
    }

    /// Independent naive statevector (LE: bit q of index = qubit q).
    fn naive_statevector(n: usize, gates: &[(String, Vec<usize>)]) -> Vec<Complex64> {
        let dim = 1usize << n;
        let mut psi = vec![Complex64::new(0.0, 0.0); dim];
        psi[0] = Complex64::new(1.0, 0.0);
        let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;
        for (name, q) in gates {
            match name.to_uppercase().as_str() {
                "H" => {
                    let (w, b) = (q[0] / 64, q[0] % 64);
                    let _ = (w, b);
                    let m = 1usize << q[0];
                    for i in 0..dim {
                        if i & m == 0 {
                            let (a, b2) = (psi[i], psi[i | m]);
                            psi[i] = (a + b2) * inv_sqrt2;
                            psi[i | m] = (a - b2) * inv_sqrt2;
                        }
                    }
                }
                "S" => {
                    let m = 1usize << q[0];
                    for i in 0..dim {
                        if i & m != 0 {
                            psi[i] *= Complex64::new(0.0, 1.0);
                        }
                    }
                }
                "SDG" => {
                    let m = 1usize << q[0];
                    for i in 0..dim {
                        if i & m != 0 {
                            psi[i] *= Complex64::new(0.0, -1.0);
                        }
                    }
                }
                "X" => {
                    let m = 1usize << q[0];
                    for i in 0..dim {
                        if i & m == 0 {
                            psi.swap(i, i | m);
                        }
                    }
                }
                "Z" => {
                    let m = 1usize << q[0];
                    for i in 0..dim {
                        if i & m != 0 {
                            psi[i] = -psi[i];
                        }
                    }
                }
                "Y" => {
                    let m = 1usize << q[0];
                    for i in 0..dim {
                        if i & m == 0 {
                            let (a, b2) = (psi[i], psi[i | m]);
                            psi[i] = Complex64::new(0.0, -1.0) * b2;
                            psi[i | m] = Complex64::new(0.0, 1.0) * a;
                        }
                    }
                }
                "CX" | "CNOT" => {
                    let mc = 1usize << q[0];
                    let mt = 1usize << q[1];
                    for i in 0..dim {
                        if i & mc != 0 && i & mt == 0 {
                            psi.swap(i, i | mt);
                        }
                    }
                }
                "CZ" => {
                    let ma = 1usize << q[0];
                    let mb = 1usize << q[1];
                    for i in 0..dim {
                        if i & ma != 0 && i & mb != 0 {
                            psi[i] = -psi[i];
                        }
                    }
                }
                "SWAP" => {
                    let ma = 1usize << q[0];
                    let mb = 1usize << q[1];
                    for i in 0..dim {
                        if i & ma == 0 && i & mb != 0 {
                            psi.swap(i, (i | ma) & !mb);
                        }
                    }
                }
                other => panic!("naive sv: unsupported {other}"),
            }
        }
        psi
    }

    fn exact_probs(n: usize, gates: &[(String, Vec<usize>)]) -> Vec<f64> {
        naive_statevector(n, gates)
            .iter()
            .map(|a| a.norm_sqr())
            .collect()
    }

    fn counts_to_probs(counts: &HashMap<String, usize>, shots: usize) -> HashMap<usize, f64> {
        counts
            .iter()
            .map(|(k, c)| {
                let idx = usize::from_str_radix(k, 2).unwrap();
                (idx, *c as f64 / shots as f64)
            })
            .collect()
    }

    fn tvd(counts: &HashMap<String, usize>, exact: &[f64], shots: usize) -> f64 {
        let probs = counts_to_probs(counts, shots);
        let mut total = 0.0;
        for (i, p) in exact.iter().enumerate() {
            let q = probs.get(&i).copied().unwrap_or(0.0);
            total += (p - q).abs();
        }
        // States present in samples but not in exact support.
        for (i, q) in probs.iter() {
            if *i >= exact.len() {
                total += q.abs();
            }
        }
        0.5 * total
    }

    fn tvd_threshold(k: usize, shots: usize) -> f64 {
        4.0 * (k as f64 / (2.0 * std::f64::consts::PI * shots as f64)).sqrt()
    }

    fn brick(n: usize, depth: usize, seed: u64) -> Vec<(String, Vec<usize>)> {
        // Seeded H/CX/S brick identical in shape to sim4_bench_suite's
        // `brick_ops` (independent RNG here: tests only need a deterministic
        // local stream, not the campaign's numpy stream).
        let mut gates: Vec<(String, Vec<usize>)> = Vec::new();
        let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for _ in 0..depth {
            for q in 0..n {
                gates.push(("H".into(), vec![q]));
            }
            for q in (0..n - 1).step_by(2) {
                gates.push(("CX".into(), vec![q, q + 1]));
            }
            for q in 0..n {
                if next() & 1 == 1 {
                    gates.push(("S".into(), vec![q]));
                }
            }
            for q in (1..n - 1).step_by(2) {
                gates.push(("CX".into(), vec![q, q + 1]));
            }
        }
        gates
    }

    fn ghz(n: usize) -> Vec<(String, Vec<usize>)> {
        let mut gates = vec![("H".to_string(), vec![0])];
        for i in 1..n {
            gates.push(("CX".to_string(), vec![0, i]));
        }
        gates
    }

    #[test]
    fn deterministic_circuit_is_bit_exact() {
        let gates = ops(&[("X", &[0]), ("CX", &[0, 1])]);
        let tab = FrameTableau::from_gate_list(2, &gates).unwrap();
        let counts = tab.sample(64, Some(5));
        assert_eq!(counts.len(), 1);
        assert_eq!(counts.values().sum::<usize>(), 64);
        assert_eq!(counts.get("11"), Some(&64));
    }

    #[test]
    fn ghz_has_only_two_branches() {
        let n = 64;
        let tab = FrameTableau::from_gate_list(n, &ghz(n)).unwrap();
        let shots = 20_000;
        let counts = tab.sample(shots, Some(11));
        assert_eq!(counts.len(), 2, "GHZ must collapse to two branches");
        let zeros = "0".repeat(n);
        let ones = "1".repeat(n);
        let c0 = *counts.get(&zeros).unwrap_or(&0);
        let c1 = *counts.get(&ones).unwrap_or(&0);
        assert_eq!(c0 + c1, shots);
        let sigma = 5.0 * (shots as f64 / 4.0).sqrt();
        assert!(
            ((c0 as f64) - shots as f64 / 2.0).abs() <= sigma,
            "GHZ branch imbalance: {c0}/{c1}"
        );
    }

    #[test]
    fn matches_exact_statevector_tvd() {
        for &n in &[6usize, 8, 10] {
            let gates = brick(n, 2, 11);
            let exact = exact_probs(n, &gates);
            let support = exact.iter().filter(|p| **p > 1e-12).count();
            let tab = FrameTableau::from_gate_list(n, &gates).unwrap();
            let shots = 40_000;
            let counts = tab.sample(shots, Some(7));
            let dist = tvd(&counts, &exact, shots);
            let thr = tvd_threshold(support, shots);
            assert!(
                dist < thr,
                "n={n}: TVD {dist:.5} >= threshold {thr:.5} (support {support})"
            );
        }
    }

    #[test]
    fn matches_sf_ir_tableau_distribution() {
        let n = 64;
        let gates = brick(n, 3, 23);
        let ours = FrameTableau::from_gate_list(n, &gates).unwrap();
        let theirs = sf_ir::stabilizer::StabilizerTableau::from_gate_list(n, &gates).unwrap();
        let shots = 30_000;
        let c_ours = ours.sample(shots, Some(3));
        let c_theirs = theirs.sample(shots, Some(4));
        let p_ours = counts_to_probs(&c_ours, shots);
        let p_theirs = counts_to_probs(&c_theirs, shots);
        let mut keys: Vec<usize> = p_ours.keys().chain(p_theirs.keys()).copied().collect();
        keys.sort_unstable();
        keys.dedup();
        let mut total = 0.0;
        for k in keys {
            let a = p_ours.get(&k).copied().unwrap_or(0.0);
            let b = p_theirs.get(&k).copied().unwrap_or(0.0);
            total += (a - b).abs();
        }
        let dist = 0.5 * total;
        // Two independent samplers of the same distribution: each has its own
        // multinomial noise; threshold covers both.
        let k_eff = p_ours.len().max(p_theirs.len());
        let thr = 2.0 * tvd_threshold(k_eff.max(1), shots);
        assert!(
            dist < thr,
            "ours-vs-sf-ir TVD {dist:.5} >= threshold {thr:.5}"
        );
    }

    #[test]
    fn plan_cache_invalidated_on_mutation() {
        let mut tab = FrameTableau::new(4);
        tab.h(0);
        tab.cnot(0, 1);
        let _ = tab.sample(128, Some(3)); // populate the cache
        tab.h(2);
        tab.cnot(2, 3);
        let mutated = tab.sample(256, Some(9));

        let gates = ops(&[("H", &[0]), ("CX", &[0, 1]), ("H", &[2]), ("CX", &[2, 3])]);
        let fresh = FrameTableau::from_gate_list(4, &gates).unwrap();
        let fresh_counts = fresh.sample(256, Some(9));
        assert_eq!(mutated, fresh_counts);
    }

    #[test]
    fn packed_and_counts_paths_agree() {
        let n = 70; // crosses the 64-bit boundary
        let gates = brick(n, 2, 5);
        let tab = FrameTableau::from_gate_list(n, &gates).unwrap();
        let shots = 500;
        let packed = tab.sample_packed(shots, Some(17));
        let words = tab.words;
        assert_eq!(packed.len(), shots * words);
        let mut manual: HashMap<String, usize> = HashMap::new();
        for i in 0..shots {
            let key = &packed[i * words..(i + 1) * words];
            let mut buf = vec![b'0'; n];
            for (k, &word) in key.iter().enumerate() {
                let mut m = word;
                while m != 0 {
                    let j = m.trailing_zeros() as usize;
                    m &= m - 1;
                    let q = 64 * k + j;
                    buf[n - 1 - q] = b'1';
                }
            }
            *manual.entry(String::from_utf8(buf).unwrap()).or_insert(0) += 1;
        }
        let counts = tab.sample(shots, Some(17));
        assert_eq!(counts, manual);
    }

    #[test]
    fn same_seed_is_deterministic() {
        let n = 32;
        let gates = brick(n, 2, 9);
        let tab = FrameTableau::from_gate_list(n, &gates).unwrap();
        let a = tab.sample_packed(300, Some(1234));
        let b = tab.sample_packed(300, Some(1234));
        assert_eq!(a, b);
    }

    #[test]
    fn no_4096_cap() {
        // The product path refuses n > 4096; this cell must not.
        let n = 4097;
        let mut tab = FrameTableau::new(n);
        tab.h(0);
        tab.cnot(0, n - 1);
        let packed = tab.sample_packed(8, Some(1));
        assert_eq!(packed.len(), 8 * tab.words);
    }

    #[test]
    fn deterministic_plan_flag() {
        let gates = ops(&[("X", &[0])]);
        let tab = FrameTableau::from_gate_list(1, &gates).unwrap();
        assert!(tab.is_deterministic_plan());
        let gates = ops(&[("H", &[0])]);
        let tab = FrameTableau::from_gate_list(1, &gates).unwrap();
        assert!(!tab.is_deterministic_plan());
        assert_eq!(tab.random_var_count(), 1);
    }

    #[test]
    fn bmi2_key_to_string_matches_scalar() {
        let scalar = |key: &[u64], n: usize| -> String {
            let mut buf = vec![b'0'; n];
            for (k, &word) in key.iter().enumerate() {
                let mut m = word;
                while m != 0 {
                    let j = m.trailing_zeros() as usize;
                    m &= m - 1;
                    buf[n - 1 - (64 * k + j)] = b'1';
                }
            }
            String::from_utf8(buf).unwrap()
        };
        let mut rng = StdRng::seed_from_u64(42);
        for &n in &[64usize, 65, 100, 127, 128, 256, 1024, 4096, 4097] {
            let words = n.div_ceil(64);
            for _ in 0..8 {
                let mut key = vec![0u64; words];
                rng.fill(&mut key[..]);
                if !n.is_multiple_of(64) {
                    let tail_mask = (1u64 << (n % 64)) - 1;
                    if let Some(last) = key.last_mut() {
                        *last &= tail_mask;
                    }
                }
                let expected = scalar(&key, n);
                assert_eq!(key_to_string(&key, n), expected, "n={n}");
                #[cfg(target_arch = "x86_64")]
                if std::arch::is_x86_feature_detected!("bmi2") {
                    // Safety: BMI2 availability checked at runtime.
                    let via_bmi2 = unsafe { key_to_string_bmi2(&key, n) };
                    assert_eq!(via_bmi2, expected, "n={n} (bmi2 kernel)");
                }
            }
        }
    }

    #[cfg(sf_cat_cxx)]
    #[test]
    fn cpp_pack_kernels_match_scalar_transpose() {
        // Bit-exactness gate for the additive C++ lane: both kernels must
        // reproduce the Rust scalar transpose of `sample_packed` for full,
        // partial and multi-word qubit counts.
        let reference = |lanes: &[u64], words: usize, cnt: usize| -> Vec<u64> {
            let mut p = vec![0u64; cnt * words];
            for (q, lane) in lanes.iter().enumerate() {
                let sh = q % 64;
                let wq = q / 64;
                for i in 0..cnt {
                    p[i * words + wq] |= ((*lane >> i) & 1) << sh;
                }
            }
            p
        };
        let mut rng = StdRng::seed_from_u64(99);
        for &n in &[64usize, 65, 128, 256, 1024] {
            let words = n.div_ceil(64);
            let mut lanes = vec![0u64; n];
            rng.fill(&mut lanes[..]);
            for &cnt in &[64usize, 37] {
                let want = reference(&lanes, words, cnt);
                for kernel in ["literal", "ds"] {
                    let mut got = vec![0u64; cnt * words];
                    crate::kernels::transpose_with(kernel, &lanes, &mut got, words, cnt);
                    assert_eq!(got, want, "n={n} cnt={cnt} kernel={kernel}");
                }
            }
        }
    }
}
