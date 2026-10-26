//! Clifford stabilizer tableau (Aaronson–Gottesman 2004) + standalone Pauli twirl.
//!
//! `StabilizerTableau` — O(n) gate updates, O(n³) sampling via AG algorithm 1.
//! Word-packed representation (ceil(n/64) u64s per row) supports n ≤ 4096.
//! `pauli_twirl_gate_list` — standalone gate-level Pauli twirl, no DAG dependency.

use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::OnceLock;

use crate::{OpType, QuantumDAG};

// ═══════════════════════════════════════════════════════════
// Stabilizer Tableau (word-packed for n ≤ 4096)
// ═══════════════════════════════════════════════════════════

pub struct StabilizerTableau {
    pub n: usize,
    words: usize,     // ceil(n / 64)
    x: Vec<Vec<u64>>, // 2n rows, each with `words` u64s
    z: Vec<Vec<u64>>,
    r: Vec<u64>, // 2n phase bits (0 or 1)
    /// Scratch buffers reused across measure_z calls to avoid per-call heap alloc.
    sx_buf: Vec<u64>, // words elements, zeroed before each measure_z
    sz_buf: Vec<u64>, // words elements, zeroed before each measure_z
    /// Cached output of `build_sample_plan()`.
    ///
    /// The plan is a pure function of the tableau, so it survives across
    /// `sample()` calls; every mutator drops it via `invalidate_plan()`
    /// (replacing the cell with a fresh `OnceLock`).
    plan_cache: OnceLock<Vec<PlanEntry>>,
}

/// One precomputed measurement of a `sample()` plan (see
/// [`StabilizerTableau::build_sample_plan`]).
///
/// In the AG pass the supports (x, z) evolve identically for every shot —
/// only the phase bits depend on the drawn random outcomes — so the pass is
/// run once with the phases kept as GF(2) linear forms over the drawn bits.
/// Each shot then just draws the fresh bits and evaluates the parities.
enum PlanEntry {
    /// Outcome is a fresh random bit; `var` is its index in the per-shot
    /// drawn-bit vector (draws happen in q order, matching the historical
    /// per-shot AG stream).
    Random { var: u32 },
    /// Outcome is the parity of previously drawn bits selected by `mask`
    /// (bit `v` of the drawn-bit vector = variable `v`), XOR `c`.
    Determined { mask: Vec<u64>, c: u8 },
}

/// Flat typed Clifford op for the fused row-major builder (`from_cops`).
/// `SX`/`SXDG`/`CZ`/`CY` are expanded into these primitives at flatten time,
/// keeping the per-row hot loop free of decomposition branches.
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

/// Minimal FxHash (Firefox hasher) for the sampler's internal counting maps.
///
/// Keys are bitstring-shaped and can be kilobytes long at n = 4096; SipHash's
/// per-byte mixing dominates the block loop there.  These maps are private
/// counters (no adversarial input), so a cheap non-cryptographic hash is
/// sufficient — the public API still returns `std::collections::HashMap`.
#[derive(Default)]
struct FxHasher {
    hash: u64,
}

const FX_SEED: u64 = 0x51_7c_c1_b7_27_22_0a_95;

/// Row×op work below which the fused tableau build stays serial (rayon task
/// spawn costs more than the streaming it saves).  Overridable via
/// `SF_STAB_PAR_MIN` so the crossover can be measured on-host rather than
/// assumed; the default preserves the historical `1 << 18` behaviour.
static SF_STAB_PAR_MIN: OnceLock<usize> = OnceLock::new();

impl std::hash::Hasher for FxHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.hash
    }

    #[inline]
    fn write(&mut self, mut bytes: &[u8]) {
        while bytes.len() >= 8 {
            let mut v = [0u8; 8];
            v.copy_from_slice(&bytes[..8]);
            self.hash = (self.hash.rotate_left(5) ^ u64::from_le_bytes(v)).wrapping_mul(FX_SEED);
            bytes = &bytes[8..];
        }
        if !bytes.is_empty() {
            let mut v = [0u8; 8];
            v[..bytes.len()].copy_from_slice(bytes);
            self.hash = (self.hash.rotate_left(5) ^ u64::from_le_bytes(v)).wrapping_mul(FX_SEED);
        }
    }

    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.hash = (self.hash.rotate_left(5) ^ i).wrapping_mul(FX_SEED);
    }

    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.write_u64(i as u64);
    }
}

#[derive(Default, Clone, Copy)]
struct FxBuildHasher;

impl std::hash::BuildHasher for FxBuildHasher {
    type Hasher = FxHasher;

    #[inline]
    fn build_hasher(&self) -> FxHasher {
        FxHasher::default()
    }
}

impl StabilizerTableau {
    pub fn new(n: usize) -> Self {
        assert!(n <= 4096, "Tableau supports n ≤ 4096");
        let words = n.div_ceil(64);
        let mut x = vec![vec![0u64; words]; 2 * n];
        let mut z = vec![vec![0u64; words]; 2 * n];
        let r = vec![0u64; 2 * n];
        for q in 0..n {
            let w = q / 64;
            let b = q % 64;
            x[q][w] = 1u64 << b; // destabilizer i = X_i
            z[n + q][w] = 1u64 << b; // stabilizer i = Z_i
        }
        Self {
            n,
            words,
            x,
            z,
            r,
            sx_buf: vec![0u64; words],
            sz_buf: vec![0u64; words],
            plan_cache: OnceLock::new(),
        }
    }

    /// Drop the cached measurement plan — any tableau mutation makes it stale.
    #[inline]
    fn invalidate_plan(&mut self) {
        self.plan_cache = OnceLock::new();
    }

    // ── Single-qubit gates ──

    pub fn h(&mut self, q: usize) {
        self.invalidate_plan();
        let w = q / 64;
        let bit = q % 64;
        let mask = 1u64 << bit;
        for i in 0..2 * self.n {
            let xq = (self.x[i][w] >> bit) & 1;
            let zq = (self.z[i][w] >> bit) & 1;
            self.r[i] ^= xq & zq;
            self.x[i][w] = (self.x[i][w] & !mask) | (zq << bit);
            self.z[i][w] = (self.z[i][w] & !mask) | (xq << bit);
        }
    }

    pub fn s(&mut self, q: usize) {
        self.invalidate_plan();
        let w = q / 64;
        for i in 0..2 * self.n {
            let xq = (self.x[i][w] >> (q % 64)) & 1;
            let zq = (self.z[i][w] >> (q % 64)) & 1;
            self.r[i] ^= xq & zq;
            self.z[i][w] ^= xq << (q % 64);
        }
    }

    pub fn sdg(&mut self, q: usize) {
        self.s(q);
        self.s(q);
        self.s(q);
    }

    pub fn x_gate(&mut self, q: usize) {
        self.invalidate_plan();
        let w = q / 64;
        for i in 0..2 * self.n {
            self.r[i] ^= (self.z[i][w] >> (q % 64)) & 1;
        }
    }

    pub fn z_gate(&mut self, q: usize) {
        self.invalidate_plan();
        let w = q / 64;
        for i in 0..2 * self.n {
            self.r[i] ^= (self.x[i][w] >> (q % 64)) & 1;
        }
    }

    pub fn y_gate(&mut self, q: usize) {
        self.invalidate_plan();
        let w = q / 64;
        for i in 0..2 * self.n {
            let xq = (self.x[i][w] >> (q % 64)) & 1;
            let zq = (self.z[i][w] >> (q % 64)) & 1;
            self.r[i] ^= xq ^ zq;
        }
    }

    // ── Two-qubit gates ──

    /// CNOT(c, t): control=c, target=t.
    /// Phase: r ^= x_c · z_t · (x_t ⊕ z_c ⊕ 1)
    pub fn cnot(&mut self, a: usize, b: usize) {
        self.invalidate_plan();
        let wa = a / 64;
        let ba = a % 64;
        let wb = b / 64;
        let bb = b % 64;
        for i in 0..2 * self.n {
            let xa = (self.x[i][wa] >> ba) & 1;
            let xb = (self.x[i][wb] >> bb) & 1;
            let za = (self.z[i][wa] >> ba) & 1;
            let zb = (self.z[i][wb] >> bb) & 1;
            // Phase: r ^= x_a · z_b · (x_b ⊕ z_a ⊕ 1)
            self.r[i] ^= xa & zb & (xb ^ za ^ 1);
            // x_b ^= x_a
            self.x[i][wb] ^= xa << bb;
            // z_a ^= z_b
            self.z[i][wa] ^= zb << ba;
        }
    }

    /// CZ(a, b) via decomposition: H(b) · CNOT(a,b) · H(b)
    pub fn cz(&mut self, a: usize, b: usize) {
        self.h(b);
        self.cnot(a, b);
        self.h(b);
    }

    pub fn swap(&mut self, a: usize, b: usize) {
        self.invalidate_plan();
        let wa = a / 64;
        let ba = a % 64;
        let wb = b / 64;
        let bb = b % 64;
        let _ma = 1u64 << ba;
        let _mb = 1u64 << bb;
        for i in 0..2 * self.n {
            let xa = (self.x[i][wa] >> ba) & 1;
            let xb = (self.x[i][wb] >> bb) & 1;
            let za = (self.z[i][wa] >> ba) & 1;
            let zb = (self.z[i][wb] >> bb) & 1;
            self.x[i][wa] = (self.x[i][wa] & !_ma) | (xb << ba);
            self.x[i][wb] = (self.x[i][wb] & !_mb) | (xa << bb);
            self.z[i][wa] = (self.z[i][wa] & !_ma) | (zb << ba);
            self.z[i][wb] = (self.z[i][wb] & !_mb) | (za << bb);
        }
    }

    /// CY = S_t · CNOT_{c,t} · S†_t
    pub fn cy(&mut self, c: usize, t: usize) {
        self.sdg(t);
        self.cnot(c, t);
        self.s(t);
    }

    // ── Gate dispatch ──

    pub fn apply_gate(&mut self, name: &str, qubits: &[usize]) -> Result<(), String> {
        match name {
            "H" => self.h(qubits[0]),
            "S" => self.s(qubits[0]),
            "SDG" => self.sdg(qubits[0]),
            "SX" => {
                self.h(qubits[0]);
                self.s(qubits[0]);
                self.h(qubits[0]);
            }
            "SXDG" => {
                self.h(qubits[0]);
                self.sdg(qubits[0]);
                self.h(qubits[0]);
            }
            "X" => self.x_gate(qubits[0]),
            "Y" => self.y_gate(qubits[0]),
            "Z" => self.z_gate(qubits[0]),
            "CX" | "CNOT" => self.cnot(qubits[0], qubits[1]),
            "CZ" => self.cz(qubits[0], qubits[1]),
            "CY" => self.cy(qubits[0], qubits[1]),
            "SWAP" => self.swap(qubits[0], qubits[1]),
            "ID" | "BARRIER" | "MEASURE" | "RESET" => {} // no-ops
            _ => return Err(format!("Unsupported gate: {}", name)),
        }
        Ok(())
    }

    /// Build and evolve a tableau from (gate_name, qubits) list.
    ///
    /// Flattens to typed ops, then runs the fused row-major builder: each
    /// tableau row consumes the whole op sequence in one pass (every AG gate
    /// update is row-local) while its words stay cache-hot, with rows split
    /// across rayon threads for large circuits.
    pub fn from_gate_list(n: usize, gates: &[(String, Vec<usize>)]) -> Result<Self, String> {
        let mut cops: Vec<COp> = Vec::with_capacity(gates.len());
        for (name, qubits) in gates {
            Self::flatten_gate(name, qubits, &mut cops)?;
        }
        Ok(Self::from_cops(n, &cops))
    }

    /// Flatten one named gate into typed ops.  Mirrors `apply_gate` exactly
    /// (same names, same decompositions, same errors).
    fn flatten_gate(name: &str, qubits: &[usize], out: &mut Vec<COp>) -> Result<(), String> {
        let q0 = qubits[0] as u32;
        match name {
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
            _ => return Err(format!("Unsupported gate: {}", name)),
        }
        Ok(())
    }

    /// Build and evolve a tableau directly from the DAG (typed fast path).
    ///
    /// Identical semantics to `from_gate_list` (BARRIER/MEASURE/RESET
    /// skipped, ID no-op, non-Clifford ops rejected) but dispatches on the
    /// typed `OpType` enum — no per-gate String allocation, uppercasing,
    /// or qubit-vec cloning — and runs the fused row-major builder.
    pub fn from_dag(dag: &QuantumDAG) -> Result<Self, String> {
        let mut cops: Vec<COp> = Vec::with_capacity(dag.graph().node_count());
        for id in dag.topological_order() {
            let op = &dag.graph()[id];
            let q = &op.qubits;
            let q0 = q[0] as u32;
            match &op.op_type {
                OpType::H => cops.push(COp::H(q0)),
                OpType::S => cops.push(COp::S(q0)),
                OpType::Sdg => cops.push(COp::Sdg(q0)),
                OpType::SX => cops.extend([COp::H(q0), COp::S(q0), COp::H(q0)]),
                OpType::X => cops.push(COp::X(q0)),
                OpType::Y => cops.push(COp::Y(q0)),
                OpType::Z => cops.push(COp::Z(q0)),
                OpType::CNOT => cops.push(COp::Cnot(q0, q[1] as u32)),
                OpType::CZ => {
                    let q1 = q[1] as u32;
                    cops.extend([COp::H(q1), COp::Cnot(q0, q1), COp::H(q1)]);
                }
                OpType::CY => {
                    let q1 = q[1] as u32;
                    cops.extend([COp::Sdg(q1), COp::Cnot(q0, q1), COp::S(q1)]);
                }
                OpType::SWAP => cops.push(COp::Swap(q0, q[1] as u32)),
                OpType::Id | OpType::Barrier | OpType::Measure | OpType::Reset => {}
                other => return Err(format!("Unsupported gate: {:?}", other)),
            }
        }
        Ok(Self::from_cops(dag.n_qubits, &cops))
    }

    /// Build a tableau from a flat typed op list (fused row-major build).
    fn from_cops(n: usize, cops: &[COp]) -> Self {
        let mut tab = Self::new(n);
        tab.apply_cops(cops);
        tab
    }

    /// Apply a flat op list with the fused row-major loop.
    ///
    /// The historical order (gate-outer, row-inner) streamed the whole
    /// 2n-row tableau from DRAM once per gate; inverting it keeps each row's
    /// words cache-resident for the entire sequence (gate updates depend only
    /// on the row's own bits), and rows are independent, so large builds are
    /// split across rayon threads.
    fn apply_cops(&mut self, cops: &[COp]) {
        self.invalidate_plan();
        if cops.is_empty() {
            return;
        }
        let rows = 2 * self.n;
        let work = rows.saturating_mul(cops.len());
        let par_min = *SF_STAB_PAR_MIN.get_or_init(|| {
            std::env::var("SF_STAB_PAR_MIN")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(1 << 18)
        });
        if work < par_min || rayon::current_num_threads() <= 1 {
            for i in 0..rows {
                Self::apply_row(cops, &mut self.x[i], &mut self.z[i], &mut self.r[i]);
            }
        } else {
            let threads = rayon::current_num_threads().max(1);
            let blk = rows.div_ceil(threads.saturating_mul(8)).max(32);
            self.x
                .par_chunks_mut(blk)
                .zip(self.z.par_chunks_mut(blk))
                .zip(self.r.par_chunks_mut(blk))
                .for_each(|((xs, zs), rs)| {
                    for i in 0..xs.len() {
                        Self::apply_row(cops, &mut xs[i], &mut zs[i], &mut rs[i]);
                    }
                });
        }
    }

    #[inline(always)]
    fn bit_pos(q: u32) -> (usize, u32, u64) {
        let b = q & 63;
        ((q >> 6) as usize, b, 1u64 << b)
    }

    /// Apply the whole op list, in order, to ONE tableau row.
    /// Semantics are identical to the per-gate methods (`h`, `s`, ...).
    fn apply_row(cops: &[COp], xr: &mut [u64], zr: &mut [u64], r: &mut u64) {
        for &op in cops {
            match op {
                COp::H(q) => {
                    let (w, b, m) = Self::bit_pos(q);
                    let xq = (xr[w] >> b) & 1;
                    let zq = (zr[w] >> b) & 1;
                    *r ^= xq & zq;
                    xr[w] = (xr[w] & !m) | (zq << b);
                    zr[w] = (zr[w] & !m) | (xq << b);
                }
                COp::S(q) => {
                    let (w, b, _) = Self::bit_pos(q);
                    let xq = (xr[w] >> b) & 1;
                    let zq = (zr[w] >> b) & 1;
                    *r ^= xq & zq;
                    zr[w] ^= xq << b;
                }
                COp::Sdg(q) => {
                    let (w, b, _) = Self::bit_pos(q);
                    for _ in 0..3 {
                        let xq = (xr[w] >> b) & 1;
                        let zq = (zr[w] >> b) & 1;
                        *r ^= xq & zq;
                        zr[w] ^= xq << b;
                    }
                }
                COp::X(q) => {
                    let (w, b, _) = Self::bit_pos(q);
                    *r ^= (zr[w] >> b) & 1;
                }
                COp::Z(q) => {
                    let (w, b, _) = Self::bit_pos(q);
                    *r ^= (xr[w] >> b) & 1;
                }
                COp::Y(q) => {
                    let (w, b, _) = Self::bit_pos(q);
                    let xq = (xr[w] >> b) & 1;
                    let zq = (zr[w] >> b) & 1;
                    *r ^= xq ^ zq;
                }
                COp::Cnot(a, b) => {
                    let (wa, ba, _) = Self::bit_pos(a);
                    let (wb, bb, _) = Self::bit_pos(b);
                    let xa = (xr[wa] >> ba) & 1;
                    let xb = (xr[wb] >> bb) & 1;
                    let za = (zr[wa] >> ba) & 1;
                    let zb = (zr[wb] >> bb) & 1;
                    *r ^= xa & zb & (xb ^ za ^ 1);
                    xr[wb] ^= xa << bb;
                    zr[wa] ^= zb << ba;
                }
                COp::Swap(a, b) => {
                    let (wa, ba, ma) = Self::bit_pos(a);
                    let (wb, bb, mb) = Self::bit_pos(b);
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

    // ── Phase of product (AG 2004) ──

    /// Phase of Pauli product in {0,1,2,3} (units of i).
    /// Word-level: uses popcount on 64-bit masks instead of per-bit iteration.
    /// ~64× faster than the naive per-bit loop.
    #[inline(always)]
    fn phase_of_product(
        x1: &[u64],
        z1: &[u64],
        x2: &[u64],
        z2: &[u64],
        n: usize,
        words: usize,
    ) -> u8 {
        let mut phase: u32 = 0;
        for w in 0..words {
            let xw1 = x1[w];
            let zw1 = z1[w];
            let xw2 = x2[w];
            let zw2 = z2[w];
            // Masks for contributions of 1 (mod 4):
            //   (Z,Y): !x1 & z1 & x2 & z2
            //   (X,Z): x1 & !z1 & !x2 & z2
            //   (Y,X): x1 & z1 & x2 & !z2
            let c1 =
                (!xw1 & zw1 & xw2 & zw2) | (xw1 & !zw1 & !xw2 & zw2) | (xw1 & zw1 & xw2 & !zw2);
            // Masks for contributions of 3 (mod 4):
            //   (Z,X): !x1 & z1 & x2 & !z2
            //   (X,Y): x1 & !z1 & x2 & z2
            //   (Y,Z): x1 & z1 & !x2 & z2
            let c3 =
                (!xw1 & zw1 & xw2 & !zw2) | (xw1 & !zw1 & xw2 & zw2) | (xw1 & zw1 & !xw2 & zw2);
            // Mask final word to the actual qubit count
            let mask = if w == words - 1 && !n.is_multiple_of(64) {
                (1u64 << (n % 64)) - 1
            } else {
                u64::MAX
            };
            let count1 = (c1 & mask).count_ones();
            let count3 = (c3 & mask).count_ones();
            phase = (phase + count1 + 3 * count3) & 3;
        }
        phase as u8
    }

    // ── Row multiplication ──

    /// Multiply row h by row i (h += i in symplectic space).
    /// Inlined into measure_z hot path — avoids function-call overhead
    /// for O(n²) calls per measurement.
    #[inline(always)]
    #[allow(dead_code)]
    fn row_mult(&mut self, h: usize, i: usize) {
        let new_phase = (2 * self.r[h]
            + 2 * self.r[i]
            + Self::phase_of_product(
                &self.x[h], &self.z[h], &self.x[i], &self.z[i], self.n, self.words,
            ) as u64)
            & 3;
        self.r[h] = (new_phase >> 1) & 1;
        for w in 0..self.words {
            self.x[h][w] ^= self.x[i][w];
            self.z[h][w] ^= self.z[i][w];
        }
    }

    // ── Measure Z_q (AG algorithm 1) ──

    /// Single-shot reference implementation.  `sample()` no longer calls it
    /// (it builds a measurement plan instead); it is kept as the single-shot
    /// reference for tests and for future mid-circuit work.
    #[allow(dead_code)]
    fn measure_z(&mut self, q: usize, rng: &mut StdRng) -> u8 {
        self.invalidate_plan();
        let n = self.n;
        let w = q / 64;
        let mask = 1u64 << (q % 64);

        // Random branch: find stabilizer row p ≥ n with x[p][q]=1
        for p in n..2 * n {
            if (self.x[p][w] & mask) != 0 {
                let outcome: u8 = rng.gen_range(0..2);

                // ── Pre-extract row p (read-only multiplier) to avoid
                //     repeated indexing + borrow-checker conflicts. ──
                let xp = self.x[p].clone();
                let zp = self.z[p].clone();
                let rp = self.r[p];
                let words = self.words;
                let xp_ptr = xp.as_ptr();
                let zp_ptr = zp.as_ptr();

                // Multiply every row i≠p where x[i][q]=1 by row p
                for i in 0..2 * n {
                    if i != p && (self.x[i][w] & mask) != 0 {
                        // ── Inlined row_mult (avoids fn-call + repeated self.x[p] indexing) ──
                        let new_phase = (2 * self.r[i]
                            + 2 * rp
                            + Self::phase_of_product(&self.x[i], &self.z[i], &xp, &zp, n, words)
                                as u64)
                            & 3;
                        self.r[i] = (new_phase >> 1) & 1;
                        // XOR loop (raw pointers — row i is exclusive, row p is const)
                        let xi_ptr = self.x[i].as_mut_ptr();
                        let zi_ptr = self.z[i].as_mut_ptr();
                        unsafe {
                            for ww in 0..words {
                                *xi_ptr.add(ww) ^= *xp_ptr.add(ww);
                                *zi_ptr.add(ww) ^= *zp_ptr.add(ww);
                            }
                        }
                    }
                }

                // Destabilizer(p-n) ← old stabilizer p
                for ww in 0..self.words {
                    self.x[p - n][ww] = self.x[p][ww];
                    self.z[p - n][ww] = self.z[p][ww];
                }
                self.r[p - n] = self.r[p];

                // Stabilizer p ← Z_q with phase=outcome
                for ww in 0..self.words {
                    self.x[p][ww] = 0;
                    self.z[p][ww] = 0;
                }
                self.z[p][w] = 1u64 << (q % 64);
                self.r[p] = outcome as u64;

                return outcome;
            }
        }

        // Deterministic branch: accumulate destabilizer rows with x[i][q]=1
        // Reuse scratch buffers to avoid per-call heap allocation.
        self.sx_buf.fill(0);
        self.sz_buf.fill(0);
        let mut sr: u64 = 0; // 2-bit phase accumulator
        let words = self.words;
        let sx_ptr = self.sx_buf.as_mut_ptr();
        let sz_ptr = self.sz_buf.as_mut_ptr();
        for i in 0..n {
            if (self.x[i][w] & mask) != 0 {
                let si = i + n;
                let phase = (sr
                    + 2 * self.r[si]
                    + Self::phase_of_product(
                        &self.sx_buf,
                        &self.sz_buf,
                        &self.x[si],
                        &self.z[si],
                        n,
                        words,
                    ) as u64)
                    & 3;
                sr = phase;
                // XOR loop — raw pointers avoid bounds checks on repeated buf access
                let xsi_ptr = self.x[si].as_ptr();
                let zsi_ptr = self.z[si].as_ptr();
                unsafe {
                    for ww in 0..words {
                        *sx_ptr.add(ww) ^= *xsi_ptr.add(ww);
                        *sz_ptr.add(ww) ^= *zsi_ptr.add(ww);
                    }
                }
            }
        }
        if (sr & 2) != 0 {
            1
        } else {
            0
        }
    }

    // ── Data export for Python interop ──

    /// Export tableau data as flat vectors: (x_flat, z_flat, r).
    /// x/z are 2n × n bit matrices packed as u8s (row-major).
    pub fn to_raw(&self) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let mut x_flat = vec![0u8; 2 * self.n * self.n];
        let mut z_flat = vec![0u8; 2 * self.n * self.n];
        for row in 0..2 * self.n {
            let base = row * self.n;
            for q in 0..self.n {
                let w = q / 64;
                let b = q % 64;
                x_flat[base + q] = ((self.x[row][w] >> b) & 1) as u8;
                z_flat[base + q] = ((self.z[row][w] >> b) & 1) as u8;
            }
        }
        let r_flat: Vec<u8> = self.r.iter().map(|&v| v as u8).collect();
        (x_flat, z_flat, r_flat)
    }

    // ── Pauli expectation value ──

    /// Return <psi|P|psi> where P is encoded as (px, pz) bit vectors.
    /// Returns 0.0 if P anticommutes with any stabilizer, ±1.0 otherwise.
    pub fn pauli_expval(&self, px: &[u8], pz: &[u8]) -> f64 {
        let n = self.n;
        assert_eq!(px.len(), n);
        assert_eq!(pz.len(), n);

        // 1. Commutation check vs stabilizer rows (n..2n-1)
        for i in n..2 * n {
            let mut symp: u64 = 0;
            for q in 0..n {
                let w = q / 64;
                let b = q % 64;
                let sx = (self.x[i][w] >> b) & 1;
                let sz = (self.z[i][w] >> b) & 1;
                symp ^= (sx * (pz[q] as u64)) ^ (sz * (px[q] as u64));
            }
            if (symp & 1) != 0 {
                return 0.0;
            }
        }

        // 2. Find which destabilizers anticommute with P
        let mut sel: Vec<usize> = Vec::new();
        for i in 0..n {
            let mut symp: u64 = 0;
            for q in 0..n {
                let w = q / 64;
                let b = q % 64;
                let dx = (self.x[i][w] >> b) & 1;
                let dz = (self.z[i][w] >> b) & 1;
                symp ^= (dx * (pz[q] as u64)) ^ (dz * (px[q] as u64));
            }
            if (symp & 1) != 0 {
                sel.push(i);
            }
        }

        // 3. Multiply selected stabilizer rows; check sign
        let mut prod_x = vec![0u8; n];
        let mut prod_z = vec![0u8; n];
        let mut prod_phase: u32 = 0;
        for &i in &sel {
            let si = n + i;
            let mut sx = vec![0u64; self.words];
            let mut sz = vec![0u64; self.words];
            for q in 0..n {
                let w = q / 64;
                let b = q % 64;
                let vx = (self.x[si][w] >> b) & 1;
                let vz = (self.z[si][w] >> b) & 1;
                if vx != 0 {
                    sx[w] |= 1 << b;
                }
                if vz != 0 {
                    sz[w] |= 1 << b;
                }
            }
            prod_phase = (prod_phase
                + 2 * self.r[si] as u32
                + Self::phase_of_product_u8(&prod_x, &prod_z, &sx, &sz, n, self.words) as u32)
                & 3;
            for q in 0..n {
                let w = q / 64;
                let b = q % 64;
                prod_x[q] ^= ((sx[w] >> b) & 1) as u8;
                prod_z[q] ^= ((sz[w] >> b) & 1) as u8;
            }
        }
        match prod_phase {
            0 => 1.0,
            2 => -1.0,
            _ => 0.0,
        }
    }

    /// Phase of product for u8-level arrays (used by pauli_expval).
    fn phase_of_product_u8(
        _x1: &[u8],
        _z1: &[u8],
        _x2: &[u64],
        _z2: &[u64],
        n: usize,
        _words: usize,
    ) -> u8 {
        const G: [[u8; 4]; 4] = [[0, 0, 0, 0], [0, 0, 3, 1], [0, 1, 0, 3], [0, 3, 1, 0]];
        let mut phase: u32 = 0;
        for q in 0..n {
            let w = q / 64;
            let b = q % 64;
            let p1 = ((_x1[q] as usize) << 1) | (_z1[q] as usize);
            let p2 = (((_x2[w] >> b) & 1) as usize) << 1 | (((_z2[w] >> b) & 1) as usize);
            phase += G[p1][p2] as u32;
        }
        (phase % 4) as u8
    }

    // ── Sampling ──

    /// Sample `shots` bitstrings and return the counts map.
    ///
    /// Bit-sliced block evaluation (64 shots per block): every plan entry's
    /// outcome for a whole block is one `u64` (bit i = shot i) — `Random`
    /// entries read a fresh word from a per-block ChaCha stream, `Determined`
    /// entries XOR their mask slices.  A block costs O(Σ|mask| + 64·n) word
    /// ops instead of 64 separate per-shot passes.
    ///
    /// Keys use the shared little-endian convention (qubit q = bit q of the
    /// integer form; the leftmost character is the highest qubit) — the same
    /// keys the Python layer previously produced by reversing the raw
    /// output, so `RunResult.counts` is unchanged.
    ///
    /// Determinism: with an explicit `seed`, the shot→randomness mapping is
    /// fixed per build (block b draws from `seed + b`); two calls with the
    /// same arguments return identical counts.
    ///
    /// The measurement plan is cached per tableau revision, so all calls
    /// after the first pay only the block evaluation.
    pub fn sample(&self, shots: usize, seed: Option<u64>) -> HashMap<String, usize> {
        if shots == 0 {
            return HashMap::new();
        }
        let base_seed = seed.unwrap_or_else(rand::random);
        let n = self.n;
        // The plan depends only on the tableau; the (possibly expensive)
        // AG pass runs once per tableau revision and is reused across
        // `sample()` calls (mutators call `invalidate_plan`).
        let plan = self.plan_cache.get_or_init(|| self.build_sample_plan());
        let var_count = plan
            .iter()
            .map(|e| match e {
                PlanEntry::Random { var } => *var as usize + 1,
                PlanEntry::Determined { .. } => 0,
            })
            .max()
            .unwrap_or(0);

        /// Shots per block — one `u64` lane per shot.
        const BLOCK: usize = 64;
        let nblocks = shots.div_ceil(BLOCK);
        // Evaluate the cached plan for one block into a `u64` per qubit
        // (bit i = shot i's outcome for q); bits >= cnt are garbage and never
        // read. Shared by both packing paths below.
        let eval_block = |b: usize| -> Vec<u64> {
            let mut rng = StdRng::seed_from_u64(base_seed.wrapping_add(b as u64));
            let mut vars = vec![0u64; var_count];
            rng.fill(&mut vars[..]);
            let mut out = vec![0u64; n];
            for (q, e) in plan.iter().enumerate() {
                out[q] = match e {
                    PlanEntry::Random { var } => vars[*var as usize],
                    PlanEntry::Determined { mask, c } => {
                        let mut lane: u64 = if *c & 1 != 0 { u64::MAX } else { 0 };
                        for (k, m) in mask.iter().enumerate() {
                            let mut m = *m;
                            while m != 0 {
                                let j = m.trailing_zeros() as usize;
                                m &= m - 1;
                                lane ^= vars[64 * k + j];
                            }
                        }
                        lane
                    }
                };
            }
            out
        };

        if n <= 64 {
            // u64-keyed counting, then ONE string materialization per distinct
            // pack across the whole run. Each block emits a
            // `HashMap<u64, usize>` (bit q of the key = that shot's outcome for
            // qubit q); the per-block maps merge on integer keys (cheap Fx
            // hashing) so the `String::from_utf8` allocation + the `acc` string
            // re-hash run once over the global distinct set instead of once per
            // block. Attribution (`SF_STAB_TRACE`) showed this per-block String
            // layer dominates the sampling window (pack >> plan-eval >> var-gen
            // on the n<=20 bench sizes), while the mask->lane XOR loop is ~1-2%
            // of it — so the win is in removing the redundant key materializer,
            // not SIMD-ing the XOR accumulate. The result map is identical
            // (same keys, same summed counts); only the materialization order
            // changes.
            let per_block: Vec<HashMap<u64, usize, FxBuildHasher>> = (0..nblocks)
                .into_par_iter()
                .map(|b| {
                    let cnt = (shots - b * BLOCK).min(BLOCK);
                    let out = eval_block(b);
                    let mut keyed: HashMap<u64, usize, FxBuildHasher> =
                        HashMap::with_hasher(FxBuildHasher);
                    for i in 0..cnt {
                        let mut pack = 0u64;
                        for (q, lane) in out.iter().enumerate() {
                            pack |= ((*lane >> i) & 1) << q;
                        }
                        *keyed.entry(pack).or_insert(0) += 1;
                    }
                    keyed
                })
                .collect();
            // Merge the per-block u64 maps (cheap integer Fx hashing, kept
            // serial), then materialize the LE strings once over the global
            // distinct set — that decode is parallelized (each task owns its
            // scratch buffer). Removes the redundant per-block String layer
            // (dominant on small-n / high-collision runs) without serialising
            // the decode on large-n / low-collision runs.
            let mut gk: HashMap<u64, usize, FxBuildHasher> = HashMap::with_hasher(FxBuildHasher);
            for m in per_block {
                for (k, v) in m {
                    *gk.entry(k).or_insert(0) += v;
                }
            }
            // LE key: leftmost char = highest qubit.
            let entries: Vec<(u64, usize)> = gk.into_iter().collect();
            let acc: HashMap<String, usize> = entries
                .into_par_iter()
                .map(|(pack, c)| {
                    let mut buf = vec![b'0'; n];
                    let mut bits = pack;
                    while bits != 0 {
                        let q = bits.trailing_zeros() as usize;
                        bits &= bits - 1;
                        buf[n - 1 - q] = b'1';
                    }
                    // Only ASCII '0'/'1' was written above.
                    (String::from_utf8(buf).unwrap(), c)
                })
                .collect();
            acc
        } else {
            // n > 64: a pack does not fit a u64. Build the LE string per shot,
            // reusing a scratch buffer; allocate only for distinct keys.
            let per_block: Vec<HashMap<String, usize, FxBuildHasher>> = (0..nblocks)
                .into_par_iter()
                .map(|b| {
                    let cnt = (shots - b * BLOCK).min(BLOCK);
                    let out = eval_block(b);
                    let mut map: HashMap<String, usize, FxBuildHasher> =
                        HashMap::with_hasher(FxBuildHasher);
                    let mut buf = String::with_capacity(n);
                    for i in 0..cnt {
                        buf.clear();
                        for q in (0..n).rev() {
                            buf.push(if (out[q] >> i) & 1 != 0 { '1' } else { '0' });
                        }
                        match map.get_mut(buf.as_str()) {
                            Some(v) => *v += 1,
                            None => {
                                map.insert(buf.clone(), 1);
                            }
                        }
                    }
                    map
                })
                .collect();
            let mut acc: HashMap<String, usize> = HashMap::new();
            for m in per_block {
                for (k, v) in m {
                    *acc.entry(k).or_insert(0) += v;
                }
            }
            acc
        }
    }

    /// Run the AG pass once, tracking each row phase as a GF(2) linear form
    /// over the drawn random bits, and emit one plan entry per qubit.
    ///
    /// This mirrors `measure_z` step for step: the pivot search and every row
    /// multiplication depend only on the supports, and phase updates are
    /// linear (bit 1 of the 2-bit phase: `r_i ^ r_p ^ (popp >> 1)` for the
    /// random branch; the deterministic accumulator `sr = (sr + 2·r_si +
    /// popp) mod 4` carries its low bit `s0` — which never depends on random
    /// bits — into the linear form).
    fn build_sample_plan(&self) -> Vec<PlanEntry> {
        let n = self.n;
        let words = self.words;
        // Upper bound on fresh draws: every qubit either draws once or is a
        // parity of earlier draws.
        let mwords = n.div_ceil(64).max(1);
        let mut tab = self.clone();
        let mut r_mask: Vec<Vec<u64>> = vec![vec![0u64; mwords]; 2 * n];
        // Gate applications (H/S/SDG/SX/SY...) and CZ/CY already set phase
        // bits in `self.r`; the linear forms model only the *drawn* bits, so
        // seed the constants from the current phases (value bit of each row).
        let mut r_const: Vec<u8> = self.r.iter().map(|&v| (v & 1) as u8).collect();
        let mut var_count: u32 = 0;
        let mut plan: Vec<PlanEntry> = Vec::with_capacity(n);

        for q in 0..n {
            let w = q / 64;
            let qmask = 1u64 << (q % 64);

            // Random branch: first stabilizer row p ≥ n with x[p][q] = 1.
            let mut p_rand: Option<usize> = None;
            for p in n..2 * n {
                if (tab.x[p][w] & qmask) != 0 {
                    p_rand = Some(p);
                    break;
                }
            }

            if let Some(p) = p_rand {
                // The multiplier row is read BEFORE row p is replaced, exactly
                // like measure_z (`let rp = self.r[p];`), so snapshot it.
                let rp_mask = r_mask[p].clone();
                let rp_const = r_const[p];
                let xp = tab.x[p].clone();
                let zp = tab.z[p].clone();

                // Multiply every row i ≠ p with x[i][q] = 1 by row p:
                //   r[i] ← r[i] ^ r[p] ^ (popp >> 1)
                for i in 0..2 * n {
                    if i != p && (tab.x[i][w] & qmask) != 0 {
                        let popp = Self::phase_of_product(&tab.x[i], &tab.z[i], &xp, &zp, n, words);
                        for k in 0..mwords {
                            r_mask[i][k] ^= rp_mask[k];
                        }
                        r_const[i] ^= rp_const ^ ((popp >> 1) & 1);
                        for ww in 0..words {
                            tab.x[i][ww] ^= xp[ww];
                            tab.z[i][ww] ^= zp[ww];
                        }
                    }
                }

                // Destabilizer(p-n) ← old stabilizer p (swap + overwrite), then
                // stabilizer p ← Z_q with the fresh random outcome as phase.
                tab.x.swap(p - n, p);
                tab.z.swap(p - n, p);
                r_mask.swap(p - n, p);
                r_const.swap(p - n, p);
                for ww in 0..words {
                    tab.x[p][ww] = 0;
                    tab.z[p][ww] = 0;
                }
                tab.z[p][w] = qmask;
                let var = var_count;
                var_count += 1;
                r_mask[p].fill(0);
                r_mask[p][(var as usize) / 64] = 1u64 << ((var as usize) % 64);
                r_const[p] = 0;
                plan.push(PlanEntry::Random { var });
            } else {
                // Deterministic branch: accumulate destabilizer rows with
                // x[i][q] = 1, exactly like measure_z (including the sx/sz
                // scratch buffers and the 2-bit phase accumulator).
                tab.sx_buf.fill(0);
                tab.sz_buf.fill(0);
                let mut s_mask = vec![0u64; mwords];
                let mut s_const: u8 = 0;
                let mut s_bit0: u8 = 0; // low bit of sr — never random
                for i in 0..n {
                    if (tab.x[i][w] & qmask) != 0 {
                        let si = i + n;
                        let popp = Self::phase_of_product(
                            &tab.sx_buf,
                            &tab.sz_buf,
                            &tab.x[si],
                            &tab.z[si],
                            n,
                            words,
                        );
                        // sr ← (sr + 2·r[si] + popp) mod 4 with sr = 2·S + s0:
                        //   S  ← S ^ r[si] ^ (popp>>1) ^ carry
                        //   s0 ← (s0 + (popp & 1)) & 1,  carry = s0 & (popp & 1)
                        let p0 = popp & 1;
                        let carry = s_bit0 & p0;
                        for k in 0..mwords {
                            s_mask[k] ^= r_mask[si][k];
                        }
                        s_const ^= r_const[si] ^ ((popp >> 1) & 1) ^ carry;
                        s_bit0 = (s_bit0 + p0) & 1;
                        for ww in 0..words {
                            tab.sx_buf[ww] ^= tab.x[si][ww];
                            tab.sz_buf[ww] ^= tab.z[si][ww];
                        }
                    }
                }
                plan.push(PlanEntry::Determined {
                    mask: s_mask,
                    c: s_const,
                });
            }
        }

        // Vars are introduced in q order, so masks can be trimmed to the
        // words actually referenced — keeps the per-shot parity loop minimal.
        let used = (var_count as usize).div_ceil(64);
        if used < mwords {
            for e in plan.iter_mut() {
                if let PlanEntry::Determined { mask, .. } = e {
                    mask.truncate(used);
                }
            }
        }
        plan
    }
}

impl Clone for StabilizerTableau {
    fn clone(&self) -> Self {
        Self {
            n: self.n,
            words: self.words,
            x: self.x.clone(),
            z: self.z.clone(),
            r: self.r.clone(),
            sx_buf: self.sx_buf.clone(),
            sz_buf: self.sz_buf.clone(),
            plan_cache: OnceLock::new(),
        }
    }
}

// ═══════════════════════════════════════════════════════════
// Standalone Pauli Twirl (gate-list level, no DAG dependency)
// ═══════════════════════════════════════════════════════════

/// CNOT twirl pairs: (P1_before, P2_before, P1_after, P2_after). 0=I,1=X,2=Z,3=Y.
const CNOT_TWIRL: &[(u8, u8, u8, u8)] = &[
    (0, 0, 0, 0),
    (0, 1, 0, 1),
    (0, 2, 2, 2),
    (0, 3, 2, 3),
    (1, 0, 1, 1),
    (1, 1, 1, 0),
    (1, 3, 3, 2),
    (2, 0, 2, 0),
    (2, 1, 2, 1),
    (2, 2, 0, 2),
    (2, 3, 0, 3),
    (3, 0, 3, 1),
    (3, 1, 3, 0),
    (3, 2, 1, 3),
];

/// CZ twirl pairs: (P1_before, P2_before, P1_after, P2_after). 0=I,1=X,2=Z,3=Y.
const CZ_TWIRL: &[(u8, u8, u8, u8)] = &[
    (0, 0, 0, 0),
    (0, 1, 2, 1),
    (0, 2, 0, 2),
    (0, 3, 2, 3),
    (1, 0, 1, 2),
    (1, 1, 3, 3),
    (1, 2, 1, 0),
    (2, 0, 2, 0),
    (2, 1, 0, 1),
    (2, 2, 2, 2),
    (2, 3, 0, 3),
    (3, 0, 3, 2),
    (3, 2, 3, 0),
    (3, 3, 1, 1),
];

fn pauli_name(idx: u8) -> &'static str {
    match idx {
        1 => "X",
        2 => "Z",
        3 => "Y",
        _ => "I",
    }
}

/// Apply Pauli twirling to a list of (gate_name, qubits, params) tuples.
pub fn pauli_twirl_gate_records(
    gates: &[(String, Vec<usize>, Vec<f64>)],
    seed: u64,
) -> Vec<(String, Vec<usize>, Vec<f64>)> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut result = Vec::with_capacity(gates.len() + gates.len() / 2);

    for (name, qubits, params) in gates {
        let upper = name.to_uppercase();
        let is_cx = upper == "CX" || upper == "CNOT";
        let is_cz = upper == "CZ";
        if (is_cx || is_cz) && qubits.len() == 2 {
            let pairs = if is_cz { CZ_TWIRL } else { CNOT_TWIRL };
            let (p1b, p2b, p1a, p2a) = pairs[rng.gen_range(0..pairs.len())];
            let q0 = qubits[0];
            let q1 = qubits[1];

            let pb1 = pauli_name(p1b);
            let pb2 = pauli_name(p2b);
            if pb1 != "I" {
                result.push((pb1.to_string(), vec![q0], vec![]));
            }
            if pb2 != "I" {
                result.push((pb2.to_string(), vec![q1], vec![]));
            }

            result.push((name.clone(), qubits.clone(), params.clone()));

            let pa1 = pauli_name(p1a);
            let pa2 = pauli_name(p2a);
            if pa1 != "I" {
                result.push((pa1.to_string(), vec![q0], vec![]));
            }
            if pa2 != "I" {
                result.push((pa2.to_string(), vec![q1], vec![]));
            }
        } else {
            result.push((name.clone(), qubits.clone(), params.clone()));
        }
    }
    result
}

/// Apply Pauli twirling to a list of (gate_name, qubits) tuples (no params).
pub fn pauli_twirl_gate_list(
    gates: &[(String, Vec<usize>)],
    seed: u64,
) -> Vec<(String, Vec<usize>)> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut result = Vec::with_capacity(gates.len() + gates.len() / 2);

    for (name, qubits) in gates {
        let upper = name.to_uppercase();
        let is_cx = upper == "CX" || upper == "CNOT";
        let is_cz = upper == "CZ";
        if (is_cx || is_cz) && qubits.len() == 2 {
            let pairs = if is_cz { CZ_TWIRL } else { CNOT_TWIRL };
            let (p1b, p2b, p1a, p2a) = pairs[rng.gen_range(0..pairs.len())];
            let q0 = qubits[0];
            let q1 = qubits[1];

            let pb1 = pauli_name(p1b);
            let pb2 = pauli_name(p2b);
            if pb1 != "I" {
                result.push((pb1.to_string(), vec![q0]));
            }
            if pb2 != "I" {
                result.push((pb2.to_string(), vec![q1]));
            }

            result.push((name.clone(), qubits.clone()));

            let pa1 = pauli_name(p1a);
            let pa2 = pauli_name(p2a);
            if pa1 != "I" {
                result.push((pa1.to_string(), vec![q0]));
            }
            if pa2 != "I" {
                result.push((pa2.to_string(), vec![q1]));
            }
        } else {
            result.push((name.clone(), qubits.clone()));
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tableau_identity() {
        let tab = StabilizerTableau::new(3);
        let counts = tab.sample(100, Some(42));
        assert_eq!(counts.get("000"), Some(&100));
    }

    #[test]
    fn test_tableau_h_gate() {
        let mut tab = StabilizerTableau::new(1);
        tab.h(0);
        let counts = tab.sample(1000, Some(42));
        let z0 = *counts.get("0").unwrap_or(&0);
        let z1 = *counts.get("1").unwrap_or(&0);
        assert!(z0 > 400 && z0 < 600, "Expected ~500 zeros, got {}", z0);
        assert!(z1 > 400 && z1 < 600, "Expected ~500 ones, got {}", z1);
    }

    #[test]
    fn test_tableau_cnot_bell() {
        let mut tab = StabilizerTableau::new(2);
        tab.h(0);
        tab.cnot(0, 1);
        let counts = tab.sample(1000, Some(42));
        let z00 = *counts.get("00").unwrap_or(&0);
        let z11 = *counts.get("11").unwrap_or(&0);
        assert!(z00 > 400 && z00 < 600, "Expected ~500 |00>, got {}", z00);
        assert!(z11 > 400 && z11 < 600, "Expected ~500 |11>, got {}", z11);
    }

    #[test]
    fn test_tableau_n_gt_64() {
        // Test tableau with n > 64 (requires word-packed representation)
        let mut tab = StabilizerTableau::new(100);
        tab.h(50);
        tab.cnot(50, 80);
        let counts = tab.sample(200, Some(123));
        // Keys are LE (qubit q = bit q of the integer form); the Bell-like
        // pair on qubits 50,80 must agree in every shot.
        for bs in counts.keys() {
            let v = u128::from_str_radix(bs, 2).unwrap();
            let b50 = (v >> 50) & 1;
            let b80 = (v >> 80) & 1;
            assert_eq!(b50, b80, "Expected correlated bits 50,80 in {}", bs);
        }
    }

    #[test]
    fn test_tableau_swap() {
        let mut tab = StabilizerTableau::new(3);
        tab.h(0);
        tab.swap(0, 2);
        let counts = tab.sample(500, Some(99));
        // Qubit 2 now has the |+> state → ~50% 0, 50% 1 at bit 2 of the LE key
        let ones: usize = counts
            .iter()
            .filter(|(bs, _)| (u64::from_str_radix(bs, 2).unwrap() >> 2) & 1 == 1)
            .map(|(_, c)| c)
            .sum();
        assert!(
            ones > 180 && ones < 320,
            "Expected ~250 ones on qubit 2, got {}",
            ones
        );
    }

    #[test]
    fn test_from_gate_list() {
        let gates: Vec<(String, Vec<usize>)> =
            vec![("H".into(), vec![0]), ("CNOT".into(), vec![0, 1])];
        let tab = StabilizerTableau::from_gate_list(2, &gates).unwrap();
        let counts = tab.sample(500, Some(42));
        let z00 = *counts.get("00").unwrap_or(&0);
        let z11 = *counts.get("11").unwrap_or(&0);
        assert!(z00 > 180 && z00 < 320);
        assert!(z11 > 180 && z11 < 320);
    }

    #[test]
    fn test_fused_builder_matches_sequential_apply() {
        // The fused row-major builder must be bit-for-bit identical to
        // applying the same gates one at a time through `apply_gate`
        // (the historical path), for every gate kind and both loop regimes
        // (serial below the rayon threshold, parallel above).
        for &n in &[3usize, 10, 70, 300, 512] {
            for &circ_seed in &[5u64, 11] {
                let mut gates = random_clifford_gates(n, circ_seed, 8 * n);
                for (k, name) in ["SXDG", "SX", "SWAP", "X", "Y", "Z"].iter().enumerate() {
                    let q0 = (3 * k + 1) % n;
                    if *name == "SWAP" {
                        gates.push((name.to_string(), vec![q0, (q0 + 2) % n]));
                    } else {
                        gates.push((name.to_string(), vec![q0]));
                    }
                }
                let mut seq = StabilizerTableau::new(n);
                for (name, qubits) in &gates {
                    seq.apply_gate(name, qubits).unwrap();
                }
                let fused = StabilizerTableau::from_gate_list(n, &gates).unwrap();
                assert_eq!(seq.x, fused.x, "x mismatch n={} seed={}", n, circ_seed);
                assert_eq!(seq.z, fused.z, "z mismatch n={} seed={}", n, circ_seed);
                assert_eq!(seq.r, fused.r, "r mismatch n={} seed={}", n, circ_seed);
            }
        }
    }

    /// Random Clifford circuit for replay tests (H/S/CX/CZ, seeded).
    fn random_clifford_gates(
        n: usize,
        circ_seed: u64,
        gate_count: usize,
    ) -> Vec<(String, Vec<usize>)> {
        use rand::SeedableRng;
        let mut grng = StdRng::seed_from_u64(circ_seed);
        let mut gates: Vec<(String, Vec<usize>)> = Vec::with_capacity(gate_count);
        for _ in 0..gate_count {
            match grng.gen_range(0..4) {
                0 => gates.push(("H".into(), vec![grng.gen_range(0..n)])),
                1 => gates.push(("S".into(), vec![grng.gen_range(0..n)])),
                2 => {
                    let q0 = grng.gen_range(0..n);
                    let mut q1 = grng.gen_range(0..n);
                    if q1 == q0 {
                        q1 = (q0 + 1) % n;
                    }
                    gates.push(("CX".into(), vec![q0, q1]));
                }
                _ => {
                    let q0 = grng.gen_range(0..n);
                    let mut q1 = grng.gen_range(0..n);
                    if q1 == q0 {
                        q1 = (q0 + 1) % n;
                    }
                    gates.push(("CZ".into(), vec![q0, q1]));
                }
            }
        }
        gates
    }

    #[test]
    fn test_sample_deterministic_and_block_totals() {
        // Same seed → identical counts; totals equal shots for block-aligned,
        // tail-block, and single-shot runs (exercises the 64-shot tail).
        let n = 5usize;
        let gates: Vec<(String, Vec<usize>)> = vec![
            ("H".into(), vec![0]),
            ("CX".into(), vec![0, 1]),
            ("H".into(), vec![2]),
            ("CZ".into(), vec![2, 3]),
            ("H".into(), vec![3]),
            ("CX".into(), vec![1, 4]),
            ("S".into(), vec![4]),
            ("H".into(), vec![4]),
        ];
        let tab = StabilizerTableau::from_gate_list(n, &gates).unwrap();
        for &shots in &[1usize, 63, 64, 65, 500] {
            let a = tab.sample(shots, Some(4242));
            let b = tab.sample(shots, Some(4242));
            assert_eq!(a, b, "same-seed mismatch shots={}", shots);
            assert_eq!(
                a.values().sum::<usize>(),
                shots,
                "total mismatch shots={}",
                shots
            );
        }
    }

    #[test]
    fn test_sample_marginals_match_exact() {
        // Statistical parity of the bit-sliced sampler vs exact <Z_q>.
        let n = 5usize;
        let gates: Vec<(String, Vec<usize>)> = vec![
            ("H".into(), vec![0]),
            ("CX".into(), vec![0, 1]),
            ("H".into(), vec![2]),
            ("CZ".into(), vec![2, 3]),
            ("H".into(), vec![3]),
            ("CX".into(), vec![1, 4]),
            ("S".into(), vec![4]),
            ("H".into(), vec![4]),
        ];
        let tab = StabilizerTableau::from_gate_list(n, &gates).unwrap();
        let shots = 20000usize;
        let counts = tab.sample(shots, Some(7));
        for q in 0..n {
            let px = vec![0u8; n];
            let mut pz = vec![0u8; n];
            pz[q] = 1;
            let exact = tab.pauli_expval(&px, &pz);
            let mut ones = 0.0f64;
            for (bs, c) in &counts {
                let v = u64::from_str_radix(bs, 2).unwrap();
                ones += ((v >> q) & 1) as f64 * *c as f64;
            }
            let m = 1.0 - 2.0 * ones / shots as f64;
            let sigma = ((1.0 - exact * exact).max(1e-12) / shots as f64).sqrt();
            assert!(
                (m - exact).abs() <= 5.0 * sigma,
                "q={} sampled={} exact={} 5sigma={}",
                q,
                m,
                exact,
                5.0 * sigma
            );
        }
    }

    #[test]
    fn test_ghz_plan_structure_and_parities() {
        // GHZ: only the first measured qubit is random; every later outcome is
        // a deterministic parity of it - exercises the Determined branch.
        let n = 8usize;
        let mut gates: Vec<(String, Vec<usize>)> = vec![("H".into(), vec![0])];
        for q in 0..(n - 1) {
            gates.push(("CX".into(), vec![q, q + 1]));
        }
        let tab = StabilizerTableau::from_gate_list(n, &gates).unwrap();
        let plan = tab.build_sample_plan();
        assert!(matches!(&plan[0], PlanEntry::Random { .. }));
        for e in plan.iter().skip(1) {
            assert!(matches!(e, PlanEntry::Determined { .. }));
        }
        let counts = tab.sample(300, Some(11));
        for bs in counts.keys() {
            assert!(
                bs == "00000000" || bs == "11111111",
                "unexpected GHZ string {}",
                bs
            );
        }
    }

    #[test]
    fn test_plan_cache_invalidated_on_mutation() {
        // A cached plan must never outlive a tableau mutation: sample (which
        // fills the cache), apply a gate, then sample again — the result must
        // match a freshly built tableau carrying the full gate list.
        let mut tab = StabilizerTableau::new(4);
        tab.h(0);
        tab.cnot(0, 1);
        let _ = tab.sample(128, Some(3)); // populates the plan cache
        tab.h(2);
        tab.cnot(2, 3);
        let mutated = tab.sample(256, Some(9));

        let mut fresh = StabilizerTableau::new(4);
        fresh.h(0);
        fresh.cnot(0, 1);
        fresh.h(2);
        fresh.cnot(2, 3);
        let expected = fresh.sample(256, Some(9));
        assert_eq!(mutated, expected);
    }

    #[test]
    fn test_pauli_twirl_preserves_count() {
        let gates = vec![
            ("H".to_string(), vec![0usize]),
            ("CX".to_string(), vec![0usize, 1usize]),
            ("H".to_string(), vec![1usize]),
        ];
        let twirled = pauli_twirl_gate_list(&gates, 42);
        assert!(twirled.len() >= 3);
    }
}
