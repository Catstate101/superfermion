//! Clifford stabilizer tableau (Aaronson–Gottesman 2004) + standalone Pauli twirl.
//!
//! `StabilizerTableau` — O(n) gate updates, O(n³) sampling via AG algorithm 1.
//! Word-packed representation (ceil(n/64) u64s per row) supports n ≤ 1024.
//! `pauli_twirl_gate_list` — standalone gate-level Pauli twirl, no DAG dependency.

use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use rayon::prelude::*;
use std::collections::HashMap;

// ═══════════════════════════════════════════════════════════
// Stabilizer Tableau (word-packed for n ≤ 1024)
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
}

impl StabilizerTableau {
    pub fn new(n: usize) -> Self {
        assert!(n <= 1024, "Tableau supports n ≤ 1024");
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
        }
    }

    // ── Single-qubit gates ──

    pub fn h(&mut self, q: usize) {
        let w = q / 64;
        let mask = 1u64 << (q % 64);
        for i in 0..2 * self.n {
            let xq = (self.x[i][w] >> (q % 64)) & 1;
            let zq = (self.z[i][w] >> (q % 64)) & 1;
            self.r[i] ^= xq & zq;
            self.x[i][w] = (self.x[i][w] & !mask) | (zq << (q % 64));
            self.z[i][w] = (self.z[i][w] & !mask) | (xq << (q % 64));
        }
    }

    pub fn s(&mut self, q: usize) {
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
        let w = q / 64;
        for i in 0..2 * self.n {
            self.r[i] ^= (self.z[i][w] >> (q % 64)) & 1;
        }
    }

    pub fn z_gate(&mut self, q: usize) {
        let w = q / 64;
        for i in 0..2 * self.n {
            self.r[i] ^= (self.x[i][w] >> (q % 64)) & 1;
        }
    }

    pub fn y_gate(&mut self, q: usize) {
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
    pub fn from_gate_list(n: usize, gates: &[(String, Vec<usize>)]) -> Result<Self, String> {
        let mut tab = Self::new(n);
        for (name, qubits) in gates {
            tab.apply_gate(name, qubits)?;
        }
        Ok(tab)
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

    fn measure_z(&mut self, q: usize, rng: &mut StdRng) -> u8 {
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

    /// Sample `shots` bitstrings. Uses rayon for multi-core parallelism.
    /// Each shot clones the tableau (measure_z is destructive, so a fresh
    /// clone is required per shot).  The O(n³) per-shot cost is dominated by
    /// the O(n²) measure_z row_mult work, not cloning.
    pub fn sample(&self, shots: usize, seed: Option<u64>) -> HashMap<String, usize> {
        if shots == 0 {
            return HashMap::new();
        }
        let base_seed = seed.unwrap_or_else(rand::random);
        let n = self.n;

        // Parallel sampling: each shot is independent (measure_z is destructive)
        (0..shots)
            .into_par_iter()
            .map(|shot_idx| {
                let mut tab = self.clone();
                let mut rng = StdRng::seed_from_u64(base_seed.wrapping_add(shot_idx as u64));
                let mut bits = Vec::with_capacity(n);
                for q in 0..n {
                    bits.push(tab.measure_z(q, &mut rng));
                }
                bits.iter()
                    .map(|b| if *b == 1 { '1' } else { '0' })
                    .collect::<String>()
            })
            .fold(
                HashMap::new,
                |mut acc: HashMap<String, usize>, bs: String| {
                    *acc.entry(bs).or_insert(0) += 1;
                    acc
                },
            )
            .reduce(
                HashMap::new,
                |mut a: HashMap<String, usize>, b: HashMap<String, usize>| {
                    for (k, v) in b {
                        *a.entry(k).or_insert(0) += v;
                    }
                    a
                },
            )
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
        // Bell-like on qubits 50,80: should only see 0 on qubit 50 XOR qubit 80
        for (bs, _) in &counts {
            let b50 = bs.chars().nth(50).unwrap();
            let b80 = bs.chars().nth(80).unwrap();
            assert_eq!(b50, b80, "Expected correlated bits 50,80 in {}", bs);
        }
    }

    #[test]
    fn test_tableau_swap() {
        let mut tab = StabilizerTableau::new(3);
        tab.h(0);
        tab.swap(0, 2);
        let counts = tab.sample(500, Some(99));
        // Qubit 2 now has the |+> state → ~50% 0, 50% 1 at position 2
        let ones: usize = counts
            .iter()
            .filter(|(bs, _)| bs.chars().nth(2).unwrap() == '1')
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
