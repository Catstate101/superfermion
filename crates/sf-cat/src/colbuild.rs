//! Column-major Clifford tableau builder — the fused build path for
//! [`FrameTableau::from_gate_list`](crate::paulframe::FrameTableau::from_gate_list).
//!
//! The row-major builder (`paulframe::apply_row`) walks every row across the
//! whole op list, so its inner iteration is *one bit of one row* and the per-op
//! `match` sits inside that loop: `2n * |ops|` branchy bit updates.
//!
//! This module builds the identical tableau in **column** order instead.  Qubit
//! `q` owns an `rw`-word bit vector over the block's rows, so
//!
//! * one gate is a handful of word-wide XOR/AND ops (the op dispatch is hoisted
//!   out of the row loop, and the loops vectorize), and
//! * a single-qubit gate touches `2 * rw` words instead of `2n` rows.
//!
//! Rows are independent under every Clifford primitive (each gate maps every
//! row through the same row-local update), so the `2n` rows are split into
//! per-thread blocks; each block runs the whole op list on its own column
//! vectors, then transposes back into the row-major tableau with a blocked
//! 64x64 bit transpose.  The result is **bit-for-bit identical** to the
//! row-major path — pinned by
//! `paulframe::tests::column_build_matches_row_build`.
//!
//! Nothing outside `sf-cat` is touched, and the row-major path stays available
//! (`SF_CAT_BUILD=rows`) as the reference.

use crate::paulframe::COp;

/// One contiguous block of row ranges, held column-major.
///
/// `x[q * rw + w]` is bit `j = 64 * w + i` of qubit `q`'s X column for block
/// row `i`; `z` has the same shape; `r[w]` holds the block's row phases.
struct Block {
    n: usize,
    rw: usize,
    rows: usize,
    x: Vec<u64>,
    z: Vec<u64>,
    r: Vec<u64>,
}

impl Block {
    /// Fresh |0...0> tableau restricted to `rows` rows starting at global row
    /// `base`: rows `[0, n)` are destabilizers `X_q`, rows `[n, 2n)` are
    /// stabilizers `Z_q` (same init as `FrameTableau::new`).
    fn new(n: usize, base: usize, rows: usize) -> Self {
        let rw = rows.div_ceil(64).max(1);
        let mut x = vec![0u64; n * rw];
        let mut z = vec![0u64; n * rw];
        for j in 0..rows {
            let gr = base + j;
            let w = j >> 6;
            let b = 1u64 << (j & 63);
            if gr < n {
                x[gr * rw + w] |= b; // destabilizer q = X_q
            } else {
                z[(gr - n) * rw + w] |= b; // stabilizer q = Z_q
            }
        }
        Self {
            n,
            rw,
            rows,
            x,
            z,
            r: vec![0u64; rw],
        }
    }

    /// Apply the whole op list to this block, one word-wide pass per gate.
    fn apply(&mut self, cops: &[COp]) {
        let rw = self.rw;
        let n = self.n;
        let (x, z, r) = (&mut self.x, &mut self.z, &mut self.r);
        for &op in cops {
            match op {
                COp::H(q) => {
                    let q = q as usize;
                    debug_assert!(q < n);
                    let o = q * rw;
                    for w in 0..rw {
                        let xw = x[o + w];
                        let zw = z[o + w];
                        r[w] ^= xw & zw;
                        x[o + w] = zw;
                        z[o + w] = xw;
                    }
                }
                COp::S(q) => op_s(x, z, r, q as usize, rw),
                COp::Sdg(q) => {
                    // Same fold as the row-major `sdg` (three S passes).
                    let q = q as usize;
                    op_s(x, z, r, q, rw);
                    op_s(x, z, r, q, rw);
                    op_s(x, z, r, q, rw);
                }
                COp::X(q) => {
                    let o = q as usize * rw;
                    for w in 0..rw {
                        r[w] ^= z[o + w];
                    }
                }
                COp::Z(q) => {
                    let o = q as usize * rw;
                    for w in 0..rw {
                        r[w] ^= x[o + w];
                    }
                }
                COp::Y(q) => {
                    let o = q as usize * rw;
                    for w in 0..rw {
                        r[w] ^= x[o + w] ^ z[o + w];
                    }
                }
                COp::Cnot(a, b) => {
                    let (oa, ob) = (a as usize * rw, b as usize * rw);
                    for w in 0..rw {
                        let xa = x[oa + w];
                        let za = z[oa + w];
                        let xb = x[ob + w];
                        let zb = z[ob + w];
                        r[w] ^= xa & zb & !(xb ^ za);
                        x[ob + w] = xb ^ xa;
                        z[oa + w] = za ^ zb;
                    }
                }
                COp::Swap(a, b) => {
                    let (oa, ob) = (a as usize * rw, b as usize * rw);
                    for w in 0..rw {
                        x.swap(oa + w, ob + w);
                        z.swap(oa + w, ob + w);
                    }
                }
            }
        }
    }

    /// Write the block back into the row-major tableau slice (block-local row
    /// indices), transposing 64 columns x 64 rows at a time.
    fn scatter(&self, x_out: &mut [u64], z_out: &mut [u64], r_out: &mut [u8], words: usize) {
        for j in 0..self.rows {
            r_out[j] = ((self.r[j >> 6] >> (j & 63)) & 1) as u8;
        }
        let mut xb = [0u64; 64];
        let mut zb = [0u64; 64];
        for w in 0..words {
            let q0 = 64 * w;
            for rb in 0..self.rw {
                let cols = self.n.saturating_sub(q0).min(64);
                for i in 0..cols {
                    let o = (q0 + i) * self.rw + rb;
                    xb[i] = self.x[o];
                    zb[i] = self.z[o];
                }
                for i in cols..64 {
                    xb[i] = 0;
                    zb[i] = 0;
                }
                transpose64(&mut xb);
                transpose64(&mut zb);
                let rows_here = self.rows.saturating_sub(64 * rb).min(64);
                let dst = 64 * rb * words + w;
                for j in 0..rows_here {
                    x_out[dst + j * words] = xb[j];
                    z_out[dst + j * words] = zb[j];
                }
            }
        }
    }
}

/// `S` on one column pair: `r ^= x & z; z ^= x` (mirrors `FrameTableau::s`).
#[inline(always)]
fn op_s(x: &mut [u64], z: &mut [u64], r: &mut [u64], q: usize, rw: usize) {
    let o = q * rw;
    for w in 0..rw {
        let xw = x[o + w];
        r[w] ^= xw & z[o + w];
        z[o + w] ^= xw;
    }
}

/// In-place 64x64 bit-matrix transpose: on return `a[c]` bit `r` equals the
/// input `a[r]` bit `c`.
///
/// Written as an explicit gather so its contract is auditable at a glance;
/// it is a one-off cost paid once per 64-row tile at the end of a build
/// (`2 * ceil(rows/64) * ceil(n/64)` tiles), far below the op pass itself.
fn transpose64(a: &mut [u64; 64]) {
    let mut out = [0u64; 64];
    for (r, dst) in out.iter_mut().enumerate() {
        let mut word = 0u64;
        for (c, &col) in a.iter().enumerate() {
            word |= ((col >> r) & 1) << c;
        }
        *dst = word;
    }
    *a = out;
}

/// Build the row-major `(x, z, r)` tableau for `cops` from |0...0>.
///
/// `par_min` is the same work threshold the row-major path uses
/// (`SF_CAT_STAB_PAR_MIN`, work = `rows * ops`); below it (or with one thread)
/// the build runs in the calling thread.
pub(crate) fn build(
    n: usize,
    cops: &[COp],
    threads: usize,
    par_min: usize,
) -> (Vec<u64>, Vec<u64>, Vec<u8>) {
    let words = n.div_ceil(64);
    let rows = 2 * n;
    let mut x = vec![0u64; rows * words];
    let mut z = vec![0u64; rows * words];
    let mut r = vec![0u8; rows];
    let threads = threads.max(1);
    let work = rows.saturating_mul(cops.len());
    if threads == 1 || work < par_min {
        let mut b = Block::new(n, 0, rows);
        b.apply(cops);
        b.scatter(&mut x, &mut z, &mut r, words);
        return (x, z, r);
    }
    // Round the per-thread slice up to a whole 64-row tile so every block
    // transposes full 64x64 matrices (and stay >= 1 tile per thread).
    let per = rows.div_ceil(threads).max(64);
    let blk = per.div_ceil(64) * 64;
    use rayon::prelude::*;
    x.par_chunks_mut(blk * words)
        .zip_eq(z.par_chunks_mut(blk * words))
        .zip_eq(r.par_chunks_mut(blk))
        .enumerate()
        .for_each(|(bi, ((xs, zs), rs))| {
            let base = bi * blk;
            let rows_b = xs.len() / words;
            let mut b = Block::new(n, base, rows_b);
            b.apply(cops);
            b.scatter(xs, zs, rs, words);
        });
    (x, z, r)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transpose64_matches_naive() {
        let mut state = 0x1234_5678_9abc_def0u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for _ in 0..32 {
            let mut a = [0u64; 64];
            for v in a.iter_mut() {
                *v = next();
            }
            let mut want = [0u64; 64];
            for (r, &row) in a.iter().enumerate() {
                for c in 0..64 {
                    if (row >> c) & 1 != 0 {
                        want[c] |= 1u64 << r;
                    }
                }
            }
            transpose64(&mut a);
            assert_eq!(a, want);
        }
        // Transposing twice is the identity.
        let mut a = [0u64; 64];
        for (i, v) in a.iter_mut().enumerate() {
            *v = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        }
        let orig = a;
        transpose64(&mut a);
        transpose64(&mut a);
        assert_eq!(a, orig);
    }
}
