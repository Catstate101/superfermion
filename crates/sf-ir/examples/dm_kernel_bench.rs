//! TEMPORARY measurement harness for density-matrix kernel variants.
//!
//! Times the current `dm_super_2q` shape (16-element joint blocks, 16×16
//! superoperator) against restructured variants, plus the 1q pair-batch
//! kernel (two 4×4 superops inside one 16-block sweep). Run with:
//!   cargo run --release -p sf-ir --example dm_kernel_bench
//!
//! Not part of the library: nothing here is linked into the wheel.

use num_complex::Complex64;
use rayon::prelude::*;
use std::time::Instant;

// ── Send+Sync wrapper for the raw buffer pointer (mirrors simd.rs DmPtr) ──
#[derive(Clone, Copy)]
struct P(*mut Complex64);
unsafe impl Send for P {}
unsafe impl Sync for P {}
impl P {
    #[inline(always)]
    unsafe fn get(&self, i: usize) -> Complex64 {
        *self.0.add(i)
    }
    #[inline(always)]
    unsafe fn set(&self, i: usize, v: Complex64) {
        *self.0.add(i) = v;
    }
    #[inline(always)]
    unsafe fn prefetch(&self, i: usize) {
        #[cfg(target_arch = "x86_64")]
        std::arch::x86_64::_mm_prefetch(self.0.add(i) as *const i8, std::arch::x86_64::_MM_HINT_T0);
        #[cfg(not(target_arch = "x86_64"))]
        {
            let _ = i;
        }
    }
}

/// Read-side peer of `P` (raw const pointer with the same Send+Sync bound).
#[derive(Clone, Copy)]
struct C(*const Complex64);
unsafe impl Send for C {}
unsafe impl Sync for C {}
impl C {
    #[inline(always)]
    unsafe fn copy_to(&self, dst: &P, i: usize, len: usize) {
        std::ptr::copy_nonoverlapping(self.0.add(i), dst.0.add(i), len);
    }
}

#[inline(always)]
fn deposit_zeros(mut g: usize, positions: &[usize]) -> usize {
    for &p in positions {
        let mask = (1usize << p) - 1;
        g = (g & mask) | ((g & !mask) << 1);
    }
    g
}

/// Joint-block geometry shared by all 2-qubit-shaped variants.
struct Geo {
    positions: [usize; 4],
    ket_off: [usize; 4],
    bra_off: [usize; 4],
    n_groups: usize,
}

fn geo(len: usize, q0: usize, q1: usize) -> Geo {
    let n = len.trailing_zeros() as usize / 2;
    let lo = q0.min(q1);
    let hi = q0.max(q1);
    let mq0 = 1usize << q0;
    let mq1 = 1usize << q1;
    let mb0 = 1usize << (n + q0);
    let mb1 = 1usize << (n + q1);
    Geo {
        positions: [lo, hi, n + lo, n + hi],
        ket_off: [0usize, mq1, mq0, mq0 | mq1],
        bra_off: [0usize, mb1, mb0, mb0 | mb1],
        n_groups: len >> 4,
    }
}

// ── Variant 1: current published shape (optional software prefetch) ──
fn super2q_cur(
    s: &mut [Complex64],
    q0: usize,
    q1: usize,
    m: &[[Complex64; 16]; 16],
    pf_dist: usize,
) {
    let g = geo(s.len(), q0, q1);
    let sp = P(s.as_mut_ptr());
    (0..g.n_groups).into_par_iter().for_each(|gi| unsafe {
        if pf_dist > 0 && gi + pf_dist < g.n_groups {
            let base = deposit_zeros(gi + pf_dist, &g.positions);
            for k in 0..4 {
                for b in 0..4 {
                    sp.prefetch(base | g.ket_off[k] | g.bra_off[b]);
                }
            }
        }
        let base = deposit_zeros(gi, &g.positions);
        let mut v = [Complex64::new(0.0, 0.0); 16];
        for k in 0..4 {
            for b in 0..4 {
                v[4 * k + b] = sp.get(base | g.ket_off[k] | g.bra_off[b]);
            }
        }
        for k in 0..4 {
            for b in 0..4 {
                let row = 4 * k + b;
                let mut acc = Complex64::new(0.0, 0.0);
                for col in 0..16 {
                    acc += m[row][col] * v[col];
                }
                sp.set(base | g.ket_off[k] | g.bra_off[b], acc);
            }
        }
    });
}

// ── Variant 2: scalar, 4 independent row accumulators (ILP) ──
fn super2q_ilp4(s: &mut [Complex64], q0: usize, q1: usize, m: &[[Complex64; 16]; 16]) {
    let g = geo(s.len(), q0, q1);
    let sp = P(s.as_mut_ptr());
    (0..g.n_groups).into_par_iter().for_each(|gi| unsafe {
        let base = deposit_zeros(gi, &g.positions);
        let mut v = [Complex64::new(0.0, 0.0); 16];
        for k in 0..4 {
            for b in 0..4 {
                v[4 * k + b] = sp.get(base | g.ket_off[k] | g.bra_off[b]);
            }
        }
        let mut out = [Complex64::new(0.0, 0.0); 16];
        for rp in 0..4 {
            let r0 = 4 * rp;
            let (mut a0, mut a1, mut a2, mut a3) = (
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
            );
            for col in 0..16 {
                let vc = v[col];
                a0 += m[r0][col] * vc;
                a1 += m[r0 + 1][col] * vc;
                a2 += m[r0 + 2][col] * vc;
                a3 += m[r0 + 3][col] * vc;
            }
            out[r0] = a0;
            out[r0 + 1] = a1;
            out[r0 + 2] = a2;
            out[r0 + 3] = a3;
        }
        for k in 0..4 {
            for b in 0..4 {
                sp.set(base | g.ket_off[k] | g.bra_off[b], out[4 * k + b]);
            }
        }
    });
}

// ── Variant 3: AVX2 row-pairs over transposed m (cmul4), serial kernel
//             called per chunk (rayon layer stays outside target_feature).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn super2q_avx2_chunk(sp: P, m_t: &[[Complex64; 16]; 16], g: &Geo, gi0: usize, gi1: usize) {
    use std::arch::x86_64::*;
    for gi in gi0..gi1 {
        let base = deposit_zeros(gi, &g.positions);
        let mut v = [Complex64::new(0.0, 0.0); 16];
        for k in 0..4 {
            for b in 0..4 {
                v[4 * k + b] = sp.get(base | g.ket_off[k] | g.bra_off[b]);
            }
        }
        let mut acc = [_mm256_setzero_pd(); 8];
        for col in 0..16 {
            let vc = _mm256_set_pd(v[col].im, v[col].re, v[col].im, v[col].re);
            let mcol = m_t[col].as_ptr();
            for p in 0..8 {
                let mv = _mm256_loadu_pd(mcol.add(2 * p) as *const f64);
                let vr = _mm256_movedup_pd(mv);
                let vi = _mm256_permute_pd(mv, 0b1111);
                let vs = _mm256_permute_pd(vc, 0b0101);
                let prod = _mm256_fmaddsub_pd(vr, vc, _mm256_mul_pd(vi, vs));
                acc[p] = _mm256_add_pd(acc[p], prod);
            }
        }
        for p in 0..8 {
            _mm256_storeu_pd(v.as_mut_ptr().add(2 * p) as *mut f64, acc[p]);
        }
        for k in 0..4 {
            for b in 0..4 {
                sp.set(base | g.ket_off[k] | g.bra_off[b], v[4 * k + b]);
            }
        }
    }
}

fn super2q_avx2(s: &mut [Complex64], q0: usize, q1: usize, m_t: &[[Complex64; 16]; 16]) {
    let g = geo(s.len(), q0, q1);
    let sp = P(s.as_mut_ptr());
    let nchunks = (rayon::current_num_threads() * 8).max(1);
    #[cfg(target_arch = "x86_64")]
    {
        let per = g.n_groups.div_ceil(nchunks);
        (0..nchunks).into_par_iter().for_each(|c| {
            let gi0 = c * per;
            let gi1 = ((c + 1) * per).min(g.n_groups);
            if gi0 < gi1 {
                unsafe { super2q_avx2_chunk(sp, m_t, &g, gi0, gi1) };
            }
        });
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = (sp, m_t, nchunks);
    }
}

// ── Variant 4: 1q pair-batch — two 4×4 superops in one 16-block sweep ──
// Applies m_a (qubit qa) then m_b (qubit qb) with the same per-element
// arithmetic order as two sequential `dm_super_1q` sweeps.
fn super1q_pair(
    s: &mut [Complex64],
    qa: usize,
    m_a: &[[Complex64; 4]; 4],
    qb: usize,
    m_b: &[[Complex64; 4]; 4],
) {
    let g = geo(s.len(), qa, qb);
    // Axis order inside the block: axis 0 = lo bit, axis 1 = hi bit.
    let lo = qa.min(qb);
    let (axis_a, axis_b) = if qa == lo {
        (0usize, 1usize)
    } else {
        (1usize, 0usize)
    };
    let sp = P(s.as_mut_ptr());
    (0..g.n_groups).into_par_iter().for_each(|gi| {
        let base = deposit_zeros(gi, &g.positions);
        let mut v = [Complex64::new(0.0, 0.0); 16];
        unsafe {
            for k in 0..4 {
                for b in 0..4 {
                    v[4 * k + b] = sp.get(base | g.ket_off[k] | g.bra_off[b]);
                }
            }
        }
        apply_1q_axis(&mut v, axis_a, m_a);
        apply_1q_axis(&mut v, axis_b, m_b);
        unsafe {
            for k in 0..4 {
                for b in 0..4 {
                    sp.set(base | g.ket_off[k] | g.bra_off[b], v[4 * k + b]);
                }
            }
        }
    });
}

/// Apply the 4×4 superop `m` to the sub-vector varying qubit `axis`
/// (0 = lo bit, 1 = hi bit) of the 16-block `v`, for each fixed
/// combination of the sibling qubit's coordinates. Mirrors the exact
/// expression order of `dm_super_1q`.
#[inline]
fn apply_1q_axis(v: &mut [Complex64; 16], axis: usize, m: &[[Complex64; 4]; 4]) {
    for other in 0..4 {
        // Coordinates of the sibling qubit (0b(k2,b2)).
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

// ── Variant 5: k-qubit batch of 1q superops (k = 2..=4) ──
// One sweep over the joint 4^k blocks; each op's 4-cycle arithmetic has
// the exact per-element expression order of `dm_super_1q`, so the result
// is identical to k sequential single-qubit sweeps.
fn super1q_batch(s: &mut [Complex64], qubits: &[usize], ms: &[[[Complex64; 4]; 4]]) {
    let k = qubits.len();
    let len = s.len();
    let n = len.trailing_zeros() as usize / 2;
    let blk_bits = 2 * k;
    let n_groups = len >> blk_bits;
    let blk = 1usize << blk_bits;
    // local bit 2i = ket of qubits[i], 2i+1 = bra of qubits[i]
    let mut off = vec![0usize; blk];
    for (j, o) in off.iter_mut().enumerate() {
        let mut v = 0usize;
        for i in 0..k {
            if (j >> (2 * i)) & 1 == 1 {
                v |= 1usize << qubits[i];
            }
            if (j >> (2 * i + 1)) & 1 == 1 {
                v |= 1usize << (n + qubits[i]);
            }
        }
        *o = v;
    }
    let sib_local: Vec<Vec<usize>> = (0..k)
        .map(|i| {
            (0..(1usize << (blk_bits - 2)))
                .map(|pat| deposit_zeros(pat, &[2 * i, 2 * i + 1]))
                .collect()
        })
        .collect();
    let pos = positions_of(qubits, n);
    let sp = P(s.as_mut_ptr());
    (0..n_groups).into_par_iter().for_each(|gi| {
        let base = deposit_zeros(gi, &pos);
        let mut v = [Complex64::new(0.0, 0.0); 256];
        unsafe {
            for (j, o) in off.iter().enumerate() {
                v[j] = sp.get(base | o);
            }
        }
        for i in 0..k {
            let m = &ms[i];
            let kl = 1usize << (2 * i);
            let bl = 1usize << (2 * i + 1);
            for &sbl in &sib_local[i] {
                let i00 = sbl;
                let i01 = sbl | kl; // ket bit set, bra clear
                let i10 = sbl | bl; // ket clear, bra set
                let i11 = sbl | kl | bl;
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
        unsafe {
            for (j, o) in off.iter().enumerate() {
                sp.set(base | o, v[j]);
            }
        }
    });
}

fn positions_of(qubits: &[usize], n: usize) -> Vec<usize> {
    let mut p: Vec<usize> = qubits.to_vec();
    p.extend(qubits.iter().map(|q| n + q));
    p.sort_unstable();
    p
}

// ── Reference: plain parallel memcpy of the buffer (machine floor) ──
fn par_copy(s: &mut [Complex64], src: &[Complex64]) {
    let n = s.len();
    let chunks = 1024;
    let cl = n / chunks;
    let dp = P(s.as_mut_ptr());
    let sp = C(src.as_ptr());
    (0..chunks).into_par_iter().for_each(|c| unsafe {
        sp.copy_to(&dp, c * cl, cl);
    });
}

/// Single 1q superop sweep (mirror of `dm_super_1q`) for the pair comparison.
fn super1q_single(s: &mut [Complex64], q: usize, m: &[[Complex64; 4]; 4]) {
    let len = s.len();
    let n = len.trailing_zeros() as usize / 2;
    let nq = n + q;
    let mq = 1usize << q;
    let mnq = 1usize << nq;
    let positions = [q, nq];
    let n_groups = len >> 2;
    let sp = P(s.as_mut_ptr());
    (0..n_groups).into_par_iter().for_each(|g| unsafe {
        let base = deposit_zeros(g, &positions);
        let i00 = base;
        let i01 = base | mq;
        let i10 = base | mnq;
        let i11 = base | mq | mnq;
        let b00 = sp.get(i00);
        let b01 = sp.get(i10);
        let b10 = sp.get(i01);
        let b11 = sp.get(i11);
        sp.set(
            i00,
            m[0][0] * b00 + m[0][1] * b01 + m[0][2] * b10 + m[0][3] * b11,
        );
        sp.set(
            i10,
            m[1][0] * b00 + m[1][1] * b01 + m[1][2] * b10 + m[1][3] * b11,
        );
        sp.set(
            i01,
            m[2][0] * b00 + m[2][1] * b01 + m[2][2] * b10 + m[2][3] * b11,
        );
        sp.set(
            i11,
            m[3][0] * b00 + m[3][1] * b01 + m[3][2] * b10 + m[3][3] * b11,
        );
    });
}

fn timeit(mut f: impl FnMut(), reps: usize) -> (f64, f64) {
    f();
    let mut ts = Vec::with_capacity(reps);
    for _ in 0..reps {
        let t = Instant::now();
        f();
        ts.push(t.elapsed().as_secs_f64() * 1e3);
    }
    ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (ts[0], ts[ts.len() / 2])
}

fn lcg(n: usize, seed: u64) -> Vec<Complex64> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let re = ((s >> 11) as f64 / (1u64 << 53) as f64) - 0.5;
            let im = ((s >> 33) as f64 / (1u64 << 31) as f64) - 0.5;
            Complex64::new(re, im)
        })
        .collect()
}

fn maxdiff(a: &[Complex64], b: &[Complex64]) -> f64 {
    let scale = b.iter().map(|x| x.norm()).fold(1e-300f64, f64::max);
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).norm())
        .fold(0.0f64, f64::max)
        / scale
}

fn main() {
    let n = 10usize;
    let len = 1usize << (2 * n);
    let base = lcg(len, 12345);
    let mut m = [[Complex64::new(0.0, 0.0); 16]; 16];
    let mr = lcg(256, 999);
    for r in 0..16 {
        for c in 0..16 {
            m[r][c] = mr[r * 16 + c];
        }
    }
    let mut m_t = [[Complex64::new(0.0, 0.0); 16]; 16];
    for r in 0..16 {
        for c in 0..16 {
            m_t[c][r] = m[r][c];
        }
    }
    let (q0, q1) = (4usize, 5usize); // adjacent-qubit cx, the bench shape

    println!(
        "len = {len} complex ({:.1} MB), threads = {}",
        (len * 16) as f64 / 1e6,
        rayon::current_num_threads()
    );

    let mut refbuf = base.clone();
    super2q_cur(&mut refbuf, q0, q1, &m, 0);

    let mut dst = vec![Complex64::new(0.0, 0.0); len];
    let (mn, md) = timeit(|| par_copy(&mut dst, &base), 15);
    println!(
        "par memcpy      : min={mn:7.3} med={md:7.3}  ({:.1} GB/s)",
        32.0 / (mn / 1e3) / 1e3
    );

    for pf in [16usize, 0] {
        let mut buf = base.clone();
        let (mn, md) = timeit(|| super2q_cur(&mut buf, q0, q1, &m, pf), 15);
        let mut one = base.clone();
        super2q_cur(&mut one, q0, q1, &m, pf);
        let d = maxdiff(&one, &refbuf);
        println!(
            "cur pf={pf:<3}      : min={mn:7.3} med={md:7.3}  rel={:.2e}",
            d
        );
    }
    {
        let mut buf = base.clone();
        let (mn, md) = timeit(|| super2q_ilp4(&mut buf, q0, q1, &m), 15);
        let mut one = base.clone();
        super2q_ilp4(&mut one, q0, q1, &m);
        let d = maxdiff(&one, &refbuf);
        println!("ilp4            : min={mn:7.3} med={md:7.3}  rel={:.2e}", d);
    }
    {
        let mut buf = base.clone();
        let (mn, md) = timeit(|| super2q_avx2(&mut buf, q0, q1, &m_t), 15);
        let mut one = base.clone();
        super2q_avx2(&mut one, q0, q1, &m_t);
        let d = maxdiff(&one, &refbuf);
        println!("avx2 rowpair    : min={mn:7.3} med={md:7.3}  rel={:.2e}", d);
    }

    // 1q pair batch vs two single sweeps
    let ma = {
        let r = lcg(16, 4242);
        let mut a = [[Complex64::new(0.0, 0.0); 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                a[i][j] = r[i * 4 + j];
            }
        }
        a
    };
    let mb = {
        let r = lcg(16, 777);
        let mut a = [[Complex64::new(0.0, 0.0); 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                a[i][j] = r[i * 4 + j];
            }
        }
        a
    };
    let mut seq = base.clone();
    let (mn_a, _) = timeit(|| super1q_single(&mut seq, 4, &ma), 15);
    let (mn_b, _) = timeit(|| super1q_single(&mut seq, 5, &mb), 15);
    println!(
        "two 1q sweeps   : min={mn_a:7.3} + {mn_b:7.3} = {:.3}",
        mn_a + mn_b
    );
    {
        let mut buf = base.clone();
        let (mn, md) = timeit(|| super1q_pair(&mut buf, 4, &ma, 5, &mb), 15);
        let mut seqb = base.clone();
        super1q_single(&mut seqb, 4, &ma);
        super1q_single(&mut seqb, 5, &mb);
        let mut one = base.clone();
        super1q_pair(&mut one, 4, &ma, 5, &mb);
        let d = maxdiff(&one, &seqb);
        println!("pair batch 1q   : min={mn:7.3} med={md:7.3}  rel={:.2e}", d);
    }
    {
        // batched with a gapped qubit pair (0 and 9) as a stress case
        let mut buf = base.clone();
        let (mn, md) = timeit(|| super1q_pair(&mut buf, 0, &ma, 9, &mb), 15);
        let mut seqb = base.clone();
        super1q_single(&mut seqb, 0, &ma);
        super1q_single(&mut seqb, 9, &mb);
        let mut one = base.clone();
        super1q_pair(&mut one, 0, &ma, 9, &mb);
        let d = maxdiff(&one, &seqb);
        println!("pair batch (0,9): min={mn:7.3} med={md:7.3}  rel={:.2e}", d);
    }
    // generalized k-qubit batch: k=2 must match the pair kernel, k=3/4 the
    // shapes used for the tail of the benchmark circuit.
    let mut alph = [[[Complex64::new(0.0, 0.0); 4]; 4]; 4];
    for (i, a) in alph.iter_mut().enumerate() {
        let r = lcg(16, 1000 + i as u64);
        for x in 0..4 {
            for y in 0..4 {
                a[x][y] = r[x * 4 + y];
            }
        }
    }
    for k in [2usize, 3, 4] {
        let qubits: Vec<usize> = (0..k).collect();
        let mut buf = base.clone();
        let (mn, md) = timeit(|| super1q_batch(&mut buf, &qubits, &alph[..k]), 15);
        let mut seqb = base.clone();
        for (i, &q) in qubits.iter().enumerate() {
            super1q_single(&mut seqb, q, &alph[i]);
        }
        let mut one = base.clone();
        super1q_batch(&mut one, &qubits, &alph[..k]);
        let d = maxdiff(&one, &seqb);
        // sequential reference time for the same k sweeps
        let mut seqbuf = base.clone();
        let mut seq_tot = 0.0f64;
        for (i, &q) in qubits.iter().enumerate() {
            let (s1, _) = timeit(|| super1q_single(&mut seqbuf, q, &alph[i]), 15);
            seq_tot += s1;
        }
        println!(
            "batch k={k}       : min={mn:7.3} med={md:7.3}  rel={:.2e}  (seq {:.3})",
            d, seq_tot
        );
    }
}
