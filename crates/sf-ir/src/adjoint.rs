//! Adjoint differentiation for parameterised quantum circuits.
//!
//! Computes the full gradient d<O>/d(theta) in O(M * 2^n) time
//! (one forward pass + one backward pass) regardless of the number
//! of parameters N.  This is a 2N-fold speedup over parameter-shift.
//!
//! Two memory strategies: when the parametrised-gate state cache fits the
//! cap the backward pass walks only the adjoint vector phi and never
//! re-applies states; above the cap intermediates are recomputed from |0>
//! (O(2^n) memory, 2M gate applications).  Sweeps above the parallel
//! threshold split across the rayon pool.

use crate::dag::{QuantumDAG, QuantumOp};
use crate::ops::{OpType, Parameter};
use crate::state::MethodError;
use num_complex::Complex64;
use rayon::prelude::*;

/// A Pauli term in an observable: coefficient * pauli_string.
/// Pauli encoding per qubit: 0=I, 1=X, 2=Y, 3=Z.
#[derive(Clone, Debug)]
pub struct PauliTerm {
    pub paulis: Vec<u8>,
    pub coef: Complex64,
}

/// Result of adjoint differentiation: gradient indexed by parameter name.
pub struct AdjointGradResult {
    pub param_names: Vec<String>,
    pub gradients: Vec<f64>,
}

fn generator_info(op: &OpType) -> Option<(&'static [u8], f64)> {
    match op {
        OpType::Rx(_) => Some((&[1], 0.5)),
        OpType::Ry(_) => Some((&[2], 0.5)),
        OpType::Rz(_) => Some((&[3], 0.5)),
        OpType::Rzz(_) => Some((&[3, 3], 0.5)),
        OpType::Rxx(_) => Some((&[1, 1], 0.5)),
        OpType::Ryy(_) => Some((&[2, 2], 0.5)),
        _ => None,
    }
}

fn get_param_name(op: &OpType) -> Option<String> {
    match op {
        OpType::Rx(Parameter::Variable { name, .. })
        | OpType::Ry(Parameter::Variable { name, .. })
        | OpType::Rz(Parameter::Variable { name, .. })
        | OpType::Rzz(Parameter::Variable { name, .. })
        | OpType::Rxx(Parameter::Variable { name, .. })
        | OpType::Ryy(Parameter::Variable { name, .. }) => Some(name.clone()),
        _ => None,
    }
}

// ─── In-place gate application (no allocation) ─────────────────────────

/// States at or above this dimension (n >= 16) may split their sweeps
/// across the rayon pool; below it the serial kernels win (and the
/// parallel paths are skipped entirely when the pool is pinned to one
/// thread).
const PAR_DIM: usize = 1 << 16;

thread_local! {
    /// Param-state cache buffer retained across calls so its pages stay
    /// resident.  A fresh (up to 512 MB) allocation re-faults every page on
    /// first touch — measured as the forward pass's dominant cost at
    /// n >= 16 (identical fwd time on consecutive calls).  Complex64 is
    /// Copy with no Drop, so the take/put-back round trip is plain memory
    /// ownership.
    #[allow(clippy::missing_const_for_thread_local)] // init already const; clippy 1.93 false positive
    static ADJ_CACHE_BUF: std::cell::RefCell<Vec<Complex64>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

fn apply_1q_inplace(
    state: &mut [Complex64],
    u: [[Complex64; 2]; 2],
    target: usize,
    n_qubits: usize,
    par: bool,
) {
    let _ = n_qubits;
    if par {
        let stride = 1usize << target;
        let block = stride << 1;
        let threads = rayon::current_num_threads().max(1);
        let want = (state.len() / (4 * threads)).max(block);
        let chunk = (want / block).max(1) * block;
        if chunk < state.len() {
            // Whole 2*stride blocks per chunk keep every chunk a valid
            // pair_pass input (chunked SIMD dispatch, no size gate).
            state
                .par_chunks_mut(chunk)
                .for_each(|c| crate::simd::pair_pass_chunk(c, stride, u));
            return;
        }
    }
    // SIMD pair transform (the scalar fallback below 8K amplitudes is the
    // formula-identical reference).
    crate::simd::pair_pass(state, 1usize << target, u);
}

/// dst = src, optionally across rayon tasks (element-wise, so identical).
#[inline]
fn copy_slice_par(dst: &mut [Complex64], src: &[Complex64], par: bool) {
    if par && src.len() >= PAR_DIM {
        let threads = rayon::current_num_threads().max(1);
        let chunk = ((src.len() / (4 * threads)).max(4096)).next_power_of_two();
        dst.par_chunks_mut(chunk)
            .zip(src.par_chunks(chunk))
            .for_each(|(d, s)| d.copy_from_slice(s));
    } else {
        dst.copy_from_slice(src);
    }
}

fn apply_2q_inplace(
    state: &mut [Complex64],
    gate: [[Complex64; 4]; 4],
    q1: usize,
    q2: usize,
    n_qubits: usize,
) {
    let dim = 1usize << n_qubits;
    let use_par = dim >= 1 << 16 && rayon::current_num_threads() > 1;
    crate::dag::inplace_2q_general(
        state,
        q1,
        q2,
        1usize << q1,
        1usize << q2,
        gate[0][0],
        gate[0][1],
        gate[0][2],
        gate[0][3],
        gate[1][0],
        gate[1][1],
        gate[1][2],
        gate[1][3],
        gate[2][0],
        gate[2][1],
        gate[2][2],
        gate[2][3],
        gate[3][0],
        gate[3][1],
        gate[3][2],
        gate[3][3],
        use_par,
    );
}

fn apply_pauli_1q_inplace(state: &mut [Complex64], pauli: u8, qubit: usize, n_qubits: usize) {
    let dim = 1 << n_qubits;
    let mask = 1usize << qubit;
    match pauli {
        1 => {
            // X: swap contiguous runs of 2^qubit (vectorized, branch-free).
            let run = mask;
            let block = run << 1;
            let p = state.as_mut_ptr();
            let n_blocks = dim / block;
            for b in 0..n_blocks {
                unsafe {
                    crate::simd::swap_runs(p, b * block, b * block + run, run);
                }
            }
        }
        2 => {
            let neg_i = Complex64::new(0.0, -1.0);
            let pos_i = Complex64::new(0.0, 1.0);
            for i in 0..dim {
                if (i & mask) == 0 {
                    let j = i | mask;
                    let a = state[i];
                    let b = state[j];
                    state[i] = neg_i * b;
                    state[j] = pos_i * a;
                }
            }
        }
        3 => {
            for i in 0..dim {
                if (i & mask) != 0 {
                    state[i] = -state[i];
                }
            }
        }
        _ => {}
    }
}

/// Partial ⟨φ|G|src⟩ over [lo, hi) for one single-qubit generator pauli
/// (1=X, 2=Y, 3=Z) with qubit mask `m`; `lo` must be a multiple of 2m.
#[inline]
fn dot_one_pauli(
    p: u8,
    phi: &[Complex64],
    src: &[Complex64],
    m: usize,
    lo: usize,
    hi: usize,
) -> Complex64 {
    let block = m << 1;
    let mut ip = Complex64::new(0.0, 0.0);
    let mut base = lo;
    while base < hi {
        match p {
            1 => {
                // X: summand_i = conj(phi[i]) · src[i ^ m].
                for (a, b) in phi[base..base + m].iter().zip(&src[base + m..base + block]) {
                    ip += a.conj() * b;
                }
                for (a, b) in phi[base + m..base + block].iter().zip(&src[base..base + m]) {
                    ip += a.conj() * b;
                }
            }
            3 => {
                // Z: upper run negated (subtracting is exact and equals
                // adding -src).
                for (a, b) in phi[base..base + m].iter().zip(&src[base..base + m]) {
                    ip += a.conj() * b;
                }
                for (a, b) in phi[base + m..base + block]
                    .iter()
                    .zip(&src[base + m..base + block])
                {
                    ip -= a.conj() * b;
                }
            }
            2 => {
                // Y: v = -i·src[i|m] / +i·src[i&!m] — an exact swap+sign.
                for (a, b) in phi[base..base + m].iter().zip(&src[base + m..base + block]) {
                    ip += a.conj() * Complex64::new(b.im, -b.re);
                }
                for (a, b) in phi[base + m..base + block].iter().zip(&src[base..base + m]) {
                    ip += a.conj() * Complex64::new(-b.im, b.re);
                }
            }
            _ => {}
        }
        base += block;
    }
    ip
}

/// ⟨φ|G|src⟩ for a generator pauli string of length 1-2, applied inline.
///
/// Mirrors `apply_pauli_1q_inplace` operation-for-operation: X permutes by
/// the qubit mask, Z flips the sign of amplitudes with the qubit set, Y
/// rotates by ±i (sign read from the input index's bit).  Every phase
/// factor is an exact ±1 / ±i multiply, so each summand — and, with the
/// ascending-i accumulation, the sum itself — is bit-identical to the
/// scratch path (copy → apply paulis → dot).  Returns None for strings of
/// length > 2 or degenerate qubit lists (caller keeps the scratch path).
#[inline]
fn gen_dot(
    phi: &[Complex64],
    src: &[Complex64],
    gen_paulis: &[u8],
    qubits: &[usize],
    n_qubits: usize,
    par: bool,
) -> Option<Complex64> {
    let dim = 1usize << n_qubits;
    match gen_paulis.len() {
        1 => {
            // Single-qubit generator (Rx/Ry/Rz): run-structured loops keep
            // both operands contiguous and branch-free, unlike a per-element
            // popcount + match scan.
            let p = gen_paulis[0];
            if p == 0 || p > 3 {
                return None;
            }
            let m = 1usize << qubits[0];
            let block = m << 1;
            if par {
                let threads = rayon::current_num_threads().max(1);
                let want = (dim / (4 * threads)).max(block);
                let chunk = (want / block).max(1) * block;
                if chunk < dim {
                    // Chunked partials, combined sequentially: summands are
                    // identical to the serial path, the grouping is not
                    // (~1 ulp).
                    let n_chunks = dim.div_ceil(chunk);
                    let partials: Vec<Complex64> = (0..n_chunks)
                        .into_par_iter()
                        .map(|ci| {
                            let lo = ci * chunk;
                            dot_one_pauli(p, phi, src, m, lo, (lo + chunk).min(dim))
                        })
                        .collect();
                    let mut ip = Complex64::new(0.0, 0.0);
                    for q in &partials {
                        ip += *q;
                    }
                    return Some(ip);
                }
            }
            Some(dot_one_pauli(p, phi, src, m, 0, dim))
        }
        2 => {
            // Two-qubit generator (Rzz/Rxx/Ryy): phase factors are exact
            // ±1/±i and the index permutation is a xor mask.  Rare in hot
            // circuits — the per-element scan is fine here.
            if qubits[0] == qubits[1] {
                return None;
            }
            let (m0, m1) = (1usize << qubits[0], 1usize << qubits[1]);
            let (mut xmask, mut zmask, mut y0, mut y1) = (0usize, 0usize, 0usize, 0usize);
            match gen_paulis[0] {
                1 => xmask |= m0,
                2 => {
                    xmask |= m0;
                    y0 = m0;
                }
                3 => zmask |= m0,
                _ => return None,
            }
            match gen_paulis[1] {
                1 => xmask |= m1,
                2 => {
                    xmask |= m1;
                    y1 = m1;
                }
                3 => zmask |= m1,
                _ => return None,
            }
            let mut ip = Complex64::new(0.0, 0.0);
            for i in 0..dim {
                let mut v = src[i ^ xmask];
                if zmask != 0 && ((i & zmask).count_ones() & 1) == 1 {
                    v = -v;
                }
                if y0 != 0 {
                    v = if (i & y0) != 0 {
                        Complex64::new(-v.im, v.re)
                    } else {
                        Complex64::new(v.im, -v.re)
                    };
                }
                if y1 != 0 {
                    v = if (i & y1) != 0 {
                        Complex64::new(-v.im, v.re)
                    } else {
                        Complex64::new(v.im, -v.re)
                    };
                }
                ip += phi[i].conj() * v;
            }
            Some(ip)
        }
        _ => None,
    }
}

// ─── Gate operation wrapper ─────────────────────────────────────────────

struct GateOp {
    op_type: OpType,
    qubits: Vec<usize>,
}

impl GateOp {
    fn unitary_2x2(&self) -> [[Complex64; 2]; 2] {
        let m = self.op_type.to_matrix();
        [[m[(0, 0)], m[(0, 1)]], [m[(1, 0)], m[(1, 1)]]]
    }

    fn unitary_4x4(&self) -> [[Complex64; 4]; 4] {
        let m = self.op_type.to_matrix();
        [
            [m[(0, 0)], m[(0, 1)], m[(0, 2)], m[(0, 3)]],
            [m[(1, 0)], m[(1, 1)], m[(1, 2)], m[(1, 3)]],
            [m[(2, 0)], m[(2, 1)], m[(2, 2)], m[(2, 3)]],
            [m[(3, 0)], m[(3, 1)], m[(3, 2)], m[(3, 3)]],
        ]
    }

    fn dagger_2x2(&self) -> [[Complex64; 2]; 2] {
        let u = self.unitary_2x2();
        [
            [u[0][0].conj(), u[1][0].conj()],
            [u[0][1].conj(), u[1][1].conj()],
        ]
    }

    fn dagger_4x4(&self) -> [[Complex64; 4]; 4] {
        let u = self.unitary_4x4();
        [
            [
                u[0][0].conj(),
                u[1][0].conj(),
                u[2][0].conj(),
                u[3][0].conj(),
            ],
            [
                u[0][1].conj(),
                u[1][1].conj(),
                u[2][1].conj(),
                u[3][1].conj(),
            ],
            [
                u[0][2].conj(),
                u[1][2].conj(),
                u[2][2].conj(),
                u[3][2].conj(),
            ],
            [
                u[0][3].conj(),
                u[1][3].conj(),
                u[2][3].conj(),
                u[3][3].conj(),
            ],
        ]
    }
}

/// Compute the adjoint gradient for a parameterised circuit.
///
/// Two variants, selected by memory: when M·2^n·16 B ≤ 512 MB the forward
/// pass caches the state AFTER every gate, so the backward pass walks phi
/// only, applies each generator to one reusable scratch buffer, and never
/// clones or re-applies states; above the cap it falls back to the original
/// recompute-from-|0> variant (O(2^n) memory, 2M gate applications).
///
/// Errors with `MethodError` when `param_values` omits any parameter that
/// appears in the circuit (including variables nested in expressions).
pub fn adjoint_grad(
    dag: &QuantumDAG,
    observable_terms: &[PauliTerm],
    param_values: &std::collections::HashMap<String, f64>,
) -> Result<AdjointGradResult, MethodError> {
    let n_qubits = dag.n_qubits;
    let dim = 1usize << n_qubits;
    // n >= 16: sweeps may split across the rayon pool (skipped entirely
    // when the pool is pinned to one thread).
    let par = dim >= PAR_DIM && rayon::current_num_threads() > 1;

    // Phase timing (SF_ADJ_TRACE=1): attribute each call's cost to
    // setup / forward / observable / per-parameter kernel / back-walk.
    // Development instrumentation only — no behavior change.
    let adj_trace = std::env::var_os("SF_ADJ_TRACE").is_some();
    let t_pre = std::time::Instant::now();
    let mut t_pre_ms = 0.0f64;
    let mut t_fwd = 0.0f64;
    let mut t_phi = 0.0f64;
    let mut t_ip_prep = 0.0f64;
    let t_ip_pauli = 0.0f64;
    let mut t_ip_dot = 0.0f64;
    let mut t_bw_psi = 0.0f64;
    let mut t_bw_phi = 0.0f64;

    let bound_dag = dag.bind(param_values);

    let order = bound_dag.topological_order();
    let ops: Vec<GateOp> = order
        .iter()
        .filter_map(|&node_id| {
            let op = &bound_dag.graph()[node_id];
            if op.op_type.is_boundary()
                || op.op_type == OpType::Barrier
                || op.op_type.is_measurement()
            {
                return None;
            }
            Some(GateOp {
                op_type: op.op_type.clone(),
                qubits: op.qubits.to_vec(),
            })
        })
        .collect();

    // Any variable still symbolic after bind() has no value in
    // `param_values`; evaluating it would panic in ops.rs. Fail with a
    // catchable error instead. variable_names() also sees variables nested
    // in parameter expressions, which dag.parameter_names() does not.
    let mut missing: Vec<String> = ops
        .iter()
        .flat_map(|g| g.op_type.parameters())
        .flat_map(|p| p.variable_names())
        .collect();
    if !missing.is_empty() {
        missing.sort();
        missing.dedup();
        return Err(MethodError(format!(
            "no value provided for parameter(s): {}. Pass all parameter \
             values via param_values= (the dag itself stays unbound)",
            missing.join(", ")
        )));
    }

    let orig_order = dag.topological_order();
    let orig_ops: Vec<&QuantumOp> = orig_order
        .iter()
        .filter_map(|&node_id| {
            let op = &dag.graph()[node_id];
            if op.op_type.is_boundary()
                || op.op_type == OpType::Barrier
                || op.op_type.is_measurement()
            {
                return None;
            }
            Some(op)
        })
        .collect();

    let param_names: Vec<String> = dag.parameter_names();
    let param_idx: std::collections::HashMap<&str, usize> = param_names
        .iter()
        .enumerate()
        .map(|(i, n)| (n.as_str(), i))
        .collect();
    let n_params = param_names.len();

    // Pre-compute all gate matrices (avoid recomputing during backward pass)
    let fwd_matrices_1q: Vec<Option<[[Complex64; 2]; 2]>> = ops
        .iter()
        .map(|g| {
            if g.qubits.len() == 1 {
                Some(g.unitary_2x2())
            } else {
                None
            }
        })
        .collect();
    let fwd_matrices_2q: Vec<Option<[[Complex64; 4]; 4]>> = ops
        .iter()
        .map(|g| {
            if g.qubits.len() == 2 {
                Some(g.unitary_4x4())
            } else {
                None
            }
        })
        .collect();
    let dag_matrices_1q: Vec<Option<[[Complex64; 2]; 2]>> = ops
        .iter()
        .map(|g| {
            if g.qubits.len() == 1 {
                Some(g.dagger_2x2())
            } else {
                None
            }
        })
        .collect();
    let dag_matrices_2q: Vec<Option<[[Complex64; 4]; 4]>> = ops
        .iter()
        .map(|g| {
            if g.qubits.len() == 2 {
                Some(g.dagger_4x4())
            } else {
                None
            }
        })
        .collect();

    // Forward pass: evolve |0> → |psi_final>. When the cache cap allows
    // (M·dim·16 B ≤ 512 MB), store the state AFTER each gate so the
    // backward pass never clones or re-applies; above the cap, fall back
    // to the recompute-from-|0> variant.
    // Only parameterised gates need a cached post-state (ψ_after_k for the
    // gradient inner product) — the back-walk itself never reads the cache.
    // Caching just those states shrinks the buffer 25-50% on HE circuits
    // and keeps n=18 d=3 under the cap.
    let mut param_slot: Vec<Option<usize>> = Vec::with_capacity(ops.len());
    let mut n_slots = 0usize;
    for op in &orig_ops {
        let slot = get_param_name(&op.original_op_type)
            .filter(|nm| param_idx.contains_key(nm.as_str()))
            .map(|_| {
                let s = n_slots;
                n_slots += 1;
                s
            });
        param_slot.push(slot);
    }
    let cache_bytes = (n_slots as u64)
        .saturating_mul(dim as u64)
        .saturating_mul(16);
    // Single contiguous cache: a Vec<Vec> allocates + zero-fills a fresh
    // psi.clone() per gate (measured as the forward pass's dominant cost at
    // n=16+). Every element is written before it is read (Complex64: Copy,
    // no Drop), so the uninit set_len is sound; a retained buffer simply
    // gets its stale bytes overwritten. The buffer is borrowed from the
    // thread-local pool and put back at the end, so repeated calls at the
    // same size reuse resident pages instead of re-faulting ~memory-size
    // in fresh allocations.
    let use_cache = cache_bytes <= 512 * 1024 * 1024;
    let mut cache: Vec<Complex64> = ADJ_CACHE_BUF.with(|c| std::mem::take(&mut *c.borrow_mut()));
    if use_cache {
        let need = n_slots * dim;
        if cache.capacity() < need {
            cache = Vec::with_capacity(need);
        }
        unsafe { cache.set_len(need) };
    }

    let mut psi = vec![Complex64::new(0.0, 0.0); dim];
    psi[0] = Complex64::new(1.0, 0.0);

    if adj_trace {
        t_pre_ms = t_pre.elapsed().as_secs_f64() * 1e3;
    }
    let t0 = std::time::Instant::now();
    for (k, gate_op) in ops.iter().enumerate() {
        match gate_op.qubits.len() {
            1 => apply_1q_inplace(
                &mut psi,
                fwd_matrices_1q[k].unwrap(),
                gate_op.qubits[0],
                n_qubits,
                par,
            ),
            2 => apply_2q_inplace(
                &mut psi,
                fwd_matrices_2q[k].unwrap(),
                gate_op.qubits[0],
                gate_op.qubits[1],
                n_qubits,
            ),
            _ => {}
        }
        if use_cache {
            if let Some(slot) = param_slot[k] {
                copy_slice_par(&mut cache[slot * dim..(slot + 1) * dim], &psi, par);
            }
        }
    }
    if adj_trace {
        t_fwd = t0.elapsed().as_secs_f64() * 1e3;
    }

    // Build phi = O|psi_final>
    let t0 = std::time::Instant::now();
    let mut phi = vec![Complex64::new(0.0, 0.0); dim];
    let mut term_state = vec![Complex64::new(0.0, 0.0); dim];
    for term in observable_terms {
        // Fused {X,Z}-only terms with at most two non-identity factors
        // (every term of the TFIM / Pauli observables in the benchmarks):
        // accumulate coef·(phase(i)·psi[i ^ xmask]) straight into phi —
        // one pass, no temporary, bit-identical to copy + pauli +
        // accumulate (phases are exact ±1).
        let mut xmask = 0usize;
        let mut zmask = 0usize;
        let mut nz = 0usize;
        let mut fusable = true;
        for (q, &p) in term.paulis.iter().enumerate() {
            match p {
                0 => {}
                1 => {
                    xmask |= 1usize << q;
                    nz += 1;
                }
                3 => {
                    zmask |= 1usize << q;
                    nz += 1;
                }
                _ => fusable = false,
            }
        }
        if nz > 2 {
            fusable = false;
        }
        if fusable {
            let coef = term.coef;
            if par {
                let threads = rayon::current_num_threads().max(1);
                let chunk = ((dim / (4 * threads)).max(4096)).next_power_of_two();
                phi.par_chunks_mut(chunk).enumerate().for_each(|(ci, ph)| {
                    let base = ci * chunk;
                    for (k, out) in ph.iter_mut().enumerate() {
                        let i = base + k;
                        let neg = zmask != 0 && ((i & zmask).count_ones() & 1) == 1;
                        let c = if neg { -coef } else { coef };
                        *out += c * psi[i ^ xmask];
                    }
                });
            } else {
                for i in 0..dim {
                    let neg = zmask != 0 && ((i & zmask).count_ones() & 1) == 1;
                    let c = if neg { -coef } else { coef };
                    phi[i] += c * psi[i ^ xmask];
                }
            }
        } else {
            copy_slice_par(&mut term_state, &psi, par);
            for (q, &p) in term.paulis.iter().enumerate() {
                if p != 0 {
                    apply_pauli_1q_inplace(&mut term_state, p, q, n_qubits);
                }
            }
            for i in 0..dim {
                phi[i] += term.coef * term_state[i];
            }
        }
    }
    if adj_trace {
        t_phi = t0.elapsed().as_secs_f64() * 1e3;
    }

    // Backward pass
    // phi holds O|psi_final> and walks backward through the gates.
    // Cached variant: psi_after_k comes from the forward cache; only phi
    // is un-applied per gate. Recompute variant: psi walks backward too
    // (the original behavior) and U_k is re-applied to a copy.
    let mut grad = vec![0.0f64; n_params];
    // Scratch for the >2-pauli generator fallback (lazily allocated; the
    // fused path never touches it), plus the recompute-variant work state
    // g_psi (fully overwritten by copy_from_slice before any read, hence
    // the uninit set_len — no per-parameter allocation).
    let mut scratch: Vec<Complex64> = Vec::new();
    // Complex64 is Copy (no Drop) and `copy_slice_par` overwrites every
    // element before any read, so the uninit set_len is sound.
    #[allow(clippy::uninit_vec)]
    let mut g_psi: Vec<Complex64> = if use_cache {
        Vec::new()
    } else {
        let mut v = Vec::with_capacity(dim);
        unsafe { v.set_len(dim) };
        v
    };

    for k in (0..ops.len()).rev() {
        let gate_op = &ops[k];
        let orig_op = &orig_ops[k];

        if !use_cache {
            let t0 = std::time::Instant::now();
            // Un-apply gate k from psi to get psi_k (state before gate k)
            match gate_op.qubits.len() {
                1 => apply_1q_inplace(
                    &mut psi,
                    dag_matrices_1q[k].unwrap(),
                    gate_op.qubits[0],
                    n_qubits,
                    par,
                ),
                2 => apply_2q_inplace(
                    &mut psi,
                    dag_matrices_2q[k].unwrap(),
                    gate_op.qubits[0],
                    gate_op.qubits[1],
                    n_qubits,
                ),
                _ => {}
            }
            if adj_trace {
                t_bw_psi += t0.elapsed().as_secs_f64() * 1e3;
            }
        }

        // Gradient contribution: <phi | G_k | psi_after_k>.
        let gen_info = generator_info(&orig_op.original_op_type);
        let param_name = get_param_name(&orig_op.original_op_type);

        if let (Some(gen_info), Some(param_name)) = (gen_info, param_name) {
            if let Some(&idx) = param_idx.get(param_name.as_str()) {
                let (gen_paulis, alpha) = gen_info;

                // grad = 2α·Im⟨φ|G_k|ψ_after_k⟩. `gen_dot` applies the
                // generator inline (exact ±1/±i phases + index permutation)
                // without materializing G|ψ⟩ — bit-identical to the old
                // scratch path, minus the per-parameter copy + pauli pass.
                let t0 = std::time::Instant::now();
                let src: &[Complex64] = if use_cache {
                    // Invariant: the enclosing guards (get_param_name hit +
                    // param_idx hit) are exactly the predicate that produced
                    // param_slot[k] = Some, so the expect cannot fire.
                    let slot = param_slot[k].expect("cached gradient gate has no cache slot");
                    &cache[slot * dim..(slot + 1) * dim]
                } else {
                    // Recomputed ψ_k: re-apply U_k into a preallocated
                    // buffer (no per-parameter allocation).
                    copy_slice_par(&mut g_psi, &psi, par);
                    match gate_op.qubits.len() {
                        1 => apply_1q_inplace(
                            &mut g_psi,
                            fwd_matrices_1q[k].unwrap(),
                            gate_op.qubits[0],
                            n_qubits,
                            par,
                        ),
                        2 => apply_2q_inplace(
                            &mut g_psi,
                            fwd_matrices_2q[k].unwrap(),
                            gate_op.qubits[0],
                            gate_op.qubits[1],
                            n_qubits,
                        ),
                        _ => {}
                    }
                    &g_psi[..]
                };
                if adj_trace {
                    t_ip_prep += t0.elapsed().as_secs_f64() * 1e3;
                }
                let t0 = std::time::Instant::now();
                let ip: Complex64 = match gen_dot(
                    phi.as_slice(),
                    src,
                    gen_paulis,
                    &gate_op.qubits,
                    n_qubits,
                    par,
                ) {
                    Some(ip) => ip,
                    None => {
                        // >2-pauli generator (not emitted by
                        // generator_info today): historical scratch path.
                        if scratch.is_empty() {
                            scratch = vec![Complex64::new(0.0, 0.0); dim];
                        }
                        scratch.copy_from_slice(src);
                        for (q_local, &pauli_id) in gen_paulis.iter().enumerate() {
                            let qubit = gate_op.qubits[q_local];
                            apply_pauli_1q_inplace(&mut scratch, pauli_id, qubit, n_qubits);
                        }
                        phi.iter()
                            .zip(scratch.iter())
                            .map(|(a, b)| a.conj() * b)
                            .sum()
                    }
                };
                if adj_trace {
                    t_ip_dot += t0.elapsed().as_secs_f64() * 1e3;
                }
                grad[idx] += 2.0 * alpha * ip.im;
            }
        }

        // Back-walk phi
        let t0 = std::time::Instant::now();
        match gate_op.qubits.len() {
            1 => apply_1q_inplace(
                &mut phi,
                dag_matrices_1q[k].unwrap(),
                gate_op.qubits[0],
                n_qubits,
                par,
            ),
            2 => apply_2q_inplace(
                &mut phi,
                dag_matrices_2q[k].unwrap(),
                gate_op.qubits[0],
                gate_op.qubits[1],
                n_qubits,
            ),
            _ => {}
        }
        if adj_trace {
            t_bw_phi += t0.elapsed().as_secs_f64() * 1e3;
        }
    }

    if adj_trace {
        eprintln!(
            "SF_ADJ_TRACE n={} dim={} gates={} params={} variant={} setup={:.1}ms \
             fwd={:.1}ms phi={:.1}ms ip[prep={:.1} pauli={:.1} dot={:.1}]ms \
             bw_psi={:.1}ms bw_phi={:.1}ms",
            n_qubits,
            dim,
            ops.len(),
            n_params,
            if use_cache { "cache" } else { "recompute" },
            t_pre_ms,
            t_fwd,
            t_phi,
            t_ip_prep,
            t_ip_pauli,
            t_ip_dot,
            t_bw_psi,
            t_bw_phi,
        );
    }

    ADJ_CACHE_BUF.with(|c| *c.borrow_mut() = cache);

    Ok(AdjointGradResult {
        param_names,
        gradients: grad,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dag::QuantumDAG;
    use crate::ops::{OpType, Parameter};

    #[test]
    fn test_rx_gradient() {
        let theta = 0.7;
        let mut dag = QuantumDAG::new(1, 0);
        dag.add_op(
            OpType::Rx(Parameter::Variable {
                name: "theta".into(),
                id: 0,
            }),
            &[0],
        );
        let obs = vec![PauliTerm {
            paulis: vec![3],
            coef: Complex64::new(1.0, 0.0),
        }];
        let mut params = std::collections::HashMap::new();
        params.insert("theta".into(), theta);
        let result = adjoint_grad(&dag, &obs, &params).unwrap();
        let expected = -theta.sin();
        assert!(
            (result.gradients[0] - expected).abs() < 1e-10,
            "Got {}, expected {}",
            result.gradients[0],
            expected
        );
    }

    #[test]
    fn test_ry_gradient() {
        let theta = 1.2;
        let mut dag = QuantumDAG::new(1, 0);
        dag.add_op(
            OpType::Ry(Parameter::Variable {
                name: "theta".into(),
                id: 0,
            }),
            &[0],
        );
        let obs = vec![PauliTerm {
            paulis: vec![3],
            coef: Complex64::new(1.0, 0.0),
        }];
        let mut params = std::collections::HashMap::new();
        params.insert("theta".into(), theta);
        let result = adjoint_grad(&dag, &obs, &params).unwrap();
        let expected = -theta.sin();
        assert!((result.gradients[0] - expected).abs() < 1e-10);
    }

    #[test]
    fn test_rz_gradient_x_observable() {
        let theta = 0.6;
        let mut dag = QuantumDAG::new(1, 0);
        dag.add_op(OpType::H, &[0]);
        dag.add_op(
            OpType::Rz(Parameter::Variable {
                name: "theta".into(),
                id: 0,
            }),
            &[0],
        );
        let obs = vec![PauliTerm {
            paulis: vec![1],
            coef: Complex64::new(1.0, 0.0),
        }];
        let mut params = std::collections::HashMap::new();
        params.insert("theta".into(), theta);
        let result = adjoint_grad(&dag, &obs, &params).unwrap();
        let expected = -theta.sin();
        assert!((result.gradients[0] - expected).abs() < 1e-10);
    }

    #[test]
    fn test_multi_param_gradient() {
        let t0 = 0.5;
        let t1 = 1.0;
        let mut dag = QuantumDAG::new(1, 0);
        dag.add_op(
            OpType::Rx(Parameter::Variable {
                name: "t0".into(),
                id: 0,
            }),
            &[0],
        );
        dag.add_op(
            OpType::Ry(Parameter::Variable {
                name: "t1".into(),
                id: 1,
            }),
            &[0],
        );
        let obs = vec![PauliTerm {
            paulis: vec![3],
            coef: Complex64::new(1.0, 0.0),
        }];
        let mut params = std::collections::HashMap::new();
        params.insert("t0".into(), t0);
        params.insert("t1".into(), t1);
        let result = adjoint_grad(&dag, &obs, &params).unwrap();
        assert_eq!(result.param_names.len(), 2);
        assert!(result.gradients[0].is_finite());
        assert!(result.gradients[1].is_finite());
    }

    #[test]
    fn test_two_qubit_rzz_gradient() {
        let theta = 0.4;
        let mut dag = QuantumDAG::new(2, 0);
        dag.add_op(OpType::H, &[0]);
        dag.add_op(OpType::H, &[1]);
        dag.add_op(
            OpType::Rzz(Parameter::Variable {
                name: "theta".into(),
                id: 0,
            }),
            &[0, 1],
        );
        let obs = vec![PauliTerm {
            paulis: vec![1, 0],
            coef: Complex64::new(1.0, 0.0),
        }];
        let mut params = std::collections::HashMap::new();
        params.insert("theta".into(), theta);
        let result = adjoint_grad(&dag, &obs, &params).unwrap();
        let expected = -(theta).sin();
        assert!((result.gradients[0] - expected).abs() < 1e-8);
    }

    #[test]
    fn test_multi_observable_terms() {
        let theta = 0.6;
        let mut dag = QuantumDAG::new(1, 0);
        dag.add_op(
            OpType::Rx(Parameter::Variable {
                name: "theta".into(),
                id: 0,
            }),
            &[0],
        );
        let obs = vec![
            PauliTerm {
                paulis: vec![3],
                coef: Complex64::new(0.5, 0.0),
            },
            PauliTerm {
                paulis: vec![1],
                coef: Complex64::new(0.3, 0.0),
            },
        ];
        let mut params = std::collections::HashMap::new();
        params.insert("theta".into(), theta);
        let result = adjoint_grad(&dag, &obs, &params).unwrap();
        let expected = -0.5 * theta.sin();
        assert!((result.gradients[0] - expected).abs() < 1e-10);
    }

    #[test]
    fn test_same_param_multiple_gates() {
        let theta = 0.7;
        let mut dag = QuantumDAG::new(2, 0);
        dag.add_op(
            OpType::Rx(Parameter::Variable {
                name: "theta".into(),
                id: 0,
            }),
            &[0],
        );
        dag.add_op(
            OpType::Rx(Parameter::Variable {
                name: "theta".into(),
                id: 0,
            }),
            &[1],
        );
        let obs = vec![PauliTerm {
            paulis: vec![3, 0],
            coef: Complex64::new(1.0, 0.0),
        }];
        let mut params = std::collections::HashMap::new();
        params.insert("theta".into(), theta);
        let result = adjoint_grad(&dag, &obs, &params).unwrap();
        let expected = -theta.sin();
        assert!((result.gradients[0] - expected).abs() < 1e-10);
    }
}
