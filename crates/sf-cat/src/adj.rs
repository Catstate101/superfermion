//! sf-cat adjoint engine (Lever 3): a compile-once reverse-mode gradient
//! handle for parameterised small-qubit circuits.
//!
//! Numerical semantics are a faithful port of `sf_ir::adjoint::adjoint_grad`
//! (same gate-matrix formulas, same pair-pass application order, same
//! generator dots, same gradient accumulation), so gradients agree with the
//! house engine to floating-point tolerance — pinned by a parity test against
//! `sf_ir::adjoint::adjoint_grad` below.  What the handle removes is sf-ir's
//! per-call fixed cost: the topological walk, `OpType` matching and
//! `bind_params`, the `HashMap` parameter lookup, the per-gate `to_matrix()`
//! heap allocations, and the per-call buffer take/put churn.  All of that is
//! resolved once in [`AdjProgram`] construction; a per-call [`grad`] is one
//! matrix-fill pass plus one forward and one backward sweep over flat arrays
//! with resident pages.
//!
//! Scope (explicit): RY/RZ gates with variable parameters and CNOT gates —
//! the benchmark ansatz — with arbitrary [`PauliTerm`] observables.  Any
//! other gate is rejected at compile time instead of being silently applied.
//!
//! [`grad`]: AdjProgram::grad

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use num_complex::Complex64;
use rayon::prelude::*;
use sf_ir::adjoint::PauliTerm;
use sf_ir::dag::QuantumDAG;
use sf_ir::ops::{OpType, Parameter};
use sf_ir::simd::{pair_pass, pair_pass_chunk};

/// Upper bound on the parameter-state cache (`n_param_gates * 2^n` complex).
/// The handle targets small-n workloads; above 64 Mi complex (1 GiB) the
/// caller should stay on `sf_ir::adjoint` (which streams over the cap).
const CACHE_ELEM_CAP: usize = 1 << 26;

/// One flat gate of the compiled program.  `slot` indexes the parameter
/// vector (the DAG's `parameter_names()` order for [`AdjProgram::from_dag`]).
type M2 = [[Complex64; 2]; 2];

#[derive(Clone, Copy, Debug)]
enum AdjGate {
    Ry {
        q: u32,
        slot: u32,
    },
    Rz {
        q: u32,
        slot: u32,
    },
    Rx {
        q: u32,
        slot: u32,
    },
    /// Hadamard / Pauli-X: constant unitaries, applied on both sweeps but
    /// carrying no gradient slot of their own.
    H {
        q: u32,
    },
    X {
        q: u32,
    },
    /// Any other constant single-qubit unitary (`S`, `T`, `P` with a bound
    /// angle, a fixed rotation, ...).  The matrix is baked at plan build time
    /// and applied on both sweeps, so it carries no gradient slot.
    Fixed {
        q: u32,
        m: M2,
    },
    /// `Cu(θ,φ,λ,γ) = diag(I, e^{iγ}·U3(θ,φ,λ))` — a 2×2 block on `tgt`
    /// applied when `ctl` is 1.  Each of the four angles is either a parameter
    /// slot (`>= 0`) or a baked constant, so one gate can differentiate up to
    /// four parameters.  `gscale` folds one of the *other* angles into γ
    /// (`γ ← γ + gscale·a[gref]`), which is how controlled-RZ is expressed
    /// exactly: `e^{-iθ/2}·U3(0,0,θ) = diag(e^{-iθ/2}, e^{iθ/2})`, so
    /// `gscale = -0.5`, `gref = 2` (the λ slot), `λ = θ`, `θ = 0`.
    Cu {
        ctl: u32,
        tgt: u32,
        slots: [i32; 4],
        consts: [f64; 4],
        gscale: f64,
        gref: u8,
    },
    Cx {
        ctl: u32,
        tgt: u32,
    },
    /// `exp(−iθ/2·Z⊗Z)`: the whole product is
    /// `diag(e^{−iθ/2}, e^{+iθ/2}, e^{+iθ/2}, e^{−iθ/2})` on `(a, b)`, so it is
    /// applied as one diagonal 4-phase pattern rather than the
    /// `Cx·Rz·Cx` triple the IR expands it to.  One traversal instead of
    /// three, in the forward pass and in both reverse sweeps.
    Rzz {
        a: u32,
        b: u32,
        slot: u32,
    },
    /// `exp(−iθ/2·P⊗P)` for `P ∈ {X, Y}`, applied as two independent 2×2
    /// rotations on the amplitude pairs `(i, i^mask)` instead of the seven-gate
    /// `(H⊗H)·Cx·Rz·Cx·(H⊗H)` chain the IR expands these to — one traversal per
    /// parameter instead of seven, in the forward pass and in both reverse
    /// sweeps.  `Y⊗Y = −(Z⊗Z)(X⊗X)`, so `yy` only flips the sign of the
    /// off-diagonal coefficient on the pairs whose two bits agree.
    Rxx {
        a: u32,
        b: u32,
        slot: u32,
        yy: bool,
    },
    /// See [`Mono`].  `qs` holds the role qubits in the gate's own order —
    /// `[a, b]` for `Swap`/`ISwap`, `[c1, c2, t]` for `Ccx`, `[c, a, b]` for
    /// `Cswap`, first listed most significant — and `k` says how many of the
    /// three are used.
    Mono {
        kind: Mono,
        qs: [u32; 3],
        k: u8,
    },
}

/// A constant two- or three-qubit gate whose matrix is monomial in the
/// computational basis: every output amplitude is one input amplitude times
/// `±1` or `±i`.  `SWAP`, `iSWAP`, `CCX` (Toffoli) and `CSWAP` (Fredkin) all
/// qualify, so each is applied as one index map instead of the chain the IR
/// expands it to — three CNOTs for `SWAP`, and for `CCX` a fifteen-gate
/// Clifford+T decomposition that the adjoint plan simply refused before this.
/// All four are their own inverse in the index map, so a daggered replay only
/// conjugates the `i` factors; being constant they own no gradient slot and
/// are just replayed on the reverse sweep.
#[derive(Clone, Copy, Debug)]
enum Mono {
    Swap,
    ISwap,
    Ccx,
    Cswap,
}

impl AdjGate {
    #[inline]
    fn is_param(&self) -> bool {
        match self {
            AdjGate::Ry { .. } | AdjGate::Rz { .. } | AdjGate::Rx { .. } => true,
            AdjGate::Rzz { .. } => true,
            AdjGate::Rxx { .. } => true,
            AdjGate::Cu { slots, .. } => slots.iter().any(|&s| s >= 0),
            _ => false,
        }
    }
}

/// Most distinct qubits one fused run may touch.  A run's block buffer holds
/// `2^MAX_RUN_Q` amplitudes, so this is also the width of the scratch both the
/// gather and the kernels work on.
const MAX_RUN_Q: usize = 4;

/// Most gates taken into one fused run: how many `(dot, apply)` steps a block
/// buffer is held live for.  Bounded so the buffer stays in L1 across gathers
/// and so a run's dots fit a fixed-size stack array.
const FUSE_RUN_MAX: usize = 4;

/// Block buffer width: one entry per amplitude the run's qubits span.
const FUSE_BLK_MAX: usize = 1 << MAX_RUN_Q;

/// Widest strip the streaming sweep will fuse over, in quantum bits.
///
/// A strip is a *contiguous* slice of both vectors that stays resident while
/// every gate of the run is applied to it, so the run costs one pass over
/// memory instead of one pass per gate.  That only holds while the strip is
/// cache-resident: this host has 8 MiB of L3 shared by four threads, and a
/// strip of `2^15` complex is 512 KiB per vector, i.e. 1 MiB per thread and
/// 4 MiB in flight -- half the L3, which leaves room for the associativity
/// conflicts.  `2^16` would put the whole L3 in flight and was measured
/// slower.  `SF_CAT_STRIP_BITS` overrides it.
const STRIP_BITS_DEFAULT: u32 = 15;

/// Most gates taken into one strip run.  Not a correctness bound: it keeps a
/// run's dots inside a fixed-size stack array and stops one run from holding a
/// thread for so long that the other strips idle.
const STRIP_RUN_MAX: usize = 64;

/// Which fused kernel a gate needs — the branch
/// [`AdjProgram::step_backward`] takes per gate, resolved once per run.
#[derive(Clone, Copy, PartialEq, Eq)]
enum FuseKind {
    /// `Ry`/`Rz`/`Rx`; the payload is [`dot_apply_pair`]'s Pauli index.
    Rot(u8),
    /// `Rzz`: a diagonal 4-phase pattern, so no pair partner.
    Zz,
    /// `Rxx`/`Ryy`: `[[c, k], [k, c]]` on the amplitude pairs.
    Xx,
    /// `H`/`X`/`Fixed`: constant, so no generator dot to accumulate.
    Const,
    /// `Cx`: a pure permutation, so no generator dot either.  `ma`/`mb` hold
    /// the control and target *masks*; the kernel wants bit indices, which
    /// [`FuseOp::step`] recovers with `trailing_zeros`.
    Cx,
}

/// One gate of a fused run, with everything its kernel needs pre-resolved to
/// *block-local* masks.
///
/// `ma`/`mb` are the masks of the lower and higher qubit of the gate inside the
/// gathered buffer (for a single-qubit gate `mb == ma`).  The kernels index by
/// a mask relative to the slice they are handed, so they run unchanged on a
/// `2^nq` block instead of on the whole state.
#[derive(Clone, Copy)]
struct FuseOp {
    kind: FuseKind,
    ma: usize,
    mb: usize,
    /// Daggered forward matrix, for `Rot` and `Const`.
    gate: M2,
    /// `zz_phase(θ, true)`, for `Zz`.
    ph: [Complex64; 2],
    /// `xx_coef(θ, yy, true)`, for `Xx`.
    c: f64,
    kk: [Complex64; 2],
    yy: bool,
    slot: u32,
}

impl FuseOp {
    /// Inert entry, so a run's op array can be a fixed-size stack array.
    #[inline]
    fn empty() -> Self {
        Self {
            kind: FuseKind::Const,
            ma: 0,
            mb: 0,
            gate: [[Complex64::new(0.0, 0.0); 2]; 2],
            ph: [Complex64::new(0.0, 0.0); 2],
            c: 0.0,
            kk: [Complex64::new(0.0, 0.0); 2],
            yy: false,
            slot: 0,
        }
    }

    /// Generator dot plus the daggered gate on one block: the same work as one
    /// per-gate arm of [`AdjProgram::step_backward`], on `2^nq` amplitudes of
    /// the state instead of all of it.
    #[inline]
    fn step(&self, pb: &mut [Complex64], sb: &mut [Complex64]) -> Complex64 {
        match self.kind {
            FuseKind::Rot(p) => dot_apply_pair(p, pb, sb, self.ma, self.gate),
            FuseKind::Zz => dot_zz_apply_pair(pb, sb, self.mb, self.ma, self.ph),
            FuseKind::Xx => dot_xx_apply_pair(pb, sb, self.mb, self.ma, self.c, self.kk, self.yy),
            FuseKind::Const => {
                // The two calls the per-gate arm makes, minus the dot.
                pair_pass_chunk(pb, self.ma, self.gate);
                pair_pass_chunk(sb, self.ma, self.gate);
                Complex64::new(0.0, 0.0)
            }
            FuseKind::Cx => {
                // `cx_apply_pair` on both vectors in one walk, as the per-gate
                // arm does; the masks are strip-relative so the slice length
                // is the only geometry it needs.
                let d = pb.len();
                cx_apply_pair(
                    pb,
                    sb,
                    d,
                    self.ma.trailing_zeros() as usize,
                    self.mb.trailing_zeros() as usize,
                );
                Complex64::new(0.0, 0.0)
            }
        }
    }
}

/// `SF_ADJ_FUSE=1` turns the layer-fused reverse sweep on.
///
/// Off by default on purpose: measured against the per-gate sweep it is
/// 0.97x-1.09x (see §40), because gathering a block in and out of the stack
/// buffers costs about as much array traffic as it saves unless a run is long.
/// It is kept as a handle rather than as the default path.
fn fuse_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| match std::env::var("SF_ADJ_FUSE") {
        Ok(v) => !matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "0" | "off" | "no" | "false" | "none"
        ),
        Err(_) => false,
    })
}

/// `SF_CAT_ADJ_STRIP=1` turns the strip-fused streaming reverse sweep on.
///
/// **Off by default because it measures neutral, not because it is wrong.**  The
/// sweep is the same kernels on the same amplitudes in the same tape order,
/// walked strip-by-strip instead of state-by-state, so it is bit-for-bit
/// equivalent up to the dots' summation order (measured `max|dg|` = 1.2e-16 on
/// n=22, L=4).  What it was meant to buy is fewer passes: a maximal run of
/// consecutive gates whose whole support fits one strip costs one read and one
/// write of both vectors instead of one per gate, and at n=22 that is 25 passes
/// per layer instead of 65.
///
/// Interleaved against the per-gate sweep on the same build, five pairs, one
/// gradient call each (`_cat_strip_ab.py`), it is **1.01x median** with the
/// per-pair ratios inside 0.95-1.05 -- and the reason is not the budget.
/// Sweeping `SF_CAT_STRIP_BITS` from 11 (32 KiB per vector, L1/L2 resident) to
/// 18 gives 4353 / 4430 / 4142 / 4396 / 4013 / 4785 / 6131 ms of reverse-sweep
/// time: flat up to 15 and then a cliff at 18 where the strip stops fitting the
/// 8 MiB L3 at all.  So the cliff is cache, but below it the fusion saves
/// nothing, which means a fused run of `m` gates still costs about `m` strip
/// passes rather than one -- the strip survives the first kernel and not the
/// fifteenth.  The per-gate path runs at a consistent 12.2 GB/s of effective
/// traffic in both the forward and the reverse sweep, and the strip does not
/// move that number.
///
/// Kept rather than removed: it is the instrument that produced that answer, it
/// is covered by `fused_sweep_matches_per_gate_sweep`, and the run detection is
/// what a cache-resident (rather than traffic-bound) kernel would need.
fn strip_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| match std::env::var("SF_CAT_ADJ_STRIP") {
        Ok(v) => !matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "0" | "off" | "no" | "false" | "none"
        ),
        Err(_) => false,
    })
}

/// Cache budget for one strip, in elements, and the widest strip a run may use.
///
/// Two constraints, whichever binds first:
///
/// * `len / threads` -- a strip wider than that leaves threads idle, and a run
///   of one gate in a strip the size of the state is the serial sweep again;
/// * [`STRIP_BITS_DEFAULT`] -- beyond the L3 budget the run stops being
///   cache-resident and each gate of the run streams the strip from DRAM,
///   which is the per-gate sweep with extra bookkeeping.
fn strip_limit(len: usize, threads: usize) -> usize {
    static BITS: OnceLock<u32> = OnceLock::new();
    let bits = *BITS.get_or_init(|| {
        std::env::var("SF_CAT_STRIP_BITS")
            .ok()
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(STRIP_BITS_DEFAULT)
    });
    let budget = 1usize.checked_shl(bits.min(20)).unwrap_or(1 << 20);
    (len / threads.max(1)).max(2).min(budget).max(2)
}

/// Most diagonal gates taken into one run.
///
/// Unlike [`STRIP_RUN_MAX`] this is not a cache budget: a diagonal run touches
/// no amplitude twice, so a run of any length is still one traversal.  The cap
/// only bounds the per-thread array of per-gate dots.
const DIAG_RUN_MAX: usize = 64;

/// One diagonal gate of a run, reduced to the numbers its kernel needs.
///
/// `mask` is the OR of the gate's qubit masks, so the generator eigenvalue of
/// amplitude `i` is `(-1)^popcount(i & mask)` -- one `popcount` serves `Rz`
/// (one bit) and `Rzz` (two bits XORed together) alike.  `t2` is the gate angle
/// indexed by that parity, i.e. `[+θ, -θ]`, so the kernel adds a number instead
/// of multiplying by a sign.
#[derive(Clone, Copy)]
struct DiagOp {
    mask: usize,
    t2: [f64; 2],
    slot: u32,
}

/// Generator parity of amplitude `i` under `op`: `0` on the `+1` eigenspace.
///
/// `Rz` is `+1` on bit 0 and `Rzz` is `+1` when the two bits agree, which are
/// the same statement once the bits are XORed: `popcount(i & mask) & 1`.
///
/// `i` must be the amplitude's index in the *whole* state, not in the chunk
/// being walked -- see [`diag_backward_blocks`].
#[inline]
fn diag_parity(op: &DiagOp, i: usize) -> usize {
    ((i & op.mask).count_ones() & 1) as usize
}

/// A run's total phase `exp(-i·a/2)`, or its conjugate when `dagger`.
///
/// The per-gate sweep multiplies `m` unit complex numbers; this takes one
/// `sin`/`cos` of their summed argument, which is cheaper and slightly better
/// conditioned -- there is no product of `m` roundings to accumulate.
#[inline]
fn diag_phase(a: f64, dagger: bool) -> Complex64 {
    let h = if dagger { 0.5 * a } else { -0.5 * a };
    Complex64::new(h.cos(), h.sin())
}

/// Reverse a diagonal run on one slice: all `m` generator dots *and* both
/// rewinds in a single pass over `phi`/`psi`.
///
/// The reason this is one pass where the per-gate sweep needs `m` is algebraic,
/// not a cache trick.  With `d_j(i)` the phase gate `j` puts on amplitude `i`
/// and `T(i) = prod_j d_j(i)`, the dot the per-gate sweep takes at gate `j` is
///
/// ```text
///   conj(phi_j(i))·s_j(i)·psi_j(i) = conj(phi_out(i))·psi_in(i)·T(i)·s_j(i)
/// ```
///
/// because `phi_j` carries `prod_{l>j} conj(d_l)` and `psi_j` carries
/// `prod_{l<=j} d_l`, and between them every gate of the run appears exactly
/// once.  So every one of the `m` dots reads the *same* per-amplitude weight
/// `z(i)`, and the only per-gate work left is a signed add.  Nothing is coupled
/// between amplitudes, which is why no strip, block or gather is involved and
/// why the win does not depend on the run fitting a cache.
/// `base` is the chunk's offset in the whole state.  The masks are absolute
/// qubit positions, so the sign has to be taken from `base + k`, not from the
/// chunk-local `k`: a chunk is an arbitrary slice of the state here, and only
/// the strip sweep was written so that its masks all sit below the strip width
/// (which is what let *it* get away with chunk-local indices).
fn diag_backward_blocks(
    pc: &mut [Complex64],
    sc: &mut [Complex64],
    base: usize,
    ops: &[DiagOp],
    dots: &mut [Complex64; DIAG_RUN_MAX],
) {
    for (k, (x, y)) in pc.iter_mut().zip(sc.iter_mut()).enumerate() {
        let i = base + k;
        let mut a = 0.0f64;
        for op in ops {
            a += op.t2[diag_parity(op, i)];
        }
        // `y` already holds `T(i)·psi_in(i)`, so this is exactly `z(i)`.
        let z = x.conj() * *y;
        let nz = -z;
        for (j, op) in ops.iter().enumerate() {
            dots[j] += if diag_parity(op, i) == 1 { nz } else { z };
        }
        let c = diag_phase(a, true);
        *x *= c;
        *y *= c;
    }
}

/// Apply a diagonal run to one slice: the whole run is a single per-amplitude
/// phase, so this is one multiply per amplitude and no dot at all.
///
/// `base` is the chunk's offset in the whole state, for the same reason as in
/// [`diag_backward_blocks`].
fn diag_forward_blocks(pc: &mut [Complex64], base: usize, ops: &[DiagOp]) {
    for (k, x) in pc.iter_mut().enumerate() {
        let i = base + k;
        let mut a = 0.0f64;
        for op in ops {
            a += op.t2[diag_parity(op, i)];
        }
        *x *= diag_phase(a, false);
    }
}

/// `SF_CAT_ADJ_DIAG=1` turns the diagonal-run sweep on -- and it is on by
/// default, because it measures as a win rather than as a wash.
///
/// A maximal run of `Rz`/`Rzz` is one diagonal operator, so the whole run costs
/// one traversal of both vectors instead of one per gate, and -- unlike
/// [`strip_enabled`] -- the run length is bounded by the tape, not by the cache:
/// a diagonal gate couples no two amplitudes, so there is nothing to gather and
/// no support to fit.  At n=22/L=4 that turns 65 tape passes per layer into 44.
///
/// Interleaved against the per-gate sweep on the same build, five pairs, one
/// gradient call each: **1.14x median** with every pair inside 1.10-1.22, and
/// `max|dg|` = 4.7e-17 against `|g|_inf` 2.869e-03 (`_cat_strip_ab.py --knob
/// SF_CAT_ADJ_DIAG`).  That is a smaller win than the 1.48x the pass count
/// suggests, and the reason is the same one §46.14 found from the other side: a
/// run of `m` gates still does `m` popcounts and `m` signed adds per amplitude,
/// so once the passes stop dominating, the per-amplitude arithmetic does.
fn diag_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| match std::env::var("SF_CAT_ADJ_DIAG") {
        Ok(v) => !matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "0" | "off" | "no" | "false" | "none"
        ),
        Err(_) => true,
    })
}

/// One thread's share of a fused run: sweep every block inside `pc`/`sc`.
///
/// A block is the `2^nq` amplitudes that agree on every bit outside `free`, so
/// blocks tile the slice and `base` — their shared non-`free` bits — is a
/// counter over them.  `spread[l]` maps the block-local index `l` to the
/// block's offset, so the gather and the scatter are `2^nq` strided loads and
/// stores while the run itself stays in the stack buffers.
fn fused_blocks(
    pc: &mut [Complex64],
    sc: &mut [Complex64],
    ops: &[FuseOp],
    spread: &[usize],
    nq: usize,
    free: usize,
) -> [Complex64; FUSE_RUN_MAX] {
    let nblk = 1usize << nq;
    let zero = Complex64::new(0.0, 0.0);
    let mut dots = [zero; FUSE_RUN_MAX];
    let mut pb = [zero; FUSE_BLK_MAX];
    let mut sb = [zero; FUSE_BLK_MAX];
    let mut base = 0usize;
    for _ in 0..pc.len() >> nq {
        for l in 0..nblk {
            pb[l] = pc[base | spread[l]];
            sb[l] = sc[base | spread[l]];
        }
        for (i, op) in ops.iter().enumerate() {
            dots[i] += op.step(&mut pb[..nblk], &mut sb[..nblk]);
        }
        for l in 0..nblk {
            pc[base | spread[l]] = pb[l];
            sc[base | spread[l]] = sb[l];
        }
        // Advance the non-`free` bits only: adds into the lowest hole and
        // clears the ones the carry passed.  Wraps to 0 after the last block.
        base = ((base | free) + 1) & !free;
    }
    dots
}

/// One thread's share of a strip run: apply every gate of the run to
/// `pc`/`sc`, in the order `ops` lists them (which is reverse tape order), and
/// return the run's dots.
///
/// This is [`fused_blocks`] without the gather.  Its ops carry masks relative to
/// the whole state, and a strip starts at a multiple of `2^bits` with every mask
/// below `2^bits`, so the low bits of an index inside the strip *are* its low
/// bits in the state and the very same kernels index correctly.  The strip is
/// the buffer, so there is nothing to gather and nothing to scatter: the only
/// traffic is the one streaming read and write of the gates' final state.
fn strip_blocks(pc: &mut [Complex64], sc: &mut [Complex64], ops: &[FuseOp]) -> Vec<Complex64> {
    let mut dots = Vec::with_capacity(ops.len());
    for op in ops {
        dots.push(op.step(pc, sc));
    }
    dots
}

/// A 1q rotation generator: which parameterised gate a `Rot` lowers to, and
/// the constant matrix it lowers to when its angle is already bound.
#[derive(Clone, Copy)]
enum Rot {
    Rx,
    Ry,
    Rz,
}

impl Rot {
    fn mat(self, t: f64) -> M2 {
        match self {
            Rot::Rx => m_rx(t),
            Rot::Ry => m_ry(t),
            Rot::Rz => m_rz(t),
        }
    }

    fn gate(self, q: u32, slot: u32) -> AdjGate {
        match self {
            Rot::Rx => AdjGate::Rx { q, slot },
            Rot::Ry => AdjGate::Ry { q, slot },
            Rot::Rz => AdjGate::Rz { q, slot },
        }
    }
}

/// A gate parameter resolved against the DAG: a live gradient slot, or a value
/// that is already bound and can be baked into the plan.
#[derive(Clone, Copy)]
enum Angle {
    Slot(u32),
    Const(f64),
}

fn angle_of(param: &Parameter, idx: &HashMap<&str, u32>) -> Result<Angle, String> {
    match param {
        Parameter::Variable { .. } => var_slot(param, idx).map(Angle::Slot),
        other => other.try_evaluate().map(Angle::Const).ok_or_else(|| {
            "cat adj: a gate parameter that neither names a differentiable \
             variable nor evaluates to a constant has no adjoint slot"
                .to_string()
        }),
    }
}

fn m_rx(t: f64) -> M2 {
    let c = Complex64::new((t / 2.0).cos(), 0.0);
    let s = Complex64::new(0.0, -(t / 2.0).sin());
    [[c, s], [s, c]]
}

fn m_ry(t: f64) -> M2 {
    let c = Complex64::new((t / 2.0).cos(), 0.0);
    let s = Complex64::new((t / 2.0).sin(), 0.0);
    [[c, -s], [s, c]]
}

fn m_rz(t: f64) -> M2 {
    let zero = Complex64::new(0.0, 0.0);
    let em = Complex64::from_polar(1.0, -t / 2.0);
    let ep = Complex64::from_polar(1.0, t / 2.0);
    [[em, zero], [zero, ep]]
}

fn m_p(t: f64) -> M2 {
    let one = Complex64::new(1.0, 0.0);
    let zero = Complex64::new(0.0, 0.0);
    [[one, zero], [zero, Complex64::from_polar(1.0, t)]]
}

fn m_h() -> M2 {
    let d = std::f64::consts::FRAC_1_SQRT_2;
    [
        [Complex64::new(d, 0.0), Complex64::new(d, 0.0)],
        [Complex64::new(d, 0.0), Complex64::new(-d, 0.0)],
    ]
}

fn m_x() -> M2 {
    let zero = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);
    [[zero, one], [one, zero]]
}

/// The `Cu` block, verbatim `sf_ir::ops::OpType::to_unitary_matrix`.
fn m_cu([t, p, l, g]: [f64; 4]) -> M2 {
    let ct = Complex64::new((t / 2.0).cos(), 0.0);
    let st = Complex64::new((t / 2.0).sin(), 0.0);
    let el = Complex64::from_polar(1.0, l);
    let ep = Complex64::from_polar(1.0, p);
    let epl = Complex64::from_polar(1.0, p + l);
    let eg = Complex64::from_polar(1.0, g);
    [[eg * ct, -eg * el * st], [eg * ep * st, eg * epl * ct]]
}

/// Compile-once adjoint program: flat gate list + preallocated, page-resident
/// state buffers.  Built once per circuit; `grad` / `expectation` may be
/// called any number of times with fresh parameter values.
pub struct AdjProgram {
    n: usize,
    dim: usize,
    p: usize,
    gates: Vec<AdjGate>,
    n_param_gates: usize,
    /// True when the reverse sweep walks `psi` back instead of reading a
    /// stored post-state per parameter gate (set above [`CACHE_ELEM_CAP`], when
    /// the cache would spill out of L3, or forced by `SF_CAT_ADJ_STREAM=1`).
    streaming: bool,
    /// Reverse-sweep thread width, read at construction from
    /// `SF_CAT_THREADS` / `SF_THREADS` (`0` = machine width, unset = 1).
    /// Per handle, so one process can A/B two widths.
    threads: usize,
    /// Whether the streaming sweep fuses consecutive small-support gates into
    /// one block traversal.  Read at construction from `SF_ADJ_FUSE`, per handle
    /// like `threads`, so one process can A/B the two sweeps.
    fuse: bool,
    /// Forward unitary per 1q gate (garbage on `Cx` entries), filled per call.
    fwd: Vec<[[Complex64; 2]; 2]>,
    /// Dagger of `fwd` (back-walk), filled per call.
    dg: Vec<[[Complex64; 2]; 2]>,
    /// Resolved `Cu` angles per gate (γ already folded into θ), per call.
    ang: Vec<[f64; 4]>,
    /// Forward state; starts at |0…0⟩ each call.
    psi: Vec<Complex64>,
    /// Post-gate state per parameterised gate, in gate order.
    cache: Vec<Complex64>,
    /// phi = O|psi_final> plus the term scratch (observable build).
    phi: Vec<Complex64>,
    term_state: Vec<Complex64>,
    grad: Vec<f64>,
}

impl AdjProgram {
    /// The benchmark ansatz: per layer, RY then RZ on every qubit (slots
    /// assigned in that order), then a CNOT ladder `(i, i+1)`.  `P = 2*n*layers`.
    pub fn from_ladder(n: usize, layers: usize) -> Self {
        let mut gates = Vec::with_capacity(layers * (3 * n));
        let mut slot = 0u32;
        for _ in 0..layers {
            for i in 0..n {
                gates.push(AdjGate::Ry { q: i as u32, slot });
                slot += 1;
                gates.push(AdjGate::Rz { q: i as u32, slot });
                slot += 1;
            }
            for i in 0..n.saturating_sub(1) {
                gates.push(AdjGate::Cx {
                    ctl: i as u32,
                    tgt: i as u32 + 1,
                });
            }
        }
        Self::new(n, gates, slot as usize)
    }

    /// Lower a `QuantumDAG` of RY/RZ (variable parameters) + CNOT gates.
    /// Parameter slots follow the DAG's `parameter_names()` order, so the
    /// gradient vector is aligned with `sf_ir::adjoint::adjoint_grad`.
    pub fn from_dag(dag: &QuantumDAG) -> Result<Self, String> {
        let names = dag.parameter_names();
        let idx: HashMap<&str, u32> = names
            .iter()
            .enumerate()
            .map(|(i, s)| (s.as_str(), i as u32))
            .collect();
        let mut gates: Vec<AdjGate> = Vec::new();
        // A `Reset` on a qubit nothing has touched yet is the identity on |0>,
        // so it must not disqualify an otherwise differentiable circuit -- the
        // stabilizer/clifford lanes already skip exactly those.  A reset *after*
        // use is a real collapse and stays refused, so `from_dag` remains safe
        // for callers that do not run `reject_dynamic_ops` first (it is public
        // and the benchmark example calls it directly).
        let nq = dag.n_qubits;
        let mut touched = vec![false; nq];
        let mut measured = vec![false; nq];
        for &node in dag.topological_order().iter() {
            let op = &dag.graph()[node];
            if op.op_type.is_boundary() || op.op_type == OpType::Barrier {
                continue;
            }
            if op.op_type.is_measurement() {
                for &q in op.qubits.iter() {
                    if q < nq {
                        measured[q] = true;
                    }
                }
                continue;
            }
            if op.op_type == OpType::Reset {
                for &q in op.qubits.iter() {
                    if q < nq && (touched[q] || measured[q]) {
                        return Err(format!(
                            "cat: adjoint plan: Reset on qubit {q} after it was used; \
                             reset means collapse-and-reprepare, which the adjoint \
                             program does not model - use sf.run(circuit, \
                             method=\"statevector\", shots=N) for a trajectory gradient"
                        ));
                    }
                }
                // Leading reset: no tombstone, no gate -- exactly the identity.
                continue;
            }
            for &q in op.qubits.iter() {
                if q < nq {
                    touched[q] = true;
                }
            }
            match &op.op_type {
                OpType::Ry(param) => {
                    let q = one_qubit(&op.qubits, "Ry")?;
                    push_rot(&mut gates, Rot::Ry, q, param, &idx)?;
                }
                OpType::Rz(param) => {
                    let q = one_qubit(&op.qubits, "Rz")?;
                    push_rot(&mut gates, Rot::Rz, q, param, &idx)?;
                }
                OpType::Rx(param) => {
                    let q = one_qubit(&op.qubits, "Rx")?;
                    push_rot(&mut gates, Rot::Rx, q, param, &idx)?;
                }
                // `P(φ) = diag(1, e^{iφ}) = e^{iφ/2}·Rz(φ)`: the extra factor is
                // a global phase, so a live `P` differentiates and applies
                // exactly like `Rz`, and a bound one keeps its exact matrix.
                OpType::P(param) | OpType::R1(param) => {
                    let q = one_qubit(&op.qubits, "P")?;
                    match angle_of(param, &idx)? {
                        Angle::Slot(slot) => gates.push(AdjGate::Rz { q, slot }),
                        Angle::Const(v) => gates.push(AdjGate::Fixed { q, m: m_p(v) }),
                    }
                }
                // `U(θ,φ,λ) = Rz(φ)·Ry(θ)·Rz(λ)` up to a global phase.
                OpType::U(t, p, l) => {
                    let q = one_qubit(&op.qubits, "U")?;
                    push_rot(&mut gates, Rot::Rz, q, l, &idx)?;
                    push_rot(&mut gates, Rot::Ry, q, t, &idx)?;
                    push_rot(&mut gates, Rot::Rz, q, p, &idx)?;
                }
                // Constant single-qubit unitaries: no slot, but they still have
                // to be applied (and undone) or the state is the wrong one.
                OpType::H => gates.push(AdjGate::H {
                    q: one_qubit(&op.qubits, "H")?,
                }),
                OpType::X => gates.push(AdjGate::X {
                    q: one_qubit(&op.qubits, "X")?,
                }),
                // Every other 1q gate with a matrix in the IR is baked here.
                OpType::Y
                | OpType::Z
                | OpType::S
                | OpType::Sdg
                | OpType::T
                | OpType::Tdg
                | OpType::SX
                | OpType::SXdg => {
                    let q = one_qubit(&op.qubits, &op.op_type.name())?;
                    let m = op.op_type.to_unitary_matrix().ok_or_else(|| {
                        format!(
                            "cat: adjoint plan: {} has no 1q matrix to bake",
                            op.op_type.name()
                        )
                    })?;
                    gates.push(AdjGate::Fixed { q, m: m2(&m)? });
                }
                OpType::Id => {}
                OpType::CNOT => {
                    let (ctl, tgt) = two_qubit(&op.qubits, "CNOT")?;
                    gates.push(AdjGate::Cx { ctl, tgt });
                }
                // `CZ = diag(1,1,1,-1)` and `CY = diag(I, Y)` are `Cu` blocks
                // with every angle bound, so no extra machinery is needed.
                // The `-1` sits on the *target*, i.e. in U3, not in γ: γ phases
                // the whole control-1 block (`diag(1,1,-1,-1)`), which is a `Z`
                // on the control, not a CZ.
                OpType::CZ => {
                    let (ctl, tgt) = two_qubit(&op.qubits, "CZ")?;
                    gates.push(cu_gate(
                        ctl,
                        tgt,
                        [
                            Angle::Const(0.0),
                            Angle::Const(0.0),
                            Angle::Const(std::f64::consts::PI),
                            Angle::Const(0.0),
                        ],
                        0.0,
                        3,
                    ));
                }
                OpType::CY => {
                    let (ctl, tgt) = two_qubit(&op.qubits, "CY")?;
                    let half = std::f64::consts::FRAC_PI_2;
                    gates.push(cu_gate(
                        ctl,
                        tgt,
                        [
                            Angle::Const(std::f64::consts::PI),
                            Angle::Const(half),
                            Angle::Const(half),
                            Angle::Const(0.0),
                        ],
                        0.0,
                        3,
                    ));
                }
                // Constant monomial gates: one index map each.  `SWAP` used
                // to lower to three CNOTs and cost three traversals; `iSWAP`
                // and `CSWAP` were refused outright, and `CCX` (which has no
                // two-qubit-only adjoint lane) with them.  All four are now
                // one traversal per sweep, on both the forward pass and the
                // reverse.
                OpType::SWAP => {
                    let (a, b) = two_qubit(&op.qubits, "SWAP")?;
                    gates.push(AdjGate::Mono {
                        kind: Mono::Swap,
                        qs: [a, b, 0],
                        k: 2,
                    });
                }
                OpType::ISWAP => {
                    let (a, b) = two_qubit(&op.qubits, "iSWAP")?;
                    gates.push(AdjGate::Mono {
                        kind: Mono::ISwap,
                        qs: [a, b, 0],
                        k: 2,
                    });
                }
                OpType::CCX => {
                    let (c1, c2, t) = three_qubit(&op.qubits, "CCX")?;
                    gates.push(AdjGate::Mono {
                        kind: Mono::Ccx,
                        qs: [c1, c2, t],
                        k: 3,
                    });
                }
                OpType::CSWAP => {
                    let (c, a, b) = three_qubit(&op.qubits, "CSWAP")?;
                    gates.push(AdjGate::Mono {
                        kind: Mono::Cswap,
                        qs: [c, a, b],
                        k: 3,
                    });
                }
                // `Rzz(θ) = CNOT·(I⊗Rz(θ))·CNOT` — exact, no phase left over.
                // With a variable angle the three gates collapse to one
                // diagonal pattern (`AdjGate::Rzz`); a bound angle keeps the
                // three-gate form, which needs no new baking path.
                OpType::Rzz(param) => {
                    let (a, b) = two_qubit(&op.qubits, "Rzz")?;
                    match angle_of(param, &idx)? {
                        Angle::Slot(s) => gates.push(AdjGate::Rzz { a, b, slot: s }),
                        Angle::Const(_) => {
                            gates.push(AdjGate::Cx { ctl: a, tgt: b });
                            push_rot(&mut gates, Rot::Rz, b, param, &idx)?;
                            gates.push(AdjGate::Cx { ctl: a, tgt: b });
                        }
                    }
                }
                // `Rxx(θ) = (H⊗H)·Rzz(θ)·(H⊗H)` since `H Z H = X`.  With a
                // variable angle the whole chain collapses to one 2×2 rotation
                // per amplitude pair (`AdjGate::Rxx`); a bound angle keeps the
                // five-gate form, which needs no new baking path.
                OpType::Rxx(param) => {
                    let (a, b) = two_qubit(&op.qubits, "Rxx")?;
                    match angle_of(param, &idx)? {
                        Angle::Slot(s) => gates.push(AdjGate::Rxx {
                            a,
                            b,
                            slot: s,
                            yy: false,
                        }),
                        Angle::Const(_) => {
                            for q in [a, b] {
                                gates.push(AdjGate::Fixed { q, m: m_h() });
                            }
                            gates.push(AdjGate::Cx { ctl: a, tgt: b });
                            push_rot(&mut gates, Rot::Rz, b, param, &idx)?;
                            gates.push(AdjGate::Cx { ctl: a, tgt: b });
                            for q in [a, b] {
                                gates.push(AdjGate::Fixed { q, m: m_h() });
                            }
                        }
                    }
                }
                // `Ryy(θ) = (Rx(π/2)⊗Rx(π/2))·Rzz(θ)·(Rx(−π/2)⊗Rx(−π/2))`
                // because `Rx(π/2) Z Rx(−π/2) = −Y`, and the sign squares out.
                // Same fusion as `Rxx`, with the agree-pair sign flipped.
                OpType::Ryy(param) => {
                    let (a, b) = two_qubit(&op.qubits, "Ryy")?;
                    match angle_of(param, &idx)? {
                        Angle::Slot(s) => gates.push(AdjGate::Rxx {
                            a,
                            b,
                            slot: s,
                            yy: true,
                        }),
                        Angle::Const(_) => {
                            let u = m_rx(std::f64::consts::FRAC_PI_2);
                            let ud = m_rx(-std::f64::consts::FRAC_PI_2);
                            for q in [a, b] {
                                gates.push(AdjGate::Fixed { q, m: u });
                            }
                            gates.push(AdjGate::Cx { ctl: a, tgt: b });
                            push_rot(&mut gates, Rot::Rz, b, param, &idx)?;
                            gates.push(AdjGate::Cx { ctl: a, tgt: b });
                            for q in [a, b] {
                                gates.push(AdjGate::Fixed { q, m: ud });
                            }
                        }
                    }
                }
                // Controlled rotations: one `Cu` block each, exact.  `CRx` is
                // `U3(θ,−π/2,π/2)`, `CP` is `Cu(0,0,φ,0)`, and `CRz` is
                // `e^{-iθ/2}·U3(0,0,θ) = diag(e^{-iθ/2}, e^{iθ/2})`, so its θ
                // rides on λ and γ (via `gscale`/`gref`) and never on U3's own
                // θ — `U3(θ,0,θ)` is *not* diagonal, it has `±sin(θ/2)` off it.
                OpType::CRx(param) => {
                    let (ctl, tgt) = two_qubit(&op.qubits, "CRx")?;
                    let half = std::f64::consts::FRAC_PI_2;
                    gates.push(cu_gate(
                        ctl,
                        tgt,
                        [
                            angle_of(param, &idx)?,
                            Angle::Const(-half),
                            Angle::Const(half),
                            Angle::Const(0.0),
                        ],
                        0.0,
                        3,
                    ));
                }
                OpType::CRz(param) => {
                    let (ctl, tgt) = two_qubit(&op.qubits, "CRz")?;
                    gates.push(cu_gate(
                        ctl,
                        tgt,
                        [
                            Angle::Const(0.0),
                            Angle::Const(0.0),
                            angle_of(param, &idx)?,
                            Angle::Const(0.0),
                        ],
                        -0.5,
                        2,
                    ));
                }
                OpType::CP(param) => {
                    let (ctl, tgt) = two_qubit(&op.qubits, "CP")?;
                    gates.push(cu_gate(
                        ctl,
                        tgt,
                        [
                            Angle::Const(0.0),
                            Angle::Const(0.0),
                            angle_of(param, &idx)?,
                            Angle::Const(0.0),
                        ],
                        0.0,
                        3,
                    ));
                }
                OpType::Cu(t, p, l, g) => {
                    let (ctl, tgt) = two_qubit(&op.qubits, "Cu")?;
                    gates.push(cu_gate(
                        ctl,
                        tgt,
                        [
                            angle_of(t, &idx)?,
                            angle_of(p, &idx)?,
                            angle_of(l, &idx)?,
                            angle_of(g, &idx)?,
                        ],
                        0.0,
                        3,
                    ));
                }
                other => {
                    return Err(format!(
                        "cat: adjoint plan: unsupported gate {}; the adjoint lane \
                         differentiates RY/RZ/RX, P/R1, U, CRX/CRZ/CP/CU and \
                         RXX/RYY/RZZ with variable parameters, and applies \
                         H/X/Y/Z/S/T/SX, CZ/CY/SWAP/iSWAP/CCX/CSWAP and CNOT \
                         as constants, so it has no derivative for this gate - \
                         rewrite the circuit to that gate set",
                        other.name()
                    ))
                }
            }
        }
        let p = names.len();
        let n = dag.n_qubits;
        if n >= 63 {
            return Err("cat adj: n must be in 1..=62".to_string());
        }
        // A cache over the cap is no longer a refusal: `new` switches this
        // plan to the streaming reverse sweep, which has the same arithmetic
        // and costs one extra daggered application per gate.
        Ok(Self::new(n, gates, p))
    }

    fn new(n: usize, gates: Vec<AdjGate>, p: usize) -> Self {
        assert!((1..63).contains(&n), "cat adj: n must be in 1..=62");
        let dim = 1usize << n;
        for g in &gates {
            match *g {
                AdjGate::Ry { q, slot } | AdjGate::Rz { q, slot } | AdjGate::Rx { q, slot } => {
                    assert!((q as usize) < n, "cat adj: qubit out of range");
                    assert!((slot as usize) < p, "cat adj: slot out of range");
                }
                AdjGate::H { q } | AdjGate::X { q } => {
                    assert!((q as usize) < n, "cat adj: qubit out of range");
                }
                AdjGate::Fixed { q, .. } => {
                    assert!((q as usize) < n, "cat adj: qubit out of range");
                }
                AdjGate::Cu {
                    ctl, tgt, slots, ..
                } => {
                    assert!(
                        (ctl as usize) < n && (tgt as usize) < n && ctl != tgt,
                        "cat adj: bad controlled-gate pair"
                    );
                    for &s in slots.iter() {
                        assert!(s < 0 || (s as usize) < p, "cat adj: slot out of range");
                    }
                }
                AdjGate::Cx { ctl, tgt } => {
                    assert!(
                        (ctl as usize) < n && (tgt as usize) < n && ctl != tgt,
                        "cat adj: bad CNOT pair"
                    );
                }
                AdjGate::Rzz { a, b, slot } => {
                    assert!(
                        (a as usize) < n && (b as usize) < n && a != b,
                        "cat adj: bad Rzz pair"
                    );
                    assert!((slot as usize) < p, "cat adj: slot out of range");
                }
                AdjGate::Rxx { a, b, slot, .. } => {
                    assert!(
                        (a as usize) < n && (b as usize) < n && a != b,
                        "cat adj: bad Rxx pair"
                    );
                    assert!((slot as usize) < p, "cat adj: slot out of range");
                }
                AdjGate::Mono { qs, k, .. } => {
                    let q = &qs[..k as usize];
                    assert!(
                        q.iter().all(|&x| (x as usize) < n),
                        "cat adj: qubit out of range"
                    );
                    // The index map reads one bit per role, so a repeated
                    // qubit would silently fold two roles onto one bit.
                    assert!(
                        q.iter().enumerate().all(|(i, x)| !q[..i].contains(x)),
                        "cat adj: repeated qubit in a constant multi-qubit gate"
                    );
                }
            }
        }
        let n_gates = gates.len();
        let n_param_gates = gates.iter().filter(|g| g.is_param()).count();
        // Above the cache cap (or when forced) the handle streams: it keeps
        // O(2^n) memory and re-walks `psi` in the reverse sweep instead of
        // storing a post-state per parameter, so no circuit size is refused.
        // Below the cap the cache still only pays while it stays resident or
        // the circuit is mostly constants — see `cache_is_worth_it`.
        let streaming = n_param_gates * dim > CACHE_ELEM_CAP
            || force_streaming()
            || (!force_cached() && !cache_is_worth_it(n_param_gates, n_gates, dim));
        let zero = Complex64::new(0.0, 0.0);
        Self {
            n,
            dim,
            p,
            gates,
            n_param_gates,
            streaming,
            threads: env_width(),
            fuse: fuse_enabled(),
            fwd: vec![[[zero; 2]; 2]; n_gates],
            dg: vec![[[zero; 2]; 2]; n_gates],
            ang: vec![[0.0; 4]; n_gates],
            psi: vec![zero; dim],
            cache: if streaming {
                Vec::new()
            } else {
                vec![zero; n_param_gates * dim]
            },
            phi: vec![zero; dim],
            term_state: vec![zero; dim],
            grad: vec![0.0; p],
        }
    }

    pub fn n_qubits(&self) -> usize {
        self.n
    }

    /// Number of parameters (gradient length).
    pub fn num_params(&self) -> usize {
        self.p
    }

    /// Fill `fwd`/`dg` for every parameterised gate from `theta`.
    /// Matrix formulas are verbatim sf-ir `OpType::to_matrix` (Ry/Rz).
    fn fill_matrices(&mut self, theta: &[f64]) {
        for k in 0..self.gates.len() {
            let m = match self.gates[k] {
                AdjGate::Ry { slot, .. } => m_ry(theta[slot as usize]),
                AdjGate::Rz { slot, .. } => {
                    // Kept alongside the matrix so the diagonal-run kernel can
                    // read θ instead of inverting `m_rz`; nothing else reads
                    // `ang` for a single-qubit gate.
                    self.ang[k][0] = theta[slot as usize];
                    m_rz(theta[slot as usize])
                }
                AdjGate::Rx { slot, .. } => m_rx(theta[slot as usize]),
                AdjGate::H { .. } => m_h(),
                AdjGate::X { .. } => m_x(),
                AdjGate::Fixed { m, .. } => m,
                AdjGate::Cu {
                    slots,
                    consts,
                    gscale,
                    gref,
                    ..
                } => {
                    let mut a = consts;
                    for i in 0..4 {
                        if slots[i] >= 0 {
                            a[i] = theta[slots[i] as usize];
                        }
                    }
                    // γ ← γ + gscale·a[gref] is what keeps controlled-RZ and
                    // the other derived blocks exact.
                    a[3] += gscale * a[gref as usize];
                    self.ang[k] = a;
                    m_cu(a)
                }
                AdjGate::Rzz { slot, .. } => {
                    // Nothing to bake: the two phases the kernels need are
                    // derived from θ at apply time (`zz_phase`).
                    self.ang[k][0] = theta[slot as usize];
                    continue;
                }
                AdjGate::Rxx { slot, .. } => {
                    // Nothing to bake: the 2×2 block is derived from θ at
                    // apply time (`xx_coef`).
                    self.ang[k][0] = theta[slot as usize];
                    continue;
                }
                AdjGate::Cx { .. } => continue,
                // Nothing to bake: the index map and its `±i` phases depend
                // only on which qubits play which role, not on `theta`.
                AdjGate::Mono { .. } => continue,
            };
            self.fwd[k] = m;
            self.dg[k] = [
                [m[0][0].conj(), m[1][0].conj()],
                [m[0][1].conj(), m[1][1].conj()],
            ];
        }
    }

    /// Forward pass: |0…0⟩ through every gate, caching the post-state of
    /// each parameterised gate (sf-ir's cached variant).
    fn forward(&mut self, diag: bool) {
        match adj_pool() {
            Some(p) => p.install(|| self.forward_sweep(diag)),
            None => self.forward_sweep(diag),
        }
    }

    /// Forward sweep; runs inside [`adj_pool`] when a width is requested.
    ///
    /// `diag` applies a maximal `Rz`/`Rzz` run as the single diagonal operator
    /// it is, which is only licenced while streaming: the cached path stores the
    /// post-state of *every* parameter gate, so it needs exactly the
    /// intermediate states the run exists to skip.
    fn forward_sweep(&mut self, diag: bool) {
        for x in self.psi.iter_mut() {
            *x = Complex64::new(0.0, 0.0);
        }
        self.psi[0] = Complex64::new(1.0, 0.0);
        let dim = self.dim;
        let mut slot = 0usize;
        let runs = diag && self.streaming;
        let mut k = 0usize;
        while k < self.gates.len() {
            if runs {
                let end = self.diag_run_from(k);
                if end - k >= 2 {
                    self.diag_forward(k, end);
                    // `slot` has to keep counting even though the run writes no
                    // cache entry, or the two bookkeeping paths drift apart.
                    for g in &self.gates[k..end] {
                        if g.is_param() {
                            slot += 1;
                        }
                    }
                    k = end;
                    continue;
                }
            }
            match self.gates[k] {
                AdjGate::Ry { q, .. }
                | AdjGate::Rz { q, .. }
                | AdjGate::Rx { q, .. }
                | AdjGate::H { q }
                | AdjGate::X { q }
                | AdjGate::Fixed { q, .. } => {
                    let m = self.fwd[k];
                    par_pair_pass(&mut self.psi, q as usize, m);
                }
                AdjGate::Cu { ctl, tgt, .. } => {
                    let m = self.fwd[k];
                    par_c1_apply(&mut self.psi, ctl as usize, tgt as usize, m);
                }
                AdjGate::Rzz { a, b, .. } => {
                    let (hm, lm) = (1usize << a.max(b), 1usize << a.min(b));
                    par_zz_apply(&mut self.psi, hm, lm, zz_phase(self.ang[k][0], false));
                }
                AdjGate::Rxx { a, b, yy, .. } => {
                    let (hm, lm) = (1usize << a.max(b), 1usize << a.min(b));
                    let (c, kc) = xx_coef(self.ang[k][0], yy, false);
                    par_xx_apply(&mut self.psi, hm, lm, c, kc);
                }
                AdjGate::Cx { ctl, tgt } => {
                    par_cx_apply(&mut self.psi, ctl as usize, tgt as usize);
                }
                AdjGate::Mono { kind, qs, k } => {
                    par_mono_apply(&mut self.psi, kind, qs, k as usize, false);
                }
            }
            if self.gates[k].is_param() {
                if !self.streaming {
                    self.cache[slot * dim..(slot + 1) * dim].copy_from_slice(&self.psi);
                }
                slot += 1;
            }
            k += 1;
        }
    }

    /// phi = Σ_terms coef · P|psi_final>, porting sf-ir's observable builder
    /// (fused X/Z ≤ 2 non-identity factors, else copy + Pauli applications).
    fn build_phi(&mut self, terms: &[PauliTerm]) {
        match adj_pool() {
            Some(p) => p.install(|| self.build_phi_sweep(terms)),
            None => self.build_phi_sweep(terms),
        }
    }

    /// Observable build; runs inside [`adj_pool`] when a width is requested.
    fn build_phi_sweep(&mut self, terms: &[PauliTerm]) {
        for x in self.phi.iter_mut() {
            *x = Complex64::new(0.0, 0.0);
        }
        for term in terms {
            let mut xmask = 0usize;
            let mut zmask = 0usize;
            let mut fusable = true;
            for (q, &p) in term.paulis.iter().enumerate() {
                match p {
                    0 => {}
                    1 => xmask |= 1usize << q,
                    3 => zmask |= 1usize << q,
                    _ => fusable = false,
                }
            }
            // The fused form below is an X permutation (`xmask`) times a Z
            // phase (`zmask`), which is exact for a Pauli string of *any*
            // weight: Z^{\otimes n} costs one pass here, not the n+2 passes of
            // the copy-and-apply fallback.  Only terms carrying Y need the
            // fallback, because Y is not a diagonal sign flip.
            if fusable {
                par_phi_add_xz(&mut self.phi, &self.psi, term.coef, xmask, zmask);
            } else {
                self.term_state.copy_from_slice(&self.psi);
                for (q, &p) in term.paulis.iter().enumerate() {
                    if p != 0 {
                        par_apply_pauli_1q(&mut self.term_state, p, q);
                    }
                }
                par_add_scaled(&mut self.phi, &self.term_state, term.coef);
            }
        }
    }

    /// Backward sweep: gradient per parameter gate (generator dot against the
    /// cached post-state), then un-apply the gate on phi.
    fn backward(&mut self) {
        match adj_pool() {
            Some(p) => p.install(|| self.backward_sweep()),
            None => self.backward_sweep(),
        }
    }

    /// Cached reverse sweep; runs inside [`adj_pool`] when a width is requested.
    fn backward_sweep(&mut self) {
        for g in self.grad.iter_mut() {
            *g = 0.0;
        }
        let dim = self.dim;
        let mut slot = self.n_param_gates;
        for k in (0..self.gates.len()).rev() {
            match self.gates[k] {
                AdjGate::Ry { q, slot: s } => {
                    slot -= 1;
                    // grad += 2·alpha·Im⟨φ|G|ψ_after⟩ with alpha = 0.5
                    // (sf-ir's generator_info factor for Rx/Ry/Rz), i.e.
                    // grad += Im⟨φ|Y|ψ_after⟩ here.  The dot and the daggered
                    // update are one traversal, so φ is read once, not twice.
                    let ip = par_dot_and_apply(
                        2,
                        &mut self.phi,
                        &self.cache[slot * dim..(slot + 1) * dim],
                        1usize << q,
                        self.dg[k],
                    );
                    self.grad[s as usize] += 2.0 * 0.5 * ip.im;
                }
                AdjGate::Rz { q, slot: s } => {
                    slot -= 1;
                    let ip = par_dot_and_apply(
                        3,
                        &mut self.phi,
                        &self.cache[slot * dim..(slot + 1) * dim],
                        1usize << q,
                        self.dg[k],
                    );
                    self.grad[s as usize] += 2.0 * 0.5 * ip.im;
                }
                AdjGate::Rx { q, slot: s } => {
                    slot -= 1;
                    let ip = par_dot_and_apply(
                        1,
                        &mut self.phi,
                        &self.cache[slot * dim..(slot + 1) * dim],
                        1usize << q,
                        self.dg[k],
                    );
                    self.grad[s as usize] += 2.0 * 0.5 * ip.im;
                }
                AdjGate::Rzz { a, b, slot: s } => {
                    slot -= 1;
                    let (hm, lm) = (1usize << a.max(b), 1usize << a.min(b));
                    let ip = par_dot_zz_apply(
                        &mut self.phi,
                        &self.cache[slot * dim..(slot + 1) * dim],
                        hm,
                        lm,
                        zz_phase(self.ang[k][0], true),
                    );
                    self.grad[s as usize] += ip.im;
                }
                AdjGate::Rxx { a, b, slot: s, yy } => {
                    slot -= 1;
                    let (hm, lm) = (1usize << a.max(b), 1usize << a.min(b));
                    let (c, kc) = xx_coef(self.ang[k][0], yy, true);
                    let ip = par_dot_xx_apply(
                        &mut self.phi,
                        &self.cache[slot * dim..(slot + 1) * dim],
                        hm,
                        lm,
                        c,
                        kc,
                        yy,
                    );
                    self.grad[s as usize] += ip.im;
                }
                AdjGate::H { q } | AdjGate::X { q } | AdjGate::Fixed { q, .. } => {
                    let m = self.dg[k];
                    par_pair_pass(&mut self.phi, q as usize, m);
                }
                AdjGate::Cu {
                    ctl,
                    tgt,
                    slots,
                    gscale,
                    gref,
                    ..
                } => {
                    let (cm, tm) = (1usize << ctl, 1usize << tgt);
                    // A `Cu` with every angle bound (CZ, CY) owns no cache
                    // entry: only the daggered re-application is due.
                    if self.gates[k].is_param() {
                        slot -= 1;
                        let a = self.ang[k];
                        let (ax, ay, az, d) = par_dot_xyz_c1(
                            &self.phi,
                            &self.cache[slot * dim..(slot + 1) * dim],
                            cm,
                            tm,
                        );
                        self.cu_grad(slots, gscale, gref, a, ax, ay, az, d);
                    }
                    let m = self.dg[k];
                    par_c1_apply(&mut self.phi, ctl as usize, tgt as usize, m);
                }
                AdjGate::Cx { ctl, tgt } => {
                    par_cx_apply(&mut self.phi, ctl as usize, tgt as usize);
                }
                AdjGate::Mono { kind, qs, k } => {
                    par_mono_apply(&mut self.phi, kind, qs, k as usize, true);
                }
            }
        }
    }

    /// Reverse sweep without the parameter-state cache: `psi` is walked
    /// backwards one gate at a time, so the state the generator dot needs is
    /// whatever `psi` holds at that point.  Same formulas and the same dot
    /// order as [`AdjProgram::backward`]; the extra cost is one daggered gate
    /// application per gate, traded against not storing P*2^n complex.
    ///
    /// Both vectors are updated in one traversal per gate (`dot_apply_pair`,
    /// `dot_xyz_pair_c1`, `c1_apply_pair`, `cx_apply_pair`), so `psi` and
    /// `phi` are each streamed once rather than twice.  Keep that property
    /// when adding a gate: the whole point of this path is that it touches
    /// memory twice per gate instead of four times.
    fn backward_streaming(&mut self, fuse: bool, strip: bool, diag: bool) {
        match adj_pool() {
            Some(p) => p.install(|| self.backward_streaming_sweep(fuse, strip, diag)),
            None => self.backward_streaming_sweep(fuse, strip, diag),
        }
    }

    /// Streaming reverse sweep; runs inside [`adj_pool`] when a width is requested.
    ///
    /// `fuse` selects between the run-fused sweep and the one-gate-at-a-time
    /// sweep it replaced.  `strip` selects the strip sweep in place of the
    /// latter, and `diag` the diagonal-run sweep -- the only one of the three
    /// whose win is algebraic rather than cache-shaped.  All are parameters
    /// rather than reads of [`fuse_enabled`], [`strip_enabled`] and
    /// [`diag_enabled`] so a test can run every shape on one program: the env
    /// switches are latched once, so they cannot A/B in-process.
    fn backward_streaming_sweep(&mut self, fuse: bool, strip: bool, diag: bool) {
        for g in self.grad.iter_mut() {
            *g = 0.0;
        }
        if fuse {
            let mut k = self.gates.len();
            while k > 0 {
                let (start, qs, nq) = self.fusable_run(k);
                // A single-gate run has nothing to share a traversal with.
                if k - start >= 2 {
                    self.fused_backward(start, k, &qs[..nq], nq);
                    k = start;
                } else {
                    k -= 1;
                    self.step_backward(k);
                }
            }
        } else if diag {
            // A whole `Rz`/`Rzz` run costs one traversal, however many qubits
            // it spans: diagonal gates never couple amplitudes.
            let mut k = self.gates.len();
            while k > 0 {
                let start = self.diag_run(k);
                if k - start >= 2 {
                    self.diag_backward(start, k);
                    k = start;
                } else {
                    k -= 1;
                    self.step_backward(k);
                }
            }
        } else if strip {
            // Same sweep, one traversal per run instead of one per gate: the
            // run's support fits a strip, so it is applied while resident.
            let limit = strip_limit(self.dim, adj_threads());
            let mut k = self.gates.len();
            while k > 0 {
                let (start, bits) = self.strip_run(k, limit);
                if k - start >= 2 {
                    self.strip_backward(start, k, bits);
                    k = start;
                } else {
                    k -= 1;
                    self.step_backward(k);
                }
            }
        } else {
            for k in (0..self.gates.len()).rev() {
                self.step_backward(k);
            }
        }
    }

    /// Longest run ending at (exclusive) `end` whose gates touch at most
    /// [`MAX_RUN_Q`] distinct qubits, and that span those qubits.
    ///
    /// Fusion needs nothing else: inside a run the gates are still stepped in
    /// reverse tape order, one after another, so non-commuting neighbours are
    /// as correct as they are in the unfused sweep.  The only requirement is
    /// that every amplitude a gate couples to sits in the same block, which
    /// holds exactly when the block covers the run's whole support.
    fn fusable_run(&self, end: usize) -> (usize, [usize; MAX_RUN_Q], usize) {
        let mut qs = [0usize; MAX_RUN_Q];
        let mut nq = 0usize;
        let mut start = end;
        while start > 0 && end - start < FUSE_RUN_MAX {
            let (qa, qb) = match self.gates[start - 1] {
                AdjGate::Ry { q, .. }
                | AdjGate::Rz { q, .. }
                | AdjGate::Rx { q, .. }
                | AdjGate::H { q }
                | AdjGate::X { q }
                | AdjGate::Fixed { q, .. } => (q as usize, usize::MAX),
                AdjGate::Rzz { a, b, .. } | AdjGate::Rxx { a, b, .. } => {
                    (a.min(b) as usize, a.max(b) as usize)
                }
                // Cu/Cx/Mono keep the per-gate path: their reverse step is not
                // one of the (dot, apply) kernels a block can host.
                _ => break,
            };
            let mut next = qs;
            let mut next_nq = nq;
            for q in [qa, qb] {
                if q == usize::MAX || next[..next_nq].contains(&q) {
                    continue;
                }
                if next_nq == MAX_RUN_Q {
                    next_nq = usize::MAX;
                    break;
                }
                next[next_nq] = q;
                next_nq += 1;
            }
            if next_nq == usize::MAX {
                break;
            }
            next[..next_nq].sort_unstable();
            qs = next;
            nq = next_nq;
            start -= 1;
        }
        (start, qs, nq)
    }

    /// Reverse-apply gates `start..end` in one traversal per block of
    /// `2^nq` amplitudes.
    ///
    /// The per-gate sweep reads and writes both vectors once per gate.  A run
    /// of `L` gates touching `k` qubits is applied instead to the `2^k`
    /// amplitudes those qubits span, gathered into a 16-element stack buffer:
    /// the run then costs one read and one write of both vectors total, i.e.
    /// `L` times less traffic.  Nothing about the arithmetic changes -- the
    /// same kernels (`dot_apply_pair`, `dot_zz_apply_pair`,
    /// `dot_xx_apply_pair`) run on the buffer, because they index by a relative
    /// bit mask and never look outside the slice they are handed.
    fn fused_backward(&mut self, start: usize, end: usize, qs: &[usize], nq: usize) {
        let nblk = 1usize << nq;
        let free = qs.iter().fold(0usize, |acc, &q| acc | (1usize << q));
        let mut spread = [0usize; FUSE_BLK_MAX];
        for (l, slot) in spread[..nblk].iter_mut().enumerate() {
            let mut g = 0usize;
            for (r, &q) in qs.iter().enumerate() {
                if (l >> r) & 1 == 1 {
                    g |= 1usize << q;
                }
            }
            *slot = g;
        }
        let local = |q: usize| 1usize << qs.iter().position(|&x| x == q).unwrap_or(0);

        let mut ops = [FuseOp::empty(); FUSE_RUN_MAX];
        let mut nops = 0usize;
        // Reverse tape order: `ops[0]` is the last gate of the run, so the
        // buffers see the same state sequence the per-gate sweep gives them.
        for k in (start..end).rev() {
            let mut op = FuseOp {
                gate: self.dg[k],
                ..FuseOp::empty()
            };
            match self.gates[k] {
                // `Rot(p)` mirrors the `p` each per-gate arm passes to
                // `par_dot_apply_pair`.
                AdjGate::Ry { q, slot } => {
                    op.kind = FuseKind::Rot(2);
                    op.ma = local(q as usize);
                    op.slot = slot;
                }
                AdjGate::Rz { q, slot } => {
                    op.kind = FuseKind::Rot(3);
                    op.ma = local(q as usize);
                    op.slot = slot;
                }
                AdjGate::Rx { q, slot } => {
                    op.kind = FuseKind::Rot(1);
                    op.ma = local(q as usize);
                    op.slot = slot;
                }
                // `H`/`X`/`Fixed` are pure rewinds, which is exactly what an
                // `empty()` op degrades to once it knows its qubit.
                AdjGate::H { q } | AdjGate::X { q } | AdjGate::Fixed { q, .. } => {
                    op.ma = local(q as usize);
                    op.mb = op.ma;
                }
                AdjGate::Rzz { a, b, slot } => {
                    op.kind = FuseKind::Zz;
                    op.ma = local(a.min(b) as usize);
                    op.mb = local(a.max(b) as usize);
                    op.ph = zz_phase(self.ang[k][0], true);
                    op.slot = slot;
                }
                AdjGate::Rxx { a, b, slot, yy } => {
                    let (c, kk) = xx_coef(self.ang[k][0], yy, true);
                    op.kind = FuseKind::Xx;
                    op.ma = local(a.min(b) as usize);
                    op.mb = local(a.max(b) as usize);
                    op.c = c;
                    op.kk = kk;
                    op.yy = yy;
                    op.slot = slot;
                }
                _ => unreachable!("fusable_run only spans fusable gates"),
            }
            ops[nops] = op;
            nops += 1;
        }

        // A block's addresses all agree above `h = qs[nq-1]`, so every block
        // sits inside one aligned window of `2^(h+1)` amplitudes.  Windows are
        // contiguous slices, which is what lets the sweep split across threads
        // with `par_chunks_mut` instead of hand-slicing by counter range (the
        // counter itself is *not* contiguous in memory).  The split is widened
        // to [`chunk_for`] so a two-qubit run does not hand Rayon one task per
        // four amplitudes.
        let win = 1usize << (qs[nq - 1] + 1);
        let chunk = chunk_for(self.dim, win);
        let parts: Vec<[Complex64; FUSE_RUN_MAX]> = if par_on(self.dim) && chunk < self.dim {
            self.phi
                .par_chunks_mut(chunk)
                .zip(self.psi.par_chunks_mut(chunk))
                .map(|(pc, sc)| fused_blocks(pc, sc, &ops[..nops], &spread[..nblk], nq, free))
                .collect()
        } else {
            vec![fused_blocks(
                &mut self.phi,
                &mut self.psi,
                &ops[..nops],
                &spread[..nblk],
                nq,
                free,
            )]
        };

        for dots in &parts {
            for (i, op) in ops[..nops].iter().enumerate() {
                match op.kind {
                    // Same coefficient the per-gate arm applies: `2·0.5·Im`
                    // for the rotation generators, plain `Im` for Rzz/Rxx.
                    FuseKind::Rot(_) => self.grad[op.slot as usize] += 2.0 * 0.5 * dots[i].im,
                    FuseKind::Zz | FuseKind::Xx => self.grad[op.slot as usize] += dots[i].im,
                    FuseKind::Const | FuseKind::Cx => {}
                }
            }
        }
    }

    /// Highest qubit a strip-hostable gate touches, or `None` for a gate a
    /// strip cannot host.
    ///
    /// `Cu`'s reverse step is four restricted dots plus a chain-rule mix, and
    /// `Mono`'s is an index map with its own kernel: neither is a (dot, apply)
    /// pair that a strip has anything to share between vectors.  Everything
    /// else -- including `Cx`, which is a permutation with no dot at all --
    /// costs one walk of both vectors and so belongs in a run.
    fn strip_gate_top(&self, k: usize) -> Option<usize> {
        match self.gates[k] {
            AdjGate::Ry { q, .. }
            | AdjGate::Rz { q, .. }
            | AdjGate::Rx { q, .. }
            | AdjGate::H { q }
            | AdjGate::X { q }
            | AdjGate::Fixed { q, .. } => Some(q as usize),
            AdjGate::Rzz { a, b, .. } | AdjGate::Rxx { a, b, .. } => {
                Some((a as usize).max(b as usize))
            }
            AdjGate::Cx { ctl, tgt } => Some((ctl as usize).max(tgt as usize)),
            _ => None,
        }
    }

    /// Longest run ending at (exclusive) `end` that one strip can host, and the
    /// strip width in bits.
    ///
    /// `limit` is the widest strip in elements ([`strip_limit`]); it is a
    /// parameter so the budget decision is testable without touching the
    /// process-wide thread width.
    ///
    /// Returned `bits` is 0 when even the last gate does not fit, which the
    /// sweep reads as "step back one gate".  The walk is backwards because the
    /// reverse sweep consumes the tape from its end, so what it takes is exactly
    /// the prefix of the remaining tape a single pass can cover: as soon as a
    /// gate needs a wider strip than the budget allows, the run stops and that
    /// gate goes out on the per-gate path.
    fn strip_run(&self, end: usize, limit: usize) -> (usize, usize) {
        let mut start = end;
        let mut bits = 0u32;
        while start > 0 && end - start < STRIP_RUN_MAX {
            let q = match self.strip_gate_top(start - 1) {
                Some(q) => q,
                None => break,
            };
            if (1usize << (q + 1)) > limit {
                break;
            }
            start -= 1;
            bits = bits.max(q as u32 + 1);
        }
        (start, bits as usize)
    }

    /// One gate of a run as a [`FuseOp`], with masks relative to the state
    /// rather than to a gathered block — a strip's low bits are the state's low
    /// bits, so the same struct and the same kernels serve both.  `dagger`
    /// selects the forward or the adjoint matrix and phase convention.
    fn strip_op(&self, k: usize, dagger: bool) -> FuseOp {
        let mut op = FuseOp {
            gate: if dagger { self.dg[k] } else { self.fwd[k] },
            ..FuseOp::empty()
        };
        match self.gates[k] {
            AdjGate::Ry { q, slot } => {
                op.kind = FuseKind::Rot(2);
                op.ma = 1usize << q;
                op.slot = slot;
            }
            AdjGate::Rz { q, slot } => {
                op.kind = FuseKind::Rot(3);
                op.ma = 1usize << q;
                op.slot = slot;
            }
            AdjGate::Rx { q, slot } => {
                op.kind = FuseKind::Rot(1);
                op.ma = 1usize << q;
                op.slot = slot;
            }
            AdjGate::H { q } | AdjGate::X { q } | AdjGate::Fixed { q, .. } => {
                op.ma = 1usize << q;
                op.mb = op.ma;
            }
            AdjGate::Rzz { a, b, slot } => {
                op.kind = FuseKind::Zz;
                op.ma = 1usize << a.min(b);
                op.mb = 1usize << a.max(b);
                op.ph = zz_phase(self.ang[k][0], dagger);
                op.slot = slot;
            }
            AdjGate::Rxx { a, b, slot, yy } => {
                let (c, kk) = xx_coef(self.ang[k][0], yy, dagger);
                op.kind = FuseKind::Xx;
                op.ma = 1usize << a.min(b);
                op.mb = 1usize << a.max(b);
                op.c = c;
                op.kk = kk;
                op.yy = yy;
                op.slot = slot;
            }
            AdjGate::Cx { ctl, tgt } => {
                op.kind = FuseKind::Cx;
                op.ma = 1usize << ctl;
                op.mb = 1usize << tgt;
            }
            _ => unreachable!("strip_run only spans strip-hostable gates"),
        }
        op
    }

    /// Reverse-apply gates `start..end` with one pass per strip.
    ///
    /// The run's gates are stepped in reverse tape order *inside* the strip, so
    /// every dot sees the state sequence the per-gate sweep gives it and the
    /// arithmetic per gate is the same arithmetic.  What changes is how often
    /// the two vectors are read and written: once for the whole run instead of
    /// once per gate.  The dots are summed per strip and then across strips,
    /// which is a different summation order from the per-gate reduction over
    /// `par_chunks_mut`; the gradient moves in its last bits and nowhere else
    /// (`strip_sweep_matches_per_gate_sweep` pins that).
    fn strip_backward(&mut self, start: usize, end: usize, bits: usize) {
        let s = 1usize << bits;
        let ops: Vec<FuseOp> = (start..end).rev().map(|k| self.strip_op(k, true)).collect();
        let parts: Vec<Vec<Complex64>> = if par_on(self.dim) && s < self.dim {
            self.phi
                .par_chunks_mut(s)
                .zip(self.psi.par_chunks_mut(s))
                .map(|(pc, sc)| strip_blocks(pc, sc, &ops))
                .collect()
        } else {
            vec![strip_blocks(&mut self.phi, &mut self.psi, &ops)]
        };
        for dots in &parts {
            for (i, op) in ops.iter().enumerate() {
                match op.kind {
                    FuseKind::Rot(_) => self.grad[op.slot as usize] += 2.0 * 0.5 * dots[i].im,
                    FuseKind::Zz | FuseKind::Xx => self.grad[op.slot as usize] += dots[i].im,
                    FuseKind::Const | FuseKind::Cx => {}
                }
            }
        }
    }

    /// Start of the diagonal run ending at (exclusive) `end`.
    ///
    /// Consecutive `Rz`/`Rzz` gates commute with each other -- they are all
    /// diagonal, so their order is immaterial -- and together they form one
    /// diagonal operator however widely their qubits spread.  That is the
    /// difference from [`AdjProgram::strip_run`] and [`AdjProgram::fusable_run`]:
    /// both of those need the run's *support* to fit a buffer, and this needs
    /// nothing, because a diagonal gate never couples two amplitudes.
    fn diag_run(&self, end: usize) -> usize {
        let mut start = end;
        while start > 0 && end - start < DIAG_RUN_MAX {
            match self.gates[start - 1] {
                AdjGate::Rz { .. } | AdjGate::Rzz { .. } => start -= 1,
                _ => break,
            }
        }
        start
    }

    /// End of the diagonal run starting at `start`.
    ///
    /// The forward-sweep twin of [`AdjProgram::diag_run`]: runs are walked the
    /// same way in both directions so a tape's diagonal blocks are recognised
    /// identically whichever sweep meets them.
    fn diag_run_from(&self, start: usize) -> usize {
        let mut end = start;
        while end < self.gates.len() && end - start < DIAG_RUN_MAX {
            match self.gates[end] {
                AdjGate::Rz { .. } | AdjGate::Rzz { .. } => end += 1,
                _ => break,
            }
        }
        end
    }

    /// [`DiagOp`] for gate `k`; only valid for `Rz`/`Rzz`.
    fn diag_op(&self, k: usize) -> DiagOp {
        let t = self.ang[k][0];
        match self.gates[k] {
            AdjGate::Rz { q, slot } => DiagOp {
                mask: 1usize << q,
                t2: [t, -t],
                slot,
            },
            AdjGate::Rzz { a, b, slot } => DiagOp {
                mask: (1usize << a) | (1usize << b),
                t2: [t, -t],
                slot,
            },
            _ => unreachable!("diag_run only spans Rz/Rzz"),
        }
    }

    /// Reverse a diagonal run: one traversal carries all of its dots.
    ///
    /// Both per-gate arms for these gates add `ip.im` to their slot, so the
    /// coefficient is uniform and no per-gate branch survives here.
    fn diag_backward(&mut self, start: usize, end: usize) {
        let ops: Vec<DiagOp> = (start..end).rev().map(|k| self.diag_op(k)).collect();
        let span = chunk_for(self.dim, 1);
        let parts: Vec<[Complex64; DIAG_RUN_MAX]> = if par_on(self.dim) && span < self.dim {
            self.phi
                .par_chunks_mut(span)
                .zip(self.psi.par_chunks_mut(span))
                .enumerate()
                .map(|(c, (pc, sc))| {
                    let mut dots = [Complex64::new(0.0, 0.0); DIAG_RUN_MAX];
                    diag_backward_blocks(pc, sc, c * span, &ops, &mut dots);
                    dots
                })
                .collect()
        } else {
            let mut dots = [Complex64::new(0.0, 0.0); DIAG_RUN_MAX];
            diag_backward_blocks(&mut self.phi, &mut self.psi, 0, &ops, &mut dots);
            vec![dots]
        };
        let mut total = [Complex64::new(0.0, 0.0); DIAG_RUN_MAX];
        for dots in &parts {
            for (j, d) in dots[..ops.len()].iter().enumerate() {
                total[j] += *d;
            }
        }
        for (j, op) in ops.iter().enumerate() {
            self.grad[op.slot as usize] += total[j].im;
        }
    }

    /// Apply a diagonal run to `psi` in one traversal (forward sweep).
    fn diag_forward(&mut self, start: usize, end: usize) {
        let ops: Vec<DiagOp> = (start..end).map(|k| self.diag_op(k)).collect();
        let span = chunk_for(self.dim, 1);
        if par_on(self.dim) && span < self.dim {
            self.psi
                .par_chunks_mut(span)
                .enumerate()
                .for_each(|(c, sl)| diag_forward_blocks(sl, c * span, &ops));
        } else {
            diag_forward_blocks(&mut self.psi, 0, &ops);
        }
    }

    /// One gate of the reverse sweep: generator dot into `grad`, then rewind
    /// both `phi` and `psi` by its dagger.
    fn step_backward(&mut self, k: usize) {
        match self.gates[k] {
            AdjGate::Ry { q, slot: s } => {
                let m = self.dg[k];
                let ip = par_dot_apply_pair(2, &mut self.phi, &mut self.psi, 1usize << q, m);
                self.grad[s as usize] += 2.0 * 0.5 * ip.im;
            }
            AdjGate::Rz { q, slot: s } => {
                let m = self.dg[k];
                let ip = par_dot_apply_pair(3, &mut self.phi, &mut self.psi, 1usize << q, m);
                self.grad[s as usize] += 2.0 * 0.5 * ip.im;
            }
            AdjGate::Rx { q, slot: s } => {
                let m = self.dg[k];
                let ip = par_dot_apply_pair(1, &mut self.phi, &mut self.psi, 1usize << q, m);
                self.grad[s as usize] += 2.0 * 0.5 * ip.im;
            }
            AdjGate::Rzz { a, b, slot: s } => {
                let (hm, lm) = (1usize << a.max(b), 1usize << a.min(b));
                let ip = par_dot_zz_apply_pair(
                    &mut self.phi,
                    &mut self.psi,
                    hm,
                    lm,
                    zz_phase(self.ang[k][0], true),
                );
                self.grad[s as usize] += ip.im;
            }
            AdjGate::Rxx { a, b, slot: s, yy } => {
                let (hm, lm) = (1usize << a.max(b), 1usize << a.min(b));
                let (c, kc) = xx_coef(self.ang[k][0], yy, true);
                let ip = par_dot_xx_apply_pair(&mut self.phi, &mut self.psi, hm, lm, c, kc, yy);
                self.grad[s as usize] += ip.im;
            }
            AdjGate::H { q } | AdjGate::X { q } | AdjGate::Fixed { q, .. } => {
                let m = self.dg[k];
                par_pair_pass(&mut self.phi, q as usize, m);
                par_pair_pass(&mut self.psi, q as usize, m);
            }
            AdjGate::Cu {
                ctl,
                tgt,
                slots,
                gscale,
                gref,
                ..
            } => {
                let (cm, tm) = (1usize << ctl, 1usize << tgt);
                let m = self.dg[k];
                if self.gates[k].is_param() {
                    let a = self.ang[k];
                    let (ax, ay, az, d) =
                        par_dot_xyz_pair_c1(&mut self.phi, &mut self.psi, cm, tm, m);
                    self.cu_grad(slots, gscale, gref, a, ax, ay, az, d);
                } else {
                    // Bound `Cu` (CZ, CY): no slot to feed, so skip the
                    // dot entirely and only rewind both vectors.
                    par_c1_apply_pair(&mut self.phi, &mut self.psi, ctl as usize, tgt as usize, m);
                }
            }
            AdjGate::Cx { ctl, tgt } => {
                par_cx_apply_pair(&mut self.phi, &mut self.psi, ctl as usize, tgt as usize);
            }
            AdjGate::Mono { kind, qs, k } => {
                // Constant: no generator dot to accumulate, so both
                // vectors just get the daggered map (once each).
                par_mono_apply(&mut self.phi, kind, qs, k as usize, true);
                par_mono_apply(&mut self.psi, kind, qs, k as usize, true);
            }
        }
    }

    /// Chain-rule accumulation for one `Cu` gate, shared by both sweeps.
    ///
    /// With `U = diag(I, e^{iγ}·U3(θ,φ,λ))` each derivative is `∂U = M·U` on
    /// the control-1 block, with the exact generators
    ///
    /// ```text
    /// θ -> (−i/2)(cosφ·Y − sinφ·X)                       (traceless)
    /// φ -> i/2·1 − i/2·Z
    /// λ -> i/2·1 − i/2(cosθ·Z + sinθ(cosφ·X + sinφ·Y))
    /// γ -> i·1
    /// ```
    ///
    /// and `∂⟨O⟩ = 2·Re⟨φ|M|ψ_after⟩`.  The `1` above is the identity *of the
    /// control-1 block*, not of the whole space, so its term `2Re(i/2·D) =
    /// −Im D` does not cancel here (`D = ⟨φ|ψ_after⟩` restricted to control 1);
    /// dropping it is a real error the moment the control is entangled.  Only
    /// fully traceless generators, like the θ one, lose their identity part.
    //
    // Nine scalars is the honest signature: one gate's four slot indices, the
    // reference-angle scale, and the five complex intermediates the sweeps
    // already hold. Bundling them into a struct would add a construction at
    // every call site in the inner sweep to satisfy a lint.
    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn cu_grad(
        &mut self,
        slots: [i32; 4],
        gscale: f64,
        gref: u8,
        a: [f64; 4],
        ax: Complex64,
        ay: Complex64,
        az: Complex64,
        d: Complex64,
    ) {
        let (ca, sa) = (a[0].cos(), a[0].sin());
        let (cb, sb) = (a[1].cos(), a[1].sin());
        let g_gam = -2.0 * d.im;
        if slots[0] >= 0 {
            self.grad[slots[0] as usize] += cb * ay.im - sb * ax.im;
        }
        if slots[1] >= 0 {
            self.grad[slots[1] as usize] += az.im - d.im;
        }
        if slots[2] >= 0 {
            self.grad[slots[2] as usize] += ca * az.im + sa * (cb * ax.im + sb * ay.im) - d.im;
        }
        if slots[3] >= 0 {
            self.grad[slots[3] as usize] += g_gam;
        }
        // γ tracks one of the other angles (controlled-RZ): that angle's
        // derivative picks up `gscale·∂γ` by the chain rule.
        let gr = gref as usize;
        if gscale != 0.0 && slots[gr] >= 0 {
            self.grad[slots[gr] as usize] += gscale * g_gam;
        }
    }

    /// Full gradient of ⟨O⟩ for `terms` at `theta`, in parameter-slot order.
    pub fn grad(&mut self, theta: &[f64], terms: &[PauliTerm]) -> Result<&[f64], String> {
        self.grad_with_fuse(theta, terms, self.fuse)
    }

    /// [`AdjProgram::grad`] with the reverse-sweep shape chosen explicitly, so
    /// the fused and per-gate sweeps can be compared on the same program.
    fn grad_with_fuse(
        &mut self,
        theta: &[f64],
        terms: &[PauliTerm],
        fuse: bool,
    ) -> Result<&[f64], String> {
        let strip = strip_enabled();
        let diag = diag_enabled();
        self.grad_with_sweeps(theta, terms, fuse, strip, diag)
    }

    /// [`AdjProgram::grad`] with every reverse-sweep shape chosen explicitly:
    /// `fuse` picks the 16-amplitude block fusion, `strip` the cache-strip
    /// fusion that the sweep takes when `fuse` is off.  Both off is the
    /// one-gate-at-a-time sweep.  Test-only since the diagonal-run sweep
    /// became the shipped default: `grad_with_fuse` now picks its flags from
    /// the two env knobs and hands them to [`AdjProgram::grad_with_sweeps`]
    /// directly, so this four-shape form survives as the A/B entry point the
    /// equivalence tests call.
    #[allow(dead_code)]
    fn grad_with_shapes(
        &mut self,
        theta: &[f64],
        terms: &[PauliTerm],
        fuse: bool,
        strip: bool,
    ) -> Result<&[f64], String> {
        self.grad_with_sweeps(theta, terms, fuse, strip, false)
    }

    /// [`AdjProgram::grad`] with the diagonal-run sweep also selectable.
    ///
    /// Kept as the one place the sweeps are dispatched so the three shapes
    /// cannot drift apart: `forward` and `backward_streaming` both take their
    /// flags from here.
    fn grad_with_sweeps(
        &mut self,
        theta: &[f64],
        terms: &[PauliTerm],
        fuse: bool,
        strip: bool,
        diag: bool,
    ) -> Result<&[f64], String> {
        if theta.len() != self.p {
            return Err(format!(
                "cat adj: theta has {} values, expected {}",
                theta.len(),
                self.p
            ));
        }
        let trace = trace_enabled();
        set_width(self.threads);
        let t0 = std::time::Instant::now();
        self.fill_matrices(theta);
        self.forward(diag);
        let t1 = std::time::Instant::now();
        self.build_phi(terms);
        let t2 = std::time::Instant::now();
        if self.streaming {
            self.backward_streaming(fuse, strip, diag);
        } else {
            self.backward();
        }
        if trace {
            let t3 = std::time::Instant::now();
            eprintln!(
                "cat-adj-trace n={} p={} gates={} streaming={} fwd={:.2}ms phi={:.2}ms bwd={:.2}ms",
                self.n,
                self.p,
                self.gates.len(),
                self.streaming,
                (t1 - t0).as_secs_f64() * 1e3,
                (t2 - t1).as_secs_f64() * 1e3,
                (t3 - t2).as_secs_f64() * 1e3,
            );
        }
        Ok(&self.grad)
    }

    /// Forward expectation ⟨ψ|O|ψ⟩ for `terms` at `theta` (chain-rule
    /// arbiter for finite-difference checks; not part of the gradient path).
    pub fn expectation(&mut self, theta: &[f64], terms: &[PauliTerm]) -> Result<f64, String> {
        if theta.len() != self.p {
            return Err(format!(
                "cat adj: theta has {} values, expected {}",
                theta.len(),
                self.p
            ));
        }
        self.fill_matrices(theta);
        set_width(self.threads);
        // The plain forward: this is the finite-difference arbiter, so it must
        // not move with the shape under test.
        self.forward(false);
        self.build_phi(terms);
        let mut acc = Complex64::new(0.0, 0.0);
        for i in 0..self.dim {
            acc += self.psi[i].conj() * self.phi[i];
        }
        Ok(acc.re)
    }
}

/// `SF_CAT_ADJ_STREAM=1` forces the streaming reverse sweep on plans small
/// enough to cache, so the two paths can be timed against each other.
fn force_streaming() -> bool {
    match std::env::var("SF_CAT_ADJ_STREAM") {
        Ok(v) => !v.is_empty() && v != "0",
        Err(_) => false,
    }
}

/// `SF_CAT_ADJ_CACHE=1` forces the cached reverse sweep, overriding
/// [`cache_is_worth_it`]; the other half of the A/B switch.
fn force_cached() -> bool {
    match std::env::var("SF_CAT_ADJ_CACHE") {
        Ok(v) => !v.is_empty() && v != "0",
        Err(_) => false,
    }
}

/// Cache footprint above which the post-state cache cannot stay in L3, so a
/// parameter gate pays a DRAM store in the forward sweep and a DRAM read in the
/// reverse one instead of two L3 hits.
const RESIDENT_CACHE_BYTES: usize = 4 << 20;

/// Whether storing a post-state per parameter gate beats re-walking `psi`.
///
/// The cache trades a store plus a load of one state per parameter gate for one
/// extra state pass per gate.  While the cache is L3-resident both sides are L3
/// traffic and the cache wins (it also keeps the reverse sweep branch-free),
/// but once it spills to DRAM the trade is ~0.70 ms per parameter gate against
/// ~0.33 ms per gate plus ~0.18 ms per parameter — measured at n = 16 on the
/// 7-lane probe, crossover at `n_param_gates / n_gates ~= 0.63`.  So: keep the
/// cache while it is resident, else stream when most gates are parameterised.
fn cache_is_worth_it(n_param_gates: usize, n_gates: usize, dim: usize) -> bool {
    let cache_bytes = n_param_gates
        .saturating_mul(dim)
        .saturating_mul(std::mem::size_of::<Complex64>());
    if cache_bytes <= RESIDENT_CACHE_BYTES {
        return true;
    }
    (n_param_gates as f64) < 0.6 * (n_gates as f64)
}

/// Phase timing on stderr when `SF_CAT_ADJ_TRACE=1` (same convention as
/// `SF_ADJ_TRACE` in sf-ir).  Off by default; costs one env lookup per call.
fn trace_enabled() -> bool {
    match std::env::var("SF_CAT_ADJ_TRACE") {
        Ok(v) => !v.is_empty() && v != "0",
        Err(_) => false,
    }
}

fn one_qubit(qubits: &[usize], name: &str) -> Result<u32, String> {
    if qubits.len() != 1 {
        return Err(format!("cat adj: {name} needs exactly one qubit"));
    }
    Ok(qubits[0] as u32)
}

fn two_qubit(qubits: &[usize], name: &str) -> Result<(u32, u32), String> {
    if qubits.len() != 2 {
        return Err(format!("cat adj: {name} needs exactly two qubits"));
    }
    Ok((qubits[0] as u32, qubits[1] as u32))
}

fn three_qubit(qubits: &[usize], name: &str) -> Result<(u32, u32, u32), String> {
    if qubits.len() != 3 {
        return Err(format!("cat adj: {name} needs exactly three qubits"));
    }
    Ok((qubits[0] as u32, qubits[1] as u32, qubits[2] as u32))
}

/// Push a 1q rotation: a live slot when the angle is a variable, a baked matrix
/// when it is already bound (bound angles used to be refused outright).
fn push_rot(
    gates: &mut Vec<AdjGate>,
    kind: Rot,
    q: u32,
    param: &Parameter,
    idx: &HashMap<&str, u32>,
) -> Result<(), String> {
    match angle_of(param, idx)? {
        Angle::Slot(slot) => gates.push(kind.gate(q, slot)),
        Angle::Const(v) => gates.push(AdjGate::Fixed { q, m: kind.mat(v) }),
    }
    Ok(())
}

/// Build a `Cu` gate from resolved angles; `Angle::Slot` becomes a live slot.
/// `gscale`/`gref` fold `γ ← γ + gscale·a[gref]`; pass `(0.0, 3)` to leave γ
/// alone (a self-reference with a zero scale is a no-op).
fn cu_gate(ctl: u32, tgt: u32, angles: [Angle; 4], gscale: f64, gref: u8) -> AdjGate {
    let mut slots = [-1i32; 4];
    let mut consts = [0.0f64; 4];
    for (i, a) in angles.iter().enumerate() {
        match *a {
            Angle::Slot(s) => slots[i] = s as i32,
            Angle::Const(v) => consts[i] = v,
        }
    }
    AdjGate::Cu {
        ctl,
        tgt,
        slots,
        consts,
        gscale,
        gref,
    }
}

fn m2(m: &[Vec<Complex64>]) -> Result<M2, String> {
    if m.len() != 2 || m[0].len() != 2 || m[1].len() != 2 {
        return Err("cat adj: expected a 2x2 unitary to bake".to_string());
    }
    Ok([[m[0][0], m[0][1]], [m[1][0], m[1][1]]])
}

fn var_slot(param: &Parameter, idx: &HashMap<&str, u32>) -> Result<u32, String> {
    match param {
        Parameter::Variable { name, .. } => idx
            .get(name.as_str())
            .copied()
            .ok_or_else(|| format!("cat adj: parameter '{name}' is not in dag.parameter_names()")),
        _ => Err("cat adj: only variable parameters are supported \
                  (found a constant or expression)"
            .to_string()),
    }
}

/// CNOT on the flat state: swap pairs (bit ctl set, bit tgt clear).
/// Exact index permutation (no arithmetic), identical to sf-ir's CNOT apply.
///
/// The mask test is hoisted to runs of `min(cm, tm)` exactly as in
/// [`c1_apply`]: the control bit is constant over such an aligned run, so this
/// is one branch per run instead of one per amplitude.  `Rzz` lowers to
/// `Cx·Rz·Cx`, so in an `Rzz` ladder CNOTs outnumber rotations two to one and
/// this loop is the hot one.
#[inline]
fn cx_apply(s: &mut [Complex64], dim: usize, ctl: usize, tgt: usize) {
    let cm = 1usize << ctl;
    let tm = 1usize << tgt;
    let run = cm.min(tm);
    let block = tm << 1;
    let mut base = 0;
    while base < dim {
        let mut k0 = 0;
        while k0 < tm {
            if (base + k0) & cm != 0 {
                for k in k0..k0 + run {
                    s.swap(base + k, base + k + tm);
                }
            }
            k0 += run;
        }
        base += block;
    }
}

/// Two-array CNOT: one index walk, both vectors swapped in place.
#[inline]
fn cx_apply_pair(a: &mut [Complex64], b: &mut [Complex64], dim: usize, ctl: usize, tgt: usize) {
    let cm = 1usize << ctl;
    let tm = 1usize << tgt;
    let run = cm.min(tm);
    let block = tm << 1;
    let mut base = 0;
    while base < dim {
        let mut k0 = 0;
        while k0 < tm {
            if (base + k0) & cm != 0 {
                for k in k0..k0 + run {
                    let i = base + k;
                    let j = i + tm;
                    a.swap(i, j);
                    b.swap(i, j);
                }
            }
            k0 += run;
        }
        base += block;
    }
}

/// The two basis patterns a monomial gate transposes, and whether the move
/// carries a factor of `i`.
///
/// Patterns are in the gate's own basis order, `|q0 q1 q2⟩` with the *first*
/// listed qubit most significant — the layout `OpType::to_matrix` uses for its
/// 8×8 block — so each entry below is that matrix read column by column.  All
/// four gates are a single transposition plus fixed points (`iSWAP` being the
/// same transposition with `i` on both directions), which is what keeps the
/// kernel down to two offsets and a swap.
#[inline]
fn mono_table(kind: Mono) -> (u8, u8, bool) {
    match kind {
        // |01> <-> |10>, no phase.
        Mono::Swap => (1, 2, false),
        // |01> -> i|10> and |10> -> i|01>.
        Mono::ISwap => (1, 2, true),
        // Two controls then the target: |110> <-> |111>, i.e. 6 <-> 7.
        Mono::Ccx => (6, 7, false),
        // Control, then the swapped pair: |101> <-> |110>, i.e. 5 <-> 6.
        Mono::Cswap => (5, 6, false),
    }
}

/// Apply a constant monomial gate ([`Mono`]) to a flat state: one traversal
/// instead of the three-to-fifteen gate chain the IR expands these to.
///
/// `start` walks the state a block at a time, the block being the span of the
/// highest involved qubit, and `u` sweeps inside a block so the mask test has
/// the same short, fixed pattern in every block — the offset arithmetic and
/// the phase test are hoisted out of the amplitude loop entirely, which is
/// what the three-CNOT chain it replaces could not avoid.
#[inline]
fn mono_apply(s: &mut [Complex64], kind: Mono, qs: [u32; 3], k: usize, dagger: bool) {
    let (p, q, phase) = mono_table(kind);
    // Pattern bit `k-1-j` is qubit `qs[j]`: the first listed qubit is the most
    // significant, matching how `OpType::to_matrix` orders a k-qubit block.
    let off = |pat: u8| {
        let mut o = 0usize;
        for j in 0..k {
            if (pat as usize >> (k - 1 - j)) & 1 == 1 {
                o |= 1usize << qs[j];
            }
        }
        o
    };
    let (op, oq) = (off(p), off(q));
    let mask = op | oq;
    let block = 1usize << (usize::BITS as usize - mask.leading_zeros() as usize);
    let dim = s.len();
    let mut start = 0usize;
    while start < dim {
        let mut u = 0usize;
        while u < block {
            if (u & mask) == 0 {
                let (i, j) = (start + u + op, start + u + oq);
                if phase {
                    // ×(±i) is a quarter turn of the two coordinates, not a
                    // general complex multiply.
                    let (a, b) = (s[i], s[j]);
                    let (na, nb) = if dagger {
                        (Complex64::new(b.im, -b.re), Complex64::new(a.im, -a.re))
                    } else {
                        (Complex64::new(-b.im, b.re), Complex64::new(-a.im, a.re))
                    };
                    s[i] = na;
                    s[j] = nb;
                } else {
                    s.swap(i, j);
                }
            }
            u += 1;
        }
        start += block;
    }
}

/// Apply a 2×2 block to the target qubit, active only when `ctl` is 1.
///
/// The control and target bits are both constant over runs of
/// `min(cm, tm)`, so the mask test is hoisted to that run: with `ctl > tgt` it
/// runs once per 2·tm block, and with `ctl < tgt` — the usual `(i, i+1)
/// ladder` — once per `cm` pairs.  Testing only the block base would silently
/// drop every second run in the second case.
#[inline]
fn c1_apply(s: &mut [Complex64], dim: usize, ctl: usize, tgt: usize, u: M2) {
    let cm = 1usize << ctl;
    let tm = 1usize << tgt;
    let run = cm.min(tm);
    let block = tm << 1;
    let mut base = 0;
    while base < dim {
        let mut k0 = 0;
        while k0 < tm {
            if (base + k0) & cm != 0 {
                for k in k0..k0 + run {
                    let lo = s[base + k];
                    let hi = s[base + k + tm];
                    s[base + k] = u[0][0] * lo + u[0][1] * hi;
                    s[base + k + tm] = u[1][0] * lo + u[1][1] * hi;
                }
            }
            k0 += run;
        }
        base += block;
    }
}

/// One traversal returning everything a `Cu` gradient needs, restricted to the
/// control-1 subspace: `⟨φ|X|src⟩`, `⟨φ|Y|src⟩`, `⟨φ|Z|src⟩` and `Σ conj(φ)·src`.
/// The three Pauli accumulations are term-for-term the `p = 1, 2, 3` branches of
/// [`dot_and_apply`], so the conventions cannot drift apart.  The control mask
/// is hoisted to runs of `min(cm, tm)` for the same reason as [`c1_apply`].
#[inline]
fn dot_xyz_c1(
    phi: &[Complex64],
    src: &[Complex64],
    cm: usize,
    tm: usize,
) -> (Complex64, Complex64, Complex64, Complex64) {
    let dim = phi.len();
    let run = cm.min(tm);
    let block = tm << 1;
    let zero = Complex64::new(0.0, 0.0);
    let (mut ax, mut ay, mut az, mut d) = (zero, zero, zero, zero);
    let mut base = 0;
    while base < dim {
        let mut k0 = 0;
        while k0 < tm {
            if (base + k0) & cm != 0 {
                for k in k0..k0 + run {
                    let lo = base + k;
                    let hi = lo + tm;
                    let (pl, ph) = (phi[lo], phi[hi]);
                    let (sl, sh) = (src[lo], src[hi]);
                    ax += pl.conj() * sh + ph.conj() * sl;
                    ay += pl.conj() * Complex64::new(sh.im, -sh.re)
                        + ph.conj() * Complex64::new(-sl.im, sl.re);
                    az += pl.conj() * sl - ph.conj() * sh;
                    d += pl.conj() * sl + ph.conj() * sh;
                }
            }
            k0 += run;
        }
        base += block;
    }
    (ax, ay, az, d)
}

/// Single-qubit Pauli application, port of sf-ir's `apply_pauli_1q_inplace`
/// (X permutes, Y rotates by ±i, Z flips the sign — all exact factors).
fn apply_pauli_1q(s: &mut [Complex64], pauli: u8, qubit: usize) {
    let dim = s.len();
    let mask = 1usize << qubit;
    match pauli {
        1 => {
            // X: swap contiguous runs of 2^qubit (pair form of swap_runs).
            let run = mask;
            let block = run << 1;
            for b in 0..dim / block {
                let base = b * block;
                for k in 0..run {
                    s.swap(base + k, base + k + run);
                }
            }
        }
        2 => {
            let neg_i = Complex64::new(0.0, -1.0);
            let pos_i = Complex64::new(0.0, 1.0);
            for i in 0..dim {
                if (i & mask) == 0 {
                    let j = i | mask;
                    let a = s[i];
                    let b = s[j];
                    s[i] = neg_i * b;
                    s[j] = pos_i * a;
                }
            }
        }
        3 => {
            for i in 0..dim {
                if (i & mask) != 0 {
                    s[i] = -s[i];
                }
            }
        }
        _ => {}
    }
}

/// Fused generator dot + daggered gate application on `phi`.
///
/// Equivalent to `dot_one_pauli(p, phi, src, m, dim)` followed by
/// `pair_pass(phi, m, gate)`, except both halves share one traversal: `phi` is
/// read once and written once instead of read twice, which saves a whole array
/// pass per parameterised gate.  The dot still runs against the pair values
/// *before* the update, and the generator cases match `dot_one_pauli` term for
/// term.
#[inline]
fn dot_and_apply(
    p: u8,
    phi: &mut [Complex64],
    src: &[Complex64],
    m: usize,
    gate: [[Complex64; 2]; 2],
) -> Complex64 {
    let dim = phi.len();
    let block = m << 1;
    let mut ip = Complex64::new(0.0, 0.0);
    let mut base = 0;
    while base < dim {
        for k in 0..m {
            let lo = phi[base + k];
            let hi = phi[base + k + m];
            match p {
                1 => {
                    // X: each run faces the opposite half of `src`.
                    ip += lo.conj() * src[base + k + m];
                    ip += hi.conj() * src[base + k];
                }
                2 => {
                    // Y: src rotated by -i on the upper run, +i on the lower.
                    let u = src[base + k + m];
                    let l = src[base + k];
                    ip += lo.conj() * Complex64::new(u.im, -u.re);
                    ip += hi.conj() * Complex64::new(-l.im, l.re);
                }
                3 => {
                    // Z: upper run subtracted (exact, same as adding -src).
                    ip += lo.conj() * src[base + k];
                    ip -= hi.conj() * src[base + k + m];
                }
                _ => {}
            }
            phi[base + k] = gate[0][0] * lo + gate[0][1] * hi;
            phi[base + k + m] = gate[1][0] * lo + gate[1][1] * hi;
        }
        base += block;
    }
    ip
}

/// `Rzz` phases as a two-entry table: `[bits agree, bits differ]`.
///
/// `exp(−iθ/2·Z⊗Z) = diag(e^{−iθ/2}, e^{+iθ/2}, e^{+iθ/2}, e^{−iθ/2})`, so the
/// phase depends only on the parity of the two bits.  `dagger` conjugates both
/// entries, which is what the reverse sweeps need.
#[inline]
fn zz_phase(t: f64, dagger: bool) -> [Complex64; 2] {
    let h = if dagger { 0.5 * t } else { -0.5 * t };
    [
        Complex64::new(h.cos(), h.sin()),
        Complex64::new(h.cos(), -h.sin()),
    ]
}

/// Multiply every amplitude by the `Rzz` phase for its parity.
///
/// `hm`/`lm` are the masks of the higher and lower qubit.  One pass over a
/// block of `2·hm`: the higher bit is fixed per half, the lower bit per run of
/// `lm`, so the inner loop is a contiguous `lm`-element scale.
#[inline]
fn zz_apply(s: &mut [Complex64], hm: usize, lm: usize, ph: [Complex64; 2]) {
    let dim = s.len();
    let block = hm << 1;
    let mut base = 0;
    while base < dim {
        for half in 0..2usize {
            let off = base + half * hm;
            let mut k0 = 0;
            while k0 < hm {
                // `half ^ bit` is the parity: 0 = agree, 1 = differ.
                let p = ph[half ^ ((k0 & lm) != 0) as usize];
                for x in &mut s[off + k0..off + k0 + lm] {
                    *x *= p;
                }
                k0 += lm;
            }
        }
        base += block;
    }
}

/// The 2×2 block of `exp(−iθ/2·P⊗P)` and its two off-diagonal coefficients.
///
/// `exp(−iθ/2·X⊗X) = cos(θ/2)·I − i·sin(θ/2)·X⊗X`, and `X⊗X` swaps the two
/// amplitudes of every pair `(i, i^mask)`, so the matrix is `[[c, k], [k, c]]`
/// with `k = −i·sin(θ/2)` on *both* pairs.  `Y⊗Y = −(Z⊗Z)(X⊗X)` is the same
/// operator with a minus sign on the pairs whose two bits agree, hence one sign
/// per pair and one variant for both gates.  Daggering is `θ → −θ`.
#[inline]
fn xx_coef(t: f64, yy: bool, dagger: bool) -> (f64, [Complex64; 2]) {
    let s = (0.5 * t).sin();
    let c = (0.5 * t).cos();
    let s = if dagger { -s } else { s };
    let k = Complex64::new(0.0, -s);
    let ka = if yy { -k } else { k };
    (c, [ka, k])
}

/// Apply `exp(−iθ/2·P⊗P)` in place: `[[c, k], [k, c]]` on every amplitude
/// pair `(i, i ^ mask)`.
///
/// `hm`/`lm` are the masks of the higher and lower qubit.  Inside a block of
/// `2·hm` the partner of `(half, x)` is `(1−half, x ^ lm)`, and the parity of
/// the pair — which picks `k` — is `half` itself.  The pairs are contiguous
/// runs of `lm`, so the inner loop stays a streaming read of two slices.
#[inline]
fn xx_apply(s: &mut [Complex64], hm: usize, lm: usize, c: f64, k: [Complex64; 2]) {
    let dim = s.len();
    let block = hm << 1;
    let mut base = 0;
    while base < dim {
        for half in 0..2usize {
            let kk = k[half];
            let off = base + half * hm;
            let off2 = base + (1 - half) * hm;
            let mut x = 0;
            while x < hm {
                for t in 0..lm {
                    let i = off + x + t;
                    let j = off2 + x + lm + t;
                    let (a, b) = (s[i], s[j]);
                    s[i] = a * c + b * kk;
                    s[j] = a * kk + b * c;
                }
                x += lm << 1;
            }
        }
        base += block;
    }
}

/// [`xx_apply`] on `phi`, fused with the `⟨φ|P|src⟩` dot the gradient needs.
/// The dot reads `src` before the daggered update, as in [`dot_zz_apply`].
#[inline]
fn dot_xx_apply(
    phi: &mut [Complex64],
    src: &[Complex64],
    hm: usize,
    lm: usize,
    c: f64,
    k: [Complex64; 2],
    yy: bool,
) -> Complex64 {
    let dim = phi.len();
    let block = hm << 1;
    let mut ip = Complex64::new(0.0, 0.0);
    let mut base = 0;
    while base < dim {
        for half in 0..2usize {
            let kk = k[half];
            // X⊗X is +1 on both pairs; Y⊗Y = −(Z⊗Z)(X⊗X) flips the agree one.
            let ps = if yy && half == 0 { -1.0 } else { 1.0 };
            let off = base + half * hm;
            let off2 = base + (1 - half) * hm;
            let mut x = 0;
            while x < hm {
                for t in 0..lm {
                    let i = off + x + t;
                    let j = off2 + x + lm + t;
                    let (a, b) = (phi[i], phi[j]);
                    ip += ps * (a.conj() * src[j] + b.conj() * src[i]);
                    phi[i] = a * c + b * kk;
                    phi[j] = a * kk + b * c;
                }
                x += lm << 1;
            }
        }
        base += block;
    }
    ip
}

/// Streaming variant: dot plus the daggered block applied to `phi` *and*
/// `src` in one traversal, mirroring [`dot_zz_apply_pair`].
#[inline]
fn dot_xx_apply_pair(
    phi: &mut [Complex64],
    src: &mut [Complex64],
    hm: usize,
    lm: usize,
    c: f64,
    k: [Complex64; 2],
    yy: bool,
) -> Complex64 {
    let dim = phi.len();
    let block = hm << 1;
    let mut ip = Complex64::new(0.0, 0.0);
    let mut base = 0;
    while base < dim {
        for half in 0..2usize {
            let kk = k[half];
            let ps = if yy && half == 0 { -1.0 } else { 1.0 };
            let off = base + half * hm;
            let off2 = base + (1 - half) * hm;
            let mut x = 0;
            while x < hm {
                for t in 0..lm {
                    let i = off + x + t;
                    let j = off2 + x + lm + t;
                    let (a, b) = (phi[i], phi[j]);
                    let (sa, sb) = (src[i], src[j]);
                    ip += ps * (a.conj() * sb + b.conj() * sa);
                    phi[i] = a * c + b * kk;
                    phi[j] = a * kk + b * c;
                    src[i] = sa * c + sb * kk;
                    src[j] = sa * kk + sb * c;
                }
                x += lm << 1;
            }
        }
        base += block;
    }
    ip
}

/// [`zz_apply`] on `phi`, fused with the `⟨φ|Z⊗Z|src⟩` dot the gradient needs.
/// Same argument order and structure as [`dot_and_apply`]: the dot is taken
/// against the values *before* the update.
#[inline]
fn dot_zz_apply(
    phi: &mut [Complex64],
    src: &[Complex64],
    hm: usize,
    lm: usize,
    ph: [Complex64; 2],
) -> Complex64 {
    let dim = phi.len();
    let block = hm << 1;
    let mut ip = Complex64::new(0.0, 0.0);
    let mut base = 0;
    while base < dim {
        for half in 0..2usize {
            let off = base + half * hm;
            let mut k0 = 0;
            while k0 < hm {
                let bit = ((k0 & lm) != 0) as usize;
                let idx = half ^ bit;
                let p = ph[idx];
                for k in k0..k0 + lm {
                    let i = off + k;
                    let a = phi[i];
                    let d = a.conj() * src[i];
                    // Z⊗Z is +1 on even parity, −1 on odd.
                    ip += if idx == 0 { d } else { -d };
                    phi[i] = a * p;
                }
                k0 += lm;
            }
        }
        base += block;
    }
    ip
}

/// Streaming variant: dot plus the daggered `Rzz` applied to `phi` *and* `src`
/// in one traversal, mirroring [`dot_apply_pair`].
#[inline]
fn dot_zz_apply_pair(
    phi: &mut [Complex64],
    src: &mut [Complex64],
    hm: usize,
    lm: usize,
    ph: [Complex64; 2],
) -> Complex64 {
    let dim = phi.len();
    let block = hm << 1;
    let mut ip = Complex64::new(0.0, 0.0);
    let mut base = 0;
    while base < dim {
        for half in 0..2usize {
            let off = base + half * hm;
            let mut k0 = 0;
            while k0 < hm {
                let bit = ((k0 & lm) != 0) as usize;
                let idx = half ^ bit;
                let p = ph[idx];
                for k in k0..k0 + lm {
                    let i = off + k;
                    let a = phi[i];
                    let b = src[i];
                    let d = a.conj() * b;
                    ip += if idx == 0 { d } else { -d };
                    phi[i] = a * p;
                    src[i] = b * p;
                }
                k0 += lm;
            }
        }
        base += block;
    }
    ip
}

/// Streaming variant of [`dot_and_apply`]: one traversal that accumulates the
/// generator dot *and* applies `gate` to `phi` *and* to `src`.
///
/// [`AdjProgram::backward_streaming`] has no cache, so it has to rewind `src`
/// gate by gate.  Splitting that from the dot kernel reads `src` twice and
/// writes it once per gate; folding both halves into one traversal reads each
/// array exactly once, and at n >= 20 a single state (16 MB) no longer fits in
/// L3, so this is a memory-traffic win, not just a loop-overhead one.  The dot
/// is taken against the pair values *before* either update, and the four
/// generator branches match [`dot_and_apply`] term for term.
#[inline]
fn dot_apply_pair(
    p: u8,
    phi: &mut [Complex64],
    src: &mut [Complex64],
    m: usize,
    gate: M2,
) -> Complex64 {
    let dim = phi.len();
    let block = m << 1;
    let mut ip = Complex64::new(0.0, 0.0);
    let mut base = 0;
    while base < dim {
        for k in 0..m {
            let lo = phi[base + k];
            let hi = phi[base + k + m];
            let sl = src[base + k];
            let sh = src[base + k + m];
            match p {
                1 => {
                    ip += lo.conj() * sh;
                    ip += hi.conj() * sl;
                }
                2 => {
                    ip += lo.conj() * Complex64::new(sh.im, -sh.re);
                    ip += hi.conj() * Complex64::new(-sl.im, sl.re);
                }
                3 => {
                    ip += lo.conj() * sl;
                    ip -= hi.conj() * sh;
                }
                _ => {}
            }
            phi[base + k] = gate[0][0] * lo + gate[0][1] * hi;
            phi[base + k + m] = gate[1][0] * lo + gate[1][1] * hi;
            src[base + k] = gate[0][0] * sl + gate[0][1] * sh;
            src[base + k + m] = gate[1][0] * sl + gate[1][1] * sh;
        }
        base += block;
    }
    ip
}

/// Streaming variant of [`dot_xyz_c1`] + two [`c1_apply`] calls: the same four
/// restricted accumulations and the same control-run loop, with the daggered
/// 2×2 block written to `phi` and to `src` in the same traversal.  A `Cu` gate
/// is the most expensive step in the streaming sweep (four arrays-pass
/// equivalents become two), so this is where the n = 20 margin comes from.
#[inline]
fn dot_xyz_pair_c1(
    phi: &mut [Complex64],
    src: &mut [Complex64],
    cm: usize,
    tm: usize,
    u: M2,
) -> (Complex64, Complex64, Complex64, Complex64) {
    let dim = phi.len();
    let run = cm.min(tm);
    let block = tm << 1;
    let zero = Complex64::new(0.0, 0.0);
    let (mut ax, mut ay, mut az, mut d) = (zero, zero, zero, zero);
    let mut base = 0;
    while base < dim {
        let mut k0 = 0;
        while k0 < tm {
            if (base + k0) & cm != 0 {
                for k in k0..k0 + run {
                    let lo = base + k;
                    let hi = lo + tm;
                    let (pl, ph) = (phi[lo], phi[hi]);
                    let (sl, sh) = (src[lo], src[hi]);
                    ax += pl.conj() * sh + ph.conj() * sl;
                    ay += pl.conj() * Complex64::new(sh.im, -sh.re)
                        + ph.conj() * Complex64::new(-sl.im, sl.re);
                    az += pl.conj() * sl - ph.conj() * sh;
                    d += pl.conj() * sl + ph.conj() * sh;
                    phi[lo] = u[0][0] * pl + u[0][1] * ph;
                    phi[hi] = u[1][0] * pl + u[1][1] * ph;
                    src[lo] = u[0][0] * sl + u[0][1] * sh;
                    src[hi] = u[1][0] * sl + u[1][1] * sh;
                }
            }
            k0 += run;
        }
        base += block;
    }
    (ax, ay, az, d)
}

/// Two-array [`c1_apply`] and [`cx_apply`]: same index walk, one traversal for
/// both vectors, so a bound-`Cu` or a CNOT in the streaming sweep does not pay
/// the mask arithmetic twice.
#[inline]
fn c1_apply_pair(a: &mut [Complex64], b: &mut [Complex64], ctl: usize, tgt: usize, u: M2) {
    let dim = a.len();
    let cm = 1usize << ctl;
    let tm = 1usize << tgt;
    let run = cm.min(tm);
    let block = tm << 1;
    let mut base = 0;
    while base < dim {
        let mut k0 = 0;
        while k0 < tm {
            if (base + k0) & cm != 0 {
                for k in k0..k0 + run {
                    let lo = base + k;
                    let hi = lo + tm;
                    let (al, ah) = (a[lo], a[hi]);
                    let (bl, bh) = (b[lo], b[hi]);
                    a[lo] = u[0][0] * al + u[0][1] * ah;
                    a[hi] = u[1][0] * al + u[1][1] * ah;
                    b[lo] = u[0][0] * bl + u[0][1] * bh;
                    b[hi] = u[1][0] * bl + u[1][1] * bh;
                }
            }
            k0 += run;
        }
        base += block;
    }
}

/// ⟨φ|G|src⟩ for one single-qubit generator pauli (1=X, 2=Y, 3=Z) with qubit
/// mask `m` — verbatim port of sf-ir's scalar `dot_one_pauli` loop structure
/// and accumulation order (`lo` is always 0 / `hi` is the full dimension here).
/// The split form is kept as the reference kernel: `dot_and_apply` folds it
/// into the phi update, and `fused_dot_and_apply_matches_split_kernels` pins
/// the two against each other.
#[allow(dead_code)]
#[inline]
fn dot_one_pauli(p: u8, phi: &[Complex64], src: &[Complex64], m: usize, dim: usize) -> Complex64 {
    let block = m << 1;
    let mut ip = Complex64::new(0.0, 0.0);
    let mut base = 0;
    while base < dim {
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

// ---------------------------------------------------------------------------
// Optional multi-thread sweeps
// ---------------------------------------------------------------------------
//
// Every gate kernel above walks the state in blocks of `2^(highest involved
// qubit + 1)` — the span in which all the affected bits are the low bits of
// the local index.  That makes a sweep embarrassingly parallel over blocks:
// each thread owns one contiguous span of the state, reads and writes only
// that span, and no amplitude moves between spans.  The only quantity that
// crosses a span boundary is a generator dot (`⟨φ|P|ψ⟩`), so those are
// collected per chunk and summed in chunk order, which keeps a given width
// deterministic.  A threaded run therefore differs from a serial one only in
// the last bits of the gradient, never in the gate arithmetic.
//
// Opt-in: with `SF_CAT_THREADS` / `SF_THREADS` unset the count is 1 and every
// helper below falls through to the serial kernel, so default results stay
// bit-identical to the single-threaded engine (`0` means machine width).

/// Construction-time thread width from the environment (`0` = machine width).
fn env_width() -> usize {
    let raw = std::env::var("SF_CAT_THREADS")
        .ok()
        .or_else(|| std::env::var("SF_THREADS").ok());
    match raw.as_deref().map(str::trim) {
        None | Some("") => 1,
        Some(s) => match s.parse::<usize>() {
            Ok(0) => std::thread::available_parallelism().map_or(1, |n| n.get()),
            Ok(n) => n.clamp(1, 64),
            Err(_) => 1,
        },
    }
}

/// Width the sweeps use right now.  `grad` / `expectation` set it from the
/// handle's own field, so two handles with different widths can be interleaved
/// in one process — which is the only way to A/B a width on a host whose
/// absolute times drift between processes.
static WIDTH: AtomicUsize = AtomicUsize::new(1);

#[inline]
fn adj_threads() -> usize {
    WIDTH.load(Ordering::Relaxed).max(1)
}

#[inline]
fn set_width(w: usize) {
    WIDTH.store(w.max(1), Ordering::Relaxed);
}

/// One dedicated pool per requested width, so a user-set width never reshapes
/// sf-ir's global pool and repeated calls reuse the same threads.
fn pool_for(w: usize) -> Option<Arc<rayon::ThreadPool>> {
    if w <= 1 {
        return None;
    }
    /// The `(width, pool)` pairs built so far; one pool per requested width.
    type Pools = Mutex<Vec<(usize, Arc<rayon::ThreadPool>)>>;
    static POOLS: OnceLock<Pools> = OnceLock::new();
    let pools = POOLS.get_or_init(|| Mutex::new(Vec::new()));
    let mut guard = pools.lock().ok()?;
    if let Some((_, p)) = guard.iter().find(|(n, _)| *n == w) {
        return Some(Arc::clone(p));
    }
    let built = rayon::ThreadPoolBuilder::new()
        .num_threads(w)
        .thread_name(|i| format!("sf-cat-adj-{i}"))
        .build()
        .ok()?;
    let p = Arc::new(built);
    guard.push((w, Arc::clone(&p)));
    Some(p)
}

/// Pool for the current width (`None` = run the serial kernels inline).
fn adj_pool() -> Option<Arc<rayon::ThreadPool>> {
    pool_for(adj_threads())
}

/// Smallest state (elements) for which chunking pays; env `SF_CAT_PAR_MIN`.
fn par_min() -> usize {
    static M: OnceLock<usize> = OnceLock::new();
    *M.get_or_init(|| {
        std::env::var("SF_CAT_PAR_MIN")
            .ok()
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(1 << 14)
    })
}

#[inline]
fn par_on(len: usize) -> bool {
    adj_threads() > 1 && len >= par_min()
}

/// Per-thread slice width: one thread's worth of state, rounded to a whole
/// number of blocks (`span`).  Splitting on `span` alone would hand Rayon
/// 32 768 jobs of two amplitudes for a gate on qubit 0, so the split is always
/// `len / threads` wide and `span` is only the alignment it must respect.
#[inline]
fn chunk_for(len: usize, span: usize) -> usize {
    let target = (len / adj_threads().max(1)).max(span);
    (span * (target / span).max(1)).min(len)
}

#[inline]
fn par_pair_pass(s: &mut [Complex64], q: usize, m: M2) {
    let span = chunk_for(s.len(), 1usize << (q + 1));
    if par_on(s.len()) && span < s.len() {
        // `pair_pass_chunk` is the variant meant for pre-split slices: it skips
        // the small-slice scalar fallback, which a chunk would otherwise hit.
        s.par_chunks_mut(span)
            .for_each(|c| pair_pass_chunk(c, 1usize << q, m));
    } else {
        pair_pass(s, 1usize << q, m);
    }
}

#[inline]
fn par_c1_apply(s: &mut [Complex64], ctl: usize, tgt: usize, u: M2) {
    let span = chunk_for(s.len(), 1usize << (ctl.max(tgt) + 1));
    if par_on(s.len()) && span < s.len() {
        s.par_chunks_mut(span).for_each(|c| {
            let d = c.len();
            c1_apply(c, d, ctl, tgt, u)
        });
    } else {
        let d = s.len();
        c1_apply(s, d, ctl, tgt, u);
    }
}

#[inline]
fn par_cx_apply(s: &mut [Complex64], ctl: usize, tgt: usize) {
    let span = chunk_for(s.len(), 1usize << (ctl.max(tgt) + 1));
    if par_on(s.len()) && span < s.len() {
        s.par_chunks_mut(span).for_each(|c| {
            let d = c.len();
            cx_apply(c, d, ctl, tgt)
        });
    } else {
        let d = s.len();
        cx_apply(s, d, ctl, tgt);
    }
}

#[inline]
fn par_zz_apply(s: &mut [Complex64], hm: usize, lm: usize, ph: [Complex64; 2]) {
    let span = chunk_for(s.len(), hm << 1);
    if par_on(s.len()) && span < s.len() {
        s.par_chunks_mut(span).for_each(|c| zz_apply(c, hm, lm, ph));
    } else {
        zz_apply(s, hm, lm, ph);
    }
}

#[inline]
fn par_xx_apply(s: &mut [Complex64], hm: usize, lm: usize, c: f64, k: [Complex64; 2]) {
    let span = chunk_for(s.len(), hm << 1);
    if par_on(s.len()) && span < s.len() {
        s.par_chunks_mut(span)
            .for_each(|ch| xx_apply(ch, hm, lm, c, k));
    } else {
        xx_apply(s, hm, lm, c, k);
    }
}

#[inline]
fn par_mono_apply(s: &mut [Complex64], kind: Mono, qs: [u32; 3], k: usize, dagger: bool) {
    let hi = (0..k).map(|j| qs[j] as usize).max().unwrap_or(0);
    let span = chunk_for(s.len(), 1usize << (hi + 1));
    if par_on(s.len()) && span < s.len() {
        s.par_chunks_mut(span)
            .for_each(|c| mono_apply(c, kind, qs, k, dagger));
    } else {
        mono_apply(s, kind, qs, k, dagger);
    }
}

#[inline]
fn par_apply_pauli_1q(s: &mut [Complex64], pauli: u8, qubit: usize) {
    let span = chunk_for(s.len(), 1usize << (qubit + 1));
    if par_on(s.len()) && span < s.len() {
        s.par_chunks_mut(span)
            .for_each(|c| apply_pauli_1q(c, pauli, qubit));
    } else {
        apply_pauli_1q(s, pauli, qubit);
    }
}

/// `phi += coef * (±1) * psi[i ^ xmask]` with the sign from `zmask`.
/// `span` covers every bit of `xmask`, so `i ^ xmask` stays inside the chunk.
#[inline]
fn par_phi_add_xz(
    phi: &mut [Complex64],
    psi: &[Complex64],
    coef: Complex64,
    xmask: usize,
    zmask: usize,
) {
    let span = chunk_for(
        phi.len(),
        if xmask == 0 {
            1
        } else {
            1usize << (usize::BITS as usize - xmask.leading_zeros() as usize)
        },
    );
    if par_on(phi.len()) && span < phi.len() {
        phi.par_chunks_mut(span)
            .enumerate()
            .zip(psi.par_chunks(span))
            .for_each(|((ci, pc), ps)| {
                let base = ci * span;
                for (i, x) in pc.iter_mut().enumerate() {
                    let g = base + i;
                    let neg = zmask != 0 && ((g & zmask).count_ones() & 1) == 1;
                    *x += if neg { -coef } else { coef } * ps[i ^ xmask];
                }
            });
    } else {
        for i in 0..phi.len() {
            let neg = zmask != 0 && ((i & zmask).count_ones() & 1) == 1;
            phi[i] += if neg { -coef } else { coef } * psi[i ^ xmask];
        }
    }
}

#[inline]
fn par_add_scaled(dst: &mut [Complex64], src: &[Complex64], coef: Complex64) {
    if par_on(dst.len()) {
        dst.par_chunks_mut(1usize << 14)
            .zip(src.par_chunks(1usize << 14))
            .for_each(|(dc, sc)| {
                for (a, b) in dc.iter_mut().zip(sc) {
                    *a += coef * *b;
                }
            });
    } else {
        for (a, b) in dst.iter_mut().zip(src) {
            *a += coef * *b;
        }
    }
}

#[inline]
fn par_dot_and_apply(
    p: u8,
    phi: &mut [Complex64],
    src: &[Complex64],
    m: usize,
    gate: M2,
) -> Complex64 {
    let span = chunk_for(phi.len(), m << 1);
    if par_on(phi.len()) && span < phi.len() {
        let parts: Vec<Complex64> = phi
            .par_chunks_mut(span)
            .zip(src.par_chunks(span))
            .map(|(pc, sc)| dot_and_apply(p, pc, sc, m, gate))
            .collect();
        parts
            .into_iter()
            .fold(Complex64::new(0.0, 0.0), |a, b| a + b)
    } else {
        dot_and_apply(p, phi, src, m, gate)
    }
}

#[inline]
fn par_dot_zz_apply(
    phi: &mut [Complex64],
    src: &[Complex64],
    hm: usize,
    lm: usize,
    ph: [Complex64; 2],
) -> Complex64 {
    let span = chunk_for(phi.len(), hm << 1);
    if par_on(phi.len()) && span < phi.len() {
        let parts: Vec<Complex64> = phi
            .par_chunks_mut(span)
            .zip(src.par_chunks(span))
            .map(|(pc, sc)| dot_zz_apply(pc, sc, hm, lm, ph))
            .collect();
        parts
            .into_iter()
            .fold(Complex64::new(0.0, 0.0), |a, b| a + b)
    } else {
        dot_zz_apply(phi, src, hm, lm, ph)
    }
}

#[inline]
fn par_dot_xx_apply(
    phi: &mut [Complex64],
    src: &[Complex64],
    hm: usize,
    lm: usize,
    c: f64,
    k: [Complex64; 2],
    yy: bool,
) -> Complex64 {
    let span = chunk_for(phi.len(), hm << 1);
    if par_on(phi.len()) && span < phi.len() {
        let parts: Vec<Complex64> = phi
            .par_chunks_mut(span)
            .zip(src.par_chunks(span))
            .map(|(pc, sc)| dot_xx_apply(pc, sc, hm, lm, c, k, yy))
            .collect();
        parts
            .into_iter()
            .fold(Complex64::new(0.0, 0.0), |a, b| a + b)
    } else {
        dot_xx_apply(phi, src, hm, lm, c, k, yy)
    }
}

#[inline]
fn par_dot_xyz_c1(
    phi: &[Complex64],
    src: &[Complex64],
    cm: usize,
    tm: usize,
) -> (Complex64, Complex64, Complex64, Complex64) {
    let span = chunk_for(phi.len(), cm.max(tm) << 1);
    if par_on(phi.len()) && span < phi.len() {
        let parts: Vec<_> = phi
            .par_chunks(span)
            .zip(src.par_chunks(span))
            .map(|(pc, sc)| dot_xyz_c1(pc, sc, cm, tm))
            .collect();
        let mut acc = (
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        );
        for (ax, ay, az, d) in parts {
            acc.0 += ax;
            acc.1 += ay;
            acc.2 += az;
            acc.3 += d;
        }
        acc
    } else {
        dot_xyz_c1(phi, src, cm, tm)
    }
}

#[inline]
fn par_dot_apply_pair(
    p: u8,
    phi: &mut [Complex64],
    src: &mut [Complex64],
    m: usize,
    gate: M2,
) -> Complex64 {
    let span = chunk_for(phi.len(), m << 1);
    if par_on(phi.len()) && span < phi.len() {
        let parts: Vec<Complex64> = phi
            .par_chunks_mut(span)
            .zip(src.par_chunks_mut(span))
            .map(|(pc, sc)| dot_apply_pair(p, pc, sc, m, gate))
            .collect();
        parts
            .into_iter()
            .fold(Complex64::new(0.0, 0.0), |a, b| a + b)
    } else {
        dot_apply_pair(p, phi, src, m, gate)
    }
}

#[inline]
fn par_dot_zz_apply_pair(
    phi: &mut [Complex64],
    src: &mut [Complex64],
    hm: usize,
    lm: usize,
    ph: [Complex64; 2],
) -> Complex64 {
    let span = chunk_for(phi.len(), hm << 1);
    if par_on(phi.len()) && span < phi.len() {
        let parts: Vec<Complex64> = phi
            .par_chunks_mut(span)
            .zip(src.par_chunks_mut(span))
            .map(|(pc, sc)| dot_zz_apply_pair(pc, sc, hm, lm, ph))
            .collect();
        parts
            .into_iter()
            .fold(Complex64::new(0.0, 0.0), |a, b| a + b)
    } else {
        dot_zz_apply_pair(phi, src, hm, lm, ph)
    }
}

#[inline]
fn par_dot_xx_apply_pair(
    phi: &mut [Complex64],
    src: &mut [Complex64],
    hm: usize,
    lm: usize,
    c: f64,
    k: [Complex64; 2],
    yy: bool,
) -> Complex64 {
    let span = chunk_for(phi.len(), hm << 1);
    if par_on(phi.len()) && span < phi.len() {
        let parts: Vec<Complex64> = phi
            .par_chunks_mut(span)
            .zip(src.par_chunks_mut(span))
            .map(|(pc, sc)| dot_xx_apply_pair(pc, sc, hm, lm, c, k, yy))
            .collect();
        parts
            .into_iter()
            .fold(Complex64::new(0.0, 0.0), |a, b| a + b)
    } else {
        dot_xx_apply_pair(phi, src, hm, lm, c, k, yy)
    }
}

#[inline]
fn par_dot_xyz_pair_c1(
    phi: &mut [Complex64],
    src: &mut [Complex64],
    cm: usize,
    tm: usize,
    u: M2,
) -> (Complex64, Complex64, Complex64, Complex64) {
    let span = chunk_for(phi.len(), cm.max(tm) << 1);
    if par_on(phi.len()) && span < phi.len() {
        let parts: Vec<_> = phi
            .par_chunks_mut(span)
            .zip(src.par_chunks_mut(span))
            .map(|(pc, sc)| dot_xyz_pair_c1(pc, sc, cm, tm, u))
            .collect();
        let mut acc = (
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        );
        for (ax, ay, az, d) in parts {
            acc.0 += ax;
            acc.1 += ay;
            acc.2 += az;
            acc.3 += d;
        }
        acc
    } else {
        dot_xyz_pair_c1(phi, src, cm, tm, u)
    }
}

#[inline]
fn par_c1_apply_pair(a: &mut [Complex64], b: &mut [Complex64], ctl: usize, tgt: usize, u: M2) {
    let span = chunk_for(a.len(), 1usize << (ctl.max(tgt) + 1));
    if par_on(a.len()) && span < a.len() {
        a.par_chunks_mut(span)
            .zip(b.par_chunks_mut(span))
            .for_each(|(ac, bc)| c1_apply_pair(ac, bc, ctl, tgt, u));
    } else {
        c1_apply_pair(a, b, ctl, tgt, u);
    }
}

#[inline]
fn par_cx_apply_pair(a: &mut [Complex64], b: &mut [Complex64], ctl: usize, tgt: usize) {
    let span = chunk_for(a.len(), 1usize << (ctl.max(tgt) + 1));
    if par_on(a.len()) && span < a.len() {
        a.par_chunks_mut(span)
            .zip(b.par_chunks_mut(span))
            .for_each(|(ac, bc)| {
                let d = ac.len();
                cx_apply_pair(ac, bc, d, ctl, tgt)
            });
    } else {
        let d = a.len();
        cx_apply_pair(a, b, d, ctl, tgt);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sf_ir::ops::Parameter;

    fn var(name: &str) -> Parameter {
        Parameter::Variable {
            name: name.into(),
            id: 0,
        }
    }

    /// Build the ladder DAG with parameter names t0..tP-1 (worker order).
    fn ladder_dag(n: usize, layers: usize) -> (QuantumDAG, usize) {
        let mut dag = QuantumDAG::new(n, 0);
        let mut slot = 0usize;
        for _ in 0..layers {
            for i in 0..n {
                dag.add_op(OpType::Ry(var(&format!("t{slot}"))), &[i]);
                slot += 1;
                dag.add_op(OpType::Rz(var(&format!("t{slot}"))), &[i]);
                slot += 1;
            }
            for i in 0..n.saturating_sub(1) {
                dag.add_op(OpType::CNOT, &[i, i + 1]);
            }
        }
        (dag, slot)
    }

    fn z_terms(n: usize) -> Vec<PauliTerm> {
        vec![PauliTerm {
            paulis: vec![3u8; n],
            coef: Complex64::new(1.0, 0.0),
        }]
    }

    fn theta(p: usize) -> Vec<f64> {
        (0..p).map(|i| 0.15 + 0.05 * ((i % 7) as f64)).collect()
    }

    #[test]
    fn cat_adj_matches_sf_ir_house_engine() {
        let (n, layers) = (6, 2);
        let (dag, p) = ladder_dag(n, layers);
        assert_eq!(p, 2 * n * layers);
        let terms = z_terms(n);
        let th = theta(p);

        let values: HashMap<String, f64> = (0..p).map(|i| (format!("t{i}"), th[i])).collect();
        let want = sf_ir::adjoint::adjoint_grad(&dag, &terms, &values).unwrap();
        assert_eq!(want.gradients.len(), p);

        let mut prog = AdjProgram::from_dag(&dag).unwrap();
        let got = prog.grad(&th, &terms).unwrap().to_vec();
        let mut max_diff = 0.0f64;
        for (a, b) in got.iter().zip(want.gradients.iter()) {
            max_diff = max_diff.max((a - b).abs());
        }
        assert!(
            max_diff < 1e-11,
            "cat vs sf-ir adjoint mismatch: max_diff={max_diff:e}"
        );

        // from_ladder produces the identical program (same layout, same slots).
        let mut lad = AdjProgram::from_ladder(n, layers);
        let got2 = lad.grad(&th, &terms).unwrap().to_vec();
        assert_eq!(got, got2, "from_ladder and from_dag must agree bit-for-bit");

        // Handle reuse with a fresh parameter vector stays aligned.
        let th2: Vec<f64> = th.iter().map(|v| v * 0.7 + 0.2).collect();
        let values2: HashMap<String, f64> = (0..p).map(|i| (format!("t{i}"), th2[i])).collect();
        let want2 = sf_ir::adjoint::adjoint_grad(&dag, &terms, &values2).unwrap();
        let got3 = prog.grad(&th2, &terms).unwrap().to_vec();
        for (a, b) in got3.iter().zip(want2.gradients.iter()) {
            assert!((a - b).abs() < 1e-11, "handle reuse mismatch");
        }
    }

    #[test]
    fn cat_adj_matches_central_fd() {
        let (n, layers) = (3, 2);
        let (dag, p) = ladder_dag(n, layers);
        let terms = z_terms(n);
        let th = theta(p);
        let mut prog = AdjProgram::from_dag(&dag).unwrap();
        let g = prog.grad(&th, &terms).unwrap().to_vec();

        let eps = 1e-5;
        let mut fd = Vec::with_capacity(p);
        for i in 0..p {
            let mut tp = th.clone();
            tp[i] += eps;
            let mut tm = th.clone();
            tm[i] -= eps;
            let ep = prog.expectation(&tp, &terms).unwrap();
            let em = prog.expectation(&tm, &terms).unwrap();
            fd.push((ep - em) / (2.0 * eps));
        }
        let mut max_err = 0.0f64;
        for (a, b) in g.iter().zip(fd.iter()) {
            max_err = max_err.max((a - b).abs());
        }
        assert!(
            max_err < 1e-7,
            "central-FD arbiter failed: max_err={max_err:e}"
        );
    }

    #[test]
    fn fused_dot_and_apply_matches_split_kernels() {
        // The fused kernel replaces dot_one_pauli() followed by pair_pass() on
        // phi, so both the generator dot and the updated phi must agree.
        let dim = 64usize;
        let m = 4usize;
        let w = |i: usize, a: f64, b: f64| {
            Complex64::new(((i as f64) * a).sin(), ((i as f64) * b).cos())
        };
        let src: Vec<Complex64> = (0..dim).map(|i| w(i, 0.13, 0.29)).collect();
        let phi0: Vec<Complex64> = (0..dim).map(|i| w(i, 0.37, 0.71)).collect();
        let gate = [
            [Complex64::new(0.6, 0.0), Complex64::new(0.0, -0.8)],
            [Complex64::new(0.0, 0.8), Complex64::new(0.6, 0.0)],
        ];
        for p in 1..=3u8 {
            let mut split = phi0.clone();
            let want = dot_one_pauli(p, &split, &src, m, dim);
            pair_pass(&mut split, m, gate);

            let mut fused = phi0.clone();
            let got = dot_and_apply(p, &mut fused, &src, m, gate);

            assert!(
                (want - got).norm() < 1e-12,
                "generator dot mismatch for p={p}: {want} vs {got}"
            );
            for i in 0..dim {
                assert!(
                    (split[i] - fused[i]).norm() < 1e-15,
                    "phi update mismatch at {i} for p={p}"
                );
            }
        }
    }

    #[test]
    fn cat_adj_handles_h_x_and_rx() {
        // H/X are constant unitaries -- applied on both sweeps, no slot -- and
        // Rx is differentiated through the X generator.  All three must match
        // the house engine, on both the cached and the streaming sweep.
        let (n, layers) = (5usize, 2usize);
        let mut dag = QuantumDAG::new(n, 0);
        let mut slot = 0usize;
        for layer in 0..layers {
            dag.add_op(OpType::H, &[layer % n]);
            dag.add_op(OpType::X, &[(layer + 1) % n]);
            for i in 0..n {
                dag.add_op(OpType::Rx(var(&format!("t{slot}"))), &[i]);
                slot += 1;
            }
            for i in 0..n.saturating_sub(1) {
                dag.add_op(OpType::CNOT, &[i, i + 1]);
            }
        }
        let p = slot;
        let terms = z_terms(n);
        let th = theta(p);
        let values: HashMap<String, f64> = (0..p).map(|i| (format!("t{i}"), th[i])).collect();
        let want = sf_ir::adjoint::adjoint_grad(&dag, &terms, &values).unwrap();

        let mut cached = AdjProgram::from_dag(&dag).unwrap();
        let got = cached.grad(&th, &terms).unwrap().to_vec();
        let mut streamed = AdjProgram::from_dag(&dag).unwrap();
        streamed.streaming = true;
        streamed.cache = Vec::new();
        let got_s = streamed.grad(&th, &terms).unwrap().to_vec();

        for (i, ((a, b), w)) in got
            .iter()
            .zip(&got_s)
            .zip(want.gradients.iter())
            .enumerate()
        {
            assert!(
                (a - w).abs() < 1e-11,
                "cached vs sf-ir mismatch at {i}: {a} vs {w}"
            );
            assert!(
                (b - w).abs() < 1e-11,
                "streaming vs sf-ir mismatch at {i}: {b} vs {w}"
            );
        }
    }

    #[test]
    fn cat_adj_streaming_matches_cached() {
        // Streaming (no parameter-state cache) must reproduce the cached
        // gradient: the reverse sweep re-walks psi instead of reading stored
        // post-states, which is what lets the handle run above the cache cap.
        let (n, layers) = (6, 3);
        let (dag, p) = ladder_dag(n, layers);
        let terms = z_terms(n);
        let th = theta(p);

        let mut cached = AdjProgram::from_dag(&dag).unwrap();
        let mut streamed = AdjProgram::from_dag(&dag).unwrap();
        streamed.streaming = true;
        streamed.cache = Vec::new();

        let a = cached.grad(&th, &terms).unwrap().to_vec();
        let b = streamed.grad(&th, &terms).unwrap().to_vec();
        let mut max_diff = 0.0f64;
        for (x, y) in a.iter().zip(b.iter()) {
            max_diff = max_diff.max((x - y).abs());
        }
        assert!(
            max_diff < 1e-11,
            "streaming vs cached adjoint mismatch: max_diff={max_diff:e}"
        );

        // Both must also agree with the house engine.
        let values: HashMap<String, f64> = (0..p).map(|i| (format!("t{i}"), th[i])).collect();
        let want = sf_ir::adjoint::adjoint_grad(&dag, &terms, &values).unwrap();
        for (x, w) in b.iter().zip(want.gradients.iter()) {
            assert!((x - w).abs() < 1e-11, "streaming vs sf-ir mismatch");
        }
    }

    #[test]
    fn cat_adj_constant_two_qubit_gates() {
        // SWAP and iSWAP are applied as one index map each rather than their
        // gate-chain expansions, on non-adjacent qubits so the pattern
        // arithmetic is exercised.  iSWAP used to be refused by the plan
        // outright; both must now reproduce the house engine on both sweeps.
        let n = 6usize;
        let mut dag = QuantumDAG::new(n, 0);
        let mut slot = 0usize;
        for _ in 0..2usize {
            for q in 0..n {
                dag.add_op(OpType::Ry(var(&format!("t{slot}"))), &[q]);
                slot += 1;
            }
            dag.add_op(OpType::SWAP, &[0, 5]);
            dag.add_op(OpType::ISWAP, &[1, 4]);
            for q in 0..n {
                dag.add_op(OpType::H, &[q]);
            }
        }
        let p = slot;
        let terms = z_terms(n);
        let th = theta(p);
        let values: HashMap<String, f64> = (0..p).map(|i| (format!("t{i}"), th[i])).collect();
        let want = sf_ir::adjoint::adjoint_grad(&dag, &terms, &values).unwrap();

        let mut cached = AdjProgram::from_dag(&dag).unwrap();
        let got = cached.grad(&th, &terms).unwrap().to_vec();
        let mut streamed = AdjProgram::from_dag(&dag).unwrap();
        streamed.streaming = true;
        streamed.cache = Vec::new();
        let got_s = streamed.grad(&th, &terms).unwrap().to_vec();

        for (i, ((a, b), w)) in got
            .iter()
            .zip(&got_s)
            .zip(want.gradients.iter())
            .enumerate()
        {
            assert!(
                (a - w).abs() < 1e-11,
                "two-qubit constants: cached vs sf-ir mismatch at {i}: {a} vs {w}"
            );
            assert!(
                (b - w).abs() < 1e-11,
                "two-qubit constants: streaming vs sf-ir mismatch at {i}: {b} vs {w}"
            );
        }
    }

    #[test]
    fn cat_adj_three_qubit_constants() {
        // CCX and CSWAP were refused by the plan before this.  sf-ir's adjoint
        // cannot arbitrate them — its apply dispatch is `1 => 1q, 2 => 2q,
        // _ => {}`, so a three-qubit gate is silently absent from its state —
        // hence the arbiter here is a central difference of this program's own
        // forward expectation, on both sweeps.
        let n = 6usize;
        let mut dag = QuantumDAG::new(n, 0);
        let mut slot = 0usize;
        for _ in 0..2usize {
            for q in 0..n {
                dag.add_op(OpType::Ry(var(&format!("t{slot}"))), &[q]);
                slot += 1;
            }
            dag.add_op(OpType::CCX, &[0, 2, 5]);
            dag.add_op(OpType::CSWAP, &[5, 1, 3]);
            dag.add_op(OpType::H, &[4]);
        }
        let p = slot;
        let terms = z_terms(n);
        let th = theta(p);

        let mut cached = AdjProgram::from_dag(&dag).unwrap();
        let got = cached.grad(&th, &terms).unwrap().to_vec();
        let mut streamed = AdjProgram::from_dag(&dag).unwrap();
        streamed.streaming = true;
        streamed.cache = Vec::new();
        let got_s = streamed.grad(&th, &terms).unwrap().to_vec();

        let eps = 1e-5;
        let mut max_err = 0.0f64;
        for i in 0..p {
            let mut tp = th.clone();
            tp[i] += eps;
            let mut tm = th.clone();
            tm[i] -= eps;
            let ep = streamed.expectation(&tp, &terms).unwrap();
            let em = streamed.expectation(&tm, &terms).unwrap();
            let fd = (ep - em) / (2.0 * eps);
            max_err = max_err.max((got[i] - fd).abs());
            max_err = max_err.max((got_s[i] - fd).abs());
        }
        assert!(
            max_err < 1e-7,
            "three-qubit constants vs central FD: max_err={max_err:e}"
        );
    }

    /// Apply a k-qubit matrix the way `OpType::to_matrix` lays it out —
    /// `|q0 … qk⟩`, first listed qubit most significant — written straight
    /// from that definition so it shares no code with the index map under test.
    fn dense_apply(
        sv: &mut [Complex64],
        get: impl Fn(usize, usize) -> Complex64,
        qubits: &[usize],
    ) {
        let k = qubits.len();
        let d = 1usize << k;
        let bits = |p: usize| {
            let mut o = 0usize;
            for j in 0..k {
                if (p >> (k - 1 - j)) & 1 == 1 {
                    o |= 1usize << qubits[j];
                }
            }
            o
        };
        let src = sv.to_vec();
        for base in 0..sv.len() {
            if base & (0..k).fold(0usize, |a, j| a | (1usize << qubits[j])) != 0 {
                continue;
            }
            for p in 0..d {
                let mut acc = Complex64::new(0.0, 0.0);
                for q in 0..d {
                    acc += get(p, q) * src[base | bits(q)];
                }
                sv[base | bits(p)] = acc;
            }
        }
    }

    #[test]
    fn cat_adj_monomial_map_matches_dense_matrices() {
        // The hand-written index maps and the plan's role order must agree
        // with the gate's own matrix, qubit for qubit: this is the check that
        // a control/target swap inside a table cannot pass.
        let n = 6usize;
        let cases = [
            (OpType::SWAP, vec![0, 4], Mono::Swap),
            (OpType::ISWAP, vec![1, 3], Mono::ISwap),
            (OpType::CCX, vec![0, 2, 5], Mono::Ccx),
            (OpType::CSWAP, vec![5, 1, 3], Mono::Cswap),
        ];
        for (op, qubits, kind) in cases {
            let k = qubits.len();
            let base: Vec<Complex64> = (0..(1usize << n))
                .map(|i| Complex64::new(((i * 37 % 11) as f64) - 5.0, ((i * 53 % 7) as f64) - 3.0))
                .collect();
            let m = op.to_matrix();
            let mut want = base.clone();
            dense_apply(&mut want, |r, c| m[(r, c)], &qubits);

            let mut qs = [0u32; 3];
            for (j, &q) in qubits.iter().enumerate() {
                qs[j] = q as u32;
            }
            let mut got = base.clone();
            mono_apply(&mut got, kind, qs, k, false);
            for i in 0..got.len() {
                assert!(
                    (got[i] - want[i]).norm() < 1e-12,
                    "{} map vs to_matrix differs at {i}: {} vs {}",
                    op.name(),
                    got[i],
                    want[i]
                );
            }
            // Every one of these is its own inverse up to the phase.
            mono_apply(&mut got, kind, qs, k, true);
            for i in 0..got.len() {
                assert!(
                    (got[i] - base[i]).norm() < 1e-12,
                    "{} dagger does not undo itself at {i}",
                    op.name()
                );
            }
        }
    }

    #[test]
    fn cat_adj_rejects_unsupported_and_bad_theta() {
        let mut dag = QuantumDAG::new(2, 0);
        dag.add_op(OpType::ECR, &[0, 1]);
        dag.add_op(OpType::Ry(var("x")), &[1]);
        let err = AdjProgram::from_dag(&dag).err().unwrap();
        assert!(err.contains("unsupported gate"), "got: {err}");

        let mut ok = QuantumDAG::new(2, 0);
        ok.add_op(OpType::Ry(var("x")), &[0]);
        ok.add_op(OpType::CNOT, &[0, 1]);
        let mut prog = AdjProgram::from_dag(&ok).unwrap();
        let terms = z_terms(2);
        assert!(prog.grad(&[0.5, 0.5], &terms).is_err());
        assert!(prog.grad(&[0.5], &terms).is_ok());
        assert_eq!(prog.num_params(), 1);
        assert_eq!(prog.n_qubits(), 2);

        // A bound angle is baked into the plan rather than refused: it carries
        // no gradient, but it still has to be applied to the state.
        let mut cst = QuantumDAG::new(1, 0);
        cst.add_op(OpType::Ry(Parameter::Const(0.4)), &[0]);
        let mut baked = AdjProgram::from_dag(&cst).unwrap();
        assert_eq!(baked.num_params(), 0);
        let e = baked.expectation(&[], &z_terms(1)).unwrap();
        assert!((e - 0.4f64.cos()).abs() < 1e-12, "baked expval {e}");
    }

    /// Every gate the fast lane used to refuse, in one circuit: the controlled
    /// rotations (`CRx`/`CRz`/`CP`/`CRy` as `Cu` with θ only, plus a four-angle
    /// `Cu`), the Pauli entanglers (`Rzz`/`Rxx`/`Ryy`), `U`/`P`/`R1` with a live
    /// angle, and the constant phase gates (`S`/`T`/`CZ`/`CY`/`SWAP`).
    #[test]
    fn cat_adj_handles_controlled_and_phased_gates() {
        let n = 6usize;
        let mut dag = QuantumDAG::new(n, 0);
        let mut k = 0usize;
        let mut nxt = || {
            let p = Parameter::Variable {
                name: format!("t{k}"),
                id: 0,
            };
            k += 1;
            p
        };

        dag.add_op(OpType::H, &[0]);
        dag.add_op(OpType::S, &[1]);
        dag.add_op(OpType::T, &[2]);
        dag.add_op(OpType::Ryy(nxt()), &[0, 1]);
        dag.add_op(OpType::Rxx(nxt()), &[1, 2]);
        dag.add_op(OpType::Rzz(nxt()), &[2, 3]);
        dag.add_op(OpType::CRx(nxt()), &[0, 2]);
        dag.add_op(OpType::CRz(nxt()), &[1, 3]);
        dag.add_op(OpType::CP(nxt()), &[2, 4]);
        // CRy is `Cu(θ, 0, 0, 0)`; the four-angle form differentiates θ and λ.
        dag.add_op(
            OpType::Cu(
                nxt(),
                Parameter::Const(0.0),
                Parameter::Const(0.0),
                Parameter::Const(0.0),
            ),
            &[0, 1],
        );
        dag.add_op(
            OpType::Cu(nxt(), Parameter::Const(0.3), nxt(), Parameter::Const(-0.2)),
            &[3, 5],
        );
        dag.add_op(OpType::CZ, &[4, 5]);
        dag.add_op(OpType::CY, &[1, 4]);
        dag.add_op(OpType::SWAP, &[0, 5]);
        dag.add_op(OpType::U(nxt(), Parameter::Const(0.4), nxt()), &[2]);
        dag.add_op(OpType::P(nxt()), &[4]);
        dag.add_op(OpType::R1(nxt()), &[5]);
        dag.add_op(OpType::Ry(nxt()), &[3]);
        dag.add_op(OpType::Rx(Parameter::Const(0.7)), &[1]);

        let p = k;
        let terms = z_terms(n);
        let th = theta(p);

        let mut cached = AdjProgram::from_dag(&dag).unwrap();
        let got = cached.grad(&th, &terms).unwrap().to_vec();
        let mut streamed = AdjProgram::from_dag(&dag).unwrap();
        streamed.streaming = true;
        streamed.cache = Vec::new();
        let got_s = streamed.grad(&th, &terms).unwrap().to_vec();

        // Two arbiters.  Central FD of *this* program's own forward pass is
        // the tight one: same lowering, so any disagreement is the reverse
        // sweep alone.  `sf_ir::adjoint` is the independent one -- a separate
        // engine, so it also catches a lowering that is self-consistent but
        // wrong (`CZ` was lowered to `diag(1,1,-1,-1)` at first and only the
        // Python cross-check against PennyLane caught it).
        let eps = 1e-5;
        let mut fd = Vec::with_capacity(p);
        for i in 0..p {
            let mut tp = th.clone();
            tp[i] += eps;
            let mut tm = th.clone();
            tm[i] -= eps;
            let ep = cached.expectation(&tp, &terms).unwrap();
            let em = cached.expectation(&tm, &terms).unwrap();
            fd.push((ep - em) / (2.0 * eps));
        }
        let values: HashMap<String, f64> = (0..p).map(|i| (format!("t{i}"), th[i])).collect();
        let want = sf_ir::adjoint::adjoint_grad(&dag, &terms, &values).unwrap();

        assert_eq!(got.len(), p);
        let mut bad = Vec::new();
        let mut worst = 0.0f64;
        for (i, ((a, b), w)) in got.iter().zip(&got_s).zip(fd.iter()).enumerate() {
            let da = (a - w).abs();
            let db = (b - w).abs();
            if da > 1e-7 || db > 1e-7 {
                bad.push(format!("t{i}: cached {a} stream {b} vs fd {w}"));
            }
            worst = worst.max(da.max(db));
        }
        assert!(bad.is_empty(), "{bad:#?}");
        assert!(worst < 1e-7, "worst {worst:e}");

        let mut bad = Vec::new();
        for (i, ((a, b), w)) in got
            .iter()
            .zip(&got_s)
            .zip(want.gradients.iter())
            .enumerate()
        {
            if (a - w).abs() > 1e-7 || (b - w).abs() > 1e-7 {
                bad.push(format!("t{i}: cached {a} stream {b} vs sf-ir {w}"));
            }
        }
        assert!(bad.is_empty(), "{bad:#?}");

        // Cached vs streaming is a much tighter statement: same formulas, so
        // only round-off may differ.
        for (i, (a, b)) in got.iter().zip(&got_s).enumerate() {
            assert!(
                (a - b).abs() < 1e-11,
                "cached vs streaming at {i}: {a} vs {b}"
            );
        }
    }

    /// `reset(0)` on a fresh qubit is |0> -> |0>, so it must not disqualify a
    /// differentiable circuit.  The gradient has to be *identical* to the same
    /// circuit without the reset -- not merely close.
    #[test]
    fn cat_adj_skips_a_leading_reset_and_refuses_a_real_one() {
        let build = |with_reset: bool| {
            let mut dag = QuantumDAG::new(2, 0);
            if with_reset {
                dag.add_op(OpType::Reset, &[0]);
            }
            dag.add_op(OpType::Ry(var("t")), &[0]);
            dag.add_op(OpType::CNOT, &[0, 1]);
            dag
        };

        let terms = z_terms(2);
        let mut plain = AdjProgram::from_dag(&build(false)).unwrap();
        let mut reset = AdjProgram::from_dag(&build(true))
            .expect("a leading reset is the identity, not a collapse");
        let a = plain.grad(&[0.7], &terms).unwrap();
        let b = reset.grad(&[0.7], &terms).unwrap();
        assert_eq!(a, b, "leading reset must not change the gradient");

        // A reset *after* use is still a collapse and still refused.
        let mut bad = QuantumDAG::new(2, 0);
        bad.add_op(OpType::Ry(var("t")), &[0]);
        bad.add_op(OpType::Reset, &[0]);
        let err = AdjProgram::from_dag(&bad).err().unwrap();
        assert!(err.contains("Reset on qubit 0 after it was used"), "{err}");
    }

    /// Fusion must be a pure memory-traffic change: the same kernels on the
    /// same gates in the same order, so the fused and per-gate sweeps may
    /// differ only by the order the dot is summed in.
    #[test]
    fn fused_sweep_matches_per_gate_sweep() {
        let n = 6usize;
        let mut dag = QuantumDAG::new(n, 0);
        let mut k = 0usize;
        let mut nxt = || {
            let p = Parameter::Variable {
                name: format!("t{k}"),
                id: 0,
            };
            k += 1;
            p
        };
        // Long single-qubit runs (these fuse), a distant pair (`Rzz 0-4`),
        // both Pauli-entangler flavours, and gates that stay on the per-gate
        // path -- the sweep has to interleave the two without changing either.
        dag.add_op(OpType::H, &[0]);
        dag.add_op(OpType::Ry(nxt()), &[1]);
        dag.add_op(OpType::Rz(nxt()), &[2]);
        dag.add_op(OpType::Rx(nxt()), &[3]);
        dag.add_op(OpType::Ry(nxt()), &[0]);
        dag.add_op(OpType::Rzz(nxt()), &[0, 4]);
        dag.add_op(OpType::Ryy(nxt()), &[1, 5]);
        dag.add_op(OpType::Rxx(nxt()), &[2, 4]);
        dag.add_op(OpType::CNOT, &[0, 1]);
        dag.add_op(OpType::S, &[2]);
        dag.add_op(OpType::Rz(nxt()), &[3]);
        dag.add_op(OpType::Rz(nxt()), &[4]);
        dag.add_op(OpType::Rz(nxt()), &[5]);
        dag.add_op(OpType::CZ, &[3, 5]);
        dag.add_op(OpType::Rz(nxt()), &[0]);
        dag.add_op(OpType::Rz(nxt()), &[1]);

        let p = k;
        let terms = z_terms(n);
        let th = theta(p);

        let build = || {
            let mut prog = AdjProgram::from_dag(&dag).unwrap();
            prog.streaming = true;
            prog.cache = Vec::new();
            prog
        };
        let gf = build()
            .grad_with_shapes(&th, &terms, true, false)
            .unwrap()
            .to_vec();
        let gp = build()
            .grad_with_shapes(&th, &terms, false, false)
            .unwrap()
            .to_vec();
        let gs = build()
            .grad_with_shapes(&th, &terms, false, true)
            .unwrap()
            .to_vec();
        let gd = build()
            .grad_with_sweeps(&th, &terms, false, false, true)
            .unwrap()
            .to_vec();

        assert_eq!(gf.len(), p);
        let mut bad = Vec::new();
        let mut worst = 0.0f64;
        for (i, (a, b)) in gf.iter().zip(&gp).enumerate() {
            let d = (a - b).abs();
            if d > 1e-12 {
                bad.push(format!("t{i}: fused {a} vs per-gate {b}"));
            }
            worst = worst.max(d);
        }
        assert!(bad.is_empty(), "{bad:#?}");
        assert!(worst < 1e-12, "worst |fused - per-gate| = {worst:e}");
        // The strip sweep is the same kernels walked strip-by-strip, so it has
        // to land on the per-gate gradient too -- the only licence it takes is
        // the summation order of the dots, which is why the bound is a tolerance
        // and not equality.
        let mut bad = Vec::new();
        let mut worst = 0.0f64;
        for (i, (a, b)) in gs.iter().zip(&gp).enumerate() {
            let d = (a - b).abs();
            if d > 1e-12 {
                bad.push(format!("t{i}: strip {a} vs per-gate {b}"));
            }
            worst = worst.max(d);
        }
        assert!(bad.is_empty(), "{bad:#?}");
        assert!(worst < 1e-12, "worst |strip - per-gate| = {worst:e}");
        // The diagonal sweep is not the same arithmetic: it sums the run's
        // angles and takes one `sin`/`cos` instead of multiplying `m` unit
        // complex numbers, so it is allowed to differ in the last bits -- but
        // only there, and the run in this circuit is 3 Rz deep plus a pair.
        let mut bad = Vec::new();
        let mut worst = 0.0f64;
        for (i, (a, b)) in gd.iter().zip(&gp).enumerate() {
            let d = (a - b).abs();
            if d > 1e-12 {
                bad.push(format!("t{i}: diag {a} vs per-gate {b}"));
            }
            worst = worst.max(d);
        }
        assert!(bad.is_empty(), "{bad:#?}");
        assert!(worst < 1e-12, "worst |diag - per-gate| = {worst:e}");
        // An all-zero gradient would pass the comparisons above for free.
        assert!(gf.iter().any(|g| g.abs() > 1e-6), "gradient is all zero");
    }

    /// The diagonal run finder must span a run that crosses qubits, and must
    /// stop at the first gate that is not diagonal -- nothing else bounds it,
    /// which is the difference from the strip and block finders.
    #[test]
    fn diag_run_spans_a_qubit_spread_and_stops_at_a_non_diagonal_gate() {
        let mut dag = QuantumDAG::new(4, 0);
        dag.add_op(OpType::Ry(var("a")), &[0]);
        dag.add_op(OpType::Rz(var("b")), &[0]);
        dag.add_op(OpType::Rz(var("c")), &[1]);
        dag.add_op(OpType::Rzz(var("d")), &[2, 3]);
        dag.add_op(OpType::Ry(var("e")), &[2]);
        let prog = AdjProgram::from_dag(&dag).unwrap();
        assert_eq!(prog.gates.len(), 5);
        // Gates 1..4 are `Rz q0`, `Rz q1`, `Rzz q2-q3`: four distinct qubits,
        // no shared support, and still one run -- a diagonal gate couples no
        // two amplitudes, so there is nothing to gather.
        assert_eq!(prog.diag_run(4), 1);
        assert_eq!(prog.diag_run_from(1), 4);
        // A non-diagonal gate is a wall in both directions.
        assert_eq!(prog.diag_run(1), 1);
        assert_eq!(prog.diag_run_from(4), 4);
        assert_eq!(prog.diag_run_from(0), 0);

        // And the cap binds, or the per-thread dot array would be unbounded.
        let mut wide = QuantumDAG::new(4, 0);
        for j in 0..70 {
            wide.add_op(OpType::Rz(var(&format!("w{j}"))), &[j % 4]);
        }
        let w = AdjProgram::from_dag(&wide).unwrap();
        assert_eq!(w.gates.len(), 70);
        assert_eq!(w.diag_run(70), 70 - DIAG_RUN_MAX);
    }

    /// The sign must come from the amplitude's index in the whole state, not
    /// from its index in the slice being walked: a chunk begins at an arbitrary
    /// offset, and the masks are absolute qubit positions.
    ///
    /// This is the contract the driver relies on when it splits the state, and
    /// it is checked directly rather than through the driver because the split
    /// is chosen from the process-wide thread width -- asserting on that from a
    /// test races every other test in the binary.
    #[test]
    fn diag_kernel_takes_its_sign_from_the_absolute_index() {
        let n = 6usize;
        let dim = 1usize << n;
        let ops = [
            DiagOp {
                mask: 1usize << 3,
                t2: [0.7, -0.7],
                slot: 0,
            },
            DiagOp {
                mask: (1usize << 1) | (1usize << 4),
                t2: [-0.35, 0.35],
                slot: 1,
            },
        ];
        let phi: Vec<Complex64> = (0..dim)
            .map(|i| Complex64::new((i as f64 * 0.37).sin(), (i as f64 * 0.11).cos()))
            .collect();
        let psi: Vec<Complex64> = (0..dim)
            .map(|i| Complex64::new((i as f64 * 0.23).cos(), (i as f64 * 0.19).sin()))
            .collect();

        let (mut p0, mut s0) = (phi.clone(), psi.clone());
        let mut d0 = [Complex64::new(0.0, 0.0); DIAG_RUN_MAX];
        diag_backward_blocks(&mut p0, &mut s0, 0, &ops, &mut d0);

        // Same data, walked as two halves, each with its own offset.
        let (mut p1, mut s1) = (phi, psi);
        let mut d1 = [Complex64::new(0.0, 0.0); DIAG_RUN_MAX];
        let h = dim / 2;
        {
            let (pa, pb) = p1.split_at_mut(h);
            let (sa, sb) = s1.split_at_mut(h);
            diag_backward_blocks(pa, sa, 0, &ops, &mut d1);
            diag_backward_blocks(pb, sb, h, &ops, &mut d1);
        }

        for j in 0..ops.len() {
            let d = (d0[j] - d1[j]).norm();
            assert!(
                d < 1e-12,
                "dot {j} differs by {d:e} between one pass and two"
            );
        }
        let mut worst = 0.0f64;
        for i in 0..dim {
            worst = worst
                .max((p0[i] - p1[i]).norm())
                .max((s0[i] - s1[i]).norm());
        }
        assert!(worst < 1e-15, "state differs by {worst:e}");
        // A one-Rz-per-half run is still a run: the dots have to be non-zero,
        // or the comparison above is satisfied by two zeroes.
        assert!(d0.iter().any(|d| d.re.abs() + d.im.abs() > 1e-9));
    }

    /// Integration: the sweep must land on the per-gate gradient at a size whose
    /// state is far above `par_min()`.  The absolute-index contract itself is
    /// pinned by `diag_kernel_takes_its_sign_from_the_absolute_index`.
    #[test]
    fn diag_sweep_matches_per_gate_on_a_large_state() {
        let n = 15usize;
        let build = || {
            let mut prog = AdjProgram::from_ladder(n, 1);
            prog.streaming = true;
            prog.cache = Vec::new();
            prog
        };
        let probe = build();
        let p = probe.p;
        let th = theta(p);
        let terms = z_terms(n);
        let gd = build()
            .grad_with_sweeps(&th, &terms, false, false, true)
            .unwrap()
            .to_vec();
        let gp = build()
            .grad_with_sweeps(&th, &terms, false, false, false)
            .unwrap()
            .to_vec();
        let mut worst = 0.0f64;
        for (a, b) in gd.iter().zip(&gp) {
            worst = worst.max((a - b).abs());
        }
        assert!(worst < 1e-12, "worst |diag - per-gate| = {worst:e}");
        assert!(gp.iter().any(|g| g.abs() > 1e-6), "gradient is all zero");
    }

    /// The strip run finder must actually return runs, and must stop exactly at
    /// the gate the cache budget cannot hold: if it silently returned one gate
    /// at a time the sweep would be the old one plus bookkeeping, and the test
    /// above would still pass.
    #[test]
    fn strip_run_spans_what_fits_and_stops_at_the_budget() {
        let prog = AdjProgram::from_ladder(6, 1);
        let n_gates = prog.gates.len();
        // A budget of the whole state: every gate of the ladder fits, so the run
        // has to span the tape instead of dribbling out one gate at a time.
        let (start, bits) = prog.strip_run(n_gates, prog.dim);
        assert_eq!(start, 0, "run covered only {} gate(s)", n_gates - start);
        // Every gate of the run has to fit the strip the run declares.
        assert!(bits >= 1 && (1usize << bits) <= prog.dim);
        // And the budget has to bind: a strip one element short of the highest
        // qubit cannot host the gate that needs it.
        let (start, bits) = prog.strip_run(n_gates, (1usize << 5) - 1);
        assert!(n_gates - start < n_gates, "the budget did not bind");
        assert!((1usize << bits) < (1usize << 5));
    }

    /// The run finder is the whole claim: if it never returns more than one
    /// gate, the sweep is the old one plus bookkeeping.
    #[test]
    fn fusable_run_stops_at_the_span_and_at_constant_gates() {
        // Ladder: Ry/Rz pairs then a CNOT ladder.  Four RY/RZ gates on two
        // qubits fuse; the CNOTs stay on the per-gate path.
        let prog = AdjProgram::from_ladder(3, 1);
        let (start, qs, nq) = prog.fusable_run(6);
        assert_eq!((start, nq), (2, 2), "expected the 4 gate run on q1/q2");
        assert_eq!(&qs[..nq], &[1, 2]);
        // Ending on a CNOT yields no run at all.
        assert_eq!(prog.fusable_run(7).2, 0);

        // Five single-qubit gates on five qubits: the run stops at
        // `MAX_RUN_Q`, so gate 0 is left to the per-gate path.
        let mut dag = QuantumDAG::new(5, 0);
        for q in 0..5 {
            dag.add_op(OpType::Ry(var(&format!("t{q}"))), &[q]);
        }
        let wide = AdjProgram::from_dag(&dag).unwrap();
        let (start, qs, nq) = wide.fusable_run(5);
        assert_eq!((start, nq), (1, MAX_RUN_Q));
        assert_eq!(&qs[..nq], &[1, 2, 3, 4]);
    }
}
