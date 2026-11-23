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

// ─── Generator table ───────────────────────────────────────────────────
//
// The adjoint gradient needs, per parameter slot, the generator G of
// ∂U/∂θ = −i·G·U.  G is Hermitian, so it always expands in the Pauli
// basis of the gate's own qubits with real coefficients — and that
// expansion is all the backward pass needs, it never materialises G.
//
// Two sources fill the table:
//
//   * `closed_form_generator` — the analytic generators: one term for the
//     plain rotations (arithmetic unchanged from the historical hot path)
//     and a short sum for the controlled variants;
//   * `complex_step_generator` — a universal fallback that differentiates
//     the gate matrix itself, so a parametric gate can never silently
//     contribute 0.0 to the gradient again (the 0.1.23 defect: P, U, CP,
//     Cu, CRx, CRz and R1 had no entry and were dropped without a trace).
//
// A parametric gate that neither source resolves is a `MethodError`,
// never a silent zero.

/// Pauli encoding of a generator factor: 0=I, 1=X, 2=Y, 3=Z.
const PAULI_I: u8 = 0;
const PAULI_X: u8 = 1;
const PAULI_Y: u8 = 2;
const PAULI_Z: u8 = 3;

/// One term of a generator's Pauli expansion: `coef · P[p₀] ⊗ P[p₁]` on
/// the gate's qubits (local index 0 pins an identity factor).  Gates here
/// are at most two-qubit, so the string never needs its own allocation.
#[derive(Clone, Copy)]
struct GenTerm {
    p: [u8; 2],
    len: u8,
    coef: f64,
}

impl GenTerm {
    #[inline]
    fn paulis(&self) -> &[u8] {
        &self.p[..self.len as usize]
    }
}

#[inline]
fn gen1(p: u8, coef: f64) -> GenTerm {
    GenTerm {
        p: [p, PAULI_I],
        len: 1,
        coef,
    }
}

#[inline]
fn gen2(a: u8, b: u8, coef: f64) -> GenTerm {
    GenTerm {
        p: [a, b],
        len: 2,
        coef,
    }
}

/// Complex 2×2 matrix, the local factor of a generator expansion.
type M2 = [[Complex64; 2]; 2];

#[inline]
fn m2_pauli(p: u8) -> M2 {
    let zero = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);
    match p {
        PAULI_X => [[zero, one], [one, zero]],
        PAULI_Y => [
            [zero, Complex64::new(0.0, -1.0)],
            [Complex64::new(0.0, 1.0), zero],
        ],
        PAULI_Z => [[one, zero], [zero, -one]],
        _ => [[one, zero], [zero, one]],
    }
}

/// Expand a Hermitian 2×2 matrix as Σ cₐ·Pₐ with cₐ = Tr(Pₐ·G)/2.
///
/// The identity component is dropped: it contributes 2c·Im⟨φ|ψ_after⟩,
/// and φ is constructed so that ⟨φ|ψ_after⟩ = ⟨O⟩ is real, so the identity
/// part of *any* generator is an exactly-zero gradient.  That is what
/// keeps the controlled-rotation generators at two terms instead of
/// three.  Terms under the 1e-12 floor are dropped as well (they are
/// rounding residue from the complex-step fallback).
fn pauli_decompose_1q(g: &M2) -> Vec<GenTerm> {
    let mut out = Vec::with_capacity(3);
    for p in [PAULI_X, PAULI_Y, PAULI_Z] {
        let m = m2_pauli(p);
        // Tr(P·G) = Σᵢⱼ Pᵢⱼ·Gⱼᵢ.
        let mut tr = Complex64::new(0.0, 0.0);
        for i in 0..2 {
            for j in 0..2 {
                tr += m[i][j] * g[j][i];
            }
        }
        let coef = tr.re * 0.5;
        if coef.abs() > 1e-12 {
            out.push(gen1(p, coef));
        }
    }
    out
}

/// Two-qubit counterpart of `pauli_decompose_1q`: c_ab = Tr((Pₐ⊗P_b)·G)/4
/// with local basis index `2·b(q₀)+b(q₁)` (the first qubit argument is the
/// high bit, matching `ops::to_unitary_matrix` and `inplace_2q_general`).
///
/// Here the identity factors are *kept* — I⊗Z and Z⊗I are not the identity
/// and do carry gradient — and only the all-identity term is dropped, for
/// the same reason as in the 1-qubit case.
fn pauli_decompose_2q(g: &nalgebra::DMatrix<Complex64>) -> Vec<GenTerm> {
    let mut out = Vec::with_capacity(15);
    for a in [PAULI_I, PAULI_X, PAULI_Y, PAULI_Z] {
        for b in [PAULI_I, PAULI_X, PAULI_Y, PAULI_Z] {
            if a == PAULI_I && b == PAULI_I {
                continue;
            }
            let ma = m2_pauli(a);
            let mb = m2_pauli(b);
            let mut tr = Complex64::new(0.0, 0.0);
            for i in 0..4 {
                for j in 0..4 {
                    tr += ma[i >> 1][j >> 1] * mb[i & 1][j & 1] * g[(j, i)];
                }
            }
            let coef = tr.re * 0.25;
            if coef.abs() > 1e-12 {
                out.push(gen2(a, b, coef));
            }
        }
    }
    out
}

/// Generator matrices of `U(θ,φ,λ)` by slot, in the `∂U/∂θ = −i·G·U`
/// convention, derived from `U·U† = I`:
///
/// * θ: `∂θU·U† = ½[[0, −e^{−iφ}], [e^{iφ}, 0]]` (φ=0 gives Y/2, i.e. Ry);
/// * φ: `∂φU·U† = i·|1⟩⟨1|` ⇒ `G = −|1⟩⟨1|`;
/// * λ: `G = −[[s², −e^{−iφ}sc], [−e^{iφ}sc, c²]]`, `s/c = sin/cos(θ/2)`.
///
/// These are the *full* generators, identity component included — the
/// decomposition drops it, and it must be dropped at the final matrix
/// width, not here: `Cu` embeds the 2×2 block into `diag(0, ·)` and a
/// constant that is identity on 2 dimensions is not identity on 4.
///
/// They stay matrix-valued because the axis depends on the other angles.
/// `None` for a slot this gate does not have.
fn u3_generator(theta: f64, phi: f64, slot: usize) -> Option<M2> {
    let zero = Complex64::new(0.0, 0.0);
    let half = |re: f64, im: f64| Complex64::new(re * 0.5, im * 0.5);
    Some(match slot {
        0 => {
            let (sp, cp) = phi.sin_cos();
            [[zero, half(-sp, -cp)], [half(-sp, cp), zero]]
        }
        1 => [[zero, zero], [zero, Complex64::new(-1.0, 0.0)]],
        2 => {
            let (sp, cp) = phi.sin_cos();
            let (sh, ch) = (theta * 0.5).sin_cos();
            let off = sh * ch;
            [
                [
                    Complex64::new(-sh * sh, 0.0),
                    Complex64::new(off * cp, -off * sp),
                ],
                [
                    Complex64::new(off * cp, off * sp),
                    Complex64::new(-ch * ch, 0.0),
                ],
            ]
        }
        _ => return None,
    })
}

/// Generator matrices of `Cu(θ,φ,λ,γ)` by slot.  `Cu = diag(I₂, e^{iγ}·U3)`
/// and both the control-0 block and the `e^{iγ}` phase are constant in
/// θ/φ/λ, so slots 0-2 carry exactly the `U3` generator embedded in the
/// control-1 block (the phase cancels against `U†`).  Slot 3 (γ) is
/// handled by the closed-form table, not here.
fn cu_generator(theta: f64, phi: f64, slot: usize) -> Option<nalgebra::DMatrix<Complex64>> {
    let u = u3_generator(theta, phi, slot)?;
    let mut g = nalgebra::DMatrix::<Complex64>::zeros(4, 4);
    for i in 0..2 {
        for j in 0..2 {
            g[(2 + i, 2 + j)] = u[i][j];
        }
    }
    Some(g)
}

/// Analytic generator for slot `slot` of `op`, as a Pauli expansion of
/// `G` in `∂U/∂θ = −i·G·U`.  `None` means "not in the closed-form table"
/// — the caller falls back to the numerical generator.
///
/// Every entry below is verified against the matrices in
/// `ops::to_unitary_matrix` by `test_every_parametric_gate_has_generator`.
fn closed_form_generator(op: &OpType, slot: usize) -> Option<Vec<GenTerm>> {
    let angle = |i: usize| op.parameters().get(i).and_then(|p| p.try_evaluate());
    Some(match op {
        OpType::Rx(_) if slot == 0 => vec![gen1(PAULI_X, 0.5)],
        OpType::Ry(_) if slot == 0 => vec![gen1(PAULI_Y, 0.5)],
        OpType::Rz(_) if slot == 0 => vec![gen1(PAULI_Z, 0.5)],
        OpType::Rxx(_) if slot == 0 => vec![gen2(PAULI_X, PAULI_X, 0.5)],
        OpType::Ryy(_) if slot == 0 => vec![gen2(PAULI_Y, PAULI_Y, 0.5)],
        OpType::Rzz(_) if slot == 0 => vec![gen2(PAULI_Z, PAULI_Z, 0.5)],
        // P(φ) = diag(1, e^{iφ}) = e^{iφ/2}·Rz(φ): the global phase is
        // unitary, so it cancels out of G — identical derivative to Rz.
        OpType::P(_) | OpType::R1(_) if slot == 0 => vec![gen1(PAULI_Z, 0.5)],
        // CP(φ) = diag(1,1,1,e^{iφ}) has G = −|11⟩⟨11| = −¼(I−Z)⊗(I−Z);
        // dropping the identity component leaves ¼(I⊗Z + Z⊗I − Z⊗Z).
        OpType::CP(_) if slot == 0 => vec![
            gen2(PAULI_I, PAULI_Z, 0.25),
            gen2(PAULI_Z, PAULI_I, 0.25),
            gen2(PAULI_Z, PAULI_Z, -0.25),
        ],
        // CRx(θ) = diag(1,1,Rx(θ)) has G = ½|1⟩⟨1|⊗X = ¼(I⊗X − Z⊗X).
        OpType::CRx(_) if slot == 0 => {
            vec![gen2(PAULI_I, PAULI_X, 0.25), gen2(PAULI_Z, PAULI_X, -0.25)]
        }
        // CRz(θ) = diag(1,1,Rz(θ)) has G = ½|1⟩⟨1|⊗Z.
        OpType::CRz(_) if slot == 0 => {
            vec![gen2(PAULI_I, PAULI_Z, 0.25), gen2(PAULI_Z, PAULI_Z, -0.25)]
        }
        // Cu(…, γ) = diag(1,1,e^{iγ}·U3) has G = −|1⟩⟨1|⊗I = ½(Z⊗I − I⊗I);
        // identity dropped → ½·Z⊗I.
        OpType::Cu(..) if slot == 3 => vec![gen2(PAULI_Z, PAULI_I, 0.5)],
        // U(θ,φ,λ) and Cu(θ,φ,λ,·) slots 0-2: matrix generator → Pauli.
        OpType::U(..) => pauli_decompose_1q(&u3_generator(angle(0)?, angle(1)?, slot)?),
        OpType::Cu(..) => pauli_decompose_2q(&cu_generator(angle(0)?, angle(1)?, slot)?),
        _ => return None,
    })
}

/// Replace parameter `slot` of `op` with a constant.  Also the canonical
/// slot ordering: `slot < op.parameters().len()` always resolves.
fn op_with_slot(op: &OpType, slot: usize, value: f64) -> Option<OpType> {
    let p = Parameter::Const(value);
    Some(match op {
        OpType::Rx(_) if slot == 0 => OpType::Rx(p),
        OpType::Ry(_) if slot == 0 => OpType::Ry(p),
        OpType::Rz(_) if slot == 0 => OpType::Rz(p),
        OpType::R1(_) if slot == 0 => OpType::R1(p),
        OpType::P(_) if slot == 0 => OpType::P(p),
        OpType::Rxx(_) if slot == 0 => OpType::Rxx(p),
        OpType::Ryy(_) if slot == 0 => OpType::Ryy(p),
        OpType::Rzz(_) if slot == 0 => OpType::Rzz(p),
        OpType::CRx(_) if slot == 0 => OpType::CRx(p),
        OpType::CRz(_) if slot == 0 => OpType::CRz(p),
        OpType::CP(_) if slot == 0 => OpType::CP(p),
        OpType::U(_, b, c) if slot == 0 => OpType::U(p, b.clone(), c.clone()),
        OpType::U(a, _, c) if slot == 1 => OpType::U(a.clone(), p, c.clone()),
        OpType::U(a, b, _) if slot == 2 => OpType::U(a.clone(), b.clone(), p),
        OpType::Cu(_, b, c, d) if slot == 0 => OpType::Cu(p, b.clone(), c.clone(), d.clone()),
        OpType::Cu(a, _, c, d) if slot == 1 => OpType::Cu(a.clone(), p, c.clone(), d.clone()),
        OpType::Cu(a, b, _, d) if slot == 2 => OpType::Cu(a.clone(), b.clone(), p, d.clone()),
        OpType::Cu(a, b, c, _) if slot == 3 => OpType::Cu(a.clone(), b.clone(), c.clone(), p),
        _ => return None,
    })
}

/// Short gate name for error messages and tests.
fn op_label(op: &OpType) -> &'static str {
    match op {
        OpType::Rx(_) => "Rx",
        OpType::Ry(_) => "Ry",
        OpType::Rz(_) => "Rz",
        OpType::R1(_) => "R1",
        OpType::P(_) => "P",
        OpType::Rxx(_) => "Rxx",
        OpType::Ryy(_) => "Ryy",
        OpType::Rzz(_) => "Rzz",
        OpType::CRx(_) => "CRx",
        OpType::CRz(_) => "CRz",
        OpType::CP(_) => "CP",
        OpType::U(..) => "U",
        OpType::Cu(..) => "Cu",
        _ => "gate",
    }
}

/// Safety net for a parametric gate that is not in the closed-form table:
/// `G = i·(∂U/∂θ)·U†` from the gate matrix itself, by a 5-point central
/// difference of `to_matrix()`.
///
/// A complex step would be exact here (every entry is sin/cos/exp of the
/// angle), but `Parameter::Const` is a real `f64`, so there is no way to
/// ask the IR for `U(θ + ih)`.  A 5-point stencil is the best available:
/// truncation is O(h⁴) and round-off ~1e-16/h, so h = 1e-4 lands near
/// 1e-12 — far inside the 1e-8 gradient tolerance, and this path is only
/// reached by gates the table does not know.
///
/// `None` when the gate exposes no parameter at `slot`, or when the whole
/// generator vanishes (an op whose matrix is not implemented returns the
/// identity from `to_matrix()`, which differentiates to zero).
fn finite_difference_generator(op: &OpType, slot: usize) -> Option<Vec<GenTerm>> {
    const H: f64 = 1e-4;
    let theta = op.parameters().get(slot)?.try_evaluate()?;
    let at = |d: f64| op_with_slot(op, slot, theta + d).map(|o| o.to_matrix());
    let (u, um1, up1) = (op.to_matrix(), at(-H)?, at(H)?);
    let (um2, up2) = (at(-2.0 * H)?, at(2.0 * H)?);
    let dim = u.nrows();
    if !matches!(dim, 2 | 4)
        || up1.nrows() != dim
        || um1.nrows() != dim
        || up2.nrows() != dim
        || um2.nrows() != dim
    {
        return None;
    }
    // ∂U/∂θ = [8(U₊ − U₋) − (U₂ − U₂₋)]/(12h), then G = i·(∂U/∂θ)·U†.
    let mut g = nalgebra::DMatrix::<Complex64>::zeros(dim, dim);
    for i in 0..dim {
        for j in 0..dim {
            let mut acc = Complex64::new(0.0, 0.0);
            for k in 0..dim {
                let du = (up1[(i, k)] - um1[(i, k)]) * 8.0 - (up2[(i, k)] - um2[(i, k)]);
                acc += du * u[(j, k)].conj();
            }
            g[(i, j)] = Complex64::new(0.0, 1.0) * acc / (12.0 * H);
        }
    }
    let terms = if dim == 2 {
        pauli_decompose_1q(&[[g[(0, 0)], g[(0, 1)]], [g[(1, 0)], g[(1, 1)]]])
    } else {
        pauli_decompose_2q(&g)
    };
    if terms.is_empty() {
        None
    } else {
        Some(terms)
    }
}

/// One gradient contribution of one gate: the parameter it differentiates
/// (index into `param_names`), its generator, and the chain-rule factor
/// from the parameter expression (1.0 for a plain variable).
struct Contribution {
    idx: usize,
    terms: Vec<GenTerm>,
    scale: f64,
}

/// Chain-rule factor ∂f/∂v of a single-variable parameter expression.
/// Central difference on the expression tree itself — exact for the affine
/// forms the IR emits (θ/2, 2·φ, θ+π/4), O(h²) otherwise.
fn expr_scale(
    p: &Parameter,
    var: &str,
    param_values: &std::collections::HashMap<String, f64>,
) -> Result<f64, MethodError> {
    let v = *param_values.get(var).ok_or_else(|| {
        MethodError(format!(
            "no value provided for parameter(s): {var}. Pass all parameter \
             values via param_values= (the dag itself stays unbound)"
        ))
    })?;
    let h = 1e-6 * v.abs().max(1.0);
    let at = |x: f64| -> Option<f64> {
        let mut m = param_values.clone();
        m.insert(var.to_string(), x);
        p.bind(&m).try_evaluate()
    };
    match (at(v + h), at(v - h)) {
        (Some(hi), Some(lo)) => Ok((hi - lo) / (2.0 * h)),
        _ => Err(MethodError(format!(
            "adjoint: cannot differentiate parameter expression of `{var}`; \
             the expression must depend on that variable alone"
        ))),
    }
}

/// Plan every gradient contribution of one gate.
///
/// A `Variable` slot contributes with scale 1, a single-variable `Expr`
/// with its chain-rule factor, a `Const` not at all.  A slot whose
/// generator cannot be resolved — including an expression over more than
/// one free variable, where no per-variable gradient exists — is an error:
/// that is the invariant that makes the 0.1.23 silent-zero class of bug
/// unrepeatable.
fn plan_contributions(
    orig: &OpType,
    bound: &OpType,
    param_idx: &std::collections::HashMap<&str, usize>,
    param_values: &std::collections::HashMap<String, f64>,
) -> Result<Vec<Contribution>, MethodError> {
    let mut out: Vec<Contribution> = Vec::new();
    for (slot, p) in orig.parameters().into_iter().enumerate() {
        let (idx, scale) = match p {
            Parameter::Const(_) => continue,
            Parameter::Variable { name, .. } => match param_idx.get(name.as_str()) {
                Some(&i) => (i, 1.0),
                None => continue,
            },
            Parameter::Expr(_) => {
                let mut names = p.variable_names();
                names.sort();
                names.dedup();
                match names.len() {
                    0 => continue,
                    1 => match param_idx.get(names[0].as_str()) {
                        Some(&i) => (i, expr_scale(p, &names[0], param_values)?),
                        None => continue,
                    },
                    n => {
                        return Err(MethodError(format!(
                            "adjoint: parameter expression of `{}` slot {} depends on {} \
                             free variables; a gradient is only defined per variable — \
                             split it into one expression per variable",
                            op_label(bound),
                            slot,
                            n
                        )))
                    }
                }
            }
        };
        let terms = closed_form_generator(bound, slot)
            .or_else(|| finite_difference_generator(bound, slot))
            .ok_or_else(|| {
                MethodError(format!(
                    "adjoint: no generator for `{}` slot {}; add it to the generator \
                     table (`closed_form_generator` / `op_with_slot`) in sf-ir/src/adjoint.rs",
                    op_label(bound),
                    slot
                ))
            })?;
        out.push(Contribution { idx, terms, scale });
    }
    Ok(out)
}

/// ⟨φ|P|src⟩ for one Pauli term over `qubits` (local ids, 0 = identity).
///
/// Identity factors are stripped so the hot path keeps the allocation-free
/// `gen_dot`; an all-identity term degenerates to the plain inner product.
fn term_dot(
    phi: &[Complex64],
    src: &[Complex64],
    paulis: &[u8],
    qubits: &[usize],
    n_qubits: usize,
    par: bool,
    scratch: &mut Vec<Complex64>,
) -> Complex64 {
    let mut sub = [PAULI_I; 4];
    let mut qsub = [0usize; 4];
    let mut n = 0usize;
    for (i, &p) in paulis.iter().enumerate() {
        if p != PAULI_I && i < qubits.len() && n < sub.len() {
            sub[n] = p;
            qsub[n] = qubits[i];
            n += 1;
        }
    }
    if n == 0 {
        return phi.iter().zip(src.iter()).map(|(a, b)| a.conj() * b).sum();
    }
    if let Some(ip) = gen_dot(phi, src, &sub[..n], &qsub[..n], n_qubits, par) {
        return ip;
    }
    if scratch.is_empty() {
        *scratch = vec![Complex64::new(0.0, 0.0); phi.len()];
    }
    scratch.copy_from_slice(src);
    for i in 0..n {
        apply_pauli_1q_inplace(scratch, sub[i], qsub[i], n_qubits);
    }
    phi.iter()
        .zip(scratch.iter())
        .map(|(a, b)| a.conj() * b)
        .sum()
}

// ─── In-place gate application (no allocation) ─────────────────────────

/// States at or above this dimension (n >= 14) split their sweeps across the
/// rayon pool; below it the serial kernels win (and the parallel paths are
/// skipped entirely when the pool is pinned to one thread).
///
/// This sat at `1 << 17` while the gate was **unresolved**: paired A/B runs
/// then read n=16 as 1.40x/1.49x *for* threads in one window and 1.03x/0.93x
/// in the next, with the serial arm reproducing and the threaded one moving by
/// a third.  The cause turned out to be structural, not thermal — each gate's
/// chunk job was dispatched into rayon's *global* pool from the calling
/// thread, so every one of the 79 passes at (16,1) had to wake parked workers.
/// Running the whole call inside a cached pool ([`adj_pool_for`]) removed that
/// cost, and the per-phase trace now reproduces in the same direction every
/// window: `def` (serial) is bracketed by two `def` legs at n=14 (3.05, 3.05
/// vs 2.80 threaded) and n=16 (15.35, 15.65 vs 13.95).  ~1.10x for threads, at
/// both sizes, with the back-walk's own phase dropping 2.55 -> 1.70 ms.  The
/// value now matches `sf-cat`'s own 1 << 14 gate.
const PAR_DIM_DEFAULT: usize = 1 << 14;

/// Parallel/serial crossover dimension for the adjoint sweeps.
/// `SF_ADJ_PAR_MIN` overrides so any n can be A/B-attributed on-host under the
/// bench thread pin without a rebuild.  Default `1<<14`: n >= 14.
#[inline]
fn adj_par_min() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("SF_ADJ_PAR_MIN")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(PAR_DIM_DEFAULT)
    })
}

/// `SF_ADJ_CACHE=1` forces the cached reverse sweep, `=0` forces the streaming
/// one; unset leaves the decision to [`adj_stream_worth_it`].
///
/// Both paths compute the same ψ_k — the cached one stores it in the forward
/// sweep, the streaming one un-applies U_k and re-applies it — so this is a
/// timing switch, not a semantics switch. That is also what makes
/// `SF_ADJ_CACHE=0` a usable parity gate: the whole gradient test suite run
/// under it exercises the streaming arithmetic on every gate type.
#[inline]
fn adj_cache_forced() -> Option<bool> {
    static V: std::sync::OnceLock<Option<bool>> = std::sync::OnceLock::new();
    *V.get_or_init(|| match std::env::var("SF_ADJ_CACHE").as_deref() {
        Ok("0") => Some(false),
        Ok("1") => Some(true),
        _ => None,
    })
}

/// `SF_ADJ_FUSE=0` turns the fused reverse kernel off; on by default.
///
/// **What the fused arm is worth, measured at the phase it changes.**  At
/// (16,1) with `SF_ADJ_TRACE=1`, five calls per arm, medians (min):
///
/// | phase | split | fused |
/// |---|---|---|
/// | `fwd` | 6.90 (6.50) | 7.60 (6.20) |
/// | `dot` | 4.00 (3.20) | 4.65 (3.70) |
/// | `bw_phi` | **1.80 (1.70)** | **0.50 (0.50)** |
///
/// The knob does exactly what it says: the separate φ pass disappears for the
/// 32 one-qubit parameter gates and the fused kernel absorbs the apply into the
/// dot (`dot` +0.5..0.65 ms).  Net ~0.7-1.0 ms of a ~14 ms call — 5-8%, real
/// but small, and `fwd` moves 0.7 ms between the two arms on code the knob does
/// not touch, which is this host's phase noise.
///
/// **A retraction.**  An earlier order-balanced, arm-alternating run of this
/// A/B read 1.63-1.68x in favour of fusion at (16,1) (split 0.469 vs fused
/// 0.788 on `cat/sf_ir`).  That was wrong: `sf-cat` itself is bimodal on this
/// host at (16,1) — 9.0-9.6 ms in some windows, 18-19 ms in others — and the
/// paired ratio inherits cat's mode, not the code's.  At `--reps 15` the *same*
/// split/fused pair read 1.157 vs 1.883 (cat in the slow mode), while
/// `sf_ir`'s own warm time was 12.8 ms in both arms.  The phase numbers above
/// are the evidence; the ratio was a cat-mode detector.
///
/// The default therefore moved on the phase measurement alone, and the
/// end-to-end paired ladder does **not** resolve it: 0.726 at (16,1) with
/// fusion against 0.719-0.727 without.  Gradients are bit-identical either
/// way (`max|dg|` 0 .. 4.8e-15 on the same runs).
#[inline]
fn adj_fuse_on() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| match std::env::var("SF_ADJ_FUSE") {
        Ok(v) => !matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "0" | "off" | "no" | "false" | "none"
        ),
        Err(_) => true,
    })
}

/// Should the reverse sweep recompute ψ_k instead of caching it?
///
/// **Re-measured on build #27 and inverted.**  `_adj_cache_ab.py` (same ABBA
/// protocol, `reps` 7, one process per leg, cat in the same process as the
/// drift control, gradients compared elementwise):
///
/// | row | cat | cached | stream | paired cached/stream |
/// |---|---|---|---|---|
/// | 14,1 | 2.223 ms | 3.356 ms | 2.594 ms | 1.17, 1.43 |
/// | 16,1 | 10.333 ms | 14.175 ms | 7.625 ms | 2.18, 1.60 |
/// | 16,2 | 20.306 ms | 25.201 ms | 14.405 ms | 1.90, 1.61 |
/// | 18,1 | 90.036 ms | 91.950 ms | 75.051 ms | 1.20, 1.26 |
/// | 20,1 | 470.962 ms | 559.968 ms | 556.092 ms | 1.02, 1.00 |
///
/// Every ratio is above 1: the streaming sweep is now 1.2-2.2x *faster* than
/// the cached one at every row where the choice exists, and `|dg|` is exactly
/// 0.0 in all rows.  The 20,1 row is the control — the 512 MB cap forces both
/// legs onto the streaming code there, so 1.02/1.00 is this host's floor.
///
/// This reverses REPORT 46.17's rule ("cached wins every row where it is
/// legal").  That measurement was taken before builds #25-#27 rewrote both
/// sweeps: replaying its n=18 row, the cached arm reproduces (145.95 ms then,
/// 91.95 ms now) while the streaming arm went 216.44 -> 75.05 ms.  Both arms
/// improved, the recompute side by more, so the ranking inverted and the old
/// numbers cannot be used to defend the old default.
///
/// The small rows put the crossover at n=12 ([[`STREAM_MIN_DIM`]]), measured
/// the same way with `reps` 15: at n=8 0.133 vs 0.126 (wash), n=10 cached
/// 0.284 vs 0.434 stream (0.64, 0.67 — cached 1.5x), n=11 0.431 vs 0.702
/// (0.50, 0.79), n=12 1.454 vs 1.170 (1.11, 1.41), n=12 L=2 2.151 vs 1.435
/// (1.18, 1.80), n=13 3.446 vs 2.850 (1.33, 1.10).  Below the crossover the
/// recompute's two extra passes over a state that already lives in L1 cost
/// more than the cache traffic they save.
///
/// `SF_ADJ_CACHE=1` still forces the cached sweep (`adj_cache_forced`), which
/// is how the A/B above and `SF_ADJ_PP` remain reachable.
fn adj_stream_worth_it(_n_param_gates: usize, _n_gates: usize, dim: usize) -> bool {
    dim >= STREAM_MIN_DIM
}

/// State length (in amplitudes) from which recomputing beats caching.  The
/// measured boundary sits between n=11 and n=12; every row with `dim` at or
/// above this used the streaming sweep in `_adj_cache_ab.py` below n=14.
const STREAM_MIN_DIM: usize = 1 << 12;

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

    /// Gradient workspace (`psi`, `phi`, `term_state`) retained across calls
    /// so their pages stay resident.  A fresh `3 · 2^n · 16 B` allocation
    /// re-faults every page on each `adjoint_grad` call — a real fixed cost
    /// across the benchmark's repeated calls, largest at big n.  Complex64 is
    /// Copy with no Drop, so the take/put-back round trip is plain memory
    /// ownership (the buffers are zero-filled, not reallocated, on take).
    #[allow(clippy::missing_const_for_thread_local)]
    static ADJ_WORK_POOL: std::cell::RefCell<Vec<Vec<Complex64>>> =
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
///
/// Gated by the same `par` flag as the sweeps.  The forward sweep copies a
/// whole state into each parameter gate's cache slot, and `SF_ADJ_TRACE` puts
/// that copy at `fwdcpy` 1.30 / 5.95 / 60.7 ms against `fwd` 2.80 / 10.05 /
/// 104.0 ms at n = 14 / 16 / 18 — i.e. 46-59% of the forward sweep, which is
/// far more than a memcpy of the same bytes should cost.  A separate, much
/// higher parallel gate (2^20 elements, so these copies run serial) was tried
/// on the theory that the excess was rayon fork/join.  It was not: with
/// `SF_ADJ_CPY_MIN=16384` (parallel, the gate below) against unset (serial)
/// the copy reads 1.100 vs 1.250 ms at n=14, 6.300 vs 5.950 at n=16 and
/// 38.550 vs 40.950 at n=18 — the same within noise, parallel ahead at two of
/// the three sizes.  The cost is memory traffic (32 MiB at n=16, 144 MiB at
/// n=18 of destination writes per call), not dispatch, so the extra gate was
/// reverted rather than kept as a knob that buys nothing.  Removing the copy
/// itself was tried and does not pay either — `adj_pp_on` has that table.
#[inline]
fn copy_slice_par(dst: &mut [Complex64], src: &[Complex64], par: bool) {
    if par {
        let threads = rayon::current_num_threads().max(1);
        let chunk = ((src.len() / (4 * threads)).max(4096)).next_power_of_two();
        dst.par_chunks_mut(chunk)
            .zip(src.par_chunks(chunk))
            .for_each(|(d, s)| d.copy_from_slice(s));
    } else {
        dst.copy_from_slice(src);
    }
}

/// Should the cached forward sweep write each gate's post-state straight into
/// its cache slot instead of applying in place and then copying?
///
/// Measured, and **off by default**.  `SF_ADJ_TRACE` per-phase medians on one
/// binary, seven repetitions per arm, each size run in both arm orders
/// (`_adj_nt_abba2.txt`); `NT` is `SF_ADJ_NT`, and every PP=1 arm has
/// `fwdcpy` = 0 exactly:
///
/// | n  | PP=0 (apply + copy) | PP=1 NT=0 | PP=1 NT=slots | PP=1 NT=all |
/// |----|---------------------|-----------|----------------|-------------|
/// | 14 | 1.45 / 1.60 ms | 1.40 / 1.10 | 1.35 / 1.40 | 1.45 / 1.70 |
/// | 16 | 7.10 / 6.85 ms | 7.75 / 7.15 | 8.40 / 7.90 | 10.10 / 9.60 |
/// | 18 | 41.50 / 50.75 ms | 59.65 / 48.50 | 40.10 / 47.95 | 50.10 / 49.20 |
///
/// The copy disappears exactly as intended and the forward sweep does not get
/// reliably faster.  At n=16 the ping-pong arm loses in both orders (the
/// working set doubles and the state already fits in cache, so there was no
/// DRAM traffic to save); at n=18 `NT=slots` is ahead in both orders by 3-5%,
/// which is smaller than the 22% drift the PP=0 arm itself shows between the
/// two orders.  The host cannot resolve it, so the arm stays one env var away.
///
/// What the work did establish, in `simd`: writing a post-state into a cold
/// slot costs a read-for-ownership of those lines, and `vmovntpd`
/// (`pair_pass_out_nt` / `cnot_out_vec`) removes it — but only where the slot
/// really is cold, which on this host means n=18: `NT=slots` beats `NT=0` in
/// both orders there (40.10 / 47.95 against 59.65 / 48.50), is *behind* it in
/// both orders at n=16 (8.40 / 7.90 against 7.75 / 7.15) and a wash at n=14.
/// Why it hurts at n=16 is not established: the stores are 32 bytes against
/// this CPU's 64-byte line, so a partial-line write-combine can cost more than
/// the RFO it was meant to save, and that is a hypothesis, not a measurement.
/// `NT=all` — streaming the working buffer a gate has just produced, which the
/// next gate reads back immediately — is the worst arm at n=14/16 and behind
/// `NT=slots` at n=18, as its doc predicted.
///
/// The same pass found the CNOT dst/src path was still a scalar gather;
/// `cnot_out_vec` is now vectorised in both store modes.  That defect was real
/// by construction (17 CNOTs per layer over 2^18 amplitudes, one scalar load
/// and store each), but its size is **not** isolated here: the PP=0 control arm
/// moved 58.65 / 48.65 -> 41.50 / 50.75 between the two ABBA runs with no code
/// change touching it, so the window drift is the same order as any kernel
/// gain and this table cannot apportion it.
fn adj_pp_on() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| match std::env::var("SF_ADJ_PP") {
        Ok(v) => !matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "0" | "off" | "no" | "false" | "none"
        ),
        Err(_) => false,
    })
}

/// Which writes of the ping-pong forward take the non-temporal store path.
///
/// A streaming store skips the read-for-ownership of a cold destination line,
/// which is exactly the cost `adj_pp_on` documents.  It is only right where the
/// line is not read soon after: a parameter gate's slot is read once, much
/// later, by the backward pass, whereas a ping-pong working buffer is read by
/// the very next gate — streaming that one trades an L3 hit for a DRAM trip.
/// `SF_ADJ_NT=0|slots|all` selects; only meaningful with `SF_ADJ_PP=1`.
#[derive(Clone, Copy, PartialEq, Eq)]
enum NtMode {
    Off,
    Slots,
    All,
}

fn adj_nt_mode() -> NtMode {
    static V: std::sync::OnceLock<NtMode> = std::sync::OnceLock::new();
    *V.get_or_init(|| match std::env::var("SF_ADJ_NT") {
        Ok(v) => match v.trim().to_ascii_lowercase().as_str() {
            "0" | "off" | "no" | "false" | "none" => NtMode::Off,
            "all" => NtMode::All,
            _ => NtMode::Slots,
        },
        Err(_) => NtMode::Slots,
    })
}

/// Borrow two distinct `dim`-sized buffers of the state cache as (dst, src),
/// so an out-of-place kernel can read one and write the other.
///
/// The two indices are always different by construction: a param gate's slot
/// is written once and never used as a working buffer, and the working
/// buffers are chosen as "whichever one is not `cur`".
#[inline]
fn two_bufs(
    cache: &mut [Complex64],
    dim: usize,
    dst: usize,
    src: usize,
) -> (&mut [Complex64], &[Complex64]) {
    debug_assert_ne!(dst, src);
    if dst < src {
        let (lo, hi) = cache.split_at_mut(src * dim);
        (&mut lo[dst * dim..(dst + 1) * dim], &hi[..dim])
    } else {
        let (lo, hi) = cache.split_at_mut(dst * dim);
        (&mut hi[..dim], &lo[src * dim..(src + 1) * dim])
    }
}

/// dst = U·src for a 1-qubit gate (`out_lo = m00·lo + m01·hi`,
/// `out_hi = m10·lo + m11·hi` over every (i, i+stride) pair).
///
/// Formula-identical to `simd::pair_pass`; `dst` and `src` must not overlap and
/// their lengths must be equal multiples of `2·stride`.  With `nt` the chunk
/// kernels take `simd::pair_pass_out_nt` first and fall back per chunk when a
/// chunk's destination is not 32-byte aligned.
#[inline]
fn pair_pass_out(
    dst: &mut [Complex64],
    src: &[Complex64],
    stride: usize,
    m: [[Complex64; 2]; 2],
    par: bool,
    nt: bool,
) {
    let block = stride << 1;
    // Vectorised per chunk (`simd::pair_pass_out`): the scalar form of this
    // loop was measured at 2.6x the in-place AVX2 apply at n=18, which ate
    // the whole saving from removing the copy.
    let pass = |d: &mut [Complex64], s: &[Complex64]| {
        if nt && crate::simd::pair_pass_out_nt(d, s, stride, m) {
            return;
        }
        crate::simd::pair_pass_out(d, s, stride, m);
    };
    if par {
        let threads = rayon::current_num_threads().max(1);
        let want = (dst.len() / (4 * threads)).max(block);
        let chunk = (want / block).max(1) * block;
        if chunk < dst.len() {
            dst.par_chunks_mut(chunk)
                .zip(src.par_chunks(chunk))
                .for_each(|(d, s)| pass(d, s));
            return;
        }
    }
    pass(dst, src);
}

/// dst = CNOT(q1 -> q2)·src: `dst[i] = src[i ^ m2]` where bit q1 is set,
/// `src[i]` otherwise.  Pure amplitude moves, so bit-identical to the in-place
/// swap kernel (`dag::inplace_cnot`).
#[inline]
fn cnot_out(dst: &mut [Complex64], src: &[Complex64], q1: usize, q2: usize, par: bool, nt: bool) {
    let m1 = 1usize << q1;
    let m2 = 1usize << q2;
    // Smallest power-of-two block covering both qubits (NOT `(m1|m2) << 1`,
    // which is 6 rather than 4 for adjacent qubits) keeps every gather
    // cache-resident and every index inside its own chunk.
    let block = m1.max(m2) << 1;
    let pass = |d: &mut [Complex64], s: &[Complex64]| {
        // Vectorised in both store modes: the scalar gather below was the
        // ping-pong arm's largest non-memory cost (it moves one amplitude at a
        // time), so it is not used when the platform can take the kernel.
        if crate::simd::cnot_out_vec(d, s, q1, q2, nt) {
            return;
        }
        for (db, sb) in d.chunks_exact_mut(block).zip(s.chunks_exact(block)) {
            for (i, v) in db.iter_mut().enumerate() {
                *v = if (i & m1) != 0 { sb[i ^ m2] } else { sb[i] };
            }
        }
    };
    if par {
        let threads = rayon::current_num_threads().max(1);
        let want = (dst.len() / (4 * threads)).max(block);
        let chunk = (want / block).max(1) * block;
        if chunk < dst.len() {
            dst.par_chunks_mut(chunk)
                .zip(src.par_chunks(chunk))
                .for_each(|(d, s)| pass(d, s));
            return;
        }
    }
    pass(dst, src);
}

fn apply_2q_inplace(
    state: &mut [Complex64],
    gate: [[Complex64; 4]; 4],
    q1: usize,
    q2: usize,
    n_qubits: usize,
) {
    let dim = 1usize << n_qubits;
    let use_par = dim >= adj_par_min() && rayon::current_num_threads() > 1;
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

/// 2-qubit apply that picks the structured kernel for permutation gates.
///
/// `inplace_2q_general` is the right kernel for a genuinely dense 2-qubit
/// unitary, but it runs it for *every* 2-qubit gate — including `CNOT`, which
/// is 15 of the 47 gates at (16,1) on the campaign ladder and costs 16 complex
/// mult-adds per 4-amplitude group there, against zero arithmetic for a swap.
/// The generic form's coefficients are exactly 0 and 1 for `CNOT`, so
/// `1·v + 0·w` reproduces the swapped amplitude bit-for-bit (signed zeros
/// aside) and the swap kernel below is not an approximation.
#[inline]
fn apply_2q_typed_inplace(
    state: &mut [Complex64],
    gate: [[Complex64; 4]; 4],
    q1: usize,
    q2: usize,
    n_qubits: usize,
    is_cnot: bool,
) {
    if is_cnot {
        let dim = 1usize << n_qubits;
        let use_par = dim >= adj_par_min() && rayon::current_num_threads() > 1;
        crate::dag::inplace_cnot(state, q1, q2, use_par);
        return;
    }
    apply_2q_inplace(state, gate, q1, q2, n_qubits);
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
    // AVX2 fast path: when the generator qubit gives long contiguous runs
    // (`m` >= the dot floor) each half-run is a whole-slice `dot_conj`, which
    // vectorises the multiply-accumulate reduction. X and Z are plain
    // conj-dots; Y is the same conj-dot rotated by the exact ±i generator
    // phase. Below the floor (small qubit index) the scalar loops below stay
    // on the elementwise reference path.
    #[cfg(target_arch = "x86_64")]
    if crate::simd::simd_enabled() && m >= crate::simd::dot_min() {
        let mut base = lo;
        while base < hi {
            let (phi_lo, phi_hi) = (&phi[base..base + m], &phi[base + m..base + block]);
            let (src_lo, src_hi) = (&src[base..base + m], &src[base + m..base + block]);
            match p {
                1 => {
                    ip += crate::simd::dot_conj(phi_lo, src_hi);
                    ip += crate::simd::dot_conj(phi_hi, src_lo);
                }
                3 => {
                    ip += crate::simd::dot_conj(phi_lo, src_lo);
                    ip -= crate::simd::dot_conj(phi_hi, src_hi);
                }
                2 => {
                    let c1 = crate::simd::dot_conj(phi_lo, src_hi);
                    ip += Complex64::new(c1.im, -c1.re);
                    let c2 = crate::simd::dot_conj(phi_hi, src_lo);
                    ip += Complex64::new(-c2.im, c2.re);
                }
                _ => {}
            }
            base += block;
        }
        return ip;
    }
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

/// ⟨φ|G|src⟩ **and** φ ← U†φ in one traversal, for a single-qubit generator
/// Pauli (1=X, 2=Y, 3=Z) with qubit mask `m`.
///
/// The split form — [`dot_one_pauli`] then `apply_1q_inplace` — reads φ twice
/// and `src` once per parameter gate; this reads each exactly once, which is
/// the difference between two passes and one over a state that no longer fits
/// L3.  Above the SIMD floor the two halves of a block are reduced with the
/// same `dot_conj` calls in the same ascending-block order as the split path
/// (so the dot is bit-identical to it) and the 2×2 update then runs on the
/// block while it is still L1-resident.  Below the floor the interleaved
/// scalar walk of `sf-cat`'s `dot_and_apply` is used: identical summands,
/// different grouping (~1 ulp).
#[inline]
fn dot_and_apply_1q(
    p: u8,
    phi: &mut [Complex64],
    src: &[Complex64],
    m: usize,
    u: [[Complex64; 2]; 2],
) -> Complex64 {
    let dim = phi.len();
    let block = m << 1;
    let mut ip = Complex64::new(0.0, 0.0);
    // Wide generator qubits: long contiguous half-runs, so the dot keeps the
    // vectorised reductions and the block is updated while it is hot.
    #[cfg(target_arch = "x86_64")]
    if crate::simd::simd_enabled() && m >= crate::simd::dot_min() {
        let mut base = 0;
        while base < dim {
            {
                let (phi_lo, phi_hi) = (&phi[base..base + m], &phi[base + m..base + block]);
                let (src_lo, src_hi) = (&src[base..base + m], &src[base + m..base + block]);
                match p {
                    1 => {
                        ip += crate::simd::dot_conj(phi_lo, src_hi);
                        ip += crate::simd::dot_conj(phi_hi, src_lo);
                    }
                    3 => {
                        ip += crate::simd::dot_conj(phi_lo, src_lo);
                        ip -= crate::simd::dot_conj(phi_hi, src_hi);
                    }
                    2 => {
                        let c1 = crate::simd::dot_conj(phi_lo, src_hi);
                        ip += Complex64::new(c1.im, -c1.re);
                        let c2 = crate::simd::dot_conj(phi_hi, src_lo);
                        ip += Complex64::new(-c2.im, c2.re);
                    }
                    _ => {}
                }
            }
            crate::simd::pair_pass_chunk(&mut phi[base..base + block], m, u);
            base += block;
        }
        return ip;
    }
    let mut base = 0;
    while base < dim {
        for k in 0..m {
            let lo = phi[base + k];
            let hi = phi[base + k + m];
            let (sl, sh) = (src[base + k], src[base + k + m]);
            match p {
                1 => {
                    ip += lo.conj() * sh;
                    ip += hi.conj() * sl;
                }
                3 => {
                    ip += lo.conj() * sl;
                    ip -= hi.conj() * sh;
                }
                2 => {
                    ip += lo.conj() * Complex64::new(sh.im, -sh.re);
                    ip += hi.conj() * Complex64::new(-sl.im, sl.re);
                }
                _ => {}
            }
            phi[base + k] = u[0][0] * lo + u[0][1] * hi;
            phi[base + k + m] = u[1][0] * lo + u[1][1] * hi;
        }
        base += block;
    }
    ip
}

/// [`dot_and_apply_1q`] across the rayon pool, chunked on whole `2m` blocks.
///
/// The chunk geometry and the ascending partial-sum order are copied from
/// [`gen_dot`]'s single-Pauli arm, so the split and fused forms group the
/// summands identically.
#[inline]
fn par_dot_and_apply_1q(
    p: u8,
    phi: &mut [Complex64],
    src: &[Complex64],
    m: usize,
    u: [[Complex64; 2]; 2],
    par: bool,
) -> Complex64 {
    let dim = phi.len();
    let block = m << 1;
    if par {
        let threads = rayon::current_num_threads().max(1);
        let want = (dim / (4 * threads)).max(block);
        let chunk = (want / block).max(1) * block;
        if chunk < dim {
            let partials: Vec<Complex64> = phi
                .par_chunks_mut(chunk)
                .zip(src.par_chunks(chunk))
                .map(|(pc, sc)| dot_and_apply_1q(p, pc, sc, m, u))
                .collect();
            let mut ip = Complex64::new(0.0, 0.0);
            for q in &partials {
                ip += *q;
            }
            return ip;
        }
    }
    dot_and_apply_1q(p, phi, src, m, u)
}

/// `Some(pauli)` when a gate's gradient is exactly one generator term of
/// weight one — the shape every `Rx`/`Ry`/`Rz` lowers to, and the only one
/// whose dot can share a traversal with the daggered φ update.  Two-qubit
/// generators, `Cu`'s four-angle sum, expression-scaled and multi-term
/// generators all keep the general split path.
fn single_pauli_generator(contribs: &[Contribution]) -> Option<u8> {
    if contribs.len() != 1 || contribs[0].terms.len() != 1 {
        return None;
    }
    match contribs[0].terms[0].paulis() {
        [p @ 1..=3] => Some(*p),
        _ => None,
    }
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
        // Allocation-free first: `to_matrix` wraps these same entries in a
        // heap-allocated `DMatrix`, and this pass asks for the unitary *and*
        // the dagger of every gate on every call.
        let mut a = [Complex64::new(0.0, 0.0); 16];
        if let Some(2) = self.op_type.write_matrix_small(&mut a) {
            return [[a[0], a[1]], [a[2], a[3]]];
        }
        let m = self.op_type.to_matrix();
        [[m[(0, 0)], m[(0, 1)]], [m[(1, 0)], m[(1, 1)]]]
    }

    fn unitary_4x4(&self) -> [[Complex64; 4]; 4] {
        let mut a = [Complex64::new(0.0, 0.0); 16];
        if let Some(4) = self.op_type.write_matrix_small(&mut a) {
            return [
                [a[0], a[1], a[2], a[3]],
                [a[4], a[5], a[6], a[7]],
                [a[8], a[9], a[10], a[11]],
                [a[12], a[13], a[14], a[15]],
            ];
        }
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
    // Install the whole call into one persistent pool (see `adj_pool_for`)
    // instead of dispatching every gate's chunk job into the *global* pool
    // from the calling thread: at (16,1) that is 79 dispatches per call, each
    // of which has to wake parked workers, and it is why `SF_ADJ_PAR_MIN=1024`
    // used to make the forward sweep slower than serial (7.3 ms -> 15.4 ms).
    if (1usize << dag.n_qubits) >= adj_par_min() && rayon::current_num_threads() > 1 {
        if let Some(p) = adj_pool_for(rayon::current_num_threads()) {
            return p.install(|| adjoint_grad_body(dag, observable_terms, param_values));
        }
    }
    adjoint_grad_body(dag, observable_terms, param_values)
}

/// One dedicated pool per requested width, built once and kept alive, the way
/// `sf-cat`'s adjoint handle does it.  Rayon parks idle workers after a short
/// spin; a sweep that re-enters the pool once per gate pays that wake-up 79
/// times per call at (16,1), which measured as *worse* than serial.  A pool
/// installed for the whole call keeps the workers hot across gates and (because
/// the pool is cached) across calls.
fn adj_pool_for(w: usize) -> Option<std::sync::Arc<rayon::ThreadPool>> {
    if w <= 1 {
        return None;
    }
    type Pools = std::sync::Mutex<Vec<(usize, std::sync::Arc<rayon::ThreadPool>)>>;
    static POOLS: std::sync::OnceLock<Pools> = std::sync::OnceLock::new();
    let pools = POOLS.get_or_init(|| std::sync::Mutex::new(Vec::new()));
    let mut guard = pools.lock().ok()?;
    if let Some((_, p)) = guard.iter().find(|(n, _)| *n == w) {
        return Some(std::sync::Arc::clone(p));
    }
    let built = rayon::ThreadPoolBuilder::new()
        .num_threads(w)
        .thread_name(|i| format!("sf-ir-adj-{i}"))
        .build()
        .ok()?;
    let p = std::sync::Arc::new(built);
    guard.push((w, std::sync::Arc::clone(&p)));
    Some(p)
}

/// Body of [`adjoint_grad`], run inside [`adj_pool_for`]'s pool when the row
/// is wide enough to thread.
fn adjoint_grad_body(
    dag: &QuantumDAG,
    observable_terms: &[PauliTerm],
    param_values: &std::collections::HashMap<String, f64>,
) -> Result<AdjointGradResult, MethodError> {
    let n_qubits = dag.n_qubits;
    let dim = 1usize << n_qubits;
    // n >= 17 (see `PAR_DIM_DEFAULT`): sweeps may split across the rayon pool
    // (skipped entirely when the pool is pinned to one thread).
    let par = dim >= adj_par_min() && rayon::current_num_threads() > 1;

    // Phase timing (SF_ADJ_TRACE=1): attribute each call's cost to
    // setup / forward / observable / per-parameter kernel / back-walk.
    // Development instrumentation only — no behavior change.
    let adj_trace = std::env::var_os("SF_ADJ_TRACE").is_some();
    let t_pre = std::time::Instant::now();
    // Setup breakdown for `SF_ADJ_TRACE`: the fixed per-call cost is what the
    // adjoint pays that a compile-once handle would not — measured as the
    // whole of the call at n <= 10 (see `setup_split` in the trace line).
    // Eight `Instant` reads per call (~0.2 us) buy the attribution.
    let mut mark = std::time::Instant::now();
    let mut marks = [0.0f64; 8];
    macro_rules! stamp {
        ($i:expr) => {{
            marks[$i] = mark.elapsed().as_secs_f64() * 1e3;
            // The final stamp's restart is deliberately unread.
            #[allow(unused_assignments)]
            {
                mark = std::time::Instant::now();
            }
        }};
    }
    let mut t_pre_ms = 0.0f64;
    let mut t_fwd = 0.0f64;
    // Split out of `t_fwd`: the cache copy alone.  The forward sweep applies
    // each parameter gate in place and *then* copies the state into its cache
    // slot, which is one extra full traversal per parameter gate -- 32 of them
    // at (16,1).  `sf-cat` instead writes the post-state straight into the slot
    // from a dst/src kernel, so this number is the upper bound on what a
    // ping-pong forward could save there.
    let mut t_fwd_cpy = 0.0f64;
    let mut t_phi = 0.0f64;
    let mut t_ip_prep = 0.0f64;
    let t_ip_pauli = 0.0f64;
    let mut t_ip_dot = 0.0f64;
    let mut t_bw_psi = 0.0f64;
    let mut t_bw_phi = 0.0f64;

    // Per-op bind instead of `dag.bind`: `bind` clones the entire graph
    // (petgraph alloc + every node) only to run `bind_params` on each op, so
    // building the bound gate list straight from the DAG is bit-identical to
    // the historical path while dropping the wasted clone and one
    // topological walk. The op_types produced here are exactly the ones
    // `dag.bind(..)` would have handed to `ops`, so every downstream gate
    // matrix and the missing-parameter check below are unchanged.
    let order = dag.topological_order();
    stamp!(0);
    let ops: Vec<GateOp> = order
        .iter()
        .filter_map(|&node_id| {
            let op = &dag.graph()[node_id];
            if op.op_type.is_boundary()
                || op.op_type == OpType::Barrier
                || op.op_type.is_measurement()
            {
                return None;
            }
            Some(GateOp {
                op_type: op.op_type.bind_params(param_values),
                qubits: op.qubits.to_vec(),
            })
        })
        .collect();
    stamp!(1);

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
    stamp!(2);

    // Same node order as `ops` (bind does not change topology), so reuse the
    // single walk instead of a second `topological_order` traversal.
    let orig_order = &order;
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
    stamp!(3);

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
    stamp!(4);

    // Forward pass: evolve |0> → |psi_final>. When the cache cap allows
    // (M·dim·16 B ≤ 512 MB), store the state AFTER each gate so the
    // backward pass never clones or re-applies; above the cap, fall back
    // to the recompute-from-|0> variant.
    // Only parameterised gates need a cached post-state (ψ_after_k for the
    // gradient inner product) — the back-walk itself never reads the cache.
    // Caching just those states shrinks the buffer 25-50% on HE circuits
    // and keeps n=18 d=3 under the cap.
    // Plan every gradient contribution once, up front: which parameter
    // each gate slot differentiates (index + chain-rule factor) and what
    // its generator is.  The cache layout and the backward pass are both
    // derived from this one plan, so they cannot disagree — the pre-0.1.24
    // code re-derived the parameter name in each place and silently
    // dropped every gate the two helpers did not know about.
    let contribs: Vec<Vec<Contribution>> = ops
        .iter()
        .zip(orig_ops.iter())
        .map(|(bound, orig)| {
            plan_contributions(
                &orig.original_op_type,
                &bound.op_type,
                &param_idx,
                param_values,
            )
        })
        .collect::<Result<Vec<_>, MethodError>>()?;
    let mut param_slot: Vec<Option<usize>> = Vec::with_capacity(ops.len());
    let mut n_slots = 0usize;
    for c in &contribs {
        if c.is_empty() {
            param_slot.push(None);
        } else {
            param_slot.push(Some(n_slots));
            n_slots += 1;
        }
    }
    let cache_bytes = (n_slots as u64)
        .saturating_mul(dim as u64)
        .saturating_mul(16);
    stamp!(5);
    // Single contiguous cache: a Vec<Vec> allocates + zero-fills a fresh
    // psi.clone() per gate (measured as the forward pass's dominant cost at
    // n=16+). Every element is written before it is read (Complex64: Copy,
    // no Drop), so the uninit set_len is sound; a retained buffer simply
    // gets its stale bytes overwritten. The buffer is borrowed from the
    // thread-local pool and put back at the end, so repeated calls at the
    // same size reuse resident pages instead of re-faulting ~memory-size
    // in fresh allocations.
    let use_cache = cache_bytes <= 512 * 1024 * 1024
        && adj_cache_forced().unwrap_or_else(|| {
            // Streaming is only a candidate when it saves DRAM traffic; the
            // measured rule lives in `adj_stream_worth_it`.
            !adj_stream_worth_it(n_slots, ops.len(), dim)
        });
    let mut cache: Vec<Complex64> = ADJ_CACHE_BUF.with(|c| std::mem::take(&mut *c.borrow_mut()));
    // Two extra buffers when the forward sweep ping-pongs (see `adj_pp_on`):
    // the working state alternates between them, so every gate can read one
    // and write the other.
    let pp = use_cache && adj_pp_on();
    let nt = if pp { adj_nt_mode() } else { NtMode::Off };
    if use_cache {
        // One spare element when the forward sweep ping-pongs: the streaming
        // store kernels need a 32-byte aligned destination base, and a
        // `Vec<Complex64>` is only 8-byte aligned by contract (16 in
        // practice), so the cache view may have to start one element later
        // (`coff`).  Every buffer is `dim * 16` bytes further on and `dim` is a
        // power of two >= 4096, so that one shift aligns all of them at once.
        let need = (n_slots + if pp { 2 } else { 0 }) * dim + if pp { 1 } else { 0 };
        if cache.capacity() < need {
            cache = Vec::with_capacity(need);
        }
        unsafe { cache.set_len(need) };
    }
    // 0 for every caller that is not ping-ponging (the default), so the
    // streaming-store offsets below collapse to the original indices.
    let coff = if pp && (cache.as_ptr() as usize) & 31 != 0 {
        1
    } else {
        0
    };
    stamp!(6);

    // Workspace buffers come from the thread-local pool (see `ADJ_WORK_POOL`)
    // so their pages stay resident across calls; zero-filled here rather than
    // reallocated.  `psi` starts in |0…0⟩, `phi` accumulates O|psi⟩ from zero;
    // `term_state` is only read in the >2-Pauli fallback after a full copy.
    let (mut psi, mut phi, mut term_state) = ADJ_WORK_POOL.with(|p| {
        let mut pool = p.borrow_mut();
        while pool.len() < 3 {
            pool.push(Vec::new());
        }
        (
            std::mem::take(&mut pool[0]),
            std::mem::take(&mut pool[1]),
            std::mem::take(&mut pool[2]),
        )
    });
    let zero = Complex64::new(0.0, 0.0);
    psi.clear();
    psi.resize(dim, zero);
    psi[0] = Complex64::new(1.0, 0.0);
    stamp!(7);

    if adj_trace {
        t_pre_ms = t_pre.elapsed().as_secs_f64() * 1e3;
    }
    let t0 = std::time::Instant::now();
    if pp {
        // Ping-pong forward (see `adj_pp_on`).  The two buffers appended to
        // the cache are the only ones a gate writes in place; everything else
        // is written where it has to live, so a parameter gate's post-state
        // lands in its slot as it is computed and the copy pass disappears.
        //
        // The cache is re-viewed from `coff`: once per call rather than an
        // offset on every index, and it is what makes `simd::pair_pass_out_nt`
        // legal (32-byte aligned destinations).
        let cache: &mut [Complex64] = &mut cache[coff..coff + (n_slots + 2) * dim];
        let (a_buf, b_buf) = (n_slots, n_slots + 1);
        let abase = a_buf * dim;
        for x in &mut cache[abase..abase + dim] {
            *x = zero;
        }
        cache[abase] = Complex64::new(1.0, 0.0);
        // Buffer holding the state produced by the last applied gate.
        let mut cur = a_buf;
        for (k, gate_op) in ops.iter().enumerate() {
            let slot = param_slot[k];
            let nq = gate_op.qubits.len();
            if nq == 2 && gate_op.op_type != OpType::CNOT {
                // No dst/src kernel for a dense 2-qubit unitary: keep it in
                // place on a working buffer, then give the slot its copy.
                // This is the pre-ping-pong cost for that gate, unchanged.
                if cur != a_buf && cur != b_buf {
                    let (d, s) = two_bufs(cache, dim, a_buf, cur);
                    d.copy_from_slice(s);
                    cur = a_buf;
                }
                apply_2q_inplace(
                    &mut cache[cur * dim..(cur + 1) * dim],
                    fwd_matrices_2q[k].unwrap(),
                    gate_op.qubits[0],
                    gate_op.qubits[1],
                    n_qubits,
                );
                if let Some(sl) = slot {
                    let tc = std::time::Instant::now();
                    let (d, s) = two_bufs(cache, dim, sl, cur);
                    d.copy_from_slice(s);
                    if adj_trace {
                        t_fwd_cpy += tc.elapsed().as_secs_f64() * 1e3;
                    }
                    cur = sl;
                }
                continue;
            }
            // Destination: the gate's own slot, or whichever working buffer
            // `cur` is not — so dst never aliases src, and a cached slot is
            // written exactly once.
            let dst = match slot {
                Some(s) => s,
                None if cur == a_buf => b_buf,
                None => a_buf,
            };
            let (d, s) = two_bufs(cache, dim, dst, cur);
            // A streaming store skips the RFO of a cold destination line, but
            // leaves the line in DRAM: correct for a slot (read once, much
            // later), wrong for a working buffer the next gate reads at once.
            let want_nt = match nt {
                NtMode::Off => false,
                NtMode::Slots => slot.is_some(),
                NtMode::All => true,
            };
            match nq {
                1 => pair_pass_out(
                    d,
                    s,
                    1usize << gate_op.qubits[0],
                    fwd_matrices_1q[k].unwrap(),
                    par,
                    want_nt,
                ),
                2 => cnot_out(d, s, gate_op.qubits[0], gate_op.qubits[1], par, want_nt),
                // A gate the sweep does not apply: the post-state is the
                // pre-state, so only a parameter gate needs the copy.
                _ => {
                    if slot.is_some() {
                        d.copy_from_slice(s);
                    } else {
                        continue;
                    }
                }
            }
            cur = dst;
        }
        // The observable pass reads `psi`; one traversal here replaces a copy
        // per parameter gate.
        let fbase = cur * dim;
        copy_slice_par(&mut psi, &cache[fbase..fbase + dim], par);
    } else {
        for (k, gate_op) in ops.iter().enumerate() {
            match gate_op.qubits.len() {
                1 => apply_1q_inplace(
                    &mut psi,
                    fwd_matrices_1q[k].unwrap(),
                    gate_op.qubits[0],
                    n_qubits,
                    par,
                ),
                2 => apply_2q_typed_inplace(
                    &mut psi,
                    fwd_matrices_2q[k].unwrap(),
                    gate_op.qubits[0],
                    gate_op.qubits[1],
                    n_qubits,
                    gate_op.op_type == OpType::CNOT,
                ),
                _ => {}
            }
            if use_cache {
                if let Some(slot) = param_slot[k] {
                    let tc = std::time::Instant::now();
                    copy_slice_par(&mut cache[slot * dim..(slot + 1) * dim], &psi, par);
                    if adj_trace {
                        t_fwd_cpy += tc.elapsed().as_secs_f64() * 1e3;
                    }
                }
            }
        }
    }
    if adj_trace {
        t_fwd = t0.elapsed().as_secs_f64() * 1e3;
    }

    // Build phi = O|psi_final>
    let t0 = std::time::Instant::now();
    phi.clear();
    phi.resize(dim, zero);
    term_state.clear();
    term_state.resize(dim, zero);
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
    // is un-applied per gate. Recompute variant: psi walks backward too,
    // and — because it is un-applied one iteration *after* the dot reads it —
    // the state it carries at the top of the body is already psi_after_k.
    let mut grad = vec![0.0f64; n_params];
    // Scratch for the >2-pauli generator fallback (lazily allocated; the
    // fused path never touches it).  The recompute variant used to keep a
    // second work state here (`g_psi`, 2^n complex) to rebuild psi_after_k
    // from psi_k; `src` below reads psi in place instead, so the buffer has
    // no consumer and the 2^n allocation is gone with it.
    let mut scratch: Vec<Complex64> = Vec::new();

    for k in (0..ops.len()).rev() {
        let gate_op = &ops[k];

        // Recompute variant: at the top of this body `psi` holds psi_after_k
        // (one un-apply removed per iteration, at the bottom), so the
        // generator dot below reads it in place.  Until this change the body
        // un-applied first and then rebuilt psi_after_k with a full copy plus
        // a re-applied U_k — two extra state passes per parameter gate, and
        // the reason the over-cap variant ran ~5 passes/gate against ~3.  The
        // dot sees the same amplitudes in the same order, so the gradients are
        // bit-identical to that version.
        //
        // Gradient contributions: 2·scale·Im⟨φ|G_k|ψ_after_k⟩, summed
        // over the generator's Pauli terms and over every slot of this
        // gate (a shared parameter name on several gates, or several slots
        // of one gate, each add their own term).
        let t0 = std::time::Instant::now();
        // Set by the fused arm below: its kernel already applied the daggered
        // gate to φ, so the trailing back-walk must not apply it again.
        let mut fused = false;
        if !contribs[k].is_empty() {
            let src: &[Complex64] = if use_cache {
                // Invariant: `param_slot[k].is_some()` iff `contribs[k]`
                // is non-empty — both are derived from the same plan, so
                // the expect cannot fire.
                let slot = param_slot[k].expect("cached gradient gate has no cache slot");
                &cache[coff + slot * dim..coff + (slot + 1) * dim]
            } else {
                // Recompute variant: `psi` already is psi_after_k — no copy,
                // no re-apply.
                &psi[..]
            };
            if adj_trace {
                t_ip_prep += t0.elapsed().as_secs_f64() * 1e3;
            }
            let t0 = std::time::Instant::now();
            // Fused arm: for the single-Pauli generator shape one traversal
            // takes the generator dot and the daggered φ update together,
            // instead of a dot pass now and an apply pass at the end of the
            // body.  The dot still sees the pre-update φ (the kernel reads
            // each pair before writing it), so this is the same arithmetic in
            // one pass.
            //
            // In the **recompute** variant this is unconditional: there the
            // pass it removes is a whole 2^n state traversal per parameter
            // gate (dot reads φ+ψ, apply reads+writes φ → 3·2^n·16 B against
            // 2·2^n·16 B fused), which is what kept the over-cap rows behind
            // `cat` — at (20,1) sf-ir ran ~5 passes/gate against ~3.  The
            // cached variant saves the φ read+write of the same pass: the
            // phase trace puts `bw_phi` at 1.80 -> 0.50 ms and `dot` at
            // 4.00 -> 4.65 ms at (16,1), so `adj_fuse_on()` now defaults on;
            // `SF_ADJ_FUSE=0` restores the split kernel.  See that function's
            // doc comment for why the end-to-end ratio does not resolve it.
            if gate_op.qubits.len() == 1 && (adj_fuse_on() || !use_cache) {
                if let (Some(pauli), Some(u)) =
                    (single_pauli_generator(&contribs[k]), dag_matrices_1q[k])
                {
                    let ip = par_dot_and_apply_1q(
                        pauli,
                        &mut phi,
                        src,
                        1usize << gate_op.qubits[0],
                        u,
                        par,
                    );
                    let c = &contribs[k][0];
                    // Same assembly as the general arm: the generator's own
                    // coefficient (0.5 for Rx/Ry/Rz), then the chain-rule scale.
                    grad[c.idx] += 2.0 * c.scale * c.terms[0].coef * ip.im;
                    fused = true;
                }
            }
            if !fused {
                for c in &contribs[k] {
                    // `term_dot` applies the generator's Paulis inline (exact
                    // ±1/±i phases + index permutation) without materializing
                    // G|ψ⟩; only terms with three or more non-identity factors
                    // reach the scratch buffer.
                    let mut ip = Complex64::new(0.0, 0.0);
                    for term in &c.terms {
                        ip += term.coef
                            * term_dot(
                                phi.as_slice(),
                                src,
                                term.paulis(),
                                &gate_op.qubits,
                                n_qubits,
                                par,
                                &mut scratch,
                            );
                    }
                    grad[c.idx] += 2.0 * c.scale * ip.im;
                }
            }
            if adj_trace {
                t_ip_dot += t0.elapsed().as_secs_f64() * 1e3;
            }
        }

        // Back-walk phi
        let t0 = std::time::Instant::now();
        if !fused {
            match gate_op.qubits.len() {
                1 => apply_1q_inplace(
                    &mut phi,
                    dag_matrices_1q[k].unwrap(),
                    gate_op.qubits[0],
                    n_qubits,
                    par,
                ),
                2 => apply_2q_typed_inplace(
                    &mut phi,
                    dag_matrices_2q[k].unwrap(),
                    gate_op.qubits[0],
                    gate_op.qubits[1],
                    n_qubits,
                    gate_op.op_type == OpType::CNOT,
                ),
                _ => {}
            }
        }
        if adj_trace {
            t_bw_phi += t0.elapsed().as_secs_f64() * 1e3;
        }

        // Recompute variant: now that both the dot (above) and the phi
        // back-walk are done with psi, un-apply gate k so the next iteration
        // — k-1 — sees psi_{k} == psi_after_{k-1} at its top.
        if !use_cache {
            let t0 = std::time::Instant::now();
            match gate_op.qubits.len() {
                1 => apply_1q_inplace(
                    &mut psi,
                    dag_matrices_1q[k].unwrap(),
                    gate_op.qubits[0],
                    n_qubits,
                    par,
                ),
                2 => apply_2q_typed_inplace(
                    &mut psi,
                    dag_matrices_2q[k].unwrap(),
                    gate_op.qubits[0],
                    gate_op.qubits[1],
                    n_qubits,
                    gate_op.op_type == OpType::CNOT,
                ),
                _ => {}
            }
            if adj_trace {
                t_bw_psi += t0.elapsed().as_secs_f64() * 1e3;
            }
        }
    }

    if adj_trace {
        eprintln!(
            "SF_ADJ_TRACE n={} dim={} gates={} params={} variant={} setup={:.1}ms \
             fwd={:.1}ms fwdcpy={:.1}ms phi={:.1}ms ip[prep={:.1} pauli={:.1} dot={:.1}]ms \
             bw_psi={:.1}ms bw_phi={:.1}ms setup_split[order={:.3} ops={:.3} \
             miss={:.3} names={:.3} mats={:.3} contrib={:.3} cache={:.3} \
             work={:.3}]ms",
            n_qubits,
            dim,
            ops.len(),
            n_params,
            if use_cache { "cache" } else { "recompute" },
            t_pre_ms,
            t_fwd,
            t_fwd_cpy,
            t_phi,
            t_ip_prep,
            t_ip_pauli,
            t_ip_dot,
            t_bw_psi,
            t_bw_phi,
            marks[0],
            marks[1],
            marks[2],
            marks[3],
            marks[4],
            marks[5],
            marks[6],
            marks[7],
        );
    }

    ADJ_CACHE_BUF.with(|c| *c.borrow_mut() = cache);
    ADJ_WORK_POOL.with(|p| {
        let mut pool = p.borrow_mut();
        pool[0] = std::mem::take(&mut psi);
        pool[1] = std::mem::take(&mut phi);
        pool[2] = std::mem::take(&mut term_state);
    });

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

    // ─── Generator-table tests ─────────────────────────────────────────

    /// Every parametric `OpType` in `ops.rs`, with distinct angles so a
    /// slot mix-up cannot hide behind two equal values.
    fn parametric_ops() -> Vec<OpType> {
        let c = |v: f64| Parameter::Const(v);
        vec![
            OpType::Rx(c(0.3)),
            OpType::Ry(c(0.4)),
            OpType::Rz(c(0.5)),
            OpType::R1(c(0.6)),
            OpType::P(c(0.7)),
            OpType::Rxx(c(0.8)),
            OpType::Ryy(c(0.9)),
            OpType::Rzz(c(1.0)),
            OpType::CRx(c(1.1)),
            OpType::CRz(c(1.2)),
            OpType::CP(c(1.3)),
            OpType::U(c(0.31), c(0.32), c(0.33)),
            OpType::Cu(c(0.41), c(0.42), c(0.43), c(0.44)),
        ]
    }

    fn bound_values() -> std::collections::HashMap<String, f64> {
        let mut m = std::collections::HashMap::new();
        m.insert("theta".to_string(), 0.9);
        m.insert("phi".to_string(), 0.4);
        m
    }

    /// `G = i·(∂U/∂θ)·U†` computed straight from the gate matrix by a
    /// 5-point central difference — independent of the generator table,
    /// which is what makes the comparison a real check.
    fn reference_generator(op: &OpType, slot: usize) -> nalgebra::DMatrix<Complex64> {
        const H: f64 = 1e-4;
        let theta = op.parameters()[slot].try_evaluate().unwrap();
        let at = |d: f64| op_with_slot(op, slot, theta + d).unwrap().to_matrix();
        let (u, um1, up1, um2, up2) = (op.to_matrix(), at(-H), at(H), at(-2.0 * H), at(2.0 * H));
        let d = u.nrows();
        let mut g = nalgebra::DMatrix::<Complex64>::zeros(d, d);
        for i in 0..d {
            for j in 0..d {
                let mut acc = Complex64::new(0.0, 0.0);
                for k in 0..d {
                    let du = (up1[(i, k)] - um1[(i, k)]) * 8.0 - (up2[(i, k)] - um2[(i, k)]);
                    acc += du * u[(j, k)].conj();
                }
                g[(i, j)] = Complex64::new(0.0, 1.0) * acc / (12.0 * H);
            }
        }
        g
    }

    /// Rebuild a generator matrix from its Pauli expansion.
    fn expand_terms(terms: &[GenTerm], dim: usize) -> nalgebra::DMatrix<Complex64> {
        let mut m =
            nalgebra::DMatrix::<Complex64>::from_element(dim, dim, Complex64::new(0.0, 0.0));
        for t in terms {
            let local: Vec<M2> = t.paulis().iter().map(|&p| m2_pauli(p)).collect();
            for i in 0..dim {
                for j in 0..dim {
                    let v = if local.len() == 2 {
                        local[0][i >> 1][j >> 1] * local[1][i & 1][j & 1]
                    } else {
                        local[0][i][j]
                    };
                    m[(i, j)] += t.coef * v;
                }
            }
        }
        m
    }

    /// The exhaustiveness gate: every parametric gate, every slot, a
    /// generator that matches the matrix derivative. Adding a parametric
    /// `OpType` without wiring it up fails here instead of returning 0.0.
    #[test]
    fn test_every_parametric_gate_has_generator() {
        for op in parametric_ops() {
            let label = op_label(&op);
            let n_slots = op.parameters().len();
            let dim = 1 << op.n_qubits();
            assert!(n_slots >= 1, "{label} lost its parameters");
            assert!(matches!(dim, 2 | 4), "{label} is not a 1- or 2-qubit gate");
            // Slot coverage: every declared slot resolves, the one past the
            // end does not (catches an off-by-one in `op_with_slot`).
            for slot in 0..n_slots {
                assert!(
                    op_with_slot(&op, slot, 0.0).is_some(),
                    "{label} slot {slot} has no substitution arm"
                );
            }
            assert!(op_with_slot(&op, n_slots, 0.0).is_none());

            for slot in 0..n_slots {
                let terms = closed_form_generator(&op, slot)
                    .or_else(|| finite_difference_generator(&op, slot))
                    .unwrap_or_else(|| panic!("{label} slot {slot}: no generator"));
                assert!(!terms.is_empty(), "{label} slot {slot}: empty generator");

                let got = expand_terms(&terms, dim);
                let want = reference_generator(&op, slot);
                // The expansion may drop the identity component (it is an
                // exactly-zero gradient), so the residual has to be a
                // multiple of the identity.  Tolerance is set by the
                // reference stencil, not by the table.
                const TOL: f64 = 1e-9;
                let mut resid_diag: Option<Complex64> = None;
                for i in 0..dim {
                    for j in 0..dim {
                        let r = got[(i, j)] - want[(i, j)];
                        if i == j {
                            match resid_diag {
                                None => resid_diag = Some(r),
                                Some(v) => assert!(
                                    (r - v).norm() < TOL,
                                    "{label} slot {slot}: identity residual not constant: \
                                     {r:?} vs {v:?}"
                                ),
                            }
                        } else {
                            assert!(
                                r.norm() < TOL,
                                "{label} slot {slot}: G[{i},{j}] off by {r:?}"
                            );
                        }
                    }
                }
                // G = i·(∂U)·U† is Hermitian for any unitary U.
                for i in 0..dim {
                    for j in 0..dim {
                        assert!(
                            (got[(i, j)] - got[(j, i)].conj()).norm() < 1e-12,
                            "{label} slot {slot}: generator is not Hermitian"
                        );
                    }
                }
            }
        }
    }

    /// The closed-form table must cover every parametric slot — the hot
    /// path never rides on a numerical derivative, and a new parametric
    /// `OpType` cannot slip in without an analytic generator.
    #[test]
    fn test_closed_form_covers_every_parametric_slot() {
        let mut covered = Vec::new();
        for op in parametric_ops() {
            for slot in 0..op.parameters().len() {
                if closed_form_generator(&op, slot).is_some() {
                    covered.push(format!("{}[{slot}]", op_label(&op)));
                }
            }
        }
        assert_eq!(
            covered,
            vec![
                "Rx[0]", "Ry[0]", "Rz[0]", "R1[0]", "P[0]", "Rxx[0]", "Ryy[0]", "Rzz[0]", "CRx[0]",
                "CRz[0]", "CP[0]", "U[0]", "U[1]", "U[2]", "Cu[0]", "Cu[1]", "Cu[2]", "Cu[3]",
            ]
        );
    }

    /// The numerical safety net must reproduce a known generator when a
    /// gate reaches it, so a future parametric `OpType` degrades to a
    /// slightly-less-precise gradient instead of a silent 0.0.
    #[test]
    fn test_numerical_fallback_matches_closed_form() {
        for op in [
            OpType::Rz(Parameter::Const(0.7)),
            OpType::Rx(Parameter::Const(-1.1)),
        ] {
            let label = op_label(&op);
            let fd = finite_difference_generator(&op, 0).expect("fallback produced nothing");
            let cf = closed_form_generator(&op, 0).unwrap();
            assert_eq!(fd.len(), cf.len(), "{label}: term count differs");
            for (a, b) in fd.iter().zip(cf.iter()) {
                assert_eq!(a.paulis(), b.paulis(), "{label}: term order differs");
                assert!(
                    (a.coef - b.coef).abs() < 1e-9,
                    "{label}: coefficient {} vs {} (want)",
                    a.coef,
                    b.coef
                );
            }
        }
        // A gate with no parameters yields nothing at all.
        assert!(finite_difference_generator(&OpType::H, 0).is_none());
    }

    /// A gate with no generator is an error, never a silent `Ok(0)`.
    #[test]
    fn test_unknown_parametric_gate_is_an_error() {
        // Neither source may invent a generator for a gate it cannot read.
        assert!(closed_form_generator(&OpType::H, 0).is_none());
        assert!(finite_difference_generator(&OpType::H, 0).is_none());

        let mut param_idx = std::collections::HashMap::new();
        param_idx.insert("theta", 0usize);
        param_idx.insert("phi", 1usize);
        let bound = OpType::Rx(Parameter::Const(0.0));
        let values = bound_values();

        // An expression over two free variables has no per-variable
        // gradient: hard error naming the cause.
        let two_var = Parameter::Expr(Box::new(crate::ops::ParameterExpr::Add(
            Parameter::Variable {
                name: "theta".into(),
                id: 0,
            },
            Parameter::Variable {
                name: "phi".into(),
                id: 1,
            },
        )));
        let err = plan_contributions(&OpType::Rx(two_var), &bound, &param_idx, &values)
            .err()
            .expect("a two-variable expression must not yield a contribution");
        assert!(
            err.0.contains("free variables"),
            "unexpected error text: {}",
            err.0
        );

        // A single-variable expression is differentiated by the chain rule:
        // ∂(θ/2)/∂θ = 0.5, with the Rx generator scaled by the same factor.
        let half = Parameter::Expr(Box::new(crate::ops::ParameterExpr::Div(
            Parameter::Variable {
                name: "theta".into(),
                id: 0,
            },
            Parameter::Const(2.0),
        )));
        let mut one = std::collections::HashMap::new();
        one.insert("theta", 0usize);
        let contribs = plan_contributions(&OpType::Rx(half), &bound, &one, &values).unwrap();
        assert_eq!(contribs.len(), 1);
        assert_eq!(contribs[0].idx, 0);
        assert!(
            (contribs[0].scale - 0.5).abs() < 1e-9,
            "scale = {}",
            contribs[0].scale
        );
        assert_eq!(contribs[0].terms.len(), 1);
        assert_eq!(contribs[0].terms[0].paulis(), &[PAULI_X]);
        assert!((contribs[0].terms[0].coef - 0.5).abs() < 1e-12);
    }

    /// The forward sweep's dst/src CNOT kernel had no coverage at all before
    /// this test — every gradient test in this module used 1-qubit gates, so a
    /// block-stride error in `cnot_out` survived a 210-test run and only
    /// panicked on the benchmark ladder.
    ///
    /// Rx(a) on |0⟩ then CNOT(0 → 1) gives cos(a/2)|00⟩ − i·sin(a/2)|11⟩, so
    /// ⟨Z₁⟩ = cos(a) and d/da = −sin(a).
    #[test]
    fn test_cnot_adjacent_gradient_matches_analytic() {
        let a = 0.83;
        let mut dag = QuantumDAG::new(2, 0);
        dag.add_op(
            OpType::Rx(Parameter::Variable {
                name: "a".into(),
                id: 0,
            }),
            &[0],
        );
        dag.add_op(OpType::CNOT, &[0, 1]);
        let obs = vec![PauliTerm {
            paulis: vec![0, 3],
            coef: Complex64::new(1.0, 0.0),
        }];
        let mut params = std::collections::HashMap::new();
        params.insert("a".into(), a);
        let result = adjoint_grad(&dag, &obs, &params).unwrap();
        assert!(
            (result.gradients[0] + a.sin()).abs() < 1e-10,
            "got {}, want {}",
            result.gradients[0],
            -a.sin()
        );
    }

    /// Same circuit with a non-adjacent control/target pair.  A block stride
    /// that is not a power of two makes `chunks_exact` yield nothing, which
    /// leaves the destination unwritten — a silently wrong gradient rather
    /// than a panic, so this case is checked explicitly.
    #[test]
    fn test_cnot_non_adjacent_gradient_matches_analytic() {
        let a = -1.21;
        let mut dag = QuantumDAG::new(3, 0);
        dag.add_op(
            OpType::Rx(Parameter::Variable {
                name: "a".into(),
                id: 0,
            }),
            &[0],
        );
        dag.add_op(OpType::CNOT, &[0, 2]);
        let obs = vec![PauliTerm {
            paulis: vec![0, 0, 3],
            coef: Complex64::new(1.0, 0.0),
        }];
        let mut params = std::collections::HashMap::new();
        params.insert("a".into(), a);
        let result = adjoint_grad(&dag, &obs, &params).unwrap();
        assert!(
            (result.gradients[0] + a.sin()).abs() < 1e-10,
            "got {}, want {}",
            result.gradients[0],
            -a.sin()
        );
    }

    /// A *dense* 2-qubit parameter gate takes the in-place arm of the
    /// ping-pong forward (no dst/src kernel for a general 4×4), so this is the
    /// one path that still copies; it must stay exact.
    ///
    /// H(0), Ry(b) on q1, CRz(θ) on (0,1): the state is
    /// (c|00⟩ + s|01⟩ + e^{−iθ/2}·c|10⟩ + e^{iθ/2}·s|11⟩)/√2 with
    /// c = cos(b/2), s = sin(b/2), so ⟨X₁⟩ = c·s·(1 + cos θ) and
    /// d/dθ = −sin(b)/2 · sin(θ).  The result is independent of the phase
    /// sign/split convention, only the relative phase θ entering.
    #[test]
    fn test_crz_gradient_matches_analytic() {
        let (b, theta) = (0.6, 1.1);
        let mut dag = QuantumDAG::new(2, 0);
        dag.add_op(OpType::H, &[0]);
        dag.add_op(OpType::Ry(Parameter::Const(b)), &[1]);
        dag.add_op(
            OpType::CRz(Parameter::Variable {
                name: "t".into(),
                id: 0,
            }),
            &[0, 1],
        );
        let obs = vec![PauliTerm {
            paulis: vec![0, 1],
            coef: Complex64::new(1.0, 0.0),
        }];
        let mut params = std::collections::HashMap::new();
        params.insert("t".into(), theta);
        let result = adjoint_grad(&dag, &obs, &params).unwrap();
        let want = -b.sin() / 2.0 * theta.sin();
        assert!(
            (result.gradients[0] - want).abs() < 1e-10,
            "got {}, want {}",
            result.gradients[0],
            want
        );
    }
}
