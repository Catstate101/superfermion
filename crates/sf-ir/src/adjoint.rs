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

/// States at or above this dimension (n >= 17) split their sweeps across the
/// rayon pool; below it the serial kernels win (and the parallel paths are
/// skipped entirely when the pool is pinned to one thread).  Bumped from the
/// historical `1<<16` after on-host attribution at the 4-thread bench pin:
/// n=16 (dim == `1<<16`) sits exactly on the old boundary and paid more in
/// per-gate rayon task-spawn / copy coordination than the parallel sweep
/// saved — the lean serial path measured ~8% faster on the bench's n=16 L=2
/// ladder (reproduced across interleaved process pairs), and the two paths
/// agree to 1 ULP (max_abs_diff 2e-16).  No state size lies between `1<<16`
/// and `1<<17`, so this changes n=16 only; n>=17 stays parallel.
const PAR_DIM_DEFAULT: usize = 1 << 17;

/// Parallel/serial crossover dimension for the adjoint sweeps.
/// `SF_ADJ_PAR_MIN` overrides so the n=16 (dim == 1<<16) boundary can be
/// A/B-attributed on-host under the bench thread pin without a rebuild:
/// raising it above 65536 forces n=16 onto the lean serial path. Default
/// `1<<16` reproduces the shipped behavior exactly.
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
#[inline]
fn copy_slice_par(dst: &mut [Complex64], src: &[Complex64], par: bool) {
    if par && src.len() >= adj_par_min() {
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
    // n >= 17 (see `PAR_DIM_DEFAULT`): sweeps may split across the rayon pool
    // (skipped entirely when the pool is pinned to one thread).
    let par = dim >= adj_par_min() && rayon::current_num_threads() > 1;

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

    // Per-op bind instead of `dag.bind`: `bind` clones the entire graph
    // (petgraph alloc + every node) only to run `bind_params` on each op, so
    // building the bound gate list straight from the DAG is bit-identical to
    // the historical path while dropping the wasted clone and one
    // topological walk. The op_types produced here are exactly the ones
    // `dag.bind(..)` would have handed to `ops`, so every downstream gate
    // matrix and the missing-parameter check below are unchanged.
    let order = dag.topological_order();
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

        // Gradient contributions: 2·scale·Im⟨φ|G_k|ψ_after_k⟩, summed
        // over the generator's Pauli terms and over every slot of this
        // gate (a shared parameter name on several gates, or several slots
        // of one gate, each add their own term).
        let t0 = std::time::Instant::now();
        if !contribs[k].is_empty() {
            let src: &[Complex64] = if use_cache {
                // Invariant: `param_slot[k].is_some()` iff `contribs[k]`
                // is non-empty — both are derived from the same plan, so
                // the expect cannot fire.
                let slot = param_slot[k].expect("cached gradient gate has no cache slot");
                &cache[slot * dim..(slot + 1) * dim]
            } else {
                // Recomputed ψ_k: re-apply U_k into a preallocated buffer
                // (no per-parameter allocation).
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
            if adj_trace {
                t_ip_dot += t0.elapsed().as_secs_f64() * 1e3;
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
}
