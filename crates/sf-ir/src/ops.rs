//! Quantum operation types for the Superfermion IR.
//!
//! This module defines the complete vocabulary of quantum operations
//! that the IR can represent. Every quantum gate, measurement, and
//! special operation is an `OpType` variant.
//!
//! # Design Rationale
//! The IR is hardware-neutral: it represents the abstract *meaning*
//! of a quantum circuit, not how any specific hardware executes it.
//! The compiler (sf-compiler) translates this to native gate sets.

use num_complex::Complex64;
use serde::{Deserialize, Serialize};
use std::f64::consts::{FRAC_PI_4, SQRT_2};

/// A variational parameter — either a fixed value or a symbolic
/// variable that gets bound at execution time.
///
/// # Physics
/// Parameters appear in rotation gates: e.g. Rx(θ) = exp(-iθX/2).
/// Symbolic parameters are the trainable variables optimized by
/// variational algorithms (VQE, QAOA, QML).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Parameter {
    /// Fixed numerical value (not trainable).
    Const(f64),

    /// Symbolic variable — trainable parameter.
    /// `name`: human-readable e.g. "theta_0", "phi_layer2_qubit3"
    /// `id`: unique integer for fast lookup in parameter arrays
    Variable { name: String, id: usize },

    /// Expression over parameters (for derived params).
    /// e.g. theta/2, 2*phi, theta + phi
    Expr(Box<ParameterExpr>),
}

/// Arithmetic expressions over parameters.
/// Enables expressions like `theta/2` or `phi + pi/4` in circuits.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ParameterExpr {
    Add(Parameter, Parameter),
    Sub(Parameter, Parameter),
    Mul(Parameter, Parameter),
    Div(Parameter, Parameter),
    Neg(Parameter),
    Sin(Parameter),
    Cos(Parameter),
    Exp(Parameter),
}

impl Parameter {
    /// Evaluate a parameter to a concrete f64 value.
    /// Panics if the parameter contains unbound variables.
    pub fn evaluate(&self) -> f64 {
        match self {
            Parameter::Const(v) => *v,
            Parameter::Variable { name, .. } => {
                panic!("Cannot evaluate unbound variable '{name}'. Call dag.bind() first.")
            }
            Parameter::Expr(expr) => expr.evaluate(),
        }
    }

    /// Try to evaluate, returning None if variables remain unbound.
    pub fn try_evaluate(&self) -> Option<f64> {
        match self {
            Parameter::Const(v) => Some(*v),
            Parameter::Variable { .. } => None,
            Parameter::Expr(expr) => expr.try_evaluate(),
        }
    }

    /// Bind a map of variable names to values.
    pub fn bind(&self, values: &std::collections::HashMap<String, f64>) -> Parameter {
        match self {
            Parameter::Const(_) => self.clone(),
            Parameter::Variable { name, .. } => {
                if let Some(&val) = values.get(name) {
                    Parameter::Const(val)
                } else {
                    self.clone()
                }
            }
            Parameter::Expr(expr) => {
                let bound = expr.bind(values);
                match bound.try_evaluate() {
                    Some(val) => Parameter::Const(val),
                    None => Parameter::Expr(Box::new(bound)),
                }
            }
        }
    }

    /// Check if this parameter is a free variable (not yet bound).
    pub fn is_variable(&self) -> bool {
        matches!(self, Parameter::Variable { .. })
    }

    /// Collect all variable names in this parameter.
    pub fn variable_names(&self) -> Vec<String> {
        match self {
            Parameter::Const(_) => vec![],
            Parameter::Variable { name, .. } => vec![name.clone()],
            Parameter::Expr(expr) => expr.variable_names(),
        }
    }
}

impl ParameterExpr {
    pub fn evaluate(&self) -> f64 {
        match self {
            ParameterExpr::Add(a, b) => a.evaluate() + b.evaluate(),
            ParameterExpr::Sub(a, b) => a.evaluate() - b.evaluate(),
            ParameterExpr::Mul(a, b) => a.evaluate() * b.evaluate(),
            ParameterExpr::Div(a, b) => a.evaluate() / b.evaluate(),
            ParameterExpr::Neg(a) => -a.evaluate(),
            ParameterExpr::Sin(a) => a.evaluate().sin(),
            ParameterExpr::Cos(a) => a.evaluate().cos(),
            ParameterExpr::Exp(a) => a.evaluate().exp(),
        }
    }

    pub fn try_evaluate(&self) -> Option<f64> {
        match self {
            ParameterExpr::Add(a, b) => Some(a.try_evaluate()? + b.try_evaluate()?),
            ParameterExpr::Sub(a, b) => Some(a.try_evaluate()? - b.try_evaluate()?),
            ParameterExpr::Mul(a, b) => Some(a.try_evaluate()? * b.try_evaluate()?),
            ParameterExpr::Div(a, b) => Some(a.try_evaluate()? / b.try_evaluate()?),
            ParameterExpr::Neg(a) => Some(-a.try_evaluate()?),
            ParameterExpr::Sin(a) => Some(a.try_evaluate()?.sin()),
            ParameterExpr::Cos(a) => Some(a.try_evaluate()?.cos()),
            ParameterExpr::Exp(a) => Some(a.try_evaluate()?.exp()),
        }
    }

    pub fn bind(&self, values: &std::collections::HashMap<String, f64>) -> ParameterExpr {
        match self {
            ParameterExpr::Add(a, b) => ParameterExpr::Add(a.bind(values), b.bind(values)),
            ParameterExpr::Sub(a, b) => ParameterExpr::Sub(a.bind(values), b.bind(values)),
            ParameterExpr::Mul(a, b) => ParameterExpr::Mul(a.bind(values), b.bind(values)),
            ParameterExpr::Div(a, b) => ParameterExpr::Div(a.bind(values), b.bind(values)),
            ParameterExpr::Neg(a) => ParameterExpr::Neg(a.bind(values)),
            ParameterExpr::Sin(a) => ParameterExpr::Sin(a.bind(values)),
            ParameterExpr::Cos(a) => ParameterExpr::Cos(a.bind(values)),
            ParameterExpr::Exp(a) => ParameterExpr::Exp(a.bind(values)),
        }
    }

    pub fn variable_names(&self) -> Vec<String> {
        match self {
            ParameterExpr::Add(a, b)
            | ParameterExpr::Sub(a, b)
            | ParameterExpr::Mul(a, b)
            | ParameterExpr::Div(a, b) => {
                let mut names = a.variable_names();
                names.extend(b.variable_names());
                names
            }
            ParameterExpr::Neg(a)
            | ParameterExpr::Sin(a)
            | ParameterExpr::Cos(a)
            | ParameterExpr::Exp(a) => a.variable_names(),
        }
    }
}

/// Every quantum operation type Superfermion supports.
///
/// This enum is the vocabulary of the IR. Adding a new gate requires:
/// 1. Add variant here
/// 2. Implement `to_unitary()` for the gate
/// 3. Add decomposition cases in sf-compiler for each hardware target
///
/// # Phase Convention
/// Rotation gates use: exp(-iθP/2) where P is the Pauli generator.
/// This matches Qiskit and most textbooks.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum OpType {
    // ─── Single-qubit gates ───────────────────────────────
    /// Hadamard: H = (X+Z)/√2
    H,
    /// Pauli-X (NOT gate): |0⟩↔|1⟩
    X,
    /// Pauli-Y: Y = iXZ
    Y,
    /// Pauli-Z: Z = diag(1, -1)
    Z,
    /// S gate: S = diag(1, i) = √Z
    S,
    /// S†: Sdg = diag(1, -i)
    Sdg,
    /// T gate: T = diag(1, e^{iπ/4}) = √S
    T,
    /// T†: Tdg = diag(1, e^{-iπ/4})
    Tdg,
    /// √X gate: SX = (I + iX)/√2
    SX,
    /// √X†
    SXdg,
    /// Identity (explicit, for timing/barriers)
    Id,

    // ─── Parameterized single-qubit rotations ─────────────
    /// Rx(θ) = exp(-iθX/2)
    Rx(Parameter),
    /// Ry(θ) = exp(-iθY/2)
    Ry(Parameter),
    /// Rz(θ) = exp(-iθZ/2)
    Rz(Parameter),
    /// R1(φ) = diag(1, e^{iφ}) — phase gate
    R1(Parameter),
    /// U(θ,φ,λ) — general SU(2) via ZYZ decomposition
    U(Parameter, Parameter, Parameter),
    /// P(λ) = diag(1, e^{iλ}) — same as R1
    P(Parameter),

    // ─── Two-qubit gates ──────────────────────────────────
    /// CNOT (Controlled-X): ctrl, tgt
    CNOT,
    /// CZ (Controlled-Z)
    CZ,
    /// CY (Controlled-Y)
    CY,
    /// SWAP
    SWAP,
    /// iSWAP
    ISWAP,
    /// Echoed cross-resonance (IBM native 2Q gate)
    ECR,

    // ─── Parameterized two-qubit ──────────────────────────
    /// Rzz(θ) = exp(-iθZZ/2)
    Rzz(Parameter),
    /// Rxx(θ) = exp(-iθXX/2)
    Rxx(Parameter),
    /// Ryy(θ) = exp(-iθYY/2)
    Ryy(Parameter),
    /// Controlled-Rx
    CRx(Parameter),
    /// Controlled-Rz
    CRz(Parameter),
    /// Controlled-Phase: CP(φ) = diag(1, 1, 1, e^{iφ})
    CP(Parameter),

    // ─── Three-qubit gates ────────────────────────────────
    /// Toffoli (CCX)
    CCX,
    /// Fredkin (CSWAP)
    CSWAP,

    // ─── Measurement ──────────────────────────────────────
    /// Computational basis measurement
    Measure,
    /// Measure all qubits
    MeasureAll,

    // ─── Special ──────────────────────────────────────────
    /// Compiler barrier (prevents optimization across)
    Barrier,
    /// Reset qubit to |0⟩
    Reset,
    /// Input boundary node (internal use)
    Input,
    /// Output boundary node (internal use)
    Output,
    /// Custom opaque gate
    Custom(String, Vec<Parameter>),
    /// Opaque unitary matrix gate -- stored without decomposition.
    /// Applied directly to statevector during simulation; decomposed
    /// into basis gates by UnitaryDecompositionPass during compilation.
    Unitary(nalgebra::DMatrix<num_complex::Complex64>),
}

impl OpType {
    /// Number of qubits this operation acts on.
    pub fn n_qubits(&self) -> usize {
        match self {
            // 0-qubit (boundary/barrier)
            OpType::Barrier | OpType::Input | OpType::Output => 0,
            // Single-qubit
            OpType::H
            | OpType::X
            | OpType::Y
            | OpType::Z
            | OpType::S
            | OpType::Sdg
            | OpType::T
            | OpType::Tdg
            | OpType::SX
            | OpType::SXdg
            | OpType::Id
            | OpType::Rx(_)
            | OpType::Ry(_)
            | OpType::Rz(_)
            | OpType::R1(_)
            | OpType::U(_, _, _)
            | OpType::P(_)
            | OpType::Measure
            | OpType::Reset => 1,
            // Two-qubit
            OpType::CNOT
            | OpType::CZ
            | OpType::CY
            | OpType::SWAP
            | OpType::ISWAP
            | OpType::ECR
            | OpType::Rzz(_)
            | OpType::Rxx(_)
            | OpType::Ryy(_)
            | OpType::CRx(_)
            | OpType::CRz(_)
            | OpType::CP(_) => 2,
            // Three-qubit
            OpType::CCX | OpType::CSWAP => 3,
            // Special
            OpType::MeasureAll => 0, // acts on all qubits
            OpType::Custom(_, _) => 0,
            OpType::Unitary(ref m) => (m.nrows() as f64).log2() as usize,
        }
    }

    /// Whether this operation is parameterized.
    pub fn is_parameterized(&self) -> bool {
        match self {
            OpType::Rx(_)
            | OpType::Ry(_)
            | OpType::Rz(_)
            | OpType::R1(_)
            | OpType::U(_, _, _)
            | OpType::P(_)
            | OpType::Rzz(_)
            | OpType::Rxx(_)
            | OpType::Ryy(_)
            | OpType::CRx(_)
            | OpType::CRz(_)
            | OpType::CP(_) => true,
            _ => false,
        }
    }

    /// Whether this is a boundary node (internal IR structure).
    pub fn is_boundary(&self) -> bool {
        matches!(self, OpType::Input | OpType::Output)
    }

    /// Whether this is a measurement operation.
    pub fn is_measurement(&self) -> bool {
        matches!(self, OpType::Measure | OpType::MeasureAll)
    }

    /// Get the 2×2 or 4×4 unitary matrix for this gate.
    /// Returns None for non-unitary operations (Measure, Barrier, etc.)
    ///
    /// # Phase Convention
    /// Rx(θ) = [[cos(θ/2), -i·sin(θ/2)], [-i·sin(θ/2), cos(θ/2)]]
    /// This is the exp(-iθX/2) convention.
    pub fn to_matrix(&self) -> nalgebra::DMatrix<num_complex::Complex64> {
        let u_vec = self.to_unitary_matrix().unwrap_or_else(|| {
            // Return IDENTITY for non-unitary ops so contraction doesn't fail.
            // (Previously this returned an all-ones matrix — a subtle correctness
            // bug that caused statevector overflow for any gate without a
            // to_unitary_matrix implementation. Identity is unitary and a safe
            // no-op fallback.)
            let dim = 1 << self.n_qubits();
            let zero = num_complex::Complex64::new(0.0, 0.0);
            let one = num_complex::Complex64::new(1.0, 0.0);
            (0..dim)
                .map(|i| (0..dim).map(|j| if i == j { one } else { zero }).collect())
                .collect()
        });

        let rows = u_vec.len();
        let cols = u_vec[0].len();
        // nalgebra::DMatrix::from_iterator expects column-major data.
        // Since u_vec is row-major (vec of rows), we must transpose the result.
        nalgebra::DMatrix::from_iterator(rows, cols, u_vec.into_iter().flatten()).transpose()
    }

    /// Get the 2×2 or 4×4 unitary matrix for this gate (as Vec<Vec>).
    pub fn to_unitary_matrix(&self) -> Option<Vec<Vec<Complex64>>> {
        let i = Complex64::i();
        let zero = Complex64::new(0.0, 0.0);
        let one = Complex64::new(1.0, 0.0);
        let neg_one = Complex64::new(-1.0, 0.0);
        let inv_sqrt2 = Complex64::new(1.0 / SQRT_2, 0.0);

        match self {
            OpType::Id => Some(vec![vec![one, zero], vec![zero, one]]),

            // H = (1/√2) [[1, 1], [1, -1]]
            OpType::H => Some(vec![
                vec![inv_sqrt2, inv_sqrt2],
                vec![inv_sqrt2, -inv_sqrt2],
            ]),

            // X = [[0, 1], [1, 0]]
            OpType::X => Some(vec![vec![zero, one], vec![one, zero]]),

            // Y = [[0, -i], [i, 0]]
            OpType::Y => Some(vec![vec![zero, -i], vec![i, zero]]),

            // Z = [[1, 0], [0, -1]]
            OpType::Z => Some(vec![vec![one, zero], vec![zero, neg_one]]),

            // S = [[1, 0], [0, i]]
            OpType::S => Some(vec![vec![one, zero], vec![zero, i]]),

            // Sdg = [[1, 0], [0, -i]]
            OpType::Sdg => Some(vec![vec![one, zero], vec![zero, -i]]),

            // T = [[1, 0], [0, e^{iπ/4}]]
            OpType::T => {
                let t_phase = Complex64::from_polar(1.0, FRAC_PI_4);
                Some(vec![vec![one, zero], vec![zero, t_phase]])
            }

            // Tdg = [[1, 0], [0, e^{-iπ/4}]]
            OpType::Tdg => {
                let t_phase = Complex64::from_polar(1.0, -FRAC_PI_4);
                Some(vec![vec![one, zero], vec![zero, t_phase]])
            }

            // SX = (1/2)[[1+i, 1-i], [1-i, 1+i]]
            OpType::SX => {
                let a = Complex64::new(0.5, 0.5); // (1+i)/2
                let b = Complex64::new(0.5, -0.5); // (1-i)/2
                Some(vec![vec![a, b], vec![b, a]])
            }

            // SXdg = (1/2)[[1-i, 1+i], [1+i, 1-i]]
            OpType::SXdg => {
                let a = Complex64::new(0.5, -0.5);
                let b = Complex64::new(0.5, 0.5);
                Some(vec![vec![a, b], vec![b, a]])
            }

            // Rx(θ) = [[cos(θ/2), -i·sin(θ/2)], [-i·sin(θ/2), cos(θ/2)]]
            OpType::Rx(param) => {
                let theta = param.evaluate();
                let c = Complex64::new((theta / 2.0).cos(), 0.0);
                let s = Complex64::new(0.0, -(theta / 2.0).sin());
                Some(vec![vec![c, s], vec![s, c]])
            }

            // Ry(θ) = exp(-iθY/2) = [[cos(θ/2), -sin(θ/2)], [sin(θ/2), cos(θ/2)]]
            OpType::Ry(param) => {
                let theta = param.evaluate();
                let c = Complex64::new((theta / 2.0).cos(), 0.0);
                let s = Complex64::new((theta / 2.0).sin(), 0.0);
                Some(vec![vec![c, -s], vec![s, c]])
            }

            // Rz(θ) = [[e^{-iθ/2}, 0], [0, e^{iθ/2}]]
            OpType::Rz(param) => {
                let theta = param.evaluate();
                let em = Complex64::from_polar(1.0, -theta / 2.0);
                let ep = Complex64::from_polar(1.0, theta / 2.0);
                Some(vec![vec![em, zero], vec![zero, ep]])
            }

            // R1(φ) = [[1, 0], [0, e^{iφ}]]
            OpType::R1(param) | OpType::P(param) => {
                let phi = param.evaluate();
                let ep = Complex64::from_polar(1.0, phi);
                Some(vec![vec![one, zero], vec![zero, ep]])
            }

            // U(θ,φ,λ) = [[cos(θ/2), -e^{iλ}·sin(θ/2)],
            //              [e^{iφ}·sin(θ/2), e^{i(φ+λ)}·cos(θ/2)]]
            OpType::U(theta, phi, lam) => {
                let t = theta.evaluate();
                let p = phi.evaluate();
                let l = lam.evaluate();
                let ct = Complex64::new((t / 2.0).cos(), 0.0);
                let st = Complex64::new((t / 2.0).sin(), 0.0);
                let el = Complex64::from_polar(1.0, l);
                let ep = Complex64::from_polar(1.0, p);
                let epl = Complex64::from_polar(1.0, p + l);
                Some(vec![vec![ct, -el * st], vec![ep * st, epl * ct]])
            }

            // CNOT = [[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]]
            OpType::CNOT => Some(vec![
                vec![one, zero, zero, zero],
                vec![zero, one, zero, zero],
                vec![zero, zero, zero, one],
                vec![zero, zero, one, zero],
            ]),

            // CZ = diag(1, 1, 1, -1)
            OpType::CZ => Some(vec![
                vec![one, zero, zero, zero],
                vec![zero, one, zero, zero],
                vec![zero, zero, one, zero],
                vec![zero, zero, zero, neg_one],
            ]),

            // SWAP = [[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]]
            OpType::SWAP => Some(vec![
                vec![one, zero, zero, zero],
                vec![zero, zero, one, zero],
                vec![zero, one, zero, zero],
                vec![zero, zero, zero, one],
            ]),

            // Rzz(θ) = diag(e^{-iθ/2}, e^{iθ/2}, e^{iθ/2}, e^{-iθ/2})
            OpType::Rzz(param) => {
                let theta = param.evaluate();
                let em = Complex64::from_polar(1.0, -theta / 2.0);
                let ep = Complex64::from_polar(1.0, theta / 2.0);
                Some(vec![
                    vec![em, zero, zero, zero],
                    vec![zero, ep, zero, zero],
                    vec![zero, zero, ep, zero],
                    vec![zero, zero, zero, em],
                ])
            }

            // Rxx(θ) = exp(-iθXX/2)
            //       = c*I⊗I - i*s*X⊗X     (c=cos(θ/2), s=sin(θ/2))
            // Matrix (basis |00>,|01>,|10>,|11>):
            //   [[c,    0,    0,    -is],
            //    [0,    c,    -is,  0  ],
            //    [0,    -is,  c,    0  ],
            //    [-is,  0,    0,    c  ]]
            OpType::Rxx(param) => {
                let theta = param.evaluate();
                let c = Complex64::new((theta / 2.0).cos(), 0.0);
                let nis = Complex64::new(0.0, -(theta / 2.0).sin()); // -i*sin
                Some(vec![
                    vec![c, zero, zero, nis],
                    vec![zero, c, nis, zero],
                    vec![zero, nis, c, zero],
                    vec![nis, zero, zero, c],
                ])
            }

            // Ryy(θ) = exp(-iθYY/2)
            //       = c*I⊗I - i*s*Y⊗Y
            // Y⊗Y has off-anti-diagonals = (-1,+1,+1,-1) effectively.
            // Matrix (basis |00>,|01>,|10>,|11>):
            //   [[c,    0,    0,    +is],
            //    [0,    c,    -is,  0  ],
            //    [0,    -is,  c,    0  ],
            //    [+is,  0,    0,    c  ]]
            OpType::Ryy(param) => {
                let theta = param.evaluate();
                let c = Complex64::new((theta / 2.0).cos(), 0.0);
                let s = (theta / 2.0).sin();
                let pis = Complex64::new(0.0, s); // +i*sin
                let nis = Complex64::new(0.0, -s); // -i*sin
                Some(vec![
                    vec![c, zero, zero, pis],
                    vec![zero, c, nis, zero],
                    vec![zero, nis, c, zero],
                    vec![pis, zero, zero, c],
                ])
            }

            // CP(φ) = diag(1, 1, 1, e^{iφ}) — controlled-phase
            OpType::CP(param) => {
                let phi = param.evaluate();
                let phase = Complex64::from_polar(1.0, phi);
                Some(vec![
                    vec![one, zero, zero, zero],
                    vec![zero, one, zero, zero],
                    vec![zero, zero, one, zero],
                    vec![zero, zero, zero, phase],
                ])
            }

            // CRx(θ): controlled-Rx with control on first qubit
            OpType::CRx(param) => {
                let theta = param.evaluate();
                let c = Complex64::new((theta / 2.0).cos(), 0.0);
                let nis = Complex64::new(0.0, -(theta / 2.0).sin());
                Some(vec![
                    vec![one, zero, zero, zero],
                    vec![zero, one, zero, zero],
                    vec![zero, zero, c, nis],
                    vec![zero, zero, nis, c],
                ])
            }

            // CRz(θ): controlled-Rz with control on first qubit
            OpType::CRz(param) => {
                let theta = param.evaluate();
                let em = Complex64::from_polar(1.0, -theta / 2.0);
                let ep = Complex64::from_polar(1.0, theta / 2.0);
                Some(vec![
                    vec![one, zero, zero, zero],
                    vec![zero, one, zero, zero],
                    vec![zero, zero, em, zero],
                    vec![zero, zero, zero, ep],
                ])
            }

            // CY: control q0, Y on q1 (basis |00>,|01>,|10>,|11>)
            // Matrix: [[1,0,0,0],[0,1,0,0],[0,0,0,-i],[0,0,i,0]]
            OpType::CY => Some(vec![
                vec![one, zero, zero, zero],
                vec![zero, one, zero, zero],
                vec![zero, zero, zero, -i],
                vec![zero, zero, i, zero],
            ]),

            // iSWAP: [[1,0,0,0],[0,0,i,0],[0,i,0,0],[0,0,0,1]]
            OpType::ISWAP => Some(vec![
                vec![one, zero, zero, zero],
                vec![zero, zero, i, zero],
                vec![zero, i, zero, zero],
                vec![zero, zero, zero, one],
            ]),

            // ECR: Echoed Cross-Resonance (OpenQASM 3 standard)
            // (1/√2) * [[0,0,1,i],[0,0,i,1],[1,-i,0,0],[-i,1,0,0]]
            OpType::ECR => {
                let s = Complex64::new(1.0 / SQRT_2, 0.0);
                let is_ = Complex64::new(0.0, 1.0 / SQRT_2);
                let nis = Complex64::new(0.0, -1.0 / SQRT_2);
                Some(vec![
                    vec![zero, zero, s, is_],
                    vec![zero, zero, is_, s],
                    vec![s, nis, zero, zero],
                    vec![nis, s, zero, zero],
                ])
            }

            // CCX (Toffoli): flips q2 iff q0=q1=1. Basis |q0 q1 q2>.
            // Swaps |110> ↔ |111> (indices 6 ↔ 7).
            OpType::CCX => {
                let mut m = vec![vec![zero; 8]; 8];
                for k in 0..8 {
                    m[k][k] = one;
                }
                m[6][6] = zero;
                m[7][7] = zero;
                m[6][7] = one;
                m[7][6] = one;
                Some(m)
            }

            // CSWAP (Fredkin): swaps q1 ↔ q2 iff q0=1. Basis |q0 q1 q2>.
            // Swaps |101> ↔ |110> (indices 5 ↔ 6).
            OpType::CSWAP => {
                let mut m = vec![vec![zero; 8]; 8];
                for k in 0..8 {
                    m[k][k] = one;
                }
                m[5][5] = zero;
                m[6][6] = zero;
                m[5][6] = one;
                m[6][5] = one;
                Some(m)
            }

            // Non-unitary operations
            OpType::Measure
            | OpType::MeasureAll
            | OpType::Barrier
            | OpType::Reset
            | OpType::Input
            | OpType::Output => None,

            OpType::Unitary(ref m) => {
                let rows = m.nrows();
                let cols = m.ncols();
                let mut result = Vec::with_capacity(rows);
                for r in 0..rows {
                    let mut row = Vec::with_capacity(cols);
                    for c in 0..cols {
                        row.push(m[(r, c)]);
                    }
                    result.push(row);
                }
                Some(result)
            }

            // For remaining gates, we can add them as needed
            _ => {
                log::warn!("Unitary not yet implemented for {:?}", self);
                None
            }
        }
    }

    /// Bind parameter values in this op type.
    pub fn bind_params(&self, values: &std::collections::HashMap<String, f64>) -> OpType {
        match self {
            OpType::Rx(p) => OpType::Rx(p.bind(values)),
            OpType::Ry(p) => OpType::Ry(p.bind(values)),
            OpType::Rz(p) => OpType::Rz(p.bind(values)),
            OpType::R1(p) => OpType::R1(p.bind(values)),
            OpType::P(p) => OpType::P(p.bind(values)),
            OpType::U(a, b, c) => OpType::U(a.bind(values), b.bind(values), c.bind(values)),
            OpType::Rzz(p) => OpType::Rzz(p.bind(values)),
            OpType::Rxx(p) => OpType::Rxx(p.bind(values)),
            OpType::Ryy(p) => OpType::Ryy(p.bind(values)),
            OpType::CRx(p) => OpType::CRx(p.bind(values)),
            OpType::CRz(p) => OpType::CRz(p.bind(values)),
            OpType::CP(p) => OpType::CP(p.bind(values)),
            OpType::Custom(name, params) => {
                let bound_params = params.iter().map(|p| p.bind(values)).collect();
                OpType::Custom(name.clone(), bound_params)
            }
            OpType::Unitary(m) => OpType::Unitary(m.clone()),
            other => other.clone(),
        }
    }

    /// Get all parameters from this op type.
    pub fn parameters(&self) -> Vec<&Parameter> {
        match self {
            OpType::Rx(p)
            | OpType::Ry(p)
            | OpType::Rz(p)
            | OpType::R1(p)
            | OpType::P(p)
            | OpType::Rzz(p)
            | OpType::Rxx(p)
            | OpType::Ryy(p)
            | OpType::CRx(p)
            | OpType::CRz(p)
            | OpType::CP(p) => vec![p],
            OpType::U(a, b, c) => vec![a, b, c],
            OpType::Custom(_, params) => params.iter().collect(),
            OpType::Unitary(_) => vec![],
            _ => vec![],
        }
    }

    /// Human-readable name for display.
    pub fn name(&self) -> String {
        match self {
            OpType::H => "H".to_string(),
            OpType::X => "X".to_string(),
            OpType::Y => "Y".to_string(),
            OpType::Z => "Z".to_string(),
            OpType::S => "S".to_string(),
            OpType::Sdg => "Sdg".to_string(),
            OpType::T => "T".to_string(),
            OpType::Tdg => "Tdg".to_string(),
            OpType::SX => "SX".to_string(),
            OpType::SXdg => "SXdg".to_string(),
            OpType::Id => "Id".to_string(),
            OpType::Rx(_) => "Rx".to_string(),
            OpType::Ry(_) => "Ry".to_string(),
            OpType::Rz(_) => "Rz".to_string(),
            OpType::R1(_) => "R1".to_string(),
            OpType::U(_, _, _) => "U".to_string(),
            OpType::P(_) => "P".to_string(),
            OpType::CNOT => "CNOT".to_string(),
            OpType::CZ => "CZ".to_string(),
            OpType::CY => "CY".to_string(),
            OpType::SWAP => "SWAP".to_string(),
            OpType::ISWAP => "iSWAP".to_string(),
            OpType::ECR => "ECR".to_string(),
            OpType::Rzz(_) => "Rzz".to_string(),
            OpType::Rxx(_) => "Rxx".to_string(),
            OpType::Ryy(_) => "Ryy".to_string(),
            OpType::CRx(_) => "CRx".to_string(),
            OpType::CRz(_) => "CRz".to_string(),
            OpType::CP(_) => "CP".to_string(),
            OpType::CCX => "CCX".to_string(),
            OpType::CSWAP => "CSWAP".to_string(),
            OpType::Measure => "Measure".to_string(),
            OpType::MeasureAll => "MeasureAll".to_string(),
            OpType::Barrier => "Barrier".to_string(),
            OpType::Reset => "Reset".to_string(),
            OpType::Input => "Input".to_string(),
            OpType::Output => "Output".to_string(),
            OpType::Custom(ref name, _) => name.clone(),
            OpType::Unitary(_) => "UNITARY".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;
    use std::f64::consts::PI;

    fn is_unitary(matrix: &[Vec<Complex64>]) -> bool {
        let n = matrix.len();
        for i in 0..n {
            for j in 0..n {
                let mut sum = Complex64::new(0.0, 0.0);
                for k in 0..n {
                    sum += matrix[k][i].conj() * matrix[k][j];
                }
                let expected = if i == j {
                    Complex64::new(1.0, 0.0)
                } else {
                    Complex64::new(0.0, 0.0)
                };
                if (sum - expected).norm() > 1e-10 {
                    return false;
                }
            }
        }
        true
    }

    fn det_2x2(m: &[Vec<Complex64>]) -> Complex64 {
        m[0][0] * m[1][1] - m[0][1] * m[1][0]
    }

    #[test]
    fn test_all_single_qubit_gates_are_unitary() {
        let gates = vec![
            OpType::H,
            OpType::X,
            OpType::Y,
            OpType::Z,
            OpType::S,
            OpType::Sdg,
            OpType::T,
            OpType::Tdg,
            OpType::SX,
            OpType::SXdg,
            OpType::Id,
        ];
        for gate in &gates {
            let u = gate
                .to_unitary_matrix()
                .unwrap_or_else(|| panic!("{:?} should have a unitary", gate));
            assert!(is_unitary(&u), "{:?} is NOT unitary! U†U ≠ I", gate);
            let det = det_2x2(&u);
            assert!(
                (det.norm() - 1.0).abs() < 1e-10,
                "{:?} |det| = {} ≠ 1",
                gate,
                det.norm()
            );
        }
    }

    #[test]
    fn test_parameterized_gates_are_unitary() {
        let angles = vec![
            0.0,
            0.5,
            1.0,
            PI / 4.0,
            PI / 2.0,
            PI,
            3.0 * PI / 2.0,
            2.0 * PI,
        ];
        for theta in &angles {
            let gates = vec![
                OpType::Rx(Parameter::Const(*theta)),
                OpType::Ry(Parameter::Const(*theta)),
                OpType::Rz(Parameter::Const(*theta)),
                OpType::R1(Parameter::Const(*theta)),
            ];
            for gate in &gates {
                let u = gate.to_unitary_matrix().unwrap();
                assert!(is_unitary(&u), "{:?} at θ={} is NOT unitary", gate, theta);
            }
        }
    }

    #[test]
    fn test_two_qubit_gates_are_unitary() {
        let gates = vec![OpType::CNOT, OpType::CZ, OpType::SWAP];
        for gate in &gates {
            let u = gate.to_unitary_matrix().unwrap();
            assert!(is_unitary(&u), "{:?} is NOT unitary", gate);
        }
    }

    #[test]
    fn test_h_squared_is_identity() {
        let h = OpType::H.to_unitary_matrix().unwrap();
        // H² = I
        let h2 = mat_mul(&h, &h);
        let id = OpType::Id.to_unitary_matrix().unwrap();
        assert_matrices_close(&h2, &id, 1e-10, "H² ≠ I");
    }

    #[test]
    fn test_x_squared_is_identity() {
        let x = OpType::X.to_unitary_matrix().unwrap();
        let x2 = mat_mul(&x, &x);
        let id = OpType::Id.to_unitary_matrix().unwrap();
        assert_matrices_close(&x2, &id, 1e-10, "X² ≠ I");
    }

    #[test]
    fn test_s_squared_is_z() {
        let s = OpType::S.to_unitary_matrix().unwrap();
        let s2 = mat_mul(&s, &s);
        let z = OpType::Z.to_unitary_matrix().unwrap();
        assert_matrices_close(&s2, &z, 1e-10, "S² ≠ Z");
    }

    #[test]
    fn test_t_squared_is_s() {
        let t = OpType::T.to_unitary_matrix().unwrap();
        let t2 = mat_mul(&t, &t);
        let s = OpType::S.to_unitary_matrix().unwrap();
        assert_matrices_close(&t2, &s, 1e-10, "T² ≠ S");
    }

    #[test]
    fn test_rx_2pi_is_neg_identity() {
        let rx = OpType::Rx(Parameter::Const(2.0 * PI))
            .to_unitary_matrix()
            .unwrap();
        let neg_id = vec![
            vec![Complex64::new(-1.0, 0.0), Complex64::new(0.0, 0.0)],
            vec![Complex64::new(0.0, 0.0), Complex64::new(-1.0, 0.0)],
        ];
        assert_matrices_close(&rx, &neg_id, 1e-10, "Rx(2π) ≠ -I");
    }

    #[test]
    fn test_rx_4pi_is_identity() {
        let rx = OpType::Rx(Parameter::Const(4.0 * PI))
            .to_unitary_matrix()
            .unwrap();
        let id = OpType::Id.to_unitary_matrix().unwrap();
        assert_matrices_close(&rx, &id, 1e-10, "Rx(4π) ≠ I");
    }

    #[test]
    fn test_cnot_squared_is_identity() {
        let cnot = OpType::CNOT.to_unitary_matrix().unwrap();
        let cnot2 = mat_mul(&cnot, &cnot);
        let dim = 4;
        let mut id4 = vec![vec![Complex64::new(0.0, 0.0); dim]; dim];
        for i in 0..dim {
            id4[i][i] = Complex64::new(1.0, 0.0);
        }
        assert_matrices_close(&cnot2, &id4, 1e-10, "CNOT² ≠ I");
    }

    #[test]
    fn test_parameter_binding() {
        let p = Parameter::Variable {
            name: "theta".into(),
            id: 0,
        };
        let mut values = std::collections::HashMap::new();
        values.insert("theta".into(), 1.5);
        let bound = p.bind(&values);
        assert_eq!(bound, Parameter::Const(1.5));
    }

    #[test]
    fn test_parameter_expression() {
        let p = Parameter::Expr(Box::new(ParameterExpr::Add(
            Parameter::Const(1.0),
            Parameter::Const(2.0),
        )));
        assert!((p.evaluate() - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_gate_qubit_counts() {
        assert_eq!(OpType::H.n_qubits(), 1);
        assert_eq!(OpType::Rx(Parameter::Const(0.5)).n_qubits(), 1);
        assert_eq!(OpType::CNOT.n_qubits(), 2);
        assert_eq!(OpType::CZ.n_qubits(), 2);
        assert_eq!(OpType::CCX.n_qubits(), 3);
    }

    // ─── Test helpers ─────────────────────────────────────
    fn mat_mul(a: &[Vec<Complex64>], b: &[Vec<Complex64>]) -> Vec<Vec<Complex64>> {
        let n = a.len();
        let mut result = vec![vec![Complex64::new(0.0, 0.0); n]; n];
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    result[i][j] += a[i][k] * b[k][j];
                }
            }
        }
        result
    }

    fn assert_matrices_close(a: &[Vec<Complex64>], b: &[Vec<Complex64>], tol: f64, msg: &str) {
        assert_eq!(a.len(), b.len(), "{msg}: dimension mismatch");
        for i in 0..a.len() {
            for j in 0..a[i].len() {
                assert!(
                    (a[i][j] - b[i][j]).norm() < tol,
                    "{msg}: [{i}][{j}] = {:?} ≠ {:?} (diff={:.2e})",
                    a[i][j],
                    b[i][j],
                    (a[i][j] - b[i][j]).norm()
                );
            }
        }
    }
}
