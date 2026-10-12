//! Representation router -- the adaptive cost model at the heart of `cat`.
//!
//! ## Why this exists
//!
//! No single simulation representation wins every workload.  Dense
//! statevector wins on shallow arbitrary circuits; the stabilizer tableau
//! wins overwhelmingly on Clifford circuits; a low-rank stabilizer
//! decomposition (Bravyi-Gosset 2019, *Simulation of quantum circuits by
//! low-rank stabilizer decompositions*) turns a small number of `T` gates
//! into a polynomial-cost problem where dense simulation is exponential.
//! The research lever is therefore not a faster kernel but a *representation
//! switch*: classify the circuit, then execute it in the cheapest regime.
//!
//! This module is the classifier + cost model.  It is deliberately pure data
//! (no simulation), so it can be unit-tested without a host and so the
//! dispatch seam ([`crate::dispatch`]) can consult it before choosing a path.
//!
//! ## Honesty note
//!
//! `select` reports the representation a *cost model* says is best; it does
//! not claim `cat` can execute every one today.  [`Representation::available_in_cat`]
//! records the execution status so callers surface an explicit error instead
//! of silently degrading to a wrong-but-cheap path.

use sf_ir::{OpType, QuantumDAG};

/// A simulation representation the router can choose between.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Representation {
    /// Aaronson-Gottesman stabilizer tableau.  Exact, polynomial, but only
    /// valid when every gate is Clifford.
    Tableau,
    /// Low-rank stabilizer decomposition.  Exact for `m` magic gates at cost
    /// scaling with `rank <= 2^m`, polynomial in `n` -- the regime where dense
    /// simulation is exponential.
    StabRank { magic_ops: usize, rank_bound: u128 },
    /// Dense complex128 statevector (`16 * 2^n` bytes).
    DenseSv,
    /// Matrix-product state / tensor network.  Approximate; the only option
    /// once both dense and low-rank regimes are out of reach.
    Mps,
}

impl Representation {
    /// Whether `cat` can execute this representation *today*.
    ///
    /// Only the dense statevector handle is wired in the current milestone;
    /// the others are surfaced as explicit errors by [`crate::dispatch`] so a
    /// routing decision never silently returns an unexecuted result.
    pub fn available_in_cat(self) -> bool {
        matches!(self, Representation::DenseSv)
    }

    /// Short stable name, for logs / error messages.
    pub fn name(self) -> &'static str {
        match self {
            Representation::Tableau => "tableau",
            Representation::StabRank { .. } => "stab-rank",
            Representation::DenseSv => "dense-sv",
            Representation::Mps => "mps",
        }
    }
}

/// Structural profile of a circuit, extracted once from a [`QuantumDAG`].
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub struct CircuitProfile {
    pub n_qubits: usize,
    /// Clifford gates in the supported set (see [`is_supported_clifford`]).
    pub clifford_ops: usize,
    /// Non-Clifford, non-parameterised gates (`T`, `Tdg`, `CCX`, `CSWAP`,
    /// opaque `Custom`/`Unitary`, and gates the tableau cannot apply like
    /// `ISWAP`/`ECR`, treated conservatively as magic).
    pub magic_ops: usize,
    /// Parameterised rotations (`Rx`, `Rz`, `Rzz`, ...).  With symbolic
    /// parameters these are non-Clifford; a *constant* Clifford angle could
    /// in principle be recognised, which is deliberately left conservative
    /// here (we classify them as magic rather than risk a wrong tableau).
    pub param_ops: usize,
    pub two_qubit_ops: usize,
    pub measurements: usize,
    pub resets: usize,
}

impl CircuitProfile {
    /// True when every unitary gate is in the supported Clifford set, so the
    /// tableau is an *exact* representation.
    pub fn is_clifford(&self) -> bool {
        self.magic_ops == 0 && self.param_ops == 0
    }

    /// Total non-Clifford unitary gates (magic sources).
    pub fn magic_total(&self) -> usize {
        self.magic_ops + self.param_ops
    }

    /// True when the circuit measures qubits (relevant to sampler routing,
    /// not to representability).
    pub fn is_sampling(&self) -> bool {
        self.measurements > 0
    }
}

/// Whether `ot` is a Clifford gate the `sf-ir` tableau can apply directly.
///
/// Deliberately conservative: gates that are *mathematically* Clifford but
/// not in the tableau's supported set (`ISWAP`, `ECR`, `SX`-family variants
/// the tableau may not expose) are excluded, so a `true` here never routes a
/// circuit to a tableau that would mis-apply a gate.
pub fn is_supported_clifford(ot: &OpType) -> bool {
    matches!(
        ot,
        OpType::H
            | OpType::X
            | OpType::Y
            | OpType::Z
            | OpType::S
            | OpType::Sdg
            | OpType::SX
            | OpType::SXdg
            | OpType::Id
            | OpType::CNOT
            | OpType::CZ
            | OpType::CY
            | OpType::SWAP
    )
}

/// Whether `ot` carries at least one angle parameter (hence magic when the
/// angle is symbolic).
pub fn is_parameterized(ot: &OpType) -> bool {
    matches!(
        ot,
        OpType::Rx(_)
            | OpType::Ry(_)
            | OpType::Rz(_)
            | OpType::R1(_)
            | OpType::U(..)
            | OpType::P(_)
            | OpType::Rzz(_)
            | OpType::Rxx(_)
            | OpType::Ryy(_)
            | OpType::CRx(_)
            | OpType::CRz(_)
            | OpType::CP(_)
            | OpType::Cu(..)
    )
}

/// Walk `dag` once and collect the structural [`CircuitProfile`].
pub fn classify(dag: &QuantumDAG) -> CircuitProfile {
    let mut p = CircuitProfile {
        n_qubits: dag.n_qubits,
        ..CircuitProfile::default()
    };
    for id in dag.topological_order() {
        let op = &dag.graph()[id];
        let ot = &op.op_type;
        // Boundary nodes (Input/Output) and barriers carry no unitary cost.
        if ot.is_boundary() || matches!(ot, OpType::Barrier) {
            continue;
        }
        match ot {
            OpType::Measure | OpType::MeasureAll => p.measurements += 1,
            OpType::Reset => p.resets += 1,
            _ => {
                let two_q = op.qubits.len() == 2;
                if two_q {
                    p.two_qubit_ops += 1;
                }
                if is_parameterized(ot) {
                    p.param_ops += 1;
                } else if is_supported_clifford(ot) {
                    p.clifford_ops += 1;
                } else {
                    p.magic_ops += 1;
                }
            }
        }
    }
    p
}

// ── Cost model ──────────────────────────────────────────────────────────────

/// Bytes for a dense complex128 statevector of `n` qubits.
pub fn dense_sv_bytes(n: usize) -> u128 {
    16u128.checked_shl(n as u32).unwrap_or(u128::MAX)
}

/// Bytes for a dense complex128 density matrix of `n` qubits.
pub fn dense_dm_bytes(n: usize) -> u128 {
    16u128.checked_shl((2 * n) as u32).unwrap_or(u128::MAX)
}

/// Bytes for a word-packed stabilizer tableau of `n` qubits (2n rows ×
/// ceil(n/64) u64s × 2 planes).
pub fn tableau_bytes(n: usize) -> u128 {
    let words = n.div_ceil(64) as u128;
    2 * (n as u128) * words * 8 * 2
}

/// Largest magic-gate count for which the `2^m`-rank decomposition is worth
/// attempting (an exponential in `m`, so this is a hard ceiling, not a
/// heuristic).
pub const STABRANK_MAX_MAGIC: usize = 24;

/// Dense statevector is chosen directly below this width because at small `n`
/// the decomposition's per-term bookkeeping overhead is not worth it.
pub const DENSE_SV_SMALL_N: usize = 22;

/// Decide the cheapest *exact-or-approximate* representation for `profile`
/// under a memory `budget_bytes`.
///
/// Decision order (documented so the choice is auditable, not magic):
///   1. all-Clifford                -> `Tableau`  (exact, poly, to 4096)
///   2. small magic + `n` not tiny  -> `StabRank` (exact, cost ~ rank^2 · poly(n))
///   3. dense statevector fits       -> `DenseSv`  (exact, any circuit)
///   4. otherwise                    -> `Mps`      (approximate fallback)
pub fn select(profile: &CircuitProfile, budget_bytes: u128) -> Representation {
    if profile.is_clifford() {
        return Representation::Tableau;
    }
    let magic = profile.magic_total();
    let rank_bound = 1u128 << magic.min(63);
    let stabrank_viable = magic <= STABRANK_MAX_MAGIC
        && (dense_sv_bytes(profile.n_qubits) > budget_bytes
            || profile.n_qubits >= DENSE_SV_SMALL_N);
    if stabrank_viable {
        return Representation::StabRank {
            magic_ops: magic,
            rank_bound,
        };
    }
    if dense_sv_bytes(profile.n_qubits) <= budget_bytes {
        return Representation::DenseSv;
    }
    Representation::Mps
}

#[cfg(test)]
mod tests {
    use super::*;
    use sf_ir::{OpType, Parameter, QuantumDAG};

    fn var(name: &str) -> Parameter {
        Parameter::Variable {
            name: name.into(),
            id: 0,
        }
    }

    fn budget_2gib() -> u128 {
        2 * 1024 * 1024 * 1024
    }

    #[test]
    fn clifford_circuit_routes_to_tableau() {
        let mut dag = QuantumDAG::new(6, 0);
        dag.add_op(OpType::H, &[0]);
        dag.add_op(OpType::CNOT, &[0, 1]);
        dag.add_op(OpType::S, &[2]);
        dag.add_op(OpType::CZ, &[1, 2]);
        dag.add_op(OpType::SWAP, &[3, 4]);
        let p = classify(&dag);
        assert!(p.is_clifford(), "profile should be pure Clifford: {p:?}");
        assert_eq!(p.clifford_ops, 5);
        assert_eq!(p.magic_total(), 0);
        assert_eq!(select(&p, budget_2gib()), Representation::Tableau);
    }

    #[test]
    fn few_t_gates_route_to_stab_rank_and_rank_bound_is_2_pow_m() {
        // n = 24 is at/above DENSE_SV_SMALL_N, so the decomposition wins on the
        // cost model even though dense (256 MiB) would still fit the budget.
        let mut dag = QuantumDAG::new(24, 0);
        dag.add_op(OpType::H, &[0]);
        dag.add_op(OpType::CNOT, &[0, 1]);
        dag.add_op(OpType::T, &[2]);
        dag.add_op(OpType::Tdg, &[3]);
        dag.add_op(OpType::T, &[4]);
        let p = classify(&dag);
        assert!(!p.is_clifford());
        assert_eq!(p.magic_total(), 3);
        match select(&p, budget_2gib()) {
            Representation::StabRank {
                magic_ops,
                rank_bound,
            } => {
                assert_eq!(magic_ops, 3);
                assert_eq!(rank_bound, 8);
            }
            other => panic!("expected StabRank, got {other:?}"),
        }
    }

    #[test]
    fn same_circuit_at_small_n_prefers_dense() {
        // The identical 3-magic circuit at n = 20 stays dense: below
        // DENSE_SV_SMALL_N the decomposition's overhead is not worth paying.
        let mut dag = QuantumDAG::new(20, 0);
        dag.add_op(OpType::H, &[0]);
        dag.add_op(OpType::CNOT, &[0, 1]);
        dag.add_op(OpType::T, &[2]);
        dag.add_op(OpType::Tdg, &[3]);
        dag.add_op(OpType::T, &[4]);
        let p = classify(&dag);
        assert_eq!(p.magic_total(), 3);
        assert_eq!(select(&p, budget_2gib()), Representation::DenseSv);
    }

    #[test]
    fn parameterised_rotations_count_as_magic_not_clifford() {
        let mut dag = QuantumDAG::new(8, 0);
        dag.add_op(OpType::Ry(var("x")), &[0]);
        dag.add_op(OpType::Rzz(var("y")), &[0, 1]);
        dag.add_op(OpType::H, &[2]);
        let p = classify(&dag);
        assert_eq!(p.param_ops, 2);
        assert_eq!(p.clifford_ops, 1);
        assert_eq!(p.magic_total(), 2);
        assert!(!p.is_clifford());
    }

    #[test]
    fn huge_non_clifford_circuit_falls_back_to_mps_over_budget() {
        // 40 qubits: dense sv is 16 * 2^40 = 16 TiB, far over a 2 GiB budget.
        let mut dag = QuantumDAG::new(40, 0);
        for q in 0..40 {
            dag.add_op(OpType::H, &[q]);
        }
        // Many magic gates so the decomposition ceiling is exceeded.
        for q in 0..30 {
            dag.add_op(OpType::T, &[q]);
        }
        let p = classify(&dag);
        assert!(p.magic_total() > STABRANK_MAX_MAGIC);
        assert!(dense_sv_bytes(p.n_qubits) > budget_2gib());
        assert_eq!(select(&p, budget_2gib()), Representation::Mps);
    }

    #[test]
    fn small_non_clifford_circuit_stays_dense() {
        let mut dag = QuantumDAG::new(8, 0);
        dag.add_op(OpType::Ry(var("x")), &[0]);
        dag.add_op(OpType::Ry(var("x")), &[1]);
        dag.add_op(OpType::Ry(var("x")), &[2]);
        dag.add_op(OpType::Ry(var("x")), &[3]);
        dag.add_op(OpType::Ry(var("x")), &[4]);
        dag.add_op(OpType::CNOT, &[0, 1]);
        let p = classify(&dag);
        // 5 magic gates, but n=8 is small: dense is exact and cheap, so the
        // router should not pay decomposition overhead.
        assert_eq!(p.magic_total(), 5);
        assert_eq!(select(&p, budget_2gib()), Representation::DenseSv);
    }

    #[test]
    fn measurements_and_resets_are_counted_but_do_not_break_clifford_routing() {
        let mut dag = QuantumDAG::new(3, 3);
        dag.add_op(OpType::H, &[0]);
        dag.add_op(OpType::CNOT, &[0, 1]);
        dag.add_measure(0, 0);
        dag.add_measure(1, 1);
        let p = classify(&dag);
        assert_eq!(p.measurements, 2);
        assert!(p.is_clifford());
        assert!(p.is_sampling());
        assert_eq!(select(&p, budget_2gib()), Representation::Tableau);
    }

    #[test]
    fn only_dense_sv_is_available_in_cat_today() {
        assert!(Representation::DenseSv.available_in_cat());
        assert!(!Representation::Tableau.available_in_cat());
        assert!(!Representation::Mps.available_in_cat());
        assert!(!Representation::StabRank {
            magic_ops: 1,
            rank_bound: 2
        }
        .available_in_cat());
    }

    #[test]
    fn cost_model_bytes_are_monotone_and_correct_at_small_n() {
        assert_eq!(dense_sv_bytes(1), 32);
        assert_eq!(dense_dm_bytes(1), 64);
        assert_eq!(dense_sv_bytes(20), 16 * 1024 * 1024);
        assert!(dense_sv_bytes(21) > dense_sv_bytes(20));
        assert!(dense_dm_bytes(10) > dense_sv_bytes(10));
        // tableau stays small far past the dense wall
        assert!(tableau_bytes(4096) < 64u128 * 1024 * 1024);
    }
}
