#![allow(clippy::new_without_default)]
#![allow(clippy::needless_range_loop)]

pub mod commutative_cancel;
pub mod decompose;
pub mod fusion;
pub mod kak;
pub mod passes;
pub mod rotation_merge;
pub mod twirl;

use rand::Rng;
use sf_ir::QuantumDAG;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum CompilerError {
    #[error("Target backend {0} does not support {1} bits, circuit has {2}")]
    QubitMismatch(String, usize, usize),
    #[error("Encountered unsupported gate {0} after decomposition")]
    UnsupportedGate(String),
    #[error("Routing failed: {0}")]
    RoutingError(String),
}

/// Target hardware backend specification.
#[derive(Clone, Debug)]
pub struct BackendSpec {
    pub name: String,
    pub native_gates: Vec<String>,
    pub connectivity: Vec<(usize, usize)>,
    pub n_qubits: usize,
    pub optimization_level: u8,
    /// Seed for the level >= 2 Pauli twirl.  ``None`` keeps the historical
    /// behaviour (a fresh thread-RNG seed per compile), which makes the gate
    /// list of a level-2 compile differ between runs while the unitary does
    /// not.  Supplying a seed makes the whole output reproducible.
    pub twirl_seed: Option<u64>,
}

impl Default for BackendSpec {
    fn default() -> Self {
        Self {
            name: "generic".to_string(),
            native_gates: vec![
                "H".to_string(),
                "X".to_string(),
                "Y".to_string(),
                "Z".to_string(),
                "CNOT".to_string(),
            ],
            connectivity: vec![],
            n_qubits: 32,
            optimization_level: 1,
            twirl_seed: None,
        }
    }
}

/// Compilation passes for circuit transformation.
pub trait Pass {
    fn name(&self) -> &str;
    fn run(&self, dag: &mut QuantumDAG) -> Result<(), CompilerError>;
}

/// Orchestrates the sequence of compilation stages.
pub struct PassManager {
    passes: Vec<Box<dyn Pass>>,
}

impl PassManager {
    pub fn new() -> Self {
        Self { passes: Vec::new() }
    }

    pub fn add_pass(&mut self, pass: Box<dyn Pass>) {
        self.passes.push(pass);
    }

    pub fn run(&self, dag: &mut QuantumDAG) -> Result<(), CompilerError> {
        for pass in &self.passes {
            log::info!("Running pass: {}", pass.name());
            pass.run(dag)?;
        }
        Ok(())
    }
}

/// High-level compiler interface.
pub struct Compiler {
    pub backend: BackendSpec,
}

impl Compiler {
    pub fn new(backend: BackendSpec) -> Self {
        Self { backend }
    }

    /// Compile a circuit for the target backend using an optimized pipeline.
    pub fn compile(&self, dag: &QuantumDAG) -> Result<QuantumDAG, CompilerError> {
        let mut result = dag.clone_dag();

        let mut manager = PassManager::new();

        // 1. Initial Gate Cancellation (Remove identity ops)
        manager.add_pass(Box::new(passes::GateCancellationPass::new()));

        // 2. High-level Decomposition (e.g. SWAP -> 3 CNOTs)
        manager.add_pass(Box::new(decompose::BasicDecomposePass::new()));

        // 3. Backend-specific target decomposition
        let is_superconducting_basis = {
            let ng = &self.backend.native_gates;
            ng.contains(&"SX".to_string())
                && (ng.contains(&"CX".to_string()) || ng.contains(&"CNOT".to_string()))
        };
        if is_superconducting_basis {
            manager.add_pass(Box::new(
                decompose::superconducting::SuperconductingDecomposePass::new(),
            ));
        }

        // 4. Target-driven basis translation (handles any target gate set)
        if !self.backend.native_gates.is_empty() {
            manager.add_pass(Box::new(decompose::basis::BasisTranslationPass::new(
                &self.backend.native_gates,
            )));
        }

        // 5. Rotation merging (collapse consecutive Rz/Rx/Ry)
        manager.add_pass(Box::new(rotation_merge::RotationMergingPass::new()));

        // 6. Gate fusion: merge consecutive 1Q gates into a single U gate
        manager.add_pass(Box::new(fusion::GateFusionPass::new()));

        manager.run(&mut result)?;

        // 5. Routing: map logical qubits to hardware topology
        if !self.backend.connectivity.is_empty() {
            let coupling = sf_router::CouplingMap::from_edges(
                self.backend.n_qubits,
                &self.backend.connectivity,
            );
            let router = sf_router::Router::new(coupling);
            let (routed_dag, _mapping) = router
                .route(&result)
                .map_err(|e| CompilerError::RoutingError(e.to_string()))?;
            result = routed_dag;

            // After routing, re-run optimization on the routed DAG
            let mut post_manager = PassManager::new();
            post_manager.add_pass(Box::new(passes::GateCancellationPass::new()));
            post_manager.add_pass(Box::new(
                commutative_cancel::CommutativeCancellationPass::new(),
            ));
            post_manager.add_pass(Box::new(kak::KakSynthesisPass::new()));
            post_manager.add_pass(Box::new(rotation_merge::RotationMergingPass::new()));
            post_manager.add_pass(Box::new(fusion::GateFusionPass::new()));
            // Re-run basis translation after KAK (KAK emits CNOT which may not be native)
            if !self.backend.native_gates.is_empty() {
                post_manager.add_pass(Box::new(decompose::basis::BasisTranslationPass::new(
                    &self.backend.native_gates,
                )));
            }
            post_manager.add_pass(Box::new(passes::GateCancellationPass::new()));
            post_manager.run(&mut result)?;
        }

        // 6. Pauli twirling at optimization level >= 2
        if self.backend.optimization_level >= 2 {
            let seed = self
                .backend
                .twirl_seed
                .unwrap_or_else(|| rand::thread_rng().gen());
            let twirl_pass = twirl::PauliTwirlPass::new(seed);
            let mut twirl_manager = PassManager::new();
            twirl_manager.add_pass(Box::new(twirl_pass));
            twirl_manager.run(&mut result)?;
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sf_ir::OpType;

    #[test]
    fn test_gate_cancellation() -> Result<(), CompilerError> {
        let mut dag = QuantumDAG::new(1, 0);
        dag.add_op(OpType::H, &[0]);
        dag.add_op(OpType::H, &[0]);
        assert_eq!(dag.gate_count(), 2);

        let compiler = Compiler::new(BackendSpec::default());
        let compiled = compiler.compile(&dag)?;

        // H * H should cancel out
        assert_eq!(compiled.gate_count(), 0);
        Ok(())
    }

    #[test]
    fn test_swap_decomposition() -> Result<(), CompilerError> {
        let mut dag = QuantumDAG::new(2, 0);
        dag.add_op(OpType::SWAP, &[0, 1]);
        assert_eq!(dag.gate_count(), 1);

        let compiler = Compiler::new(BackendSpec::default());
        let compiled = compiler.compile(&dag)?;

        // SWAP should become 3 CNOTs
        assert_eq!(compiled.gate_count(), 3);
        assert_eq!(compiled.count_ops_of_type("CNOT"), 3);
        Ok(())
    }

    #[test]
    fn test_mixed_optimization() -> Result<(), CompilerError> {
        let mut dag = QuantumDAG::new(2, 0);
        dag.add_op(OpType::H, &[0]);
        dag.add_op(OpType::H, &[0]); // Cancels
        dag.add_op(OpType::SWAP, &[0, 1]); // 3 CNOTs
        dag.add_op(OpType::X, &[1]);
        dag.add_op(OpType::X, &[1]); // Cancels

        let compiler = Compiler::new(BackendSpec::default());
        let compiled = compiler.compile(&dag)?;

        assert_eq!(compiled.gate_count(), 3);
        Ok(())
    }

    #[test]
    fn test_compile_with_connectivity_constraints() -> Result<(), CompilerError> {
        let backend = BackendSpec {
            n_qubits: 6,
            connectivity: vec![(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)],
            ..Default::default()
        };

        let mut dag = QuantumDAG::new(6, 0);
        dag.add_op(OpType::H, &[0]);
        dag.add_op(OpType::CNOT, &[0, 5]);
        dag.add_op(OpType::CNOT, &[5, 0]);
        dag.add_op(OpType::CNOT, &[0, 3]);

        let compiler = Compiler::new(backend);
        let compiled = compiler.compile(&dag)?;

        assert_eq!(compiled.n_qubits, 6);
        // Compiled circuit must produce some gates (routing + optimization may reduce count)
        assert!(compiled.gate_count() > 0);
        Ok(())
    }

    #[test]
    fn test_cz_basis_compilation() -> Result<(), CompilerError> {
        let backend = BackendSpec {
            name: "cz_test".to_string(),
            native_gates: vec!["RZ".into(), "SX".into(), "X".into(), "CZ".into()],
            connectivity: vec![(0, 1)],
            n_qubits: 2,
            optimization_level: 1,
            twirl_seed: None,
        };

        let mut dag = QuantumDAG::new(2, 0);
        dag.add_op(OpType::H, &[0]);
        dag.add_op(OpType::CNOT, &[0, 1]);

        let compiler = Compiler::new(backend);
        let compiled = compiler.compile(&dag)?;

        assert_eq!(
            compiled.count_ops_of_type("CNOT"),
            0,
            "No CNOT should remain in CZ basis, got {} CNOTs. Total gates: {}",
            compiled.count_ops_of_type("CNOT"),
            compiled.gate_count()
        );
        assert!(
            compiled.count_ops_of_type("CZ") > 0 || compiled.gate_count() > 0,
            "Should have CZ gates or fused equivalents"
        );
        Ok(())
    }

    /// Diagnostic + regression: the multi-trial router must not change the
    /// relative order of a 1q gate and the 2q gate that follows it.  The
    /// backward SABRE pass traverses `rev_gates`, so returning its DAG yields
    /// the program executed in reverse; this pins the order down.
    #[test]
    fn test_router_preserves_gate_order() {
        fn show(tag: &str, dag: &QuantumDAG) {
            let recs: Vec<String> = dag
                .to_gate_records()
                .into_iter()
                .map(|(n, q, _)| format!("{}{:?}", n, q))
                .collect();
            println!("  {:<26} {}", tag, recs.join(" "));
        }

        let backend = BackendSpec {
            name: "path5".to_string(),
            native_gates: vec!["RZ".into(), "SX".into(), "X".into(), "CX".into()],
            connectivity: vec![(0, 1), (1, 2), (2, 3), (3, 4)],
            n_qubits: 5,
            optimization_level: 0,
            twirl_seed: None,
        };

        for (label, first_2q_input) in [("H(0) then CX(0,4)", false), ("CX(0,4) then H(0)", true)] {
            println!("\n=== {label}");
            let mut dag = QuantumDAG::new(5, 0);
            if first_2q_input {
                dag.add_op(OpType::CNOT, &[0, 4]);
                dag.add_op(OpType::H, &[0]);
            } else {
                dag.add_op(OpType::H, &[0]);
                dag.add_op(OpType::CNOT, &[0, 4]);
            }

            let mut result = dag.clone_dag();
            let mut pre = PassManager::new();
            pre.add_pass(Box::new(passes::GateCancellationPass::new()));
            pre.add_pass(Box::new(decompose::BasicDecomposePass::new()));
            pre.add_pass(Box::new(
                decompose::superconducting::SuperconductingDecomposePass::new(),
            ));
            pre.add_pass(Box::new(decompose::basis::BasisTranslationPass::new(
                &backend.native_gates,
            )));
            pre.add_pass(Box::new(rotation_merge::RotationMergingPass::new()));
            pre.add_pass(Box::new(fusion::GateFusionPass::new()));
            pre.run(&mut result).expect("pre passes");
            show("after pre passes", &result);

            let coupling =
                sf_router::CouplingMap::from_edges(backend.n_qubits, &backend.connectivity);
            let router = sf_router::Router::new(coupling.clone());
            let (routed, _mapping) = router.route(&result).expect("route");
            result = routed;
            show("after routing", &result);

            let steps: Vec<(&str, Box<dyn Pass>)> = vec![
                (
                    "cancellation",
                    Box::new(passes::GateCancellationPass::new()),
                ),
                (
                    "commutative_cancel",
                    Box::new(commutative_cancel::CommutativeCancellationPass::new()),
                ),
                ("kak", Box::new(kak::KakSynthesisPass::new())),
                (
                    "rotation_merge",
                    Box::new(rotation_merge::RotationMergingPass::new()),
                ),
                ("fusion", Box::new(fusion::GateFusionPass::new())),
                (
                    "basis_translation",
                    Box::new(decompose::basis::BasisTranslationPass::new(
                        &backend.native_gates,
                    )),
                ),
                (
                    "cancellation",
                    Box::new(passes::GateCancellationPass::new()),
                ),
            ];
            for (name, pass) in steps {
                pass.run(&mut result).expect("post pass");
                show(name, &result);
            }

            // The contract: a 1q gate that precedes the 2q gate in the source
            // must still precede it in the compiled circuit (and vice versa).
            // This is exactly what the backward SABRE DAG breaks.
            let mut first_1q = None;
            let mut first_2q = None;
            for (i, (name, qubits, _)) in result.to_gate_records().iter().enumerate() {
                match qubits.len() {
                    1 if first_1q.is_none() => first_1q = Some(i),
                    2 if first_2q.is_none() => first_2q = Some(i),
                    _ => {
                        let _ = name;
                    }
                }
            }
            let (i1, i2) = (first_1q.unwrap(), first_2q.unwrap());
            if first_2q_input {
                assert!(
                    i2 < i1,
                    "{label}: the 2q gate moves before the 1q gate that must follow it \
                     (2q at {}, 1q at {})",
                    i2,
                    i1
                );
            } else {
                assert!(
                    i1 < i2,
                    "{label}: the 1q gate moves after the 2q gate that must follow it \
                     (1q at {}, 2q at {})",
                    i1,
                    i2
                );
            }
        }
    }

    /// Randomised property search over the router (the path that carried the
    /// reversed-DAG defect).  A seeded LCG builds 30 circuits per topology;
    /// each must come back with
    ///   * every two-qubit gate on a coupling edge,
    ///   * every wire inside the device,
    ///   * the same unitary up to a wire relabelling (the multi-trial router
    ///     may start from a random layout, so all 5! relabellings are tried),
    ///   * identical output when the same seed is run twice.
    #[test]
    fn test_random_programmes_route_to_equivalent_circuits() {
        use num_complex::Complex64;

        struct Lcg(u64);
        impl Lcg {
            fn next(&mut self) -> u64 {
                self.0 = self
                    .0
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                self.0 >> 33
            }
            fn below(&mut self, n: u64) -> u64 {
                self.next() % n
            }
        }

        fn wire_perms() -> Vec<[usize; 5]> {
            fn go(k: usize, cur: &mut [usize; 5], used: &mut [bool; 5], out: &mut Vec<[usize; 5]>) {
                if k == 5 {
                    out.push(*cur);
                    return;
                }
                for w in 0..5 {
                    if !used[w] {
                        used[w] = true;
                        cur[k] = w;
                        go(k + 1, cur, used, out);
                        used[w] = false;
                    }
                }
            }
            let mut out = Vec::new();
            go(0, &mut [0; 5], &mut [false; 5], &mut out);
            out
        }

        /// Basis-index permutation induced by a wire permutation.
        fn permute_index(i: usize, sigma: &[usize; 5]) -> usize {
            let mut out = 0;
            for q in 0..5 {
                if (i >> q) & 1 == 1 {
                    out |= 1 << sigma[q];
                }
            }
            out
        }

        /// Smallest max|U_out[i,j] - U_in[out^-1(i), in^-1(j)]| over all input
        /// relabellings.  The router may start a trial on a random layout, so
        /// the left (output) side is fixed by the mapping it returns and only
        /// the right (input) side is unknown.
        fn best_input_relabel_error(
            u_in: &nalgebra::DMatrix<Complex64>,
            u_out: &nalgebra::DMatrix<Complex64>,
            sigma_out: &[usize; 5],
            perms: &[[usize; 5]],
        ) -> f64 {
            let mut inv_out = [0usize; 5];
            for q in 0..5 {
                inv_out[sigma_out[q]] = q;
            }
            let mut best = f64::INFINITY;
            for sigma_in in perms {
                let mut inv_in = [0usize; 5];
                for q in 0..5 {
                    inv_in[sigma_in[q]] = q;
                }
                let mut worst: f64 = 0.0;
                for i in 0..u_out.nrows() {
                    let ii = permute_index(i, &inv_out);
                    for j in 0..u_out.ncols() {
                        let jj = permute_index(j, &inv_in);
                        let d = (u_out[(i, j)] - u_in[(ii, jj)]).norm();
                        if d > worst {
                            worst = d;
                        }
                    }
                    if worst >= best {
                        break;
                    }
                }
                if worst < best {
                    best = worst;
                }
                if best < 1e-12 {
                    break;
                }
            }
            best
        }

        let topologies: [(&str, Vec<(usize, usize)>); 2] = [
            ("path5", vec![(0, 1), (1, 2), (2, 3), (3, 4)]),
            ("tee5", vec![(0, 1), (1, 2), (2, 3), (1, 4)]),
        ];
        let perms = wire_perms();

        let mut checked = 0usize;
        for (topology, edges) in topologies {
            let coupling = sf_router::CouplingMap::from_edges(5, &edges);
            let mut rng = Lcg(0x5EED_1234_ABCD_0001);
            for case in 0..30 {
                let depth = 1 + rng.below(8) as usize;
                let mut dag = QuantumDAG::new(5, 0);
                for _ in 0..depth {
                    if rng.below(100) < 45 {
                        let q = rng.below(5) as usize;
                        let theta =
                            sf_ir::Parameter::Const(0.1 + 0.7 * (rng.below(1000) as f64) / 1000.0);
                        let op = match rng.below(9) {
                            0 => OpType::H,
                            1 => OpType::X,
                            2 => OpType::Y,
                            3 => OpType::Z,
                            4 => OpType::S,
                            5 => OpType::T,
                            6 => OpType::Rx(theta),
                            7 => OpType::Ry(theta),
                            _ => OpType::Rz(theta),
                        };
                        dag.add_op(op, &[q]);
                    } else {
                        let a = rng.below(5) as usize;
                        let mut b = rng.below(5) as usize;
                        while b == a {
                            b = rng.below(5) as usize;
                        }
                        let op = match rng.below(3) {
                            0 => OpType::CNOT,
                            1 => OpType::CZ,
                            _ => OpType::SWAP,
                        };
                        dag.add_op(op, &[a, b]);
                    }
                }

                let router = sf_router::Router::new(coupling.clone());
                let (routed, mapping) = router
                    .route(&dag)
                    .unwrap_or_else(|e| panic!("{topology} case {case}: route failed: {e}"));
                let mut sigma_out = [0usize; 5];
                for q in 0..5 {
                    sigma_out[q] = mapping.logical_to_physical(sf_ir::LogicalQubit(q)).0;
                }

                for (name, qubits, _) in routed.to_gate_records() {
                    assert!(
                        qubits.iter().all(|&q| q < 5),
                        "{topology} case {case}: {name} left the device: {qubits:?}"
                    );
                    if qubits.len() == 2 {
                        let (a, b) = (qubits[0], qubits[1]);
                        assert!(
                            edges.contains(&(a, b)) || edges.contains(&(b, a)),
                            "{topology} case {case}: {name} on non-adjacent {a},{b}"
                        );
                    }
                }

                let err = best_input_relabel_error(
                    &dag.to_unitary(),
                    &routed.to_unitary(),
                    &sigma_out,
                    &perms,
                );
                let script: Vec<String> = dag
                    .to_gate_records()
                    .into_iter()
                    .map(|(n, q, _)| format!("{}{:?}", n, q))
                    .collect();
                assert!(
                    err < 1e-9,
                    "{topology} case {case}: routed unitary differs by {err:.3e} \
                     under every wire relabelling; input {script:?}"
                );

                let (again, _again_mapping) = sf_router::Router::new(coupling.clone())
                    .route(&dag)
                    .expect("second route");
                let first: Vec<_> = routed
                    .to_gate_records()
                    .into_iter()
                    .map(|(n, q, _)| (n, q))
                    .collect();
                let second: Vec<_> = again
                    .to_gate_records()
                    .into_iter()
                    .map(|(n, q, _)| (n, q))
                    .collect();
                assert_eq!(first, second, "{topology} case {case}: not deterministic");
                checked += 1;
            }
        }
        // Metric sanity: a genuinely different circuit must be rejected, or a
        // passing run above would mean nothing.  U and U^dagger have identical
        // |entries|, so this also rules out an absolute-value comparison.
        let mut probe = QuantumDAG::new(5, 0);
        probe.add_op(OpType::H, &[0]);
        probe.add_op(OpType::Rz(sf_ir::Parameter::Const(0.37)), &[0]);
        probe.add_op(OpType::CNOT, &[0, 1]);
        let probe_u = probe.to_unitary();
        let identity = [0usize, 1, 2, 3, 4];
        let wrong = best_input_relabel_error(&probe_u.adjoint(), &probe_u, &identity, &perms);
        assert!(
            wrong > 1e-3,
            "the relabelling metric accepts U^dagger (err {wrong:.3e})"
        );

        println!("{checked} random circuits routed and checked");
    }

    #[test]
    fn test_compiled_circuit_preserves_qubit_count() -> Result<(), CompilerError> {
        let backend = BackendSpec {
            n_qubits: 5,
            connectivity: vec![(0, 1), (1, 2), (2, 3), (3, 4)],
            ..Default::default()
        };

        let mut dag = QuantumDAG::new(5, 0);
        dag.add_op(OpType::H, &[0]);
        dag.add_op(OpType::CNOT, &[0, 4]);
        dag.add_op(OpType::SWAP, &[1, 3]);

        let compiler = Compiler::new(backend);
        let compiled = compiler.compile(&dag)?;

        assert_eq!(compiled.n_qubits, 5);
        assert_eq!(dag.n_qubits, compiled.n_qubits);
        Ok(())
    }
}
