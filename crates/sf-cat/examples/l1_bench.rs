//! Lever-1 A/B example: Pauli-frame sampler vs sf-ir's tableau sampler,
//! same process, same op stream, interleaved rounds.
//!
//! Usage: `l1_bench <job.json>` where the job is
//! `{"n": 64, "ops": [["h",0],["cx",0,1],...], "shots": 2000, "reps": 7,
//!   "seed": 123}`.
//!
//! Prints one JSON line: per-lane build / first-sample (incl. plan build) /
//! sample-only medians, plus key counts and the number of random plan vars.
//! The Python driver (`sim4_bench_suite/_sf_cat_l1_ab.py`) interleaves this
//! process with stim lanes for the same-window A/B.
//!
//! With `"mode": "adj"` (fields: n, layers, theta, reps, fd_check) the same
//! JSON-line contract carries the adjoint lane instead: the cat compile-once
//! handle vs `sf_ir::adjoint::adjoint_grad` in one process, with a cat-vs-
//! sf-ir gradient parity check and an optional central finite-difference
//! arbiter.  The Python driver `_sf_cat_adj_ab.py` interleaves it with the
//! qulacs backprop lane.
//!
//! With `"mode": "adjops"` (fields: n, ops, theta, obs, reps, fd_check) the
//! same adjoint lanes run on an arbitrary RY/RZ(slot-indexed)+CNOT geometry
//! built from `ops`, with the observable from `obs` (an IXYZ string, default
//! Z^n).  Used by `sim4_bench_suite/_sf_cat_adj_scope_ab.py` for ansatz-
//! geometry scope measurements.
//!
//! With `"mode": "dm"` (fields: n, ops, reps) the same JSON-line contract
//! carries the sf-ir density-matrix lane: `QuantumDAG::simulate_dm` (the
//! streaming pair-pass engine) vs the statevector engine on identical ops,
//! with layout-invariant gates (trace ~1, purity ~1, rho diagonal == |psi|^2).
//! `cat`'s own DM mode is a deliberate hold-out (R6/R8 measured no same-window
//! win vs the routed aer path), so this lane benching the sf-ir engine is
//! disclosed as such in the report -- never reframed as a cat-mode result.

use num_complex::Complex64;
use sf_cat::adj::AdjProgram;
use sf_cat::paulframe::FrameTableau;
use sf_ir::adjoint::PauliTerm;
use sf_ir::{OpType, Parameter, QuantumDAG};

fn median_ms(mut xs: Vec<f64>) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let m = xs.len() / 2;
    if xs.len() % 2 == 1 {
        xs[m]
    } else {
        0.5 * (xs[m - 1] + xs[m])
    }
}

fn parse_ops(v: &serde_json::Value) -> Vec<(String, Vec<usize>)> {
    v.as_array()
        .expect("ops must be an array")
        .iter()
        .map(|op| {
            let arr = op.as_array().expect("op must be an array");
            let name = arr[0].as_str().expect("op name must be a string");
            let qubits: Vec<usize> = arr[1..]
                .iter()
                .map(|q| q.as_u64().expect("qubit must be an int") as usize)
                .collect();
            // sf-ir's from_gate_list matches upper-case names; normalize here.
            (name.to_uppercase(), qubits)
        })
        .collect()
}

fn main() {
    let path = std::env::args().nth(1).expect("usage: l1_bench <job.json>");
    let raw = std::fs::read_to_string(&path).expect("read job json");
    let job: serde_json::Value = serde_json::from_str(&raw).expect("parse job json");

    if job["mode"].as_str() == Some("adj") {
        println!("{}", run_adj(&job));
        return;
    }

    if job["mode"].as_str() == Some("adjops") {
        println!("{}", run_adjops(&job));
        return;
    }

    if job["mode"].as_str() == Some("dm") {
        println!("{}", run_dm(&job));
        return;
    }

    let n = job["n"].as_u64().expect("n") as usize;
    let shots = job["shots"].as_u64().expect("shots") as usize;
    let reps = job["reps"].as_u64().unwrap_or(7) as usize;
    let seed = job["seed"].as_u64().unwrap_or(7);
    let gates = parse_ops(&job["ops"]);
    let rounds = 4usize;

    let mut out = serde_json::json!({
        "n": n,
        "shots": shots,
        "reps": reps,
        "seed": seed,
        "rayon_threads": rayon::current_num_threads(),
        "ops_len": gates.len(),
    });

    // ── Lane cat_pf (this crate) ──
    let mut b_build = Vec::new();
    let mut b_first = Vec::new();
    let mut b_pack = Vec::new();
    let mut b_counts = Vec::new();
    let mut b_cold = Vec::new();
    let mut b_plan = Vec::new();
    let mut vars = 0usize;
    let mut keys_cat = 0usize;
    for r in 0..rounds {
        if r % 2 == 0 {
            let t_cold = std::time::Instant::now();
            let t = std::time::Instant::now();
            let tab = FrameTableau::from_gate_list(n, &gates).expect("cat build");
            b_build.push(t.elapsed().as_secs_f64() * 1e3);
            let t = std::time::Instant::now();
            let c = tab.sample(shots, Some(seed));
            b_first.push(t.elapsed().as_secs_f64() * 1e3);
            b_cold.push(t_cold.elapsed().as_secs_f64() * 1e3);
            keys_cat = c.len();
            vars = tab.random_var_count();
            let t = std::time::Instant::now();
            let packed = tab.sample_packed(shots, Some(seed));
            b_pack.push(t.elapsed().as_secs_f64() * 1e3);
            assert_eq!(packed.len(), shots * tab.words());
            let t = std::time::Instant::now();
            let c2 = tab.sample(shots, Some(seed));
            b_counts.push(t.elapsed().as_secs_f64() * 1e3);
            assert_eq!(c2.len(), keys_cat);
            // Attribution: plan build alone (tableau build excluded).
            let tab3 = FrameTableau::from_gate_list(n, &gates).expect("cat build attr");
            let t = std::time::Instant::now();
            let _ = tab3.random_var_count();
            b_plan.push(t.elapsed().as_secs_f64() * 1e3);
        } else {
            let t_cold = std::time::Instant::now();
            let t = std::time::Instant::now();
            let tab = FrameTableau::from_gate_list(n, &gates).expect("cat build");
            b_build.push(t.elapsed().as_secs_f64() * 1e3);
            let t = std::time::Instant::now();
            let _ = tab.sample(shots, Some(seed));
            b_first.push(t.elapsed().as_secs_f64() * 1e3);
            b_cold.push(t_cold.elapsed().as_secs_f64() * 1e3);
            let t = std::time::Instant::now();
            let _ = tab.sample_packed(shots, Some(seed));
            b_pack.push(t.elapsed().as_secs_f64() * 1e3);
            let t = std::time::Instant::now();
            let _ = tab.sample(shots, Some(seed));
            b_counts.push(t.elapsed().as_secs_f64() * 1e3);
        }
    }
    out["cat_pf"] = serde_json::json!({
        "cold_ms": median_ms(b_cold),
        "plan_ms": median_ms(b_plan),
        "build_ms": median_ms(b_build),
        "first_sample_ms": median_ms(b_first),
        "sample_pack_ms": median_ms(b_pack),
        "sample_counts_ms": median_ms(b_counts),
        "keys": keys_cat,
        "random_vars": vars,
        "pack_kernel": sf_cat::kernels::current_kernel(),
    });

    // ── Lane sf_ir (sf-ir's StabilizerTableau; capped at 4096 by its own
    // assert, so only run when n fits) ──
    if n <= 4096 {
        let mut b_build = Vec::new();
        let mut b_first = Vec::new();
        let mut b_sample = Vec::new();
        let mut b_cold = Vec::new();
        let mut keys_ir = 0usize;
        for r in 0..rounds {
            if r % 2 == 1 {
                let t_cold = std::time::Instant::now();
                let t = std::time::Instant::now();
                let tab = sf_ir::stabilizer::StabilizerTableau::from_gate_list(n, &gates)
                    .expect("sf-ir build");
                b_build.push(t.elapsed().as_secs_f64() * 1e3);
                let t = std::time::Instant::now();
                let c = tab.sample(shots, Some(seed));
                b_first.push(t.elapsed().as_secs_f64() * 1e3);
                b_cold.push(t_cold.elapsed().as_secs_f64() * 1e3);
                keys_ir = c.len();
                let t = std::time::Instant::now();
                let c2 = tab.sample(shots, Some(seed));
                b_sample.push(t.elapsed().as_secs_f64() * 1e3);
                assert_eq!(c2.len(), keys_ir);
            } else {
                let t_cold = std::time::Instant::now();
                let t = std::time::Instant::now();
                let tab = sf_ir::stabilizer::StabilizerTableau::from_gate_list(n, &gates)
                    .expect("sf-ir build");
                b_build.push(t.elapsed().as_secs_f64() * 1e3);
                let t = std::time::Instant::now();
                let _ = tab.sample(shots, Some(seed));
                b_first.push(t.elapsed().as_secs_f64() * 1e3);
                b_cold.push(t_cold.elapsed().as_secs_f64() * 1e3);
                let t = std::time::Instant::now();
                let _ = tab.sample(shots, Some(seed));
                b_sample.push(t.elapsed().as_secs_f64() * 1e3);
            }
        }
        out["sf_ir"] = serde_json::json!({
            "cold_ms": median_ms(b_cold),
            "build_ms": median_ms(b_build),
            "first_sample_ms": median_ms(b_first),
            "sample_ms": median_ms(b_sample),
            "keys": keys_ir,
        });
    } else {
        out["sf_ir"] = serde_json::json!({"status": "cap_refused (sf-ir assert n<=4096)"});
    }

    println!("{}", out);
}

/// Adjoint lane: the cat compile-once handle vs `sf_ir::adjoint::adjoint_grad`
/// on the identical ladder ansatz (worker `adj_layout`), observable Z^n.
/// One JSON line; timings are medians over `reps` calls, the slower sf-ir
/// setup (DAG build + parameter map) is reported separately from per-call
/// time so the comparison is call-for-call.
fn run_adj(job: &serde_json::Value) -> serde_json::Value {
    let n = job["n"].as_u64().expect("n") as usize;
    let layers = job["layers"].as_u64().expect("layers") as usize;
    let reps = job["reps"].as_u64().unwrap_or(7) as usize;
    let p = 2 * n * layers;
    let theta: Vec<f64> = job["theta"]
        .as_array()
        .expect("theta array")
        .iter()
        .map(|v| v.as_f64().expect("theta value"))
        .collect();
    assert_eq!(theta.len(), p, "theta length must equal 2*n*layers");
    let fd_check = job["fd_check"].as_bool().unwrap_or(false);
    let terms = vec![PauliTerm {
        paulis: vec![3u8; n],
        coef: Complex64::new(1.0, 0.0),
    }];

    // ── Lane cat: compile-once handle ──
    let t = std::time::Instant::now();
    let mut prog = AdjProgram::from_ladder(n, layers);
    let compile_ms = t.elapsed().as_secs_f64() * 1e3;
    let t_cold = std::time::Instant::now();
    let _ = prog.grad(&theta, &terms).expect("cat grad");
    let cold_ms = t_cold.elapsed().as_secs_f64() * 1e3;
    let mut t_cat = Vec::with_capacity(reps);
    let mut g_cat = Vec::new();
    for _ in 0..reps {
        let t = std::time::Instant::now();
        g_cat = prog.grad(&theta, &terms).expect("cat grad").to_vec();
        t_cat.push(t.elapsed().as_secs_f64() * 1e3);
    }
    let expval = prog.expectation(&theta, &terms).expect("cat expval");

    // ── Lane sf_ir: the same DAG the wheel lane lowers, in-process ──
    let t = std::time::Instant::now();
    let mut dag = QuantumDAG::new(n, 0);
    let mut slot = 0usize;
    for _ in 0..layers {
        for i in 0..n {
            dag.add_op(
                OpType::Ry(Parameter::Variable {
                    name: format!("t{slot}"),
                    id: slot,
                }),
                &[i],
            );
            slot += 1;
            dag.add_op(
                OpType::Rz(Parameter::Variable {
                    name: format!("t{slot}"),
                    id: slot,
                }),
                &[i],
            );
            slot += 1;
        }
        for i in 0..n.saturating_sub(1) {
            dag.add_op(OpType::CNOT, &[i, i + 1]);
        }
    }
    let dag_build_ms = t.elapsed().as_secs_f64() * 1e3;
    let values: std::collections::HashMap<String, f64> =
        (0..p).map(|i| (format!("t{i}"), theta[i])).collect();
    let t = std::time::Instant::now();
    let r = sf_ir::adjoint::adjoint_grad(&dag, &terms, &values).expect("sf_ir adj");
    let setup_ms = t.elapsed().as_secs_f64() * 1e3;
    let g_ir = r.gradients;
    let mut t_ir = Vec::with_capacity(reps);
    for _ in 0..reps {
        let t = std::time::Instant::now();
        let _ = sf_ir::adjoint::adjoint_grad(&dag, &terms, &values).expect("sf_ir adj");
        t_ir.push(t.elapsed().as_secs_f64() * 1e3);
    }

    // ── Cross-checks: house-engine parity + optional central FD ──
    let parity_max_diff = g_cat
        .iter()
        .zip(g_ir.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    let mut fd_max_err = f64::NAN;
    if fd_check {
        let eps = 1e-5;
        let mut fd = Vec::with_capacity(p);
        for i in 0..p {
            let mut tp = theta.clone();
            tp[i] += eps;
            let mut tm = theta.clone();
            tm[i] -= eps;
            let ep = prog.expectation(&tp, &terms).unwrap();
            let em = prog.expectation(&tm, &terms).unwrap();
            fd.push((ep - em) / (2.0 * eps));
        }
        fd_max_err = g_cat
            .iter()
            .zip(fd.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
    }
    let grad_norm = g_cat.iter().map(|v| v * v).sum::<f64>().sqrt();
    let grad_head: Vec<f64> = g_cat
        .iter()
        .take(4)
        .map(|v| (v * 1e6).round() / 1e6)
        .collect();

    serde_json::json!({
        "mode": "adj",
        "n": n,
        "layers": layers,
        "p": p,
        "reps": reps,
        "rayon_threads": rayon::current_num_threads(),
        "cat": {
            "compile_ms": compile_ms,
            "cold_ms": cold_ms,
            "grad_ms": median_ms(t_cat),
        },
        "sf_ir": {
            "dag_build_ms": dag_build_ms,
            "setup_ms": setup_ms,
            "grad_ms": median_ms(t_ir),
        },
        "parity_max_diff": parity_max_diff,
        "fd_check": fd_check,
        "fd_eps": 1e-5,
        "fd_max_err": if fd_check {
            serde_json::json!(fd_max_err)
        } else {
            serde_json::Value::Null
        },
        "expval": expval,
        "grad_norm": grad_norm,
        "grad_head": grad_head,
        "grad": g_cat,
    })
}

/// Adjoint lane, arbitrary geometry: same JSON-line contract as `"adj"`, but
/// the circuit comes from `"ops"` (RY/RZ entries carry `[qubit, slot]`) and
/// the observable from `"obs"` (an IXYZ string, default Z^n).  Both lanes get
/// the identical DAG, so parity is call-for-call.
fn run_adjops(job: &serde_json::Value) -> serde_json::Value {
    let n = job["n"].as_u64().expect("n") as usize;
    let reps = job["reps"].as_u64().unwrap_or(7) as usize;
    let fd_check = job["fd_check"].as_bool().unwrap_or(false);
    let theta: Vec<f64> = job["theta"]
        .as_array()
        .expect("theta array")
        .iter()
        .map(|v| v.as_f64().expect("theta value"))
        .collect();
    let obs = job["obs"]
        .as_str()
        .map(str::to_string)
        .unwrap_or_else(|| "Z".repeat(n));
    assert_eq!(obs.len(), n, "obs length must equal n");
    let paulis: Vec<u8> = obs
        .chars()
        .map(|c| match c {
            'I' => 0u8,
            'X' => 1,
            'Y' => 2,
            'Z' => 3,
            other => panic!("adjops: bad observable char {other}"),
        })
        .collect();
    let terms = vec![PauliTerm {
        paulis,
        coef: Complex64::new(1.0, 0.0),
    }];
    let ops = parse_ops(&job["ops"]);

    // ── The shared DAG (both lanes) ──
    let t = std::time::Instant::now();
    let mut dag = QuantumDAG::new(n, 0);
    for (name, qs) in &ops {
        match name.as_str() {
            "RY" | "RZ" => {
                assert_eq!(qs.len(), 2, "RY/RZ ops must be [qubit, slot]");
                let slot = qs[1];
                let param = Parameter::Variable {
                    name: format!("t{slot}"),
                    id: slot,
                };
                let op = if name == "RY" {
                    OpType::Ry(param)
                } else {
                    OpType::Rz(param)
                };
                dag.add_op(op, &[qs[0]]);
            }
            "CX" | "CNOT" => {
                assert_eq!(qs.len(), 2, "CX ops must be [ctl, tgt]");
                dag.add_op(OpType::CNOT, &[qs[0], qs[1]]);
            }
            other => panic!("adjops: unsupported op {other}"),
        }
    }
    let p = theta.len();
    let names = dag.parameter_names();
    let expected: Vec<String> = (0..p).map(|i| format!("t{i}")).collect();
    assert_eq!(
        names, expected,
        "adjops: parameter names must be t0..tP-1 in order"
    );
    let dag_build_ms = t.elapsed().as_secs_f64() * 1e3;

    // ── Lane cat: compile-once handle ──
    let t = std::time::Instant::now();
    let mut prog = AdjProgram::from_dag(&dag).expect("cat adjops program");
    let compile_ms = t.elapsed().as_secs_f64() * 1e3;
    let t_cold = std::time::Instant::now();
    let _ = prog.grad(&theta, &terms).expect("cat grad");
    let cold_ms = t_cold.elapsed().as_secs_f64() * 1e3;
    let mut t_cat = Vec::with_capacity(reps);
    let mut g_cat = Vec::new();
    for _ in 0..reps {
        let t = std::time::Instant::now();
        g_cat = prog.grad(&theta, &terms).expect("cat grad").to_vec();
        t_cat.push(t.elapsed().as_secs_f64() * 1e3);
    }
    let expval = prog.expectation(&theta, &terms).expect("cat expval");

    // ── Lane sf_ir ──
    let values: std::collections::HashMap<String, f64> =
        (0..p).map(|i| (format!("t{i}"), theta[i])).collect();
    let t = std::time::Instant::now();
    let r = sf_ir::adjoint::adjoint_grad(&dag, &terms, &values).expect("sf_ir adj");
    let setup_ms = t.elapsed().as_secs_f64() * 1e3;
    let g_ir = r.gradients;
    let mut t_ir = Vec::with_capacity(reps);
    for _ in 0..reps {
        let t = std::time::Instant::now();
        let _ = sf_ir::adjoint::adjoint_grad(&dag, &terms, &values).expect("sf_ir adj");
        t_ir.push(t.elapsed().as_secs_f64() * 1e3);
    }

    // ── Cross-checks: house-engine parity + optional central FD ──
    let parity_max_diff = g_cat
        .iter()
        .zip(g_ir.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    let mut fd_max_err = f64::NAN;
    if fd_check {
        let eps = 1e-5;
        let mut fd = Vec::with_capacity(p);
        for i in 0..p {
            let mut tp = theta.clone();
            tp[i] += eps;
            let mut tm = theta.clone();
            tm[i] -= eps;
            let ep = prog.expectation(&tp, &terms).unwrap();
            let em = prog.expectation(&tm, &terms).unwrap();
            fd.push((ep - em) / (2.0 * eps));
        }
        fd_max_err = g_cat
            .iter()
            .zip(fd.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
    }
    let grad_norm = g_cat.iter().map(|v| v * v).sum::<f64>().sqrt();
    let grad_head: Vec<f64> = g_cat
        .iter()
        .take(4)
        .map(|v| (v * 1e6).round() / 1e6)
        .collect();

    serde_json::json!({
        "mode": "adjops",
        "n": n,
        "p": p,
        "reps": reps,
        "rayon_threads": rayon::current_num_threads(),
        "obs": obs,
        "cat": {
            "compile_ms": compile_ms,
            "cold_ms": cold_ms,
            "grad_ms": median_ms(t_cat),
        },
        "sf_ir": {
            "dag_build_ms": dag_build_ms,
            "setup_ms": setup_ms,
            "grad_ms": median_ms(t_ir),
        },
        "parity_max_diff": parity_max_diff,
        "fd_check": fd_check,
        "fd_eps": 1e-5,
        "fd_max_err": if fd_check {
            serde_json::json!(fd_max_err)
        } else {
            serde_json::Value::Null
        },
        "expval": expval,
        "grad_norm": grad_norm,
        "grad_head": grad_head,
        "grad": g_cat,
    })
}

/// Build a DAG from parsed ops for the density-matrix lane (Clifford set only:
/// the noiseless ladder family H/S/CX; anything else is a hard error).
fn build_dm_dag(n: usize, ops: &[(String, Vec<usize>)]) -> QuantumDAG {
    let mut dag = QuantumDAG::new(n, 0);
    for (name, qs) in ops {
        let ot = match name.as_str() {
            "H" => OpType::H,
            "X" => OpType::X,
            "Y" => OpType::Y,
            "Z" => OpType::Z,
            "S" => OpType::S,
            "SDG" => OpType::Sdg,
            "SX" => OpType::SX,
            "SXDG" => OpType::SXdg,
            "CX" | "CNOT" => OpType::CNOT,
            "CZ" => OpType::CZ,
            "CY" => OpType::CY,
            "SWAP" => OpType::SWAP,
            other => panic!("dm: unsupported op {other}"),
        };
        dag.add_op(ot, qs);
    }
    dag
}

/// Density-matrix lane: `QuantumDAG::simulate_dm` (the sf-ir streaming
/// pair-pass engine, engine order `i = ket | (bra << n)`) vs the statevector
/// engine on identical ops.  Gates read off the engine-order buffer:
/// trace(rho) ~ 1, purity Tr rho^2 ~ 1, and the rho diagonal equals |psi|^2
/// elementwise.  `cat`'s DM mode is a deliberate hold-out (R6/R8 measured no
/// same-window win vs the routed aer path), so this lane benches the sf-ir
/// engine through the same harness -- disclosed, never reframed as a cat win.
fn run_dm(job: &serde_json::Value) -> serde_json::Value {
    let n = job["n"].as_u64().expect("n") as usize;
    let reps = job["reps"].as_u64().unwrap_or(7) as usize;
    let ops = parse_ops(&job["ops"]);

    // cold: DAG construction + first simulate_dm in ONE timed region.
    let t_cold = std::time::Instant::now();
    let _ = build_dm_dag(n, &ops).simulate_dm();
    let cold_ms = t_cold.elapsed().as_secs_f64() * 1e3;

    // build attribution (no simulate).
    let t = std::time::Instant::now();
    let dag = build_dm_dag(n, &ops);
    let build_ms = t.elapsed().as_secs_f64() * 1e3;

    // warm passes (excluded) then interleaved medians (dm <-> sv).
    let mut data = dag.simulate_dm();
    let mut psi = dag.simulate();
    let mut t_dm = Vec::with_capacity(reps);
    let mut t_sv = Vec::with_capacity(reps);
    for r in 0..reps {
        if r % 2 == 0 {
            let t = std::time::Instant::now();
            data = dag.simulate_dm();
            t_dm.push(t.elapsed().as_secs_f64() * 1e3);
            let t = std::time::Instant::now();
            psi = dag.simulate();
            t_sv.push(t.elapsed().as_secs_f64() * 1e3);
        } else {
            let t = std::time::Instant::now();
            psi = dag.simulate();
            t_sv.push(t.elapsed().as_secs_f64() * 1e3);
            let t = std::time::Instant::now();
            data = dag.simulate_dm();
            t_dm.push(t.elapsed().as_secs_f64() * 1e3);
        }
    }

    // layout-invariant correctness gates, engine-order reads.
    let d = 1usize << n;
    let trace: f64 = (0..d).map(|k| data[k | (k << n)].re).sum();
    let purity: f64 = data.iter().map(|z| z.norm_sqr()).sum();
    let mut prob_match = true;
    let mut max_diag_diff = 0.0f64;
    for k in 0..d {
        let diff = (data[k | (k << n)].re - psi[k].norm_sqr()).abs();
        if diff > max_diag_diff {
            max_diag_diff = diff;
        }
        if diff > 1e-8 {
            prob_match = false;
        }
    }

    serde_json::json!({
        "mode": "dm",
        "n": n,
        "reps": reps,
        "rayon_threads": rayon::current_num_threads(),
        "ops_len": ops.len(),
        "cold_ms": cold_ms,
        "build_ms": build_ms,
        "dm_ms": median_ms(t_dm),
        "sv_ms": median_ms(t_sv),
        "trace": trace,
        "purity": purity,
        "prob_match": prob_match,
        "max_diag_diff": max_diag_diff,
        "engine": "sf-ir streaming pair-pass (QuantumDAG::simulate_dm)",
        "cat_dm_status": "held out (R6/R8: no same-window win vs aer) -- measured via sf-ir, disclosed",
    })
}
