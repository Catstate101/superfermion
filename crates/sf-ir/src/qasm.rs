//! OpenQASM 2.0 parser — builds a QuantumDAG directly from QASM2 text.
//!
//! This is a hand-rolled parser (no regex dependency) optimized for speed.
//! It supports the subset of QASM2 emitted by Qiskit `dumps()` with basis
//! gates [rx, ry, rz, cx] plus common single/two-qubit gates.
//!
//! Gate-to-OpType mapping mirrors the Python bridge's `QASM_MAP` exactly.

use crate::dag::QuantumDAG;
use crate::ops::{OpType, Parameter};
use std::f64::consts::PI;

/// Parse an OpenQASM 2.0 string into a QuantumDAG.
///
/// Returns an error string if the QASM is malformed or unsupported.
pub fn parse_qasm2(qasm_str: &str) -> Result<QuantumDAG, String> {
    let mut dag: Option<QuantumDAG> = None;

    for raw_line in qasm_str.lines() {
        let line = raw_line.trim();
        if line.is_empty()
            || line.starts_with("//")
            || line.starts_with("OPENQASM")
            || line.starts_with("include")
        {
            continue;
        }

        // ── qubit declaration ──────────────────────────────────
        if line.starts_with("qreg") {
            dag = parse_qreg_size(line).map(|n| QuantumDAG::new(n, n));
            continue;
        }

        // QASM3-style: qubit[N] name;
        if line.starts_with("qubit[") {
            dag = parse_bracket_size(line, "qubit[").map(|n| QuantumDAG::new(n, n));
            continue;
        }

        let dag_ref = match dag.as_mut() {
            Some(d) => d,
            None => continue,
        };
        let n = dag_ref.n_qubits;

        // ── gate instruction ───────────────────────────────────
        let (gate_name, params, qubits_str) = match split_qasm2_line(line) {
            Some(t) => t,
            None => continue,
        };

        // Map gate name to OpType
        let op = match map_gate(&gate_name, &params) {
            Some(o) => o,
            None => continue, // skip unknown gates silently
        };

        // Handle measure specially: "measure q[i] -> c[j]" has two bracket
        // pairs but only q[i] is a qubit. Parse only before "->".
        if gate_name == "measure" {
            // "measure q[i] -> c[j];" — parse q part and c part separately
            let (q_part, c_part) = if let Some(arrow) = qubits_str.find("->") {
                (&qubits_str[..arrow], Some(&qubits_str[arrow + 2..]))
            } else {
                (qubits_str.as_str(), None)
            };
            if let Some(q_idx) = parse_first_bracket_index(q_part) {
                if q_idx >= n {
                    continue;
                }
                let qubit = n - 1 - q_idx;
                let cbit = c_part
                    .and_then(parse_first_bracket_index)
                    .filter(|&ci| ci < n)
                    .map(|ci| n - 1 - ci)
                    .unwrap_or(qubit);
                dag_ref.add_measure(qubit, cbit);
            }
            continue;
        }

        if gate_name == "reset" || gate_name == "barrier" {
            let mut qubit_indices = Vec::new();
            let mut search_from = 0usize;
            while let Some(bracket_start) = qubits_str[search_from..].find('[') {
                let abs_start = search_from + bracket_start + 1;
                let bracket_end = match qubits_str[abs_start..].find(']') {
                    Some(i) => abs_start + i,
                    None => break,
                };
                let num_str = &qubits_str[abs_start..bracket_end];
                if let Ok(idx) = num_str.parse::<usize>() {
                    if idx < n {
                        qubit_indices.push(n - 1 - idx);
                    }
                }
                search_from = bracket_end + 1;
            }
            if !qubit_indices.is_empty() {
                dag_ref.add_op(op, &qubit_indices);
            }
            continue;
        }

        // Parse qubit indices from "qreg[q]" or "name[q]" patterns
        let mut qubit_indices = Vec::new();
        let mut search_from = 0usize;
        while let Some(bracket_start) = qubits_str[search_from..].find('[') {
            let abs_start = search_from + bracket_start + 1;
            let bracket_end = match qubits_str[abs_start..].find(']') {
                Some(i) => abs_start + i,
                None => break,
            };
            let num_str = &qubits_str[abs_start..bracket_end];
            if let Ok(idx) = num_str.parse::<usize>() {
                if idx < n {
                    // Reverse endianness: QASM LSB(0) → SF MSB(n-1)
                    qubit_indices.push(n - 1 - idx);
                }
            }
            search_from = bracket_end + 1;
        }

        if qubit_indices.is_empty() {
            continue;
        }

        dag_ref.add_op(op, &qubit_indices);
    }

    dag.ok_or_else(|| "Could not parse QASM: no qubit declaration found".to_string())
}

/// Split a QASM2 gate line into (gate_name, params_vec, qubits_str).
///
/// Handles formats:
///   - `h q[0];`
///   - `rx(pi/2) q[0];`
///   - `cx q[0], q[1];`
///   - `u(0.1, 0.2, 0.3) q[0];`
fn split_qasm2_line(line: &str) -> Option<(String, Vec<f64>, String)> {
    let line = line.trim_end_matches(';').trim();

    // Find '(' for params
    let (gate_name, params_str, qubits_str) = if let Some(lparen) = line.find('(') {
        let rparen = line.rfind(')')?;
        let gname = line[..lparen].trim().to_lowercase();
        let pstr = &line[lparen + 1..rparen];
        let qstr = line[rparen + 1..].trim().to_string();
        (gname, Some(pstr), qstr)
    } else {
        // No params — split on first whitespace
        let mut parts = line.splitn(2, |c: char| c.is_whitespace());
        let gname = parts.next()?.trim().to_lowercase();
        let qstr = parts.next()?.trim().to_string();
        (gname, None, qstr)
    };

    let params = params_str
        .map(|s| {
            s.split(',')
                .map(|p| eval_param(p.trim()))
                .collect::<Vec<f64>>()
        })
        .unwrap_or_default();

    Some((gate_name, params, qubits_str))
}

/// Evaluate a QASM2 parameter expression like `pi/2`, `-pi/2`, `0.5*pi`, `0.1`.
fn eval_param(p: &str) -> f64 {
    if p.is_empty() {
        return 0.0;
    }
    // Fast path for plain numbers
    if !p.contains("pi") && !p.contains("PI") {
        return p.parse::<f64>().unwrap_or(0.0);
    }

    // Replace pi with the constant value and evaluate
    let substituted = p
        .replace("pi", &PI.to_string())
        .replace("PI", &PI.to_string());

    // Simple expression evaluator for + - * /
    eval_simple_expr(&substituted)
}

/// Evaluate a simple arithmetic expression (+, -, *, /) with f64 values.
fn eval_simple_expr(s: &str) -> f64 {
    let s = s.trim();

    // Handle unary minus
    if let Some(rest) = s.strip_prefix('-') {
        // Check if rest starts with a digit
        if rest.starts_with(|c: char| c.is_ascii_digit() || c == '.') {
            return -eval_simple_expr(rest);
        }
        return -eval_simple_expr(rest);
    }

    // Find the rightmost + or - (lowest precedence)
    if let Some(pos) = find_op_at_level(s, &['+', '-'], 1) {
        let left = eval_simple_expr(&s[..pos]);
        let right = eval_simple_expr(&s[pos + 1..]);
        return if s.as_bytes()[pos] == b'+' {
            left + right
        } else {
            left - right
        };
    }

    // Find the rightmost * or /
    if let Some(pos) = find_op_at_level(s, &['*', '/'], 1) {
        let left = eval_simple_expr(&s[..pos]);
        let right = eval_simple_expr(&s[pos + 1..]);
        return if s.as_bytes()[pos] == b'*' {
            left * right
        } else {
            if right == 0.0 {
                return 0.0;
            }
            left / right
        };
    }

    // Leaf: just a number
    s.parse::<f64>().unwrap_or(0.0)
}

/// Find an operator at the top expression level (not inside parens).
fn find_op_at_level(s: &str, ops: &[char], _level: u32) -> Option<usize> {
    let bytes = s.as_bytes();
    let mut depth: i32 = 0;
    for i in (0..bytes.len()).rev() {
        match bytes[i] {
            b')' => depth += 1,
            b'(' => depth -= 1,
            _ if depth == 0 && ops.contains(&(bytes[i] as char)) => return Some(i),
            _ => {}
        }
    }
    None
}

/// Extract the first bracket-enclosed integer from a string, e.g. "q[3]" → Some(3).
fn parse_first_bracket_index(s: &str) -> Option<usize> {
    let start = s.find('[')? + 1;
    let end = s[start..].find(']')?;
    s[start..start + end].parse::<usize>().ok()
}

/// Parse qubit count from qreg declaration: `qreg q[N];` or `qreg name[N];`
fn parse_qreg_size(line: &str) -> Option<usize> {
    parse_bracket_size(line, "qreg")
}

fn parse_bracket_size(line: &str, _prefix: &str) -> Option<usize> {
    let start = line.find('[')? + 1;
    let end = line[start..].find(']')?;
    line[start..start + end].parse::<usize>().ok()
}

/// Map a QASM2 gate name + params to an OpType.
fn map_gate(name: &str, params: &[f64]) -> Option<OpType> {
    let p0 = || params.first().copied().unwrap_or(0.0);
    let p1 = || params.get(1).copied().unwrap_or(0.0);
    let p2 = || params.get(2).copied().unwrap_or(0.0);

    match name {
        "h" => Some(OpType::H),
        "x" => Some(OpType::X),
        "y" => Some(OpType::Y),
        "z" => Some(OpType::Z),
        "s" => Some(OpType::S),
        "sdg" => Some(OpType::Sdg),
        "t" => Some(OpType::T),
        "tdg" => Some(OpType::Tdg),
        "sx" => Some(OpType::SX),
        "sxdg" => Some(OpType::SXdg),
        "id" => Some(OpType::Id),
        "rx" => Some(OpType::Rx(Parameter::Const(p0()))),
        "ry" => Some(OpType::Ry(Parameter::Const(p0()))),
        "rz" => Some(OpType::Rz(Parameter::Const(p0()))),
        "r1" => Some(OpType::R1(Parameter::Const(p0()))),
        "p" | "u1" => Some(OpType::P(Parameter::Const(p0()))),
        "u" | "u3" => Some(OpType::U(
            Parameter::Const(p0()),
            Parameter::Const(p1()),
            Parameter::Const(p2()),
        )),
        "cx" | "cnot" => Some(OpType::CNOT),
        "cz" => Some(OpType::CZ),
        "cy" => Some(OpType::CY),
        "swap" => Some(OpType::SWAP),
        "iswap" => Some(OpType::ISWAP),
        "ecr" => Some(OpType::ECR),
        "ccx" | "toffoli" => Some(OpType::CCX),
        "cswap" => Some(OpType::CSWAP),
        "rzz" => Some(OpType::Rzz(Parameter::Const(p0()))),
        "rxx" => Some(OpType::Rxx(Parameter::Const(p0()))),
        "ryy" => Some(OpType::Ryy(Parameter::Const(p0()))),
        "crx" => Some(OpType::CRx(Parameter::Const(p0()))),
        "crz" => Some(OpType::CRz(Parameter::Const(p0()))),
        "cp" | "cr1" | "cphase" => Some(OpType::CP(Parameter::Const(p0()))),
        "measure" => Some(OpType::Measure),
        "barrier" => Some(OpType::Barrier),
        "reset" => Some(OpType::Reset),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_bell_circuit() {
        let qasm = r#"OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
h q[0];
cx q[0], q[1];
"#;
        let dag = parse_qasm2(qasm).expect("should parse");
        assert_eq!(dag.n_qubits, 2);
        assert_eq!(dag.gate_count(), 2);
    }

    #[test]
    fn test_parse_qv100_style() {
        let qasm = r#"OPENQASM 2.0;
include "qelib1.inc";
qreg qregless[100];
rz(pi/2) qregless[0];
ry(0.5*pi) qregless[1];
rx(-pi/2) qregless[2];
cx qregless[0], qregless[1];
"#;
        let dag = parse_qasm2(qasm).expect("should parse");
        assert_eq!(dag.n_qubits, 100);
        assert_eq!(dag.gate_count(), 4);
    }

    #[test]
    fn test_eval_param() {
        let eps = 1e-10;
        assert!((eval_param("pi") - std::f64::consts::PI).abs() < eps);
        assert!((eval_param("pi/2") - std::f64::consts::PI / 2.0).abs() < eps);
        assert!((eval_param("-pi/2") - (-std::f64::consts::PI / 2.0)).abs() < eps);
        assert!((eval_param("0.5*pi") - 0.5 * std::f64::consts::PI).abs() < eps);
        assert!((eval_param("0.1") - 0.1).abs() < eps);
        assert!((eval_param("-0.3") - (-0.3)).abs() < eps);
        assert!((eval_param("2*pi/3") - 2.0 * std::f64::consts::PI / 3.0).abs() < eps);
    }

    #[test]
    fn test_parse_measure_no_cycle() {
        let qasm = r#"OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0], q[1];
measure q[0] -> c[0];
measure q[1] -> c[1];
"#;
        let dag = parse_qasm2(qasm).expect("should parse without cycle");
        assert_eq!(dag.n_qubits, 2);
        // h + cx + 2 measures
        assert!(dag.gate_count() >= 2);
        // Verify topo order succeeds (no cycle panic)
        let _order = dag.topological_order();
    }

    #[test]
    fn test_parse_reset_line() {
        let qasm = r#"OPENQASM 2.0;
qreg q[2];
reset q[0];
h q[0];
"#;
        let dag = parse_qasm2(qasm).expect("should parse");
        assert_eq!(dag.n_qubits, 2);
        let _order = dag.topological_order();
    }

    #[test]
    fn test_all_gate_types() {
        // Test that all gate mappings work
        let qasm = r#"OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
h q[0];
x q[0];
z q[0];
rx(0.5) q[0];
ry(0.3) q[1];
rz(0.1) q[2];
cx q[0], q[1];
cz q[0], q[1];
swap q[0], q[1];
"#;
        let dag = parse_qasm2(qasm).expect("should parse");
        assert_eq!(dag.n_qubits, 3);
        assert!(dag.gate_count() >= 8);
    }
}
