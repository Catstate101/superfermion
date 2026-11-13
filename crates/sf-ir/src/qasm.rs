//! OpenQASM 2.0 parser — builds a QuantumDAG directly from QASM2 text.
//!
//! This is a hand-rolled parser (no regex dependency) optimized for speed.
//! It supports the subset of QASM2 emitted by Qiskit `dumps()` with basis
//! gates [rx, ry, rz, cx] plus common single/two-qubit gates.
//!
//! Gate-to-OpType mapping mirrors the Python bridge's `QASM_MAP` exactly.
//!
//! Endianness: SF, Qiskit and OpenQASM all use the little-endian (q0 = LSB)
//! qubit order, so a QASM label `q[i]` maps straight to SF qubit `i` with no
//! reversal. This keeps `to_qasm` -> `from_qasm2` an involution and matches
//! the pure-Python fallback in `superfermion.bridge`.

use crate::dag::QuantumDAG;
use crate::ops::{OpType, Parameter};
use std::collections::HashMap;
use std::f64::consts::PI;

/// Parse an OpenQASM 2.0 string into a QuantumDAG.
///
/// Returns an error string if the QASM is malformed or unsupported.
pub fn parse_qasm2(qasm_str: &str) -> Result<QuantumDAG, String> {
    let mut dag: Option<QuantumDAG> = None;

    // Register table: name -> (offset, size).  OpenQASM 2 allows several qreg
    // declarations, and a whole register is a valid operand (`h q;`), which
    // addresses every qubit of that register.  The DAG used to be rebuilt on
    // every qreg line, so `qreg a[1]; qreg b[1]; x b[0];` produced a 1-qubit
    // circuit and `cx a[0], b[0];` collided on qubit 0.
    let mut qregs: HashMap<String, (usize, usize)> = HashMap::new();
    let mut cregs: HashMap<String, (usize, usize)> = HashMap::new();
    let mut n_qubits = 0usize;
    let mut n_cbits = 0usize;

    // ── `gate` declarations ────────────────────────────────────
    // qelib1-style macros are expanded before the line loop, so a body whose
    // name has no OpType (`ch`, `cry`, `cu1`, `c3x`, ...) still imports
    // exactly instead of being rejected as an unsupported gate.
    // `QELIB1_EXTRA` seeds the standard library's own bodies for the gates
    // `map_gate` does not implement natively; a declaration in the input
    // overrides the seed of the same name.
    let (_, mut declarations) = strip_gate_decls(QELIB1_EXTRA);
    let (stripped, user_decls) = strip_gate_decls(qasm_str);
    for (name, decl) in user_decls {
        declarations.insert(name, decl);
    }
    let expanded;
    let src: &str = if declarations.is_empty() {
        qasm_str
    } else {
        expanded = inline_gate_calls(&stripped, &declarations)?;
        &expanded
    };

    for (lineno, raw_line) in split_qasm2_statements(src).into_iter() {
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
            if dag.is_some() {
                return Err(format!(
                    "line {}: qreg declaration after the first instruction: {line}",
                    lineno + 1
                ));
            }
            match parse_named_register(line) {
                Some((name, size)) => {
                    qregs.insert(name, (n_qubits, size));
                    n_qubits += size;
                }
                None => {
                    return Err(format!(
                        "line {}: malformed qreg declaration: {line}",
                        lineno + 1
                    ))
                }
            }
            continue;
        }

        // QASM3-style: qubit[N] name;
        if line.starts_with("qubit[") {
            if dag.is_some() {
                return Err(format!(
                    "line {}: qubit declaration after the first instruction: {line}",
                    lineno + 1
                ));
            }
            match parse_sized_register(line) {
                Some((name, size)) => {
                    qregs.insert(name, (n_qubits, size));
                    n_qubits += size;
                }
                None => {
                    return Err(format!(
                        "line {}: malformed qubit declaration: {line}",
                        lineno + 1
                    ))
                }
            }
            continue;
        }

        // Classical register declarations carry no quantum operation. The DAG
        // is created with one classical bit per qubit, so the size is unused
        // here but the line is legal QASM and must not be treated as a gate.
        if line.starts_with("creg") {
            if dag.is_some() {
                return Err(format!(
                    "line {}: creg declaration after the first instruction: {line}",
                    lineno + 1
                ));
            }
            match parse_named_register(line) {
                Some((name, size)) => {
                    cregs.insert(name, (n_cbits, size));
                    n_cbits += size;
                }
                None => {
                    return Err(format!(
                        "line {}: malformed creg declaration: {line}",
                        lineno + 1
                    ))
                }
            }
            continue;
        }

        // A bare `barrier;` addresses every qubit and carries no semantics
        // (`to_qasm` drops BARRIER for the same reason).
        if line.trim_end_matches(';').trim() == "barrier" {
            continue;
        }

        if n_qubits == 0 {
            return Err(format!(
                "line {}: instruction before any qreg declaration: {line}",
                lineno + 1
            ));
        }
        // Without an explicit creg the DAG keeps the historical one classical
        // bit per qubit, so `measure q[i]` still targets c[i].
        let cbit_width = if n_cbits == 0 { n_qubits } else { n_cbits };
        let dag_ref = dag.get_or_insert_with(|| QuantumDAG::new(n_qubits, cbit_width));
        let n = dag_ref.n_qubits;

        // ── classical feed-forward ─────────────────────────────
        // `if (c == value) gate ...` guards the next instruction.  Without
        // this the `if` head reached `split_qasm2_line_checked`, which tried
        // to evaluate `c==1` as an angle expression and failed.
        let (statement, condition) = match parse_if_condition(line, &cregs, lineno)? {
            Some((cond, rest)) => (rest, Some(cond)),
            None => (line, None),
        };

        // ── gate instruction ───────────────────────────────────
        // The checked variant reports an angle expression it cannot evaluate
        // instead of folding it to 0.0 (see `eval_expr`), which used to
        // produce a silent zero rotation from a "successful" parse.
        let (gate_name, params, qubits_str) = match split_qasm2_line_checked(statement) {
            Ok(t) => t,
            Err(msg) => return Err(format!("line {}: {msg} in '{line}'", lineno + 1)),
        };

        // Map gate name to OpType. An unmapped name is an error, not a skip:
        // dropping it returns a circuit that is missing operations while
        // looking like a successful parse, and the caller has no way to tell.
        let op = match map_gate(&gate_name, &params) {
            Some(o) => o,
            None => {
                return Err(format!(
                    "line {}: unsupported gate '{gate_name}'",
                    lineno + 1
                ))
            }
        };

        // Handle measure specially: "measure q[i] -> c[j]" has two bracket
        // pairs but only q[i] is a qubit. Parse only before "->".
        // A guard is legal on a measurement in OpenQASM 2
        // (`if(c==1) measure q[0] -> c[0];`); it selects whether the qubit
        // is measured and the classical bit updated, exactly like QASM3's
        // `if (c[0] == true) c[0] = measure q[0];` and Qiskit's if_test.
        if gate_name == "measure" {
            let (q_part, c_part) = if let Some(arrow) = qubits_str.find("->") {
                (&qubits_str[..arrow], Some(&qubits_str[arrow + 2..]))
            } else {
                (qubits_str.as_str(), None)
            };
            let q_ops = resolve_operands(q_part, &qregs, n, line, lineno + 1)?;
            let c_ops = match c_part {
                Some(rest) if !rest.trim().is_empty() => Some(resolve_operands(
                    rest,
                    &cregs,
                    dag_ref.n_cbits,
                    line,
                    lineno + 1,
                )?),
                _ => None,
            };
            let q_indices = expand_operands(&q_ops, "measure", line, lineno + 1)?;
            let c_indices = match &c_ops {
                Some(ops) => Some(expand_operands(ops, "measure", line, lineno + 1)?),
                None => None,
            };
            match c_indices {
                None => {
                    for row in &q_indices {
                        match condition {
                            Some((cbit, value)) => {
                                dag_ref.add_measure_conditioned(row[0], row[0], cbit, value);
                            }
                            None => {
                                dag_ref.add_measure(row[0], row[0]);
                            }
                        }
                    }
                }
                Some(c_rows) if c_rows.len() == q_indices.len() => {
                    for (q_row, c_row) in q_indices.iter().zip(c_rows.iter()) {
                        match condition {
                            Some((cbit, value)) => {
                                dag_ref.add_measure_conditioned(q_row[0], c_row[0], cbit, value);
                            }
                            None => {
                                dag_ref.add_measure(q_row[0], c_row[0]);
                            }
                        }
                    }
                }
                Some(_) => {
                    return Err(format!(
                        "line {}: measure qubit and classical operands have different sizes in '{line}'",
                        lineno + 1
                    ))
                }
            }
            continue;
        }

        if gate_name == "reset" || gate_name == "barrier" {
            let operands = resolve_operands(&qubits_str, &qregs, n, line, lineno + 1)?;
            if gate_name == "barrier" {
                if condition.is_some() {
                    return Err(format!(
                        "line {}: a barrier cannot be conditioned: {line}",
                        lineno + 1
                    ));
                }
                // A barrier spans every operand and has no gate semantics, so
                // it is not broadcast; unequal register sizes are legal here.
                let mut indices = Vec::new();
                for operand in &operands {
                    match operand {
                        Operand::Index(i) => indices.push(*i),
                        Operand::Register(offset, size) => indices.extend(*offset..*offset + *size),
                    }
                }
                dag_ref.add_op(op.clone(), &indices);
            } else {
                for inst in expand_operands(&operands, &gate_name, line, lineno + 1)? {
                    match condition {
                        Some((cbit, value)) => {
                            dag_ref.add_op_conditioned(op.clone(), &inst, cbit, value);
                        }
                        None => {
                            dag_ref.add_op(op.clone(), &inst);
                        }
                    }
                }
            }
            continue;
        }

        let operands = resolve_operands(&qubits_str, &qregs, n, line, lineno + 1)?;
        let instances = expand_operands(&operands, &gate_name, line, lineno + 1)?;

        // A single gate application may not name the same qubit twice: the DAG
        // would gain a self-loop and `add_op` asserts on it.  Report that as a
        // parse error carrying the line number instead of letting the assert
        // surface as a PanicException.  One-line (minified) programs used to be
        // read as one giant gate whose operand list was every `q[i]` on the
        // line; re-importing such an export crashed the interpreter's caller.
        for inst in &instances {
            for i in 0..inst.len() {
                for j in (i + 1)..inst.len() {
                    if inst[i] == inst[j] {
                        return Err(format!(
                            "line {}: gate '{}' names qubit {} more than once in '{}' ({} operands given; one gate application must name distinct qubits)",
                            lineno + 1,
                            gate_name,
                            inst[i],
                            line,
                            inst.len()
                        ));
                    }
                }
            }
        }

        // A gate application must name exactly as many qubits as the gate
        // acts on.  Without this, `u3(t,p,l) q[4], q[2], q[3];` (one-qubit
        // gate, three operands) parsed "successfully" into a three-wire record
        // holding a 2x2 matrix, and `QuantumDAG::to_unitary` then indexed out
        // of bounds -- a Rust panic that reached Python as PanicException.
        // `OpType::n_qubits` is the authority, so this cannot drift from the
        // gate table.
        let expected = op.n_qubits();
        if expected != 0 && instances[0].len() != expected {
            return Err(format!(
                "line {}: gate '{}' takes {} qubit operand(s) but {} were given in '{}'",
                lineno + 1,
                gate_name,
                expected,
                instances[0].len(),
                line
            ));
        }

        for inst in &instances {
            match condition {
                Some((cbit, value)) => {
                    dag_ref.add_op_conditioned(op.clone(), inst, cbit, value);
                }
                None => {
                    dag_ref.add_op(op.clone(), inst);
                }
            }
        }
    }

    // A program may contain declarations only (`qreg q[2];` with no
    // instructions): the DAG is created lazily at the first instruction,
    // so materialise the empty one here instead of reporting a parse error.
    if dag.is_none() {
        if n_qubits == 0 {
            return Err("Could not parse QASM: no qubit declaration found".to_string());
        }
        let cbit_width = if n_cbits == 0 { n_qubits } else { n_cbits };
        dag = Some(QuantumDAG::new(n_qubits, cbit_width));
    }
    Ok(dag.expect("the empty DAG is materialised above"))
}

/// A parsed ``if`` guard paired with the statement it applies to.
type GuardedStatement<'a> = Option<((usize, u64), &'a str)>;

/// Parse a leading OpenQASM 2 ``if (creg == value)`` / ``if (creg[i] == value)``.
///
/// Returns the resolved ``(cbit, value)`` condition together with the guarded
/// statement, or ``None`` when the line is not conditional.  The DAG stores a
/// single classical-bit test per gate, so a multi-bit whole-register
/// comparison (legal QASM 2) cannot be represented and is an error rather
/// than a silent drop.
fn parse_if_condition<'a>(
    line: &'a str,
    cregs: &HashMap<String, (usize, usize)>,
    lineno: usize,
) -> Result<GuardedStatement<'a>, String> {
    let rest = match line.strip_prefix("if") {
        Some(rest) => rest.trim_start(),
        None => return Ok(None),
    };
    let rest = match rest.strip_prefix('(') {
        Some(rest) => rest,
        None => {
            return Err(format!(
                "line {}: malformed if statement (missing '('): {line}",
                lineno + 1
            ))
        }
    };
    let eq = match rest.find("==") {
        Some(pos) => pos,
        None => {
            return Err(format!(
                "line {}: malformed if statement (missing '=='): {line}",
                lineno + 1
            ))
        }
    };
    let lhs = rest[..eq].trim();
    let after = &rest[eq + 2..];
    let close = match after.find(')') {
        Some(pos) => pos,
        None => {
            return Err(format!(
                "line {}: malformed if statement (missing ')'): {line}",
                lineno + 1
            ))
        }
    };
    let value: u64 = match after[..close].trim().parse() {
        Ok(v) => v,
        Err(_) => {
            return Err(format!(
                "line {}: if statement value is not an integer: {line}",
                lineno + 1
            ))
        }
    };
    let statement = after[close + 1..].trim();
    if statement.is_empty() {
        return Err(format!(
            "line {}: if statement guards no instruction: {line}",
            lineno + 1
        ));
    }
    let (name, index) = match lhs.find('[') {
        Some(open) => {
            let inner = lhs[open + 1..].trim_end_matches(']').trim();
            let idx: usize = match inner.parse() {
                Ok(i) => i,
                Err(_) => {
                    return Err(format!(
                        "line {}: if statement index is not an integer: {line}",
                        lineno + 1
                    ))
                }
            };
            (lhs[..open].trim(), Some(idx))
        }
        None => (lhs, None),
    };
    let (offset, size) = match cregs.get(name) {
        Some(span) => *span,
        None => {
            return Err(format!(
                "line {}: if statement names unknown classical register '{name}'",
                lineno + 1
            ))
        }
    };
    let index = match index {
        Some(i) if i < size => i,
        Some(i) => {
            return Err(format!(
                "line {}: if statement names classical bit {i} outside register '{name}' (size {size})",
                lineno + 1
            ))
        }
        None if size == 1 => 0,
        None => {
            return Err(format!(
                "line {}: if statement compares the whole {size}-bit register '{name}', but the IR stores one cbit condition per gate",
                lineno + 1
            ))
        }
    };
    if value > 1 {
        return Err(format!(
            "line {}: if statement compares {name}[{index}] with {value}, but the IR stores one cbit condition (0 or 1)",
            lineno + 1
        ));
    }
    Ok(Some(((offset + index, value), statement)))
}

/// Split a QASM2 source into `(0-based line number, statement)` pairs.
///
/// QASM2 does not require one statement per line: `qreg q[2]; h q[0]; h q[1];`
/// is legal and minified programs put a whole circuit on one line.  The line
/// loop used to hand the entire line to `split_qasm2_line_checked`, which reads
/// a single head, so every statement after the first on a line was dropped --
/// or, when the head carried an angle, folded in every `q[i]` on the line as an
/// operand.  Splitting here keeps the line the statement started on so error
/// messages still point at the right place.
fn split_qasm2_statements(src: &str) -> Vec<(usize, String)> {
    let mut out = Vec::new();
    let mut current = String::new();
    let mut depth = 0i32; // parentheses of a parameter list
    let mut in_string = false;
    for (idx, raw) in src.lines().enumerate() {
        for ch in strip_line_comment(raw).chars() {
            match ch {
                '"' => {
                    in_string = !in_string;
                    current.push(ch);
                }
                '(' if !in_string => {
                    depth += 1;
                    current.push(ch);
                }
                ')' if !in_string => {
                    depth -= 1;
                    current.push(ch);
                }
                ';' if !in_string && depth <= 0 => {
                    current.push(ch);
                    let stmt = current.trim().to_string();
                    if !stmt.is_empty() {
                        out.push((idx, stmt));
                    }
                    current.clear();
                }
                _ => current.push(ch),
            }
        }
        // A statement may continue on the next line, so only flush here when
        // the buffer already looks complete: `inline_gate_calls` emits one
        // expansion per line *without* a ';', and gluing those would merge
        // separate gates into one gate with a merged operand list.
        let done = !in_string
            && depth <= 0
            && (current.trim_end().ends_with(';') || current.contains(']'));
        if done {
            let stmt = current.trim().to_string();
            if !stmt.is_empty() {
                out.push((idx, stmt));
            }
            current.clear();
        } else {
            current.push(' ');
        }
    }
    let tail = current.trim().to_string();
    if !tail.is_empty() {
        out.push((src.lines().count().saturating_sub(1), tail));
    }
    out
}

/// Drop a `//` comment, leaving any `//` inside a quoted string alone.
fn strip_line_comment(raw: &str) -> &str {
    if let Some(pos) = raw.find("//") {
        if raw[..pos].matches('"').count().is_multiple_of(2) {
            return &raw[..pos];
        }
    }
    raw
}

/// One operand of a QASM 2 gate application.
enum Operand {
    /// A single qubit, already translated to the circuit-wide index.
    Index(usize),
    /// A whole register: (offset, size).
    Register(usize, usize),
}

/// Parse the operand list of a statement (`q`, `q[0]`, `a[1], b[0]`).
///
/// Register names are resolved against the declaration table, so `b[0]`
/// after `qreg a[1]; qreg b[1];` is qubit 1 and not qubit 0.
fn resolve_operands(
    operands: &str,
    regs: &HashMap<String, (usize, usize)>,
    width: usize,
    line: &str,
    lineno: usize,
) -> Result<Vec<Operand>, String> {
    let mut result = Vec::new();
    for raw in operands.split(',') {
        let token = raw.trim();
        if token.is_empty() {
            continue;
        }
        let (name, index) = match token.find('[') {
            Some(start) => {
                let end = match token[start..].find(']') {
                    Some(i) => start + i,
                    None => {
                        return Err(format!(
                            "line {lineno}: unterminated qubit index in '{line}'"
                        ))
                    }
                };
                let name = token[..start].trim();
                let idx_str = token[start + 1..end].trim();
                let idx = match idx_str.parse::<usize>() {
                    Ok(i) => Some(i),
                    Err(_) => {
                        return Err(format!(
                            "line {lineno}: qubit index '{idx_str}' is not an integer in '{line}'"
                        ))
                    }
                };
                (name, idx)
            }
            None => (token, None),
        };
        let (offset, size) = match regs.get(name) {
            Some(span) => *span,
            None => {
                return Err(format!(
                    "line {lineno}: unknown qubit register '{name}' in '{line}'"
                ))
            }
        };
        match index {
            Some(idx) => {
                if idx >= size {
                    return Err(format!(
                        "line {lineno}: qubit index {idx} out of range for register '{name}' (size {size}) in '{line}'"
                    ));
                }
                let absolute = offset + idx;
                if absolute >= width {
                    return Err(format!(
                        "line {lineno}: qubit index {absolute} out of range for {width} qubits in '{line}'"
                    ));
                }
                result.push(Operand::Index(absolute));
            }
            None => result.push(Operand::Register(offset, size)),
        }
    }
    if result.is_empty() {
        return Err(format!(
            "line {lineno}: statement names no qubit operand in '{line}'"
        ));
    }
    Ok(result)
}

/// Expand whole-register operands into one index list per gate application.
///
/// OpenQASM 2 broadcasts a gate over a register: `h q;` is one `h` per
/// qubit, and `cx a, b;` pairs the i-th qubit of each register.  Mixing a
/// whole register with an indexed operand is not a valid application.
fn expand_operands(
    operands: &[Operand],
    gate_name: &str,
    line: &str,
    lineno: usize,
) -> Result<Vec<Vec<usize>>, String> {
    if operands.iter().all(|op| matches!(op, Operand::Index(_))) {
        return Ok(vec![operands
            .iter()
            .map(|op| match op {
                Operand::Index(i) => *i,
                Operand::Register(..) => unreachable!(),
            })
            .collect()]);
    }
    if operands.iter().any(|op| matches!(op, Operand::Index(_))) {
        return Err(format!(
            "line {lineno}: gate '{gate_name}' mixes whole-register and indexed operands in '{line}'"
        ));
    }
    let sizes: Vec<usize> = operands
        .iter()
        .map(|op| match op {
            Operand::Register(_, size) => *size,
            Operand::Index(_) => unreachable!(),
        })
        .collect();
    if sizes.windows(2).any(|w| w[0] != w[1]) {
        return Err(format!(
            "line {lineno}: gate '{gate_name}' operands have different register sizes in '{line}'"
        ));
    }
    let size = sizes[0];
    Ok((0..size)
        .map(|k| {
            operands
                .iter()
                .map(|op| match op {
                    Operand::Register(offset, _) => offset + k,
                    Operand::Index(i) => *i,
                })
                .collect()
        })
        .collect())
}

/// Parse `qreg name[N];` (or `creg name[N];`) into (name, size).
fn parse_named_register(line: &str) -> Option<(String, usize)> {
    let size = parse_qreg_size(line)?;
    let open = line.find('[')?;
    let name = line[..open].split_whitespace().nth(1)?.to_string();
    if name.is_empty() {
        return None;
    }
    Some((name, size))
}

/// Parse the QASM 3 spelling `qubit[N] name;` into (name, size).
fn parse_sized_register(line: &str) -> Option<(String, usize)> {
    let size = parse_bracket_size(line, "qubit[")?;
    let open = line.find('[')?;
    let close = line[open..].find(']')? + open;
    let name = line[close + 1..].trim().trim_end_matches(';').trim();
    if name.is_empty() {
        return None;
    }
    Some((name.to_string(), size))
}

/// Split a QASM2 gate line into (gate_name, params_vec, qubits_str).
///
/// Handles formats:
///   - `h q[0];`
///   - `rx(pi/2) q[0];`
///   - `cx q[0], q[1];`
///   - `u(0.1, 0.2, 0.3) q[0];`
///
/// NOTE: `parse_qasm2` uses [`split_qasm2_line_checked`] instead, which
/// reports an unevaluable angle expression rather than folding it to 0.0.
/// This lenient variant is kept (unused by the parser) so existing callers
/// outside this module keep the previous behaviour.
#[allow(dead_code)]
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
#[allow(dead_code)]
fn eval_param(p: &str) -> f64 {
    if p.is_empty() {
        return 0.0;
    }
    // Full OpenQASM 2.0 grammar first (see `eval_expr`).  The legacy
    // `pi`-substitution evaluator below stays as the fallback, so every
    // expression it used to accept still evaluates to the same value.
    if let Some(v) = eval_expr(p) {
        return v;
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
#[allow(dead_code)]
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
#[allow(dead_code)]
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

/// Split a QASM2 gate line into (gate_name, params_vec, qubits_str), reporting
/// an angle expression that cannot be evaluated instead of folding it to 0.0.
///
/// The lenient [`split_qasm2_line`] is kept for callers that predate this
/// check; `parse_qasm2` uses this one so a bad parameter is a parse error.
fn split_qasm2_line_checked(line: &str) -> Result<(String, Vec<f64>, String), String> {
    let line = line.trim_end_matches(';').trim();

    let (gate_name, params_str, qubits_str) = if let Some(lparen) = line.find('(') {
        let rparen = match line.rfind(')') {
            Some(r) if r > lparen => r,
            _ => return Err("unterminated parameter list".to_string()),
        };
        let gname = line[..lparen].trim().to_lowercase();
        let pstr = &line[lparen + 1..rparen];
        let qstr = line[rparen + 1..].trim().to_string();
        (gname, Some(pstr), qstr)
    } else {
        let mut parts = line.splitn(2, |c: char| c.is_whitespace());
        let gname = match parts.next() {
            Some(g) => g.trim().to_lowercase(),
            None => return Err("empty statement".to_string()),
        };
        let qstr = match parts.next() {
            Some(q) => q.trim().to_string(),
            None => return Err(format!("gate '{gname}' names no operand")),
        };
        (gname, None, qstr)
    };

    let mut params = Vec::new();
    if let Some(s) = params_str {
        for raw in s.split(',') {
            let raw = raw.trim();
            if raw.is_empty() {
                continue;
            }
            match eval_expr(raw) {
                Some(v) => params.push(v),
                None => return Err(format!("cannot evaluate parameter '{raw}'")),
            }
        }
    }
    Ok((gate_name, params, qubits_str))
}

// ═══════════════════════════════════════════════════════════════
// Angle expressions (OpenQASM 2.0 numeric grammar)
// ═══════════════════════════════════════════════════════════════

/// Recursive-descent parser for an OpenQASM 2.0 angle expression.
///
/// ```text
/// expr  := term  (('+' | '-') term)*
/// term  := unary (('*' | '/') unary)*
/// unary := ('+' | '-')* power
/// power := atom ('^' unary)?
/// atom  := number | 'pi' | ident '(' expr ')' | '(' expr ')'
/// ```
///
/// `eval_param` used to substitute the digits of `pi` and run a splitter that
/// only understood `+ - * /` between bare numbers.  `pi*-0.3` happened to work
/// because the substituted digits formed a valid f64, while `(0.3)*pi`,
/// `sin(pi/2)*0.3` and `0.15+0.15` all fell through to
/// `parse::<f64>().unwrap_or(0.0)` and became 0.0 -- a silent zero rotation.
struct ExprParser<'a> {
    src: &'a str,
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> ExprParser<'a> {
    fn new(src: &'a str) -> Self {
        ExprParser {
            src,
            bytes: src.as_bytes(),
            pos: 0,
        }
    }

    fn skip_ws(&mut self) {
        while self.pos < self.bytes.len() && self.bytes[self.pos].is_ascii_whitespace() {
            self.pos += 1;
        }
    }

    fn peek(&mut self) -> Option<u8> {
        self.skip_ws();
        self.bytes.get(self.pos).copied()
    }

    fn bump(&mut self) -> Option<u8> {
        let c = self.peek()?;
        self.pos += 1;
        Some(c)
    }

    fn expr(&mut self) -> Option<f64> {
        let mut v = self.term()?;
        loop {
            match self.peek() {
                Some(b'+') => {
                    self.pos += 1;
                    v += self.term()?;
                }
                Some(b'-') => {
                    self.pos += 1;
                    v -= self.term()?;
                }
                _ => return Some(v),
            }
        }
    }

    fn term(&mut self) -> Option<f64> {
        let mut v = self.unary()?;
        loop {
            match self.peek() {
                Some(b'*') => {
                    self.pos += 1;
                    v *= self.unary()?;
                }
                Some(b'/') => {
                    self.pos += 1;
                    let d = self.unary()?;
                    if d == 0.0 {
                        return None;
                    }
                    v /= d;
                }
                _ => return Some(v),
            }
        }
    }

    fn unary(&mut self) -> Option<f64> {
        match self.peek() {
            Some(b'-') => {
                self.pos += 1;
                Some(-self.unary()?)
            }
            Some(b'+') => {
                self.pos += 1;
                self.unary()
            }
            _ => self.power(),
        }
    }

    fn power(&mut self) -> Option<f64> {
        let base = self.atom()?;
        if self.peek() == Some(b'^') {
            self.pos += 1;
            return Some(base.powf(self.unary()?));
        }
        Some(base)
    }

    fn atom(&mut self) -> Option<f64> {
        match self.peek()? {
            b'(' => {
                self.pos += 1;
                let v = self.expr()?;
                if self.bump() != Some(b')') {
                    return None;
                }
                Some(v)
            }
            c if c.is_ascii_digit() || c == b'.' => self.number(),
            c if c.is_ascii_alphabetic() || c == b'_' => self.ident(),
            _ => None,
        }
    }

    fn number(&mut self) -> Option<f64> {
        let start = self.pos;
        while self.pos < self.bytes.len() {
            let c = self.bytes[self.pos];
            let exp_sign = (c == b'+' || c == b'-')
                && self.pos > start
                && (self.bytes[self.pos - 1] == b'e' || self.bytes[self.pos - 1] == b'E');
            if c.is_ascii_digit() || c == b'.' || c == b'e' || c == b'E' || exp_sign {
                self.pos += 1;
            } else {
                break;
            }
        }
        self.src[start..self.pos].parse::<f64>().ok()
    }

    fn ident(&mut self) -> Option<f64> {
        let start = self.pos;
        while self.pos < self.bytes.len() {
            let c = self.bytes[self.pos];
            if c.is_ascii_alphanumeric() || c == b'_' {
                self.pos += 1;
            } else {
                break;
            }
        }
        let name = self.src[start..self.pos].to_ascii_lowercase();
        if name == "pi" {
            return Some(PI);
        }
        if self.peek() != Some(b'(') {
            return None;
        }
        self.pos += 1;
        let arg = self.expr()?;
        if self.bump() != Some(b')') {
            return None;
        }
        match name.as_str() {
            "sin" => Some(arg.sin()),
            "cos" => Some(arg.cos()),
            "tan" => Some(arg.tan()),
            "exp" => Some(arg.exp()),
            "ln" => Some(arg.ln()),
            "sqrt" => Some(arg.sqrt()),
            _ => None,
        }
    }
}

/// Evaluate an OpenQASM 2.0 angle expression, or `None` if it is malformed.
fn eval_expr(src: &str) -> Option<f64> {
    let mut parser = ExprParser::new(src);
    let value = parser.expr()?;
    if parser.peek().is_some() {
        // Trailing text (e.g. `0.1.2`) is not an expression: refuse it.
        return None;
    }
    if value.is_finite() {
        Some(value)
    } else {
        None
    }
}

// ═══════════════════════════════════════════════════════════════
// `gate` declarations (qelib1-style macros)
// ═══════════════════════════════════════════════════════════════

/// One parsed `gate name(params) qargs { body }` declaration.
struct GateDecl {
    params: Vec<String>,
    qargs: Vec<String>,
    body: Vec<String>,
}

/// qelib1 gates that `map_gate` has no `OpType` for, as declarations.
///
/// The parser skips `include "qelib1.inc";` (the file only carries
/// declarations), so these are the standard library's own bodies, copied
/// verbatim.  Expanding them keeps `from_qasm` total over qelib1 instead of
/// rejecting `ch`, `cry`, `csx`, `c3x` and friends as unsupported gates.
const QELIB1_EXTRA: &str = r"
gate u0(gamma) q { U(0,0,0) q; }
gate ch a,b { h b; sdg b; cx a,b; h b; t b; cx a,b; t b; h b; s b; x b; s a; }
gate csx a,b { h b; cu1(pi/2) a,b; h b; }
gate cry(lambda) a,b { ry(lambda/2) b; cx a,b; ry(-lambda/2) b; cx a,b; }
gate rccx a,b,c { u2(0,pi) c; u1(pi/4) c; cx b, c; u1(-pi/4) c; cx a, c; u1(pi/4) c; cx b, c; u1(-pi/4) c; u2(0,pi) c; }
gate rc3x a,b,c,d { u2(0,pi) d; u1(pi/4) d; cx c,d; u1(-pi/4) d; u2(0,pi) d; cx a,d; u1(pi/4) d; cx b,d; u1(-pi/4) d; cx a,d; u1(pi/4) d; cx b,d; u1(-pi/4) d; u2(0,pi) d; u1(pi/4) d; cx c,d; u1(-pi/4) d; u2(0,pi) d; }
gate c3x a,b,c,d { h d; p(pi/8) a; p(pi/8) b; p(pi/8) c; p(pi/8) d; cx a, b; p(-pi/8) b; cx a, b; cx b, c; p(-pi/8) c; cx a, c; p(pi/8) c; cx b, c; p(-pi/8) c; cx a, c; cx c, d; p(-pi/8) d; cx b, d; p(pi/8) d; cx c, d; p(-pi/8) d; cx a, d; p(pi/8) d; cx c, d; p(-pi/8) d; cx b, d; p(pi/8) d; cx c, d; p(-pi/8) d; cx a, d; h d; }
gate c3sqrtx a,b,c,d { h d; cu1(pi/8) a,d; h d; cx a,b; h d; cu1(-pi/8) b,d; h d; cx a,b; h d; cu1(pi/8) b,d; h d; cx b,c; h d; cu1(-pi/8) c,d; h d; cx a,c; h d; cu1(pi/8) c,d; h d; cx b,c; h d; cu1(-pi/8) c,d; h d; cx a,c; h d; cu1(pi/8) c,d; h d; }
gate c4x a,b,c,d,e { h e; cu1(pi/2) d,e; h e; c3x a,b,c,d; h e; cu1(-pi/2) d,e; h e; c3x a,b,c,d; c3sqrtx a,b,c,e; }
";

/// Whether `src` starts with the keyword `kw` at byte offset `at`.
fn keyword_at(src: &str, at: usize, kw: &str) -> bool {
    let rest = match src.get(at..) {
        Some(r) => r,
        None => return false,
    };
    if !rest.starts_with(kw) {
        return false;
    }
    match rest[kw.len()..].chars().next() {
        Some(c) => c.is_whitespace(),
        None => false,
    }
}

/// Remove every `gate ... { ... }` declaration from `src`.
///
/// Returns the remaining text (each declaration replaced by the same number of
/// newlines, so line numbers in error messages still match the input) and the
/// declarations themselves, keyed by gate name.
fn strip_gate_decls(src: &str) -> (String, HashMap<String, GateDecl>) {
    let bytes = src.as_bytes();
    let mut out = String::with_capacity(src.len());
    let mut decls: HashMap<String, GateDecl> = HashMap::new();
    let mut copied = 0usize;
    let mut i = 0usize;

    while i < bytes.len() {
        let at_line_start = i == 0 || bytes[i - 1] == b'\n';
        if at_line_start {
            let mut j = i;
            while j < bytes.len() && (bytes[j] == b' ' || bytes[j] == b'\t') {
                j += 1;
            }
            if keyword_at(src, j, "gate") {
                // A declaration ends at the '}' that closes its body (QASM2
                // writes `gate ... { ... }` with no trailing ';').  A ';'
                // immediately after that brace is consumed with it.
                let mut k = j;
                let mut depth = 0i32;
                while k < bytes.len() {
                    match bytes[k] {
                        b'{' => depth += 1,
                        b'}' => {
                            depth -= 1;
                            if depth <= 0 {
                                k += 1;
                                break;
                            }
                        }
                        _ => {}
                    }
                    k += 1;
                }
                // Skip a trailing ';' (and the rest of that line's blanks).
                let mut trailing = k;
                while trailing < bytes.len()
                    && (bytes[trailing] == b' ' || bytes[trailing] == b'\t')
                {
                    trailing += 1;
                }
                if trailing < bytes.len() && bytes[trailing] == b';' {
                    k = trailing + 1;
                }
                if let Some((name, decl)) = parse_gate_decl(&src[j..k]) {
                    decls.insert(name, decl);
                }
                out.push_str(&src[copied..j]);
                for _ in 0..src[j..k].matches('\n').count() {
                    out.push('\n');
                }
                copied = k;
                i = k;
                continue;
            }
        }
        i += 1;
    }
    out.push_str(&src[copied..]);
    (out, decls)
}

/// Parse `gate name(p1, p2) a, b { stmt; ... }` into its parts.
fn parse_gate_decl(text: &str) -> Option<(String, GateDecl)> {
    let text = text.trim_start();
    if !text.starts_with("gate") {
        return None;
    }
    let text = text[4..].trim_start();
    let brace = text.find('{')?;
    let close = text.rfind('}')?;
    if close < brace {
        return None;
    }
    let header = &text[..brace];
    let body_text = &text[brace + 1..close];

    let (name, params, qargs) = if let Some(lp) = header.find('(') {
        let rp = header.rfind(')')?;
        if rp < lp {
            return None;
        }
        (
            header[..lp].trim().to_string(),
            split_ident_list(&header[lp + 1..rp]),
            split_ident_list(&header[rp + 1..]),
        )
    } else {
        let mut it = header.splitn(2, |c: char| c.is_whitespace());
        let name = it.next()?.trim().to_string();
        (name, Vec::new(), split_ident_list(it.next().unwrap_or("")))
    };
    if name.is_empty() {
        return None;
    }

    // Comments are line-scoped; drop them before splitting on ';' so a
    // comment can never be mistaken for a statement.
    let mut statements = String::new();
    for line in body_text.lines() {
        let code = line.split("//").next().unwrap_or("");
        statements.push(' ');
        statements.push_str(code.trim());
    }
    let body = statements
        .split(';')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();

    Some((
        name,
        GateDecl {
            params,
            qargs,
            body,
        },
    ))
}

/// Split a comma-separated identifier list (`"a, b, c"`), dropping blanks.
fn split_ident_list(s: &str) -> Vec<String> {
    s.split(',')
        .map(|p| p.trim().to_string())
        .filter(|p| !p.is_empty())
        .collect()
}

/// Replace the identifier `ident` in `s` with `with`, respecting word edges.
fn replace_ident(s: &str, ident: &str, with: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut word = String::new();
    for ch in s.chars() {
        if ch.is_alphanumeric() || ch == '_' {
            word.push(ch);
        } else {
            if !word.is_empty() {
                if word == ident {
                    out.push_str(with);
                } else {
                    out.push_str(&word);
                }
                word.clear();
            }
            out.push(ch);
        }
    }
    if !word.is_empty() {
        if word == ident {
            out.push_str(with);
        } else {
            out.push_str(&word);
        }
    }
    out
}

/// Expand one statement, inlining it (recursively) when it names a declared
/// gate.  Statements that are not gate calls are copied through unchanged.
fn expand_line(
    line: &str,
    decls: &HashMap<String, GateDecl>,
    depth: usize,
    out: &mut Vec<String>,
) -> Result<(), String> {
    let t = line.trim();
    if t.is_empty() {
        // Keep the line, so `lineno` in parse errors still matches the input.
        out.push(String::new());
        return Ok(());
    }
    if t.starts_with("//") {
        out.push(t.to_string());
        return Ok(());
    }
    let (name, params, args) = match split_call(t) {
        Some(v) => v,
        None => {
            out.push(t.to_string());
            return Ok(());
        }
    };
    let decl = match decls.get(&name) {
        Some(d) => d,
        None => {
            out.push(t.to_string());
            return Ok(());
        }
    };
    if depth >= 64 {
        return Err(format!("gate '{name}' expands recursively (depth > 64)"));
    }
    if params.len() != decl.params.len() {
        return Err(format!(
            "gate '{name}' expects {} parameter(s), got {}",
            decl.params.len(),
            params.len()
        ));
    }
    if args.len() != decl.qargs.len() {
        return Err(format!(
            "gate '{name}' expects {} qubit operand(s), got {}",
            decl.qargs.len(),
            args.len()
        ));
    }
    for stmt in &decl.body {
        let mut s = stmt.clone();
        // Formal parameters become the call-site expression, parenthesised so
        // `u1(theta/2)` with `theta=0.4` reads `u1((0.4)/2)`.
        for (formal, actual) in decl.params.iter().zip(params.iter()) {
            s = replace_ident(&s, formal, &format!("({actual})"));
        }
        for (formal, actual) in decl.qargs.iter().zip(args.iter()) {
            s = replace_ident(&s, formal, actual);
        }
        expand_line(&s, decls, depth + 1, out)?;
    }
    Ok(())
}

/// Inline every call to a declared gate in `src`.
fn inline_gate_calls(src: &str, decls: &HashMap<String, GateDecl>) -> Result<String, String> {
    let mut out: Vec<String> = Vec::new();
    for line in src.lines() {
        expand_line(line, decls, 0, &mut out)?;
    }
    Ok(out.join("\n"))
}

/// Split `name(p1, p2) a, b` into its raw (un-evaluated) parts.
fn split_call(line: &str) -> Option<(String, Vec<String>, Vec<String>)> {
    let line = line.trim();
    let line = line.strip_suffix(';').unwrap_or(line).trim();
    let (name, params, rest) = if let Some(lp) = line.find('(') {
        let rp = line.rfind(')')?;
        if rp < lp {
            return None;
        }
        (
            line[..lp].trim().to_string(),
            split_ident_list(&line[lp + 1..rp]),
            line[rp + 1..].trim().to_string(),
        )
    } else {
        let mut it = line.splitn(2, |c: char| c.is_whitespace());
        let name = it.next()?.trim().to_string();
        (name, Vec::new(), it.next()?.trim().to_string())
    };
    if name.is_empty() {
        return None;
    }
    let args = split_ident_list(&rest);
    Some((name, params, args))
}

/// Extract the first bracket-enclosed integer from a string, e.g. "q[3]" → Some(3).
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
    let p3 = || params.get(3).copied().unwrap_or(0.0);

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
        // u2(phi, lambda) is u3(pi/2, phi, lambda) -- a qelib1 gate that
        // Qiskit emits in some exports. Without this it used to be dropped
        // silently by the caller's `None =>` arm.
        "u2" => Some(OpType::U(
            Parameter::Const(PI / 2.0),
            Parameter::Const(p0()),
            Parameter::Const(p1()),
        )),
        "cu" | "cu3" => Some(OpType::Cu(
            Parameter::Const(p0()),
            Parameter::Const(p1()),
            Parameter::Const(p2()),
            Parameter::Const(p3()),
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
        // `cu1(lambda)` is diag(1, 1, 1, e^{i*lambda}) -- the same matrix as
        // `cp`/`cphase`, so it maps onto the native CP arm rather than being
        // expanded through its five-gate qelib1 body.
        "cp" | "cr1" | "cphase" | "cu1" => Some(OpType::CP(Parameter::Const(p0()))),
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
    fn test_unsupported_gate_is_an_error_not_a_silent_drop() {
        // Regression: an unmapped gate used to be skipped, so the parse
        // "succeeded" with operations missing. The caller cannot detect that.
        let qasm = "OPENQASM 2.0;\nqreg q[1];\nfoo q[0];\n";
        let err = parse_qasm2(qasm)
            .err()
            .expect("unsupported gate must not parse");
        assert!(err.contains("foo"), "error should name the gate: {err}");
        assert!(err.contains("line 3"), "error should carry the line: {err}");
    }

    #[test]
    fn test_wrong_operand_count_is_an_error_not_a_panic() {
        // Regression: a one-qubit gate naming three qubits used to parse into
        // a three-wire record carrying a 2x2 matrix; `to_unitary` then indexed
        // out of bounds (panicked at dag.rs's 3+ qubit branch).
        let qasm = "OPENQASM 2.0;\nqreg q[5];\nu3(0.0, 0.0, 0.0) q[4], q[2], q[3];\n";
        let err = parse_qasm2(qasm)
            .err()
            .expect("wrong operand count must not parse");
        assert!(
            err.contains("qubit operand"),
            "error should name the operand count, got: {err}"
        );
    }

    #[test]
    fn test_out_of_range_qubit_is_an_error_not_a_silent_drop() {
        let qasm = "OPENQASM 2.0;\nqreg q[2];\nx q[5];\n";
        let err = parse_qasm2(qasm)
            .err()
            .expect("out-of-range qubit must not parse");
        assert!(err.contains("out of range"), "{err}");
    }

    #[test]
    fn test_u2_expands_to_u3_pi_half() {
        // u2(phi, lambda) == u3(pi/2, phi, lambda); it used to be dropped.
        let qasm = "OPENQASM 2.0;\nqreg q[1];\nu2(0.3, 0.7) q[0];\n";
        let dag = parse_qasm2(qasm).expect("u2 should parse");
        assert_eq!(dag.gate_count(), 1);
    }

    #[test]
    fn test_creg_and_bare_barrier_are_not_gates() {
        let qasm =
            "OPENQASM 2.0;\nqreg q[2];\ncreg c[2];\nbarrier;\nh q[0];\nmeasure q[0] -> c[0];\n";
        let dag = parse_qasm2(qasm).expect("declarations must not be treated as gates");
        assert_eq!(dag.n_qubits, 2);
    }

    #[test]
    fn test_instruction_before_qreg_is_an_error() {
        let qasm = "OPENQASM 2.0;\nh q[0];\nqreg q[1];\n";
        let err = parse_qasm2(qasm)
            .err()
            .expect("instruction before qreg must not parse");
        assert!(err.contains("before any qreg"), "{err}");
    }

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
    fn test_angle_expression_arithmetic_without_pi_is_not_zero() {
        // Regression (SF-19): `0.15+0.15` used to hit
        // `parse::<f64>().unwrap_or(0.0)` and become 0.0.
        let eps = 1e-12;
        assert!((eval_expr("0.15+0.15").unwrap() - 0.3).abs() < eps);
        assert!((eval_expr("1 - 0.25").unwrap() - 0.75).abs() < eps);
        assert!((eval_expr("6/4").unwrap() - 1.5).abs() < eps);
    }

    #[test]
    fn test_angle_expression_parens_and_functions() {
        // Regression (SF-19): `(0.3)*pi` and `sin(pi/2)*0.3` used to become
        // leaves of the legacy splitter and evaluate to 0.0.
        let pi = std::f64::consts::PI;
        let eps = 1e-12;
        assert!((eval_expr("(0.3)*pi").unwrap() - 0.3 * pi).abs() < eps);
        assert!((eval_expr("sin(pi/2)*0.3").unwrap() - 0.3).abs() < eps);
        assert!((eval_expr("cos(0)").unwrap() - 1.0).abs() < eps);
        assert!((eval_expr("pi*-0.3").unwrap() + 0.3 * pi).abs() < eps);
        assert!((eval_expr("-(0.1+0.2)").unwrap() + 0.3).abs() < eps);
        assert!((eval_expr("2^3").unwrap() - 8.0).abs() < eps);
    }

    #[test]
    fn test_angle_expression_rejects_garbage() {
        assert!(eval_expr("1.0.2").is_none());
        assert!(eval_expr("").is_none());
        assert!(eval_expr("theta").is_none());
        assert!(eval_expr("1/0").is_none());
        // An unevaluable parameter must be an error, not a silent 0.0.
        let qasm = "OPENQASM 2.0;\nqreg q[1];\nrx(theta) q[0];\n";
        let err = parse_qasm2(qasm).err().expect("must not parse");
        assert!(err.contains("cannot evaluate parameter"), "{err}");
    }

    #[test]
    fn test_parameter_expression_reaches_the_gate_record() {
        let qasm = "OPENQASM 2.0;\nqreg q[1];\nrx(0.15+0.15) q[0];\n";
        let dag = parse_qasm2(qasm).expect("should parse");
        let recs = dag.to_gate_records();
        assert_eq!(recs.len(), 1);
        assert!((recs[0].2[0] - 0.3).abs() < 1e-12, "{recs:?}");
    }

    #[test]
    fn test_qelib1_extra_declaration_expands_instead_of_erroring() {
        // Regression: `ch` has no OpType, so it used to be rejected as
        // "unsupported gate". Its qelib1 body is 7 gates: h, sdg, cx, h, t,
        // cx, t, h, s, x, s -- 11 operations.
        let qasm = "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[2];\nch q[0], q[1];\n";
        let dag = parse_qasm2(qasm).expect("ch should expand");
        assert_eq!(dag.gate_count(), 11);
        let recs = dag.to_gate_records();
        assert!(recs.iter().any(|(n, _, _)| n.eq_ignore_ascii_case("CX")));
    }

    #[test]
    fn test_qelib1_extra_cry_and_c3x_expand() {
        let qasm = "OPENQASM 2.0;\nqreg q[4];\ncry(0.4) q[0], q[1];\nc3x q[0], q[1], q[2], q[3];\n";
        let dag = parse_qasm2(qasm).expect("cry + c3x should expand");
        // cry is 4 gates; c3x is 31 gates in qelib1.
        assert_eq!(dag.gate_count(), 35);
    }

    #[test]
    fn test_user_gate_declaration_is_inlined() {
        let qasm =
            "OPENQASM 2.0;\nqreg q[2];\ngate mybell a, b { h a; cx a, b; }\nmybell q[0], q[1];\n";
        let dag = parse_qasm2(qasm).expect("user declaration should inline");
        assert_eq!(dag.gate_count(), 2);
    }

    #[test]
    fn test_gate_declaration_arity_is_checked() {
        let qasm = "OPENQASM 2.0;\nqreg q[2];\ncry(0.1, 0.2) q[0], q[1];\n";
        let err = parse_qasm2(qasm)
            .err()
            .expect("arity mismatch must not parse");
        assert!(err.contains("parameter"), "{err}");
    }

    #[test]
    fn test_duplicate_qubit_in_one_application_is_an_error() {
        // A minified program used to be read as one gate whose operands were
        // every q[i] on the line; re-importing such an export hit the DAG's
        // duplicate-qubit assert and surfaced as a PyO3 PanicException.
        let qasm = "OPENQASM 2.0;\nqreg q[2];\nh q[0], q[0];\n";
        let err = parse_qasm2(qasm)
            .err()
            .expect("repeated operand must not reach the DAG");
        assert!(err.contains("more than once"), "{err}");
        assert!(err.contains("line 3"), "{err}");
    }

    #[test]
    fn test_whole_register_measure() {
        // `measure q -> c;` is legal QASM 2 and is the final statement random
        // generators write.  It used to be rejected for having no operand.
        let qasm = "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[3];\ncreg c[3];\nh q[0];\nmeasure q -> c;\n";
        let dag = parse_qasm2(qasm).expect("whole-register measure must parse");
        assert_eq!(dag.n_qubits, 3);
        let _order = dag.topological_order();
    }

    #[test]
    fn test_several_statements_on_one_line_are_all_parsed() {
        // Legal QASM2, and the shape "minified" programs take.  It used to
        // yield one gate: the line loop read a single head and dropped the
        // rest of the line.
        let qasm = "OPENQASM 2.0;\nqreg q[2]; h q[0]; h q[1];\n";
        let dag = parse_qasm2(qasm).expect("two statements on one line");
        assert_eq!(dag.n_qubits, 2);
        assert_eq!(dag.gate_count(), 2);
    }

    #[test]
    fn test_whole_program_on_one_line() {
        let qasm = "OPENQASM 2.0; include \"qelib1.inc\"; qreg q[2]; \
                    h q[0]; cx q[0], q[1];\n";
        let dag = parse_qasm2(qasm).expect("header and body can share a line");
        assert_eq!(dag.n_qubits, 2);
        assert_eq!(dag.gate_count(), 2);
    }

    #[test]
    fn test_statements_split_at_semicolons_inside_expresssions_are_kept() {
        // A ';' inside a quoted string or a parenthesised expression must not
        // be treated as a statement boundary.
        let qasm = "OPENQASM 2.0;\nqreg q[1];\nrx((0.1+0.2)) q[0];\n";
        let dag = parse_qasm2(qasm).expect("one statement");
        assert_eq!(dag.gate_count(), 1);
    }

    #[test]
    fn test_cu1_maps_to_the_native_cp_arm() {
        let qasm = "OPENQASM 2.0;\nqreg q[2];\ncu1(0.5) q[0], q[1];\n";
        let dag = parse_qasm2(qasm).expect("cu1 is native");
        assert_eq!(dag.gate_count(), 1);
        let recs = dag.to_gate_records();
        assert!(
            recs[0].2.iter().any(|p| (p - 0.5).abs() < 1e-12),
            "{recs:?}"
        );
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

    #[test]
    fn test_little_endian_passthrough_not_reversed() {
        // Regression: q[i] must map to SF qubit i (no n-1-i reversal), so that
        // to_qasm -> from_qasm2 is an involution, matching the Python bridge.
        let qasm = r#"OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
x q[0];
cx q[0], q[2];
swap q[1], q[2];
"#;
        let dag = parse_qasm2(qasm).expect("should parse");
        let recs = dag.to_gate_records();
        // X must act on qubit 0 (never reversed to n-1).
        assert!(recs
            .iter()
            .any(|(n, q, _)| n.to_uppercase() == "X" && q == &[0]));
        // cx control/target must stay [0, 2], never reversed to [2, 0].
        assert!(recs
            .iter()
            .any(|(n, q, _)| matches!(n.to_uppercase().as_str(), "CX" | "CNOT") && q == &[0, 2]));
        assert!(recs
            .iter()
            .any(|(n, q, _)| n.to_uppercase() == "SWAP" && q == &[1, 2]));
    }

    #[test]
    fn test_multiple_qregs_keep_register_offsets() {
        // Regression (SF-10a): each qreg declaration used to rebuild the DAG,
        // so only the last register survived and took index 0.
        let qasm = "OPENQASM 2.0;\nqreg a[1];\nqreg b[1];\nx b[0];\n";
        let dag = parse_qasm2(qasm).expect("two qregs should parse");
        assert_eq!(dag.gate_count(), 1);
        let recs = dag.to_gate_records();
        assert_eq!(recs[0].0.to_uppercase(), "X");
        assert_eq!(recs[0].1, vec![1]);
    }

    #[test]
    fn test_whole_register_operand_broadcasts() {
        // Regression (SF-09): `h q;` used to fail arity validation instead of
        // applying H to every qubit of the register.
        let qasm = "OPENQASM 2.0;\nqreg q[2];\nh q;\n";
        let dag = parse_qasm2(qasm).expect("whole-register gate should parse");
        assert_eq!(dag.gate_count(), 2);
        let recs = dag.to_gate_records();
        assert_eq!(recs[0].1, vec![0]);
        assert_eq!(recs[1].1, vec![1]);
    }

    #[test]
    fn test_cross_register_two_qubit_gate_uses_offsets() {
        // Regression (SF-10b): `cx a, b;` used to collide both registers on
        // [0, 0] instead of pairing a[i] with b[i].
        let qasm = "OPENQASM 2.0;\nqreg a[2];\nqreg b[2];\ncx a, b;\n";
        let dag = parse_qasm2(qasm).expect("cross-register cx should parse");
        let recs = dag.to_gate_records();
        assert_eq!(recs.len(), 2);
        assert_eq!(recs[0].1, vec![0, 2]);
        assert_eq!(recs[1].1, vec![1, 3]);
    }

    #[test]
    fn test_measure_pairs_register_bits() {
        let qasm = "OPENQASM 2.0;\nqreg q[2];\ncreg c[2];\nmeasure q -> c;\n";
        let dag = parse_qasm2(qasm).expect("whole-register measure should parse");
        let measured: Vec<(Vec<usize>, usize)> = dag
            .to_instructions()
            .iter()
            .filter(|op| matches!(op.op_type, OpType::Measure))
            .map(|op| (op.qubits.to_vec(), op.classical_bits[0]))
            .collect();
        assert_eq!(measured, vec![(vec![0], 0), (vec![1], 1)]);
    }

    #[test]
    fn test_if_condition_is_attached_to_the_next_gate() {
        let qasm = "OPENQASM 2.0;\nqreg q[1];\ncreg c[1];\nif(c==1) x q[0];\n";
        let dag = parse_qasm2(qasm).expect("if statement should parse");
        let instructions = dag.to_instructions();
        let op = instructions
            .iter()
            .find(|op| matches!(op.op_type, OpType::X))
            .expect("X should be recorded");
        assert_eq!(op.condition, Some((0, 1)));
    }

    #[test]
    fn test_mixing_whole_register_and_indexed_operands_is_rejected() {
        let qasm = "OPENQASM 2.0;\nqreg a[2];\nqreg b[2];\ncx a, b[0];\n";
        let err = parse_qasm2(qasm).err().expect("mixed operands must fail");
        assert!(err.contains("mixes whole-register"), "{err}");
    }

    #[test]
    fn test_declaration_only_program_is_an_empty_dag() {
        let qasm = "OPENQASM 2.0;\nqreg q[2];\n";
        let dag = parse_qasm2(qasm).expect("declarations only should parse");
        assert_eq!(dag.n_qubits, 2);
        assert_eq!(dag.gate_count(), 0);
    }

    #[test]
    fn test_conditioned_records_keep_classical_bits() {
        let qasm = "OPENQASM 2.0;\nqreg q[2];\ncreg c[2];\nmeasure q[1] -> c[0];\n";
        let dag = parse_qasm2(qasm).expect("measure should parse");
        let records = dag.to_gate_records_with_conditions();
        let measure = records
            .iter()
            .find(|record| record.0.eq_ignore_ascii_case("MEASURE"))
            .expect("measurement record");
        assert_eq!(measure.1, vec![1]);
        assert_eq!(measure.4, vec![0]);
    }
}
