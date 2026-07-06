//! Rust-native serialization for QuantumDAG.
//!
//! Provides JSON and binary (bincode) serialization for circuits.

use crate::QuantumDAG;
use serde::{Deserialize, Serialize};

/// Serializable snapshot of a QuantumDAG.
///
/// Since petgraph's StableDiGraph is not trivially serializable in a
/// portable way, we convert to an instruction list first.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SerializedCircuit {
    pub n_qubits: usize,
    pub n_cbits: usize,
    #[serde(alias = "gates")]
    pub instructions: Vec<SerializedOp>,
    #[serde(default)]
    pub metadata: SerializedMetadata,
    #[serde(default)]
    pub parameters: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SerializedOp {
    #[serde(alias = "name")]
    pub op_type: String,
    pub qubits: Vec<usize>,
    #[serde(default)]
    pub classical_bits: Vec<usize>,
    pub params: Vec<f64>,
    #[serde(default)]
    pub param_names: Vec<String>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct SerializedMetadata {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default, alias = "target_backend")]
    pub target_backend: Option<String>,
    #[serde(default)]
    pub optimization_level: u8,
}

impl SerializedCircuit {
    /// Serialize a QuantumDAG to a portable format.
    pub fn from_dag(dag: &QuantumDAG) -> Self {
        let instructions: Vec<SerializedOp> = dag
            .to_instructions()
            .iter()
            .map(|op| {
                let mut params = Vec::new();
                let mut param_names = Vec::new();
                for p in op.op_type.parameters() {
                    match p {
                        crate::ops::Parameter::Const(v) => {
                            params.push(*v);
                            param_names.push(String::new());
                        }
                        crate::ops::Parameter::Variable { name, .. } => {
                            params.push(0.0);
                            param_names.push(name.clone());
                        }
                        crate::ops::Parameter::Expr(_) => {
                            if let Some(v) = p.try_evaluate() {
                                params.push(v);
                            } else {
                                params.push(0.0);
                            }
                            param_names.push(String::new());
                        }
                    }
                }
                SerializedOp {
                    op_type: op.op_type.name().to_string(),
                    qubits: op.qubits.to_vec(),
                    classical_bits: op.classical_bits.to_vec(),
                    params,
                    param_names,
                }
            })
            .collect();

        SerializedCircuit {
            n_qubits: dag.n_qubits,
            n_cbits: dag.n_cbits,
            instructions,
            metadata: SerializedMetadata {
                name: dag.metadata.name.clone(),
                target_backend: dag.metadata.target_backend.clone(),
                optimization_level: dag.metadata.optimization_level,
            },
            parameters: dag.parameter_names(),
        }
    }

    /// Serialize to JSON string.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Deserialize from JSON string.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// Rebuild a QuantumDAG from this serialized form.
    pub fn to_dag(&self) -> QuantumDAG {
        let mut dag = QuantumDAG::new(self.n_qubits, self.n_cbits);
        dag.metadata.name = self.metadata.name.clone();
        dag.metadata.target_backend = self.metadata.target_backend.clone();
        dag.metadata.optimization_level = self.metadata.optimization_level;

        for inst in &self.instructions {
            let op_type = self.rebuild_op_type(inst);
            if let Some(op) = op_type {
                if inst.classical_bits.is_empty() {
                    dag.add_op(op, &inst.qubits);
                } else {
                    // Measurement
                    for &cbit in &inst.classical_bits {
                        dag.add_measure(inst.qubits[0], cbit);
                    }
                }
            }
        }

        dag
    }

    fn rebuild_op_type(&self, inst: &SerializedOp) -> Option<crate::OpType> {
        use crate::ops::Parameter;
        use crate::OpType;

        let param = |idx: usize| -> Parameter {
            if idx < inst.param_names.len() && !inst.param_names[idx].is_empty() {
                Parameter::Variable {
                    name: inst.param_names[idx].clone(),
                    id: 0,
                }
            } else if idx < inst.params.len() {
                Parameter::Const(inst.params[idx])
            } else {
                Parameter::Const(0.0)
            }
        };

        match inst.op_type.as_str() {
            "H" => Some(OpType::H),
            "X" => Some(OpType::X),
            "Y" => Some(OpType::Y),
            "Z" => Some(OpType::Z),
            "S" => Some(OpType::S),
            "Sdg" => Some(OpType::Sdg),
            "T" => Some(OpType::T),
            "Tdg" => Some(OpType::Tdg),
            "SX" => Some(OpType::SX),
            "SXdg" => Some(OpType::SXdg),
            "Id" => Some(OpType::Id),
            "Rx" => Some(OpType::Rx(param(0))),
            "Ry" => Some(OpType::Ry(param(0))),
            "Rz" => Some(OpType::Rz(param(0))),
            "R1" => Some(OpType::R1(param(0))),
            "P" => Some(OpType::P(param(0))),
            "U" => Some(OpType::U(param(0), param(1), param(2))),
            "CNOT" => Some(OpType::CNOT),
            "CZ" => Some(OpType::CZ),
            "CY" => Some(OpType::CY),
            "SWAP" => Some(OpType::SWAP),
            "iSWAP" => Some(OpType::ISWAP),
            "ECR" => Some(OpType::ECR),
            "Rzz" => Some(OpType::Rzz(param(0))),
            "Rxx" => Some(OpType::Rxx(param(0))),
            "Ryy" => Some(OpType::Ryy(param(0))),
            "CRx" => Some(OpType::CRx(param(0))),
            "CRz" => Some(OpType::CRz(param(0))),
            "CCX" => Some(OpType::CCX),
            "CSWAP" => Some(OpType::CSWAP),
            "Measure" => None, // Handled separately
            "Barrier" => Some(OpType::Barrier),
            "Reset" => Some(OpType::Reset),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::OpType;

    #[test]
    fn test_json_roundtrip() {
        let mut dag = QuantumDAG::new(2, 0);
        dag.add_op(OpType::H, &[0]);
        dag.add_op(OpType::CNOT, &[0, 1]);

        let serialized = SerializedCircuit::from_dag(&dag);
        let json = serialized.to_json().unwrap();
        let deserialized = SerializedCircuit::from_json(&json).unwrap();
        let rebuilt = deserialized.to_dag();

        assert_eq!(rebuilt.n_qubits, 2);
        assert_eq!(rebuilt.gate_count(), 2);
    }

    #[test]
    fn test_serialized_circuit_fields() {
        let mut dag = QuantumDAG::new(3, 1);
        dag.metadata.name = Some("test".to_string());
        dag.add_op(OpType::H, &[0]);
        dag.add_op(OpType::Rx(crate::ops::Parameter::Const(1.57)), &[1]);

        let s = SerializedCircuit::from_dag(&dag);
        assert_eq!(s.n_qubits, 3);
        assert_eq!(s.n_cbits, 1);
        assert_eq!(s.instructions.len(), 2);

        // Gates are on different qubits — topological order may vary
        let op_types: Vec<&str> = s.instructions.iter().map(|i| i.op_type.as_str()).collect();
        assert!(op_types.contains(&"H"));
        assert!(op_types.contains(&"Rx"));

        let rx_inst = s.instructions.iter().find(|i| i.op_type == "Rx").unwrap();
        assert!((rx_inst.params[0] - 1.57).abs() < 1e-10);
    }
}
