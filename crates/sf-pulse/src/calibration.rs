//! Gate calibration — mapping quantum gates to calibrated pulse schedules.

use crate::schedule::{ChannelId, PulseSchedule};
use crate::waveforms::PulseEnvelope;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Calibration for a single gate on specific qubits.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GateCalibration {
    pub gate_name: String,
    pub qubits: Vec<usize>,
    pub fidelity: f64,
    pub duration_dt: usize,
}

impl GateCalibration {
    pub fn key(&self) -> String {
        let qs: Vec<String> = self.qubits.iter().map(|q| q.to_string()).collect();
        format!("{}({})", self.gate_name, qs.join(","))
    }
}

/// Database of gate calibrations for a backend.
#[derive(Clone, Debug, Default)]
pub struct CalibrationDatabase {
    pub backend_name: String,
    pub dt_ns: f64,
    calibrations: HashMap<String, GateCalibration>,
}

impl CalibrationDatabase {
    pub fn new(backend: impl Into<String>, dt_ns: f64) -> Self {
        Self {
            backend_name: backend.into(),
            dt_ns,
            calibrations: HashMap::new(),
        }
    }

    pub fn add(&mut self, cal: GateCalibration) {
        self.calibrations.insert(cal.key(), cal);
    }

    pub fn get(&self, gate: &str, qubits: &[usize]) -> Option<&GateCalibration> {
        let qs: Vec<String> = qubits.iter().map(|q| q.to_string()).collect();
        let key = format!("{}({})", gate, qs.join(","));
        self.calibrations.get(&key)
    }

    /// Add default transmon calibrations for a qubit.
    pub fn add_defaults_for_qubit(&mut self, qubit: usize) {
        self.add(GateCalibration {
            gate_name: "x".to_string(),
            qubits: vec![qubit],
            fidelity: 0.9995,
            duration_dt: 160,
        });
        self.add(GateCalibration {
            gate_name: "sx".to_string(),
            qubits: vec![qubit],
            fidelity: 0.9998,
            duration_dt: 160,
        });
        self.add(GateCalibration {
            gate_name: "rz".to_string(),
            qubits: vec![qubit],
            fidelity: 1.0,
            duration_dt: 0, // Virtual Z
        });
    }

    /// Add default CX calibration for a qubit pair.
    pub fn add_cx_calibration(&mut self, control: usize, target: usize) {
        self.add(GateCalibration {
            gate_name: "cx".to_string(),
            qubits: vec![control, target],
            fidelity: 0.995,
            duration_dt: 640,
        });
    }

    /// Build a pulse schedule for a gate.
    pub fn build_schedule(&self, gate: &str, qubits: &[usize]) -> Option<PulseSchedule> {
        let cal = self.get(gate, qubits)?;
        let mut sched = PulseSchedule::new(format!("{}_{}", gate, cal.key()));

        match gate {
            "x" => {
                sched.play(
                    PulseEnvelope::drag(cal.duration_dt, 40.0, 0.5, 0.3),
                    ChannelId::drive(qubits[0]),
                    None,
                );
            }
            "sx" => {
                sched.play(
                    PulseEnvelope::drag(cal.duration_dt, 40.0, 0.25, 0.3),
                    ChannelId::drive(qubits[0]),
                    None,
                );
            }
            "rz" => {
                sched.shift_phase(0.0, ChannelId::drive(qubits[0]));
            }
            "cx" => {
                let cr_idx = qubits[0] * 2 + if qubits[1] > qubits[0] { 0 } else { 1 };
                sched.play(
                    PulseEnvelope::gaussian_square(cal.duration_dt, 64.0, 0.3, 400),
                    ChannelId::control(cr_idx),
                    None,
                );
            }
            _ => return None,
        }

        Some(sched)
    }

    pub fn n_calibrations(&self) -> usize {
        self.calibrations.len()
    }

    pub fn list_gates(&self) -> Vec<&GateCalibration> {
        self.calibrations.values().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calibration_database() {
        let mut db = CalibrationDatabase::new("ibm_brisbane", 0.222);
        db.add_defaults_for_qubit(0);
        db.add_defaults_for_qubit(1);
        db.add_cx_calibration(0, 1);

        assert_eq!(db.n_calibrations(), 7); // 3*2 + 1

        let x_cal = db.get("x", &[0]).unwrap();
        assert_eq!(x_cal.duration_dt, 160);
        assert!(x_cal.fidelity > 0.999);

        let cx_cal = db.get("cx", &[0, 1]).unwrap();
        assert_eq!(cx_cal.duration_dt, 640);
    }

    #[test]
    fn test_build_schedule() {
        let mut db = CalibrationDatabase::new("test", 0.222);
        db.add_defaults_for_qubit(0);

        let sched = db.build_schedule("x", &[0]).unwrap();
        assert_eq!(sched.duration(), 160);
    }
}
