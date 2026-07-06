//! Superfermion Pulse Crate — Rust-native pulse-level quantum control.
//!
//! Provides high-performance waveform generation, schedule construction,
//! and gate calibration for pulse-level programming.

pub mod waveforms;
pub mod schedule;
pub mod calibration;

pub use waveforms::{Waveform, WaveformType, PulseEnvelope};
pub use schedule::{PulseSchedule, PulseInstruction, ChannelId, ChannelType};
pub use calibration::{GateCalibration, CalibrationDatabase};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crate_imports() {
        let _ = WaveformType::Gaussian;
        let _ = ChannelType::Drive;
    }
}
