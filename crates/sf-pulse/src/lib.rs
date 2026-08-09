//! Superfermion Pulse Crate — Rust-native pulse-level quantum control.
//!
//! Provides high-performance waveform generation, schedule construction,
//! and gate calibration for pulse-level programming.

pub mod calibration;
pub mod schedule;
pub mod waveforms;

pub use calibration::{CalibrationDatabase, GateCalibration};
pub use schedule::{ChannelId, ChannelType, PulseInstruction, PulseSchedule};
pub use waveforms::{PulseEnvelope, Waveform, WaveformType};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crate_imports() {
        let _ = WaveformType::Gaussian;
        let _ = ChannelType::Drive;
    }
}
