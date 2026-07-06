//! Pulse schedule — time-ordered sequence of instructions on channels.

use crate::waveforms::PulseEnvelope;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Channel type.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ChannelType {
    Drive,
    Control,
    Measure,
    Acquire,
}

/// Channel identifier.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ChannelId {
    pub channel_type: ChannelType,
    pub index: usize,
}

impl ChannelId {
    pub fn drive(qubit: usize) -> Self {
        Self { channel_type: ChannelType::Drive, index: qubit }
    }
    pub fn control(index: usize) -> Self {
        Self { channel_type: ChannelType::Control, index }
    }
    pub fn measure(qubit: usize) -> Self {
        Self { channel_type: ChannelType::Measure, index: qubit }
    }
    pub fn acquire(qubit: usize) -> Self {
        Self { channel_type: ChannelType::Acquire, index: qubit }
    }

    pub fn name(&self) -> String {
        let prefix = match self.channel_type {
            ChannelType::Drive => "d",
            ChannelType::Control => "u",
            ChannelType::Measure => "m",
            ChannelType::Acquire => "a",
        };
        format!("{}{}", prefix, self.index)
    }
}

/// A single pulse instruction.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PulseInstruction {
    pub envelope: PulseEnvelope,
    pub channel: ChannelId,
    pub t0: usize,
    pub phase: f64,
    pub frequency: f64,
}

impl PulseInstruction {
    pub fn t1(&self) -> usize {
        self.t0 + self.envelope.duration
    }
}

/// A pulse schedule — collection of timed instructions on channels.
#[derive(Clone, Debug, Default)]
pub struct PulseSchedule {
    pub name: String,
    instructions: Vec<PulseInstruction>,
    /// Per-channel cursor for sequential appending
    cursors: HashMap<String, usize>,
}

impl PulseSchedule {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            instructions: Vec::new(),
            cursors: HashMap::new(),
        }
    }

    /// Add a pulse instruction at a specific time.
    pub fn play(&mut self, envelope: PulseEnvelope, channel: ChannelId, t0: Option<usize>) -> &mut Self {
        let ch_name = channel.name();
        let start = t0.unwrap_or_else(|| *self.cursors.get(&ch_name).unwrap_or(&0));

        self.instructions.push(PulseInstruction {
            envelope: envelope.clone(),
            channel,
            t0: start,
            phase: 0.0,
            frequency: 0.0,
        });

        self.cursors.insert(ch_name, start + envelope.duration);
        self
    }

    /// Add a phase shift on a channel.
    pub fn shift_phase(&mut self, phase: f64, channel: ChannelId) -> &mut Self {
        let ch_name = channel.name();
        let t0 = *self.cursors.get(&ch_name).unwrap_or(&0);
        self.instructions.push(PulseInstruction {
            envelope: PulseEnvelope::square(0, 0.0),
            channel,
            t0,
            phase,
            frequency: 0.0,
        });
        self
    }

    /// Barrier: synchronize channels to the latest cursor.
    pub fn barrier(&mut self) -> &mut Self {
        let max_t = self.cursors.values().copied().max().unwrap_or(0);
        for cursor in self.cursors.values_mut() {
            *cursor = max_t;
        }
        self
    }

    /// Total schedule duration.
    pub fn duration(&self) -> usize {
        self.instructions
            .iter()
            .map(|inst| inst.t1())
            .max()
            .unwrap_or(0)
    }

    /// Number of instructions.
    pub fn n_instructions(&self) -> usize {
        self.instructions.len()
    }

    /// Get all instructions sorted by time.
    pub fn instructions(&self) -> Vec<&PulseInstruction> {
        let mut sorted: Vec<&PulseInstruction> = self.instructions.iter().collect();
        sorted.sort_by_key(|inst| (inst.t0, inst.channel.name()));
        sorted
    }

    /// List of channels used.
    pub fn channels(&self) -> Vec<String> {
        let mut chs: Vec<String> = self.cursors.keys().cloned().collect();
        chs.sort();
        chs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_schedule_basic() {
        let mut sched = PulseSchedule::new("test");
        sched.play(
            PulseEnvelope::gaussian(160, 40.0, 0.5),
            ChannelId::drive(0),
            None,
        );
        assert_eq!(sched.duration(), 160);
        assert_eq!(sched.n_instructions(), 1);
    }

    #[test]
    fn test_schedule_sequential() {
        let mut sched = PulseSchedule::new("test");
        sched.play(PulseEnvelope::gaussian(100, 25.0, 0.5), ChannelId::drive(0), None);
        sched.play(PulseEnvelope::gaussian(100, 25.0, 0.3), ChannelId::drive(0), None);
        assert_eq!(sched.duration(), 200);
    }

    #[test]
    fn test_schedule_barrier() {
        let mut sched = PulseSchedule::new("test");
        sched.play(PulseEnvelope::gaussian(200, 50.0, 0.5), ChannelId::drive(0), None);
        sched.play(PulseEnvelope::gaussian(100, 25.0, 0.3), ChannelId::drive(1), None);
        sched.barrier();
        sched.play(PulseEnvelope::gaussian(50, 12.0, 0.4), ChannelId::drive(1), None);
        assert_eq!(sched.duration(), 250);
    }
}
