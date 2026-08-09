//! Classical registers — creg management for measurement results and conditionals.

use serde::{Deserialize, Serialize};

/// A classical register holding measurement results.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ClassicalRegister {
    pub name: String,
    pub size: usize,
    /// Current values (if simulated)
    values: Vec<u8>,
}

impl ClassicalRegister {
    pub fn new(name: impl Into<String>, size: usize) -> Self {
        Self {
            name: name.into(),
            size,
            values: vec![0; size],
        }
    }

    pub fn get(&self, index: usize) -> u8 {
        assert!(
            index < self.size,
            "Classical bit index {} out of range (size {})",
            index,
            self.size
        );
        self.values[index]
    }

    pub fn set(&mut self, index: usize, value: u8) {
        assert!(index < self.size);
        self.values[index] = value & 1; // Only 0 or 1
    }

    /// Read the whole register as a little-endian integer.
    pub fn read_int(&self) -> u64 {
        let mut result = 0u64;
        for (i, &v) in self.values.iter().enumerate() {
            result |= (v as u64) << i;
        }
        result
    }

    pub fn reset(&mut self) {
        self.values.fill(0);
    }
}

/// Manager for multiple classical registers in a circuit.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ClassicalRegFile {
    registers: Vec<ClassicalRegister>,
    /// Total flat index → (register_index, bit_index)
    flat_map: Vec<(usize, usize)>,
}

impl ClassicalRegFile {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a new register, return its starting flat index.
    pub fn add_register(&mut self, name: impl Into<String>, size: usize) -> usize {
        let start = self.flat_map.len();
        let reg_idx = self.registers.len();
        self.registers.push(ClassicalRegister::new(name, size));
        for bit in 0..size {
            self.flat_map.push((reg_idx, bit));
        }
        start
    }

    /// Total number of classical bits across all registers.
    pub fn total_bits(&self) -> usize {
        self.flat_map.len()
    }

    /// Number of registers.
    pub fn n_registers(&self) -> usize {
        self.registers.len()
    }

    /// Get a bit by flat index.
    pub fn get_bit(&self, flat_idx: usize) -> u8 {
        let (reg, bit) = self.flat_map[flat_idx];
        self.registers[reg].get(bit)
    }

    /// Set a bit by flat index.
    pub fn set_bit(&mut self, flat_idx: usize, value: u8) {
        let (reg, bit) = self.flat_map[flat_idx];
        self.registers[reg].set(bit, value);
    }

    /// Read an entire register by name.
    pub fn read_register(&self, name: &str) -> Option<u64> {
        self.registers
            .iter()
            .find(|r| r.name == name)
            .map(|r| r.read_int())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classical_register() {
        let mut reg = ClassicalRegister::new("c", 4);
        reg.set(0, 1);
        reg.set(2, 1);
        assert_eq!(reg.get(0), 1);
        assert_eq!(reg.get(1), 0);
        assert_eq!(reg.read_int(), 0b0101); // bits 0 and 2 set
    }

    #[test]
    fn test_reg_file() {
        let mut rf = ClassicalRegFile::new();
        rf.add_register("c0", 4);
        rf.add_register("c1", 2);
        assert_eq!(rf.total_bits(), 6);
        assert_eq!(rf.n_registers(), 2);

        rf.set_bit(0, 1); // c0[0]
        rf.set_bit(4, 1); // c1[0]
        assert_eq!(rf.get_bit(0), 1);
        assert_eq!(rf.read_register("c0"), Some(1));
        assert_eq!(rf.read_register("c1"), Some(1));
    }
}
