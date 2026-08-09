//! Pulse waveform envelope generators.

use num_complex::Complex64;
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

/// Waveform envelope type.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum WaveformType {
    Gaussian,
    DRAG,
    Square,
    GaussianSquare,
    Cosine,
    Custom,
}

/// Pulse envelope parameters.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PulseEnvelope {
    pub waveform_type: WaveformType,
    pub duration: usize,
    pub amp: f64,
    pub sigma: f64,
    pub beta: f64,    // DRAG coefficient
    pub width: usize, // GaussianSquare flat-top width
}

impl PulseEnvelope {
    pub fn gaussian(duration: usize, sigma: f64, amp: f64) -> Self {
        Self {
            waveform_type: WaveformType::Gaussian,
            duration,
            amp,
            sigma,
            beta: 0.0,
            width: 0,
        }
    }

    pub fn drag(duration: usize, sigma: f64, amp: f64, beta: f64) -> Self {
        Self {
            waveform_type: WaveformType::DRAG,
            duration,
            amp,
            sigma,
            beta,
            width: 0,
        }
    }

    pub fn square(duration: usize, amp: f64) -> Self {
        Self {
            waveform_type: WaveformType::Square,
            duration,
            amp,
            sigma: 0.0,
            beta: 0.0,
            width: 0,
        }
    }

    pub fn gaussian_square(duration: usize, sigma: f64, amp: f64, width: usize) -> Self {
        Self {
            waveform_type: WaveformType::GaussianSquare,
            duration,
            amp,
            sigma,
            width,
            beta: 0.0,
        }
    }
}

/// A waveform — computed samples from an envelope.
#[derive(Clone, Debug)]
pub struct Waveform {
    pub envelope: PulseEnvelope,
    pub samples: Vec<Complex64>,
}

impl Waveform {
    /// Generate waveform samples from an envelope.
    pub fn from_envelope(env: &PulseEnvelope) -> Self {
        let samples = match env.waveform_type {
            WaveformType::Gaussian => Self::gen_gaussian(env),
            WaveformType::DRAG => Self::gen_drag(env),
            WaveformType::Square => Self::gen_square(env),
            WaveformType::GaussianSquare => Self::gen_gaussian_square(env),
            WaveformType::Cosine => Self::gen_cosine(env),
            WaveformType::Custom => vec![Complex64::new(0.0, 0.0); env.duration],
        };
        Self {
            envelope: env.clone(),
            samples,
        }
    }

    fn gen_gaussian(env: &PulseEnvelope) -> Vec<Complex64> {
        let center = env.duration as f64 / 2.0;
        (0..env.duration)
            .map(|t| {
                let x = (t as f64 - center) / env.sigma;
                Complex64::new(env.amp * (-0.5 * x * x).exp(), 0.0)
            })
            .collect()
    }

    fn gen_drag(env: &PulseEnvelope) -> Vec<Complex64> {
        let center = env.duration as f64 / 2.0;
        (0..env.duration)
            .map(|t| {
                let x = (t as f64 - center) / env.sigma;
                let gauss = env.amp * (-0.5 * x * x).exp();
                let d_gauss = -x / env.sigma * gauss;
                Complex64::new(gauss, env.beta * d_gauss)
            })
            .collect()
    }

    fn gen_square(env: &PulseEnvelope) -> Vec<Complex64> {
        vec![Complex64::new(env.amp, 0.0); env.duration]
    }

    fn gen_gaussian_square(env: &PulseEnvelope) -> Vec<Complex64> {
        let rise_time = (env.duration - env.width) / 2;
        (0..env.duration)
            .map(|t| {
                let val = if t < rise_time {
                    let x = (t as f64 - rise_time as f64) / env.sigma;
                    (-0.5 * x * x).exp()
                } else if t < rise_time + env.width {
                    1.0
                } else {
                    let x = (t as f64 - (rise_time + env.width) as f64) / env.sigma;
                    (-0.5 * x * x).exp()
                };
                Complex64::new(env.amp * val, 0.0)
            })
            .collect()
    }

    fn gen_cosine(env: &PulseEnvelope) -> Vec<Complex64> {
        (0..env.duration)
            .map(|t| {
                let phase = PI * t as f64 / env.duration as f64;
                Complex64::new(env.amp * phase.sin().powi(2), 0.0)
            })
            .collect()
    }

    /// Duration in samples.
    pub fn duration(&self) -> usize {
        self.samples.len()
    }

    /// Peak amplitude.
    pub fn peak_amp(&self) -> f64 {
        self.samples
            .iter()
            .map(|s| s.norm())
            .fold(0.0_f64, f64::max)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gaussian() {
        let env = PulseEnvelope::gaussian(160, 40.0, 0.5);
        let wf = Waveform::from_envelope(&env);
        assert_eq!(wf.duration(), 160);
        assert!(wf.peak_amp() <= 0.5 + 1e-10);
    }

    #[test]
    fn test_drag() {
        let env = PulseEnvelope::drag(160, 40.0, 0.5, 0.3);
        let wf = Waveform::from_envelope(&env);
        assert_eq!(wf.duration(), 160);
        // DRAG should have imaginary components
        assert!(wf.samples.iter().any(|s| s.im.abs() > 1e-10));
    }

    #[test]
    fn test_square() {
        let env = PulseEnvelope::square(100, 0.7);
        let wf = Waveform::from_envelope(&env);
        assert_eq!(wf.duration(), 100);
        for s in &wf.samples {
            assert!((s.re - 0.7).abs() < 1e-10);
        }
    }

    #[test]
    fn test_gaussian_square() {
        let env = PulseEnvelope::gaussian_square(640, 64.0, 0.3, 400);
        let wf = Waveform::from_envelope(&env);
        assert_eq!(wf.duration(), 640);
        // Middle should be at full amp
        assert!((wf.samples[320].re - 0.3).abs() < 1e-10);
    }
}
