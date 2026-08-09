//! GPU-accelerated quantum simulation via CUDA (cudarc).
//!
//! Provides statevector simulation on NVIDIA GPUs using PTX kernels
//! launched through cudarc. Dynamically loads the CUDA driver at runtime —
//! compiles on machines without CUDA, gracefully returns `is_available() = false`.

use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use num_complex::Complex64;
use std::collections::HashMap;
use std::sync::{Arc, OnceLock};
use thiserror::Error;

static GPU_STATE: OnceLock<Option<GpuState>> = OnceLock::new();

const KERNEL_SOURCE: &str = include_str!("kernels/gates.cu");

#[derive(Error, Debug)]
pub enum GpuError {
    #[error("No CUDA GPU detected. Use device='cpu'.")]
    NotAvailable,
    #[error("Circuit has {n_qubits} qubits — requires {required_mb}MB VRAM. Your GPU has {available_mb}MB.")]
    InsufficientVram {
        n_qubits: usize,
        required_mb: u64,
        available_mb: u64,
    },
    #[error("CUDA error: {0}")]
    Cuda(String),
}

#[derive(Debug, Clone)]
pub struct GpuInfo {
    pub name: String,
    pub total_memory_mb: u64,
}

struct GpuState {
    _ctx: Arc<CudaContext>,
    module: Arc<CudaModule>,
    stream: Arc<CudaStream>,
}

fn init_gpu() -> Option<GpuState> {
    let ctx = CudaContext::new(0).ok()?;
    let stream = ctx.default_stream();

    let (major, minor) = ctx.compute_capability().ok()?;
    let arch_str: String = format!("sm_{}{}", major, minor);
    let arch: &'static str = Box::leak(arch_str.into_boxed_str());

    let opts = cudarc::nvrtc::CompileOptions {
        arch: Some(arch),
        ..Default::default()
    };
    let ptx = cudarc::nvrtc::compile_ptx_with_opts(KERNEL_SOURCE, opts).ok()?;
    let module = ctx.load_module(ptx).ok()?;
    Some(GpuState {
        _ctx: ctx,
        module,
        stream,
    })
}

fn get_state() -> Result<&'static GpuState, GpuError> {
    let state = GPU_STATE.get_or_init(init_gpu);
    state.as_ref().ok_or(GpuError::NotAvailable)
}

/// Returns true if a CUDA GPU is available and kernels compile successfully.
pub fn is_available() -> bool {
    get_state().is_ok()
}

/// Diagnostic: returns detailed reason why GPU init failed (or "ok").
#[allow(clippy::needless_return)]
pub fn diagnose() -> String {
    match CudaContext::new(0) {
        Err(e) => return format!("CudaContext::new(0) failed: {:?}", e),
        Ok(ctx) => {
            let (major, minor) = match ctx.compute_capability() {
                Err(e) => return format!("compute_capability failed: {:?}", e),
                Ok(cc) => cc,
            };
            let arch_str = format!("sm_{}{}", major, minor);
            let arch: &'static str = Box::leak(arch_str.into_boxed_str());
            let opts = cudarc::nvrtc::CompileOptions {
                arch: Some(arch),
                ..Default::default()
            };
            match cudarc::nvrtc::compile_ptx_with_opts(KERNEL_SOURCE, opts) {
                Err(e) => {
                    return format!("compile_ptx failed (arch=sm_{}{}): {:?}", major, minor, e)
                }
                Ok(ptx) => match ctx.load_module(ptx) {
                    Err(e) => return format!("load_module failed: {:?}", e),
                    Ok(_) => return format!("ok (sm_{}{}, GPU ready)", major, minor),
                },
            }
        }
    }
}

/// Returns GPU hardware info, or None if unavailable.
pub fn gpu_info() -> Option<GpuInfo> {
    let state = get_state().ok()?;
    let total_mem = state.stream.context().total_mem().ok()?;
    Some(GpuInfo {
        name: "CUDA Device 0".to_string(),
        total_memory_mb: (total_mem / (1024 * 1024)) as u64,
    })
}

/// Gate operation for the GPU simulator.
pub struct GateOp {
    pub name: String,
    pub qubits: Vec<usize>,
    /// Row-major matrix entries (real parts). For 1q: 4 elements; for 2q: 16 elements.
    /// For diagonal 1q gates: 2 elements (d0, d1).
    pub matrix_re: Vec<f64>,
    /// Row-major matrix entries (imaginary parts).
    pub matrix_im: Vec<f64>,
    /// True for diagonal gates (RZ, S, T, P) — only 2 diagonal elements needed.
    pub is_diagonal: bool,
}

fn cuda_err(e: impl std::fmt::Debug) -> GpuError {
    GpuError::Cuda(format!("{:?}", e))
}

/// Simulate a circuit on GPU, returning the final statevector.
pub fn simulate_statevector(n_qubits: usize, gates: &[GateOp]) -> Result<Vec<Complex64>, GpuError> {
    let state = get_state()?;
    let dim: u64 = 1u64 << n_qubits;

    let required_bytes = 2 * dim * 8;
    let total_mem = state.stream.context().total_mem().map_err(cuda_err)? as u64;
    if required_bytes > total_mem * 9 / 10 {
        return Err(GpuError::InsufficientVram {
            n_qubits,
            required_mb: required_bytes / (1024 * 1024),
            available_mb: total_mem / (1024 * 1024),
        });
    }

    // Initialize |0...0> on GPU
    let mut state_re_host = vec![0.0f64; dim as usize];
    state_re_host[0] = 1.0;
    let state_im_host = vec![0.0f64; dim as usize];

    let mut dev_re: CudaSlice<f64> = state.stream.clone_htod(&state_re_host).map_err(cuda_err)?;
    let mut dev_im: CudaSlice<f64> = state.stream.clone_htod(&state_im_host).map_err(cuda_err)?;

    let fn_1q: CudaFunction = state
        .module
        .load_function("apply_gate_1q")
        .map_err(cuda_err)?;
    let fn_diag: CudaFunction = state
        .module
        .load_function("apply_diagonal_1q")
        .map_err(cuda_err)?;
    let fn_2q: CudaFunction = state
        .module
        .load_function("apply_gate_2q")
        .map_err(cuda_err)?;

    for gate in gates {
        match gate.qubits.len() {
            1 => {
                let target = gate.qubits[0] as i32;
                let n_q = n_qubits as i32;

                if gate.is_diagonal {
                    let cfg = LaunchConfig::for_num_elems(dim as u32);
                    unsafe {
                        state
                            .stream
                            .launch_builder(&fn_diag)
                            .arg(&mut dev_re)
                            .arg(&mut dev_im)
                            .arg(&n_q)
                            .arg(&target)
                            .arg(&gate.matrix_re[0])
                            .arg(&gate.matrix_im[0])
                            .arg(&gate.matrix_re[1])
                            .arg(&gate.matrix_im[1])
                            .launch(cfg)
                    }
                    .map_err(cuda_err)?;
                } else {
                    let num_pairs = (dim / 2) as u32;
                    let cfg = LaunchConfig::for_num_elems(num_pairs);
                    unsafe {
                        state
                            .stream
                            .launch_builder(&fn_1q)
                            .arg(&mut dev_re)
                            .arg(&mut dev_im)
                            .arg(&n_q)
                            .arg(&target)
                            .arg(&gate.matrix_re[0])
                            .arg(&gate.matrix_im[0])
                            .arg(&gate.matrix_re[1])
                            .arg(&gate.matrix_im[1])
                            .arg(&gate.matrix_re[2])
                            .arg(&gate.matrix_im[2])
                            .arg(&gate.matrix_re[3])
                            .arg(&gate.matrix_im[3])
                            .launch(cfg)
                    }
                    .map_err(cuda_err)?;
                }
            }
            2 => {
                let q0 = gate.qubits[0] as i32;
                let q1 = gate.qubits[1] as i32;
                let n_q = n_qubits as i32;
                let num_quads = (dim / 4) as u32;
                let cfg = LaunchConfig::for_num_elems(num_quads);

                let mat_re_dev: CudaSlice<f64> =
                    state.stream.clone_htod(&gate.matrix_re).map_err(cuda_err)?;
                let mat_im_dev: CudaSlice<f64> =
                    state.stream.clone_htod(&gate.matrix_im).map_err(cuda_err)?;

                unsafe {
                    state
                        .stream
                        .launch_builder(&fn_2q)
                        .arg(&mut dev_re)
                        .arg(&mut dev_im)
                        .arg(&n_q)
                        .arg(&q0)
                        .arg(&q1)
                        .arg(&mat_re_dev)
                        .arg(&mat_im_dev)
                        .launch(cfg)
                }
                .map_err(cuda_err)?;
            }
            _ => {
                return Err(GpuError::Cuda(format!(
                    "GPU supports 1- and 2-qubit gates only, got {}-qubit gate '{}'",
                    gate.qubits.len(),
                    gate.name
                )));
            }
        }
    }

    let result_re: Vec<f64> = state.stream.clone_dtoh(&dev_re).map_err(cuda_err)?;
    let result_im: Vec<f64> = state.stream.clone_dtoh(&dev_im).map_err(cuda_err)?;

    let sv: Vec<Complex64> = result_re
        .iter()
        .zip(result_im.iter())
        .map(|(&re, &im)| Complex64::new(re, im))
        .collect();

    Ok(sv)
}

/// Simulate and sample measurement outcomes on GPU.
pub fn simulate_and_sample(
    n_qubits: usize,
    gates: &[GateOp],
    shots: u64,
    seed: u64,
) -> Result<HashMap<u64, u64>, GpuError> {
    let sv = simulate_statevector(n_qubits, gates)?;

    use rand::prelude::*;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let probs: Vec<f64> = sv.iter().map(|c| c.norm_sqr()).collect();

    let mut counts: HashMap<u64, u64> = HashMap::new();
    for _ in 0..shots {
        let r: f64 = rng.gen();
        let mut cumulative = 0.0;
        let mut outcome = probs.len() as u64 - 1;
        for (i, &p) in probs.iter().enumerate() {
            cumulative += p;
            if r < cumulative {
                outcome = i as u64;
                break;
            }
        }
        *counts.entry(outcome).or_insert(0) += 1;
    }

    Ok(counts)
}
