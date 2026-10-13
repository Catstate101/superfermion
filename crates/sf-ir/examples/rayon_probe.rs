//! Rayon/sampler ground-truth probe (desktop dev tree, not shipped).
use rayon::prelude::*;
use sf_ir::stabilizer::StabilizerTableau;
use std::time::Instant;

fn main() {
    println!("rayon threads: {}", rayon::current_num_threads());

    // 1) Parallel sum smoke test: wall time reveals whether workers engage.
    let t = Instant::now();
    let s: u64 = (0..8_000_000u64)
        .into_par_iter()
        .map(|x| x.wrapping_mul(2_654_435_761))
        .sum();
    println!(
        "par sum = {} in {:?} (serial would be ~4-8ms)",
        s,
        t.elapsed()
    );

    // 2) Stabilizer sampler at n=20 (the tvd row shape).
    let n = 20usize;
    let mut gates: Vec<(String, Vec<usize>)> = Vec::new();
    for _layer in 0..8 {
        let mut q = 0usize;
        while q < n - 1 {
            gates.push(("H".into(), vec![q]));
            gates.push(("CX".into(), vec![q, q + 1]));
            q += 2;
        }
        let mut q = 1usize;
        while q < n - 1 {
            gates.push(("H".into(), vec![q]));
            gates.push(("CX".into(), vec![q, q + 1]));
            q += 2;
        }
    }
    let tab = StabilizerTableau::from_gate_list(n, &gates).unwrap();

    // warm the plan cache
    let _ = tab.sample(64, Some(3));

    for shots in [64usize, 640, 6400, 12800, 128000] {
        let t = Instant::now();
        let c = tab.sample(shots, Some(3));
        let el = t.elapsed();
        let blocks = shots.div_ceil(64) as f64;
        println!(
            "sample({:6}) -> {:5} keys in {:>10.3?}  ({:8.2} us/block)",
            shots,
            c.len(),
            el,
            el.as_micros() as f64 / blocks
        );
    }
}
