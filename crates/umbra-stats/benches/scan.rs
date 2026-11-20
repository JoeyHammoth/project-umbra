//! A6 scan cost: 10⁶ candidates in 2-D and in 3-D (spec §5.11's budget row).
//!
//! The criterion this measures is "1e6 candidates under 2 s in 3-D and 50 ms in
//! 2-D". Both systems here are built with **parallel rays through every voxel
//! along one axis**, which makes each voxel's footprint small and is the
//! friendliest geometry the scan can meet. A real cone-beam scene spreads the
//! same opacity over more rays, so treat the numbers as a lower bound: they
//! answer "is the kernel's constant factor in the right decade", not "is every
//! 3-D experiment inside 2 s".
//!
//! Run with `cargo bench -p umbra-stats --bench scan`.

use std::time::{Duration, Instant};

use umbra_core::{Csr, Grid2, Grid3, GridAny};
use umbra_forward::{RayCell, System};
use umbra_stats::{scan, ScanConfig, ScanProblem};

/// Two-dimensional scene: one ray per row, `n` cells long.
fn scene_2d(n: usize) -> (System, Vec<f32>, Vec<f32>, Vec<f32>) {
    let grid = Grid2::new([n as u32, n as u32], [0.0, 0.0], 0.01);
    let mut indptr = vec![0u32];
    let mut indices = Vec::new();
    let mut data = Vec::new();
    for row in 0..n {
        for column in 0..n {
            indices.push((row * n + column) as u32);
            data.push(1.0f32);
        }
        indptr.push(indices.len() as u32);
    }
    finish(Csr::new(indptr, indices, data), GridAny::Two(grid), n)
}

/// Three-dimensional scene: one ray along `z` for each `(x, y)` column.
fn scene_3d(n: usize) -> (System, Vec<f32>, Vec<f32>, Vec<f32>) {
    let grid = Grid3::new([n as u32, n as u32, n as u32], [0.0; 3], 0.01);
    let plane = n * n;
    let mut indptr = vec![0u32];
    let mut indices = Vec::new();
    let mut data = Vec::new();
    for column in 0..plane {
        for k in 0..n {
            indices.push((k * plane + column) as u32);
            data.push(1.0f32);
        }
        indptr.push(indices.len() as u32);
    }
    finish(Csr::new(indptr, indices, data), GridAny::Three(grid), plane)
}

fn finish(l: Csr<f32>, grid: GridAny, rays: usize) -> (System, Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut density = vec![2.65f32; grid.cell_count()];
    // A buried inclusion, so the scan has something to find and the amplitude
    // path is exercised rather than only the H0 arithmetic.
    let voxels = grid.cell_count();
    let start = voxels / 3;
    let end = start + voxels / 50;
    density[start..end].fill(1.0);
    let system = System {
        l,
        rays: (0..rays)
            .map(|_| RayCell {
                det: 0,
                zen: 0.0,
                az: 0.0,
                k_scale: 1.0,
            })
            .collect(),
        grid,
    };
    let opacity = system.opacity(&density);
    let lambda0: Vec<f32> = opacity
        .iter()
        .map(|&x| 1.0e3 * (-x / 1.0e3).exp())
        .collect();
    let dlambda_dx: Vec<f32> = lambda0.iter().map(|&l| -l / 1.0e3).collect();
    (system, density, lambda0, dlambda_dx)
}

fn ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1.0e3
}

fn main() {
    // One scale is exactly the "10⁶ candidates" the budget row names; the second
    // line reports what a multi-scale bank costs on the same scene, because a
    // real scan runs a bank and the budget does not say so.
    let single = ScanConfig {
        scales_voxels: vec![1.0],
        top_k: 10,
        ..ScanConfig::default()
    };
    let multi = ScanConfig {
        scales_voxels: vec![1.0, 2.0],
        ..single.clone()
    };

    for (label, (system, density, lambda0, dlambda_dx), budget) in [
        ("2-D", scene_2d(1000), 50.0f64),
        ("3-D", scene_3d(100), 2000.0f64),
    ] {
        let counts: Vec<f32> = lambda0.iter().map(|l| l * 1.02).collect();
        let problem = ScanProblem {
            system: &system,
            counts: &counts,
            lambda0: &lambda0,
            dlambda_dx: &dlambda_dx,
            density: &density,
            attenuation_gcm2: 1.0e3,
        };
        for (name, cfg) in [("one scale", &single), ("two scales", &multi)] {
            // One warm-up so the first allocation does not land in the measurement.
            let _ = scan(&problem, cfg);
            let mut samples = Vec::new();
            let mut correlations = Vec::new();
            let mut peak = 0.0f32;
            for _ in 0..3 {
                let started = Instant::now();
                let result = scan(&problem, cfg);
                samples.push(started.elapsed());
                correlations.push(result.timing.correlate);
                peak = result.peak.map_or(0.0, |c| c.z);
            }
            samples.sort();
            correlations.sort();
            let total = ms(samples[samples.len() / 2]);
            let correlation = ms(correlations[correlations.len() / 2]);
            println!(
                "{label} {name}: {} candidates, {} rays, {} voxels — median {total:>7.1} ms total, \
                 {correlation:>7.1} ms correlation of {:>5.0} ms budget (peak Z {peak:.2})",
                system.voxel_count() * cfg.scales_voxels.len(),
                system.ray_count(),
                system.voxel_count(),
                budget,
            );
        }
    }
}
